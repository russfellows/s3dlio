use anyhow::Result;
use crate::object_store::ObjectStore;
use super::manifest::{Manifest, ShardMeta};
use super::paths::{KeyLayout, Strategy};
use super::latest::{Latest, write_latest};

pub struct Writer<'a> {
    pub store: &'a dyn ObjectStore,
    pub base_uri: String,    // e.g., "s3://bucket/prefix" | "file:///dir" | "direct:///dir" | "az://account/container/prefix"
    pub world_size: u32,
    pub rank: u32,
    pub strategy: Strategy,
    pub multipart_threshold: usize, // bytes; above this use put_multipart
    pub part_size: Option<usize>,   // forwarded to put_multipart
}

impl<'a> Writer<'a> {
    pub fn new(
        store: &'a dyn ObjectStore,
        base_uri: String,
        world_size: u32,
        rank: u32,
    ) -> Self {
        Self {
            store,
            base_uri,
            world_size,
            rank,
            strategy: Strategy::Flat,
            multipart_threshold: 100 * 1024 * 1024, // 100MB default
            part_size: None,
        }
    }

    pub fn with_strategy(mut self, strategy: Strategy) -> Self {
        self.strategy = strategy;
        self
    }

    pub fn with_multipart_threshold(mut self, threshold: usize) -> Self {
        self.multipart_threshold = threshold;
        self
    }

    pub fn with_part_size(mut self, part_size: usize) -> Self {
        self.part_size = Some(part_size);
        self
    }

    /// Upload a shard for this rank and return metadata
    pub async fn put_shard(&self, layout: &KeyLayout, data: &[u8]) -> Result<ShardMeta> {
        let key = layout.shard_key(self.rank, self.strategy);
        let uri = layout.to_uri(&self.base_uri, &key);
        
        // Choose upload method based on size
        if data.len() >= self.multipart_threshold {
            self.store.put_multipart(&uri, data, self.part_size).await?;
        } else {
            self.store.put(&uri, data).await?;
        }
        
        // Collect metadata
        let meta = self.store.stat(&uri).await?;
        Ok(ShardMeta {
            rank: self.rank,
            key: key,
            size: meta.size,
            etag: meta.e_tag,
            checksum: None, // TODO: Add checksum support if needed
        })
    }

    /// Upload a shard with custom key (for flexibility)
    pub async fn put_shard_with_key(&self, key: &str, data: &[u8]) -> Result<ShardMeta> {
        let uri = format!("{}/{}", self.base_uri.trim_end_matches('/'), key);
        
        if data.len() >= self.multipart_threshold {
            self.store.put_multipart(&uri, data, self.part_size).await?;
        } else {
            self.store.put(&uri, data).await?;
        }
        
        let meta = self.store.stat(&uri).await?;
        Ok(ShardMeta {
            rank: self.rank,
            key: key.to_string(),
            size: meta.size,
            etag: meta.e_tag,
            checksum: None,
        })
    }

    /// Write a manifest (typically called by rank 0)
    pub async fn write_manifest(&self, layout: &KeyLayout, manifest: &Manifest) -> Result<String> {
        let manifest_key = layout.manifest_key();
        let uri = layout.to_uri(&self.base_uri, &manifest_key);
        let bytes = serde_json::to_vec_pretty(manifest)?;
        self.store.put(&uri, &bytes).await?;
        Ok(manifest_key)
    }

    /// Update the latest pointer (typically called by rank 0)
    pub async fn publish_latest(&self, layout: &KeyLayout, manifest_rel_key: &str) -> Result<()> {
        let latest_uri = layout.to_uri(&self.base_uri, &layout.latest_key());
        let latest = Latest {
            manifest_key: manifest_rel_key.to_string(),
            global_step: layout.step,
            ts: layout.ts.clone(),
        };
        write_latest(self.store, &latest_uri, &latest).await
    }

    /// Convenience method to save a complete checkpoint in one call
    /// This is for single-rank or when coordination is handled externally
    pub async fn save_checkpoint(
        &self,
        step: u64,
        epoch: u64,
        framework: &str,
        data: &[u8],
        user_meta: Option<serde_json::Value>,
    ) -> Result<String> {
        let layout = KeyLayout::new(self.base_uri.clone(), step);
        
        // Create manifest
        let mut manifest = Manifest::new(
            framework.to_string(),
            step,
            epoch,
            self.world_size,
        );
        manifest.user_meta = user_meta;

        // Upload shard
        let shard_meta = self.put_shard(&layout, data).await?;
        manifest.add_shard(shard_meta);

        // If this is a single-rank checkpoint or we're rank 0, finalize
        if self.world_size == 1 || self.rank == 0 {
            manifest.mark_complete();
            let manifest_key = self.write_manifest(&layout, &manifest).await?;
            self.publish_latest(&layout, &manifest_key).await?;
            Ok(manifest_key)
        } else {
            // Multi-rank case: just return the shard info
            // Caller needs to coordinate with rank 0 to finalize
            Ok(format!("shard-{}-uploaded", self.rank))
        }
    }

    /// Distributed checkpoint: each rank uploads its shard
    pub async fn save_distributed_shard(
        &self,
        step: u64,
        _epoch: u64,
        _framework: &str,
        data: &[u8],
    ) -> Result<(KeyLayout, ShardMeta)> {
        let layout = KeyLayout::new(self.base_uri.clone(), step);
        let shard_meta = self.put_shard(&layout, data).await?;
        Ok((layout, shard_meta))
    }

    /// Finalize distributed checkpoint (rank 0 only)
    pub async fn finalize_distributed_checkpoint(
        &self,
        layout: &KeyLayout,
        framework: &str,
        epoch: u64,
        shards: Vec<ShardMeta>,
        user_meta: Option<serde_json::Value>,
    ) -> Result<String> {
        let mut manifest = Manifest::new(
            framework.to_string(),
            layout.step,
            epoch,
            self.world_size,
        );
        manifest.user_meta = user_meta;

        // Add all shards
        for shard in shards {
            manifest.add_shard(shard);
        }

        // Sort and validate
        manifest.sort_shards();
        manifest.mark_complete();
        manifest.validate()?;

        // Write manifest and update latest
        let manifest_key = self.write_manifest(layout, &manifest).await?;
        self.publish_latest(layout, &manifest_key).await?;

        Ok(manifest_key)
    }

    /// Check if a checkpoint exists
    pub async fn checkpoint_exists(&self, step: u64) -> Result<bool> {
        let layout = KeyLayout::new(self.base_uri.clone(), step);
        let latest_uri = layout.to_uri(&self.base_uri, &layout.latest_key());
        self.store.exists(&latest_uri).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object_store::store_for_uri;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_single_rank_checkpoint() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let base_uri = format!("file://{}", temp_dir.path().display());
        let store = store_for_uri(&base_uri)?;

        let writer = Writer::new(&*store, base_uri.clone(), 1, 0);
        let data = b"checkpoint data for rank 0";

        let manifest_key = writer.save_checkpoint(
            100,
            10,
            "torch",
            data,
            Some(serde_json::json!({"model": "resnet50"})),
        ).await?;

        assert!(manifest_key.contains("manifests/ckpt-100-"));
        assert!(writer.checkpoint_exists(100).await?);

        Ok(())
    }

    #[tokio::test]
    async fn test_distributed_checkpoint() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let base_uri = format!("file://{}", temp_dir.path().display());
        let store = store_for_uri(&base_uri)?;

        // Simulate 2 ranks
        let writer_0 = Writer::new(&*store, base_uri.clone(), 2, 0);
        let writer_1 = Writer::new(&*store, base_uri.clone(), 2, 1);

        // Each rank saves its shard
        let (layout, shard_0) = writer_0.save_distributed_shard(
            200, 20, "torch", b"rank 0 data"
        ).await?;

        let (_, shard_1) = writer_1.save_distributed_shard(
            200, 20, "torch", b"rank 1 data"
        ).await?;

        // Rank 0 finalizes
        let manifest_key = writer_0.finalize_distributed_checkpoint(
            &layout,
            "torch",
            20,
            vec![shard_0, shard_1],
            None,
        ).await?;

        assert!(manifest_key.contains("manifests/ckpt-200-"));
        assert!(writer_0.checkpoint_exists(200).await?);

        Ok(())
    }

    #[tokio::test]
    async fn test_different_strategies() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let base_uri = format!("file://{}", temp_dir.path().display());
        let store = store_for_uri(&base_uri)?;

        for strategy in [Strategy::Flat, Strategy::RoundRobin, Strategy::Binary] {
            let writer = Writer::new(&*store, base_uri.clone(), 4, 0)
                .with_strategy(strategy);
            
            let layout = KeyLayout::new(base_uri.clone(), 300);
            let data = format!("data for strategy {:?}", strategy);
            
            let shard_meta = writer.put_shard(&layout, data.as_bytes()).await?;
            assert_eq!(shard_meta.rank, 0);
            assert!(shard_meta.size > 0);
        }

        Ok(())
    }
}
