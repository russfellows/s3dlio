pub mod manifest;
pub mod paths;
pub mod writer;
pub mod reader;
pub mod latest;

use anyhow::Result;
use bytes::Bytes;
use crate::object_store::{store_for_uri, ObjectStore};
pub use manifest::{Manifest, ShardMeta};
pub use paths::{KeyLayout, Strategy};
pub use writer::Writer;
pub use reader::Reader;
pub use latest::Latest;

/// Configuration for checkpoint operations
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    pub strategy: Strategy,
    pub multipart_threshold: usize,
    pub part_size: Option<usize>,
    pub enable_validation: bool,
    pub concurrent_uploads: bool,
    pub compression: Option<crate::object_store::CompressionConfig>,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            strategy: Strategy::Flat,
            multipart_threshold: 100 * 1024 * 1024, // 100MB
            part_size: None,
            enable_validation: true,
            concurrent_uploads: true,
            compression: None,
        }
    }
}

impl CheckpointConfig {
    pub fn new() -> Self {
        Self::default()
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

    pub fn with_validation(mut self, enable: bool) -> Self {
        self.enable_validation = enable;
        self
    }

    pub fn with_concurrent_uploads(mut self, enable: bool) -> Self {
        self.concurrent_uploads = enable;
        self
    }

    pub fn with_compression(mut self, compression: crate::object_store::CompressionConfig) -> Self {
        self.compression = Some(compression);
        self
    }

    /// Optimized for S3/Azure with hot-spot avoidance
    pub fn cloud_optimized() -> Self {
        Self {
            strategy: Strategy::RoundRobin,
            multipart_threshold: 64 * 1024 * 1024, // 64MB for cloud
            part_size: Some(8 * 1024 * 1024), // 8MB parts
            enable_validation: true,
            concurrent_uploads: true,
            compression: None,
        }
    }

    /// Optimized for local file systems
    pub fn local_optimized() -> Self {
        Self {
            strategy: Strategy::Flat,
            multipart_threshold: 256 * 1024 * 1024, // 256MB for local
            part_size: None,
            enable_validation: false, // Skip validation for local FS
            concurrent_uploads: true,
            compression: None,
        }
    }

    /// High-performance config for large scale training
    pub fn high_performance() -> Self {
        Self {
            strategy: Strategy::Binary,
            multipart_threshold: 32 * 1024 * 1024, // 32MB
            part_size: Some(16 * 1024 * 1024), // 16MB parts
            enable_validation: false,
            concurrent_uploads: true,
            compression: None,
        }
    }
}

/// Main checkpoint store interface
pub struct CheckpointStore {
    pub uri: String,
    pub store: Box<dyn ObjectStore>,
    pub config: CheckpointConfig,
}

impl CheckpointStore {
    /// Create a new checkpoint store with default configuration
    pub fn open(uri: &str) -> Result<Self> {
        let store = store_for_uri(uri)?;
        Ok(Self {
            uri: uri.to_string(),
            store,
            config: CheckpointConfig::default(),
        })
    }

    /// Create a checkpoint store with custom configuration
    pub fn open_with_config(uri: &str, config: CheckpointConfig) -> Result<Self> {
        let store = store_for_uri(uri)?;
        Ok(Self {
            uri: uri.to_string(),
            store,
            config,
        })
    }

    /// Create a writer for this store
    pub fn writer(&self, world_size: u32, rank: u32) -> Writer<'_> {
        let mut writer = Writer::new(&*self.store, self.uri.clone(), world_size, rank)
            .with_strategy(self.config.strategy)
            .with_multipart_threshold(self.config.multipart_threshold)
            .with_part_size(self.config.part_size.unwrap_or(8 * 1024 * 1024));
        
        if let Some(compression) = &self.config.compression {
            writer = writer.with_compression(compression.clone());
        }
        
        writer
    }

    /// Create a reader for this store
    pub fn reader(&self) -> Reader<'_> {
        Reader::new(&*self.store, self.uri.clone())
    }

    /// Convenience method to save a single-rank checkpoint
    pub async fn save(
        &self,
        step: u64,
        epoch: u64,
        framework: &str,
        data: &[u8],
        user_meta: Option<serde_json::Value>,
    ) -> Result<String> {
        let writer = self.writer(1, 0);
        writer.save_checkpoint(step, epoch, framework, data, user_meta).await
    }

    /// Convenience method to load the latest checkpoint
    pub async fn load_latest(&self) -> Result<Option<Bytes>> {
        let reader = self.reader();
        match reader.load_latest_manifest().await? {
            Some(manifest) => {
                if manifest.world_size == 1 {
                    let data = reader.read_shard_by_rank(&manifest, 0).await?;
                    Ok(Some(data))
                } else {
                    anyhow::bail!("Cannot load multi-rank checkpoint with load_latest(). Use reader() directly.");
                }
            }
            None => Ok(None),
        }
    }

    /// Get information about available checkpoints
    pub async fn list_checkpoints(&self) -> Result<Vec<CheckpointInfo>> {
        let reader = self.reader();
        let checkpoints = reader.list_checkpoints().await?;
        
        let mut infos = Vec::new();
        for (step, timestamp, status) in checkpoints {
            if let Some(manifest) = reader.find_manifest_by_step(step).await? {
                infos.push(CheckpointInfo {
                    step,
                    epoch: manifest.epoch,
                    timestamp,
                    status,
                    framework: manifest.framework,
                    world_size: manifest.world_size,
                    total_size: manifest.shards.iter().map(|s| s.size).sum(),
                });
            }
        }
        
        Ok(infos)
    }

    /// Delete a checkpoint
    pub async fn delete_checkpoint(&self, step: u64) -> Result<bool> {
        let reader = self.reader();
        reader.delete_checkpoint(step).await
    }

    /// Validate a checkpoint
    pub async fn validate_checkpoint(&self, step: Option<u64>) -> Result<Vec<String>> {
        if !self.config.enable_validation {
            return Ok(vec!["Validation disabled in config".to_string()]);
        }

        let reader = self.reader();
        let manifest = match step {
            Some(s) => reader.find_manifest_by_step(s).await?,
            None => reader.load_latest_manifest().await?,
        };

        match manifest {
            Some(m) => reader.validate_checkpoint(&m).await,
            None => Ok(vec!["No checkpoint found".to_string()]),
        }
    }

    /// Check if the store is accessible
    pub async fn health_check(&self) -> Result<()> {
        // Try to list the base directory
        self.store.list(&self.uri, false).await?;
        Ok(())
    }
}

/// Information about a checkpoint
#[derive(Debug, Clone)]
pub struct CheckpointInfo {
    pub step: u64,
    pub epoch: u64,
    pub timestamp: String,
    pub status: String,
    pub framework: String,
    pub world_size: u32,
    pub total_size: u64,
}

impl CheckpointInfo {
    pub fn is_complete(&self) -> bool {
        self.status == "complete"
    }

    pub fn is_single_rank(&self) -> bool {
        self.world_size == 1
    }

    pub fn size_mb(&self) -> f64 {
        self.total_size as f64 / (1024.0 * 1024.0)
    }

    pub fn size_gb(&self) -> f64 {
        self.total_size as f64 / (1024.0 * 1024.0 * 1024.0)
    }
}

/// High-level convenience functions
impl CheckpointStore {
    /// Auto-detect optimal configuration based on URI scheme
    pub fn open_auto_config(uri: &str) -> Result<Self> {
        let config = if uri.starts_with("s3://") || uri.starts_with("az://") {
            CheckpointConfig::cloud_optimized()
        } else {
            CheckpointConfig::local_optimized()
        };
        
        Self::open_with_config(uri, config)
    }

    /// Create a high-performance store for large-scale training
    pub fn open_high_performance(uri: &str) -> Result<Self> {
        Self::open_with_config(uri, CheckpointConfig::high_performance())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_checkpoint_store_basic() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let uri = format!("file://{}", temp_dir.path().display());
        
        let store = CheckpointStore::open(&uri)?;
        
        // Health check
        store.health_check().await?;

        // Save a checkpoint
        let data = b"test checkpoint data";
        let manifest_key = store.save(100, 10, "torch", data, None).await?;
        assert!(manifest_key.contains("manifests/ckpt-100-"));

        // Load it back
        let loaded = store.load_latest().await?.unwrap();
        assert_eq!(&loaded[..], data);

        // List checkpoints
        let infos = store.list_checkpoints().await?;
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].step, 100);
        assert!(infos[0].is_complete());
        assert!(infos[0].is_single_rank());

        Ok(())
    }

    #[tokio::test]
    async fn test_checkpoint_store_configs() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let uri = format!("file://{}", temp_dir.path().display());

        // Test different configurations
        let _store1 = CheckpointStore::open_auto_config(&uri)?;
        let _store2 = CheckpointStore::open_high_performance(&uri)?;
        
        let custom_config = CheckpointConfig::new()
            .with_strategy(Strategy::Binary)
            .with_validation(false);
        let _store3 = CheckpointStore::open_with_config(&uri, custom_config)?;

        Ok(())
    }

    #[tokio::test]
    async fn test_distributed_workflow() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let uri = format!("file://{}", temp_dir.path().display());
        
        let store = CheckpointStore::open(&uri)?;

        // Simulate distributed training with 2 ranks
        let writer_0 = store.writer(2, 0);
        let writer_1 = store.writer(2, 1);

        // Each rank saves its shard
        let (layout, shard_0) = writer_0.save_distributed_shard(
            200, 20, "torch", b"rank 0 model state"
        ).await?;

        let (_, shard_1) = writer_1.save_distributed_shard(
            200, 20, "torch", b"rank 1 model state"
        ).await?;

        // Rank 0 finalizes the checkpoint
        let manifest_key = writer_0.finalize_distributed_checkpoint(
            &layout,
            "torch",
            20,
            vec![shard_0, shard_1],
            Some(serde_json::json!({"model": "gpt", "layers": 24})),
        ).await?;

        assert!(manifest_key.contains("manifests/ckpt-200-"));

        // Read back the distributed checkpoint
        let reader = store.reader();
        let manifest = reader.load_latest_manifest().await?.unwrap();
        assert_eq!(manifest.world_size, 2);
        assert_eq!(manifest.framework, "torch");

        let all_shards = reader.read_all_shards(&manifest).await?;
        assert_eq!(all_shards.len(), 2);

        // Validate the checkpoint
        let errors = store.validate_checkpoint(Some(200)).await?;
        assert!(errors.is_empty(), "Validation errors: {:?}", errors);

        Ok(())
    }
}
