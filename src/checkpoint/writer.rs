// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use super::latest::{write_latest, Latest};
use super::manifest::{Manifest, ShardMeta};
use super::paths::{KeyLayout, Strategy};
use crate::object_store::ObjectStore;
use anyhow::Result;
use bytes::Bytes;

/// A shard payload accepted by the writer's one-shot APIs ([`Writer::put_shard`],
/// [`Writer::put_shard_with_key`], [`Writer::save_checkpoint`],
/// [`Writer::save_distributed_shard`], `CheckpointStore::save`).
///
/// RULE #1 (zero copy): the store receives the caller's own [`Bytes`]. `Bytes`, `&Bytes` (a
/// refcount clone) and `Vec<u8>` (which `Bytes::from` adopts without copying) are passed along by
/// reference count; a Python buffer export is wrapped by `Bytes::from_owner` before it gets here.
/// The other borrowed forms (`&[u8]`, `&Vec<u8>`, `&[u8; N]`, `&mut Vec<u8>`, `&mut [u8]`,
/// `&Box<[u8]>`, `&Arc<[u8]>`, `&Cow<[u8]>`, `&BytesMut`) exist so existing callers compile: a
/// borrowed slice has no owner a refcounted `Bytes` could keep alive, so those forms take ONE
/// copy, made only where a `Bytes` is really needed (the uncompressed upload). A compressing
/// writer only needs a slice ([`as_shard_slice`](Self::as_shard_slice)): the zstd encoder is the
/// transform, so there the borrowed forms are not copied at all. Pass a `Bytes` to avoid the copy.
///
/// Because the parameter is generic, Rust no longer auto-derefs a reference into `&[u8]`: a
/// reference to a type not listed above (`&&[u8]`, `&String`, `&Rc<[u8]>`, ...) needs
/// `.as_slice()` / `&x[..]` at the call site.
pub trait IntoShardBytes: Send {
    /// The payload as a slice (no copy), for writers that only read it.
    fn as_shard_slice(&self) -> &[u8];
    /// Convert into the refcounted payload handed to the object store.
    fn into_shard_bytes(self) -> Bytes;
}

impl IntoShardBytes for Bytes {
    fn as_shard_slice(&self) -> &[u8] {
        self
    }
    fn into_shard_bytes(self) -> Bytes {
        self
    }
}

impl IntoShardBytes for &Bytes {
    fn as_shard_slice(&self) -> &[u8] {
        self
    }
    fn into_shard_bytes(self) -> Bytes {
        // A refcount clone, not a copy of the payload.
        self.clone()
    }
}

impl IntoShardBytes for Vec<u8> {
    fn as_shard_slice(&self) -> &[u8] {
        self
    }
    fn into_shard_bytes(self) -> Bytes {
        Bytes::from(self)
    }
}

/// The borrowed (compat) forms: one copy, only when a `Bytes` is needed.
macro_rules! borrowed_shard_data {
    ($($ty:ty => |$this:ident| $slice:expr),+ $(,)?) => {$(
        impl IntoShardBytes for $ty {
            fn as_shard_slice(&self) -> &[u8] {
                let $this = self;
                $slice
            }
            fn into_shard_bytes(self) -> Bytes {
                // Compatibility form: a borrowed slice cannot be kept alive by a `Bytes`, so this
                // is the one unavoidable copy of the borrowed API (never made when compressing);
                // callers that own the data pass `Bytes`.
                Bytes::copy_from_slice(self.as_shard_slice())
            }
        }
    )+};
}

borrowed_shard_data! {
    &[u8] => |t| t,
    &mut [u8] => |t| t,
    &Vec<u8> => |t| t.as_slice(),
    &mut Vec<u8> => |t| t.as_slice(),
    &Box<[u8]> => |t| t,
    &std::sync::Arc<[u8]> => |t| t,
    &std::borrow::Cow<'_, [u8]> => |t| t,
    &bytes::BytesMut => |t| t,
}

impl<const N: usize> IntoShardBytes for &[u8; N] {
    fn as_shard_slice(&self) -> &[u8] {
        self.as_slice()
    }
    fn into_shard_bytes(self) -> Bytes {
        // Compatibility form, see the other borrowed forms.
        Bytes::copy_from_slice(self.as_slice())
    }
}

pub struct Writer<'a> {
    pub store: &'a dyn ObjectStore,
    pub base_uri: String, // e.g., "s3://bucket/prefix" | "file:///dir" | "direct:///dir" | "az://account/container/prefix"
    pub world_size: u32,
    pub rank: u32,
    pub strategy: Strategy,
    pub multipart_threshold: usize, // bytes; above this use put_multipart
    pub part_size: Option<usize>,   // forwarded to put_multipart
    pub compression: Option<crate::object_store::CompressionConfig>,
}

impl<'a> Writer<'a> {
    pub fn new(store: &'a dyn ObjectStore, base_uri: String, world_size: u32, rank: u32) -> Self {
        Self {
            store,
            base_uri,
            world_size,
            rank,
            strategy: Strategy::Flat,
            multipart_threshold: 100 * 1024 * 1024, // 100MB default
            part_size: None,
            compression: None,
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

    pub fn with_compression(mut self, compression: crate::object_store::CompressionConfig) -> Self {
        self.compression = Some(compression);
        self
    }

    /// Upload a shard for this rank and return metadata.
    ///
    /// Zero copy for a `Bytes` payload ([`IntoShardBytes`]): the store receives the caller's
    /// `Bytes`. With compression configured the compressing writer reads it as a slice (the
    /// zstd transform is the only copy).
    pub async fn put_shard<D: IntoShardBytes>(
        &self,
        layout: &KeyLayout,
        data: D,
    ) -> Result<ShardMeta> {
        let key = layout.shard_key(self.rank, self.strategy);
        let uri = layout.to_uri(&self.base_uri, &key);

        // If compression is enabled, use streaming writer for compression support
        if let Some(compression) = &self.compression {
            let (mut writer, _) = self.get_shard_writer(layout).await?;
            writer.write_chunk(data.as_shard_slice()).await?;
            // The checksum of the logical (uncompressed) bytes, taken before finalize consumes
            // the writer.
            let checksum = writer.checksum();
            writer.finalize().await?;

            // A compressing writer stores the shard under `stored_name`; the manifest and the
            // stat use the same rule, so they name the object the writer created.
            let compressed_uri = compression.stored_name(&uri);
            let compressed_key = compression.stored_name(&key);

            let meta = self.store.stat(&compressed_uri).await?;
            Ok(ShardMeta {
                rank: self.rank,
                key: compressed_key,
                size: meta.size,
                etag: meta.e_tag,
                checksum,
            })
        } else {
            // Choose upload method based on size for uncompressed data
            // The store needs a refcounted `Bytes`: made here, only on this path, so a compressing
            // save from a borrowed slice never copies it.
            let data = data.into_shard_bytes();
            if data.len() >= self.multipart_threshold {
                self.store
                    .put_multipart(&uri, data.clone(), self.part_size)
                    .await?;
            } else {
                self.store.put(&uri, data.clone()).await?;
            }

            // Collect metadata
            let meta = self.store.stat(&uri).await?;
            Ok(ShardMeta {
                rank: self.rank,
                key,
                size: meta.size,
                etag: meta.e_tag,
                checksum: Some(crate::checksum::crc32c_label(crate::checksum::crc32c_of(
                    &data,
                ))),
            })
        }
    }

    /// Upload a shard with custom key (for flexibility). Zero copy for a `Bytes` payload, like
    /// [`put_shard`](Self::put_shard).
    pub async fn put_shard_with_key<D: IntoShardBytes>(
        &self,
        key: &str,
        data: D,
    ) -> Result<ShardMeta> {
        let uri = format!("{}/{}", self.base_uri.trim_end_matches('/'), key);

        // If compression is enabled, use streaming writer for compression support
        if let Some(compression) = &self.compression {
            let mut writer = self.get_shard_writer_with_key(key).await?;
            writer.write_chunk(data.as_shard_slice()).await?;
            // The checksum of the logical (uncompressed) bytes, taken before finalize consumes
            // the writer.
            let checksum = writer.checksum();
            writer.finalize().await?;

            // A compressing writer stores the shard under `stored_name`; the manifest and the
            // stat use the same rule, so they name the object the writer created.
            let compressed_uri = compression.stored_name(&uri);
            let compressed_key = compression.stored_name(key);

            let meta = self.store.stat(&compressed_uri).await?;
            Ok(ShardMeta {
                rank: self.rank,
                key: compressed_key,
                size: meta.size,
                etag: meta.e_tag,
                checksum,
            })
        } else {
            // Choose upload method based on size for uncompressed data
            // The store needs a refcounted `Bytes`: made here, only on this path, so a compressing
            // save from a borrowed slice never copies it.
            let data = data.into_shard_bytes();
            if data.len() >= self.multipart_threshold {
                self.store
                    .put_multipart(&uri, data.clone(), self.part_size)
                    .await?;
            } else {
                self.store.put(&uri, data.clone()).await?;
            }

            let meta = self.store.stat(&uri).await?;
            Ok(ShardMeta {
                rank: self.rank,
                key: key.to_string(),
                size: meta.size,
                etag: meta.e_tag,
                checksum: Some(crate::checksum::crc32c_label(crate::checksum::crc32c_of(
                    &data,
                ))),
            })
        }
    }

    /// Get a streaming writer for a shard (zero-copy version of put_shard)
    /// This enables writing large shards without buffering the entire content in memory
    pub async fn get_shard_writer(
        &self,
        layout: &KeyLayout,
    ) -> Result<(Box<dyn crate::object_store::ObjectWriter>, String)> {
        let key = layout.shard_key(self.rank, self.strategy);
        let uri = layout.to_uri(&self.base_uri, &key);

        let writer = if let Some(compression) = &self.compression {
            self.store
                .get_writer_with_compression(&uri, *compression)
                .await?
        } else {
            self.store.get_writer(&uri).await?
        };

        Ok((writer, key))
    }

    /// Get a streaming writer with custom key (for flexibility)
    pub async fn get_shard_writer_with_key(
        &self,
        key: &str,
    ) -> Result<Box<dyn crate::object_store::ObjectWriter>> {
        let uri = format!("{}/{}", self.base_uri.trim_end_matches('/'), key);

        if let Some(compression) = &self.compression {
            self.store
                .get_writer_with_compression(&uri, *compression)
                .await
        } else {
            self.store.get_writer(&uri).await
        }
    }

    /// The name a shard written through this Writer is stored (and must be recorded) under:
    /// the compression extension is added by the one rule, [`CompressionConfig::stored_name`],
    /// when this Writer is configured with compression. Idempotent, so a caller that already
    /// passes the stored name is unaffected.
    ///
    /// [`CompressionConfig::stored_name`]: crate::object_store::CompressionConfig::stored_name
    fn recorded_key(&self, key: String) -> String {
        match &self.compression {
            Some(compression) => compression.stored_name(&key),
            None => key,
        }
    }

    /// Helper to create ShardMeta after streaming write is complete.
    ///
    /// `key` is the key the shard writer was created for (`get_shard_writer`'s second value);
    /// when this Writer has compression the recorded and statted name is the stored one
    /// (`shard.bin.zst`), by the same rule the writers use.
    pub async fn finalize_shard_meta(&self, layout: &KeyLayout, key: String) -> Result<ShardMeta> {
        let key = self.recorded_key(key);
        let uri = layout.to_uri(&self.base_uri, &key);
        let meta = self.store.stat(&uri).await?;
        Ok(ShardMeta {
            rank: self.rank,
            key,
            size: meta.size,
            etag: meta.e_tag,
            checksum: None, // TODO: Add checksum support if needed
        })
    }

    /// Helper to create ShardMeta with checksum from writer before finalizing
    pub async fn finalize_shard_meta_with_checksum(
        &self,
        layout: &KeyLayout,
        key: String,
        checksum: Option<String>,
    ) -> Result<ShardMeta> {
        let key = self.recorded_key(key);
        let uri = layout.to_uri(&self.base_uri, &key);
        let meta = self.store.stat(&uri).await?;
        Ok(ShardMeta {
            rank: self.rank,
            key,
            size: meta.size,
            etag: meta.e_tag,
            checksum,
        })
    }

    /// Complete helper: finalize writer and create ShardMeta with checksum
    pub async fn finalize_writer_to_shard_meta(
        &self,
        layout: &KeyLayout,
        key: String,
        writer: Box<dyn crate::object_store::ObjectWriter>,
    ) -> Result<ShardMeta> {
        let checksum = writer.checksum();
        // A compressing writer stores the shard under the compression extension
        // (`shard.bin.zst`), and the manifest must name the stored object.
        let compression = writer.compression();
        let key = compression.stored_name(&key);
        writer.finalize().await?;
        self.finalize_shard_meta_with_checksum(layout, key, checksum)
            .await
    }

    /// Write a manifest (typically called by rank 0)
    pub async fn write_manifest(&self, layout: &KeyLayout, manifest: &Manifest) -> Result<String> {
        let manifest_key = layout.manifest_key();
        let uri = layout.to_uri(&self.base_uri, &manifest_key);
        let bytes = serde_json::to_vec_pretty(manifest)?;
        self.store.put(&uri, Bytes::from(bytes)).await?;
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
    pub async fn save_checkpoint<D: IntoShardBytes>(
        &self,
        step: u64,
        epoch: u64,
        framework: &str,
        data: D,
        user_meta: Option<serde_json::Value>,
    ) -> Result<String> {
        let layout = KeyLayout::new(self.base_uri.clone(), step);

        // Create manifest
        let mut manifest = Manifest::new(framework.to_string(), step, epoch, self.world_size);
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
    pub async fn save_distributed_shard<D: IntoShardBytes>(
        &self,
        step: u64,
        _epoch: u64,
        _framework: &str,
        data: D,
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
        let mut manifest =
            Manifest::new(framework.to_string(), layout.step, epoch, self.world_size);
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

        let manifest_key = writer
            .save_checkpoint(
                100,
                10,
                "torch",
                data,
                Some(serde_json::json!({"model": "resnet50"})),
            )
            .await?;

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
        let (layout, shard_0) = writer_0
            .save_distributed_shard(200, 20, "torch", b"rank 0 data")
            .await?;

        let (_, shard_1) = writer_1
            .save_distributed_shard(200, 20, "torch", b"rank 1 data")
            .await?;

        // Rank 0 finalizes
        let manifest_key = writer_0
            .finalize_distributed_checkpoint(&layout, "torch", 20, vec![shard_0, shard_1], None)
            .await?;

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
            let writer = Writer::new(&*store, base_uri.clone(), 4, 0).with_strategy(strategy);

            let layout = KeyLayout::new(base_uri.clone(), 300);
            let data = format!("data for strategy {:?}", strategy);

            let shard_meta = writer.put_shard(&layout, data.as_bytes()).await?;
            assert_eq!(shard_meta.rank, 0);
            assert!(shard_meta.size > 0);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_streaming_shard_writer() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let base_uri = format!("file://{}", temp_dir.path().display());

        let store = store_for_uri(&base_uri)?;
        let writer = Writer::new(&*store, base_uri, 1, 0);

        let layout = KeyLayout::new("checkpoints".to_string(), 42);

        // Test streaming shard writing
        {
            let (mut shard_writer, key) = writer.get_shard_writer(&layout).await?;

            // Write data in chunks (simulating streaming from Python/training)
            let chunk1 = b"first chunk of tensor data";
            let chunk2 = b"second chunk of tensor data";
            let chunk3 = b"final chunk of tensor data";

            shard_writer.write_chunk(chunk1).await?;
            shard_writer.write_chunk(chunk2).await?;
            shard_writer.write_chunk(chunk3).await?;

            let total_bytes = shard_writer.bytes_written();
            assert_eq!(
                total_bytes,
                (chunk1.len() + chunk2.len() + chunk3.len()) as u64
            );

            // Finalize the write
            shard_writer.finalize().await?;

            // Create metadata after streaming write
            let shard_meta = writer.finalize_shard_meta(&layout, key).await?;
            assert_eq!(shard_meta.rank, 0);
            assert_eq!(shard_meta.size, total_bytes);

            println!("✓ Streaming shard writer test passed");
        }

        // Test custom key streaming
        {
            let custom_key = "custom_shard_stream.bin";
            let mut shard_writer = writer.get_shard_writer_with_key(custom_key).await?;

            let data = b"streaming checkpoint data with custom key";
            shard_writer.write_chunk(data).await?;

            let bytes_written = shard_writer.bytes_written();
            shard_writer.finalize().await?;

            // Verify the file was written correctly
            let uri = format!("{}/{}", writer.base_uri.trim_end_matches('/'), custom_key);
            let read_data = store.get(&uri).await?;
            assert_eq!(read_data.as_ref(), data);
            assert_eq!(bytes_written, data.len() as u64);

            println!("✓ Custom key streaming writer test passed");
        }

        // Test write cancellation
        {
            let (mut shard_writer, _key) = writer.get_shard_writer(&layout).await?;

            shard_writer
                .write_chunk(b"some data that will be cancelled")
                .await?;
            assert!(shard_writer.bytes_written() > 0);

            // Cancel the write
            shard_writer.cancel().await?;

            println!("✓ Streaming writer cancellation test passed");
        }

        Ok(())
    }
}
