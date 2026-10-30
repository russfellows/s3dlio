// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use tracing::{debug, info, trace};
// src/file_store.rs
//
// FileSystemObjectStore implementation for POSIX file I/O
// This provides the same ObjectStore interface for local filesystem operations

use crate::checksum::Crc32c;
use crate::data_loader::options::PageCacheMode;
use crate::object_store::WriterOptions;
use crate::page_cache::apply_page_cache_hint;
use crate::range_engine_generic::{RangeEngine, RangeEngineConfig};
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::fs;

use crate::constants::DEFAULT_FILE_RANGE_ENGINE_THRESHOLD;
use crate::object_store::{
    CompressionConfig, HeadMetadata, ObjectMetadata, ObjectProperties, ObjectStore, ObjectWriter,
    ReceiveProgress,
};

/// A file modification time as an RFC 3339 UTC string (#263), the form `ObjectMetadata::last_modified`
/// carries for every other backend.
///
/// The sub-second part the filesystem reported is kept (`2025-05-16T09:46:12.500Z`); a whole-second
/// time has none (`2025-05-16T09:46:12Z`).  A time before the Unix epoch, or one outside the
/// range RFC 3339 here can express, has no such string and is reported as absent rather than
/// guessed.  Shared by the `file://` and the direct-I/O store.
pub(crate) fn modified_time_rfc3339(time: std::time::SystemTime) -> Option<String> {
    let since_epoch = time
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .ok()?;
    let secs = i64::try_from(since_epoch.as_secs()).ok()?;
    let datetime = chrono::DateTime::from_timestamp(secs, since_epoch.subsec_nanos())?;
    Some(datetime.to_rfc3339_opts(chrono::SecondsFormat::AutoSi, true))
}

/// Configuration for FileSystemObjectStore
///
/// **Performance Note**: Range parallelism may be **counterproductive** for local files
/// - Sequential reads are typically faster due to OS page cache optimization
/// - Concurrent ranges introduce seek overhead and disk contention
/// - RangeEngine is **disabled by default** (v0.9.6+) due to performance testing
/// - Benefits are primarily for network storage (S3/Azure/GCS)
#[derive(Debug, Clone)]
pub struct FileSystemConfig {
    /// Enable concurrent range downloads for large files
    /// Default: false (v0.9.6+) - local files rarely benefit from range parallelism
    pub enable_range_engine: bool,

    /// Range engine configuration
    pub range_engine: RangeEngineConfig,

    /// Page cache behavior hint (maps to posix_fadvise on Linux/Unix)
    /// - None: Use Auto mode (Sequential for large files >=64MB, Random for small)
    /// - Some(mode): Explicitly set Sequential, Random, DontNeed, or Normal
    ///   Default: None (Auto mode)
    pub page_cache_mode: Option<PageCacheMode>,
}

impl Default for FileSystemConfig {
    fn default() -> Self {
        Self {
            enable_range_engine: false, // Disabled by default due to local FS seek overhead (v0.9.6+)
            range_engine: RangeEngineConfig {
                min_split_size: DEFAULT_FILE_RANGE_ENGINE_THRESHOLD,
                ..Default::default()
            },
            page_cache_mode: None, // Use Auto mode by default
        }
    }
}

/// FileSystem adapter that implements ObjectStore for local POSIX file operations.
///
/// URI Mapping:
/// - `file:///absolute/path/to/file` -> `/absolute/path/to/file`
/// - `file://./relative/path/to/file` -> `./relative/path/to/file`
/// - `file://../relative/path/to/file` -> `../relative/path/to/file`
///
/// Container Operations:
/// - create_container: creates directories
/// - delete_container: removes empty directories
///
/// Performance:
/// - Files >= 4MB use concurrent range downloads (configurable)
/// - Files < 4MB use simple sequential reads
/// - Expected 30-50% throughput improvement for large files
#[derive(Clone)]
pub struct FileSystemObjectStore {
    config: Arc<FileSystemConfig>,
    size_cache: Arc<crate::object_size_cache::ObjectSizeCache>,
}

/// Fill `buf` completely from `file` starting at byte `offset` with positional reads
/// (`pread(2)`), straight into the caller's memory; an end of file before `buf` is full is an
/// `UnexpectedEof` error.
#[cfg(unix)]
pub(crate) fn read_exact_at_std(
    file: &std::fs::File,
    buf: &mut [u8],
    offset: u64,
) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}

#[cfg(not(unix))]
pub(crate) fn read_exact_at_std(
    file: &std::fs::File,
    buf: &mut [u8],
    offset: u64,
) -> std::io::Result<()> {
    use std::io::{Read, Seek};
    let mut file = file;
    file.seek(std::io::SeekFrom::Start(offset))?;
    file.read_exact(buf)
}

/// Read from `file` at byte `offset` into `buf` with positional reads (`pread(2)`) until `buf`
/// is full or the file ends; returns the number of bytes read (less than `buf.len()` only at the
/// end of the file). Straight into the caller's memory: nothing is buffered or copied.
pub(crate) fn read_full_at_std(
    file: &std::fs::File,
    buf: &mut [u8],
    offset: u64,
) -> std::io::Result<usize> {
    let mut total = 0usize;
    while total < buf.len() {
        let at = offset.saturating_add(total as u64);
        #[cfg(unix)]
        let n = {
            use std::os::unix::fs::FileExt;
            match file.read_at(&mut buf[total..], at) {
                Ok(n) => n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        };
        #[cfg(not(unix))]
        let n = {
            use std::io::{Read, Seek};
            let mut f = file;
            f.seek(std::io::SeekFrom::Start(at))?;
            match f.read(&mut buf[total..]) {
                Ok(n) => n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        };
        if n == 0 {
            break; // end of file
        }
        total += n;
    }
    Ok(total)
}

/// Write `data` completely to `file` at its current position with plain `write(2)` calls
/// straight from the caller's memory (`&File` implements `Write`, so a shared handle works).
pub(crate) fn write_all_std(file: &std::fs::File, data: &[u8]) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = file;
    file.write_all(data)
}

/// Write a BORROWED slice completely to `file` without copying it (Rule #1).
///
/// A borrowed `&[u8]` cannot be moved into `spawn_blocking` without a copy, so the write runs on
/// the calling task's thread: on the multi-thread runtime `block_in_place` hands the worker's
/// other tasks to a fresh thread first (the same off-loading `spawn_blocking` gives); a
/// current-thread runtime has no other thread to hand them to (and `block_in_place` would panic
/// there), so the write simply runs inline. Owned data should use `spawn_blocking` instead.
pub(crate) fn write_borrowed_blocking(file: &std::fs::File, data: &[u8]) -> std::io::Result<()> {
    match tokio::runtime::Handle::try_current().map(|h| h.runtime_flavor()) {
        Ok(tokio::runtime::RuntimeFlavor::MultiThread) => {
            tokio::task::block_in_place(|| write_all_std(file, data))
        }
        _ => write_all_std(file, data),
    }
}

/// Streaming writer for filesystem operations
///
/// Rule #1 (zero copy): the file is a plain `std::fs::File` written directly from the
/// caller's bytes. `tokio::fs::File::write_all` copies every chunk into tokio's own internal
/// buffer (up to 2 MiB per call) before a blocking task writes it, which is a payload copy
/// the data never needed.
pub struct FileSystemWriter {
    file: Option<Arc<std::fs::File>>,
    path: PathBuf,
    bytes_written: u64,
    compressed_bytes: u64,
    finalized: bool,
    hasher: Crc32c,
    compression: CompressionConfig,
    compressor: Option<zstd::Encoder<'static, Vec<u8>>>,
    /// The blocking write of an owned chunk that has been submitted and not yet seen to finish.
    ///
    /// It lives HERE, not in the `write_owned_bytes` future: a caller may drop that future, and
    /// a write that nobody tracks any more could run beside the next write (same cursor, wrong
    /// order, wrong checksum) or outlive `finalize`'s sync. Every operation that touches the
    /// file first calls `drain_pending`, which waits for it and reports its error.
    pending_write: Option<tokio::task::JoinHandle<std::io::Result<()>>>,
    /// Test-only gate: the blocking write of an owned chunk waits here (see `tests::WriteGate`).
    #[cfg(test)]
    write_gate: Option<Arc<tests::WriteGate>>,
}

impl FileSystemWriter {
    async fn new(path: PathBuf) -> Result<Self> {
        Self::new_with_compression(path, CompressionConfig::None).await
    }

    pub async fn new_with_compression(
        path: PathBuf,
        compression: CompressionConfig,
    ) -> Result<Self> {
        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }

        // Determine final path with compression extension if needed
        // (the one suffix rule, shared with the checkpoint manifest names)
        let final_path = compression.stored_path(&path);

        // Create and open the file for writing
        let file = fs::File::create(&final_path).await?.into_std().await;

        let compressor = match compression {
            CompressionConfig::None => None,
            CompressionConfig::Zstd { level } => {
                let mut encoder = zstd::Encoder::new(Vec::new(), level)?;
                encoder.include_checksum(false)?; // We handle checksums ourselves
                Some(encoder)
            }
        };

        Ok(Self {
            file: Some(Arc::new(file)),
            path: final_path,
            bytes_written: 0,
            compressed_bytes: 0,
            finalized: false,
            hasher: Crc32c::new(),
            compression,
            compressor,
            pending_write: None,
            #[cfg(test)]
            write_gate: None,
        })
    }

    /// Wait for the owned write still in flight, if any, and report its outcome (an I/O error,
    /// or the failure of the blocking task itself).
    ///
    /// The handle is awaited by reference and cleared only after it completed: if the caller
    /// drops THIS future mid-wait, the write stays tracked and the next call drains it.  The
    /// outcome is reported once; the call that observes it consumes it.
    async fn drain_pending(&mut self) -> Result<()> {
        let Some(handle) = self.pending_write.as_mut() else {
            return Ok(());
        };
        let joined = handle.await;
        self.pending_write = None;
        joined
            .map_err(|e| anyhow::anyhow!("owned file write task did not complete: {e}"))?
            .context("owned file write failed")?;
        Ok(())
    }
}

#[async_trait]
impl ObjectWriter for FileSystemWriter {
    async fn write_chunk(&mut self, chunk: &[u8]) -> Result<()> {
        debug!(
            "FileSystemWriter::write_chunk: writing {} bytes",
            chunk.len()
        );
        if self.finalized {
            bail!("Cannot write to finalized writer");
        }
        // An earlier owned write (possibly abandoned by its caller) must finish first, in order.
        self.drain_pending().await?;

        // Update checksum with original data (before compression)
        self.hasher.update(chunk);
        self.bytes_written += chunk.len() as u64;

        if let Some(ref mut file) = self.file {
            if let Some(ref mut compressor) = self.compressor {
                // Compress the chunk using std::io::Write trait
                use std::io::Write;
                // (Compression is a transform: its output is new data, not a copy of the input.)
                compressor.write_all(chunk)?;

                // Don't update compressed_bytes here - we don't know the compressed size until finalize()
            } else {
                // No compression - write straight from the caller's slice, no intermediate
                // buffer (see `write_borrowed_blocking` for why it runs on this thread).
                // Trade-off, documented: on a multi-thread runtime the write is off-loaded with
                // `block_in_place`; on a CURRENT-THREAD runtime it runs inline and blocks that
                // thread for the duration of the write (a borrowed slice cannot be moved to
                // another thread without the copy Rule #1 forbids).
                // Owned data goes through `write_owned_bytes`, which uses `spawn_blocking`.
                write_borrowed_blocking(file, chunk)?;
            }
            Ok(())
        } else {
            bail!("Writer has been finalized or cancelled");
        }
    }

    /// Owned data goes to a blocking task that writes it as-is: no copy, and the runtime's
    /// workers are never blocked.
    async fn write_owned_bytes(&mut self, data: Vec<u8>) -> Result<()> {
        if self.finalized {
            bail!("Cannot write to finalized writer");
        }
        // The previous owned write, if any, finishes (and reports its error) first; this must
        // come before the counters and the checksum change, so a dropped wait changes nothing.
        self.drain_pending().await?;
        if self.compressor.is_some() {
            // Compression path: the encoder consumes the slice (a transform).
            return self.write_chunk(&data).await;
        }
        let Some(file) = self.file.as_ref().map(Arc::clone) else {
            bail!("Writer has been finalized or cancelled");
        };
        self.hasher.update(&data);
        self.bytes_written += data.len() as u64;
        #[cfg(test)]
        let gate = self.write_gate.clone();
        // The Vec is moved into the job: no copy.  The handle is kept in the writer (not only
        // in this future), so dropping this future cannot detach the write unseen.
        self.pending_write = Some(tokio::task::spawn_blocking(move || {
            #[cfg(test)]
            if let Some(gate) = gate {
                gate.pass()?;
            }
            write_all_std(&file, &data)
        }));
        self.drain_pending().await
    }

    async fn finalize(mut self: Box<Self>) -> Result<()> {
        debug!("FileSystemWriter::finalize: starting finalization");
        if self.finalized {
            return Ok(());
        }

        // Never sync (or return) while an earlier owned write is still running.
        self.drain_pending().await?;

        if let Some(file) = self.file.take() {
            // Finalize compression if enabled
            let final_compressed = match self.compressor.take() {
                Some(compressor) => {
                    debug!(
                        "FileSystemWriter::finalize: finalizing compression, path: {}",
                        self.path.display()
                    );
                    let out = compressor.finish()?;
                    self.compressed_bytes = out.len() as u64; // Update with actual compressed size
                    Some(out)
                }
                None => None,
            };

            // The compressed tail is an owned Vec: moved into the blocking task, not copied.
            tokio::task::spawn_blocking(move || -> std::io::Result<()> {
                if let Some(out) = final_compressed {
                    write_all_std(&file, &out)?;
                }
                file.sync_all()
            })
            .await??;
        }

        self.finalized = true;
        Ok(())
    }

    fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    fn checksum(&self) -> Option<String> {
        Some(format!("crc32c:{:08x}", self.hasher.clone().finalize()))
    }

    fn compressed_bytes(&self) -> u64 {
        self.compressed_bytes
    }

    fn compression(&self) -> CompressionConfig {
        self.compression
    }

    fn compression_ratio(&self) -> f64 {
        if self.bytes_written == 0 || !self.compression.is_enabled() {
            1.0
        } else if self.compressed_bytes == 0 {
            // Not yet finalized - can't determine ratio
            1.0
        } else {
            self.compressed_bytes as f64 / self.bytes_written as f64
        }
    }

    async fn cancel(mut self: Box<Self>) -> Result<()> {
        self.finalized = true;
        // Let a running write finish before the file is removed, so it cannot outlive the
        // cleanup; its outcome no longer matters.
        if let Err(e) = self.drain_pending().await {
            debug!("FileSystemWriter::cancel: discarding the failure of a pending write: {e:#}");
        }
        if let Some(_file) = self.file.take() {
            // File will be closed when dropped
        }

        // Remove the partial file
        if self.path.exists() {
            let _ = fs::remove_file(&self.path).await; // Ignore errors
        }

        Ok(())
    }
}

impl Default for FileSystemObjectStore {
    fn default() -> Self {
        Self::new()
    }
}

impl FileSystemObjectStore {
    /// Create a new FileSystemObjectStore with default configuration
    ///
    /// v0.9.10: Uses TTL=0 for size cache (effectively disabled) because:
    /// - File metadata can change rapidly on disk
    /// - Local stat operations are fast (<1ms vs 10-50ms for network storage)
    /// - Cache provides no meaningful performance benefit
    pub fn new() -> Self {
        use std::time::Duration;
        Self {
            config: Arc::new(FileSystemConfig::default()),
            size_cache: Arc::new(crate::object_size_cache::ObjectSizeCache::new(
                Duration::from_secs(0),
            )),
        }
    }

    /// Create with custom configuration
    ///
    /// v0.9.10: TTL=0 for file:// backend (no caching needed for local filesystem)
    pub fn with_config(config: FileSystemConfig) -> Self {
        use std::time::Duration;
        Self {
            config: Arc::new(config),
            size_cache: Arc::new(crate::object_size_cache::ObjectSizeCache::new(
                Duration::from_secs(0),
            )),
        }
    }

    /// Convert a URI to a filesystem path.
    ///
    /// `pub(crate)` so the op-log's tests can check that its `file://` handling
    /// resolves to the same path (src/s3_logger.rs).
    pub(crate) fn uri_to_path(uri: &str) -> Result<PathBuf> {
        if !uri.starts_with("file://") {
            bail!("FileSystemObjectStore expects file:// URI, got: {}", uri);
        }

        // Strip file:// prefix
        let path = &uri[7..];
        Ok(PathBuf::from(path))
    }

    /// Convert a filesystem path back to a URI for list operations
    fn path_to_uri(path: &Path) -> String {
        if path.is_absolute() {
            format!("file://{}", path.display())
        } else {
            path.display().to_string()
        }
    }

    /// Create ObjectMetadata from filesystem metadata
    async fn metadata_from_path(path: &Path) -> Result<ObjectMetadata> {
        let metadata = fs::metadata(path).await?;

        // Create ObjectMetadata compatible with S3ObjectStat
        let last_modified = metadata.modified().ok().and_then(modified_time_rfc3339);

        let file_hash = crate::file_get_into::stat_etag(metadata.len(), metadata.modified().ok());

        Ok(ObjectMetadata {
            size: metadata.len(),
            last_modified,
            e_tag: Some(file_hash),
            content_type: None, // Could infer from file extension
            content_language: None,
            content_encoding: None,
            cache_control: None,
            content_disposition: None,
            expires: None,
            storage_class: Some("STANDARD".to_string()), // Default for files
            server_side_encryption: None,
            ssekms_key_id: None,
            sse_customer_algorithm: None,
            version_id: None,
            replication_status: None,
            metadata: HashMap::new(),
        })
    }

    /// Recursively collect files in a directory
    async fn collect_files_recursive(
        dir: &Path,
        prefix: &str,
        results: &mut Vec<String>,
    ) -> Result<()> {
        let mut entries = fs::read_dir(dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let entry_path = entry.path();
            let file_name = entry_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("");

            if entry_path.is_dir() {
                let new_prefix = if prefix.is_empty() {
                    file_name.to_string()
                } else {
                    format!("{}/{}", prefix, file_name)
                };
                Box::pin(Self::collect_files_recursive(
                    &entry_path,
                    &new_prefix,
                    results,
                ))
                .await?;
            } else {
                let file_uri = Self::path_to_uri(&entry_path);
                results.push(file_uri);
            }
        }

        Ok(())
    }

    /// Download using RangeEngine for concurrent range requests
    async fn get_with_range_engine(&self, uri: &str, file_size: u64) -> Result<Bytes> {
        let engine = RangeEngine::new(self.config.range_engine.clone());

        // Rule #1 (zero copy): every range is read with a positional read straight into its own
        // segment of the engine's ONE master buffer (`download_into_master`). Going through
        // `get_range` would allocate a `Bytes` per range and copy each into the master buffer.
        // The page-cache hint is the one `get_range` applies to range reads.
        let path = Self::uri_to_path(uri)?;
        let cache_mode = self.config.page_cache_mode.unwrap_or(PageCacheMode::Random);
        let fill_range = move |offset: u64, mut segment: BytesMut| {
            let path = path.clone();
            async move {
                tokio::task::spawn_blocking(move || -> Result<(BytesMut, usize)> {
                    let file = std::fs::File::open(&path)?;
                    let _ = apply_page_cache_hint(&file, cache_mode, file_size);
                    let read = read_full_at_std(&file, &mut segment, offset)?;
                    Ok((segment, read))
                })
                .await?
            }
        };

        let (bytes, stats) = engine
            .download_into_master(file_size, false, fill_range, None)
            .await?;

        info!(
            "RangeEngine downloaded {} bytes in {} ranges: {:.2} MB/s ({:.2} Gbps)",
            stats.bytes_downloaded,
            stats.ranges_processed,
            stats.throughput_mbps(),
            stats.throughput_gbps()
        );

        Ok(bytes)
    }

    /// Simple sequential read for small files
    async fn get_simple(&self, uri: &str, file_size: u64) -> Result<Bytes> {
        let path = Self::uri_to_path(uri)?;

        // Open file and apply page cache hint based on size
        // Apply page cache hint (use configured mode or Auto by default)
        let cache_mode = self.config.page_cache_mode.unwrap_or(PageCacheMode::Auto);
        debug!(
            "Applying page cache hint {:?} for file {} (size: {} bytes)",
            cache_mode, uri, file_size
        );
        let capacity = usize::try_from(file_size)
            .map_err(|_| anyhow::anyhow!("file {uri} of {file_size} bytes is too large to read"))?;

        // Rule #1 (zero copy): read with plain `std::fs` in a blocking task, straight into the
        // destination Vec (which becomes the returned `Bytes`). `tokio::fs::File::read*` reads
        // into tokio's own internal buffer (up to 2 MiB) and copies from there into the caller's
        // buffer, one extra copy of every byte.
        let data = tokio::task::spawn_blocking(move || -> Result<Vec<u8>> {
            use std::io::Read;
            let mut file = std::fs::File::open(&path)?;
            let _ = apply_page_cache_hint(&file, cache_mode, file_size);
            // Pre-allocate to avoid reallocation churn (v0.9.9+)
            let mut data = Vec::with_capacity(capacity);
            file.read_to_end(&mut data)?;
            Ok(data)
        })
        .await??;

        // Convert to Bytes (cheap, just wraps in Arc)
        Ok(Bytes::from(data))
    }
}

/// Create a directory and any missing parents (`mkdir -p`); an existing directory is
/// success. Shared by the `file://` and `direct://` stores: O_DIRECT has no meaning
/// for a directory operation, so `direct://` behaves exactly like `file://` (#229).
pub(crate) async fn mkdir_path(path: &Path) -> Result<()> {
    tokio::fs::create_dir_all(path)
        .await
        .with_context(|| format!("Failed to create directory: {}", path.display()))
}

/// Remove a directory. `recursive == false` removes only an EMPTY directory (a
/// non-empty one is an error and nothing is removed); `recursive == true` removes
/// the whole tree. A directory that does not exist is success, so the call is
/// idempotent. Shared by the `file://` and `direct://` stores (#229).
pub(crate) async fn rmdir_path(path: &Path, recursive: bool) -> Result<()> {
    let result = if recursive {
        tokio::fs::remove_dir_all(path).await
    } else {
        tokio::fs::remove_dir(path).await
    };

    match result {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            // Directory doesn't exist - treat as success (idempotent operation)
            tracing::debug!(
                "Directory already removed or never existed: {}",
                path.display()
            );
            Ok(())
        }
        Err(e) if recursive => Err(e)
            .with_context(|| format!("Failed to remove directory recursively: {}", path.display())),
        Err(e) => Err(e).with_context(|| {
            format!(
                "Failed to remove directory (not empty?): {}",
                path.display()
            )
        }),
    }
}

#[async_trait]
impl ObjectStore for FileSystemObjectStore {
    async fn get(&self, uri: &str) -> Result<Bytes> {
        if !uri.starts_with("file://") {
            bail!("FileSystemObjectStore expected file:// URI");
        }
        let path = Self::uri_to_path(uri)?;
        debug!("FileSystemObjectStore::get uri='{}'", uri);

        if !path.exists() {
            bail!("File not found: {}", path.display());
        }

        if !path.is_file() {
            bail!("Path is not a file: {}", path.display());
        }

        // Get file size to decide download strategy
        let metadata = fs::metadata(&path).await?;
        let file_size = metadata.len();

        // Use RangeEngine for large files if enabled
        if self.config.enable_range_engine && file_size >= self.config.range_engine.min_split_size {
            trace!(
                "File size {} >= threshold {}, using RangeEngine for {}",
                file_size,
                self.config.range_engine.min_split_size,
                uri
            );
            return self.get_with_range_engine(uri, file_size).await;
        }

        // Simple sequential read for small files
        trace!(
            "File size {} < threshold {}, using simple read for {}",
            file_size,
            self.config.range_engine.min_split_size,
            uri
        );
        self.get_simple(uri, file_size).await
    }

    /// Caller-buffer GET: `pread` straight into `dst` (#295, #267); see
    /// [`crate::file_get_into`] for the contract, the zero-copy and the cancellation story.
    async fn get_into(&self, uri: &str, dst: &mut [u8], progress: &ReceiveProgress) -> Result<()> {
        if !uri.starts_with("file://") {
            bail!("FileSystemObjectStore expected file:// URI");
        }
        let path = Self::uri_to_path(uri)?;
        debug!(
            "FileSystemObjectStore::get_into uri='{}', expected_len={}",
            uri,
            dst.len()
        );
        let spec = crate::file_get_into::LocalRead {
            path: &path,
            provider: "file",
            direct: None,
            page_cache: Some(self.config.page_cache_mode.unwrap_or(PageCacheMode::Auto)),
        };
        crate::file_get_into::get_into(&spec, dst, progress).await
    }

    /// Typed HEAD from `stat`: `Ok(None)` for a missing file, an error for a directory.
    async fn head(&self, uri: &str) -> Result<Option<HeadMetadata>> {
        if !uri.starts_with("file://") {
            bail!("FileSystemObjectStore expected file:// URI");
        }
        let path = Self::uri_to_path(uri)?;
        debug!("FileSystemObjectStore::head uri='{}'", uri);
        crate::file_get_into::head(&path, "file").await
    }

    async fn get_range(&self, uri: &str, offset: u64, length: Option<u64>) -> Result<Bytes> {
        if !uri.starts_with("file://") {
            bail!("FileSystemObjectStore expected file:// URI");
        }
        let path = Self::uri_to_path(uri)?;
        debug!(
            "FileSystemObjectStore::get_range uri='{}', offset={}, length={:?}",
            uri, offset, length
        );

        if !path.exists() {
            bail!("File not found: {}", path.display());
        }

        if !path.is_file() {
            bail!("Path is not a file: {}", path.display());
        }

        // Get file size and apply page cache hint
        let metadata = fs::metadata(&path).await?;
        let file_size = metadata.len();

        // Apply page cache hint (use configured mode, or Random for range reads by default)
        let cache_mode = self.config.page_cache_mode.unwrap_or(PageCacheMode::Random);
        debug!("Applying page cache hint {:?} for range read {} offset={} length={:?} (file size: {} bytes)", 
               cache_mode, uri, offset, length, file_size);

        // Calculate how many bytes are actually available from offset to end of file
        let bytes_available = file_size.saturating_sub(offset);
        let read_length = match length {
            Some(len) => len.min(bytes_available), // Read at most what's available
            None => bytes_available,               // Read to end
        };

        if read_length == 0 {
            // Nothing to read (offset >= file_size)
            return Ok(Bytes::new());
        }
        let read_len = usize::try_from(read_length)
            .map_err(|_| anyhow::anyhow!("range of {read_length} bytes is too large to read"))?;

        // Rule #1 (zero copy): a positional read with plain `std::fs` in a blocking task, straight
        // into the destination Vec (which becomes the returned `Bytes`). `tokio::fs::File::read*`
        // goes through tokio's internal buffer (up to 2 MiB) and copies from there, an extra copy
        // of every byte.
        let buffer = tokio::task::spawn_blocking(move || -> Result<Vec<u8>> {
            let file = std::fs::File::open(&path)?;
            let _ = apply_page_cache_hint(&file, cache_mode, file_size);
            let mut buffer = vec![0u8; read_len];
            read_exact_at_std(&file, &mut buffer, offset)?;
            Ok(buffer)
        })
        .await??;

        // Convert to Bytes (cheap, just wraps in Arc)
        Ok(Bytes::from(buffer))
    }

    async fn put(&self, uri: &str, data: Bytes) -> Result<()> {
        if !uri.starts_with("file://") {
            bail!("FileSystemObjectStore expected file:// URI");
        }
        let path = Self::uri_to_path(uri)?;
        debug!(
            "FileSystemObjectStore::put uri='{}', {} bytes",
            uri,
            data.len()
        );

        // Create parent directories if they don't exist
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }

        // `tokio::fs::write` copies its input into an owned buffer (issue #238), so a
        // large PUT briefly held two full copies. Move the `Bytes` into a blocking task
        // and write from its slice: the payload is never duplicated.
        tokio::task::spawn_blocking(move || std::fs::write(&path, data.as_ref())).await??;
        Ok(())
    }

    async fn put_multipart(&self, uri: &str, data: Bytes, _part_size: Option<usize>) -> Result<()> {
        if !uri.starts_with("file://") {
            bail!("FileSystemObjectStore expected file:// URI");
        }
        debug!(
            "FileSystemObjectStore::put_multipart uri='{}', {} bytes",
            uri,
            data.len()
        );
        // For filesystem, multipart is the same as regular put
        // In a more sophisticated implementation, we could write in chunks
        self.put(uri, data).await
    }

    async fn list(&self, uri_prefix: &str, recursive: bool) -> Result<Vec<String>> {
        if !uri_prefix.starts_with("file://") {
            bail!("FileSystemObjectStore expected file:// URI");
        }
        let base_path = Self::uri_to_path(uri_prefix)?;
        debug!(
            "FileSystemObjectStore::list prefix='{}', recursive={}",
            uri_prefix, recursive
        );
        let mut results = Vec::new();

        if !base_path.exists() {
            return Ok(results); // Empty list for non-existent paths
        }

        if base_path.is_file() {
            // If the prefix points to a file, return just that file
            results.push(Self::path_to_uri(&base_path));
            return Ok(results);
        }

        if base_path.is_dir() {
            if recursive {
                Self::collect_files_recursive(&base_path, "", &mut results).await?;
            } else {
                // Non-recursive: direct children — files as-is, directories with trailing slash
                // (mirrors S3 "common prefix" semantics so callers can distinguish them).
                let mut entries = fs::read_dir(&base_path).await?;
                while let Some(entry) = entries.next_entry().await? {
                    let entry_path = entry.path();
                    if entry_path.is_file() {
                        results.push(Self::path_to_uri(&entry_path));
                    } else if entry_path.is_dir() {
                        let mut dir_uri = Self::path_to_uri(&entry_path);
                        if !dir_uri.ends_with('/') {
                            dir_uri.push('/');
                        }
                        results.push(dir_uri);
                    }
                }
            }
        }

        Ok(results)
    }

    fn list_stream<'a>(
        &'a self,
        uri_prefix: &'a str,
        recursive: bool,
    ) -> std::pin::Pin<Box<dyn futures::stream::Stream<Item = Result<String>> + Send + 'a>> {
        Box::pin(async_stream::stream! {
            // For file://, delegate to buffered list() since filesystem operations are fast
            // Streaming doesn't provide significant benefit for local filesystem
            match self.list(uri_prefix, recursive).await {
                Ok(keys) => {
                    for key in keys {
                        yield Ok(key);
                    }
                }
                Err(e) => yield Err(e),
            }
        })
    }

    async fn stat(&self, uri: &str) -> Result<ObjectMetadata> {
        if !uri.starts_with("file://") {
            bail!("FileSystemObjectStore expected file:// URI");
        }
        let path = Self::uri_to_path(uri)?;
        debug!("FileSystemObjectStore::stat uri='{}'", uri);

        if !path.exists() {
            bail!("File not found: {}", path.display());
        }

        if !path.is_file() {
            bail!("Path is not a file: {}", path.display());
        }

        Self::metadata_from_path(&path).await
    }

    async fn delete(&self, uri: &str) -> Result<()> {
        if !uri.starts_with("file://") {
            bail!("FileSystemObjectStore expected file:// URI");
        }
        let path = Self::uri_to_path(uri)?;
        debug!("FileSystemObjectStore::delete uri='{}'", uri);

        if !path.exists() {
            // Already deleted, consider it success
            return Ok(());
        }

        if path.is_file() {
            fs::remove_file(&path).await?;
        } else if path.is_dir() {
            fs::remove_dir_all(&path).await?;
        }

        Ok(())
    }

    async fn delete_batch(&self, uris: &[String]) -> Result<()> {
        // FileSystemObjectStore: delete files concurrently
        use futures::stream::{self, StreamExt};
        debug!("FileSystemObjectStore::delete_batch {} URIs", uris.len());

        let max_concurrency = (uris.len() / 10).clamp(10, 100);
        let uris_owned: Vec<String> = uris.to_vec();

        let deletions = stream::iter(uris_owned)
            .map(|uri| async move { self.delete(&uri).await })
            .buffer_unordered(max_concurrency);

        let results: Vec<Result<()>> = deletions.collect().await;
        for result in results {
            result?;
        }

        Ok(())
    }

    async fn delete_prefix(&self, uri_prefix: &str) -> Result<()> {
        if !uri_prefix.starts_with("file://") {
            bail!("FileSystemObjectStore expected file:// URI");
        }
        let base_path = Self::uri_to_path(uri_prefix)?;
        debug!(
            "FileSystemObjectStore::delete_prefix prefix='{}'",
            uri_prefix
        );

        if !base_path.exists() {
            return Ok(()); // Nothing to delete
        }

        if base_path.is_file() {
            fs::remove_file(&base_path).await?;
        } else if base_path.is_dir() {
            // Collect all files under the prefix and delete them
            let files = self.list(uri_prefix, true).await?;
            for file_uri in files {
                self.delete(&file_uri).await?;
            }

            // Try to remove the directory if it's empty
            if (fs::remove_dir(&base_path).await).is_err() {
                // Directory might not be empty due to subdirectories
                // For now, we'll leave non-empty directories
            }
        }

        Ok(())
    }

    async fn create_container(&self, name: &str) -> Result<()> {
        let path = PathBuf::from(name);
        fs::create_dir_all(&path).await?;
        Ok(())
    }

    async fn delete_container(&self, name: &str) -> Result<()> {
        let path = PathBuf::from(name);

        if !path.exists() {
            return Ok(()); // Already deleted
        }

        if path.is_dir() {
            fs::remove_dir(&path).await?; // Only removes empty directories
        } else {
            bail!("Path is not a directory: {}", path.display());
        }

        Ok(())
    }

    /// `rename` is the filesystem's atomic `rename(2)` (below), not copy + delete.
    fn has_native_rename(&self) -> bool {
        true
    }

    async fn rename(&self, src_uri: &str, dst_uri: &str) -> Result<()> {
        if !src_uri.starts_with("file://") {
            bail!("FileSystemObjectStore expected file:// URI for source");
        }
        if !dst_uri.starts_with("file://") {
            bail!("FileSystemObjectStore expected file:// URI for destination");
        }

        let src_path = Self::uri_to_path(src_uri)?;
        let dst_path = Self::uri_to_path(dst_uri)?;

        if !src_path.exists() {
            bail!("Source file not found: {}", src_path.display());
        }

        // Create parent directories for destination if they don't exist
        if let Some(parent) = dst_path.parent() {
            fs::create_dir_all(parent).await?;
        }

        // Use tokio::fs::rename for atomic filesystem rename operation
        fs::rename(&src_path, &dst_path).await?;
        Ok(())
    }

    async fn get_writer(&self, uri: &str) -> Result<Box<dyn ObjectWriter>> {
        if !uri.starts_with("file://") {
            bail!("FileSystemObjectStore expected file:// URI");
        }

        let path = Self::uri_to_path(uri)?;
        let writer = FileSystemWriter::new(path).await?;
        Ok(Box::new(writer))
    }

    async fn get_writer_with_compression(
        &self,
        uri: &str,
        compression: CompressionConfig,
    ) -> Result<Box<dyn ObjectWriter>> {
        if !uri.starts_with("file://") {
            bail!("FileSystemObjectStore expected file:// URI");
        }

        let path = Self::uri_to_path(uri)?;
        let writer = FileSystemWriter::new_with_compression(path, compression).await?;
        Ok(Box::new(writer))
    }

    async fn create_writer(
        &self,
        uri: &str,
        options: WriterOptions,
    ) -> Result<Box<dyn ObjectWriter>> {
        if !uri.starts_with("file://") {
            bail!("FileSystemObjectStore expected file:// URI");
        }

        let path = Self::uri_to_path(uri)?;
        if let Some(compression) = options.compression {
            let writer = FileSystemWriter::new_with_compression(path, compression).await?;
            Ok(Box::new(writer))
        } else {
            let writer = FileSystemWriter::new(path).await?;
            Ok(Box::new(writer))
        }
    }

    /// Pre-stat objects and populate the size cache
    ///
    /// v0.9.10: For file:// backend, size_cache has TTL=0 (effectively disabled)
    /// because local filesystem metadata operations are fast (~1ms vs 10-50ms for
    /// network storage) and files can change rapidly on disk.
    ///
    /// This method still works to maintain API compatibility, but provides minimal
    /// performance benefit for local files.
    async fn pre_stat_and_cache(&self, uris: &[String], max_concurrent: usize) -> Result<usize> {
        // Use default concurrent pre_stat_objects implementation
        let size_map = self.pre_stat_objects(uris, max_concurrent).await?;

        // Populate size cache with results (TTL=0 means immediate expiration)
        self.seed_size_cache(&size_map).await;

        Ok(size_map.len())
    }

    /// Fill the size cache with sizes the caller already learned from `stat` calls
    /// (no request is made); see [`ObjectStore::seed_size_cache`].
    async fn seed_size_cache(&self, sizes: &HashMap<String, u64>) {
        for (uri, size) in sizes.iter() {
            self.size_cache.put(uri.clone(), *size).await;
        }
    }
    // =========================================================================
    // Metadata Operations - ported from sai3-bench fs_metadata.rs
    // =========================================================================

    async fn mkdir(&self, uri: &str) -> Result<()> {
        let path = Self::uri_to_path(uri)?;
        mkdir_path(&path).await
    }

    async fn rmdir(&self, uri: &str, recursive: bool) -> Result<()> {
        let path = Self::uri_to_path(uri)?;
        rmdir_path(&path, recursive).await
    }

    async fn update_metadata(&self, _uri: &str, _metadata: &HashMap<String, String>) -> Result<()> {
        // File systems don't support custom metadata keys like cloud storage
        bail!("Custom metadata not supported for file:// backend")
    }

    async fn update_properties(&self, uri: &str, properties: &ObjectProperties) -> Result<()> {
        let path = Self::uri_to_path(uri)?;

        // File systems have limited property support
        // We can only really handle storage_class as a hint (ignored for now)
        if properties.content_type.is_some()
            || properties.cache_control.is_some()
            || properties.content_encoding.is_some()
        {
            bail!("HTTP properties (content-type, cache-control, etc.) not supported for file:// backend")
        }

        // storage_class could map to filesystem features (btrfs compression, ZFS tiers)
        // but for now we just accept it as a no-op
        if let Some(ref storage_class) = properties.storage_class {
            tracing::debug!(
                "Ignoring storage_class '{}' for file:// backend at {}",
                storage_class,
                path.display()
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod modified_time_tests {
    use super::modified_time_rfc3339;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    fn at(secs: u64, nanos: u32) -> SystemTime {
        UNIX_EPOCH + Duration::new(secs, nanos)
    }

    /// A whole-second UTC time is an RFC 3339 string, not a `SystemTime { .. }` dump (#263).
    #[test]
    fn utc_time_is_rfc3339() {
        assert_eq!(
            modified_time_rfc3339(at(1_747_388_772, 0)).as_deref(),
            Some("2025-05-16T09:46:12Z")
        );
        assert_eq!(
            modified_time_rfc3339(UNIX_EPOCH).as_deref(),
            Some("1970-01-01T00:00:00Z")
        );
    }

    /// The fraction the filesystem reported is kept, down to the nanosecond.
    #[test]
    fn fractional_time_keeps_its_precision() {
        for (nanos, want) in [
            (500_000_000, "2025-05-16T09:46:12.500Z"),
            (123_456_000, "2025-05-16T09:46:12.123456Z"),
            (123_456_789, "2025-05-16T09:46:12.123456789Z"),
        ] {
            assert_eq!(
                modified_time_rfc3339(at(1_747_388_772, nanos)).as_deref(),
                Some(want),
                "{nanos} ns"
            );
        }
    }

    /// The output parses back to the same instant, so nothing but formatting changed.
    #[test]
    fn output_round_trips_to_the_same_instant() {
        let text = modified_time_rfc3339(at(1_900_000_000, 7_008_009)).expect("present");
        let parsed = chrono::DateTime::parse_from_rfc3339(&text).expect("RFC 3339");
        assert_eq!(parsed.timestamp(), 1_900_000_000);
        assert_eq!(parsed.timestamp_subsec_nanos(), 7_008_009);
    }

    /// A time with no RFC 3339 form is absent, never an empty string or a made-up time: before the
    /// epoch, and past the range the calendar type can hold.
    #[test]
    fn unrepresentable_time_is_absent() {
        assert_eq!(
            modified_time_rfc3339(UNIX_EPOCH - Duration::from_secs(1)),
            None
        );
        assert_eq!(modified_time_rfc3339(at(10_000_000_000_000, 0)), None);
    }
}

/// Tests for the ordering and completion tracking of an owned write (stage G review, G-F3).
///
/// A blocking write is held at a gate so the tests control exactly when it can finish: no test
/// here depends on how long a write takes. The "must not overtake" checks wait a fixed
/// interval, which can only make a regression pass by luck of slowness, never fail a correct
/// implementation.
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::sync::mpsc::{error::TryRecvError, UnboundedReceiver};

    /// A short wait for something that must NOT happen.
    const NEGATIVE_WAIT: Duration = Duration::from_millis(300);
    /// The deadline for everything that must happen; a regression fails instead of hanging.
    const DEADLINE: Duration = Duration::from_secs(10);

    /// Holds the blocking write of an owned chunk until the test releases it.
    pub(super) struct WriteGate {
        entered: tokio::sync::mpsc::UnboundedSender<()>,
        release: std::sync::Mutex<std::sync::mpsc::Receiver<bool>>,
    }

    impl WriteGate {
        /// Runs on the blocking thread: announce arrival, then wait for the test's verdict
        /// (`true` = go on and write, `false` = fail). A dropped test (sender gone) or a stuck
        /// one (20 s) also fails the write, so no blocking thread outlives a failing test.
        pub(super) fn pass(&self) -> std::io::Result<()> {
            self.entered
                .send(())
                .map_err(|_| std::io::Error::other("write gate: test is gone"))?;
            let release = self
                .release
                .lock()
                .map_err(|_| std::io::Error::other("write gate poisoned"))?;
            match release.recv_timeout(Duration::from_secs(20)) {
                Ok(true) => Ok(()),
                Ok(false) => Err(std::io::Error::other("injected write failure")),
                Err(_) => Err(std::io::Error::other("write gate: never released")),
            }
        }
    }

    struct Rig {
        writer: FileSystemWriter,
        entered: UnboundedReceiver<()>,
        release: std::sync::mpsc::Sender<bool>,
        dir: tempfile::TempDir,
    }

    impl Rig {
        async fn new() -> Self {
            let dir = tempfile::tempdir().expect("temp dir");
            let mut writer = FileSystemWriter::new_with_compression(
                dir.path().join("out.bin"),
                CompressionConfig::None,
            )
            .await
            .expect("writer");
            let (entered_tx, entered) = tokio::sync::mpsc::unbounded_channel();
            let (release, release_rx) = std::sync::mpsc::channel();
            writer.write_gate = Some(Arc::new(WriteGate {
                entered: entered_tx,
                release: std::sync::Mutex::new(release_rx),
            }));
            Self {
                writer,
                entered,
                release,
                dir,
            }
        }

        fn file(&self) -> PathBuf {
            self.dir.path().join("out.bin")
        }

        /// Submit an owned write and DROP its future while the blocking job is held at the gate.
        async fn submit_and_drop(&mut self, data: Vec<u8>) {
            // `timeout` polls the future once before it looks at the clock: the write is
            // submitted, cannot complete (it is gated), and the future is dropped.
            let outcome =
                tokio::time::timeout(Duration::ZERO, self.writer.write_owned_bytes(data)).await;
            assert!(outcome.is_err(), "a gated write must not complete");
            self.arrived().await;
        }

        /// Wait until a blocking write job has reached the gate.
        async fn arrived(&mut self) {
            tokio::time::timeout(DEADLINE, self.entered.recv())
                .await
                .expect("a write job reached the gate in time")
                .expect("gate channel open");
        }

        fn fail_next(&self) {
            self.release.send(false).expect("gate open");
        }
    }

    /// No further job may have reached the gate.
    fn assert_no_second_job(entered: &mut UnboundedReceiver<()>, what: &str) {
        assert_eq!(
            entered.try_recv(),
            Err(TryRecvError::Empty),
            "{what}: a second write job started while the first was still pending"
        );
    }

    fn allow(release: &std::sync::mpsc::Sender<bool>) {
        release.send(true).expect("gate open");
    }

    fn crc_label(data: &[u8]) -> String {
        format!("crc32c:{:08x}", crc32c::crc32c(data))
    }

    /// A second write must wait for a dropped write, and then land after it.
    #[tokio::test]
    async fn next_write_waits_for_a_dropped_owned_write_and_keeps_order() {
        let mut rig = Rig::new().await;
        let go = rig.release.clone();
        let first = vec![0xAA_u8; 1000];
        let second = vec![0xBB_u8; 777];
        rig.submit_and_drop(first.clone()).await;

        let mut next = Box::pin(rig.writer.write_owned_bytes(second.clone()));
        assert!(
            tokio::time::timeout(NEGATIVE_WAIT, &mut next)
                .await
                .is_err(),
            "the next write finished while the first was still blocked"
        );
        assert_no_second_job(&mut rig.entered, "write_owned_bytes");

        allow(&go); // the first write
        allow(&go); // the second write
        tokio::time::timeout(DEADLINE, next)
            .await
            .expect("second write finished")
            .expect("second write succeeded");

        let mut expected = first;
        expected.extend_from_slice(&second);
        assert_eq!(rig.writer.bytes_written(), expected.len() as u64);
        assert_eq!(rig.writer.checksum(), Some(crc_label(&expected)));
        let path = rig.file();
        tokio::time::timeout(DEADLINE, Box::new(rig.writer).finalize())
            .await
            .expect("finalize finished")
            .expect("finalize succeeded");
        assert_eq!(std::fs::read(path).expect("read file"), expected);
    }

    /// The borrowed-slice write path honors the same ordering.
    #[tokio::test]
    async fn borrowed_write_waits_for_a_dropped_owned_write() {
        let mut rig = Rig::new().await;
        let go = rig.release.clone();
        let first = vec![0x11_u8; 4096];
        let second = vec![0x22_u8; 123];
        rig.submit_and_drop(first.clone()).await;

        {
            let mut next = Box::pin(rig.writer.write_chunk(&second));
            assert!(
                tokio::time::timeout(NEGATIVE_WAIT, &mut next)
                    .await
                    .is_err(),
                "write_chunk finished while the owned write was still blocked"
            );
            allow(&go); // the owned write
            tokio::time::timeout(DEADLINE, next)
                .await
                .expect("write_chunk finished")
                .expect("write_chunk succeeded");
        }

        let mut expected = first;
        expected.extend_from_slice(&second);
        assert_eq!(rig.writer.checksum(), Some(crc_label(&expected)));
        let path = rig.file();
        tokio::time::timeout(DEADLINE, Box::new(rig.writer).finalize())
            .await
            .expect("finalize finished")
            .expect("finalize succeeded");
        assert_eq!(std::fs::read(path).expect("read file"), expected);
    }

    /// `finalize` must not sync and return while an earlier write is still running; a write
    /// whose caller was dropped while it waited never started and so never lands.
    #[tokio::test]
    async fn finalize_waits_for_a_dropped_owned_write_and_a_dropped_drain_keeps_tracking() {
        let mut rig = Rig::new().await;
        let go = rig.release.clone();
        let first = vec![0xC3_u8; 2048];
        rig.submit_and_drop(first.clone()).await;

        // A second write is dropped while it waits for the first: its drain is cancelled.
        let outcome =
            tokio::time::timeout(NEGATIVE_WAIT, rig.writer.write_owned_bytes(vec![0xEE; 64])).await;
        assert!(outcome.is_err(), "second write must still be waiting");
        assert_no_second_job(&mut rig.entered, "write_owned_bytes");

        let path = rig.file();
        let mut finalize = Box::pin(Box::new(rig.writer).finalize());
        assert!(
            tokio::time::timeout(NEGATIVE_WAIT, &mut finalize)
                .await
                .is_err(),
            "finalize returned while the owned write was still blocked"
        );
        allow(&go);
        tokio::time::timeout(DEADLINE, finalize)
            .await
            .expect("finalize finished")
            .expect("finalize succeeded");
        assert_eq!(std::fs::read(path).expect("read file"), first);
    }

    /// An error of a write whose caller was dropped is reported by the next call, whichever it is.
    #[tokio::test]
    async fn error_of_a_dropped_owned_write_reaches_the_next_owned_write() {
        let mut rig = Rig::new().await;
        rig.submit_and_drop(vec![1_u8; 100]).await;
        rig.fail_next();
        let err = tokio::time::timeout(DEADLINE, rig.writer.write_owned_bytes(vec![2_u8; 10]))
            .await
            .expect("write returned")
            .expect_err("the earlier failure must be reported");
        assert!(
            format!("{err:#}").contains("injected write failure"),
            "{err:#}"
        );
    }

    #[tokio::test]
    async fn error_of_a_dropped_owned_write_reaches_the_next_borrowed_write() {
        let mut rig = Rig::new().await;
        rig.submit_and_drop(vec![1_u8; 100]).await;
        rig.fail_next();
        let err = tokio::time::timeout(DEADLINE, rig.writer.write_chunk(&[2_u8; 10]))
            .await
            .expect("write returned")
            .expect_err("the earlier failure must be reported");
        assert!(
            format!("{err:#}").contains("injected write failure"),
            "{err:#}"
        );
    }

    #[tokio::test]
    async fn error_of_a_dropped_owned_write_reaches_finalize() {
        let mut rig = Rig::new().await;
        rig.submit_and_drop(vec![1_u8; 100]).await;
        rig.fail_next();
        let err = tokio::time::timeout(DEADLINE, Box::new(rig.writer).finalize())
            .await
            .expect("finalize returned")
            .expect_err("finalize must not report success after a failed write");
        assert!(
            format!("{err:#}").contains("injected write failure"),
            "{err:#}"
        );
    }

    /// `cancel` deletes the partial file only after the running write is done, so the write
    /// cannot re-create or outlive the cleanup.
    #[tokio::test]
    async fn cancel_waits_for_a_dropped_owned_write_then_removes_the_file() {
        let mut rig = Rig::new().await;
        let go = rig.release.clone();
        rig.submit_and_drop(vec![9_u8; 512]).await;
        let path = rig.file();
        let mut cancel = Box::pin(Box::new(rig.writer).cancel());
        assert!(
            tokio::time::timeout(NEGATIVE_WAIT, &mut cancel)
                .await
                .is_err(),
            "cancel returned while the owned write was still blocked"
        );
        assert!(
            path.exists(),
            "the file is removed only after the write is done"
        );
        allow(&go);
        tokio::time::timeout(DEADLINE, cancel)
            .await
            .expect("cancel finished")
            .expect("cancel succeeded");
        assert!(!path.exists(), "cancel removes the partial file");
    }
}
