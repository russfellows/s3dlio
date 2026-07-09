// src/range_engine_generic.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// ── Range engine history and rationale ──────────────────────────────────────
//
// There were briefly TWO range-engine files in this codebase:
//
//   range_engine.rs          (S3-specific, removed in v0.9.90)
//   range_engine_generic.rs  (this file — the one that actually works)
//
// range_engine.rs was an early, never-completed experiment.  It depended on
// sharded_client.rs (also removed in v0.9.90), which proposed splitting HTTP
// traffic across multiple AWS SDK client instances to reduce per-client
// contention.  That premise is incorrect for reqwest 0.13 + HTTP/2: the
// connection pool is already concurrent by design, so multiplying client
// instances adds memory/connection overhead without benefit.  The key blocker
// was that range_engine.rs::get_range() returned Bytes::new() — a stub that
// was never implemented — so no data was ever actually returned.
//
// This file (range_engine_generic.rs) is the production engine.  It is:
//   • Backend-agnostic — accepts any async `fn(offset, length) -> Result<Bytes>`
//   • Used by file_store.rs, file_store_direct.rs, object_store.rs (Azure/GCS)
//   • Stream-based with controlled concurrency (Semaphore + buffered())
//   • Disabled by default for local file backends (seek overhead outweighs gain)
//
// FUTURE WORK — S3 range engine
// ────────────────────────────────
// S3 range GETs — how they work vs Azure/GCS
// ─────────────────────────────────────────────
// S3ObjectStore does NOT use this engine, but S3 range GETs (including
// concurrent splitting) ARE fully implemented and working via a separate path:
//
//   • S3ObjectStore::get_range()         → single-range HTTP GET with Range: header
//   • S3ObjectStore::get_optimized()     → calls get_object_concurrent_range_async()
//   • S3ObjectStore::get_range_optimized() → calls get_object_concurrent_range_async()
//   • s3_utils::get_object_concurrent_range_async() — the actual implementation:
//       HEAD to get object size, then splits into chunks and fires concurrent
//       GetObject requests with Range: bytes=N-M headers via FuturesUnordered.
//       Chunks are sorted and assembled lock-free.  Controlled by env var
//       S3DLIO_ENABLE_RANGE_OPTIMIZATION (on by default since v0.9.60).
//
// Azure and GCS use this generic engine instead (via get_with_range_engine()).
//
// FUTURE CONSOLIDATION (non-urgent)
// The two approaches solve the same problem differently.  Migrating S3 to use
// this engine would mean passing self.get_range() as the closure, exactly like
// Azure/GCS.  That would unify the code paths and give S3 the engine's
// cancellation-token and adaptive-chunk features.  An `enable_range_engine`
// flag in an S3Config struct would mirror the Azure/GCS pattern.  This is
// not urgent — both implementations are correct and performant.

use anyhow::{bail, Result};
use bytes::{Bytes, BytesMut};
use futures::stream::{FuturesOrdered, StreamExt};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::constants::{
    DEFAULT_FILE_RANGE_ENGINE_THRESHOLD, DEFAULT_RANGE_ENGINE_CHUNK_SIZE,
    DEFAULT_RANGE_ENGINE_MAX_CONCURRENT, DEFAULT_RANGE_TIMEOUT_SECS,
};
use crate::data_loader::parallel_fetch::DropCancel;

/// Compute the inclusive end-of-range byte offset for a `bytes=start-end`
/// style range request, given a start `offset` and a nonzero byte `length`.
///
/// audit #152 findings 2.5/2.6/2.7 (bug B4): the S3, Azure, and
/// community-GCS `get_range()` implementations each computed this as a bare
/// `offset + length - 1`, which underflows — wrapping to a huge value in
/// release builds, panicking in debug builds — whenever `length == 0`.
///
/// Callers MUST short-circuit `length == 0` themselves before calling this
/// helper (return an empty buffer without issuing a range request at all —
/// RFC 7233 range units can't express an empty range, so there is no valid
/// inclusive end to compute). As defense in depth, this function still
/// returns `Err` rather than underflowing if it's ever called with
/// `length == 0` anyway. It also guards `offset + length` overflowing
/// `u64` via `checked_add`, returning an error instead of silently
/// wrapping.
pub fn range_end_inclusive(offset: u64, length: u64) -> Result<u64> {
    let len_minus_one = length.checked_sub(1).ok_or_else(|| {
        anyhow::anyhow!(
            "range_end_inclusive called with length=0 for offset={offset} — \
             callers must short-circuit zero-length ranges before requesting \
             a byte range (a zero-byte range has no valid inclusive end)"
        )
    })?;
    offset.checked_add(len_minus_one).ok_or_else(|| {
        anyhow::anyhow!("range end overflow: offset={offset} + length={length} exceeds u64::MAX")
    })
}

/// Configuration for range-based concurrent downloads
///
/// **Performance Considerations**:
/// - **Local File Systems**: Range parallelism may be **slower** due to:
///   - Seek overhead (random access vs sequential)
///   - Disk I/O contention
///   - Page cache already optimizes sequential reads
///   - Consider disabling or using higher thresholds (16-64MB) for file:// URIs
///
/// - **Network Storage (S3/Azure/GCS)**: Benefits significantly from range parallelism:
///   - Hides network latency with concurrent requests
///   - 30-50% throughput improvement for large files
///   - Lower thresholds (4MB) work well
///
/// - **DirectIO**: Limited benefit since O_DIRECT already bypasses page cache
///   - Higher threshold (16MB) recommended due to alignment overhead
///   - Lower concurrency (16) to avoid excessive parallel seeks
#[derive(Debug, Clone)]
pub struct RangeEngineConfig {
    /// Size of each range chunk in bytes (default: 64MB)
    pub chunk_size: usize,

    /// Maximum concurrent range requests (default: 32)
    pub max_concurrent_ranges: usize,

    /// Minimum object size to trigger range splitting (default: 4MB)
    /// Objects smaller than this use simple single-request downloads
    /// **WARNING**: For local filesystems, consider higher thresholds or disable entirely
    pub min_split_size: u64,

    /// Timeout per range request (default: 30s)
    pub range_timeout: Duration,
}

impl Default for RangeEngineConfig {
    fn default() -> Self {
        Self {
            chunk_size: DEFAULT_RANGE_ENGINE_CHUNK_SIZE,
            max_concurrent_ranges: DEFAULT_RANGE_ENGINE_MAX_CONCURRENT,
            min_split_size: DEFAULT_FILE_RANGE_ENGINE_THRESHOLD,
            range_timeout: Duration::from_secs(DEFAULT_RANGE_TIMEOUT_SECS),
        }
    }
}

/// Statistics collected during range download
#[derive(Debug, Clone)]
pub struct RangeDownloadStats {
    /// Total bytes downloaded
    pub bytes_downloaded: u64,

    /// Number of range requests made
    pub ranges_processed: usize,

    /// Total elapsed time
    pub elapsed_time: Duration,

    /// Average throughput in bytes per second
    pub throughput_bps: u64,
}

impl RangeDownloadStats {
    /// Throughput in megabytes per second
    pub fn throughput_mbps(&self) -> f64 {
        (self.throughput_bps as f64) / (1024.0 * 1024.0)
    }

    /// Throughput in gigabits per second
    pub fn throughput_gbps(&self) -> f64 {
        (self.throughput_bps as f64 * 8.0) / (1_000_000_000.0)
    }
}

/// Universal range-based download engine
///
/// This engine provides high-performance concurrent downloads for ANY backend
/// that implements async `get_range(offset, length)`. It uses:
///
/// - Stream-based architecture with `stream::iter().buffered()`
/// - Controlled concurrency via semaphore
/// - Cancellation token support for clean shutdown
/// - Timeout per range request
/// - Ordered reassembly of chunks
///
/// # Example
///
/// ```ignore
/// use s3dlio::range_engine_generic::{RangeEngine, RangeEngineConfig};
///
/// # async fn example() -> anyhow::Result<()> {
/// let engine = RangeEngine::new(RangeEngineConfig::default());
///
/// // Works with ANY async get_range function
/// let get_range = |offset, length| async move {
///     // Your backend's get_range implementation
///     my_backend.get_range(offset, length).await
/// };
///
/// let (bytes, stats) = engine.download(file_size, get_range, None).await?;
/// println!("Downloaded {} bytes in {} ranges at {} MB/s",
///     stats.bytes_downloaded, stats.ranges_processed, stats.throughput_mbps());
/// # Ok(())
/// # }
/// ```
pub struct RangeEngine {
    config: RangeEngineConfig,
    concurrency_limiter: Arc<Semaphore>,
}

impl RangeEngine {
    /// Create a new range engine with the given configuration
    pub fn new(config: RangeEngineConfig) -> Self {
        let concurrency_limiter = Arc::new(Semaphore::new(config.max_concurrent_ranges));
        Self {
            config,
            concurrency_limiter,
        }
    }

    /// Create with default configuration
    pub fn with_defaults() -> Self {
        Self::new(RangeEngineConfig::default())
    }

    /// Download object using concurrent range requests
    ///
    /// This method automatically decides the optimal download strategy:
    /// - Small objects (< min_split_size): Single request
    /// - Large objects: Concurrent range requests with streaming
    ///
    /// # Arguments
    ///
    /// * `object_size` - Total size of the object in bytes
    /// * `get_range` - Async function that fetches a range: `fn(offset, length) -> Future<Result<Bytes>>`
    /// * `cancel` - Optional cancellation token for clean shutdown
    ///
    /// # Returns
    ///
    /// Tuple of (downloaded bytes, statistics)
    pub async fn download<F, Fut>(
        &self,
        object_size: u64,
        get_range: F,
        cancel: Option<CancellationToken>,
    ) -> Result<(Bytes, RangeDownloadStats)>
    where
        F: Fn(u64, u64) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<Bytes>> + Send,
    {
        // audit #152 bug 2.8 (D9): a zero-byte object is a legitimate,
        // existing object (an empty file) -- not an error. Previously
        // this bailed with "Cannot download zero-sized object", turning
        // a perfectly normal empty read into a hard failure for any
        // caller whose object size happened to be 0.
        if object_size == 0 {
            return Ok((
                Bytes::new(),
                RangeDownloadStats {
                    bytes_downloaded: 0,
                    ranges_processed: 0,
                    elapsed_time: Duration::ZERO,
                    throughput_bps: 0,
                },
            ));
        }

        let start_time = Instant::now();

        // Small objects: use single request (no overhead from range splitting)
        if object_size < self.config.min_split_size {
            tracing::debug!(
                "Object size {} < threshold {}, using single request",
                object_size,
                self.config.min_split_size
            );
            return self
                .download_single(object_size, get_range, start_time)
                .await;
        }

        // Large objects: use concurrent range requests with streams
        tracing::debug!(
            "Object size {} >= threshold {}, using concurrent ranges",
            object_size,
            self.config.min_split_size
        );
        self.download_with_ranges(object_size, get_range, cancel, start_time)
            .await
    }

    /// Download using a single range request (for small objects)
    async fn download_single<F, Fut>(
        &self,
        object_size: u64,
        get_range: F,
        start_time: Instant,
    ) -> Result<(Bytes, RangeDownloadStats)>
    where
        F: Fn(u64, u64) -> Fut + Send + Sync,
        Fut: std::future::Future<Output = Result<Bytes>> + Send,
    {
        // Fetch entire object as single range
        let bytes = tokio::time::timeout(self.config.range_timeout, get_range(0, object_size))
            .await
            .map_err(|_| {
                anyhow::anyhow!(
                    "Single request timeout after {:?}",
                    self.config.range_timeout
                )
            })?
            .map_err(|e| anyhow::anyhow!("Single request failed: {}", e))?;

        let bytes_downloaded = bytes.len() as u64;
        let elapsed = start_time.elapsed();

        let stats = RangeDownloadStats {
            bytes_downloaded,
            ranges_processed: 1,
            elapsed_time: elapsed,
            throughput_bps: Self::calculate_throughput(bytes_downloaded, elapsed),
        };

        Ok((bytes, stats))
    }

    /// Download using concurrent range requests (stream-based for large objects)
    async fn download_with_ranges<F, Fut>(
        &self,
        object_size: u64,
        get_range: F,
        cancel: Option<CancellationToken>,
        start_time: Instant,
    ) -> Result<(Bytes, RangeDownloadStats)>
    where
        F: Fn(u64, u64) -> Fut + Send + Sync + Clone + 'static,
        Fut: std::future::Future<Output = Result<Bytes>> + Send,
    {
        // Calculate optimal range splits
        let ranges = self.calculate_ranges(object_size);
        let n_ranges = ranges.len();

        tracing::debug!(
            "Splitting {} bytes into {} ranges of ~{} bytes each",
            object_size,
            n_ranges,
            self.config.chunk_size
        );

        let semaphore = Arc::clone(&self.concurrency_limiter);
        let timeout = self.config.range_timeout;

        // Task-level parallelism (issue #148 site 3.1f): each range fetch
        // is `tokio::spawn`'d as its own task so tokio can distribute the
        // per-range CPU work (signing, header parsing, body assembly)
        // across worker threads. The prior `.buffered(N)` pattern polled
        // every future inside this task's poll cycle.
        //
        // Ordering + peak-memory preservation: results are consumed via
        // `FuturesOrdered<JoinHandle<...>>` in submission order — that
        // preserves the running-write-offset assembly (short read from
        // range k does NOT leave a zero-filled hole at range k+1's
        // offset) and matches the previous `.buffered()` semantics.
        //
        // Bounded spawn pool: we prime the pool with `max_concurrent`
        // spawns, then produce a new spawn each time we consume a result.
        // Without this cap all N spawns would be alive simultaneously,
        // each holding a chunk-sized Bytes in its JoinHandle after
        // completion — driving peak memory to ~2× total_size (observed
        // failure of `range_engine_download_peak_memory_bounded`).
        //
        // DropCancel guards mid-flight drop; external cancel is honored
        // via a second select! arm.
        let internal_cancel = CancellationToken::new();
        let _drop_cancel = DropCancel(internal_cancel.clone());

        let spawn_fetch = |idx: usize, offset: u64, length: u64| {
            let get_range = get_range.clone();
            let semaphore = Arc::clone(&semaphore);
            let external_cancel = cancel.clone();
            let internal_token = internal_cancel.clone();
            tokio::spawn(async move {
                let _permit = semaphore
                    .acquire_owned()
                    .await
                    .map_err(|e| anyhow::anyhow!("Semaphore acquisition failed: {}", e))?;

                let fetch = async {
                    let bytes = tokio::time::timeout(timeout, get_range(offset, length))
                        .await
                        .map_err(|_| {
                            anyhow::anyhow!(
                                "Range {} timeout after {:?} (offset={}, length={})",
                                idx,
                                timeout,
                                offset,
                                length
                            )
                        })?
                        .map_err(|e| {
                            anyhow::anyhow!(
                                "Range {} request failed (offset={}, length={}): {}",
                                idx,
                                offset,
                                length,
                                e
                            )
                        })?;
                    // Locked contract (issue #152 / audit finding f39,
                    // docs/implementation-plans/v0.9.109-audit-fix-plan.md
                    // §3 bug A1): any short OR over read is a hard error.
                    // Assembly below writes each chunk at a running
                    // cursor rather than its declared range offset, so a
                    // length mismatch on any non-final chunk — even one
                    // that's later "compensated" by a mismatch elsewhere
                    // and sums to the right total — silently shifts every
                    // subsequent chunk to the wrong position. There is no
                    // safe way to tolerate a mismatch here; it must abort
                    // the whole download.
                    if bytes.len() != length as usize {
                        return Err(anyhow::anyhow!(
                            "Range {} short/over read: expected {} bytes at offset {}, got {} \
                             bytes (last_range={})",
                            idx,
                            length,
                            offset,
                            bytes.len(),
                            idx == n_ranges - 1
                        ));
                    }
                    Ok::<(usize, Bytes), anyhow::Error>((idx, bytes))
                };

                let external_wait = async {
                    match external_cancel {
                        Some(t) => t.cancelled().await,
                        None => std::future::pending::<()>().await,
                    }
                };

                tokio::select! {
                    _ = internal_token.cancelled() => {
                        Err(anyhow::anyhow!("Range {} cancelled (drop)", idx))
                    }
                    _ = external_wait => {
                        Err(anyhow::anyhow!("Range {} cancelled by user", idx))
                    }
                    r = fetch => r,
                }
            })
        };

        let pool_cap = self.config.max_concurrent_ranges;
        let mut pending: FuturesOrdered<JoinHandle<Result<(usize, Bytes)>>> = FuturesOrdered::new();
        let mut next_range = 0usize;
        while next_range < n_ranges && pending.len() < pool_cap {
            let (offset, length) = ranges[next_range];
            pending.push_back(spawn_fetch(next_range, offset, length));
            next_range += 1;
        }

        // Pre-allocate the master output buffer (issue #148, audit
        // §3.3a / Patch 3). Peak live memory is bounded by
        // `pool_cap * chunk_size` (in-flight/queued) + `master`.
        let mut master = BytesMut::zeroed(object_size as usize);
        let mut write_offset: usize = 0;
        let mut ranges_seen: usize = 0;
        let mut first_err: Option<anyhow::Error> = None;

        while let Some(join_res) = pending.next().await {
            match join_res {
                Ok(Ok((idx, bytes))) => {
                    if first_err.is_none() {
                        let len = bytes.len();
                        tracing::trace!("Assembling range {} ({} bytes)", idx, len);

                        let end = write_offset
                            .checked_add(len)
                            .ok_or_else(|| anyhow::anyhow!("range assembly offset overflow"))?;
                        if end > master.len() {
                            bail!(
                                "range {} would write {}..{} but master buffer is only {} bytes",
                                idx,
                                write_offset,
                                end,
                                master.len()
                            );
                        }
                        master[write_offset..end].copy_from_slice(&bytes);
                        write_offset = end;
                        ranges_seen += 1;
                    }
                    drop(bytes);
                }
                Ok(Err(e)) => {
                    if first_err.is_none() {
                        first_err = Some(e);
                        internal_cancel.cancel();
                    }
                }
                Err(join_err) if join_err.is_panic() => {
                    if first_err.is_none() {
                        first_err =
                            Some(anyhow::anyhow!("range fetch task panicked: {}", join_err));
                        internal_cancel.cancel();
                    }
                }
                Err(_cancelled) => {
                    // Task was cancelled via the select! arm — expected
                    // during shutdown after an earlier error fired cancel.
                }
            }

            // Refill the spawn pool as slots free up.
            if first_err.is_none() && next_range < n_ranges {
                let (offset, length) = ranges[next_range];
                pending.push_back(spawn_fetch(next_range, offset, length));
                next_range += 1;
            }
        }

        if let Some(e) = first_err {
            return Err(e);
        }
        let _ = ranges_seen;

        // Defense in depth: every chunk that reached this point matched
        // its declared range length exactly (enforced above), and
        // `calculate_ranges` partitions `object_size` exactly, so
        // `write_offset` must equal `master.len()` here. This should be
        // unreachable — if it ever fires, something upstream violated
        // the per-chunk length invariant without going through the
        // error path above, which is itself a bug worth surfacing
        // loudly rather than silently truncating (the old behavior,
        // which is the root cause this fix closes out).
        if write_offset != master.len() {
            bail!(
                "range assembly invariant violated: wrote {} bytes but master buffer is {} \
                 bytes (object_size drifted from the sum of range lengths)",
                write_offset,
                master.len()
            );
        }

        let bytes_downloaded = master.len() as u64;
        let elapsed = start_time.elapsed();

        let stats = RangeDownloadStats {
            bytes_downloaded,
            ranges_processed: n_ranges,
            elapsed_time: elapsed,
            throughput_bps: Self::calculate_throughput(bytes_downloaded, elapsed),
        };

        tracing::info!(
            "Downloaded {} bytes in {} ranges: {:.2} MB/s ({:.2} Gbps)",
            stats.bytes_downloaded,
            stats.ranges_processed,
            stats.throughput_mbps(),
            stats.throughput_gbps()
        );

        Ok((master.freeze(), stats))
    }

    /// Calculate optimal range splits for an object
    ///
    /// Splits the object into chunks of approximately chunk_size,
    /// with the last chunk possibly being smaller.
    fn calculate_ranges(&self, object_size: u64) -> Vec<(u64, u64)> {
        let mut ranges = Vec::new();
        let mut offset = 0u64;
        let chunk_size = self.config.chunk_size as u64;

        while offset < object_size {
            let remaining = object_size - offset;
            let length = remaining.min(chunk_size);
            ranges.push((offset, length));
            offset += length;
        }

        ranges
    }

    /// Calculate throughput in bytes per second
    fn calculate_throughput(bytes: u64, elapsed: Duration) -> u64 {
        let elapsed_secs = elapsed.as_secs_f64();
        if elapsed_secs > 0.0 {
            (bytes as f64 / elapsed_secs) as u64
        } else {
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // RED-then-GREEN regression tests for s3dlio issue #152 findings
    // 2.5/2.6/2.7 (bug B4): `offset + length - 1` underflows when
    // `length == 0`. These exercise the shared `range_end_inclusive`
    // helper directly; separate tests exercise the S3/Azure/GCS
    // `get_range()` call sites that consume it.

    #[test]
    fn range_end_inclusive_zero_length_is_an_error_not_an_underflow() {
        // Pre-fix, the call sites computed `offset + 0 - 1` directly,
        // which panics (debug builds, overflow-checks-on) or wraps to
        // u64::MAX - 1 + offset (release builds) instead of ever reaching
        // this helper. The helper itself must never underflow either.
        let result = range_end_inclusive(100, 0);
        assert!(
            result.is_err(),
            "length=0 must be rejected, not silently underflow"
        );
    }

    #[test]
    fn range_end_inclusive_single_byte_at_offset_zero() {
        assert_eq!(range_end_inclusive(0, 1).unwrap(), 0);
    }

    #[test]
    fn range_end_inclusive_normal_range() {
        // offset=100, length=50 -> bytes 100..=149
        assert_eq!(range_end_inclusive(100, 50).unwrap(), 149);
    }

    #[test]
    fn range_end_inclusive_overflow_is_an_error() {
        let result = range_end_inclusive(u64::MAX - 1, 10);
        assert!(
            result.is_err(),
            "offset+length overflowing u64 must be an error, not a silent wrap"
        );
    }

    #[tokio::test]
    async fn test_zero_sized_object_returns_empty_bytes_not_error() {
        // RED-then-GREEN regression test for s3dlio issue #152 bug 2.8 (D9).
        // Pre-fix, `download()` unconditionally `bail!`-ed on
        // `object_size == 0`, treating a legitimate empty object as an
        // error. An empty object is not a failure -- it should download
        // as zero bytes, same as reading an empty file.
        let engine = RangeEngine::new(RangeEngineConfig {
            min_split_size: 1, // force any nonzero size through the split path
            ..RangeEngineConfig::default()
        });

        let get_range = move |_offset: u64, _length: u64| async move { Ok(Bytes::new()) };

        let (bytes, stats) = engine
            .download(0, get_range, None)
            .await
            .expect("downloading a zero-sized object must succeed, not error");

        assert_eq!(bytes.len(), 0);
        assert_eq!(stats.bytes_downloaded, 0);
        assert_eq!(stats.ranges_processed, 0);
    }

    #[tokio::test]
    async fn test_small_object_single_request() {
        let engine = RangeEngine::with_defaults();

        // 1MB object (< 4MB threshold)
        let object_size = 1024 * 1024;
        let data = vec![0u8; object_size as usize];

        let get_range = move |offset: u64, length: u64| {
            let data = data.clone();
            async move {
                Ok(Bytes::from(
                    data[offset as usize..(offset + length) as usize].to_vec(),
                ))
            }
        };

        let (bytes, stats) = engine.download(object_size, get_range, None).await.unwrap();

        assert_eq!(bytes.len(), object_size as usize);
        assert_eq!(stats.ranges_processed, 1);
        assert_eq!(stats.bytes_downloaded, object_size);
    }

    #[tokio::test]
    async fn test_large_object_concurrent_ranges() {
        let engine = RangeEngine::new(RangeEngineConfig {
            chunk_size: 1024 * 1024, // 1MB chunks
            max_concurrent_ranges: 4,
            min_split_size: 2 * 1024 * 1024, // 2MB threshold
            range_timeout: Duration::from_secs(5),
        });

        // 10MB object (> 2MB threshold)
        let object_size = 10 * 1024 * 1024;
        let data = (0..object_size)
            .map(|i| (i % 256) as u8)
            .collect::<Vec<_>>();

        let concurrent_count = Arc::new(AtomicUsize::new(0));
        let max_concurrent = Arc::new(AtomicUsize::new(0));

        let get_range = {
            let data = data.clone();
            let concurrent_count = Arc::clone(&concurrent_count);
            let max_concurrent = Arc::clone(&max_concurrent);

            move |offset: u64, length: u64| {
                let data = data.clone();
                let concurrent_count = Arc::clone(&concurrent_count);
                let max_concurrent = Arc::clone(&max_concurrent);

                async move {
                    // Track concurrency
                    let current = concurrent_count.fetch_add(1, Ordering::SeqCst) + 1;
                    max_concurrent.fetch_max(current, Ordering::SeqCst);

                    // Simulate network delay
                    tokio::time::sleep(Duration::from_millis(10)).await;

                    let result =
                        Bytes::from(data[offset as usize..(offset + length) as usize].to_vec());

                    concurrent_count.fetch_sub(1, Ordering::SeqCst);
                    Ok(result)
                }
            }
        };

        let (bytes, stats) = engine.download(object_size, get_range, None).await.unwrap();

        // Verify correctness
        assert_eq!(bytes.len(), object_size as usize);
        assert_eq!(stats.bytes_downloaded, object_size);
        assert_eq!(stats.ranges_processed, 10); // 10 x 1MB chunks

        // Verify concurrency occurred
        let max_concurrent_seen = max_concurrent.load(Ordering::SeqCst);
        assert!(
            max_concurrent_seen > 1,
            "Expected concurrent execution, got max_concurrent={}",
            max_concurrent_seen
        );
        assert!(
            max_concurrent_seen <= 4,
            "Should not exceed max_concurrent_ranges"
        );

        // Verify data integrity
        for (i, &byte) in bytes.iter().enumerate() {
            assert_eq!(byte, (i % 256) as u8, "Data mismatch at byte {}", i);
        }
    }

    #[tokio::test]
    async fn test_cancellation() {
        let engine = RangeEngine::new(RangeEngineConfig {
            chunk_size: 1024 * 1024,
            max_concurrent_ranges: 4,
            min_split_size: 2 * 1024 * 1024,
            range_timeout: Duration::from_secs(5),
        });

        let object_size = 10 * 1024 * 1024;
        let cancel_token = CancellationToken::new();

        let get_range = {
            let cancel_token = cancel_token.clone();
            move |_offset: u64, _length: u64| {
                let cancel_token = cancel_token.clone();
                async move {
                    // Cancel after first request
                    cancel_token.cancel();
                    tokio::time::sleep(Duration::from_millis(100)).await;
                    Ok(Bytes::from(vec![0u8; 1024 * 1024]))
                }
            }
        };

        let result = engine
            .download(object_size, get_range, Some(cancel_token.clone()))
            .await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cancelled"));
    }

    #[tokio::test]
    async fn test_timeout() {
        let engine = RangeEngine::new(RangeEngineConfig {
            chunk_size: 1024 * 1024,
            max_concurrent_ranges: 2,
            min_split_size: 2 * 1024 * 1024,
            range_timeout: Duration::from_millis(100), // Very short timeout
        });

        let object_size = 5 * 1024 * 1024;

        let get_range = |_offset: u64, _length: u64| async move {
            // Simulate slow request (exceeds timeout)
            tokio::time::sleep(Duration::from_secs(1)).await;
            Ok(Bytes::from(vec![0u8; 1024 * 1024]))
        };

        let result = engine.download(object_size, get_range, None).await;

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("timeout"));
    }
}
