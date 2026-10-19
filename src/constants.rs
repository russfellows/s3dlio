// src/constants.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use once_cell::sync::Lazy;
use rand::RngCore;
use std::sync::Arc;

/// Default page size fallback when system detection fails (4096 bytes)
pub const DEFAULT_PAGE_SIZE: usize = 4096;

/// Default buffer alignment for I/O operations (same as page size)
pub const DEFAULT_BUFFER_ALIGNMENT: usize = DEFAULT_PAGE_SIZE;

/// Default minimum I/O size for direct I/O operations (64KB for better performance)
pub const DEFAULT_MIN_IO_SIZE: usize = 64 * 1024;

/// Bytes in one MiB, for the `*_MB` environment variables.
pub const BYTES_PER_MIB: usize = 1024 * 1024;

/// Default size, in bytes, of one `direct://` (O_DIRECT) bounce buffer (16 MiB).
///
/// O_DIRECT needs the buffer ADDRESS, the file offset and the length of every I/O to be
/// multiples of the device alignment (4096). A PUT source whose address is already aligned is
/// written straight from the caller's memory in one `write` and never uses a bounce buffer.
/// Every other PUT, and (optionally, see [`ENV_DIRECT_IO_GET_MODE`]) every GET, moves its data
/// through the shared per-NUMA-node pool of warm, page-aligned bounce buffers described in
/// `direct_bounce.rs`: buffers are allocated and first-touched once per node and reused, used in
/// ping-pong pairs so the copy of chunk `k+1` overlaps the I/O of chunk `k`.
///
/// Why 16 MiB: the I/O is large enough that per-syscall cost is noise, while the pooled bounce
/// memory stays small and bounded: `nodes x pairs x 2 x size` (default: 4 pairs x 2 x 16 MiB =
/// 128 MiB per NUMA node), shared by every in-flight request in both directions. Several small
/// pairs rather than one big pair let several requests use warm buffers at once, so a long
/// transfer does not block the short ones behind it. (The default shape is provisional: chosen
/// from a discrete-event model of the pool, to be confirmed with native counters on hardware.
/// Set the two environment variables to restore the earlier 1 pair x 64 MiB.)
/// (Pooled path only: private buffers made when the pool cannot serve
/// a request, or by the opt-in `fallback` policy, are outside it; see `direct_bounce.rs`.) Override with [`ENV_DIRECT_IO_BOUNCE_SIZE_MB`], clamped to
/// [`MIN_DIRECT_IO_BOUNCE_SIZE_MB`]..=[`MAX_DIRECT_IO_BOUNCE_SIZE_MB`].
pub const DEFAULT_DIRECT_IO_BOUNCE_SIZE: usize = 16 * BYTES_PER_MIB;

/// Smallest accepted `direct://` bounce-buffer size in MiB.
pub const MIN_DIRECT_IO_BOUNCE_SIZE_MB: usize = 4;

/// Largest accepted `direct://` bounce-buffer size in MiB.
pub const MAX_DIRECT_IO_BOUNCE_SIZE_MB: usize = 64;

/// Environment variable overriding [`DEFAULT_DIRECT_IO_BOUNCE_SIZE`], in MiB.
///
/// Clamped to 4..=64; `0` or an unparseable value selects the default (16).
/// Example: `S3DLIO_DIRECT_IO_BOUNCE_SIZE_MB=64` (the pre-0.9.128-round-7 size).
pub const ENV_DIRECT_IO_BOUNCE_SIZE_MB: &str = "S3DLIO_DIRECT_IO_BOUNCE_SIZE_MB";

/// Default number of ping-pong buffer PAIRS per NUMA node (1 pair = 2 buffers).
pub const DEFAULT_DIRECT_IO_BOUNCE_PAIRS: usize = 4;

/// Largest accepted pairs-per-node value.
pub const MAX_DIRECT_IO_BOUNCE_PAIRS: usize = 64;

/// Environment variable setting the number of buffer pairs per NUMA node (1..=64). More pairs
/// let more requests use warm buffers at once, at `2 x size` more memory per pair per node.
pub const ENV_DIRECT_IO_BOUNCE_PAIRS: &str = "S3DLIO_DIRECT_IO_BOUNCE_PAIRS_PER_NODE";

/// Environment variable choosing what a request does when every pair of its node is in use:
/// `wait` (default; strict FIFO across PUTs and GETs, so neither direction starves the other)
/// or `fallback` (use a private buffer of one bounce size for this request: no pipelining, extra
/// memory of one buffer per concurrent request beyond the pool, counted in `bounce_stats()`).
pub const ENV_DIRECT_IO_BOUNCE_WHEN_BUSY: &str = "S3DLIO_DIRECT_IO_BOUNCE_WHEN_BUSY";

/// Environment variable choosing how a `direct://` GET reads: `direct` (default for now) reads
/// straight into one fresh page-aligned buffer of the object's size, zero copies but fresh page
/// faults on every call; `pooled` reads through the shared bounce pool (warm buffers, one copy
/// into the destination). Which is faster depends on load; this stays `direct` until measured.
pub const ENV_DIRECT_IO_GET_MODE: &str = "S3DLIO_DIRECT_IO_GET_MODE";

/// A failed creation of a NUMA node's bounce pool is retried after this long (30 s); until then
/// requests use per-call buffers.
pub const BOUNCE_POOL_RETRY: std::time::Duration = std::time::Duration::from_secs(30);

/// How often a request waiting for a bounce buffer re-checks whether it was cancelled (25 ms).
pub const BOUNCE_CANCEL_POLL: std::time::Duration = std::time::Duration::from_millis(25);

/// Size, in bytes, of the aligned staging buffer a `direct://` STREAMING writer
/// (`DirectIOWriter`) fills before each O_DIRECT write (64 MiB). A stream's final size is
/// unknown and many writers may be open at once; a page-aligned chunk bypasses it entirely.
pub const DEFAULT_DIRECT_IO_WRITER_STAGING_SIZE: usize = 64 * BYTES_PER_MIB;

/// Resolve the effective bounce-buffer size in bytes (clamped, see
/// [`ENV_DIRECT_IO_BOUNCE_SIZE_MB`]).
pub fn direct_io_bounce_size() -> usize {
    // The `.env` policy first (#243): a public setting reader.
    crate::dotenv::apply();
    parse_direct_io_bounce_size(std::env::var(ENV_DIRECT_IO_BOUNCE_SIZE_MB).ok().as_deref())
}

/// Pure parser behind [`direct_io_bounce_size`]: MiB text to bytes, clamped to the accepted
/// range, the default on anything unusable (unset, empty, non-numeric, zero).
pub fn parse_direct_io_bounce_size(raw: Option<&str>) -> usize {
    raw.and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&mb| mb > 0)
        .map(|mb| mb.clamp(MIN_DIRECT_IO_BOUNCE_SIZE_MB, MAX_DIRECT_IO_BOUNCE_SIZE_MB))
        .map(|mb| mb * BYTES_PER_MIB)
        .unwrap_or(DEFAULT_DIRECT_IO_BOUNCE_SIZE)
}

/// Pure parser for [`ENV_DIRECT_IO_BOUNCE_PAIRS`]: clamped to 1..=64, default 4.
pub fn parse_direct_io_bounce_pairs(raw: Option<&str>) -> usize {
    raw.and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .map(|n| n.min(MAX_DIRECT_IO_BOUNCE_PAIRS))
        .unwrap_or(DEFAULT_DIRECT_IO_BOUNCE_PAIRS)
}

#[cfg(test)]
mod direct_io_bounce_size_tests {
    use super::*;

    #[test]
    fn unusable_values_select_the_default() {
        for raw in [
            None,
            Some(""),
            Some("abc"),
            Some("0"),
            Some("-4"),
            Some("1.5"),
        ] {
            assert_eq!(
                parse_direct_io_bounce_size(raw),
                DEFAULT_DIRECT_IO_BOUNCE_SIZE,
                "{raw:?}"
            );
        }
    }

    #[test]
    fn sizes_are_clamped_to_4_to_64_mib() {
        assert_eq!(parse_direct_io_bounce_size(Some("1")), 4 * BYTES_PER_MIB);
        assert_eq!(parse_direct_io_bounce_size(Some("16")), 16 * BYTES_PER_MIB);
        assert_eq!(parse_direct_io_bounce_size(Some(" 8 ")), 8 * BYTES_PER_MIB);
        assert_eq!(
            parse_direct_io_bounce_size(Some("1024")),
            64 * BYTES_PER_MIB
        );
        assert_eq!(
            parse_direct_io_bounce_size(Some(&usize::MAX.to_string())),
            64 * BYTES_PER_MIB
        );
    }

    #[test]
    fn default_is_the_documented_16_mib() {
        assert_eq!(DEFAULT_DIRECT_IO_BOUNCE_SIZE, 16 * BYTES_PER_MIB);
        assert_eq!(parse_direct_io_bounce_size(None), 16 * BYTES_PER_MIB);
    }

    #[test]
    fn pairs_are_clamped_to_1_to_64_and_default_to_4() {
        assert_eq!(DEFAULT_DIRECT_IO_BOUNCE_PAIRS, 4);
        assert_eq!(parse_direct_io_bounce_pairs(None), 4);
        assert_eq!(parse_direct_io_bounce_pairs(Some("0")), 4);
        assert_eq!(parse_direct_io_bounce_pairs(Some("x")), 4);
        assert_eq!(parse_direct_io_bounce_pairs(Some("1")), 1);
        assert_eq!(parse_direct_io_bounce_pairs(Some("3")), 3);
        assert_eq!(parse_direct_io_bounce_pairs(Some("9999")), 64);
    }
}

/// Largest number of bytes Linux transfers in ONE `read`/`write` system call
/// (`0x7ffff000`, just under 2 GiB, rounded down to a page): a larger request is
/// silently shortened by the kernel. A `direct://` transfer bigger than this is issued as
/// consecutive writes of at most this size.
pub const MAX_SINGLE_IO_BYTES: usize = 0x7fff_f000;

/// Default multipart upload threshold for S3 (32 MB)
pub const DEFAULT_S3_MULTIPART_THRESHOLD: usize = 32 * 1024 * 1024;

/// Default multipart upload part size for S3 (16 MB)
pub const DEFAULT_S3_MULTIPART_PART_SIZE: usize = 16 * 1024 * 1024;

/// Default multipart upload part size for Azure (16 MB)
pub const DEFAULT_AZURE_MULTIPART_PART_SIZE: usize = 16 * 1024 * 1024;

/// Default Azure SDK managed-download partition size (32 MiB).
pub const DEFAULT_AZURE_DOWNLOAD_PART_SIZE: usize = 32 * 1024 * 1024;

/// Default Azure SDK managed-download concurrency.
pub const DEFAULT_AZURE_DOWNLOAD_CONCURRENCY: usize = 16;

/// Smallest partition size a sequential Azure download plans its requests with (32 MiB).
///
/// A cached object size is only a hint, and may be stale. The SDK splits a download into
/// partitions of the size it is given, so a stale 0- or 1-byte hint turned a re-read of a blob
/// that had grown to 1 MiB into about a million one-byte requests. The partition size is
/// therefore never smaller than this, whatever the hint says.
pub const AZURE_MIN_DOWNLOAD_PARTITION_SIZE: usize = 32 * 1024 * 1024;

/// Largest buffer an Azure range read reserves up front (128 MiB), however long a range the
/// caller asked for. The buffer still grows if the server sends more, so this only bounds what
/// a caller-chosen length (`get_range(.., Some(1 << 40))` to mean "to the end") can reserve.
pub const AZURE_RANGE_PRESIZE_MAX: u64 = 128 * 1024 * 1024;

/// Most Azure service clients (one per account URL) kept for reuse; see
/// `azure_client::AzureBlob::with_default_credential_from_url`.
pub const AZURE_SERVICE_CLIENT_CACHE_MAX: usize = 64;

/// Selects `auto`, `azure-sdk`, `s3dlio`, or `sequential` Azure downloads.
pub const ENV_AZURE_DOWNLOAD_MODE: &str = "S3DLIO_AZURE_DOWNLOAD_MODE";

/// Selects the concurrent engine used by Azure `auto` mode.
pub const ENV_AZURE_CONCURRENT_ENGINE: &str = "S3DLIO_AZURE_CONCURRENT_ENGINE";

/// Azure object-size threshold in MiB for switching `auto` mode to concurrency.
pub const ENV_AZURE_DOWNLOAD_THRESHOLD_MB: &str = "S3DLIO_AZURE_DOWNLOAD_THRESHOLD_MB";

/// Azure SDK managed-download request concurrency.
pub const ENV_AZURE_DOWNLOAD_CONCURRENCY: &str = "S3DLIO_AZURE_DOWNLOAD_CONCURRENCY";

/// Azure SDK managed-download partition size in MiB.
pub const ENV_AZURE_DOWNLOAD_PART_SIZE_MB: &str = "S3DLIO_AZURE_DOWNLOAD_PART_SIZE_MB";

/// Maximum number of parts in a multipart upload
pub const MAX_MULTIPART_PARTS: usize = 10000;

/// Default TCP connect timeout in seconds.
///
/// Covers only the TCP handshake (SYN → SYN-ACK).  On a LAN/datacenter network
/// a connect that hasn't completed in 20 s indicates a dead host, wrong IP, a
/// firewall silently dropping packets, or — under heavy cold-start fan-out —
/// an endpoint whose TCP accept queue is briefly saturated.  Default bumped
/// from 10 s to 20 s in v0.9.102 after mlcommons/storage#506 showed the
/// hardcoded 5 s SDK ceiling was the trigger for warmup dispatch failures
/// on RetinaNet B200 (2 048 concurrent connects to a single endpoint).
/// Override with `S3DLIO_CONNECT_TIMEOUT_SECS`.
pub const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 20;

/// Environment variable name for the TCP connect-timeout override.
///
/// See [`DEFAULT_CONNECT_TIMEOUT_SECS`].  Consumed by every HTTP-client builder
/// in s3dlio via [`connect_timeout_secs()`] so both the reqwest transport and
/// the AWS-SDK `TimeoutConfig` layer agree on a single value — and on the same
/// number reported in the docs.
///
/// Use a longer value under cold-start fan-out (e.g. mlperf-storage retinanet
/// runs that issue thousands of concurrent connects at warmup): the default
/// 10 s may be too aggressive when the endpoint's TCP accept queue is briefly
/// saturated.  Example: `S3DLIO_CONNECT_TIMEOUT_SECS=30`.
pub const ENV_CONNECT_TIMEOUT_SECS: &str = "S3DLIO_CONNECT_TIMEOUT_SECS";

/// Resolve the effective TCP connect timeout in seconds.
///
/// Reads [`ENV_CONNECT_TIMEOUT_SECS`] each call (cheap; called only at HTTP
/// client construction time, which happens once per process); falls back to
/// [`DEFAULT_CONNECT_TIMEOUT_SECS`] when unset or unparseable.  Keep this in
/// the constants module so reqwest_client.rs, s3_client.rs, and any future
/// HTTP transport read from the same source.
pub fn connect_timeout_secs() -> u64 {
    // The `.env` policy first (#243, review R9): a public setting reader, so a first call sees the opted-in file's value.
    crate::dotenv::apply();
    std::env::var(ENV_CONNECT_TIMEOUT_SECS)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_CONNECT_TIMEOUT_SECS)
}

/// Default operation timeout in seconds.
///
/// Covers the full request/response cycle once the TCP connection is established
/// (send request headers + body, wait for response headers + body).  60 s is
/// sufficient for objects up to ~6 GB at 100 MB/s and ~60 GB at 1 GB/s.
/// Override with `S3DLIO_OPERATION_TIMEOUT_SECS`.
pub const DEFAULT_OPERATION_TIMEOUT_SECS: u64 = 60;

/// Default max retry attempts for AWS SDK operations.
///
/// Matches the AWS SDK's own default (3 attempts: 1 initial + 2 retries).
/// Override with `S3DLIO_MAX_RETRY_ATTEMPTS`:
///   - `1` — fast-fail at warmup (no retries; surface dispatch failures in
///     ~connect_timeout instead of ~3× connect_timeout + exponential backoff)
///   - `5` or higher — ride out flaky-network bursts
pub const DEFAULT_MAX_RETRY_ATTEMPTS: u32 = 3;

/// Environment variable name for the SDK retry-count override.
///
/// See [`DEFAULT_MAX_RETRY_ATTEMPTS`].  Read by [`max_retry_attempts()`]
/// and applied via `aws_config::retry::RetryConfig` in `s3_client.rs`.
pub const ENV_MAX_RETRY_ATTEMPTS: &str = "S3DLIO_MAX_RETRY_ATTEMPTS";

/// Resolve the effective max retry attempts for AWS SDK calls.
///
/// Reads [`ENV_MAX_RETRY_ATTEMPTS`] each call (cheap; called only at HTTP
/// client construction time); falls back to [`DEFAULT_MAX_RETRY_ATTEMPTS`]
/// when unset or unparseable.  Clamped to a minimum of 1 (zero retries
/// would mean "do not even attempt", which the SDK rejects).
pub fn max_retry_attempts() -> u32 {
    // The `.env` policy first (#243, review R9): a public setting reader, so a first call sees the opted-in file's value.
    crate::dotenv::apply();
    std::env::var(ENV_MAX_RETRY_ATTEMPTS)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_RETRY_ATTEMPTS)
        .max(1)
}

/// Default maximum number of entries in an [`ObjectSizeCache`]
/// (`crate::object_size_cache`), including the process-global one used by
/// the S3 read path.
///
/// 1 Mi (1,048,576) entries.  A cache entry costs roughly 200 bytes for a
/// ~60-character URI (hash-map bucket and load-factor slack, the shared URI
/// allocation, and a 32-byte write-order list node; 150-300 bytes depending
/// on URI length), so a full cache is on the order of 200-300 MB -- reached
/// only by a process that has stat'ed or read more than a million distinct
/// objects within the cache TTL.
///
/// Eviction is oldest-written-first, not least-recently-read.  A loop that
/// cycles over more distinct objects than the limit (an epoch over a dataset
/// larger than the limit) evicts each size just before it is needed again and
/// gets almost no cache hits.  For such datasets raise the limit above the
/// object count.  Override with `S3DLIO_SIZE_CACHE_MAX_ENTRIES`.
pub const DEFAULT_SIZE_CACHE_MAX_ENTRIES: usize = 1_048_576;

/// Environment variable name for the size-cache entry limit.
///
/// See [`DEFAULT_SIZE_CACHE_MAX_ENTRIES`].  Read by [`size_cache_max_entries()`].
pub const ENV_SIZE_CACHE_MAX_ENTRIES: &str = "S3DLIO_SIZE_CACHE_MAX_ENTRIES";

/// Parse a raw `S3DLIO_SIZE_CACHE_MAX_ENTRIES` value: unset or unparseable
/// gives [`DEFAULT_SIZE_CACHE_MAX_ENTRIES`], and the result is at least 1.
pub(crate) fn parse_size_cache_max_entries(raw: Option<&str>) -> usize {
    raw.and_then(|s| s.trim().parse().ok())
        .unwrap_or(DEFAULT_SIZE_CACHE_MAX_ENTRIES)
        .max(1)
}

/// Resolve the effective size-cache entry limit.
///
/// Reads [`ENV_SIZE_CACHE_MAX_ENTRIES`] ONCE per process (like the other
/// cached `S3DLIO_*` settings): size caches are built per store, possibly per
/// request, so the environment is not consulted again after the first call --
/// set the variable before the first cache is created.  Unset or unparseable
/// uses [`DEFAULT_SIZE_CACHE_MAX_ENTRIES`]; clamped to a minimum of 1.
pub fn size_cache_max_entries() -> usize {
    static LIMIT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        // Latched for the process: the `.env` policy first (#243).
        crate::dotenv::apply();
        parse_size_cache_max_entries(std::env::var(ENV_SIZE_CACHE_MAX_ENTRIES).ok().as_deref())
    })
}

/// Resolve the MPI/distributed-training world size for this process.
///
/// Checked env vars (first one found wins):
///   - `OMPI_COMM_WORLD_SIZE` — Open MPI
///   - `PMI_SIZE`             — MPICH / Slurm PMI
///   - `WORLD_SIZE`           — PyTorch distributed / torchrun
///
/// Falls back to `1` (single process, i.e. "no MPI detected") when none are
/// set or unparseable. Never returns 0.
pub fn mpi_world_size() -> usize {
    // The `.env` policy first (#243, review R9): the world size may come from the opted-in file (also covers
    // `mpi_aware_thread_budget`, which reads it through this function).
    crate::dotenv::apply();
    std::env::var("OMPI_COMM_WORLD_SIZE")
        .or_else(|_| std::env::var("PMI_SIZE"))
        .or_else(|_| std::env::var("WORLD_SIZE"))
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1)
        .max(1)
}

/// Per-process worker-thread budget: available CPUs divided evenly across
/// the MPI/distributed-training world size (see [`mpi_world_size`]), floor 1.
///
/// This is the single source of truth for "how many Tokio/Rayon worker
/// threads should THIS process claim" under `mpirun -n N` / `torchrun
/// --nproc-per-node=N` launches, where every rank shares the same host and
/// an unconstrained per-process `num_cpus::get()` default causes N-way CPU
/// oversubscription. Used by both [`crate::s3_client::get_runtime_threads`]'s
/// default fallback and [`crate::s3_client::configure_thread_pools`], so a
/// caller gets the same budget whether or not it opts in explicitly.
pub fn mpi_aware_thread_budget() -> usize {
    let world_size = mpi_world_size();
    let cpus = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    (cpus / world_size).max(1)
}

/// Resolve the effective per-process worker-thread budget for any s3dlio
/// thread pool: `S3DLIO_RT_THREADS` env var override (subject to the
/// [`crate::s3_client::clamped_env_rt_threads`] sanity clamp), else the
/// MPI-aware default, else the pre-MPI-awareness `max(4, num_cpus)` floor.
///
/// This is the same precedence [`crate::s3_client::get_runtime_threads`]
/// applies to the global s3dlio Tokio runtime, factored out here so any
/// *other* dedicated pool s3dlio owns (e.g. checkpointing's own runtime --
/// see `build_checkpoint_runtime` in `python_api/python_aiml_api.rs`) can
/// get identical, single-env-knob sizing without duplicating the logic.
///
/// # Resolution order (first match wins)
/// 1. `S3DLIO_RT_THREADS` — user override, applies to every pool.  Values
///    below `RT_THREADS_LIMIT/4` are clamped up to `RT_THREADS_LIMIT` to
///    defend against downstream miscomputation (e.g. DLIO's
///    `write_threads * 1.5` with `write_threads` defaulted to 1); set
///    `S3DLIO_RT_THREADS_UNSAFE=1` to bypass the clamp.
/// 2. [`mpi_aware_thread_budget`] when `mpi_world_size() > 1`
/// 3. `max(4, num_cpus)` — floor at 4 for tiny single/dual-core VMs
pub fn effective_thread_budget() -> usize {
    // The `.env` policy first (#243, review R9): `S3DLIO_RT_THREADS` and the world size may come from the opted-in file.
    crate::dotenv::apply();
    if let Some(n) = crate::s3_client::clamped_env_rt_threads() {
        return n;
    }
    if mpi_world_size() > 1 {
        return mpi_aware_thread_budget();
    }
    let cpus = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);
    std::cmp::max(4, cpus)
}

/// Default concurrent upload limit for multipart operations
pub const DEFAULT_CONCURRENT_UPLOADS: usize = 32;

/// Buffer size for streaming operations (1 MB)
pub const DEFAULT_STREAM_BUFFER_SIZE: usize = 1024 * 1024;

/// Maximum buffer size for in-memory operations (256 MB)
pub const MAX_MEMORY_BUFFER_SIZE: usize = 256 * 1024 * 1024;

/// Minimum S3 multipart upload part size (5 MB - AWS requirement)
pub const MIN_S3_MULTIPART_PART_SIZE: usize = 5 * 1024 * 1024;

/// Default buffer capacity for multipart operations (2 MB)
pub const DEFAULT_MULTIPART_BUFFER_CAPACITY: usize = 2 * 1024 * 1024;

/// Maximum number of concurrent jobs (parallel workers) for any CLI command.
///
/// Chosen to accommodate the largest reasonable real-world deployment:
/// a 256-core host driving 16 S3 endpoints with ~16 async tasks per core
/// gives 256 × 16 = 4,096.  Beyond this, connection overhead and OS scheduler
/// pressure outweigh any throughput gains.
pub const MAX_JOBS: usize = 4_096;

/// Maximum number of endpoints in a multi-endpoint configuration.
///
/// A multi-endpoint list always has at least 1 entry (single endpoint, no load
/// balancing overhead beyond one extra indirection) and at most this many entries.
/// 32 covers the largest anticipated deployments (e.g. 32 independent storage
/// nodes each with its own IP and bucket).
pub const MAX_ENDPOINTS: usize = 32;

/// Returns the maximum allowed number of endpoints in a multi-endpoint configuration.
///
/// Downstream tools (e.g. sai3-bench) should call this rather than hard-coding the
/// limit, so the single source of truth remains in s3dlio.
///
/// # Example
/// ```rust
/// assert!(s3dlio::constants::max_endpoints() >= 1);
/// ```
pub fn max_endpoints() -> usize {
    MAX_ENDPOINTS
}

// ============================================================================
// RangeEngine Configuration Constants
// ============================================================================
//
// RangeEngine provides concurrent range downloads for large objects, splitting
// them into parallel requests to maximize throughput on high-bandwidth networks.
//
// **IMPORTANT: RangeEngine is DISABLED BY DEFAULT as of v0.9.6**
// Performance testing revealed that the extra HEAD/STAT request on every GET
// operation causes up to 50% slowdown for typical workloads. RangeEngine must
// be explicitly enabled for large-file workloads where benefits outweigh overhead.
//
// PERFORMANCE CONSIDERATIONS:
// - Small objects (< 16 MiB): HEAD overhead outweighs any parallel benefit
// - Medium objects (16-64 MiB): 20-40% faster ONLY if RangeEngine enabled
// - Large objects (> 64 MiB): 30-60% faster ONLY if RangeEngine enabled
// - Typical workloads: Slower due to 2x requests (HEAD + GET) per object
//
// THRESHOLD SELECTION:
// The 16 MiB default threshold (when RangeEngine is enabled) was chosen to:
// 1. Skip range splitting for typical objects that don't benefit
// 2. Only engage RangeEngine for moderately large files (>= 16 MiB)
// 3. Balance performance vs overhead for large-file workloads
//
// HOW TO ENABLE:
// ```rust
// // Enable for Azure large-file workloads
// let config = AzureConfig {
//     enable_range_engine: true,  // Must explicitly enable
//     range_engine: RangeEngineConfig {
//         min_split_size: 16 * 1024 * 1024,  // 16 MiB default
//         ..Default::default()
//     },
// };
// let store = AzureObjectStore::with_config(config);
// ```
//
// For small-object benchmarks, leave RangeEngine disabled (default behavior).
// ============================================================================

/// Default chunk size for RangeEngine concurrent downloads (64 MB)
/// Used for splitting large objects into parallel range requests
pub const DEFAULT_RANGE_ENGINE_CHUNK_SIZE: usize = 64 * 1024 * 1024;

/// Default maximum concurrent ranges for RangeEngine (32)
/// Controls parallelism for concurrent range downloads
pub const DEFAULT_RANGE_ENGINE_MAX_CONCURRENT: usize = 32;

/// Universal default minimum object size to trigger RangeEngine (now 32 MiB, v0.9.60+)
///
/// This threshold applies to all storage backends (S3, Azure, GCS, file://, direct://)
/// when RangeEngine is enabled.
///
/// **S3 inline optimization** (`S3DLIO_ENABLE_RANGE_OPTIMIZATION`, enabled by default in v0.9.60):
/// Uses this 32 MiB threshold unless overridden by `S3DLIO_RANGE_THRESHOLD_MB`.
/// Set `S3DLIO_ENABLE_RANGE_OPTIMIZATION=0` to disable.
///
/// **Per-store RangeEngine** (`enable_range_engine` config field):
/// Still `false` by default on all backends — must be explicitly enabled via config.
/// This avoids the HEAD stat overhead for workloads with small or mixed objects.
///
/// **Performance Impact (when RangeEngine is active):**
/// - Objects < 32 MiB: Single GET request (fast path, no range splitting)
/// - Objects >= 32 MiB: HEAD + concurrent range GETs (RangeEngine path)
///
/// **When to Override:**
/// - Set `S3DLIO_RANGE_THRESHOLD_MB` to change S3 inline optimization threshold
/// - Set higher (e.g., 64 MiB) to avoid HEAD overhead for large-object workloads
/// - Set lower (e.g., 4 MiB) for dedicated large-file high-latency networks
///
/// **Example Configuration (per-store RangeEngine):**
/// ```ignore
/// use s3dlio::object_store::{GcsConfig, RangeEngineConfig};
///
/// // Enable RangeEngine for large-file workload
/// let config = GcsConfig {
///     enable_range_engine: true,  // Must explicitly enable for per-store RangeEngine
///     range_engine: RangeEngineConfig {
///         min_split_size: 32 * 1024 * 1024,  // 32 MiB threshold
///         ..Default::default()
///     },
/// };
/// ```
pub const DEFAULT_RANGE_ENGINE_THRESHOLD: u64 = 32 * 1024 * 1024;

/// Legacy alias for S3 backend (now uses universal threshold)
/// Kept for backward compatibility, but DEFAULT_RANGE_ENGINE_THRESHOLD is preferred
#[deprecated(since = "0.9.5", note = "Use DEFAULT_RANGE_ENGINE_THRESHOLD instead")]
pub const DEFAULT_S3_RANGE_ENGINE_THRESHOLD: u64 = DEFAULT_RANGE_ENGINE_THRESHOLD;

/// Legacy alias for Azure backend (now uses universal threshold)
/// Kept for backward compatibility, but DEFAULT_RANGE_ENGINE_THRESHOLD is preferred
#[deprecated(since = "0.9.5", note = "Use DEFAULT_RANGE_ENGINE_THRESHOLD instead")]
pub const DEFAULT_AZURE_RANGE_ENGINE_THRESHOLD: u64 = DEFAULT_RANGE_ENGINE_THRESHOLD;

/// Legacy alias for GCS backend (now uses universal threshold)
/// Kept for backward compatibility, but DEFAULT_RANGE_ENGINE_THRESHOLD is preferred
#[deprecated(since = "0.9.5", note = "Use DEFAULT_RANGE_ENGINE_THRESHOLD instead")]
pub const DEFAULT_GCS_RANGE_ENGINE_THRESHOLD: u64 = DEFAULT_RANGE_ENGINE_THRESHOLD;

/// Minimum object size to trigger RangeEngine for File backend (16 MiB)
/// Local file systems benefit less from range parallelism due to seek overhead,
/// but can still benefit for very large files on fast SSDs/NVMe storage
pub const DEFAULT_FILE_RANGE_ENGINE_THRESHOLD: u64 = DEFAULT_RANGE_ENGINE_THRESHOLD;

/// Minimum object size to trigger RangeEngine for DirectIO backend (16 MiB)
/// DirectIO already bypasses page cache, so range parallelism has limited benefit
/// Higher threshold reduces overhead from O_DIRECT alignment requirements
pub const DEFAULT_DIRECTIO_RANGE_ENGINE_THRESHOLD: u64 = DEFAULT_RANGE_ENGINE_THRESHOLD;

/// Default range request timeout (30 seconds)
pub const DEFAULT_RANGE_TIMEOUT_SECS: u64 = 30;

// =============================================================================
// CLI Progress Display Constants
// =============================================================================

/// Rolling-window duration (seconds) used to compute the current obj/s rate in
/// the `s3-cli get` progress bar (URI-prefix mode).
///
/// Every 500 ms the background updater records a (timestamp, completed_count)
/// sample.  Samples older than this window are evicted; the rate is calculated
/// as Δcompleted / Δtime across the oldest surviving sample, giving a responsive
/// but stable "current rate" display.
///
/// **Why 5 s?**
/// - Short enough to track real acceleration / deceleration (e.g. GCS RAPID
///   warming up over the first few seconds).
/// - Long enough to smooth out single-object jitter without lag.
///
/// **Note on PUT (put_many_cmd):** the PUT progress bar drives indicatif's
/// built-in {bytes_per_sec} field, which uses indicatif's own internal
/// exponential moving average — this constant does not apply there.
pub const CLI_RATE_WINDOW_SECS: u64 = 5;

/// Page size bounds for validation
pub const MIN_PAGE_SIZE: usize = 512;
pub const MAX_PAGE_SIZE: usize = 64 * 1024;

/// URI scheme constants
pub const SCHEME_FILE: &str = "file://";
pub const SCHEME_DIRECT: &str = "direct://";
pub const SCHEME_S3: &str = "s3://";
pub const SCHEME_AZURE: &str = "az://";
pub const SCHEME_GCS: &str = "gs://";

/// File I/O operation modes
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoMode {
    /// Regular buffered I/O using OS page cache
    Buffered,
    /// Direct I/O bypassing OS page cache
    Direct,
}

impl IoMode {
    /// Returns the appropriate URI scheme for this I/O mode
    pub fn uri_scheme(&self) -> &'static str {
        match self {
            IoMode::Buffered => SCHEME_FILE,
            IoMode::Direct => SCHEME_DIRECT,
        }
    }
}

// =============================================================================
// Data Generation Constants
// =============================================================================

/// Block size for LEGACY data_gen.rs algorithm (4096 bytes = 4 KiB)
/// This is the fundamental unit for the old BASE_BLOCK-based algorithm.
/// Changed to 4096 (standard page size) for:
/// - Memory page alignment (optimal for direct I/O)
/// - Better compatibility with O_DIRECT operations
/// - Standard filesystem block size
pub const BLK_SIZE: usize = 4096;

/// Half block size for LEGACY data_gen.rs internal calculations
pub const HALF_BLK: usize = BLK_SIZE / 2;

/// Data generation block size for OPTIMIZED data_gen_alt.rs algorithm (1 MiB)
///
/// **PERFORMANCE OPTIMIZATION (backported from dgen-rs v0.1.5)**:
/// Optimal L3 cache utilization provides ~3x performance improvement:
/// - Better parallelization across cores  
/// - Reduced thread pool overhead
/// - Optimal for modern CPUs (Emerald Rapid, Sapphire Rapids)
/// - 34% performance boost vs 64 KB blocks
///
/// **Benchmarks (dgen-rs v0.1.5)**:
/// - UMA systems: 10.80 GB/s per core (C4-16, 8 cores)
/// - Aggregate: 86-163 GB/s on single-socket systems
/// - Compression ratio 2.0: 1.3-1.5x additional speedup
///
/// Data generation block size (1 MiB) - used by data_gen_alt for controlled
/// deduplication and compression patterns. Named specifically to avoid
/// confusion with other block size concepts (filesystem, storage, etc.).
pub const DGEN_BLOCK_SIZE: usize = 1024 * 1024; // 1 MiB

/// Modification region size for randomization (32 bytes)
/// This determines the size of regions that get randomized within blocks
pub const MOD_SIZE: usize = 32;

// =============================================================================
// Op-Log Streaming Constants
// =============================================================================

/// Default chunk size for streaming op-log reads (1 MB)
/// Used for BufReader capacity when parsing op-log files
pub const DEFAULT_OPLOG_CHUNK_SIZE: usize = 1024 * 1024;

/// Default op-log entry buffer capacity (1024 entries)
/// Channel buffer size for background parsing thread
pub const DEFAULT_OPLOG_ENTRY_BUFFER: usize = 1024;

/// Environment variable for op-log read buffer size
pub const ENV_OPLOG_READ_BUF: &str = "S3DLIO_OPLOG_READ_BUF";

/// Environment variable for op-log chunk size
pub const ENV_OPLOG_CHUNK_SIZE: &str = "S3DLIO_OPLOG_CHUNK_SIZE";

// =============================================================================
// HTTP/2 and Connection Pool Environment Variables and Defaults
// =============================================================================
//
// These constants centralise every knob that controls the HTTP client used by
// s3dlio's AWS SDK / reqwest layer.  They are intentionally all in one place
// so that tooling authors (sai3-bench, dl-driver, warpio …) can discover and
// override them without hunting through implementation files.
//
// USAGE FOR DOWNSTREAM LIBRARY USERS
// -----------------------------------
// Read the current effective settings at runtime via `H2WindowConfig::from_env()`.
// Override individual settings by setting the corresponding environment variable
// before constructing an S3 client.
//
// Example (sai3-bench):
// ```rust
// use s3dlio::constants::{ENV_S3DLIO_H2C, ENV_H2_STREAM_WINDOW_MB};
// std::env::set_var(ENV_S3DLIO_H2C, "1");
// std::env::set_var(ENV_H2_STREAM_WINDOW_MB, "16");
// ```
// =============================================================================

// ── HTTP version control ───────────────────────────────────────────────────

/// Environment variable controlling HTTP/2 mode on plain `http://` endpoints.
///
/// | Value | Behaviour |
/// |-------|-----------|
/// | `1`, `true`, `yes`, `on`, `enable` | Force h2c (HTTP/2 prior-knowledge cleartext); no fallback |
/// | `0`, `false`, `no`, `off`, `disable` | Force HTTP/1.1; skip auto-probe |
/// | *(unset)* | HTTP/1.1 (default — equivalent to `0`) |
///
/// **Default is HTTP/1.1** for `http://`. Set `S3DLIO_H2C=1` to opt in to h2c on `http://`.
/// See also [`ENV_S3DLIO_ENABLE_HTTP2`] for a master switch that opts both `http://` and
/// `https://` into HTTP/2 in one setting.
pub const ENV_S3DLIO_H2C: &str = "S3DLIO_H2C";

/// Default HTTP/2 cleartext (h2c) behaviour when [`ENV_S3DLIO_H2C`] is not set: `false` (HTTP/1.1).
///
/// **Default is `false`** since v0.9.92. Benchmarking on `http://` endpoints showed HTTP/2
/// reduces throughput relative to HTTP/1.1 with an unlimited connection pool. Set `S3DLIO_H2C=1`
/// to opt in to h2c for storage systems that require it.
pub const DEFAULT_H2C_ENABLED: bool = false;

/// Environment variable controlling HTTP/2 mode on `https://` endpoints.
///
/// | Value | Behaviour |
/// |-------|-----------|
/// | `1`, `true`, `yes`, `on`, `enable` | Opt in to HTTP/2 (via TLS ALPN); apply H2 window tuning |
/// | `0`, `false`, `no`, `off`, `disable`, *(unset)* | HTTP/1.1 (default — reqwest builder calls `.http1_only()`) |
///
/// **Default changed in v0.9.108 (issue #148)**: previously `https://` endpoints always
/// negotiated HTTP/2 via TLS ALPN unconditionally. Benchmarking against real-world S3-compatible
/// endpoints showed HTTP/2 is frequently slower than HTTP/1.1 for this workload class,
/// primarily due to single-connection flow-control constraints even with adaptive windows.
/// The default is now HTTP/1.1 — matching the `http://` default. Set `S3DLIO_HTTPS_H2=1`
/// to restore the pre-v0.9.108 behaviour, or set [`ENV_S3DLIO_ENABLE_HTTP2`] to opt in
/// to HTTP/2 on both `http://` and `https://` in one variable.
///
/// **BREAKING CHANGE from v0.9.106**: any deployment that relied on ALPN-negotiated H2
/// on `https://` will now see HTTP/1.1 unless this var (or the master switch) is set.
pub const ENV_S3DLIO_HTTPS_H2: &str = "S3DLIO_HTTPS_H2";

/// Default HTTP/2 opt-in state for `https://` when [`ENV_S3DLIO_HTTPS_H2`] is unset:
/// `false` (HTTP/1.1). See [`ENV_S3DLIO_HTTPS_H2`] for rationale.
pub const DEFAULT_HTTPS_H2_ENABLED: bool = false;

/// Master switch: opt in to HTTP/2 on **both** `http://` and `https://` in one variable.
///
/// | Value | Behaviour |
/// |-------|-----------|
/// | `1`, `true`, `yes`, `on`, `enable` | Equivalent to setting both `S3DLIO_H2C=1` and `S3DLIO_HTTPS_H2=1` |
/// | `0`, `false`, `no`, `off`, `disable`, *(unset)* | No effect; the per-scheme vars decide |
///
/// Precedence: HTTP/2 is enabled for scheme *S* iff the per-scheme variable for *S* is
/// truthy **or** this master switch is truthy. Setting this switch cannot *disable* H2
/// on a scheme where the per-scheme var already enabled it — but that's a non-issue
/// since both defaults are already "off".
pub const ENV_S3DLIO_ENABLE_HTTP2: &str = "S3DLIO_ENABLE_HTTP2";

// ── Payload signing bypass ─────────────────────────────────────────────────

/// Environment variable to disable SigV4 payload (body) signing on S3 PUT requests.
///
/// When set to `1`, `true`, `yes`, `on`, or `enable`, the AWS SDK will send
/// `x-amz-content-sha256: UNSIGNED-PAYLOAD` instead of computing a SHA-256
/// digest of the entire request body on every PUT.  This eliminates per-request
/// CPU cost proportional to object size and can substantially improve PUT
/// throughput on endpoints that don't enforce payload integrity (e.g. MinIO,
/// s3-ultra, Ceph RGW in path-style mode).
///
/// **WARNING**: Disabling payload signing removes a data-integrity check.  Only
/// use this against trusted, internal storage endpoints.  Never enable on
/// production AWS S3 workloads where data integrity is critical.
pub const ENV_UNSIGNED_PAYLOAD: &str = "S3DLIO_UNSIGNED_PAYLOAD";

/// Default payload signing behaviour: `false` (signed, safe default).
pub const DEFAULT_UNSIGNED_PAYLOAD: bool = false;

// ── Connection pool tunables ───────────────────────────────────────────────

/// Maximum idle connections kept in the pool per host (default: [`DEFAULT_POOL_MAX_IDLE_PER_HOST`]).
///
/// The default is unlimited (`usize::MAX`); idle connections are still evicted
/// after `S3DLIO_POOL_IDLE_TIMEOUT_SECS` (default 90 s) of inactivity.
/// Set to a positive integer to impose a hard ceiling — useful for
/// memory-constrained environments or testing specific pool-pressure scenarios.
pub const ENV_POOL_MAX_IDLE_PER_HOST: &str = "S3DLIO_POOL_MAX_IDLE_PER_HOST";

/// Default maximum idle connections per host in the reqwest connection pool.
///
/// Set to [`usize::MAX`] (unlimited) so the pool always retains one idle
/// connection per concurrent worker.  Under high concurrency (e.g. t=128) a
/// fixed cap of 32 caused 96 workers to tear down their connection on every
/// request completion, paying a full TCP handshake penalty each round-trip
/// (~794 µs overhead on loopback, ~5 µs without).
///
/// Connections are still cleaned up by `pool_idle_timeout` (default 90 s),
/// so the pool shrinks automatically when load drops.
///
/// Override with `S3DLIO_POOL_MAX_IDLE_PER_HOST=<n>` to impose a hard ceiling.
pub const DEFAULT_POOL_MAX_IDLE_PER_HOST: usize = usize::MAX;

/// Idle connection pool timeout in seconds (default: [`DEFAULT_POOL_IDLE_TIMEOUT_SECS`]).
///
/// Connections idle longer than this are closed and removed from the pool.
/// Reduce for bursty workloads on shared networks; increase for long-running
/// steady-state benchmarks to avoid reconnect overhead.
pub const ENV_POOL_IDLE_TIMEOUT_SECS: &str = "S3DLIO_POOL_IDLE_TIMEOUT_SECS";

/// Default idle connection pool timeout in seconds.
pub const DEFAULT_POOL_IDLE_TIMEOUT_SECS: u64 = 90;

// ── HTTP/2 flow-control window tunables ───────────────────────────────────
//
// HTTP/2 uses per-stream and per-connection *receive* flow-control windows.
// The windows advertise how many bytes the *client* is willing to buffer from
// the server before issuing WINDOW_UPDATE credits.
//
// Impact by operation:
//   GETs  — directly gates download throughput (server sends body data)
//   PUTs  — response bodies are tiny (~200 B); negligible impact on upload rate
//
// Two modes are available (mutually exclusive):
//
//   ADAPTIVE (default, recommended)
//     `http2_adaptive_window(true)` — hyper/h2 sends periodic H2 PING frames,
//     measures RTT, and computes BDP = bandwidth × RTT.  Issues WINDOW_UPDATE
//     frames proactively to keep the window ≥ BDP.  Self-tunes from 64 KB to
//     hundreds of MB automatically.  Works for all object sizes without manual
//     tuning.
//
//   STATIC (override)
//     Set S3DLIO_H2_STREAM_WINDOW_MB and/or S3DLIO_H2_CONN_WINDOW_MB.
//     Adaptive mode is disabled; the fixed values are used instead.
//     Useful for deterministic benchmarks or servers that ignore PINGs.
//
// NOTE: adaptive_window overrides static sizes — they are mutually exclusive
// in the reqwest/hyper API.  Static values are only applied when adaptive is OFF.
//
// The maximum allowed by the HTTP/2 spec is 2^31-1 = 2,147,483,647 bytes (~2 GiB).
// Practical upper limit is typically 256 MiB (beyond that servers may not honour it).

/// Per-stream HTTP/2 receive window size in **MiB** (static mode only).
///
/// Only consulted when `S3DLIO_H2_ADAPTIVE_WINDOW` is unset or `0`.
/// When set, adaptive window mode is disabled and this exact size is used.
///
/// Typical values:
/// - `4`  — 4 MiB: good for LAN with moderate object sizes (up to ~64 MiB)
/// - `16` — 16 MiB: high-throughput LAN / NVMe-oF storage
/// - `64` — 64 MiB: ultra-low-latency fabric (InfiniBand, 200GbE)
///
/// Must fit in a `u32` and must not exceed 2047 MiB (HTTP/2 spec limit ~2 GiB).
pub const ENV_H2_STREAM_WINDOW_MB: &str = "S3DLIO_H2_STREAM_WINDOW_MB";

/// Default per-stream HTTP/2 receive window size used in static mode (MiB).
pub const DEFAULT_H2_STREAM_WINDOW_MB: u32 = 4;

/// Per-connection HTTP/2 receive window size in **MiB** (static mode only).
///
/// The connection window is the aggregate across all concurrent streams on
/// one TCP connection.  Should be at least `stream_window × max_concurrent_streams`
/// to avoid the connection window becoming the bottleneck.
///
/// Only consulted when `S3DLIO_H2_ADAPTIVE_WINDOW` is unset or `0`.
/// Defaults to `4 × S3DLIO_H2_STREAM_WINDOW_MB` when unset.
pub const ENV_H2_CONN_WINDOW_MB: &str = "S3DLIO_H2_CONN_WINDOW_MB";

/// Default per-connection HTTP/2 receive window size used in static mode (MiB).
/// Set to 4× the default stream window to avoid connection-level bottlenecks.
pub const DEFAULT_H2_CONN_WINDOW_MB: u32 = DEFAULT_H2_STREAM_WINDOW_MB * 4;

/// Enable or disable HTTP/2 adaptive window mode (default: **enabled**).
///
/// | Value | Behaviour |
/// |-------|-----------|
/// | *(unset)* | Adaptive ON (default) |
/// | `1`, `true`, `yes`, `on` | Adaptive ON explicitly |
/// | `0`, `false`, `no`, `off` | Adaptive OFF; use static window values |
///
/// When adaptive is ON, the values in `S3DLIO_H2_STREAM_WINDOW_MB` and
/// `S3DLIO_H2_CONN_WINDOW_MB` are **ignored** (reqwest/hyper override them).
pub const ENV_H2_ADAPTIVE_WINDOW: &str = "S3DLIO_H2_ADAPTIVE_WINDOW";

/// Maximum legal HTTP/2 flow-control window size per the spec (2^31 - 1 bytes).
/// Attempting to set a larger value is a protocol error.
pub const H2_MAX_WINDOW_BYTES: u32 = 0x7FFF_FFFF;

/// Maximum practical per-stream or per-connection window size we will accept
/// from env-var input (256 MiB).  Values above this are silently clamped.
pub const H2_WINDOW_MB_HARD_CAP: u32 = 256;

// ── Public configuration view for downstream library users ────────────────

/// Resolved HTTP/2 window configuration, ready for use when building a reqwest client.
///
/// Downstream crates (sai3-bench, dl-driver, …) can call [`H2WindowConfig::from_env`]
/// to inspect the active configuration, or construct it manually to override settings
/// without touching environment variables.
///
/// # Example (sai3-bench)
/// ```rust,no_run
/// use s3dlio::constants::H2WindowConfig;
/// let cfg = H2WindowConfig::from_env();
/// println!("adaptive={} stream={}MiB conn={}MiB", cfg.adaptive, cfg.stream_window_mb, cfg.conn_window_mb);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct H2WindowConfig {
    /// `true` = hyper BDP adaptive window (overrides static sizes).
    /// `false` = static sizes below are used.
    pub adaptive: bool,
    /// Per-stream receive window in MiB (only used when `adaptive == false`).
    pub stream_window_mb: u32,
    /// Per-connection receive window in MiB (only used when `adaptive == false`).
    pub conn_window_mb: u32,
}

impl Default for H2WindowConfig {
    /// Returns the default configuration: adaptive mode on, static sizes at their defaults.
    fn default() -> Self {
        Self {
            adaptive: true,
            stream_window_mb: DEFAULT_H2_STREAM_WINDOW_MB,
            conn_window_mb: DEFAULT_H2_CONN_WINDOW_MB,
        }
    }
}

impl H2WindowConfig {
    /// Build the active configuration by reading the three env vars.
    ///
    /// Parsing rules:
    /// - `S3DLIO_H2_ADAPTIVE_WINDOW` unset or truthy → adaptive ON
    /// - `S3DLIO_H2_ADAPTIVE_WINDOW` falsy → adaptive OFF, static sizes used
    /// - `S3DLIO_H2_STREAM_WINDOW_MB` / `S3DLIO_H2_CONN_WINDOW_MB` are clamped
    ///   to [`H2_WINDOW_MB_HARD_CAP`] and must be > 0.
    pub fn from_env() -> Self {
        // The `.env` policy first (#243, review R9): a public setting reader, so a first call sees the opted-in file's values.
        crate::dotenv::apply();
        // ── Adaptive flag ──────────────────────────────────────────────────
        let adaptive = match std::env::var(ENV_H2_ADAPTIVE_WINDOW) {
            Err(_) => true, // default: adaptive ON
            Ok(v) => !matches!(
                v.to_lowercase().as_str(),
                "0" | "false" | "no" | "off" | "disable"
            ),
        };

        // ── Static window sizes (only meaningful when adaptive == false) ───
        let stream_mb = std::env::var(ENV_H2_STREAM_WINDOW_MB)
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|&v| v > 0)
            .map(|v| v.min(H2_WINDOW_MB_HARD_CAP))
            .unwrap_or(DEFAULT_H2_STREAM_WINDOW_MB);

        let conn_mb = std::env::var(ENV_H2_CONN_WINDOW_MB)
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .filter(|&v| v > 0)
            .map(|v| v.min(H2_WINDOW_MB_HARD_CAP))
            .unwrap_or(stream_mb.saturating_mul(4).min(H2_WINDOW_MB_HARD_CAP));

        Self {
            adaptive,
            stream_window_mb: stream_mb,
            conn_window_mb: conn_mb,
        }
    }

    /// Convert the stream window to bytes for use in the reqwest builder.
    #[inline]
    pub fn stream_window_bytes(&self) -> u32 {
        self.stream_window_mb
            .saturating_mul(1024 * 1024)
            .min(H2_MAX_WINDOW_BYTES)
    }

    /// Convert the connection window to bytes for use in the reqwest builder.
    #[inline]
    pub fn conn_window_bytes(&self) -> u32 {
        self.conn_window_mb
            .saturating_mul(1024 * 1024)
            .min(H2_MAX_WINDOW_BYTES)
    }
}

// =============================================================================
// Custom Endpoint Environment Variables
// =============================================================================

/// Primary environment variable for custom Azure Blob Storage endpoint
/// Example: AZURE_STORAGE_ENDPOINT=http://localhost:10000
pub const ENV_AZURE_STORAGE_ENDPOINT: &str = "AZURE_STORAGE_ENDPOINT";

/// Alternative environment variable for custom Azure Blob Storage endpoint
/// Example: AZURE_BLOB_ENDPOINT_URL=http://localhost:10000
pub const ENV_AZURE_BLOB_ENDPOINT_URL: &str = "AZURE_BLOB_ENDPOINT_URL";

// =============================================================================
// Azure credential selection (issue #253)
// =============================================================================

/// Selects the Azure credential: auto | cli | managed-identity | client-secret |
/// client-certificate | workload-identity | sas | none (case-insensitive).
pub const ENV_AZURE_CREDENTIAL: &str = "S3DLIO_AZURE_CREDENTIAL";
/// The SAS token appended to every Azure request URL in `sas` mode (leading `?` optional).
pub const ENV_AZURE_SAS_TOKEN: &str = "S3DLIO_AZURE_SAS_TOKEN";
/// Standard Azure identity variables s3dlio reads (see docs/Environment_Variables.md).
pub const ENV_AZURE_CLIENT_ID: &str = "AZURE_CLIENT_ID";
pub const ENV_AZURE_TENANT_ID: &str = "AZURE_TENANT_ID";
pub const ENV_AZURE_CLIENT_SECRET: &str = "AZURE_CLIENT_SECRET";
pub const ENV_AZURE_CLIENT_CERTIFICATE_PATH: &str = "AZURE_CLIENT_CERTIFICATE_PATH";
pub const ENV_AZURE_CLIENT_CERTIFICATE_PASSWORD: &str = "AZURE_CLIENT_CERTIFICATE_PASSWORD";
pub const ENV_AZURE_FEDERATED_TOKEN_FILE: &str = "AZURE_FEDERATED_TOKEN_FILE";
/// Set by Azure App Service / Arc / Functions hosts that offer a managed-identity endpoint.
pub const ENV_AZURE_IDENTITY_ENDPOINT: &str = "IDENTITY_ENDPOINT";
/// Set by Azure ML / Cloud Shell hosts that offer a managed-identity endpoint.
pub const ENV_AZURE_MSI_ENDPOINT: &str = "MSI_ENDPOINT";

/// How long `auto` waits for the Azure instance metadata service (169.254.169.254:80) when it
/// asks whether this is an Azure VM: this long to connect, then this long (one overall
/// deadline) to read the answer, so a probe takes at most twice this (0.5 s). On a VM the
/// answer is immediate; elsewhere the connection fails or times out. The probe runs on the
/// blocking pool, and only when no other chain member decides first.
pub const AZURE_IMDS_PROBE_TIMEOUT_MS: u64 = 250;

/// How long a "the metadata service does not look like Azure" answer is believed, in seconds, so
/// a failing `auto` chain does not probe again on every request.
pub const AZURE_IMDS_NEGATIVE_CACHE_SECS: u64 = 30;

/// Deadline for `auto`'s managed-identity attempt, in milliseconds (the SDK's own retry schedule
/// for a metadata service that answers 404/410/5xx is about 72 s). Override with
/// `S3DLIO_AZURE_AUTO_MI_TIMEOUT_MS`.
pub const DEFAULT_AZURE_AUTO_MI_TIMEOUT_MS: u64 = 5000;
pub const ENV_AZURE_AUTO_MI_TIMEOUT_MS: &str = "S3DLIO_AZURE_AUTO_MI_TIMEOUT_MS";

// =============================================================================
// GCS gRPC Read Timeout / Progress Constants
// =============================================================================

/// Number of bytes in one GiB, used for size-scaled timeout calculations.
pub const BYTES_PER_GIB: u64 = 1024 * 1024 * 1024;

/// Environment variable to force a fixed per-chunk read timeout (seconds).
///
/// - `0` disables chunk timeout entirely.
/// - unset uses size-scaled defaults.
pub const ENV_GCS_READ_CHUNK_TIMEOUT_SECS: &str = "S3DLIO_GCS_READ_CHUNK_TIMEOUT_SECS";

/// Environment variable to force a fixed per-attempt bidi connect timeout (seconds).
///
/// - `0` is ignored (falls back to scaled defaults).
/// - unset uses size-scaled defaults.
pub const ENV_GCS_BIDI_ATTEMPT_TIMEOUT_SECS: &str = "S3DLIO_GCS_BIDI_ATTEMPT_TIMEOUT_SECS";

/// Environment variable controlling INFO progress cadence while waiting for chunks.
pub const ENV_GCS_READ_PROGRESS_INTERVAL_SECS: &str = "S3DLIO_GCS_READ_PROGRESS_INTERVAL_SECS";

/// Environment variable for timeout slope (seconds per GiB).
pub const ENV_GCS_READ_TIMEOUT_SECS_PER_GIB: &str = "S3DLIO_GCS_READ_TIMEOUT_SECS_PER_GIB";

/// Environment variable for minimum timeout floor (seconds).
pub const ENV_GCS_READ_TIMEOUT_MIN_SECS: &str = "S3DLIO_GCS_READ_TIMEOUT_MIN_SECS";

/// Environment variable for maximum timeout cap (seconds).
pub const ENV_GCS_READ_TIMEOUT_MAX_SECS: &str = "S3DLIO_GCS_READ_TIMEOUT_MAX_SECS";

/// Default timeout slope: each GiB contributes this many seconds.
pub const DEFAULT_GCS_READ_TIMEOUT_SECS_PER_GIB: u64 = 45;

/// Minimum timeout floor (seconds).
pub const DEFAULT_GCS_READ_TIMEOUT_MIN_SECS: u64 = 12;

/// Maximum timeout cap (seconds).
pub const DEFAULT_GCS_READ_TIMEOUT_MAX_SECS: u64 = 300;

/// Default INFO progress interval while chunk waits are in progress.
pub const DEFAULT_GCS_READ_PROGRESS_INTERVAL_SECS: u64 = 5;

/// Custom GCS endpoint URL for the official client (an emulator or test server). Wins over
/// [`ENV_STORAGE_EMULATOR_HOST`]. The client then uses anonymous credentials.
/// Example: GCS_ENDPOINT_URL=http://localhost:4443
pub const ENV_GCS_ENDPOINT_URL: &str = "GCS_ENDPOINT_URL";

/// GCS emulator convention environment variable (STORAGE_EMULATOR_HOST=host:port).
/// If the value has no `http://` or `https://` scheme, `http://` is prepended. The same
/// rules apply as for [`ENV_GCS_ENDPOINT_URL`]: anonymous credentials.
/// Example: STORAGE_EMULATOR_HOST=localhost:4443
pub const ENV_STORAGE_EMULATOR_HOST: &str = "STORAGE_EMULATOR_HOST";

/// A base random block of BLK_SIZE bytes, generated once and shared.
/// Used by the single-pass data generation algorithm.
pub static A_BASE_BLOCK: Lazy<Arc<Vec<u8>>> = Lazy::new(|| {
    let mut rng = rand::rngs::ThreadRng::default();
    let mut block = vec![0u8; BLK_SIZE];
    rng.fill_bytes(&mut block[..]);
    Arc::new(block)
});

/// Original base random block of BLK_SIZE bytes, generated once.
/// Used by the two-pass reference implementation for comparison.
pub static BASE_BLOCK: Lazy<Vec<u8>> = Lazy::new(|| {
    let mut rng = rand::rngs::ThreadRng::default();
    let mut block = vec![0u8; BLK_SIZE];
    rng.fill_bytes(&mut block[..]);
    block
});

/// Upper bound on how long the Python `atexit` hook waits, with the GIL
/// released, for in-flight async result deliveries to finish before the
/// interpreter finalizes (issue #192).
///
/// A delivery normally completes in microseconds once the exiting thread
/// releases the GIL; the bound exists only so a pathologically stuck delivery
/// can never hang process exit.
pub const ASYNC_EXIT_DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

/// How long the buffer-export releaser thread waits after being woken before it takes the
/// GIL (25 ms). A `put_bytes` caller, or the `put_bytes_async` delivery path, normally drains
/// the released exports itself within microseconds; the grace period lets them, so the
/// releaser finds the queue empty and never contends for the GIL on the hot path. It only
/// matters when nobody drains (for example a cancelled awaitable).
pub const EXPORT_RELEASE_GRACE: std::time::Duration = std::time::Duration::from_millis(25);

/// Upper bound (30 s) on how long the interpreter-exit callback waits, with the GIL detached, for
/// the buffer-export releaser thread to leave an attach scope it had already entered (issue
/// #238, Python 3.12 shutdown race). Generous because a draining releaser can run an exporter's
/// Python `__release_buffer__`; it exists only so a wedged exporter can never hang exit forever.
/// When it is exceeded the event is logged and counted
/// (`s3dlio._put_export_stats()["shutdown_attach_wait_timeouts"]`) and exit continues.
pub const RELEASER_ATTACH_DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// How often (50 ms) the awaitable of `put_bytes_async(borrow=True)` re-checks the export barrier
/// when the barrier machinery itself failed (an internal error, for example a failed callback
/// registration) and it can no longer be notified: it polls instead of completing unsafely.
pub const EXPORT_BARRIER_POLL_INTERVAL: std::time::Duration = std::time::Duration::from_millis(50);

/// Default for how long a put waits, after the store finished with it, for every native reader
/// of a borrowed Python buffer to let go (30 s). If a backend still holds a clone of the payload
/// after this, the buffer stays PINNED (never released while it can still be read) and the missed
/// deadline is logged and counted (`s3dlio._put_export_stats()["quiesce_timeouts"]`). The blocking
/// `put_bytes` then raises `ExportStillReferencedError` (the data may have been written, but the
/// buffer must not be reused); the awaitable of `put_bytes_async(borrow=True)` does NOT complete:
/// it warns and keeps waiting for the readers, so cancellation or a timeout wrapper can never turn
/// a still-borrowed buffer into a reuse point. Override with [`ENV_EXPORT_QUIESCE_TIMEOUT_SECS`].
/// The blocking wait is not interruptible by signals.
pub const EXPORT_QUIESCE_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Environment variable overriding [`EXPORT_QUIESCE_TIMEOUT`], in seconds (fractions allowed,
/// at least 0.001; unusable values select the default).
pub const ENV_EXPORT_QUIESCE_TIMEOUT_SECS: &str = "S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS";

/// The effective quiesce timeout (reads [`ENV_EXPORT_QUIESCE_TIMEOUT_SECS`] on each call). A set
/// but unusable value falls back to the default, with a one-time warning.
pub fn export_quiesce_timeout() -> std::time::Duration {
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    crate::dotenv::apply();
    match std::env::var(ENV_EXPORT_QUIESCE_TIMEOUT_SECS) {
        Err(_) => EXPORT_QUIESCE_TIMEOUT,
        Ok(raw) => parse_export_quiesce_timeout_checked(&raw).unwrap_or_else(|| {
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                tracing::warn!(
                    "s3dlio: ignoring {ENV_EXPORT_QUIESCE_TIMEOUT_SECS}={raw:?} (expected seconds \
                     between 0.001 and 86400); using the default {EXPORT_QUIESCE_TIMEOUT:?}"
                );
            }
            EXPORT_QUIESCE_TIMEOUT
        }),
    }
}

/// Parse a seconds value: `Some` only for a finite number in `0.001..=86400`.
pub fn parse_export_quiesce_timeout_checked(raw: &str) -> Option<std::time::Duration> {
    raw.trim()
        .parse::<f64>()
        .ok()
        .filter(|v| v.is_finite() && (0.001..=86_400.0).contains(v))
        .map(std::time::Duration::from_secs_f64)
}

/// Pure parser behind [`export_quiesce_timeout`] (default on anything unusable).
pub fn parse_export_quiesce_timeout(raw: Option<&str>) -> std::time::Duration {
    raw.and_then(parse_export_quiesce_timeout_checked)
        .unwrap_or(EXPORT_QUIESCE_TIMEOUT)
}

#[cfg(test)]
mod export_quiesce_timeout_tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn usable_values_are_parsed() {
        let secs = |s: &str| parse_export_quiesce_timeout(Some(s));
        assert_eq!(secs("5"), Duration::from_secs(5));
        assert_eq!(
            secs(" 5 "),
            Duration::from_secs(5),
            "surrounding whitespace"
        );
        assert_eq!(secs("+5"), Duration::from_secs(5));
        assert_eq!(secs("1e-3"), Duration::from_millis(1));
        assert_eq!(secs("0.001"), Duration::from_millis(1));
        assert_eq!(secs("2.5"), Duration::from_millis(2500));
        assert_eq!(secs("86400"), Duration::from_secs(86_400));
    }

    #[test]
    fn unusable_values_select_the_default() {
        for raw in [
            "", "0", "-1", "-0.5", "0.0009", "NaN", "nan", "inf", "-inf", "infinity", "86401",
            "1e9", "garbage", "5s", "1,5",
        ] {
            assert_eq!(
                parse_export_quiesce_timeout(Some(raw)),
                EXPORT_QUIESCE_TIMEOUT,
                "{raw:?}"
            );
            assert!(
                parse_export_quiesce_timeout_checked(raw).is_none(),
                "{raw:?}"
            );
        }
        assert_eq!(parse_export_quiesce_timeout(None), EXPORT_QUIESCE_TIMEOUT);
    }
}

/// Payload size below which the default `put_bytes_async` snapshot is copied with the GIL
/// held (256 KiB): releasing and re-taking the GIL costs more than copying that much.
pub const SNAPSHOT_GIL_DETACH_THRESHOLD: usize = 256 * 1024;

#[cfg(test)]
mod h2_window_tests {
    use super::*;

    #[test]
    fn test_h2_window_config_default() {
        let cfg = H2WindowConfig::default();
        assert!(cfg.adaptive, "default should be adaptive ON");
        assert_eq!(cfg.stream_window_mb, DEFAULT_H2_STREAM_WINDOW_MB);
        assert_eq!(cfg.conn_window_mb, DEFAULT_H2_CONN_WINDOW_MB);
        assert_eq!(
            cfg.conn_window_mb,
            cfg.stream_window_mb * 4,
            "default conn window should be 4x stream window"
        );
    }

    #[test]
    fn test_h2_window_config_stream_to_bytes() {
        let cfg = H2WindowConfig {
            adaptive: false,
            stream_window_mb: 4,
            conn_window_mb: 16,
        };
        assert_eq!(cfg.stream_window_bytes(), 4 * 1024 * 1024);
        assert_eq!(cfg.conn_window_bytes(), 16 * 1024 * 1024);
    }

    #[test]
    fn test_h2_window_config_hard_cap_enforced() {
        // 300 MiB exceeds H2_WINDOW_MB_HARD_CAP (256)
        let cfg = H2WindowConfig {
            adaptive: false,
            stream_window_mb: 300,
            conn_window_mb: 300,
        };
        // bytes() doesn't clamp to hard cap — the cap is enforced in from_env parsing.
        // Verify the hard cap constant itself is sane.
        const {
            assert!(
                H2_WINDOW_MB_HARD_CAP <= 2047,
                "hard cap must be below the HTTP/2 spec max of ~2 GiB"
            );
            assert!(H2_MAX_WINDOW_BYTES == 0x7FFF_FFFF);
        }
        // bytes() uses saturating_mul + min(H2_MAX_WINDOW_BYTES)
        assert!(cfg.stream_window_bytes() <= H2_MAX_WINDOW_BYTES);
    }

    #[test]
    fn test_h2_window_conn_defaults_to_4x_stream() {
        // When conn is unset, from_env() sets it to 4× stream.
        // We test the struct math directly.
        let stream = 8u32;
        let conn = stream.saturating_mul(4).min(H2_WINDOW_MB_HARD_CAP);
        assert_eq!(conn, 32);
    }

    #[test]
    fn test_h2_window_adaptive_default_true() {
        // Simulate "unset" — H2WindowConfig::default() should give adaptive=true.
        let cfg = H2WindowConfig::default();
        assert!(cfg.adaptive);
    }
}

#[cfg(test)]
mod connect_timeout_tests {
    use super::*;

    // These tests mutate process-global env, so each takes the crate-wide
    // guard `crate::test_env` (issue #216): a lock private to this module
    // cannot protect variables that tests in other modules also touch.

    fn with_var<F: FnOnce()>(value: Option<&str>, body: F) {
        let _guard = crate::test_env::lock();
        let _env = crate::test_env::EnvRestore::capture(&[ENV_CONNECT_TIMEOUT_SECS]);
        match value {
            Some(v) => std::env::set_var(ENV_CONNECT_TIMEOUT_SECS, v),
            None => std::env::remove_var(ENV_CONNECT_TIMEOUT_SECS),
        }
        body();
    }

    #[test]
    fn test_connect_timeout_falls_back_to_default_when_unset() {
        with_var(None, || {
            assert_eq!(connect_timeout_secs(), DEFAULT_CONNECT_TIMEOUT_SECS);
        });
    }

    #[test]
    fn test_connect_timeout_honors_env_override() {
        with_var(Some("30"), || {
            assert_eq!(connect_timeout_secs(), 30);
        });
    }

    #[test]
    fn test_connect_timeout_unparseable_falls_back_to_default() {
        with_var(Some("not-a-number"), || {
            assert_eq!(connect_timeout_secs(), DEFAULT_CONNECT_TIMEOUT_SECS);
        });
    }

    #[test]
    fn test_connect_timeout_empty_falls_back_to_default() {
        with_var(Some(""), || {
            assert_eq!(connect_timeout_secs(), DEFAULT_CONNECT_TIMEOUT_SECS);
        });
    }
}

#[cfg(test)]
mod size_cache_limit_tests {
    use super::*;

    #[test]
    fn test_size_cache_limit_default_when_unset() {
        assert_eq!(
            parse_size_cache_max_entries(None),
            DEFAULT_SIZE_CACHE_MAX_ENTRIES
        );
    }

    #[test]
    fn test_size_cache_limit_honors_override() {
        assert_eq!(parse_size_cache_max_entries(Some("1234")), 1234);
        assert_eq!(parse_size_cache_max_entries(Some(" 77 ")), 77);
    }

    #[test]
    fn test_size_cache_limit_garbage_falls_back_and_zero_clamps() {
        assert_eq!(
            parse_size_cache_max_entries(Some("lots")),
            DEFAULT_SIZE_CACHE_MAX_ENTRIES
        );
        assert_eq!(parse_size_cache_max_entries(Some("0")), 1);
    }

    #[test]
    fn test_size_cache_limit_is_read_once() {
        // Whatever the first reader saw is what every later call returns.
        let first = size_cache_max_entries();
        assert_eq!(size_cache_max_entries(), first);
        assert!(first >= 1);
    }
}

#[cfg(test)]
mod max_retry_tests {
    use super::*;

    fn with_var<F: FnOnce()>(value: Option<&str>, body: F) {
        let _guard = crate::test_env::lock();
        let _env = crate::test_env::EnvRestore::capture(&[ENV_MAX_RETRY_ATTEMPTS]);
        match value {
            Some(v) => std::env::set_var(ENV_MAX_RETRY_ATTEMPTS, v),
            None => std::env::remove_var(ENV_MAX_RETRY_ATTEMPTS),
        }
        body();
    }

    #[test]
    fn test_max_retry_default_when_unset() {
        with_var(None, || {
            assert_eq!(max_retry_attempts(), DEFAULT_MAX_RETRY_ATTEMPTS);
        });
    }

    #[test]
    fn test_max_retry_honors_override() {
        with_var(Some("5"), || {
            assert_eq!(max_retry_attempts(), 5);
        });
    }

    #[test]
    fn test_max_retry_one_is_valid_fast_fail() {
        // Operators may set 1 to surface dispatch failures in one connect
        // budget instead of three.
        with_var(Some("1"), || {
            assert_eq!(max_retry_attempts(), 1);
        });
    }

    #[test]
    fn test_max_retry_zero_clamped_to_one() {
        // SDK rejects max_attempts=0; the helper clamps to 1.
        with_var(Some("0"), || {
            assert_eq!(max_retry_attempts(), 1);
        });
    }

    #[test]
    fn test_max_retry_unparseable_falls_back_to_default() {
        with_var(Some("not-a-number"), || {
            assert_eq!(max_retry_attempts(), DEFAULT_MAX_RETRY_ATTEMPTS);
        });
    }
}

#[cfg(test)]
mod mpi_thread_budget_tests {
    use super::*;

    // mpi_world_size() checks 3 env vars; these tests mutate all of them and
    // must run serially both against each other and against every other test
    // in this binary that touches the same vars (effective_thread_budget_tests
    // and s3_client's mpi_aware_runtime_sizing_tests do).  That is why they take
    // the crate-wide guard `crate::test_env`, not a lock of their own (#216).
    const VARS: [&str; 3] = ["OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "WORLD_SIZE"];

    fn with_clean_env<F: FnOnce()>(body: F) {
        let _guard = crate::test_env::lock();
        let _env = crate::test_env::EnvRestore::capture(&VARS);
        for v in VARS {
            std::env::remove_var(v);
        }
        body();
    }

    #[test]
    fn test_mpi_world_size_defaults_to_one_when_unset() {
        with_clean_env(|| {
            assert_eq!(mpi_world_size(), 1);
        });
    }

    #[test]
    fn test_mpi_world_size_honors_ompi_var() {
        with_clean_env(|| {
            std::env::set_var("OMPI_COMM_WORLD_SIZE", "4");
            assert_eq!(mpi_world_size(), 4);
        });
    }

    #[test]
    fn test_mpi_world_size_honors_pmi_var() {
        with_clean_env(|| {
            std::env::set_var("PMI_SIZE", "8");
            assert_eq!(mpi_world_size(), 8);
        });
    }

    #[test]
    fn test_mpi_world_size_honors_torchrun_var() {
        with_clean_env(|| {
            std::env::set_var("WORLD_SIZE", "2");
            assert_eq!(mpi_world_size(), 2);
        });
    }

    #[test]
    fn test_mpi_world_size_ompi_wins_over_others() {
        with_clean_env(|| {
            std::env::set_var("OMPI_COMM_WORLD_SIZE", "4");
            std::env::set_var("PMI_SIZE", "16");
            std::env::set_var("WORLD_SIZE", "32");
            assert_eq!(mpi_world_size(), 4);
        });
    }

    #[test]
    fn test_mpi_world_size_zero_clamped_to_one() {
        // A malformed/zero world size must never divide a thread budget by 0.
        with_clean_env(|| {
            std::env::set_var("OMPI_COMM_WORLD_SIZE", "0");
            assert_eq!(mpi_world_size(), 1);
        });
    }

    #[test]
    fn test_mpi_world_size_unparseable_falls_back_to_one() {
        with_clean_env(|| {
            std::env::set_var("OMPI_COMM_WORLD_SIZE", "not-a-number");
            assert_eq!(mpi_world_size(), 1);
        });
    }

    #[test]
    fn test_mpi_aware_thread_budget_matches_manual_formula() {
        with_clean_env(|| {
            std::env::set_var("OMPI_COMM_WORLD_SIZE", "4");
            let cpus = std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4);
            assert_eq!(mpi_aware_thread_budget(), (cpus / 4).max(1));
        });
    }

    #[test]
    fn test_mpi_aware_thread_budget_single_process_equals_cpu_count() {
        with_clean_env(|| {
            let cpus = std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4);
            assert_eq!(mpi_aware_thread_budget(), cpus);
        });
    }

    #[test]
    fn test_mpi_aware_thread_budget_never_zero_even_with_huge_world_size() {
        with_clean_env(|| {
            std::env::set_var("OMPI_COMM_WORLD_SIZE", "999999");
            assert_eq!(mpi_aware_thread_budget(), 1);
        });
    }
}

#[cfg(test)]
mod effective_thread_budget_tests {
    use super::*;
    use std::sync::atomic::Ordering;

    const VARS: [&str; 5] = [
        "S3DLIO_RT_THREADS",
        "S3DLIO_RT_THREADS_UNSAFE",
        "OMPI_COMM_WORLD_SIZE",
        "PMI_SIZE",
        "WORLD_SIZE",
    ];

    fn with_clean_env<F: FnOnce()>(body: F) {
        let _guard = crate::test_env::lock();
        let _env = crate::test_env::EnvRestore::capture(&VARS);
        let _limit = crate::test_env::AtomicRestore::capture(&crate::s3_client::RT_THREADS_LIMIT);
        for v in VARS {
            std::env::remove_var(v);
        }
        // effective_thread_budget delegates to clamped_env_rt_threads, which
        // reads RT_THREADS_LIMIT.  Reset it so a prior test in the same
        // binary doesn't leak a nonzero limit into these assertions.
        crate::s3_client::RT_THREADS_LIMIT.store(0, Ordering::Relaxed);

        body();
    }

    /// #216: a failed assertion inside the body must not leak the variables or
    /// RT_THREADS_LIMIT into later tests (the tokio guard never poisons).
    #[test]
    fn with_clean_env_restores_when_the_body_panics() {
        let snapshot = || {
            let _guard = crate::test_env::lock();
            let vars: Vec<Option<String>> = VARS.iter().map(|v| std::env::var(v).ok()).collect();
            (
                vars,
                crate::s3_client::RT_THREADS_LIMIT.load(Ordering::Relaxed),
            )
        };
        let before = snapshot();

        let outcome = std::panic::catch_unwind(|| {
            with_clean_env(|| {
                std::env::set_var("S3DLIO_RT_THREADS", "2");
                std::env::set_var("OMPI_COMM_WORLD_SIZE", "4");
                crate::s3_client::RT_THREADS_LIMIT.store(28, Ordering::Relaxed);
                panic!("simulated failed assertion");
            });
        });
        assert!(outcome.is_err(), "the body was expected to panic");

        assert_eq!(snapshot(), before, "state leaked out of a panicking body");
    }

    #[test]
    fn test_effective_budget_single_process_matches_max_4_cores_floor() {
        with_clean_env(|| {
            let cpus = std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4);
            assert_eq!(effective_thread_budget(), std::cmp::max(4, cpus));
        });
    }

    #[test]
    fn test_effective_budget_mpi_aware_when_world_size_set() {
        with_clean_env(|| {
            std::env::set_var("OMPI_COMM_WORLD_SIZE", "4");
            let cpus = std::thread::available_parallelism()
                .map(|p| p.get())
                .unwrap_or(4);
            assert_eq!(effective_thread_budget(), (cpus / 4).max(1));
        });
    }

    #[test]
    fn test_effective_budget_explicit_env_var_wins_over_everything() {
        with_clean_env(|| {
            std::env::set_var("S3DLIO_RT_THREADS", "2");
            std::env::set_var("OMPI_COMM_WORLD_SIZE", "4");
            assert_eq!(effective_thread_budget(), 2);
        });
    }

    /// RED-then-GREEN regression for the DLIO unet3d datagen bottleneck:
    /// with `RT_THREADS_LIMIT=28` (auto-set at import) and
    /// `S3DLIO_RT_THREADS=1` (miscomputed by DLIO's
    /// `write_threads * 1.5` at write_threads=1),
    /// `effective_thread_budget()` must clamp the env-var value up to the
    /// configured target rather than starve every downstream pool it
    /// sizes (including the checkpoint runtime).
    #[test]
    fn test_effective_budget_clamps_grossly_underprovisioned_env_var() {
        with_clean_env(|| {
            std::env::set_var("S3DLIO_RT_THREADS", "1");
            crate::s3_client::RT_THREADS_LIMIT.store(28, Ordering::Relaxed);
            assert_eq!(effective_thread_budget(), 28);
        });
    }

    #[test]
    fn test_effective_budget_unsafe_bypass_respects_low_env_var() {
        with_clean_env(|| {
            std::env::set_var("S3DLIO_RT_THREADS", "1");
            std::env::set_var("S3DLIO_RT_THREADS_UNSAFE", "1");
            crate::s3_client::RT_THREADS_LIMIT.store(28, Ordering::Relaxed);
            assert_eq!(effective_thread_budget(), 1);
        });
    }
}
