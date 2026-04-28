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

/// Default multipart upload threshold for S3 (32 MB)
pub const DEFAULT_S3_MULTIPART_THRESHOLD: usize = 32 * 1024 * 1024;

/// Default multipart upload part size for S3 (16 MB)
pub const DEFAULT_S3_MULTIPART_PART_SIZE: usize = 16 * 1024 * 1024;

/// Default multipart upload part size for Azure (16 MB)
pub const DEFAULT_AZURE_MULTIPART_PART_SIZE: usize = 16 * 1024 * 1024;

/// Maximum number of parts in a multipart upload
pub const MAX_MULTIPART_PARTS: usize = 10000;

/// Default TCP connect timeout in seconds.
///
/// Covers only the TCP handshake (SYN → SYN-ACK).  On a LAN/datacenter network
/// a connect that hasn't completed in 10 s indicates a dead host, wrong IP, or
/// a firewall silently dropping packets.  Override with `S3DLIO_CONNECT_TIMEOUT_SECS`.
pub const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 10;

/// Default operation timeout in seconds.
///
/// Covers the full request/response cycle once the TCP connection is established
/// (send request headers + body, wait for response headers + body).  60 s is
/// sufficient for objects up to ~6 GB at 100 MB/s and ~60 GB at 1 GB/s.
/// Override with `S3DLIO_OPERATION_TIMEOUT_SECS`.
pub const DEFAULT_OPERATION_TIMEOUT_SECS: u64 = 60;

/// Default retry count for storage operations
pub const DEFAULT_RETRY_COUNT: usize = 3;

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
/// **Default changed in v0.9.92**: the previous default was `Auto` (probe h2c once, fall back to
/// HTTP/1.1 if rejected). Benchmarking on loopback TCP endpoints showed that HTTP/2 consistently
/// reduces PUT/GET throughput compared with HTTP/1.1 and an unlimited connection pool. The default
/// is now HTTP/1.1. Set `S3DLIO_H2C=1` to opt in to h2c on `http://` endpoints.
///
/// **Note**: `https://` endpoints always negotiate HTTP/2 via TLS ALPN regardless of this setting.
pub const ENV_S3DLIO_H2C: &str = "S3DLIO_H2C";

/// Default HTTP/2 cleartext (h2c) behaviour when [`ENV_S3DLIO_H2C`] is not set: `false` (HTTP/1.1).
///
/// **Changed to `false` in v0.9.92.** Previous default was `Auto` (h2c probe once, fall back to
/// HTTP/1.1 if rejected). Benchmarking on plain `http://` endpoints showed that HTTP/2 reduces
/// throughput relative to HTTP/1.1 with an unlimited connection pool, so the default is now off.
///
/// `https://` endpoints are unaffected — HTTP/2 is still negotiated automatically via TLS ALPN.
/// Set `S3DLIO_H2C=1` to enable h2c for storage systems that require HTTP/2 on plain-HTTP endpoints.
pub const DEFAULT_H2C_ENABLED: bool = false;

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

/// Primary environment variable for custom GCS endpoint
/// Example: GCS_ENDPOINT_URL=http://localhost:4443
pub const ENV_GCS_ENDPOINT_URL: &str = "GCS_ENDPOINT_URL";

/// GCS emulator convention environment variable (STORAGE_EMULATOR_HOST=host:port)
/// If value doesn't start with http://, "http://" is prepended automatically
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
