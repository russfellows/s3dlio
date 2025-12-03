// src/constants.rs
//
// Centralized constants for s3dlio to avoid hardcoded values throughout the codebase

use std::sync::Arc;
use once_cell::sync::Lazy;
use rand::RngCore;

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

/// Default timeout for storage operations (seconds)
pub const DEFAULT_OPERATION_TIMEOUT_SECS: u64 = 300; // 5 minutes

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

/// Universal default minimum object size to trigger RangeEngine (16 MiB)
/// 
/// This threshold applies to all storage backends (S3, Azure, GCS, file://, direct://)
/// when RangeEngine is explicitly enabled. **RangeEngine is disabled by default as of v0.9.6.**
///
/// **Why 16 MiB?**
/// - Avoids range splitting for typical objects (configs, logs, small files)
/// - Only engages RangeEngine for moderately large files (media, datasets, archives)
/// - When enabled, balances performance vs overhead for large-file workloads
/// 
/// **Performance Impact (when RangeEngine is enabled):**
/// - Objects < 16 MiB: Single GET request (fast path, no range splitting)
/// - Objects >= 16 MiB: HEAD + concurrent range GETs (RangeEngine path)
/// 
/// **When to Override:**
/// - Set higher (e.g., 64 MiB) if most objects are large but still want to avoid HEAD overhead
/// - Set lower (e.g., 4 MiB) for dedicated large-file workloads on high-latency networks
/// - Leave RangeEngine disabled (default) for mixed or small-object workloads
///
/// **Example Configuration:**
/// ```rust
/// use s3dlio::object_store::{GcsConfig, RangeEngineConfig};
/// 
/// // Enable RangeEngine for large-file workload
/// let config = GcsConfig {
///     enable_range_engine: true,  // Must explicitly enable (disabled by default)
///     range_engine: RangeEngineConfig {
///         min_split_size: 16 * 1024 * 1024,  // 16 MiB threshold (default)
///         ..Default::default()
///     },
/// };
/// ```
pub const DEFAULT_RANGE_ENGINE_THRESHOLD: u64 = 16 * 1024 * 1024;

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

/// Block size for data generation (512 bytes)
/// This is the fundamental unit for deduplication and compression calculations
pub const BLK_SIZE: usize = 512;

/// Half block size for internal calculations
pub const HALF_BLK: usize = BLK_SIZE / 2;

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
// Custom Endpoint Environment Variables
// =============================================================================

/// Primary environment variable for custom Azure Blob Storage endpoint
/// Example: AZURE_STORAGE_ENDPOINT=http://localhost:10000
pub const ENV_AZURE_STORAGE_ENDPOINT: &str = "AZURE_STORAGE_ENDPOINT";

/// Alternative environment variable for custom Azure Blob Storage endpoint
/// Example: AZURE_BLOB_ENDPOINT_URL=http://localhost:10000
pub const ENV_AZURE_BLOB_ENDPOINT_URL: &str = "AZURE_BLOB_ENDPOINT_URL";

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
