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

/// Page size bounds for validation
pub const MIN_PAGE_SIZE: usize = 512;
pub const MAX_PAGE_SIZE: usize = 64 * 1024;

/// URI scheme constants
pub const SCHEME_FILE: &str = "file://";
pub const SCHEME_DIRECT: &str = "direct://";
pub const SCHEME_S3: &str = "s3://";
pub const SCHEME_AZURE: &str = "az://";

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

/// Storage backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageBackend {
    /// Local filesystem (buffered or direct)
    File(IoMode),
    /// Amazon S3
    S3,
    /// Azure Blob Storage
    Azure,
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

impl StorageBackend {
    /// Returns the URI scheme for this backend
    pub fn uri_scheme(&self) -> &'static str {
        match self {
            StorageBackend::File(mode) => mode.uri_scheme(),
            StorageBackend::S3 => SCHEME_S3,
            StorageBackend::Azure => SCHEME_AZURE,
        }
    }
}
