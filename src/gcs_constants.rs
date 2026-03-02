// src/gcs_constants.rs
//
// ── Single source of truth for all GCS / gRPC tuning parameters in s3dlio ──
//
// Vendor-layer constants (HTTP/2 window size, gRPC write chunk size) are
// defined in:
//   vendor/google-cloud-gax-internal/src/gcs_constants.rs
// and re-exported here so application code has exactly ONE import for
// everything GCS-related.
//
// To change any default:
//   • Numeric defaults  → edit vendor/google-cloud-gax-internal/src/gcs_constants.rs
//   • Application defaults (channels, deletes) → edit the constants below
//   • Environment variable names → update BOTH files and keep them in sync

// ── Re-exports from the vendor layer ─────────────────────────────────────────

/// Re-export all vendor-layer constants so callers only need
/// `use crate::gcs_constants::*;` (or `use s3dlio::gcs_constants::*;`).
pub use google_cloud_gax_internal::gcs_constants::{
    GCS_SERVER_MAX_MESSAGE_SIZE,
    DEFAULT_GRPC_WRITE_CHUNK_SIZE,
    MAX_GRPC_WRITE_CHUNK_SIZE,
    DEFAULT_WINDOW_MIB,
    ENV_GRPC_INITIAL_WINDOW_MIB,
    ENV_GRPC_WRITE_CHUNK_SIZE,
};

// ── Application-layer constants ───────────────────────────────────────────────

/// Minimum (floor) gRPC subchannel count when auto-detecting from CPU count.
///
/// Each subchannel is an independent HTTP/2 TCP connection to GCS.  Fewer
/// channels than concurrent jobs forces multiple streams to share one
/// connection's flow-control window, collapsing effective per-stream bandwidth.
///
/// Empirical data (c4-standard-8, RAPID bucket, 1 000 × ~30 MiB objects):
///   •  8 subchannels, 64 jobs →  ~845 MiB/s  (8 streams / conn)
///   • 32 subchannels, 64 jobs → ~3.10 GiB/s  (2 streams / conn)
///   • 64 subchannels, 64 jobs → ~3.62 GiB/s  (1 stream / conn) ← optimal
///
/// 64 is a safe floor for workloads with up to 64 concurrent jobs on any VM.
pub const GCS_MIN_CHANNELS: usize = 64;

/// Maximum number of concurrent GCS object DELETE operations.
///
/// Used as the permit count for the semaphore that throttles bulk deletes.
pub const GCS_MAX_CONCURRENT_DELETES: usize = 64;

/// Environment variable to force the gRPC subchannel count.
///
/// Overrides both `set_gcs_channel_count()` and the CPU-based auto-detect.
/// Takes effect only before the first GCS client is initialized.
pub const ENV_GCS_GRPC_CHANNELS: &str = "S3DLIO_GCS_GRPC_CHANNELS";

/// Environment variable that controls per-bucket RAPID / appendable-write mode.
///
/// | Value                    | Effect                              |
/// |--------------------------|-------------------------------------|
/// | `auto` (default)         | detect per-bucket via Storage Layout API |
/// | `true` / `1` / `yes`    | force RAPID on for every bucket     |
/// | `false` / `0` / `no`    | disable RAPID for every bucket      |
pub const ENV_GCS_RAPID: &str = "S3DLIO_GCS_RAPID";
