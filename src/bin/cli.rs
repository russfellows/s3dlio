// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! CLI supporting `list`, `get`, `delete`, `put`, and related commands.
//!
//! Examples:
//! ```bash
//! s3Rust-cli list        s3://bucket/prefix/
//! s3Rust-cli stat        s3://bucket/prefix/
//! s3Rust-cli get         s3://bucket/key.npz           # single
//! s3Rust-cli get         s3://bucket/prefix/ -j 128    # many
//! s3Rust-cli delete      s3://bucket/prefix/           # delete all under prefix
//! s3Rust-cli put         s3://bucket/key               # put one or more object
//! s3Rust-cli upload      local-file s3://bucket/key    # upload one or more objects
//! s3Rust-cli download    s3://bucket/key local-file    # download  one or more object
//!
//! # Multi-backend support (S3, Azure, GCS, file://, direct://)
//! s3dlio upload files/*.txt s3://bucket/data/           # S3
//! s3dlio upload files/*.txt az://account/container/     # Azure
//! s3dlio upload files/*.txt gs://bucket/data/           # Google Cloud Storage
//! s3dlio upload files/*.txt file:///local/path/         # Local filesystem
//! s3dlio upload files/*.txt direct:///fast/storage/     # DirectIO
//! ```

use anyhow::{bail, Context, Result};
use clap::{ArgAction, Parser, Subcommand};
use futures_util::stream::{FuturesUnordered, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use std::io::{self, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};
use tracing_subscriber::EnvFilter;

// Import shared functions from the crate.
use s3dlio::{
    config::{Config, DataGenMode},
    constants::MAX_JOBS,
    data_gen::generate_object,
    list_containers,
    multi_endpoint::{LoadBalanceStrategy, MultiEndpointStore},
    object_store::{store_for_uri_with_logger, ObjectStore},
    parse_s3_uri, ContainerInfo, ObjectType, DEFAULT_OBJECT_SIZE,
};

use s3dlio::progress::{ProgressCallback, S3ProgressTracker};
use s3dlio::{finalize_op_logger, global_logger, init_op_logger};
use s3dlio::{generic_download_objects_with_summary, generic_upload_files_with_summary};

/// Returns per-crate log-level caps that prevent deadlocks.
///
/// h2, hyper, tonic, and the AWS SDK crates deadlock when debug/trace
/// logging is active inside `tokio::spawn` blocks.  These directives are
/// applied on top of whatever `RUST_LOG` says, so `RUST_LOG=debug` still
/// works for *our* code while keeping the problematic crates at `warn`.
fn crate_log_caps() -> Vec<tracing_subscriber::filter::Directive> {
    [
        "h2=warn",
        "hyper=warn",
        "hyper_util=warn",
        "tonic=warn",
        "tower=warn",
        "reqwest=warn",
        "aws_config=warn",
        "aws_runtime=warn",
        "aws_sdk_s3=warn",
        "aws_smithy_runtime=warn",
        "aws_smithy_runtime_api=warn",
        "aws_smithy_http=warn",
        "aws_credential_types=warn",
        "aws_sigv4=warn",
        "tracing=warn",
    ]
    .iter()
    .filter_map(|s| s.parse().ok())
    .collect()
}

/// Macro to safely print with broken pipe handling
macro_rules! safe_println {
    ($($arg:tt)*) => {
        match writeln!(io::stdout(), $($arg)*) {
            Ok(_) => {},
            Err(e) if e.kind() == ErrorKind::BrokenPipe => {
                // Gracefully exit on broken pipe (e.g., when piped to head/tail)
                std::process::exit(0);
            }
            Err(e) => return Err(e.into())
        }
    };
}

// -- Commands

#[derive(Parser)]
#[command(author, version, about)]
struct Cli {
    /// Turn on verbose (info‑level) logging New, counts the number of v's
    #[arg(short = 'v',
        long,
        action = ArgAction::Count,
        help = "Increase log verbosity: -v = Info, -vv = Debug",
    )]
    verbose: u8,

    /// Write warp‑replay compatible op‑log (.tsv.zst). Disabled if not provided.
    #[arg(long = "op-log", value_name = "FILE")]
    op_log: Option<PathBuf>,

    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List all top-level containers (buckets, containers, or directories)
    /// for the specified storage backend.
    ///
    /// Pass a URI or scheme to select the backend:
    ///   s3://               AWS S3 or S3-compatible (default, uses AWS_ENDPOINT_URL)
    ///   gs://               Google Cloud Storage (needs GOOGLE_CLOUD_PROJECT)
    ///   az://accountname    Azure Blob Storage containers for that account
    ///   file:///path        Local filesystem directories under /path
    ListBuckets {
        /// Backend URI or scheme.  Examples:
        ///   s3://  gs://  gcs://  az://mystorageaccount  file:///mnt/data
        #[arg(default_value = "s3://")]
        uri: String,
    },

    /// Create a bucket/container/directory for any supported backend (s3://, gs://, az://, file://, direct://).
    #[clap(name = "create-bucket")]
    CreateBucket {
        /// Storage URI of the bucket/container to create
        /// (e.g. s3://my-bucket, gs://my-bucket, az://account/container, file:///mnt/data)
        uri: String,
    },

    /// Delete a bucket/container/directory for any supported backend. The bucket must be empty.
    #[clap(name = "delete-bucket")]
    DeleteBucket {
        /// Storage URI of the bucket/container to delete
        /// (e.g. s3://my-bucket, gs://my-bucket, az://account/container, file:///mnt/data)
        uri: String,
    },

    /// Stat object, show size & last modify date of a single object
    Stat {
        /// Full S3 URI (e.g. s3://bucket/prefix/key)
        uri: String,
    },
    /// Delete one object or every object that matches the prefix.
    #[clap(visible_alias = "rm")]
    Delete {
        /// S3 URI (single key or prefix ending with `/`).
        uri: String,

        /// Batch size (number of keys per DeleteObjects API call).
        /// The S3 API accepts up to 1,000 keys per batch request.
        #[arg(short = 'j', long = "jobs", default_value_t = 1000, value_parser = parse_jobs)]
        jobs: usize,

        /// Perform the operation recursively.
        #[clap(short, long)]
        recursive: bool,

        /// Optional regex pattern to filter objects to delete (applied client-side, works with all backends).
        #[clap(short, long)]
        pattern: Option<String>,
    },
    /// Download one or many objects concurrently.
    Get {
        /// S3 URI – can be a full key or a prefix ending with `/`.
        uri: Option<String>,

        /// Maximum concurrent GET requests (default 64, max 4096).
        /// Accepts human suffixes: k=1,000  m=1,000,000  (e.g. 64, 128, 256).
        #[arg(short = 'j', long = "jobs", default_value_t = 64, value_parser = parse_jobs)]
        jobs: usize,

        /// Alternative: maximum concurrent GET requests (for mp compatibility).
        /// Accepts human suffixes: k=1,000  m=1,000,000 (max 4096).
        #[arg(long = "concurrent", value_parser = parse_jobs)]
        concurrent: Option<usize>,

        /// File containing list of S3 URIs to download (one per line).
        #[arg(long = "keylist")]
        keylist: Option<PathBuf>,

        /// Perform the operation recursively.
        #[clap(short, long)]
        recursive: bool,

        /// Byte offset for range request (optional, for single-object GET only)
        #[arg(long = "offset")]
        offset: Option<u64>,

        /// Number of bytes to read for range request (optional, if not specified reads to end)
        #[arg(long = "length")]
        length: Option<u64>,

        /// Comma-separated S3 endpoints (host:port) for multi-endpoint GET.
        /// Round-robin load balancing across all endpoints. Requires an s3:// URI.
        /// Example: --endpoints=10.9.0.17:9000,10.9.0.18:9000
        #[arg(long = "endpoints", value_name = "HOST:PORT[,HOST:PORT...]")]
        endpoints: Option<String>,
    },
    /// Upload one or more objects concurrently, uses ObjectType format filled with random data.
    Put {
        /// S3 URI prefix (e.g. s3://bucket/prefix)
        uri_prefix: String,

        /// Optionally create the bucket if it does not exist.
        #[arg(short = 'c', long = "create-bucket", action)]
        create_bucket_flag: bool,

        /// Deduplication factor (integer ≥1). 1 => fully unique.
        #[arg(short = 'd', long = "dedup", default_value_t = 1)]
        dedup_f: usize,

        /// Maximum concurrent uploads (jobs), but is modified to be min(jobs, num).
        /// Accepts human suffixes: k=1,000 (e.g. 32, 64, 256, max is: 4096).
        #[arg(short = 'j', long = "jobs", default_value_t = 32, value_parser = parse_jobs)]
        jobs: usize,

        /// Number of objects to create and upload.
        /// Accepts human suffixes: 1k=1,000  1m=1,000,000  1g=1,000,000,000
        /// and binary suffixes: 1ki=1,024  1mi=1,048,576  1gi=1,073,741,824.
        /// Examples: 100000000  100m  1g
        #[arg(short = 'n', long = "num", default_value_t = 1, value_parser = parse_human_count)]
        num: usize,

        /// Specify Type of object to generate:
        #[arg( short = 'o', long = "object-type", value_enum, ignore_case = true, default_value_t = ObjectType::Raw)]
        // Without value_parser [] values are case insensitive
        object_type: ObjectType,

        /// Object size in bytes or human units (examples: 8388608, 8MB, 8MiB, 8m, 8M, 8g, 8GiB).
        #[arg(short = 's', long = "size", default_value_t = DEFAULT_OBJECT_SIZE, value_parser = parse_human_size)]
        size: usize,

        /// Template for names. Use '{}' for replacement, first '{}' is object number, 2nd is total count.
        #[arg(short = 't', long = "template", default_value = "object_{}_of_{}.dat")]
        template: String,

        /// Compression factor (integer ≥1). 1 => fully random.
        #[arg(short = 'x', long = "compress", default_value_t = 1)]
        compress_f: usize,

        /// Data generation mode for performance optimization
        #[arg(long = "data-gen-mode", value_enum, ignore_case = true, default_value_t = s3dlio::config::DataGenMode::Streaming)]
        data_gen_mode: s3dlio::config::DataGenMode,

        /// Chunk size for streaming generation mode (bytes)
        #[arg(long = "chunk-size", default_value_t = 256 * 1024)]
        chunk_size: usize,

        /// Comma-separated S3 endpoints (host:port) for multi-endpoint PUT.
        /// Round-robin load balancing across all endpoints. Requires an s3:// URI.
        /// Example: --endpoints=10.9.0.17:9000,10.9.0.18:9000
        #[arg(long = "endpoints", value_name = "HOST:PORT[,HOST:PORT...]")]
        endpoints: Option<String>,
    },
    /// Upload local files to any storage backend (S3, Azure, GCS, file://, direct://), supports glob and regex patterns
    Upload {
        /// One or more local files, directories, glob patterns ('*','?'), or regex patterns
        #[arg(required = true)]
        files: Vec<PathBuf>,
        /// Destination URI (s3://bucket/, az://container/, gs://bucket/, file:///path/, direct:///path/) **ending with '/'**
        dest: String,
        /// Maximum parallel uploads (default 32, max 4096).
        /// Accepts human suffixes: k=1,000  m=1,000,000 (e.g. 32, 64, 256).
        #[arg(short = 'j', long = "jobs", default_value_t = 32, value_parser = parse_jobs)]
        jobs: usize,
        /// Create the bucket if it doesn’t exist
        #[arg(short = 'c', long = "create-bucket")]
        create_bucket: bool,
    },
    /// Download objects from any storage backend (S3, Azure, GCS, file://, direct://), supports glob and regex patterns
    Download {
        /// Source URI with optional patterns – supports full URIs, prefixes, globs ('*','?'), and regex patterns
        src: String,
        /// Local directory to write into
        dest_dir: PathBuf,
        /// Maximum parallel downloads (default 64, max 4096).
        /// Accepts human suffixes: k=1,000  m=1,000,000 (e.g. 64, 128, 256).
        #[arg(short = 'j', long = "jobs", default_value_t = 64, value_parser = parse_jobs)]
        jobs: usize,
        /// Download recursively
        #[clap(short, long)]
        recursive: bool,
    },

    /// Copy between local paths and storage URIs.
    ///
    /// Direction is inferred automatically:
    /// - local -> URI  : upload
    /// - URI   -> local: download
    Cp {
        /// Source local path or storage URI
        src: String,
        /// Destination local path or storage URI
        dest: String,
        /// Maximum parallel operations (default 32, max 4096).
        /// Accepts human suffixes: k=1,000  m=1,000,000 (e.g. 32, 64, 256).
        #[arg(short = 'j', long = "jobs", default_value_t = 32, value_parser = parse_jobs)]
        jobs: usize,
        /// Recursive mode for URI-prefix downloads
        #[clap(short, long)]
        recursive: bool,
        /// Create destination bucket/container when uploading (if supported)
        #[arg(short = 'c', long = "create-bucket")]
        create_bucket: bool,
    },

    /// List objects at a storage URI (s3://, az://, gs://, file://, direct://)
    #[clap(name = "list", visible_alias = "ls")]
    GenericList {
        /// Storage URI (e.g. s3://bucket/prefix/, az://account/container/, gs://bucket/prefix/, file:///path/)
        uri: String,

        /// List objects recursively
        #[clap(short, long)]
        recursive: bool,

        /// Optional regex pattern to filter results (applied client-side, works with all backends)
        /// Example: '.*\.txt$' to match only .txt files
        #[clap(short, long)]
        pattern: Option<String>,

        /// Count objects only (don't print URIs) - faster for large result sets
        #[clap(short, long)]
        count_only: bool,
    },

    /// Generate NVIDIA DALI-compatible index file for a TFRecord file
    #[clap(name = "tfrecord-index")]
    TfrecordIndex {
        /// Path to TFRecord file (input)
        tfrecord_path: PathBuf,

        /// Path to index file (output, typically .idx extension)
        #[arg(default_value = None)]
        index_path: Option<PathBuf>,
    },
}

// -----------------------------------------------------------------------------
// Command implementations
// -----------------------------------------------------------------------------

/// List top-level containers for the given backend URI (bucket/container/directory).
async fn list_buckets_cmd(uri: &str) -> Result<()> {
    safe_println!("Listing containers for '{}'...", uri);

    let containers: Vec<ContainerInfo> = list_containers(uri)?;

    if containers.is_empty() {
        safe_println!("No containers found.");
        return Ok(());
    }

    safe_println!("\nFound {} container(s):", containers.len());
    safe_println!("{:<40} {:<30} {}", "Name", "URI", "Creation Date");
    safe_println!("{}", "-".repeat(90));

    for c in containers {
        let date = c.creation_date.as_deref().unwrap_or("-");
        safe_println!("{:<40} {:<30} {}", c.name, c.uri, date);
    }

    Ok(())
}

/// Check if AWS credentials are available for S3 operations
fn check_aws_credentials() -> Result<()> {
    if std::env::var("AWS_ACCESS_KEY_ID").is_err()
        || std::env::var("AWS_SECRET_ACCESS_KEY").is_err()
    {
        bail!("Missing required AWS environment variables. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (and optionally AWS_REGION) either in your environment or in a .env file.");
    }
    Ok(())
}

/// Check if credentials are needed based on URI scheme
fn requires_aws_credentials(uri: &str) -> bool {
    // Only S3 URIs require AWS credentials
    // Other schemes (gs://, az://, file://, direct://) use their own auth methods
    uri.starts_with("s3://")
}

/// Extract the container/bucket/path component expected by each backend's
/// `create_container` / `delete_container` from a full storage URI.
///
/// Scheme → extracted name:
///   s3://bucket[/prefix]              → "bucket"
///   gs://bucket[/prefix]              → "bucket"
///   gcs://bucket[/prefix]             → "bucket"
///   az://account/container[/prefix]   → "account/container"
///   azure://account/container[/…]     → "account/container"
///   file:///path[/…]                  → "/path[/…]"
///   direct:///path[/…]                → "/path[/…]"
fn extract_container_name(uri: &str) -> Result<String> {
    if let Some(rest) = uri.strip_prefix("s3://") {
        let bucket = rest.split('/').next().unwrap_or("").trim_end_matches('/');
        if bucket.is_empty() {
            bail!("No bucket name found in S3 URI: {}", uri);
        }
        Ok(bucket.to_string())
    } else if let Some(rest) = uri
        .strip_prefix("gs://")
        .or_else(|| uri.strip_prefix("gcs://"))
    {
        let bucket = rest.split('/').next().unwrap_or("").trim_end_matches('/');
        if bucket.is_empty() {
            bail!("No bucket name found in GCS URI: {}", uri);
        }
        Ok(bucket.to_string())
    } else if let Some(rest) = uri
        .strip_prefix("az://")
        .or_else(|| uri.strip_prefix("azure://"))
    {
        // Azure expects "account/container"
        let parts: Vec<&str> = rest.splitn(3, '/').collect();
        let account = parts.first().copied().unwrap_or("");
        let container = parts.get(1).copied().unwrap_or("").trim_end_matches('/');
        if account.is_empty() || container.is_empty() {
            bail!("Azure URI must be az://account/container; got: {}", uri);
        }
        Ok(format!("{}/{}", account, container))
    } else if let Some(rest) = uri.strip_prefix("file://") {
        let path = rest.trim_end_matches('/');
        if path.is_empty() {
            bail!("No path found in file URI: {}", uri);
        }
        Ok(path.to_string())
    } else if let Some(rest) = uri.strip_prefix("direct://") {
        let path = rest.trim_end_matches('/');
        if path.is_empty() {
            bail!("No path found in direct URI: {}", uri);
        }
        Ok(path.to_string())
    } else {
        bail!("Unsupported URI scheme for create/delete-bucket: {}", uri);
    }
}

fn is_storage_uri(path_or_uri: &str) -> bool {
    path_or_uri.starts_with("s3://")
        || path_or_uri.starts_with("gs://")
        || path_or_uri.starts_with("gcs://")
        || path_or_uri.starts_with("az://")
        || path_or_uri.starts_with("azure://")
        || path_or_uri.starts_with("file://")
        || path_or_uri.starts_with("direct://")
}

/// Strip visual thousands-separator characters from a human-readable number string.
///
/// Always strips `_` (universal). Also strips the locale thousands-separator
/// character when it is a single non-alphanumeric character (e.g. `,` in
/// en-US, `.` in de-DE).  Only the *thousands* separator is stripped — the
/// *decimal-point* character is never touched, so `8.5m` still fails cleanly
/// in en-US locales.
///
/// Applied to the raw input before digit scanning so that `100_000_000`,
/// `100,000,000`, and `100.000.000` all parse correctly.
fn strip_separators(s: &str) -> String {
    let locale_sep: Option<char> = {
        #[cfg(unix)]
        unsafe {
            let lc = libc::localeconv();
            if !lc.is_null() {
                let sep_ptr = (*lc).thousands_sep;
                if !sep_ptr.is_null() {
                    if let Ok(sep_str) = std::ffi::CStr::from_ptr(sep_ptr).to_str() {
                        let mut chars = sep_str.chars();
                        chars
                            .next()
                            .filter(|&ch| chars.next().is_none() && !ch.is_alphanumeric())
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        }
        #[cfg(not(unix))]
        {
            None
        }
    };
    s.chars()
        .filter(|&c| c != '_' && Some(c) != locale_sep)
        .collect()
}

/// Parse a human-readable count (number of objects, jobs, etc.) from a string.
///
/// Accepts a plain integer or a number followed by a metric (SI) or binary (IEC) suffix.
/// Suffixes are **case-insensitive**.
///
/// | Suffix              | Meaning         | Value             |
/// |---------------------|-----------------|-------------------|
/// | *(none)*            | exact count     | 1                 |
/// | `k`, `kb`           | kilo (SI)       | 1,000             |
/// | `m`, `mb`           | mega (SI)       | 1,000,000         |
/// | `g`, `gb`           | giga (SI)       | 1,000,000,000     |
/// | `ki`, `kib`         | kibi (binary)   | 1,024             |
/// | `mi`, `mib`         | mebi (binary)   | 1,048,576         |
/// | `gi`, `gib`         | gibi (binary)   | 1,073,741,824     |
///
/// # Examples
///
/// ```text
/// 100000000  =>  100,000,000   (plain integer)
/// 100m       =>  100,000,000   (SI mega)
/// 1g         =>    1,000,000,000
/// 384k       =>      384,000
/// 1ki        =>        1,024   (binary kibi)
/// 1Mi        =>    1,048,576   (binary mebi)
/// ```
fn parse_human_count(input: &str) -> std::result::Result<usize, String> {
    let stripped = strip_separators(input.trim());
    let trimmed = stripped.as_str();
    if trimmed.is_empty() {
        return Err("count cannot be empty".to_string());
    }

    let numeric_len = trimmed.chars().take_while(|c| c.is_ascii_digit()).count();

    if numeric_len == 0 {
        return Err(format!("invalid count '{}': must start with digits", input));
    }

    let (value_str, suffix_str) = trimmed.split_at(numeric_len);
    let value: u128 = value_str
        .parse()
        .map_err(|_| format!("invalid count '{}': invalid number", input))?;

    let suffix = suffix_str.trim().to_ascii_lowercase();
    let multiplier: u128 = match suffix.as_str() {
        "" => 1,
        "k" | "kb" => 1_000,
        "m" | "mb" => 1_000_000,
        "g" | "gb" => 1_000_000_000,
        "ki" | "kib" => 1_024,
        "mi" | "mib" => 1_048_576,
        "gi" | "gib" => 1_073_741_824,
        _ => {
            return Err(format!(
                "invalid count '{}': unsupported suffix '{}'. \
                 Use k/m/g for SI (1k=1,000  1m=1,000,000  1g=1,000,000,000) \
                 or ki/mi/gi for binary (1ki=1,024  1mi=1,048,576  1gi=1,073,741,824)",
                input, suffix_str
            ))
        }
    };

    let count = value
        .checked_mul(multiplier)
        .ok_or_else(|| format!("invalid count '{}': value is too large", input))?;

    usize::try_from(count).map_err(|_| {
        format!(
            "invalid count '{}': value exceeds platform usize limit",
            input
        )
    })
}

/// Like `parse_human_count` but enforces `MAX_JOBS` as an upper bound.
///
/// Used for `--jobs` arguments across all subcommands.  The ceiling is defined
/// in `s3dlio::constants::MAX_JOBS` (currently 4,096).
fn parse_jobs(input: &str) -> std::result::Result<usize, String> {
    let n = parse_human_count(input)?;
    if n > MAX_JOBS {
        return Err(format!(
            "invalid jobs value '{}': {} exceeds the maximum of {} jobs. \
             On even the largest systems (256-core hosts) values above {} are \
             counterproductive due to connection overhead.",
            input, n, MAX_JOBS, MAX_JOBS
        ));
    }
    Ok(n)
}

fn parse_human_size(input: &str) -> std::result::Result<usize, String> {
    let stripped = strip_separators(input.trim());
    let trimmed = stripped.as_str();
    if trimmed.is_empty() {
        return Err("size cannot be empty".to_string());
    }

    let numeric_len = trimmed.chars().take_while(|c| c.is_ascii_digit()).count();

    if numeric_len == 0 {
        return Err(format!("invalid size '{}': must start with digits", input));
    }

    let (value_str, suffix_str) = trimmed.split_at(numeric_len);
    let value: u128 = value_str
        .parse()
        .map_err(|_| format!("invalid size '{}': invalid number", input))?;

    let suffix = suffix_str.trim().to_ascii_lowercase();
    let multiplier: u128 = match suffix.as_str() {
        "" | "b" => 1,
        "k" | "kb" => 1_000,
        "m" | "mb" => 1_000_000,
        "g" | "gb" => 1_000_000_000,
        "ki" | "kib" => 1_024,
        "mi" | "mib" => 1_048_576,
        "gi" | "gib" => 1_073_741_824,
        _ => {
            return Err(format!(
                "invalid size '{}': unsupported suffix '{}'. Supported: k, m, g, kb, mb, gb, ki, mi, gi, kib, mib, gib",
                input, suffix_str
            ))
        }
    };

    let bytes = value
        .checked_mul(multiplier)
        .ok_or_else(|| format!("invalid size '{}': value is too large", input))?;

    usize::try_from(bytes)
        .map_err(|_| format!("invalid size '{}': value exceeds platform size", input))
}

/// Main CLI function
#[tokio::main]
async fn main() -> Result<()> {
    // Loads any variables from .env file that are not already set
    dotenvy::dotenv().ok();

    let cli = Cli::parse();

    // Initialize tracing with env filter (compatible with dl-driver/s3-bench)
    // NOTE: 2025-12-03 - Debug-level logging for ALL crates causes hangs during AWS SDK operations.
    // Root cause: h2, hyper, and AWS SDK crates deadlock when debug/trace logging is active
    // inside tokio::spawn blocks (tracing subscriber contention with async I/O).
    // Fix: Always cap known-problematic crates at warn, regardless of RUST_LOG setting.
    // Bug filed: https://github.com/awslabs/aws-sdk-rust/issues/1388
    // See also: https://github.com/russfellows/s3dlio/issues/105
    let safe_filter = match cli.verbose {
        0 => "warn".to_string(),
        1 => "warn,s3dlio=info".to_string(), // -v: our code at info, deps at warn
        _ => "warn,s3dlio=debug".to_string(), // -vv: our code at debug, deps at warn
    };

    // Start from RUST_LOG if present, otherwise use our safe default.
    // Then forcibly cap problematic crates at warn to prevent deadlocks,
    // even if RUST_LOG=debug or RUST_LOG=trace was set globally.
    let base_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(&safe_filter));
    let filter = crate_log_caps()
        .into_iter()
        .fold(base_filter, |f, d| f.add_directive(d));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .init();

    // NOTE: tracing_log::LogTracer disabled 2025-12-03
    // This bridge from log->tracing causes deadlocks when AWS SDK logs from within
    // async OnceCell initialization. The AWS SDK uses the `log` crate internally.
    // Symptom: s3-cli hangs with -v or -vv flags.
    // tracing_log::LogTracer::init().ok();

    // If set, start the op‑logger *before* the first S3 call
    if let Some(ref path) = cli.op_log {
        if let Err(e) = init_op_logger(path.to_string_lossy()) {
            eprintln!("failed to start op‑logger `{}`: {e}", path.display());
        }
    }

    match cli.cmd {
        // List containers: multi-backend (s3://, gs://, az://, file://, direct://)
        Command::ListBuckets { uri } => {
            // S3 scheme requires AWS credentials up-front; other backends
            // discover their own credentials lazily.
            if uri.is_empty() || uri.starts_with("s3://") {
                check_aws_credentials()?;
            }
            list_buckets_cmd(&uri).await?;
        }

        Command::CreateBucket { uri } => {
            if requires_aws_credentials(&uri) {
                check_aws_credentials()?;
            }
            let name = extract_container_name(&uri)?;
            let logger = global_logger();
            let store = store_for_uri_with_logger(&uri, logger)?;
            info!("Attempting to create bucket/container: {}...", uri);
            store.create_container(&name).await?;
            safe_println!("Successfully created or verified '{}'.", uri);
        }

        Command::DeleteBucket { uri } => {
            if requires_aws_credentials(&uri) {
                check_aws_credentials()?;
            }
            let name = extract_container_name(&uri)?;
            let logger = global_logger();
            let store = store_for_uri_with_logger(&uri, logger)?;
            info!("Attempting to delete bucket/container: {}...", uri);
            store.delete_container(&name).await?;
            safe_println!("Successfully deleted '{}'.", uri);
        }

        Command::Stat { uri } => {
            // Only check AWS credentials if using S3
            if requires_aws_credentials(&uri) {
                check_aws_credentials()?;
            }
            stat_cmd(&uri).await?
        }

        // New, with regex and recursion
        Command::Get {
            uri,
            jobs,
            concurrent,
            keylist,
            recursive,
            offset,
            length,
            endpoints,
        } => {
            // Only check AWS credentials if URI is S3
            if let Some(uri_str) = &uri {
                if requires_aws_credentials(uri_str) {
                    check_aws_credentials()?;
                }
            }
            get_cmd(
                uri.as_deref(),
                jobs,
                concurrent,
                keylist.as_deref(),
                recursive,
                offset,
                length,
                endpoints.as_deref(),
            )
            .await?
        }

        Command::Delete {
            uri,
            jobs,
            recursive,
            pattern,
        } => {
            // Check AWS credentials for S3 URIs
            if requires_aws_credentials(&uri) {
                check_aws_credentials()?;
            }
            delete_cmd(&uri, jobs, recursive, pattern.as_deref()).await?
        }

        Command::Upload {
            files,
            dest,
            jobs,
            create_bucket,
        } => {
            // Pre-tune GCS subchannels to match upload concurrency.
            s3dlio::set_gcs_channel_count(jobs);

            // Check AWS credentials only for S3 destinations
            if requires_aws_credentials(&dest) {
                check_aws_credentials()?;
            }
            // Calculate total size of files to upload for progress tracking
            let mut total_bytes = 0u64;
            let mut file_count = 0;

            for pattern in &files {
                let s = pattern.to_string_lossy();
                if s.contains('*') || s.contains('?') {
                    for entry in glob::glob(&s)? {
                        match entry {
                            Ok(path) => {
                                if let Ok(metadata) = std::fs::metadata(&path) {
                                    if metadata.is_file() {
                                        total_bytes += metadata.len();
                                        file_count += 1;
                                    }
                                }
                            }
                            Err(e) => warn!("Glob error: {}", e),
                        }
                    }
                } else {
                    let path = std::path::Path::new(s.as_ref());
                    if let Ok(metadata) = std::fs::metadata(path) {
                        if metadata.is_file() {
                            total_bytes += metadata.len();
                            file_count += 1;
                        }
                    }
                }
            }

            if file_count == 0 {
                bail!("No files found to upload");
            }

            // Create progress tracker
            let progress_tracker =
                Arc::new(S3ProgressTracker::new("UPLOAD", file_count, total_bytes));
            let progress_callback =
                Arc::new(ProgressCallback::new(progress_tracker.clone(), file_count));

            // Handle bucket creation for backends that support it
            if create_bucket {
                let logger = global_logger();
                if let Ok(store) = store_for_uri_with_logger(&dest, logger) {
                    // Extract bucket/container name from URI
                    if dest.starts_with("s3://") {
                        if let Ok((bucket, _)) = parse_s3_uri(&dest) {
                            if let Err(e) = store.create_container(&bucket).await {
                                warn!("Failed to create bucket {}: {}", bucket, e);
                            }
                        }
                    } else if dest.starts_with("az://") || dest.starts_with("azure://") {
                        // For Azure, extract container name
                        let parts: Vec<&str> = dest
                            .trim_start_matches("az://")
                            .trim_start_matches("azure://")
                            .split('/')
                            .collect();
                        if let Some(container) = parts.first() {
                            if let Err(e) = store.create_container(container).await {
                                warn!("Failed to create container {}: {}", container, e);
                            }
                        }
                    } else if dest.starts_with("gs://") || dest.starts_with("gcs://") {
                        // For GCS, extract bucket name
                        let parts: Vec<&str> = dest
                            .trim_start_matches("gs://")
                            .trim_start_matches("gcs://")
                            .split('/')
                            .collect();
                        if let Some(bucket) = parts.first() {
                            if let Err(e) = store.create_container(bucket).await {
                                warn!("Failed to create GCS bucket {}: {}", bucket, e);
                            }
                        }
                    }
                    // File backends don't need bucket creation, directories are created automatically
                }
            }

            let t0 = Instant::now();
            let summary =
                generic_upload_files_with_summary(&dest, &files, jobs, Some(progress_callback))
                    .await?;
            let elapsed = t0.elapsed();

            // Finish progress bar
            progress_tracker.finish("Upload", total_bytes, elapsed);

            safe_println!(
                "UPLOAD summary: attempted={}, succeeded={}, failed={}",
                summary.attempted,
                summary.succeeded,
                summary.failed
            );

            if summary.failed > 0 {
                bail!(
                    "UPLOAD completed with failures: attempted={}, succeeded={}, failed={}",
                    summary.attempted,
                    summary.succeeded,
                    summary.failed
                );
            }
        }

        Command::Download {
            src,
            dest_dir,
            jobs,
            recursive,
        } => {
            // Pre-tune GCS subchannels to match download concurrency.
            s3dlio::set_gcs_channel_count(jobs);

            // Check AWS credentials only for S3 sources
            if requires_aws_credentials(&src) {
                check_aws_credentials()?;
            }
            // Use generic object store to list objects (works with all backends)
            let logger = global_logger();
            let store = store_for_uri_with_logger(&src, logger)?;
            let keys = store.list(&src, recursive).await?;

            if keys.is_empty() {
                bail!("No objects matched for download");
            }

            // Create progress tracker with unknown total bytes initially
            let progress_tracker =
                Arc::new(S3ProgressTracker::new("DOWNLOAD", keys.len() as u64, 0));
            let progress_callback = Arc::new(ProgressCallback::new(
                progress_tracker.clone(),
                keys.len() as u64,
            ));

            let t0 = Instant::now();
            let summary = generic_download_objects_with_summary(
                &src,
                &dest_dir,
                jobs,
                recursive,
                Some(progress_callback.clone()),
            )
            .await?;
            let elapsed = t0.elapsed();

            // Get final byte count from progress callback and update the progress bar total
            let total_bytes = progress_callback
                .bytes_transferred
                .load(std::sync::atomic::Ordering::Relaxed);
            progress_callback.update_total_bytes(total_bytes);
            progress_tracker.finish("Download", total_bytes, elapsed);

            safe_println!(
                "DOWNLOAD summary: attempted={}, succeeded={}, failed={}",
                summary.attempted,
                summary.succeeded,
                summary.failed
            );

            if summary.failed > 0 {
                bail!(
                    "DOWNLOAD completed with failures: attempted={}, succeeded={}, failed={}",
                    summary.attempted,
                    summary.succeeded,
                    summary.failed
                );
            }
        }

        Command::Cp {
            src,
            dest,
            jobs,
            recursive,
            create_bucket,
        } => {
            let src_is_uri = is_storage_uri(&src);
            let dest_is_uri = is_storage_uri(&dest);

            match (src_is_uri, dest_is_uri) {
                (false, true) => {
                    if requires_aws_credentials(&dest) {
                        check_aws_credentials()?;
                    }

                    if create_bucket {
                        let logger = global_logger();
                        if let Ok(store) = store_for_uri_with_logger(&dest, logger) {
                            if dest.starts_with("s3://") {
                                if let Ok((bucket, _)) = parse_s3_uri(&dest) {
                                    if let Err(e) = store.create_container(&bucket).await {
                                        warn!("Failed to create bucket {}: {}", bucket, e);
                                    }
                                }
                            } else if dest.starts_with("az://") || dest.starts_with("azure://") {
                                let parts: Vec<&str> = dest
                                    .trim_start_matches("az://")
                                    .trim_start_matches("azure://")
                                    .split('/')
                                    .collect();
                                if let Some(container) = parts.first() {
                                    if let Err(e) = store.create_container(container).await {
                                        warn!("Failed to create container {}: {}", container, e);
                                    }
                                }
                            } else if dest.starts_with("gs://") || dest.starts_with("gcs://") {
                                let parts: Vec<&str> = dest
                                    .trim_start_matches("gs://")
                                    .trim_start_matches("gcs://")
                                    .split('/')
                                    .collect();
                                if let Some(bucket) = parts.first() {
                                    if let Err(e) = store.create_container(bucket).await {
                                        warn!("Failed to create GCS bucket {}: {}", bucket, e);
                                    }
                                }
                            }
                        }
                    }

                    let files = vec![PathBuf::from(&src)];
                    let summary =
                        generic_upload_files_with_summary(&dest, &files, jobs, None).await?;
                    safe_println!(
                        "CP summary (upload): attempted={}, succeeded={}, failed={}",
                        summary.attempted,
                        summary.succeeded,
                        summary.failed
                    );
                    if summary.failed > 0 {
                        bail!(
                            "cp (upload) completed with failures: attempted={}, succeeded={}, failed={}",
                            summary.attempted,
                            summary.succeeded,
                            summary.failed
                        );
                    }
                }
                (true, false) => {
                    if requires_aws_credentials(&src) {
                        check_aws_credentials()?;
                    }

                    let summary = generic_download_objects_with_summary(
                        &src,
                        Path::new(&dest),
                        jobs,
                        recursive,
                        None,
                    )
                    .await?;
                    safe_println!(
                        "CP summary (download): attempted={}, succeeded={}, failed={}",
                        summary.attempted,
                        summary.succeeded,
                        summary.failed
                    );
                    if summary.failed > 0 {
                        bail!(
                            "cp (download) completed with failures: attempted={}, succeeded={}, failed={}",
                            summary.attempted,
                            summary.succeeded,
                            summary.failed
                        );
                    }
                }
                (true, true) => {
                    bail!("cp with URI->URI is not supported yet; use get/put, upload/download, or backend-native copy");
                }
                (false, false) => {
                    bail!("cp expects one local path and one storage URI");
                }
            }
        }

        Command::Put {
            uri_prefix,
            create_bucket_flag,
            num,
            template,
            jobs,
            size,
            object_type,
            dedup_f,
            compress_f,
            data_gen_mode,
            chunk_size,
            endpoints,
        } => {
            // Check AWS credentials only for S3 operations
            if requires_aws_credentials(&uri_prefix) {
                check_aws_credentials()?;

                // Handle S3-specific bucket creation
                if create_bucket_flag {
                    let (bucket, _prefix) = parse_s3_uri(&uri_prefix)?;
                    if let Err(e) = s3dlio::s3_utils::create_bucket(&bucket) {
                        eprintln!("Warning: failed to create bucket {}: {}", bucket, e);
                    }
                }
            }

            put_many_cmd(
                &uri_prefix,
                num,
                &template,
                jobs,
                size,
                object_type,
                dedup_f,
                compress_f,
                data_gen_mode,
                chunk_size,
                endpoints.as_deref(),
            )
            .await?
        }

        Command::GenericList {
            uri,
            recursive,
            pattern,
            count_only,
        } => {
            // Check AWS credentials only for S3 URIs
            if requires_aws_credentials(&uri) {
                check_aws_credentials()?;
            }
            generic_list_cmd(&uri, recursive, pattern.as_deref(), count_only).await?
        }

        Command::TfrecordIndex {
            tfrecord_path,
            index_path,
        } => {
            // Check AWS credentials only for S3 paths
            let tfrecord_str = tfrecord_path.to_string_lossy();
            if requires_aws_credentials(&tfrecord_str) {
                check_aws_credentials()?;
            }
            tfrecord_index_cmd(&tfrecord_path, index_path.as_deref())?
        }
    } // End of match cli.cmd

    // If set, finalize the op‑logger
    if cli.op_log.is_some() {
        finalize_op_logger();
    }

    Ok(())
}

/// Generic list command that works with any storage backend
/// Supports optional client-side regex filtering
/// Uses streaming for memory efficiency and visible progress
async fn generic_list_cmd(
    uri: &str,
    recursive: bool,
    pattern: Option<&str>,
    count_only: bool,
) -> Result<()> {
    use futures::stream::StreamExt;
    use regex::Regex;

    debug!(
        "generic_list_cmd: uri='{}', recursive={}, pattern={:?}, count_only={}",
        uri, recursive, pattern, count_only
    );

    // Helper to format numbers with commas
    fn format_with_commas(n: f64) -> String {
        let s = format!("{:.0}", n);
        let mut result = String::new();
        for (i, c) in s.chars().rev().enumerate() {
            if i > 0 && i % 3 == 0 {
                result.push(',');
            }
            result.push(c);
        }
        result.chars().rev().collect()
    }

    let logger = global_logger();
    let store = store_for_uri_with_logger(uri, logger)?;

    // Compile regex pattern if provided
    let re = pattern
        .map(|pat| Regex::new(pat).with_context(|| format!("Invalid regex pattern: '{}'", pat)))
        .transpose()?;

    let mut stream = store.list_stream(uri, recursive);
    let count = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let stdout = io::stdout();
    let mut out = stdout.lock();

    // Create progress bar for count-only mode with background update task
    let start = std::time::Instant::now();
    let progress_handle = if count_only {
        let pb = ProgressBar::new_spinner();
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] {msg}")
                .unwrap(),
        );
        pb.set_message("Listing objects...");
        pb.enable_steady_tick(std::time::Duration::from_millis(100));

        // Spawn background task to update progress every second
        let count_clone = count.clone();
        let pb_clone = pb.clone();
        let handle = tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));
            interval.tick().await; // First tick is immediate, skip it
            loop {
                interval.tick().await;
                let current_count = count_clone.load(std::sync::atomic::Ordering::Relaxed);
                let elapsed = start.elapsed().as_secs_f64();
                let rate = if elapsed > 0.0 {
                    current_count as f64 / elapsed
                } else {
                    0.0
                };
                pb_clone.set_message(format!(
                    "{} objects ({} obj/s)",
                    format_with_commas(current_count as f64),
                    format_with_commas(rate)
                ));
            }
        });
        Some((pb, handle))
    } else {
        None
    };

    while let Some(result) = stream.next().await {
        let key = result?;

        // Apply client-side regex filtering if pattern provided
        if let Some(ref regex) = re {
            if !regex.is_match(&key) {
                continue;
            }
        }

        count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Print URI unless --count-only
        if !count_only {
            if let Err(e) = writeln!(out, "{}", key) {
                if e.kind() == io::ErrorKind::BrokenPipe {
                    return Ok(()); // Handle cases like piping to `head`
                } else {
                    return Err(e.into());
                }
            }
        }
    }

    // Final summary
    let final_count = count.load(std::sync::atomic::Ordering::Relaxed);
    if let Some((pb, handle)) = progress_handle {
        handle.abort(); // Stop the background update task
        pb.finish_and_clear();
    }
    let elapsed = start.elapsed().as_secs_f64();
    let rate = final_count as f64 / elapsed;
    writeln!(
        out,
        "Total objects: {} ({:.3}s, rate: {} objects/s)",
        final_count,
        elapsed,
        format_with_commas(rate)
    )?;
    Ok(())
}

/// TFRecord index generation command
/// Creates NVIDIA DALI-compatible index files for TFRecord files
fn tfrecord_index_cmd(tfrecord_path: &PathBuf, index_path: Option<&Path>) -> Result<()> {
    use s3dlio::tfrecord_index::write_index_for_tfrecord_file;

    // Determine output path: use provided or default to input + ".idx"
    let output_path = match index_path {
        Some(p) => p.to_path_buf(),
        None => {
            let mut p = tfrecord_path.clone();
            let current_name = p
                .file_name()
                .context("Invalid TFRecord path")?
                .to_string_lossy()
                .to_string();
            p.set_file_name(format!("{}.idx", current_name));
            p
        }
    };

    info!("Indexing TFRecord file: {}", tfrecord_path.display());
    info!("Output index file: {}", output_path.display());

    let start = Instant::now();
    let num_records = write_index_for_tfrecord_file(tfrecord_path, &output_path)
        .map_err(|e| anyhow::anyhow!("Failed to create index: {}", e))?;
    let elapsed = start.elapsed();

    safe_println!(
        "Successfully indexed {} records in {:.2?}",
        num_records,
        elapsed
    );
    safe_println!("Index file: {}", output_path.display());
    safe_println!("Format: NVIDIA DALI compatible (text, space-separated)");

    Ok(())
}

/// Stat command: provide info on a single object (universal, works with all backends)
async fn stat_cmd(uri: &str) -> Result<()> {
    debug!("stat_cmd: uri='{}'", uri);
    let logger = global_logger();
    let store = store_for_uri_with_logger(uri, logger)?;
    let os = store.stat(uri).await?;
    debug!("stat_cmd: uri='{}', size={}", uri, os.size);

    // Always show core fields
    safe_println!("URI             : {}", uri);
    safe_println!("Size            : {}", os.size);

    // Only show optional fields if they have values
    if let Some(ref lm) = os.last_modified {
        safe_println!("LastModified    : {}", lm);
    }
    if let Some(ref et) = os.e_tag {
        safe_println!("ETag            : {}", et);
    }
    if let Some(ref ct) = os.content_type {
        safe_println!("Content-Type    : {}", ct);
    }
    if let Some(ref cl) = os.content_language {
        safe_println!("Content-Language: {}", cl);
    }
    if let Some(ref ce) = os.content_encoding {
        safe_println!("Content-Encoding: {}", ce);
    }
    if let Some(ref sc) = os.storage_class {
        safe_println!("StorageClass    : {}", sc);
    }
    if let Some(ref vid) = os.version_id {
        safe_println!("VersionId       : {}", vid);
    }
    if let Some(ref rs) = os.replication_status {
        safe_println!("ReplicationStat : {}", rs);
    }
    if let Some(ref sse) = os.server_side_encryption {
        safe_println!("SSE             : {}", sse);
    }
    if let Some(ref kmsid) = os.ssekms_key_id {
        safe_println!("SSE-KMS Key ID  : {}", kmsid);
    }

    // Show user metadata if present
    if !os.metadata.is_empty() {
        safe_println!("User Metadata:");
        for (k, v) in os.metadata {
            safe_println!("  {} = {}", k, v);
        }
    }
    Ok(())
}

/// Get command: downloads objects matching a key, prefix, or pattern.
#[allow(clippy::too_many_arguments)]
async fn get_cmd(
    uri: Option<&str>,
    jobs: usize,
    concurrent: Option<usize>,
    keylist: Option<&std::path::Path>,
    recursive: bool,
    offset: Option<u64>,
    length: Option<u64>,
    endpoints: Option<&str>,
) -> Result<()> {
    // Determine concurrency level (prefer concurrent over jobs for mp compatibility)
    let concurrency = concurrent.unwrap_or(jobs);

    // Pre-tune GCS subchannels to match concurrency before the first GCS client init.
    // The GCS client is a process-wide singleton (OnceCell); this must be called
    // before any GCS store is created.  S3DLIO_GCS_GRPC_CHANNELS env var overrides.
    s3dlio::set_gcs_channel_count(concurrency);

    // Validate range request parameters
    if (offset.is_some() || length.is_some()) && (keylist.is_some() || recursive) {
        bail!("Range requests (--offset/--length) are only supported for single-object GET, not with --keylist or --recursive");
    }

    // Handle keylist mode vs URI mode
    if let Some(keylist_path) = keylist {
        // Read URIs from keylist file
        let keylist_content = std::fs::read_to_string(keylist_path)
            .with_context(|| format!("Failed to read keylist file: {:?}", keylist_path))?;

        let mut uris: Vec<String> = keylist_content
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(|line| line.to_string())
            .collect();

        if uris.is_empty() {
            bail!("No URIs found in keylist file: {:?}", keylist_path);
        }

        // When --endpoints is given, build a shared MultiEndpointStore.
        // Rewrite all keylist URIs to use the first endpoint as the base;
        // the store will round-robin them across all endpoints on each get().
        let multi_store: Option<Arc<MultiEndpointStore>> = if let Some(endpoints_str) = endpoints {
            let first_endpoint = endpoints_str
                .split(',')
                .next()
                .unwrap_or("")
                .trim()
                .to_string();
            if first_endpoint.is_empty() {
                bail!("--endpoints requires at least one host:port");
            }
            // Rewrite each URI: s3://bucket/path → s3://first_endpoint:port/bucket/path
            uris = uris
                .iter()
                .map(|u| {
                    let path = u.strip_prefix("s3://").ok_or_else(|| {
                        anyhow::anyhow!("--endpoints requires s3:// URIs in keylist, got: {}", u)
                    })?;
                    Ok(format!("s3://{}/{}", first_endpoint, path))
                })
                .collect::<Result<Vec<_>>>()?;
            // Build endpoint roots: s3://host1:port/, s3://host2:port/, ...
            let endpoint_uris = build_s3_host_root_uris(endpoints_str)?;
            let store =
                MultiEndpointStore::new(endpoint_uris, LoadBalanceStrategy::RoundRobin, None)
                    .context("Failed to create multi-endpoint store")?;
            Some(Arc::new(store))
        } else {
            None
        };

        println!(
            "Processing {} URIs from keylist with concurrency {}",
            uris.len(),
            concurrency
        );

        // Create progress tracker for keylist mode
        let progress_tracker = Arc::new(S3ProgressTracker::new("GET", uris.len() as u64, 0));
        let progress_callback = Arc::new(ProgressCallback::new(
            progress_tracker.clone(),
            uris.len() as u64,
        ));

        let t0 = Instant::now();

        // Use universal ObjectStore interface for parallel downloads
        let sem = Arc::new(tokio::sync::Semaphore::new(concurrency));
        let mut futs = futures_util::stream::FuturesUnordered::new();
        let mut total_bytes = 0u64;
        let mut success_count = 0u64;
        let mut failure_count = 0u64;

        for uri in &uris {
            let sem = sem.clone();
            let progress = progress_callback.clone();
            let uri = uri.clone();
            let logger = global_logger();
            let multi = multi_store.clone();

            futs.push(tokio::spawn(async move {
                let _permit = sem.acquire_owned().await.unwrap();
                let data = if let Some(store) = multi {
                    store.get(&uri).await?
                } else {
                    let store = store_for_uri_with_logger(&uri, logger)?;
                    store.get(&uri).await?
                };
                let byte_count = data.len() as u64;

                // Update progress
                progress.object_completed(byte_count);

                Ok::<(String, u64), anyhow::Error>((uri, byte_count))
            }));
        }

        while let Some(result) = futs.next().await {
            match result {
                Ok(Ok((uri, byte_count))) => {
                    total_bytes += byte_count;
                    success_count += 1;
                    info!("Downloaded {} bytes from {}", byte_count, uri);
                }
                Ok(Err(e)) => {
                    failure_count += 1;
                    warn!("GET failed: {}", e);
                }
                Err(e) => {
                    failure_count += 1;
                    warn!("GET task join failed: {}", e);
                }
            }
        }

        let dt = t0.elapsed();

        // Update final progress
        progress_callback.update_total_bytes(total_bytes);
        progress_tracker.finish("Download", total_bytes, dt);

        // Report benchmark results (data is immediately discarded for benchmarking)
        let throughput_mb_s = (total_bytes as f64 / (1024.0 * 1024.0)) / dt.as_secs_f64();
        println!(
            "Benchmarked {} objects: {} bytes in {:.3}s ({:.2} MB/s)",
            uris.len(),
            total_bytes,
            dt.as_secs_f64(),
            throughput_mb_s
        );
        safe_println!(
            "GET summary: attempted={}, succeeded={}, failed={}",
            uris.len(),
            success_count,
            failure_count
        );

        if failure_count > 0 {
            bail!(
                "GET completed with failures: attempted={}, succeeded={}, failed={}",
                uris.len(),
                success_count,
                failure_count
            );
        }

        return Ok(());
    }

    // Original URI-based mode - use universal ObjectStore interface
    let uri_str =
        uri.ok_or_else(|| anyhow::anyhow!("Either --uri or --keylist must be provided"))?;

    // Only check AWS credentials if using S3 backend
    if requires_aws_credentials(uri_str) {
        check_aws_credentials()?;
    }

    // Handle single-object range request
    if let Some(offset_val) = offset {
        // Range request mode - single object only
        if recursive {
            bail!("Range requests (--offset/--length) cannot be combined with --recursive");
        }

        let logger = global_logger();
        let store = store_for_uri_with_logger(uri_str, logger)?;

        let t0 = Instant::now();
        let data = store.get_range(uri_str, offset_val, length).await?;
        let dt = t0.elapsed();

        let byte_count = data.len() as u64;
        let throughput_mb_s = (byte_count as f64 / (1024.0 * 1024.0)) / dt.as_secs_f64();

        let range_desc = if let Some(len) = length {
            format!("bytes {}-{}", offset_val, offset_val + len - 1)
        } else {
            format!("bytes {}-EOF", offset_val)
        };

        println!(
            "Range GET {}: {} bytes in {:.3}s ({:.2} MB/s)",
            range_desc,
            byte_count,
            dt.as_secs_f64(),
            throughput_mb_s
        );

        return Ok(());
    }

    // Use universal ObjectStore to list objects (works with all backends).
    // When --endpoints is given, list from the first endpoint then distribute GETs
    // across all endpoints via a shared MultiEndpointStore.
    let logger = global_logger();
    let multi_store: Option<Arc<MultiEndpointStore>> = if let Some(endpoints_str) = endpoints {
        if !uri_str.starts_with("s3://") {
            bail!("--endpoints requires an s3:// URI");
        }
        // Extract the bucket/prefix path from the URI and build per-endpoint URIs.
        let path = uri_str.strip_prefix("s3://").unwrap_or(uri_str);
        let path = if path.ends_with('/') {
            path.to_string()
        } else {
            format!("{}/", path)
        };
        let endpoint_uris = build_s3_endpoint_uris(endpoints_str, &format!("s3://{}", path))?;
        let store = MultiEndpointStore::new(endpoint_uris, LoadBalanceStrategy::RoundRobin, None)
            .context("Failed to create multi-endpoint store")?;
        Some(Arc::new(store))
    } else {
        None
    };
    let store = store_for_uri_with_logger(uri_str, logger)?;

    // When the URI is a directory prefix (ends with '/'), list recursively even if the
    // user did not pass -r.  A directory URI with recursive=false returns only the
    // immediate virtual sub-prefixes (entries ending in '/'), which are not real objects
    // and cannot be GET-ted; the user always wants the leaf objects inside the prefix.
    let do_recursive = recursive || uri_str.ends_with('/');

    // === Shared real-time progress state ===
    // These atomics are updated by spawned download tasks so the background
    // progress updater always sees current counts without any locking on the
    // hot path.
    let keys_listed = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let keys_done = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let bytes_done = Arc::new(std::sync::atomic::AtomicU64::new(0));

    // Rolling rate window for obj/s computation — see CLI_RATE_WINDOW_SECS in constants.rs.
    // Sampled every 500 ms by the background task.
    let rate_window: Arc<std::sync::Mutex<std::collections::VecDeque<(Instant, u64)>>> =
        Arc::new(std::sync::Mutex::new(std::collections::VecDeque::new()));

    // Bytes-based progress bar — same visual style as the old S3ProgressTracker.
    // We start with length=0 and update it each tick using the estimated total bytes
    // (avg_bytes_per_object × remaining_objects + bytes_already_done), so indicatif
    // can compute {bytes_per_sec} and {eta} natively from real byte throughput.
    let pb = ProgressBar::new(0);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(
                "GET: {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] \
                 {bytes}/{total_bytes} ({bytes_per_sec}, ETA: {eta}) | {msg}",
            )
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏  "),
    );
    pb.enable_steady_tick(std::time::Duration::from_millis(80));

    let t0 = Instant::now();

    // Background task: updates the progress bar every 500 ms.
    //
    // - pb.set_position(bytes) drives all indicatif-computed fields: {bytes},
    //   {bytes_per_sec}, and {eta}.
    // - pb.set_length(estimated_total) gives indicatif a moving-but-converging
    //   estimate of total bytes so the bar fill and ETA are meaningful from the
    //   start, before listing completes.
    // - {msg} carries object count and the rolling obj/s rate (window = CLI_RATE_WINDOW_SECS).
    let pb_bg = pb.clone();
    let listed_bg = Arc::clone(&keys_listed);
    let done_bg = Arc::clone(&keys_done);
    let bytes_bg = Arc::clone(&bytes_done);
    let window_bg = Arc::clone(&rate_window);
    let updater = tokio::spawn(async move {
        use std::sync::atomic::Ordering::Relaxed;
        let mut interval = tokio::time::interval(std::time::Duration::from_millis(500));
        interval.tick().await; // skip the immediate first tick
        loop {
            interval.tick().await;
            let listed = listed_bg.load(Relaxed);
            let completed = done_bg.load(Relaxed);
            let bytes = bytes_bg.load(Relaxed);

            // Drive the bytes bar so indicatif tracks real byte throughput.
            pb_bg.set_position(bytes);

            // Estimate total bytes = bytes so far + avg_bytes/obj × remaining objects.
            // This keeps the bar fill and ETA reasonable even while listing is still
            // ongoing.  Once listing finishes the estimate converges to reality.
            if let Some(avg_bytes_per_obj) = bytes.checked_div(completed) {
                let remaining = listed.saturating_sub(completed);
                let estimated_total = bytes + remaining * avg_bytes_per_obj;
                pb_bg.set_length(estimated_total);
            }

            // Rolling obj/s rate — window length from CLI_RATE_WINDOW_SECS in constants.rs.
            let now = Instant::now();
            let rate_obj_s = {
                let mut win = window_bg.lock().unwrap();
                win.push_back((now, completed));
                // Evict samples outside the rolling window.
                while win
                    .front()
                    .map(|(t, _)| {
                        now.duration_since(*t)
                            > std::time::Duration::from_secs(
                                s3dlio::constants::CLI_RATE_WINDOW_SECS,
                            )
                    })
                    .unwrap_or(false)
                {
                    win.pop_front();
                }
                if win.len() >= 2 {
                    let (oldest_t, oldest_c) = *win.front().unwrap();
                    let span_s = now.duration_since(oldest_t).as_secs_f64();
                    if span_s > 0.1 {
                        completed.saturating_sub(oldest_c) as f64 / span_s
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            };

            pb_bg.set_message(format!(
                "{}/{} objects | {:.1} obj/s ({}s avg)",
                completed,
                listed,
                rate_obj_s,
                s3dlio::constants::CLI_RATE_WINDOW_SECS,
            ));
        }
    });

    // === Semaphore-bounded streaming pipeline ===
    //
    // The semaphore permit is acquired in the main task *before* spawning each
    // tokio task.  This keeps the live-task count pinned to exactly `concurrency`
    // regardless of how many objects exist in the prefix.  While the main task is
    // suspended on `acquire_owned()`, the already-running tasks release permits as
    // they finish — no deadlock is possible.
    //
    // Keys arrive from the listing stream one-by-one, so BidiReadObject streams open
    // progressively instead of all simultaneously — reducing thundering-herd pressure
    // on GCS RAPID read-ahead buffers.
    let sem = Arc::new(tokio::sync::Semaphore::new(concurrency));
    let mut futs = futures_util::stream::FuturesUnordered::new();
    let mut stream = store.list_stream(uri_str, do_recursive);
    let mut failure_count = 0u64;

    // Phase 1: consume the listing stream, spawning one bounded download per key.
    while let Some(result) = stream.next().await {
        let key = result.with_context(|| format!("list error for '{}'", uri_str))?;
        if key.ends_with('/') {
            continue; // skip virtual directory entries (common prefixes)
        }
        keys_listed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Block here until a concurrency slot is available, then spawn immediately.
        let permit = Arc::clone(&sem)
            .acquire_owned()
            .await
            .expect("semaphore closed");
        let logger = global_logger();
        let done_task = Arc::clone(&keys_done);
        let bytes_task = Arc::clone(&bytes_done);
        let multi = multi_store.clone();

        futs.push(tokio::spawn(async move {
            let _permit = permit; // released when this block exits
            let data = if let Some(store) = multi {
                store.get(&key).await?
            } else {
                let store = store_for_uri_with_logger(&key, logger)?;
                store.get(&key).await?
            };
            let byte_count = data.len() as u64;
            // Update shared progress atomics on success so the background task
            // sees live counts without waiting for Phase 2.
            done_task.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            bytes_task.fetch_add(byte_count, std::sync::atomic::Ordering::Relaxed);
            Ok::<u64, anyhow::Error>(byte_count)
        }));
    }

    let keys_seen = keys_listed.load(std::sync::atomic::Ordering::Relaxed);
    if keys_seen == 0 {
        updater.abort();
        pb.finish_and_clear();
        bail!("No objects match pattern '{}'", uri_str);
    }

    // Phase 2: drain all in-flight downloads and collect errors.
    // Byte/object counts are already tracked via atomics; this loop only gathers failures.
    while let Some(result) = futs.next().await {
        match result {
            Ok(Ok(_)) => {} // atomics already updated inside the task
            Ok(Err(e)) => {
                failure_count += 1;
                warn!("GET failed: {}", e);
            }
            Err(e) => {
                failure_count += 1;
                warn!("GET task join failed: {}", e);
            }
        }
    }

    // Stop the background updater and display final summary on the progress bar.
    updater.abort();
    let dt = t0.elapsed();
    let total_bytes = bytes_done.load(std::sync::atomic::Ordering::Relaxed);
    let success_count = keys_done.load(std::sync::atomic::Ordering::Relaxed);
    // Land the bar at exactly 100 % so the fill is complete.
    // indicatif already renders {bytes}/{total_bytes} and {bytes_per_sec} from the
    // bar template, so {msg} only carries what indicatif cannot: obj count and obj/s.
    pb.set_length(total_bytes);
    pb.set_position(total_bytes);
    let avg_obj_s = success_count as f64 / dt.as_secs_f64();
    pb.finish_with_message(format!(
        "{}/{} objects | {:.1} obj/s avg",
        success_count, keys_seen, avg_obj_s,
    ));

    safe_println!(
        "GET summary: attempted={}, succeeded={}, failed={}",
        keys_seen,
        success_count,
        failure_count,
    );

    if failure_count > 0 {
        bail!(
            "GET completed with failures: attempted={}, succeeded={}, failed={}",
            keys_seen,
            success_count,
            failure_count,
        );
    }

    Ok(())
}

// NOTE: mp-get CLI command removed.
//
// The process-spawn approach had worse real-world performance than a single
// `get -j N` invocation. Each worker process pays full cold-start costs
// (Tokio runtime init, AWS SDK init, HTTP connection pool build-up), whereas
// a single async process amortizes all of that once and handles thousands of
// concurrent tasks with negligible overhead.
//
// The `mp` module and Python `mp_get()` binding are retained unchanged.
// If multi-process GET is worth revisiting, consider:
//   - Pre-warming worker processes and reusing them across runs
//   - NUMA-local process pinning for very large NUMA server configs
//   - Measuring Tokio reactor saturation at >1024 concurrent tasks
//   - Always wiring op_log_dir so byte throughput is properly reported
//   - Listing real prefix objects instead of synthesizing keys from a template
//
// For now: `s3dlio get s3://bucket/prefix/ -j 256 -r` is the recommended path.

/// Delete command: deletes objects matching a key, prefix, or pattern.
async fn delete_cmd(uri: &str, _jobs: usize, recursive: bool, pattern: Option<&str>) -> Result<()> {
    use indicatif::{ProgressBar, ProgressStyle};
    use regex::Regex;
    use s3dlio::object_store::store_for_uri_with_logger;
    use std::sync::Arc;

    // Use generic object store to delete objects (works with all backends)
    let logger = global_logger();
    let store = Arc::new(store_for_uri_with_logger(uri, logger)?);

    // If recursive or URI ends with '/', delete prefix; otherwise check if it's a pattern
    if recursive || uri.ends_with('/') {
        if let Some(pat) = pattern {
            // Delete with regex filter
            let mut keys = store.list(uri, recursive).await?;
            let re =
                Regex::new(pat).with_context(|| format!("Invalid regex pattern: '{}'", pat))?;
            keys.retain(|k| re.is_match(k));

            if keys.is_empty() {
                eprintln!("No objects match pattern '{}' under prefix '{}'", pat, uri);
                return Ok(());
            }

            let total = keys.len();
            info!(
                "Deleting {} objects matching pattern '{}' under prefix '{}'",
                total, pat, uri
            );
            eprintln!(
                "Found {} objects matching pattern '{}' (using adaptive concurrency: ~{})",
                total,
                pat,
                if total < 10 {
                    1
                } else if total < 100 {
                    10
                } else {
                    (total / 10).min(1000)
                }
            );

            // Create progress bar for deletion
            let pb = ProgressBar::new(total as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("Deleting: {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} objects ({per_sec}, ETA: {eta})")
                    .unwrap()
                    .progress_chars("█▉▊▋▌▍▎▏  ")
            );

            // Clone progress bar for the callback
            let pb_clone = pb.clone();

            // Use batch delete (backends optimize internally - S3 uses DeleteObjects API)
            for chunk in keys.chunks(1000) {
                store.delete_batch(chunk).await?;
                pb_clone.inc(chunk.len() as u64);
            }

            pb.finish_with_message(format!("Deleted {} objects matching pattern", total));
            eprintln!(
                "\nSuccessfully deleted {} objects matching pattern '{}'",
                total, pat
            );
        } else {
            // Delete entire prefix without filter - use streaming pipeline with progress bar
            info!(
                "Starting streaming list+delete pipeline for prefix: {}",
                uri
            );

            // Create indeterminate progress bar (we don't know total count yet)
            let pb = ProgressBar::new_spinner();
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("Deleting: {spinner:.green} [{elapsed_precise}] {msg}")
                    .unwrap(),
            );
            pb.enable_steady_tick(std::time::Duration::from_millis(100));
            pb.set_message("Starting pipeline...");

            // Channel for passing batches from lister to deleter (buffer 10 batches)
            let (tx, mut rx) = tokio::sync::mpsc::channel::<Vec<String>>(10);

            // Spawn lister task - streams results and batches them
            let store_list = store.clone();
            let uri_list = uri.to_string();
            let pb_list = pb.clone();
            let lister = tokio::spawn(async move {
                use futures::stream::StreamExt;

                let mut stream = store_list.list_stream(&uri_list, recursive);
                let mut batch = Vec::with_capacity(1000);
                let mut total_listed = 0u64;

                while let Some(result) = stream.next().await {
                    match result {
                        Ok(key) => {
                            batch.push(key);
                            total_listed += 1;

                            // Send batch when it reaches 1000 objects
                            if batch.len() >= 1000 {
                                if tx.send(batch.clone()).await.is_err() {
                                    // Receiver dropped (error in deleter)
                                    break;
                                }
                                batch.clear();

                                // Update progress bar with nice formatting
                                if total_listed.is_multiple_of(1000) {
                                    pb_list.set_message(format!(
                                        "Listed: {} | Deleting in background...",
                                        total_listed
                                    ));
                                }
                            }
                        }
                        Err(e) => {
                            pb_list.finish_with_message(format!("List error: {}", e));
                            break;
                        }
                    }
                }

                // Send final partial batch if any
                if !batch.is_empty() {
                    let _ = tx.send(batch).await;
                }

                total_listed
            });

            // Deleter task - receives batches and deletes them IN PARALLEL
            let store_delete = store.clone();
            let pb_delete = pb.clone();
            let deleter = tokio::spawn(async move {
                use futures::stream::{FuturesUnordered, StreamExt};

                let mut total_deleted = 0u64;
                let delete_start = std::time::Instant::now();
                let max_concurrent_deletes = 10; // Allow 10 concurrent DeleteObjects requests
                let mut active_deletes = FuturesUnordered::new();

                loop {
                    // Try to receive batches while we have room for more concurrent deletes
                    while active_deletes.len() < max_concurrent_deletes {
                        match rx.try_recv() {
                            Ok(batch) => {
                                let store_clone = store_delete.clone();
                                let batch_size = batch.len();

                                // Spawn delete as a future
                                active_deletes.push(tokio::spawn(async move {
                                    store_clone.delete_batch(&batch).await.map(|_| batch_size)
                                }));
                            }
                            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                                // No more batches available right now
                                break;
                            }
                            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => {
                                // Lister finished, process remaining deletes
                                break;
                            }
                        }
                    }

                    // Wait for at least one delete to complete if we have any active
                    if !active_deletes.is_empty() {
                        match active_deletes.next().await {
                            Some(Ok(Ok(batch_size))) => {
                                total_deleted += batch_size as u64;

                                // Update progress bar with rate information
                                let elapsed = delete_start.elapsed().as_secs_f64().max(0.001);
                                let rate = (total_deleted as f64 / elapsed) as u64;
                                pb_delete.set_message(format!(
                                    "Deleted: {} ({}/s, {} concurrent)",
                                    total_deleted,
                                    rate,
                                    active_deletes.len() + 1
                                ));
                            }
                            Some(Ok(Err(e))) => {
                                pb_delete.finish_with_message(format!("Delete error: {}", e));
                                return Err(e);
                            }
                            Some(Err(e)) => {
                                pb_delete.finish_with_message(format!("Task error: {}", e));
                                return Err(anyhow::anyhow!("Delete task failed: {}", e));
                            }
                            None => break, // No more active deletes
                        }
                    } else {
                        // Check if lister is done
                        match rx.try_recv() {
                            Err(tokio::sync::mpsc::error::TryRecvError::Disconnected) => break,
                            Err(tokio::sync::mpsc::error::TryRecvError::Empty) => {
                                // Wait a bit for more work
                                tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
                            }
                            Ok(batch) => {
                                // Got a batch, process it
                                let store_clone = store_delete.clone();
                                let batch_size = batch.len();
                                active_deletes.push(tokio::spawn(async move {
                                    store_clone.delete_batch(&batch).await.map(|_| batch_size)
                                }));
                            }
                        }
                    }
                }

                // Wait for any remaining deletes to complete
                while let Some(result) = active_deletes.next().await {
                    match result {
                        Ok(Ok(batch_size)) => {
                            total_deleted += batch_size as u64;
                            let elapsed = delete_start.elapsed().as_secs_f64().max(0.001);
                            let rate = (total_deleted as f64 / elapsed) as u64;
                            pb_delete
                                .set_message(format!("Deleted: {} ({}/s)", total_deleted, rate));
                        }
                        Ok(Err(e)) => {
                            pb_delete.finish_with_message(format!("Delete error: {}", e));
                            return Err(e);
                        }
                        Err(e) => {
                            pb_delete.finish_with_message(format!("Task error: {}", e));
                            return Err(anyhow::anyhow!("Delete task failed: {}", e));
                        }
                    }
                }

                Ok::<u64, anyhow::Error>(total_deleted)
            });

            // Wait for both tasks to complete
            let total_listed = lister.await?;
            let total_deleted = deleter.await??;

            if total_listed == 0 {
                pb.finish_with_message("No objects found");
                eprintln!("No objects found under prefix: {}", uri);
                return Ok(());
            }

            pb.finish_with_message(format!("✓ Deleted {} objects", total_deleted));
            eprintln!("Successfully deleted all objects under prefix: {}", uri);
        }
    } else {
        // First, try to list objects with this URI as a prefix to see if it matches multiple objects
        let mut keys = store.list(uri, false).await?;

        // Apply regex filter if provided
        if let Some(pat) = pattern {
            let re =
                Regex::new(pat).with_context(|| format!("Invalid regex pattern: '{}'", pat))?;
            keys.retain(|k| re.is_match(k));
        }

        if keys.is_empty() {
            // No objects found with this prefix, try as exact object name
            info!("Deleting single object: {}", uri);
            store.delete(uri).await?;
            eprintln!("Successfully deleted 1 object: {}", uri);
        } else if keys.len() == 1 && keys[0] == uri && pattern.is_none() {
            // Exactly one object and it matches the URI exactly (no pattern filter)
            info!("Deleting single object: {}", uri);
            store.delete(uri).await?;
            eprintln!("Successfully deleted 1 object: {}", uri);
        } else {
            // Multiple objects match the prefix (or pattern filter was applied)
            let total_count = keys.len();
            info!("Deleting {} objects matching prefix: {}", total_count, uri);
            if let Some(pat) = pattern {
                eprintln!(
                    "Found {} objects matching prefix '{}' and pattern '{}'",
                    total_count, uri, pat
                );
            } else {
                eprintln!("Found {} objects matching prefix '{}'", total_count, uri);
            }

            // Create progress bar for deletion
            let pb = ProgressBar::new(total_count as u64);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("Deleting: {spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} objects ({per_sec}, ETA: {eta})")
                    .unwrap()
                    .progress_chars("█▉▊▋▌▍▎▏  ")
            );

            for key in &keys {
                store.delete(key).await?;
                info!("Deleted: {}", key);
                pb.inc(1);
            }

            pb.finish_with_message(format!("Completed deletion of {} objects", total_count));
            if let Some(pat) = pattern {
                eprintln!(
                    "\nSuccessfully deleted {} objects matching pattern '{}'",
                    total_count, pat
                );
            } else {
                eprintln!("\nSuccessfully deleted {} objects", total_count);
            }
        }
    }

    Ok(())
}

/// Build per-endpoint S3 URIs from a comma-separated endpoint list and a base S3 URI.
///
/// For example, `endpoints = "10.9.0.17:9000,10.9.0.18:9000"` and
/// `base_uri = "s3://mybucket/bench/"` produces:
/// `["s3://10.9.0.17:9000/mybucket/bench/", "s3://10.9.0.18:9000/mybucket/bench/"]`
///
/// The base URI must start with `s3://`.
fn build_s3_endpoint_uris(endpoints: &str, base_uri: &str) -> Result<Vec<String>> {
    let path = base_uri
        .strip_prefix("s3://")
        .ok_or_else(|| anyhow::anyhow!("--endpoints requires an s3:// URI, got: {}", base_uri))?;
    let path = if path.ends_with('/') {
        path.to_string()
    } else {
        format!("{}/", path)
    };
    let uris: Vec<String> = endpoints
        .split(',')
        .map(|h| h.trim())
        .filter(|h| !h.is_empty())
        .map(|h| format!("s3://{}/{}", h, path))
        .collect();
    if uris.is_empty() {
        bail!("--endpoints requires at least one host:port value");
    }
    Ok(uris)
}

/// Build bare per-endpoint root URIs (`s3://host:port/`) from a comma-separated endpoint list.
///
/// Used for keylist mode where object paths are preserved and only the endpoint is replaced.
fn build_s3_host_root_uris(endpoints: &str) -> Result<Vec<String>> {
    let uris: Vec<String> = endpoints
        .split(',')
        .map(|h| h.trim())
        .filter(|h| !h.is_empty())
        .map(|h| format!("s3://{}/", h))
        .collect();
    if uris.is_empty() {
        bail!("--endpoints requires at least one host:port value");
    }
    Ok(uris)
}

/// Put command supports 1 or more objects, also takes our ObjectType
#[allow(clippy::too_many_arguments)]
async fn put_many_cmd(
    uri_prefix: &str,
    num: usize,
    template: &str,
    jobs: usize,
    size: usize,
    object_type: s3dlio::ObjectType,
    dedup_f: usize,
    compress_f: usize,
    data_gen_mode: DataGenMode,
    chunk_size: usize,
    endpoints: Option<&str>,
) -> Result<()> {
    // Pre-tune GCS subchannels to match upload concurrency before the first GCS op.
    s3dlio::set_gcs_channel_count(jobs);

    // Ensure prefix ends with '/' for consistent URI generation
    let mut prefix = uri_prefix.to_string();
    if !prefix.ends_with('/') {
        prefix.push('/');
    }

    // When --endpoints is given, construct per-endpoint URIs and build a MultiEndpointStore.
    // Object URIs are generated using the first endpoint as the base; the store rewrites
    // each URI to the selected (round-robin) endpoint on every put() call.
    let multi_store: Option<Arc<MultiEndpointStore>> = if let Some(endpoints_str) = endpoints {
        let endpoint_uris = build_s3_endpoint_uris(endpoints_str, &prefix)?;
        let store = MultiEndpointStore::new(endpoint_uris, LoadBalanceStrategy::RoundRobin, None)
            .context("Failed to create multi-endpoint store")?;
        // Use the first endpoint's URI as the base for generating object URIs.
        prefix = store.get_endpoint_configs()[0].0.clone();
        Some(Arc::new(store))
    } else {
        None
    };

    // Generate the full list of URIs (always using `prefix` which may now be endpoint-specific).
    let mut uris = Vec::with_capacity(num);

    // Now replace brackets with values
    for i in 0..num {
        // replace first {} with the index, second {} with the total count
        let object_name =
            template
                .replacen("{}", &i.to_string(), 1)
                .replacen("{}", &num.to_string(), 1);
        let full_uri = format!("{}{}", prefix, object_name);
        uris.push(full_uri);
    }

    // Find the lesser of the number of jobs or number of objects
    let effective_jobs = std::cmp::min(jobs, num);
    let total_bytes = num * size;

    // Create progress bar for upload operation
    let progress_tracker = Arc::new(S3ProgressTracker::new(
        "PUT",
        num as u64,
        total_bytes as u64,
    ));
    let progress_callback = Arc::new(ProgressCallback::new(progress_tracker.clone(), num as u64));
    progress_tracker
        .progress_bar
        .set_message(format!("Preparing to upload {} objects...", num));

    let t0 = Instant::now();

    // Generate test data
    let config = Config::new_with_defaults(object_type, 1, size, dedup_f, compress_f)
        .with_data_gen_mode(data_gen_mode)
        .with_chunk_size(chunk_size);
    let data = generate_object(&config)?;

    // Use universal ObjectStore interface for parallel uploads
    let sem = Arc::new(tokio::sync::Semaphore::new(effective_jobs));
    let mut futs = FuturesUnordered::new();
    let logger = global_logger();

    for uri in &uris {
        let sem = sem.clone();
        let progress = progress_callback.clone();
        let uri = uri.clone();
        let data = data.clone();
        let logger = logger.clone();
        let multi = multi_store.clone();

        futs.push(tokio::spawn(async move {
            let _permit = sem.acquire_owned().await.unwrap();
            let byte_count = data.len() as u64;
            if let Some(store) = multi {
                // Multi-endpoint: store routes round-robin and rewrites the URI.
                store.put(&uri, data).await?;
            } else {
                let store = store_for_uri_with_logger(&uri, logger)?;
                // Bytes passed directly (zero-copy, reference counted)
                store.put(&uri, data).await?;
            }

            // Update progress
            progress.object_completed(byte_count);

            Ok::<(String, u64), anyhow::Error>((uri, byte_count))
        }));
    }

    let mut total_uploaded_bytes = 0u64;
    let mut success_count = 0u64;
    let mut failure_count = 0u64;
    while let Some(result) = futs.next().await {
        match result {
            Ok(Ok((uri, byte_count))) => {
                total_uploaded_bytes += byte_count;
                success_count += 1;
                info!("Uploaded {} bytes to {}", byte_count, uri);
            }
            Ok(Err(e)) => {
                failure_count += 1;
                warn!("PUT failed: {}", e);
            }
            Err(e) => {
                failure_count += 1;
                warn!("PUT task join failed: {}", e);
            }
        }
    }

    let elapsed = t0.elapsed();

    // Update final progress and finish
    progress_callback.update_total_bytes(total_uploaded_bytes);
    progress_tracker.finish("Upload", total_uploaded_bytes, elapsed);

    let proto_tag = s3dlio::reqwest_client::observed_http_version_str()
        .map(|v| format!(", protocol={v}"))
        .unwrap_or_default();
    safe_println!(
        "PUT summary: attempted={}, succeeded={}, failed={}{}",
        num,
        success_count,
        failure_count,
        proto_tag
    );

    if failure_count > 0 {
        bail!(
            "PUT completed with failures: attempted={}, succeeded={}, failed={}",
            num,
            success_count,
            failure_count
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        build_s3_endpoint_uris, build_s3_host_root_uris, extract_container_name, parse_human_count,
        parse_human_size, parse_jobs, Cli, Command,
    };
    use clap::Parser;
    use s3dlio::constants::MAX_JOBS;

    // ------------------------------------------------------------------
    // extract_container_name — no network access, pure string parsing
    // ------------------------------------------------------------------

    #[test]
    fn extract_s3_simple_bucket() {
        assert_eq!(
            extract_container_name("s3://my-bucket").unwrap(),
            "my-bucket"
        );
    }

    #[test]
    fn extract_s3_bucket_with_prefix() {
        // Only the bucket portion should be returned
        assert_eq!(
            extract_container_name("s3://my-bucket/some/prefix/").unwrap(),
            "my-bucket"
        );
    }

    #[test]
    fn extract_s3_trailing_slash_only() {
        // "s3://" alone has no bucket — should error
        assert!(extract_container_name("s3://").is_err());
    }

    #[test]
    fn extract_gcs_gs_prefix() {
        assert_eq!(
            extract_container_name("gs://my-gcs-bucket").unwrap(),
            "my-gcs-bucket"
        );
    }

    #[test]
    fn extract_gcs_gcs_prefix() {
        assert_eq!(
            extract_container_name("gcs://my-gcs-bucket/prefix/").unwrap(),
            "my-gcs-bucket"
        );
    }

    #[test]
    fn extract_gcs_missing_bucket() {
        assert!(extract_container_name("gs://").is_err());
    }

    #[test]
    fn extract_azure_account_and_container() {
        assert_eq!(
            extract_container_name("az://myaccount/mycontainer").unwrap(),
            "myaccount/mycontainer"
        );
    }

    #[test]
    fn extract_azure_with_prefix() {
        // Prefix beyond account/container is ignored; only account/container returned
        assert_eq!(
            extract_container_name("az://myaccount/mycontainer/some/prefix/").unwrap(),
            "myaccount/mycontainer"
        );
    }

    #[test]
    fn extract_azure_missing_container() {
        assert!(extract_container_name("az://myaccount").is_err());
        assert!(extract_container_name("az://myaccount/").is_err());
    }

    #[test]
    fn extract_azure_missing_account() {
        assert!(extract_container_name("az://").is_err());
    }

    #[test]
    fn extract_azure_prefix_variant() {
        assert_eq!(
            extract_container_name("azure://acc/cont").unwrap(),
            "acc/cont"
        );
    }

    #[test]
    fn extract_file_absolute_path() {
        assert_eq!(
            extract_container_name("file:///mnt/data").unwrap(),
            "/mnt/data"
        );
    }

    #[test]
    fn extract_file_nested_path() {
        assert_eq!(
            extract_container_name("file:///mnt/data/subdir/").unwrap(),
            "/mnt/data/subdir"
        );
    }

    #[test]
    fn extract_file_empty_path() {
        assert!(extract_container_name("file://").is_err());
    }

    #[test]
    fn extract_direct_path() {
        assert_eq!(
            extract_container_name("direct:///nvme/data").unwrap(),
            "/nvme/data"
        );
    }

    #[test]
    fn extract_direct_trailing_slash_stripped() {
        assert_eq!(
            extract_container_name("direct:///nvme/data/").unwrap(),
            "/nvme/data"
        );
    }

    #[test]
    fn extract_unknown_scheme_errors() {
        assert!(extract_container_name("ftp://some-host/bucket").is_err());
        assert!(extract_container_name("http://example.com/bucket").is_err());
        assert!(extract_container_name("just-a-name").is_err());
    }

    // ------------------------------------------------------------------
    // parse_human_size — existing tests
    // ------------------------------------------------------------------

    #[test]
    fn parse_human_size_accepts_bytes_and_suffixes() {
        assert_eq!(parse_human_size("8").unwrap(), 8);
        assert_eq!(parse_human_size("8b").unwrap(), 8);
        assert_eq!(parse_human_size("8K").unwrap(), 8_000);
        assert_eq!(parse_human_size("8m").unwrap(), 8_000_000);
        assert_eq!(parse_human_size("8MB").unwrap(), 8_000_000);
        assert_eq!(parse_human_size("8mb").unwrap(), 8_000_000);
        assert_eq!(parse_human_size("8G").unwrap(), 8_000_000_000);
        assert_eq!(parse_human_size("8MiB").unwrap(), 8 * 1_048_576);
        assert_eq!(parse_human_size("8mib").unwrap(), 8 * 1_048_576);
        assert_eq!(parse_human_size("8Gi").unwrap(), 8 * 1_073_741_824);
        assert_eq!(parse_human_size("8KiB").unwrap(), 8 * 1_024);
    }

    #[test]
    fn parse_human_size_rejects_invalid_values() {
        assert!(parse_human_size("").is_err());
        assert!(parse_human_size("mb").is_err());
        assert!(parse_human_size("8XB").is_err());
        assert!(parse_human_size("8.5MB").is_err());
    }

    // ------------------------------------------------------------------
    // parse_human_count — SI and binary count suffixes
    // ------------------------------------------------------------------

    #[test]
    fn parse_human_count_accepts_plain_integers() {
        assert_eq!(parse_human_count("0").unwrap(), 0);
        assert_eq!(parse_human_count("1").unwrap(), 1);
        assert_eq!(parse_human_count("100000000").unwrap(), 100_000_000);
        assert_eq!(parse_human_count("384").unwrap(), 384);
    }

    #[test]
    fn parse_human_count_si_suffixes() {
        // SI: k=1,000  m=1,000,000  g=1,000,000,000 (case-insensitive)
        assert_eq!(parse_human_count("1k").unwrap(), 1_000);
        assert_eq!(parse_human_count("1K").unwrap(), 1_000);
        assert_eq!(parse_human_count("1kb").unwrap(), 1_000);
        assert_eq!(parse_human_count("1KB").unwrap(), 1_000);
        assert_eq!(parse_human_count("100m").unwrap(), 100_000_000);
        assert_eq!(parse_human_count("100M").unwrap(), 100_000_000);
        assert_eq!(parse_human_count("100mb").unwrap(), 100_000_000);
        assert_eq!(parse_human_count("1g").unwrap(), 1_000_000_000);
        assert_eq!(parse_human_count("1G").unwrap(), 1_000_000_000);
        assert_eq!(parse_human_count("1gb").unwrap(), 1_000_000_000);
        assert_eq!(parse_human_count("384k").unwrap(), 384_000);
    }

    #[test]
    fn parse_human_count_binary_suffixes() {
        // Binary: ki=1,024  mi=1,048,576  gi=1,073,741,824 (case-insensitive)
        assert_eq!(parse_human_count("1ki").unwrap(), 1_024);
        assert_eq!(parse_human_count("1Ki").unwrap(), 1_024);
        assert_eq!(parse_human_count("1kib").unwrap(), 1_024);
        assert_eq!(parse_human_count("1KiB").unwrap(), 1_024);
        assert_eq!(parse_human_count("1mi").unwrap(), 1_048_576);
        assert_eq!(parse_human_count("1Mi").unwrap(), 1_048_576);
        assert_eq!(parse_human_count("1mib").unwrap(), 1_048_576);
        assert_eq!(parse_human_count("1MiB").unwrap(), 1_048_576);
        assert_eq!(parse_human_count("1gi").unwrap(), 1_073_741_824);
        assert_eq!(parse_human_count("1Gi").unwrap(), 1_073_741_824);
        assert_eq!(parse_human_count("1gib").unwrap(), 1_073_741_824);
    }

    #[test]
    fn parse_human_count_rejects_invalid_values() {
        assert!(parse_human_count("").is_err()); // empty
        assert!(parse_human_count("m").is_err()); // suffix without number
        assert!(parse_human_count("8XB").is_err()); // unknown suffix
        assert!(parse_human_count("8.5m").is_err()); // decimal not supported (en-US locale)
        assert!(parse_human_count("8b").is_err()); // 'b' is not a valid count suffix
    }

    #[test]
    fn parse_human_count_strips_separators() {
        // Underscore is always stripped (universal visual separator)
        assert_eq!(parse_human_count("100_000_000").unwrap(), 100_000_000);
        assert_eq!(parse_human_count("1_000").unwrap(), 1_000);
        assert_eq!(parse_human_count("100_000m").unwrap(), 100_000 * 1_000_000);
        // Leading/trailing underscores around the suffix are fine
        assert_eq!(parse_human_count("1_000k").unwrap(), 1_000_000);
    }

    #[test]
    fn parse_human_size_strips_separators() {
        // Underscore is always stripped
        assert_eq!(parse_human_size("8_388_608").unwrap(), 8_388_608);
        assert_eq!(parse_human_size("1_024k").unwrap(), 1_024_000);
        assert_eq!(parse_human_size("1_048_576b").unwrap(), 1_048_576);
    }

    // ------------------------------------------------------------------
    // parse_jobs — wraps parse_human_count with MAX_JOBS ceiling
    // ------------------------------------------------------------------

    #[test]
    fn parse_jobs_accepts_reasonable_values() {
        assert_eq!(parse_jobs("1").unwrap(), 1);
        assert_eq!(parse_jobs("32").unwrap(), 32);
        assert_eq!(parse_jobs("64").unwrap(), 64);
        assert_eq!(parse_jobs("256").unwrap(), 256);
        assert_eq!(parse_jobs("1000").unwrap(), 1000); // delete default
        assert_eq!(parse_jobs(&MAX_JOBS.to_string()).unwrap(), MAX_JOBS); // exactly at ceiling
    }

    #[test]
    fn parse_jobs_rejects_above_max() {
        assert!(parse_jobs(&(MAX_JOBS + 1).to_string()).is_err());
        assert!(parse_jobs("5000").is_err());
        assert!(parse_jobs("1m").is_err()); // 1,000,000 >> MAX_JOBS
    }

    // ------------------------------------------------------------------
    // --endpoints helper functions — pure string logic, no network access
    // ------------------------------------------------------------------

    #[test]
    fn build_s3_endpoint_uris_two_endpoints() {
        let uris = build_s3_endpoint_uris("10.9.0.17:9000,10.9.0.18:9000", "s3://mybucket/bench/")
            .unwrap();
        assert_eq!(uris.len(), 2);
        assert_eq!(uris[0], "s3://10.9.0.17:9000/mybucket/bench/");
        assert_eq!(uris[1], "s3://10.9.0.18:9000/mybucket/bench/");
    }

    #[test]
    fn build_s3_endpoint_uris_adds_trailing_slash() {
        // base URI without trailing slash should still produce correct endpoint URIs
        let uris = build_s3_endpoint_uris("10.9.0.17:9000", "s3://mybucket/prefix").unwrap();
        assert_eq!(uris[0], "s3://10.9.0.17:9000/mybucket/prefix/");
    }

    #[test]
    fn build_s3_endpoint_uris_ignores_whitespace_and_empty_entries() {
        let uris =
            build_s3_endpoint_uris(" 10.9.0.17:9000 , , 10.9.0.18:9000 ", "s3://bucket/").unwrap();
        // Empty entry between commas and surrounding whitespace should be stripped
        assert_eq!(uris.len(), 2);
        assert_eq!(uris[0], "s3://10.9.0.17:9000/bucket/");
        assert_eq!(uris[1], "s3://10.9.0.18:9000/bucket/");
    }

    #[test]
    fn build_s3_endpoint_uris_rejects_non_s3_uri() {
        assert!(build_s3_endpoint_uris("host:9000", "az://myaccount/mycontainer/").is_err());
        assert!(build_s3_endpoint_uris("host:9000", "file:///mnt/data/").is_err());
    }

    #[test]
    fn build_s3_endpoint_uris_rejects_empty_list() {
        assert!(build_s3_endpoint_uris("  ,  ", "s3://bucket/").is_err());
    }

    #[test]
    fn build_s3_host_root_uris_two_endpoints() {
        let uris = build_s3_host_root_uris("10.9.0.17:9000,10.9.0.18:9000").unwrap();
        assert_eq!(uris.len(), 2);
        assert_eq!(uris[0], "s3://10.9.0.17:9000/");
        assert_eq!(uris[1], "s3://10.9.0.18:9000/");
    }

    // ------------------------------------------------------------------
    // --endpoints CLI argument parsing — verifies clap wiring end-to-end
    // ------------------------------------------------------------------

    #[test]
    fn cli_put_endpoints_parsed_into_struct() {
        let cli = Cli::try_parse_from([
            "s3-cli",
            "put",
            "--endpoints=10.9.0.17:9000,10.9.0.18:9000",
            "s3://mybucket/bench/",
        ])
        .expect("should parse successfully");

        if let Command::Put { endpoints, .. } = cli.cmd {
            assert_eq!(
                endpoints.as_deref(),
                Some("10.9.0.17:9000,10.9.0.18:9000"),
                "--endpoints value should be passed through verbatim"
            );
        } else {
            panic!("expected Command::Put");
        }
    }

    #[test]
    fn cli_get_endpoints_parsed_into_struct() {
        let cli = Cli::try_parse_from([
            "s3-cli",
            "get",
            "--endpoints=10.9.0.17:9000,10.9.0.18:9000",
            "s3://mybucket/bench/",
        ])
        .expect("should parse successfully");

        if let Command::Get { endpoints, .. } = cli.cmd {
            assert_eq!(
                endpoints.as_deref(),
                Some("10.9.0.17:9000,10.9.0.18:9000"),
                "--endpoints value should be passed through verbatim"
            );
        } else {
            panic!("expected Command::Get");
        }
    }

    #[test]
    fn cli_put_endpoints_absent_when_not_provided() {
        let cli = Cli::try_parse_from(["s3-cli", "put", "s3://mybucket/bench/"])
            .expect("should parse successfully");

        if let Command::Put { endpoints, .. } = cli.cmd {
            assert!(
                endpoints.is_none(),
                "--endpoints should be None when not provided"
            );
        } else {
            panic!("expected Command::Put");
        }
    }
}
