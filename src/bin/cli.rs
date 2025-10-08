//
// Copyright, 2025.  Signal65 / Futurum Group.
// 
//! CLI supporting `list`, `get`, `delete`, `put`, and `putmany`.
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
use std::io::{self, Write, ErrorKind};
use std::path::{Path, PathBuf};
use std::str::FromStr; // For the custom S3Path type
use std::time::Instant;
use std::sync::Arc;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;
use tempfile::NamedTempFile;
use glob;
use futures_util::stream::{FuturesUnordered, StreamExt};



// Import shared functions from the crate.
use s3dlio::{
    parse_s3_uri,
    DEFAULT_OBJECT_SIZE, ObjectType,
    list_buckets,
    object_store::store_for_uri_with_logger,
    mp,
    config::{DataGenMode, Config},
    data_gen::generate_object,
};

use s3dlio::{generic_upload_files, generic_download_objects};
use s3dlio::{init_op_logger, finalize_op_logger, global_logger};
use s3dlio::progress::{S3ProgressTracker, ProgressCallback};

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



// --- Define the S3Path struct for clap to use as a custom parser.
// This struct neatly encapsulates the bucket and key.
#[derive(Clone, Debug)]
struct S3Path {
    bucket: String,
    key: String,
}

impl S3Path {
    fn bucket(&self) -> &str {
        &self.bucket
    }

    fn key(&self) -> &str {
        &self.key
    }
}

/// Implement `FromStr` so that `clap` can parse a string like "s3://bucket/key"
/// directly into our `S3Path` struct.
impl FromStr for S3Path {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (bucket, key) = parse_s3_uri(s)?;
        Ok(S3Path {
            bucket,
            key,
        })
    }
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

    /// List all S3 buckets in the account.
    ListBuckets,

    /// Create a new S3 bucket.
    CreateBucket {
        /// The name of the bucket to create (e.g., my-new-bucket)
        bucket_name: String,
    },

    /// Delete an S3 bucket. The bucket must be empty.
    DeleteBucket {
        /// The name of the bucket to delete (e.g., my-old-bucket)
        bucket_name: String,
    },

    /// List objects in an S3 path, the final part of path is treated as a regex.
    /// Example: s3-cli list s3://my-bucket/data/.*\\.csv
    List {
        #[clap(value_parser)]
        s3_path: S3Path,

        /// List objects recursively
        #[clap(short, long)]
        recursive: bool,
    },
    /*
     * Old
     *
    /// List keys that start with the given prefix.
    List {
        /// S3 URI (e.g. s3://bucket/prefix/)
        uri: String,
    },
    */
    /// Stat object, show size & last modify date of a single object 
    Stat {
        /// Full S3 URI (e.g. s3://bucket/prefix/key)
        uri: String,
    },
    /// Delete one object or every object that matches the prefix.
    Delete {
        /// S3 URI (single key or prefix ending with `/`).
        uri: String,

        /// Batch size (number of parallel delete calls).
        #[arg(short = 'j', long = "jobs", default_value_t = 1000)]
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
        
        /// Maximum concurrent GET requests.
        #[arg(short = 'j', long = "jobs", default_value_t = 64)]
        jobs: usize,
        
        /// Alternative: maximum concurrent GET requests (for mp compatibility).
        #[arg(long = "concurrent")]
        concurrent: Option<usize>,
        
        /// File containing list of S3 URIs to download (one per line).
        #[arg(long = "keylist")]
        keylist: Option<PathBuf>,
        
        /// Perform the operation recursively.
        #[clap(short, long)]
        recursive: bool,
    },
    /// Multi-process GET for maximum throughput (warp-level performance).
    MpGet {
        /// S3 URI prefix for objects to download (e.g. s3://bucket/prefix/)
        uri: String,
        
        /// Number of worker processes to spawn
        #[arg(short = 'p', long = "procs", default_value_t = 4)]
        procs: usize,
        
        /// Concurrent operations per worker process
        #[arg(short = 'j', long = "jobs", default_value_t = 64)]
        jobs: usize,
        
        /// Number of objects to download (for testing/benchmarking)
        #[arg(short = 'n', long = "num", default_value_t = 1000)]
        num: usize,
        
        /// Object size in bytes (for generated object names)
        #[arg(short = 's', long = "size", default_value_t = 1048576)]
        size: usize,
        
        /// Template for object names (use {} for number placeholder)
        #[arg(short = 't', long = "template", default_value = "object_{}.dat")]
        template: String,
        
        /// Progress reporting interval in seconds
        #[arg(short = 'i', long = "interval", default_value_t = 1.0)]
        interval: f64,
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
        #[arg(short = 'j', long = "jobs", default_value_t = 32)]
        jobs: usize,

        /// Number of objects to create and upload.
        #[arg(short = 'n', long = "num", default_value_t = 1)]
        num: usize,

        /// Specify Type of object to generate: 
        #[arg( short = 'o', long = "object-type", value_enum, ignore_case = true, default_value_t = ObjectType::Raw)] // Without value_parser [] values are case insensitive
        object_type: ObjectType,

        /// Object size in bytes (default 20 MB).
        #[arg(short = 's', long = "size", default_value_t = DEFAULT_OBJECT_SIZE)]
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
    },
    /// Upload local files to any storage backend (S3, Azure, GCS, file://, direct://), supports glob and regex patterns
    Upload {
        /// One or more local files, directories, glob patterns ('*','?'), or regex patterns
        #[arg(required = true)]
        files: Vec<PathBuf>,
        /// Destination URI (s3://bucket/, az://container/, gs://bucket/, file:///path/, direct:///path/) **ending with '/'**
        dest: String,
        /// Maximum parallel uploads
        #[arg(short = 'j', long = "jobs", default_value_t = 32)]
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
        /// Maximum parallel downloads
        #[arg(short = 'j', long = "jobs", default_value_t = 64)]
        jobs: usize,
        /// Download recursively
        #[clap(short, long)] 
        recursive: bool,     
    },

    /// List objects using generic storage URI (supports s3://, az://, gs://, file://, direct://)
    /// Example: s3dlio ls s3://bucket/prefix/ -r
    /// Example: s3dlio ls gs://bucket/prefix/ -r
    #[clap(name = "ls")]
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
    },

    /// Generate NVIDIA DALI-compatible index file for TFRecord files
    /// Format: "{offset} {size}\n" (space-separated ASCII text)
    /// Example: s3dlio tfrecord-index train.tfrecord train.tfrecord.idx
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

/// List all S3 buckets in the account.
async fn list_buckets_cmd() -> Result<()> {
    safe_println!("Listing all S3 buckets...");
    
    let buckets = list_buckets()?;
    
    if buckets.is_empty() {
        safe_println!("No buckets found in this S3 account.");
        return Ok(());
    }
    
    safe_println!("\nFound {} bucket(s):", buckets.len());
    safe_println!("{:<30} {}", "Bucket Name", "Creation Date");
    safe_println!("{}", "-".repeat(60));
    
    for bucket in buckets {
        safe_println!("{:<30} {}", bucket.name, bucket.creation_date);
    }
    
    Ok(())
}


/// Check if AWS credentials are available for S3 operations
fn check_aws_credentials() -> Result<()> {
    if std::env::var("AWS_ACCESS_KEY_ID").is_err() || std::env::var("AWS_SECRET_ACCESS_KEY").is_err() {
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

/// Main CLI function
#[tokio::main]
async fn main() -> Result<()> {
    // Loads any variables from .env file that are not already set
    dotenvy::dotenv().ok();

    let cli = Cli::parse();

    // Initialise logging once, based on how many `-v` flags were given:
    // Initialize tracing subscriber based on verbosity
    let filter = match cli.verbose {
        0 => "warn",        // no -v: WARN level
        1 => "info",        // -v: INFO level
        _ => "debug",       // -vv or more: DEBUG level
    };
    
    // Initialize tracing with env filter (compatible with dl-driver/s3-bench)
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new(filter)))
        .with_target(false)
        .init();
    
    // Initialize tracing-log bridge to capture log crate messages from dependencies
    tracing_log::LogTracer::init().ok();


    // If set, start the op‑logger *before* the first S3 call
    if let Some(ref path) = cli.op_log {
        if let Err(e) = init_op_logger(path.to_string_lossy()) {
            eprintln!("failed to start op‑logger `{}`: {e}", path.display());
        }
    }

    match cli.cmd {
        // List all buckets command (S3-specific)
        Command::ListBuckets => {
            check_aws_credentials()?;
            list_buckets_cmd().await?;
        },

        // New, create-bucket command (S3-specific)
        Command::CreateBucket { bucket_name } => {
            check_aws_credentials()?;
            // Reliably clean the input to get a valid bucket name by
            // stripping the protocol prefix and any trailing slashes.
            let final_bucket_name = bucket_name
                .strip_prefix("s3://")
                .unwrap_or(&bucket_name)
                .trim_end_matches('/');

            info!("Attempting to create bucket: {}...", final_bucket_name);
            s3dlio::s3_utils::create_bucket(final_bucket_name)?;
            safe_println!("Successfully created or verified bucket '{}'.", final_bucket_name);
        },

        Command::DeleteBucket { bucket_name } => {
            check_aws_credentials()?;
            // Reliably clean the input to get a valid bucket name
            let final_bucket_name = bucket_name
                .strip_prefix("s3://")
                .unwrap_or(&bucket_name)
                .trim_end_matches('/');

            info!("Attempting to delete bucket: {}...", final_bucket_name);
            s3dlio::s3_utils::delete_bucket(final_bucket_name)?;
            safe_println!("Successfully deleted bucket '{}'.", final_bucket_name);
        }

        // --- Update the `List` command handler.
        // It now unpacks `s3_path` and `recursive` and calls our new backend function.
        Command::List { s3_path, recursive } => {
            check_aws_credentials()?;  // S3-only command
            // Deprecation warning
            eprintln!("WARNING: The 'list' command is deprecated and S3-only.");
            eprintln!("Please use 'ls' for universal multi-backend support:");
            eprintln!("  s3-cli ls s3://bucket/prefix/ -r");
            eprintln!("  s3-cli ls s3://bucket/prefix/ -p '.*\\.txt$'  # with regex pattern");
            eprintln!("This command will be removed in v0.9.0.");
            eprintln!();
            
            let keys = s3dlio::s3_utils::list_objects(s3_path.bucket(), s3_path.key(), recursive)?;
    
            let stdout = io::stdout();
            let mut out = stdout.lock();
            for key in &keys {
                if let Err(e) = writeln!(out, "{}", key) {
                    if e.kind() == io::ErrorKind::BrokenPipe {
                        return Ok(()); // Handle cases like piping to `head`
                    } else {
                        return Err(e.into());
                    }
                }
            }
            writeln!(out, "\nTotal objects: {}", keys.len())?;
        }
    
        Command::Stat { uri } => {
            // Only check AWS credentials if using S3
            if requires_aws_credentials(&uri) {
                check_aws_credentials()?;
            }
            stat_cmd(&uri).await?
        },
    
        // New, with regex and recursion
        Command::Get { uri, jobs, concurrent, keylist, recursive } => {
            // Only check AWS credentials if URI is S3
            if let Some(uri_str) = &uri {
                if requires_aws_credentials(uri_str) {
                    check_aws_credentials()?;
                }
            }
            get_cmd(uri.as_deref(), jobs, concurrent, keylist.as_deref(), recursive).await?
        }
        
        // Multi-process GET for maximum throughput
        Command::MpGet { uri, procs, jobs, num, size, template, interval } => {
            // Check AWS credentials for S3 URIs
            if requires_aws_credentials(&uri) {
                check_aws_credentials()?;
            }
            mp_get_cmd(&uri, procs, jobs, num, size, &template, interval)?
        }
    
        Command::Delete { uri, jobs, recursive, pattern } => {
            // Check AWS credentials for S3 URIs
            if requires_aws_credentials(&uri) {
                check_aws_credentials()?;
            }
            delete_cmd(&uri, jobs, recursive, pattern.as_deref()).await?
        },
    
    
        Command::Upload { files, dest, jobs, create_bucket } => {
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
            let progress_tracker = Arc::new(S3ProgressTracker::new("UPLOAD", file_count, total_bytes));
            let progress_callback = Arc::new(ProgressCallback::new(progress_tracker.clone(), file_count));

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
                        let parts: Vec<&str> = dest.trim_start_matches("az://").trim_start_matches("azure://").split('/').collect();
                        if let Some(container) = parts.get(0) {
                            if let Err(e) = store.create_container(container).await {
                                warn!("Failed to create container {}: {}", container, e);
                            }
                        }
                    } else if dest.starts_with("gs://") || dest.starts_with("gcs://") {
                        // For GCS, extract bucket name
                        let parts: Vec<&str> = dest.trim_start_matches("gs://").trim_start_matches("gcs://").split('/').collect();
                        if let Some(bucket) = parts.get(0) {
                            if let Err(e) = store.create_container(bucket).await {
                                warn!("Failed to create GCS bucket {}: {}", bucket, e);
                            }
                        }
                    }
                    // File backends don't need bucket creation, directories are created automatically
                }
            }

            let t0 = Instant::now();
            let result = generic_upload_files(&dest, &files, jobs, Some(progress_callback)).await;
            let elapsed = t0.elapsed();

            // Finish progress bar
            progress_tracker.finish("Upload", total_bytes, elapsed);
            
            result?
        }
    
        Command::Download { src, dest_dir, jobs, recursive } => {
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
            let progress_tracker = Arc::new(S3ProgressTracker::new("DOWNLOAD", keys.len() as u64, 0));
            let progress_callback = Arc::new(ProgressCallback::new(progress_tracker.clone(), keys.len() as u64));

            let t0 = Instant::now();
            let result = generic_download_objects(&src, &dest_dir, jobs, recursive, Some(progress_callback.clone())).await;
            let elapsed = t0.elapsed();

            // Get final byte count from progress callback and update the progress bar total
            let total_bytes = progress_callback.bytes_transferred.load(std::sync::atomic::Ordering::Relaxed);
            progress_callback.update_total_bytes(total_bytes);
            progress_tracker.finish("Download", total_bytes, elapsed);
            
            result?
        }
    
        Command::Put { uri_prefix, create_bucket_flag, num, template, jobs, size, object_type, dedup_f, compress_f, data_gen_mode, chunk_size } => {
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
    
            put_many_cmd(&uri_prefix, num, &template, jobs, size, object_type, dedup_f, compress_f, data_gen_mode, chunk_size).await?

        }

        Command::GenericList { uri, recursive, pattern } => {
            // Check AWS credentials only for S3 URIs
            if requires_aws_credentials(&uri) {
                check_aws_credentials()?;
            }
            generic_list_cmd(&uri, recursive, pattern.as_deref()).await?
        }

        Command::TfrecordIndex { tfrecord_path, index_path } => {
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
async fn generic_list_cmd(uri: &str, recursive: bool, pattern: Option<&str>) -> Result<()> {
    use regex::Regex;
    
    let logger = global_logger();
    let store = store_for_uri_with_logger(uri, logger)?;
    let mut keys = store.list(uri, recursive).await?;

    // Apply client-side regex filtering if pattern provided
    if let Some(pat) = pattern {
        let re = Regex::new(pat)
            .with_context(|| format!("Invalid regex pattern: '{}'", pat))?;
        keys.retain(|k| re.is_match(k));
    }

    let stdout = io::stdout();
    let mut out = stdout.lock();
    for key in &keys {
        if let Err(e) = writeln!(out, "{}", key) {
            if e.kind() == io::ErrorKind::BrokenPipe {
                return Ok(()); // Handle cases like piping to `head`
            } else {
                return Err(e.into());
            }
        }
    }
    writeln!(out, "
Total objects: {}", keys.len())?;
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
            let current_name = p.file_name()
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
    
    safe_println!("Successfully indexed {} records in {:.2?}", num_records, elapsed);
    safe_println!("Index file: {}", output_path.display());
    safe_println!("Format: NVIDIA DALI compatible (text, space-separated)");
    
    Ok(())
}

/// Stat command: provide info on a single object (universal, works with all backends)
async fn stat_cmd(uri: &str) -> Result<()> {
    let logger = global_logger();
    let store = store_for_uri_with_logger(uri, logger)?;
    let os = store.stat(uri).await?;
    
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
        for (k,v) in os.metadata {
            safe_println!("  {} = {}", k, v);
        }
    }
    Ok(())
}

/// Get command: downloads objects matching a key, prefix, or pattern.
async fn get_cmd(uri: Option<&str>, jobs: usize, concurrent: Option<usize>, keylist: Option<&std::path::Path>, recursive: bool) -> Result<()> {
    // Determine concurrency level (prefer concurrent over jobs for mp compatibility)
    let concurrency = concurrent.unwrap_or(jobs);
    
    // Handle keylist mode vs URI mode
    if let Some(keylist_path) = keylist {
        // Read URIs from keylist file
        let keylist_content = std::fs::read_to_string(keylist_path)
            .with_context(|| format!("Failed to read keylist file: {:?}", keylist_path))?;
        
        let uris: Vec<String> = keylist_content
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(|line| line.to_string())
            .collect();
            
        if uris.is_empty() {
            bail!("No URIs found in keylist file: {:?}", keylist_path);
        }
        
        println!("Processing {} URIs from keylist with concurrency {}", uris.len(), concurrency);
        
        // Create progress tracker for keylist mode
        let progress_tracker = Arc::new(S3ProgressTracker::new("GET", uris.len() as u64, 0));
        let progress_callback = Arc::new(ProgressCallback::new(progress_tracker.clone(), uris.len() as u64));
        
        let t0 = Instant::now();
        
        // Use universal ObjectStore interface for parallel downloads
        let sem = Arc::new(tokio::sync::Semaphore::new(concurrency));
        let mut futs = futures_util::stream::FuturesUnordered::new();
        let mut total_bytes = 0u64;
        
        for uri in &uris {
            let sem = sem.clone();
            let progress = progress_callback.clone();
            let uri = uri.clone();
            let logger = global_logger();
            
            futs.push(tokio::spawn(async move {
                let _permit = sem.acquire_owned().await.unwrap();
                let store = store_for_uri_with_logger(&uri, logger)?;
                let data = store.get(&uri).await?;
                let byte_count = data.len() as u64;
                
                // Update progress
                progress.object_completed(byte_count);
                
                Ok::<(String, u64), anyhow::Error>((uri, byte_count))
            }));
        }
        
        while let Some(result) = futs.next().await {
            let (uri, byte_count) = result??;
            total_bytes += byte_count;
            info!("Downloaded {} bytes from {}", byte_count, uri);
        }
        
        let dt = t0.elapsed();
        
        // Update final progress
        progress_callback.update_total_bytes(total_bytes);
        progress_tracker.finish("Download", total_bytes, dt);
        
        // Report benchmark results (data is immediately discarded for benchmarking)
        let throughput_mb_s = (total_bytes as f64 / (1024.0 * 1024.0)) / dt.as_secs_f64();
        println!("Benchmarked {} objects: {} bytes in {:.3}s ({:.2} MB/s)", 
                 uris.len(), total_bytes, dt.as_secs_f64(), throughput_mb_s);
        
        return Ok(());
    }
    
    // Original URI-based mode - use universal ObjectStore interface
    let uri_str = uri.ok_or_else(|| anyhow::anyhow!("Either --uri or --keylist must be provided"))?;
    
    // Only check AWS credentials if using S3 backend
    if requires_aws_credentials(uri_str) {
        check_aws_credentials()?;
    }

    // Use universal ObjectStore to list objects (works with all backends)
    let logger = global_logger();
    let store = store_for_uri_with_logger(uri_str, logger)?;
    let keys = store.list(uri_str, recursive).await?;

    if keys.is_empty() {
        bail!("No objects match pattern '{}'", uri_str);
    }

    // Create progress tracker
    let progress_tracker = Arc::new(S3ProgressTracker::new("GET", keys.len() as u64, 0));
    let progress_callback = Arc::new(ProgressCallback::new(progress_tracker.clone(), keys.len() as u64));

    let t0 = Instant::now();
    
    // Use universal ObjectStore interface for parallel downloads
    let sem = Arc::new(tokio::sync::Semaphore::new(jobs));
    let mut futs = futures_util::stream::FuturesUnordered::new();
    let mut total_bytes = 0u64;
    
    for uri in &keys {
        let sem = sem.clone();
        let progress = progress_callback.clone();
        let uri = uri.clone();
        let logger = global_logger();
        
        futs.push(tokio::spawn(async move {
            let _permit = sem.acquire_owned().await.unwrap();
            let store = store_for_uri_with_logger(&uri, logger)?;
            let data = store.get(&uri).await?;
            let byte_count = data.len() as u64;
            
            // Update progress
            progress.object_completed(byte_count);
            
            Ok::<(String, u64), anyhow::Error>((uri, byte_count))
        }));
    }
    
    while let Some(result) = futs.next().await {
        let (uri, byte_count) = result??;
        total_bytes += byte_count;
        info!("Downloaded {} bytes from {}", byte_count, uri);
    }
    
    let dt = t0.elapsed();

    // Update final progress and finish
    progress_callback.update_total_bytes(total_bytes);
    progress_tracker.finish("Download", total_bytes, dt);

    Ok(())
}

/// Multi-process GET command for maximum throughput
fn mp_get_cmd(uri: &str, procs: usize, jobs: usize, num: usize, _size: usize, template: &str, _interval: f64) -> Result<()> {
    let (bucket, key_prefix) = parse_s3_uri(uri)?;
    
    println!("Multi-process GET starting with {} processes, {} jobs per process", procs, jobs);
    println!("Target: {} objects from s3://{}/{}", num, bucket, key_prefix);
    
    // Generate object keys based on template and number
    let mut keys = Vec::new();
    for i in 0..num {
        let key = if template.contains("{}") {
            template.replace("{}", &i.to_string())
        } else {
            format!("{}{}", template, i)
        };
        keys.push(format!("s3://{}/{}{}", bucket, key_prefix, key));
    }
    
    // Create temporary keylist file
    let keylist_file = NamedTempFile::new()?;
    let keylist_path = keylist_file.path().to_path_buf();
    
    // Write keys to file
    {
        let mut file = std::fs::File::create(&keylist_path)?;
        for key in &keys {
            writeln!(file, "{}", key)?;
        }
    }
    
    // Get the current executable path for worker processes
    let current_exe = std::env::current_exe()?;
    
    // Configure multi-process GET (without oplog for now - will add proper zstd support later) 
    let config = mp::MpGetConfigBuilder::new()
        .procs(procs)
        .concurrent_per_proc(jobs)
        .keylist(keylist_path)
        .worker_cmd(current_exe.to_string_lossy().to_string())
        .passthrough_io(false)  // Disable debug output for clean results
        .build();
    
    // Run the multi-process operation
    let result = mp::run_get_shards(&config)?;
    
    // Print summary
    println!("\n=== Multi-Process GET Summary ===");
    println!("Total operations: {}", result.total_objects);
    println!("Total bytes: {}", result.total_bytes);
    println!("Duration: {:.2}s", result.elapsed_seconds);
    println!("Throughput: {:.2} MB/s", result.throughput_mbps());
    println!("Operations/sec: {:.2}", result.ops_per_second());
    
    println!("\nPer-worker performance:");
    for worker in &result.per_worker {
        let worker_throughput = if result.elapsed_seconds > 0.0 {
            (worker.bytes as f64 / (1024.0 * 1024.0)) / result.elapsed_seconds
        } else {
            0.0
        };
        println!("  Worker {}: {} ops, {} bytes, {:.2} MB/s", 
                 worker.worker_id, worker.objects, worker.bytes, worker_throughput);
    }
    
    Ok(())
}

/// Delete command: deletes objects matching a key, prefix, or pattern.
async fn delete_cmd(uri: &str, _jobs: usize, recursive: bool, pattern: Option<&str>) -> Result<()> {
    use s3dlio::object_store::store_for_uri_with_logger;
    use regex::Regex;
    use indicatif::{ProgressBar, ProgressStyle};
    
    // Use generic object store to delete objects (works with all backends)
    let logger = global_logger();
    let store = store_for_uri_with_logger(uri, logger)?;
    
    // If recursive or URI ends with '/', delete prefix; otherwise check if it's a pattern
    if recursive || uri.ends_with('/') {
        if let Some(pat) = pattern {
            // Delete with regex filter
            let mut keys = store.list(uri, recursive).await?;
            let re = Regex::new(pat)
                .with_context(|| format!("Invalid regex pattern: '{}'", pat))?;
            keys.retain(|k| re.is_match(k));
            
            if keys.is_empty() {
                eprintln!("No objects match pattern '{}' under prefix '{}'", pat, uri);
                return Ok(());
            }
            
            info!("Deleting {} objects matching pattern '{}' under prefix '{}'", keys.len(), pat, uri);
            eprintln!("Found {} objects matching pattern '{}'", keys.len(), pat);
            
            // Create progress bar for deletion
            let pb = ProgressBar::new(keys.len() as u64);
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
            
            pb.finish_with_message(format!("Deleted {} objects matching pattern", keys.len()));
            eprintln!("\nSuccessfully deleted {} objects matching pattern '{}'", keys.len(), pat);
        } else {
            // Delete entire prefix without filter - need to list first for progress
            info!("Listing objects under prefix: {}", uri);
            eprintln!("Listing objects to delete...");
            let keys = store.list(uri, recursive).await?;
            
            if keys.is_empty() {
                eprintln!("No objects found under prefix: {}", uri);
                return Ok(());
            }
            
            let total_count = keys.len();
            info!("Deleting {} objects under prefix '{}'", total_count, uri);
            eprintln!("Found {} objects to delete", total_count);
            
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
            eprintln!("\nSuccessfully deleted {} objects under prefix: {}", total_count, uri);
        }
    } else {
        // First, try to list objects with this URI as a prefix to see if it matches multiple objects
        let mut keys = store.list(uri, false).await?;
        
        // Apply regex filter if provided
        if let Some(pat) = pattern {
            let re = Regex::new(pat)
                .with_context(|| format!("Invalid regex pattern: '{}'", pat))?;
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
                eprintln!("Found {} objects matching prefix '{}' and pattern '{}'", total_count, uri, pat);
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
                eprintln!("\nSuccessfully deleted {} objects matching pattern '{}'", total_count, pat);
            } else {
                eprintln!("\nSuccessfully deleted {} objects", total_count);
            }
        }
    }
    
    Ok(())
}



/// Put command supports 1 or more objects, also takes our ObjectType
async fn put_many_cmd(uri_prefix: &str, num: usize, template: &str, jobs: usize, size: usize, object_type: s3dlio::ObjectType, dedup_f: usize, compress_f: usize, data_gen_mode: DataGenMode, chunk_size: usize) -> Result<()> {
    // Ensure prefix ends with '/' for consistent URI generation
    let mut prefix = uri_prefix.to_string();
    if !prefix.ends_with('/') {
        prefix.push('/');
    }
    
    // Generate the full list of URIs.
    let mut uris = Vec::with_capacity(num);

    // Now replace brackets with values
    for i in 0..num {
        // replace first {} with the index, second {} with the total count
        let object_name = template
            .replacen("{}", &i.to_string(), 1)
            .replacen("{}", &num.to_string(), 1);
        let full_uri = format!("{}{}", prefix, object_name);
        uris.push(full_uri);
    }

    // Find the lesser of the number of jobs or number of objects
    let effective_jobs = std::cmp::min(jobs, num);
    let total_bytes = num * size;
    
    // Create progress bar for upload operation
    let progress_tracker = Arc::new(S3ProgressTracker::new("PUT", num as u64, total_bytes as u64));
    let progress_callback = Arc::new(ProgressCallback::new(progress_tracker.clone(), num as u64));
    progress_tracker.progress_bar.set_message(format!("Preparing to upload {} objects...", num));

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
        
        futs.push(tokio::spawn(async move {
            let _permit = sem.acquire_owned().await.unwrap();
            let store = store_for_uri_with_logger(&uri, logger)?;
            store.put(&uri, &data).await?;
            let byte_count = data.len() as u64;
            
            // Update progress
            progress.object_completed(byte_count);
            
            Ok::<(String, u64), anyhow::Error>((uri, byte_count))
        }));
    }
    
    let mut total_uploaded_bytes = 0u64;
    while let Some(result) = futs.next().await {
        let (uri, byte_count) = result??;
        total_uploaded_bytes += byte_count;
        info!("Uploaded {} bytes to {}", byte_count, uri);
    }
    
    let elapsed = t0.elapsed();

    // Update final progress and finish
    progress_callback.update_total_bytes(total_uploaded_bytes);
    progress_tracker.finish("Upload", total_uploaded_bytes, elapsed);
    
    Ok(())
}

