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
//! ```

use anyhow::{bail, Context, Result};
use clap::{ArgAction, Parser, Subcommand};
use std::io::{self, Write, ErrorKind};
use std::path::PathBuf;
use std::str::FromStr; // For the custom S3Path type
use std::time::Instant;
use log::LevelFilter;
use log::info;
use tempfile::NamedTempFile;



// Import shared functions from the crate.
use s3dlio::{
    delete_objects, get_objects_parallel, parse_s3_uri, stat_object_uri, 
    put_objects_with_random_data_and_type, DEFAULT_OBJECT_SIZE, ObjectType,
    list_buckets,
    object_store::store_for_uri,
    mp,
    config::{DataGenMode, Config},
};

use s3dlio::s3_copy::{upload_files, download_objects};
use s3dlio::{init_op_logger, finalize_op_logger};
use s3dlio::progress::S3ProgressTracker;

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
    /// Upload local files (supports glob patterns) to S3, concurrently to jobs
    Upload {
        /// One or more local files or glob patterns ('*' and '?')
        #[arg(required = true)]
        files: Vec<PathBuf>,
        /// S3 prefix **ending with '/'**
        dest: String,
        /// Maximum parallel uploads
        #[arg(short = 'j', long = "jobs", default_value_t = 32)]
        jobs: usize,
        /// Create the bucket if it doesn’t exist
        #[arg(short = 'c', long = "create-bucket")]
        create_bucket: bool,
    },
    /// Download object(s) to named directory (uses globbing pattern match)
    Download {
        /// S3 URI – can be a full key or prefix/glob with '*' or '?'.
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

    /// List objects using generic storage URI (supports s3://, az://, file://, direct://)
    /// Example: s3dlio ls s3://bucket/prefix/ -r
    #[clap(name = "ls")]
    GenericList {
        /// Storage URI (e.g. s3://bucket/prefix/, az://account/container/, file:///path/)
        uri: String,

        /// List objects recursively
        #[clap(short, long)]
        recursive: bool,
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


/// Main CLI function
#[tokio::main]
async fn main() -> Result<()> {
    // Loads any variables from .env file that are not already set
    dotenvy::dotenv().ok();

    // Optionally, pre-check required AWS variables.
    if std::env::var("AWS_ACCESS_KEY_ID").is_err() || std::env::var("AWS_SECRET_ACCESS_KEY").is_err() {
        eprintln!("Error: Missing required environment variables. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (and optionally AWS_REGION) either in your environment or in a .env file.");
        std::process::exit(1);
    }


    let cli = Cli::parse();

    // Initialise logging once, based on how many `-v` flags were given:
    let level = match cli.verbose {
        0 => LevelFilter::Warn,   // no -v
        1 => LevelFilter::Info,   // -v
        _ => LevelFilter::Debug,  // -vv or more
    };
    env_logger::Builder::from_default_env()
        .filter_level(level)
        .init();


    // If set, start the op‑logger *before* the first S3 call
    if let Some(ref path) = cli.op_log {
        if let Err(e) = init_op_logger(path.to_string_lossy()) {
            eprintln!("failed to start op‑logger `{}`: {e}", path.display());
        }
    }

    match cli.cmd {
        // List all buckets command
        Command::ListBuckets => {
            list_buckets_cmd().await?;
        },

        // New, create-bucket command
        Command::CreateBucket { bucket_name } => {
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
    
        Command::Stat { uri } => stat_cmd(&uri)?,
    
        // New, with regex and recursion
        Command::Get { uri, jobs, concurrent, keylist, recursive } => {
            get_cmd(uri.as_deref(), jobs, concurrent, keylist.as_deref(), recursive)?
        }
        
        // Multi-process GET for maximum throughput
        Command::MpGet { uri, procs, jobs, num, size, template, interval } => {
            mp_get_cmd(&uri, procs, jobs, num, size, &template, interval)?
        }
    
        Command::Delete { uri, jobs, recursive } => delete_cmd(&uri, jobs, recursive)?,
    
    
        Command::Upload { files, dest, jobs, create_bucket } => {
            let spinner = S3ProgressTracker::spinner("UPLOAD");
            spinner.set_message(format!("Uploading files to {}...", dest));
            let result = upload_files(&dest, &files, jobs, create_bucket);
            spinner.finish_with_message("Upload complete!");
            result?
        }
    
        Command::Download { src, dest_dir, jobs, recursive } => {
            let spinner = S3ProgressTracker::spinner("DOWNLOAD");
            spinner.set_message(format!("Downloading from {} to {:?}...", src, dest_dir));
            let result = download_objects(&src, &dest_dir, jobs, recursive);
            spinner.finish_with_message("Download complete!");
            result?
        }
    
        Command::Put { uri_prefix, create_bucket_flag, num, template, jobs, size, object_type, dedup_f, compress_f, data_gen_mode, chunk_size } => {
            let (bucket, _prefix) = parse_s3_uri(&uri_prefix)?;
            if create_bucket_flag {
                if let Err(e) = s3dlio::s3_utils::create_bucket(&bucket) {
                    eprintln!("Warning: failed to create bucket {}: {}", bucket, e);
                }
            }
    
            put_many_cmd(&uri_prefix, num, &template, jobs, size, object_type, dedup_f, compress_f, data_gen_mode, chunk_size)?

        }

        Command::GenericList { uri, recursive } => {
            generic_list_cmd(&uri, recursive).await?
        }

    } // End of match cli.cmd

    // If set, finalize the op‑logger 
    if cli.op_log.is_some() {
        finalize_op_logger();
    }

    Ok(())
}

/// Generic list command that works with any storage backend
async fn generic_list_cmd(uri: &str, recursive: bool) -> Result<()> {
    let store = store_for_uri(uri)?;
    let keys = store.list(uri, recursive).await?;

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

/// Stat command: provide info on a single object
fn stat_cmd(uri: &str) -> Result<()> {
    let os = stat_object_uri(uri)?;
    safe_println!("Size            : {}", os.size);
    safe_println!("LastModified    : {:?}", os.last_modified);
    safe_println!("ETag            : {:?}", os.e_tag);
    safe_println!("Content-Type    : {:?}", os.content_type);
    safe_println!("Content-Language: {:?}", os.content_language);
    safe_println!("StorageClass    : {:?}", os.storage_class);
    safe_println!("VersionId       : {:?}", os.version_id);
    safe_println!("ReplicationStat : {:?}", os.replication_status);
    safe_println!("SSE             : {:?}", os.server_side_encryption);
    safe_println!("SSE-KMS Key ID  : {:?}", os.ssekms_key_id);
    if !os.metadata.is_empty() {
        safe_println!("User Metadata:");
        for (k,v) in os.metadata {
            safe_println!("  {} = {}", k, v);
        }
    }
    Ok(())
}

/// Get command: downloads objects matching a key, prefix, or pattern.
fn get_cmd(uri: Option<&str>, jobs: usize, concurrent: Option<usize>, keylist: Option<&std::path::Path>, recursive: bool) -> Result<()> {
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
        
        let t0 = Instant::now();
        let results = get_objects_parallel(&uris, concurrency)?;
        let dt = t0.elapsed();
        
        let total_bytes: usize = results.iter().map(|(_, bytes)| bytes.len()).sum();
        
        // Report benchmark results (data is immediately discarded for benchmarking)
        let throughput_mb_s = (total_bytes as f64 / (1024.0 * 1024.0)) / dt.as_secs_f64();
        println!("Benchmarked {} objects: {} bytes in {:.3}s ({:.2} MB/s)", 
                 results.len(), total_bytes, dt.as_secs_f64(), throughput_mb_s);
        
        return Ok(());
    }
    
    // Original URI-based mode
    let uri_str = uri.ok_or_else(|| anyhow::anyhow!("Either --uri or --keylist must be provided"))?;
    let (bucket, mut key_pattern) = s3dlio::parse_s3_uri(uri_str)?;

    /*
     * Old
     *
    // If the pattern is not a glob/regex and not a prefix, it's a single key.
    // Otherwise, use list_objects to find all matching keys.
    let keys_to_get = if !key_pattern.contains('*') && !key_pattern.contains('?') && !key_pattern.ends_with('/') {
        vec![key_pattern]
    } else {
        s3dlio::s3_utils::list_objects(&bucket, &key_pattern, recursive)?
    };
    */
    
    // New, simpler
    // If the path ends with a slash, automatically append a wildcard to search inside it.
    if key_pattern.ends_with('/') {
        key_pattern.push_str(".*");
    }

    // List the objects using the potentially modified pattern.
    let keys_to_get = s3dlio::s3_utils::list_objects(&bucket, &key_pattern, recursive)?;

    if keys_to_get.is_empty() {
        bail!("No objects match pattern '{}' in bucket '{}'", uri_str, bucket);
    }

    let uris: Vec<String> = keys_to_get.into_iter().map(|k| format!("s3://{}/{}", bucket, k)).collect();
    
    // Create progress bar for download operation
    let progress = S3ProgressTracker::new("GET", uris.len() as u64, 0); // We don't know total bytes yet
    progress.progress_bar.set_message(format!("Preparing to download {} objects...", uris.len()));

    let t0 = Instant::now();
    let results = get_objects_parallel(&uris, jobs)?;
    
    let total_bytes: usize = results.iter().map(|(_, bytes)| bytes.len()).sum();
    let dt = t0.elapsed();

    // Finish the progress bar with completion stats
    progress.finish("Download", total_bytes as u64, dt);

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
fn delete_cmd(uri: &str, _jobs: usize, recursive: bool) -> Result<()> {
    let (bucket, mut key_pattern) = s3dlio::parse_s3_uri(uri)?;

    /*
     * Old
     *
    // If the pattern is not a glob/regex and not a prefix, it's a single key.
    // Otherwise, use list_objects to find all matching keys.
    let keys_to_delete = if !key_pattern.contains('*') && !key_pattern.contains('?') && !key_pattern.ends_with('/') {
        vec![key_pattern]
    } else {
        s3dlio::s3_utils::list_objects(&bucket, &key_pattern, recursive)?
    };
    */

    // New, simpler
    // If the path ends with a slash, automatically append a wildcard to search inside it.
    if key_pattern.ends_with('/') {
        key_pattern.push_str(".*");
    }

    // List the objects using the potentially modified pattern.
    let keys_to_delete = s3dlio::s3_utils::list_objects(&bucket, &key_pattern, recursive)?;


    if keys_to_delete.is_empty() {
        bail!("No objects to delete under the specified URI");
    }

    eprintln!("Deleting {} objects…", keys_to_delete.len());
    delete_objects(&bucket, &keys_to_delete)?;
    eprintln!("Done.");
    Ok(())
}



/// Put command supports 1 or more objects, also takes our ObjectType
fn put_many_cmd(uri_prefix: &str, num: usize, template: &str, jobs: usize, size: usize, object_type: s3dlio::ObjectType, dedup_f: usize, compress_f: usize, data_gen_mode: DataGenMode, chunk_size: usize) -> Result<()> {
    // Parse the prefix into bucket and key prefix.
    let (bucket, mut prefix) = parse_s3_uri(uri_prefix)?;
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
        let full_uri = format!("s3://{}/{}{}", bucket, prefix, object_name);
        uris.push(full_uri);
    }

    // Find the lesser of the number of jobs or number of objects
    let effective_jobs = std::cmp::min(jobs, num);
    let total_bytes = num * size;
    
    // Create progress bar for upload operation
    let progress = S3ProgressTracker::new("PUT", num as u64, total_bytes as u64);
    progress.progress_bar.set_message(format!("Preparing to upload {} objects...", num));

    let t0 = Instant::now();
    // Create config with specified data generation mode and chunk size
    let config = Config::new_with_defaults(object_type, 1, size, dedup_f, compress_f)
        .with_data_gen_mode(data_gen_mode)
        .with_chunk_size(chunk_size);
    put_objects_with_random_data_and_type(&uris, size, effective_jobs, config)?;
    let elapsed = t0.elapsed();

    // Finish the progress bar with completion stats
    progress.finish("Upload", total_bytes as u64, elapsed);
    Ok(())
}

