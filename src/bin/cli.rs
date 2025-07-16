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
//! s3Rust-cli put         s3://bucket/key               # put one or more object
//! ```

use anyhow::{bail, Context, Result};
use clap::{ArgAction, Parser, Subcommand};
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Instant;
use log::LevelFilter;

// Import shared functions from the crate.
use s3dlio::{
    delete_objects, get_object_uri, get_objects_parallel, list_objects, parse_s3_uri,
    stat_object_uri, put_objects_with_random_data_and_type, DEFAULT_OBJECT_SIZE, ObjectType,
};

use s3dlio::s3_copy::{upload_files, download_objects};

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

    #[command(subcommand)]
    cmd: Command,
}


#[derive(Subcommand)]
enum Command {
    /// List keys that start with the given prefix.
    List {
        /// S3 URI (e.g. s3://bucket/prefix/)
        uri: String,
    },
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
    },
    /// Download one or many objects concurrently.
    Get {
        /// S3 URI – can be a full key or a prefix ending with `/`.
        uri: String,
        
        /// Maximum concurrent GET requests.
        #[arg(short = 'j', long = "jobs", default_value_t = 64)]
        jobs: usize,
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
    },
}


/// Main CLI function
fn main() -> Result<()> {
    // Loads any variables from .env file that are not already set
    dotenvy::dotenv().ok();

    // Optionally, pre-check required AWS variables.
    if std::env::var("AWS_ACCESS_KEY_ID").is_err() || std::env::var("AWS_SECRET_ACCESS_KEY").is_err() {
        eprintln!("Error: Missing required environment variables. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (and optionally AWS_REGION) either in your environment or in a .env file.");
        std::process::exit(1);
    }


    let cli = Cli::parse();

    // 1️⃣  Set up logging early – once per process.
    /*
    if cli.verbose {
        env_logger::Builder::from_default_env()
            //.filter_level(LevelFilter::Debug)   // Shows Error, Warn, Info and Debug levels 
            .filter_level(LevelFilter::Info)   // Shows Error, Warn and Info levels 
            .init();
    } else {
        env_logger::Builder::from_default_env()
            .filter_level(LevelFilter::Warn)   // Shows Error and Warn levels 
            .init();
    }
    */
    // Initialise logging once, based on how many `-v` flags were given:
    let level = match cli.verbose {
        0 => LevelFilter::Warn,   // no -v
        1 => LevelFilter::Info,   // -v
        _ => LevelFilter::Debug,  // -vv or more
    };
    env_logger::Builder::from_default_env()
        .filter_level(level)
        .init();


    match cli.cmd {
        Command::List { uri } => list_cmd(&uri),

        Command::Stat { uri } => stat_cmd(&uri),

        Command::Get { uri, jobs } => get_cmd(&uri, jobs),

        Command::Delete { uri, jobs } => delete_cmd(&uri, jobs),

        Command::Upload { files, dest, jobs, create_bucket } => {
            upload_files(&dest, &files, jobs, create_bucket)
        }

        Command::Download { src, dest_dir, jobs } => {
            download_objects(&src, &dest_dir, jobs)
        }

        Command::Put { uri_prefix, create_bucket_flag, num, template, jobs, size, object_type, dedup_f, compress_f } => {
            let (bucket, _prefix) = parse_s3_uri(&uri_prefix)?;
            if create_bucket_flag {
                if let Err(e) = s3dlio::s3_utils::create_bucket(&bucket) {
                    eprintln!("Warning: failed to create bucket {}: {}", bucket, e);
                }
            }
            
            put_many_cmd(&uri_prefix, num, &template, jobs, size, object_type, dedup_f, compress_f)

        }
    }
}


/// List command: supports glob matching on keys (after the last '/').
fn list_cmd(uri: &str) -> Result<()> {
    let (bucket, key_pattern) = parse_s3_uri(uri)?;

    let (effective_prefix, glob_pattern) = if let Some(pos) = key_pattern.rfind('/') {
        (&key_pattern[..=pos], &key_pattern[pos+1..])
    } else {
        ("", key_pattern.as_str())
    };

    let mut keys = list_objects(&bucket, effective_prefix)?;

/*
 * OLD: could cause listing of children it shouldn't
 *
    if glob_pattern.contains('*') {
         let regex_pattern = format!("^{}$", regex::escape(glob_pattern).replace("\\*", ".*"));
         let re = regex::Regex::new(&regex_pattern)
             .with_context(|| "Invalid regex pattern generated from glob")?;
         keys = keys.into_iter()
             .filter(|k| {
                 let basename = k.rsplit('/').next().unwrap_or(k);
                 re.is_match(basename)
             })
             .collect();
    }
*/

    // Always filter by the basename, whether exact or glob.
    let regex_pattern = format!(
        "^{}$",
        regex::escape(glob_pattern).replace("\\*", ".*")
    );
    let re = regex::Regex::new(&regex_pattern)
        .with_context(|| "Invalid regex pattern generated from glob")?;
    keys = keys
        .into_iter()
        .filter(|k| {
            let basename = k.rsplit('/').next().unwrap_or(k);
            re.is_match(basename)
        })
        .collect();

    let stdout = io::stdout();
    let mut out = stdout.lock();
    for key in &keys {
        if let Err(e) = writeln!(out, "{}", key) {
            if e.kind() == io::ErrorKind::BrokenPipe {
                return Ok(());
            } else {
                return Err(e.into());
            }
        }
    }
    writeln!(out, "\nTotal objects: {}", keys.len())?;
    Ok(())
}

/// Stat command: provide info on a single object
fn stat_cmd(uri: &str) -> Result<()> {
    let os = stat_object_uri(uri)?;
    println!("Size            : {}", os.size);
    println!("LastModified    : {:?}", os.last_modified);
    println!("ETag            : {:?}", os.e_tag);
    println!("Content-Type    : {:?}", os.content_type);
    println!("Content-Language: {:?}", os.content_language);
    println!("StorageClass    : {:?}", os.storage_class);
    println!("VersionId       : {:?}", os.version_id);
    println!("ReplicationStat : {:?}", os.replication_status);
    println!("SSE             : {:?}", os.server_side_encryption);
    println!("SSE-KMS Key ID  : {:?}", os.ssekms_key_id);
    if !os.metadata.is_empty() {
        println!("User Metadata:");
        for (k,v) in os.metadata {
            println!("  {} = {}", k, v);
        }
    }
    Ok(())
}


/// Get command: supports glob matching on keys (after the last '/').
fn get_cmd(uri: &str, jobs: usize) -> Result<()> {
    let (bucket, key_or_prefix) = parse_s3_uri(uri)?;

    if key_or_prefix.contains('*') {
        let (effective_prefix, glob_pattern) = if let Some(pos) = key_or_prefix.rfind('/') {
            (&key_or_prefix[..=pos], &key_or_prefix[pos+1..])
        } else {
            ("", key_or_prefix.as_str())
        };
        let mut keys = list_objects(&bucket, effective_prefix)?;
        if glob_pattern.contains('*') {
            let regex_pattern = format!("^{}$", regex::escape(glob_pattern).replace("\\*", ".*"));
            let re = regex::Regex::new(&regex_pattern)
                .with_context(|| "Invalid regex pattern generated from glob")?;
            keys = keys.into_iter()
                .filter(|k| {
                    let basename = k.rsplit('/').next().unwrap_or(k);
                    re.is_match(basename)
                })
                .collect();
        }
        if keys.is_empty() {
            bail!("No objects match pattern '{}' in bucket '{}'", key_or_prefix, bucket);
        }
        let uris: Vec<String> = keys.into_iter().map(|k| format!("s3://{}/{}", bucket, k)).collect();
        eprintln!("Fetching {} objects with {} jobs…", uris.len(), jobs);
        let t0 = Instant::now();
        let total_bytes: usize = get_objects_parallel(&uris, jobs)?
            .into_iter()
            .map(|(_, bytes)| bytes.len())
            .sum();
        let dt = t0.elapsed();
        eprintln!(
            "downloaded {:.2} MB in {:?} ({:.2} MB/s)",
            total_bytes as f64 / 1_048_576.0,
            dt,
            total_bytes as f64 / 1_048_576.0 / dt.as_secs_f64()
        );
    } else if key_or_prefix.ends_with('/') || key_or_prefix.is_empty() {
        let prefix = key_or_prefix;
        let keys = list_objects(&bucket, &prefix)?;
        if keys.is_empty() {
            bail!("No objects match prefix '{}' in bucket '{}'", prefix, bucket);
        }
        let uris: Vec<String> = keys.into_iter().map(|k| format!("s3://{}/{}", bucket, k)).collect();
        eprintln!("Fetching {} objects with {} jobs…", uris.len(), jobs);
        let t0 = Instant::now();
        let total_bytes: usize = get_objects_parallel(&uris, jobs)?
            .into_iter()
            .map(|(_, bytes)| bytes.len())
            .sum();
        let dt = t0.elapsed();
        eprintln!(
            "downloaded {:.2} MB in {:?} ({:.2} MB/s)",
            total_bytes as f64 / 1_048_576.0,
            dt,
            total_bytes as f64 / 1_048_576.0 / dt.as_secs_f64()
        );
    } else {
        let full_uri = format!("s3://{}/{}", bucket, key_or_prefix);
        let t0 = Instant::now();
        let bytes = get_object_uri(&full_uri)?;
        println!(
            "downloaded {} bytes in {:?} ({:.2} MB/s)",
            bytes.len(),
            t0.elapsed(),
            bytes.len() as f64 / 1_048_576.0 / t0.elapsed().as_secs_f64()
        );
    }
    Ok(())
}


/// Delete command: supports glob matching on keys (after the last '/').
fn delete_cmd(uri: &str, _jobs: usize) -> Result<()> {
    let (bucket, key_or_pattern) = parse_s3_uri(uri)?;

    let keys_to_delete = if key_or_pattern.contains('*') {
        let (effective_prefix, glob_pattern) = if let Some(pos) = key_or_pattern.rfind('/') {
            (&key_or_pattern[..=pos], &key_or_pattern[pos+1..])
        } else {
            ("", key_or_pattern.as_str())
        };

        let mut keys = list_objects(&bucket, effective_prefix)?;
        if glob_pattern.contains('*') {
            let regex_pattern = format!("^{}$", regex::escape(glob_pattern).replace("\\*", ".*"));
            let re = regex::Regex::new(&regex_pattern)
                .with_context(|| "Invalid regex pattern generated from glob")?;
            keys = keys.into_iter()
                .filter(|k| {
                    let basename = k.rsplit('/').next().unwrap_or(k);
                    re.is_match(basename)
                })
                .collect();
        }
        keys
    } else if key_or_pattern.ends_with('/') || key_or_pattern.is_empty() {
        list_objects(&bucket, &key_or_pattern)?
    } else {
        vec![key_or_pattern]
    };

    if keys_to_delete.is_empty() {
        bail!("No objects to delete under the specified URI");
    }

    eprintln!("Deleting {} objects…", keys_to_delete.len());
    delete_objects(&bucket, &keys_to_delete)?;
    eprintln!("Done.");
    Ok(())
}


/// Put command supports 1 or more objects, also takes our ObjectType
fn put_many_cmd(uri_prefix: &str, num: usize, template: &str, jobs: usize, size: usize, object_type: s3dlio::ObjectType, dedup_f: usize, compress_f: usize) -> Result<()> {
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
    let t0 = Instant::now();

    put_objects_with_random_data_and_type(&uris, size, effective_jobs, object_type, dedup_f, compress_f)?;

    let elapsed = t0.elapsed();
    let total_bytes = num * size;
    let mb_total = total_bytes as f64 / (1024.0 * 1024.0);
    let throughput = mb_total / elapsed.as_secs_f64();
    let objects_per_sec = num as f64 / elapsed.as_secs_f64();
    println!(
        "Uploaded {} objects (total {} bytes) in {:?} ({:.2} objects/s, {:.2} MB/s)",
        num, total_bytes, elapsed, objects_per_sec, throughput
    );
    Ok(())
}

