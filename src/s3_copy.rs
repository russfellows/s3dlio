// src/s3_copy.rs

use anyhow::{bail, Result};
use aws_sdk_s3::primitives::ByteStream;
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use glob::glob;
use tracing::{info, debug, warn, error};
//use regex::Regex;
use std::{fs, path::{Path}, sync::Arc};
use tokio::{fs as async_fs, sync::Semaphore};

use crate::s3_utils::{
    parse_s3_uri, create_bucket, list_objects, get_object, put_object_async,
};
use crate::progress::ProgressCallback;

// ──────────────────────────────────────────────────────────────────────────────
// Functions that leverage existing s3_utils code
// Will upload local files → S3  &  download S3 → local files
// ──────────────────────────────────────────────────────────────────────────────

/// Upload local files (supports globs) into `s3://bucket/prefix/…`,
/// streaming each file off disk with up to `max_in_flight` tasks.
/// If `do_create_bucket` is true, we attempt to create the bucket first.
pub async fn upload_files<P: AsRef<Path>>(  
    dest_prefix: &str,
    patterns: &[P],
    max_in_flight: usize,
    do_create_bucket: bool,
    progress_callback: Option<Arc<ProgressCallback>>,
) -> Result<()> {
    let (bucket, key_prefix) = parse_s3_uri(dest_prefix)?;
    // Allow empty prefix (bucket root) or a non-empty prefix that ends with '/'
    if !key_prefix.is_empty() && !key_prefix.ends_with('/') {
        bail!("dest_prefix must end with '/' if specifying a non-root prefix");
    }

    // Expand globs and single paths
    let mut paths = Vec::new();
    for pat in patterns {
        let s = pat.as_ref().to_string_lossy();
        if s.contains('*') || s.contains('?') {
            for entry in glob(&s)? {
                match entry {
                    Ok(pb) => paths.push(pb),
                    Err(e) => warn!("Glob error for pattern {}: {}", s, e),
                }
            }
        } else {
            paths.push(pat.as_ref().to_path_buf());
        }
    }
    if paths.is_empty() {
        bail!("No files matched for upload");
    }

    if do_create_bucket {
        create_bucket(&bucket)?;
    }

    // Cap the number of concurrent tasks to the number of items
    let effective_jobs = std::cmp::min(max_in_flight, paths.len());

    // INFO: start of bulk upload
    info!(
        "Starting upload of {} file(s) to {} (jobs={})",
        paths.len(),
        dest_prefix,
        effective_jobs
    );
    // DEBUG: details of create_bucket and requested jobs
    debug!(
        "upload_files debug: create_bucket={}, requested_jobs={}",
        do_create_bucket,
        max_in_flight
    );

    let sem = Arc::new(Semaphore::new(effective_jobs));
    let result = {
        let mut futs = FuturesUnordered::new();
        for path in paths.clone() {
            debug!("queueing upload for {:?}", path);
            let sem = sem.clone();
            let bucket = bucket.clone();
            let progress = progress_callback.clone();
            let fname = path.file_name()
                .ok_or_else(|| anyhow::anyhow!("Bad path {:?}", path))?
                .to_string_lossy();
            let key = format!("{}{}", key_prefix, fname);

            // Get file size for progress tracking
            let file_size = fs::metadata(&path)?.len();

            futs.push(tokio::spawn(async move {
                debug!("starting upload of {:?} → s3://{}/{}", path, bucket, key);
                let _permit = sem.acquire_owned().await.unwrap();
                let body = ByteStream::from_path(&path).await?;
                put_object_async(&bucket, &key, &body.collect().await?.into_bytes()).await?;
                debug!("finished upload of {:?} → s3://{}/{}", path, bucket, key);
                
                // Update progress if callback provided
                if let Some(ref progress) = progress {
                    progress.object_completed(file_size);
                }
                
                Ok::<(), anyhow::Error>(())
            }));
        }

        while let Some(join_res) = futs.next().await {
            join_res??;
        }
        Ok(())
    };

    // INFO: result of bulk upload
    match &result {
        Ok(()) => info!(
            "Finished upload of {} file(s) to {}",
            paths.len(),
            dest_prefix
        ),
        Err(e) => error!("Upload failed: {}", e),
    }
    result
}


/// Download one key, every key under a prefix, or glob/regex-match objects into `dest_dir/`,
/// creating files with their original basenames, up to `max_in_flight` at once.
pub async fn download_objects(
    src_uri: &str,
    dest_dir: &Path,
    max_in_flight: usize,
    recursive: bool,
    progress_callback: Option<Arc<ProgressCallback>>,
) -> Result<()> {
    let (bucket, mut key_pattern) = parse_s3_uri(src_uri)?;

    /*
     * Old
     *
    // If the pattern is not a glob/regex and not a prefix, it's a single key.
    // Otherwise, use our powerful list_objects function to find all matching keys.
    let keys = if !key_pattern.ends_with('/') && !key_pattern.contains('*') && !key_pattern.contains('?') {
        vec![key_pattern.clone()]
    } else {
        // Use the `recursive` parameter passed in from the caller
        list_objects(&bucket, &key_pattern, recursive)?
    };
    */

    // New, simpler
    // If the path ends with a slash, automatically append a wildcard to search inside it.
    if key_pattern.ends_with('/') {
        key_pattern.push_str(".*");
    }

    // List the objects using the potentially modified pattern.
    let keys = list_objects(&bucket, &key_pattern, recursive)?;

    if keys.is_empty() {
        bail!("No objects matched for download");
    }

    // sync—make sure the dir exists
    fs::create_dir_all(dest_dir)?;

    // Cap the number of concurrent tasks to the number of items
    let effective_jobs = std::cmp::min(max_in_flight, keys.len());

    // INFO: start of bulk download
    info!(
        "Starting download of {} object(s) from {} to {:?} (jobs={})",
        keys.len(),
        src_uri,
        dest_dir,
        effective_jobs
    );
    // DEBUG: details of requested jobs
    debug!("download_objects debug: requested_jobs={}", max_in_flight);

    let sem = Arc::new(Semaphore::new(effective_jobs));
    let result = {
        let mut futs = FuturesUnordered::new();
        for k in keys.clone() {
            let sem = sem.clone();
            let bucket = bucket.clone();
            let out_dir = dest_dir.to_path_buf();
            let progress = progress_callback.clone();

            futs.push(tokio::spawn(async move {
                debug!("starting download of s3://{}/{} → {:?}", bucket, k, out_dir);
                let _permit = sem.acquire_owned().await.unwrap();
                // Skip "directories" which are objects that end with a slash
                if k.ends_with('/') {
                    return Ok::<(), anyhow::Error>(());
                }
                let bytes = get_object(&bucket, &k).await?;
                let byte_count = bytes.len() as u64;
                let fname = Path::new(&k)
                    .file_name()
                    .ok_or_else(|| anyhow::anyhow!("Bad key: {}", k))?;
                let out_path = out_dir.join(fname);
                async_fs::write(&out_path, &bytes).await?;
                debug!("finished download of s3://{}/{} → {:?}", bucket, k, out_path);
                
                // Update progress if callback provided
                if let Some(ref progress) = progress {
                    progress.object_completed(byte_count);
                }
                
                Ok::<(), anyhow::Error>(())
            }));
        }

        while let Some(join_res) = futs.next().await {
            join_res??;
        }
        Ok(())
    };

    // INFO: result of bulk download
    match &result {
        Ok(()) => info!(
            "Finished download of {} object(s) to {:?}",
            keys.len(),
            dest_dir
        ),
        Err(e) => error!("Download failed: {}", e),
    }
    result
}


