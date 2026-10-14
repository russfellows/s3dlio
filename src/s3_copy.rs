// src/s3_copy.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use anyhow::{bail, Result};
use aws_sdk_s3::primitives::ByteStream;
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use glob::glob;
use tracing::{debug, error, info, warn};
//use regex::Regex;
use std::{fs, path::Path, sync::Arc};
use tokio::{fs as async_fs, sync::Semaphore};

use crate::delete_batch_plan::parse_s3_uri_for_global_client;
use crate::object_store::{store_for_uri, ObjectStore};
use crate::progress::ProgressCallback;
use crate::s3_utils::{
    create_bucket_for_uri_async, get_object, list_objects_with_pattern, parse_s3_uri_full,
    put_object_async,
};

// ──────────────────────────────────────────────────────────────────────────────
// Functions that leverage existing s3_utils code
// Will upload local files → S3  &  download S3 → local files
// ──────────────────────────────────────────────────────────────────────────────

/// Upload local files (supports globs) into `s3://bucket/prefix/…`,
/// streaming each file off disk with up to `max_in_flight` tasks.
/// If `do_create_bucket` is true, we attempt to create the bucket first.
///
/// `dest_prefix` may name a server (`s3://host:port/bucket/prefix/`): the bucket is
/// then created on, and the files are uploaded to, THAT server through a
/// per-endpoint client (issue #199); a portless IP address is an error.
pub async fn upload_files<P: AsRef<Path>>(
    dest_prefix: &str,
    patterns: &[P],
    max_in_flight: usize,
    do_create_bucket: bool,
    progress_callback: Option<Arc<ProgressCallback>>,
) -> Result<()> {
    let dest = parse_s3_uri_full(dest_prefix)?;
    let (bucket, key_prefix) = (dest.bucket, dest.key);
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
        create_bucket_for_uri_async(dest_prefix).await?;
    }

    // A destination that names a server uploads through that server's client; a
    // plain bucket uses the process-global client (and its op-log path) as before.
    let endpoint_store: Option<Arc<dyn ObjectStore>> = match &dest.endpoint {
        Some(_) => Some(Arc::from(store_for_uri(dest_prefix)?)),
        None => None,
    };

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
        do_create_bucket, max_in_flight
    );

    let sem = Arc::new(Semaphore::new(effective_jobs));
    let result = {
        let mut futs = FuturesUnordered::new();
        for path in paths.clone() {
            debug!("queueing upload for {:?}", path);
            let sem = sem.clone();
            let bucket = bucket.clone();
            let progress = progress_callback.clone();
            let endpoint_store = endpoint_store.clone();
            let endpoint = dest.endpoint.clone();
            let fname = path
                .file_name()
                .ok_or_else(|| anyhow::anyhow!("Bad path {:?}", path))?
                .to_string_lossy();
            let key = format!("{}{}", key_prefix, fname);

            // Get file size for progress tracking
            let file_size = fs::metadata(&path)?.len();

            futs.push(tokio::spawn(async move {
                debug!("starting upload of {:?} → s3://{}/{}", path, bucket, key);
                let _permit = sem.acquire_owned().await.unwrap();
                let body = ByteStream::from_path(&path).await?;
                let bytes = body.collect().await?.into_bytes();
                match (&endpoint_store, &endpoint) {
                    (Some(store), Some(authority)) => {
                        let uri = format!("s3://{authority}/{bucket}/{key}");
                        store.put(&uri, bytes).await?;
                    }
                    _ => put_object_async(&bucket, &key, bytes).await?,
                }
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

/// Download every object whose key starts with the LITERAL prefix named by `src_uri`
/// into `dest_dir/`, creating files with their original basenames, up to
/// `max_in_flight` at once.
///
/// The URI is a literal key prefix, never a regular expression (issue #213):
/// `s3://b/dir/` downloads everything under `dir/`; `s3://b/dir/file.txt` downloads
/// `dir/file.txt` and any other key that starts with it (`dir/file.txt.bak`); `.`,
/// `(`, `[`, `*` and the like are ordinary characters. To select by regular
/// expression use [`download_objects_matching`].
///
/// With `recursive == false` only the direct children of the prefix are listed (and
/// the "directories" among them are skipped).
pub async fn download_objects(
    src_uri: &str,
    dest_dir: &Path,
    max_in_flight: usize,
    recursive: bool,
    progress_callback: Option<Arc<ProgressCallback>>,
) -> Result<()> {
    download_objects_matching(
        src_uri,
        None,
        dest_dir,
        max_in_flight,
        recursive,
        progress_callback,
    )
    .await
}

/// [`download_objects`] narrowed by an EXPLICIT regular expression (issue #213).
///
/// `src_uri` is the literal key prefix to list; `pattern`, when given, is matched
/// (unanchored) against each listed key relative to the listing base (see
/// [`crate::list_pattern`]), so `("s3://b/dl/", r"\.npz$")` downloads the `.npz`
/// objects under `dl/`. An invalid pattern fails before anything is listed.
pub async fn download_objects_matching(
    src_uri: &str,
    pattern: Option<&str>,
    dest_dir: &Path,
    max_in_flight: usize,
    recursive: bool,
    progress_callback: Option<Arc<ProgressCallback>>,
) -> Result<()> {
    let (bucket, key_prefix) = parse_s3_uri_for_global_client(src_uri)?;

    // List the objects under the literal prefix, narrowed by the explicit pattern.
    let keys = list_objects_with_pattern(&bucket, &key_prefix, pattern, recursive)?;

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
                debug!(
                    "finished download of s3://{}/{} → {:?}",
                    bucket, k, out_path
                );

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
