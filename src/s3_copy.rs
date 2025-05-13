// src/s3_copy.rs

use anyhow::{bail, Result};
use aws_sdk_s3::primitives::ByteStream;
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use std::{fs, path::Path, sync::Arc};
use tokio::{fs as async_fs, runtime::Runtime, sync::Semaphore};

use crate::s3_utils::{
    parse_s3_uri, create_bucket, list_objects, get_object, put_object_async,
};

// ──────────────────────────────────────────────────────────────────────────────
// Functions that leverage existing s3_utils code
// Will upload local files → S3  &  download S3 → local files
// ──────────────────────────────────────────────────────────────────────────────

/// Upload a list of local files into `s3://bucket/prefix/…`,
/// streaming each file off disk with up to `max_in_flight` tasks.
/// If `do_create_bucket` is true, we attempt to create the bucket first.
pub fn upload_files<P: AsRef<Path>>(  
    dest_prefix: &str,
    paths: &[P],
    max_in_flight: usize,
    do_create_bucket: bool,
) -> Result<()> {
    let (bucket, key_prefix) = parse_s3_uri(dest_prefix)?;
    // Allow empty prefix (bucket root) or a non-empty prefix that ends with '/'
    if !key_prefix.is_empty() && !key_prefix.ends_with('/') {
        bail!("dest_prefix must end with '/' if specifying a non-root prefix");
    }

    if do_create_bucket {
        create_bucket(&bucket)?;
    }

    // Cap the number of concurrent tasks to the number of items
    let effective_jobs = std::cmp::min(max_in_flight, paths.len());

    // build a Tokio runtime for our async work
    let rt = Runtime::new()?;

    // INFO: start of bulk upload
    log::info!(
        "Starting upload of {} file(s) to {} (jobs={})",
        paths.len(),
        dest_prefix,
        effective_jobs
    );
    // DEBUG: details of create_bucket and original limit
    log::debug!("upload_files debug: create_bucket={}, requested_jobs={}",
        do_create_bucket, max_in_flight);

    let sem = Arc::new(Semaphore::new(effective_jobs));
    let result = rt.block_on(async {
        let mut futs = FuturesUnordered::new();
        for p in paths {
            log::debug!("queueing upload for {:?}", p.as_ref());
            let sem = sem.clone();
            let bucket = bucket.clone();
            let key = {
                let fname = p
                    .as_ref()
                    .file_name()
                    .ok_or_else(|| anyhow::anyhow!("Bad path {:?}", p.as_ref()))?
                    .to_string_lossy();
                format!("{}{}", key_prefix, fname)
            };
            let path = p.as_ref().to_path_buf();

            futs.push(tokio::spawn(async move {
                log::debug!("starting upload of {:?} → s3://{}/{}", path, bucket, key);
                let _permit = sem.acquire_owned().await.unwrap();
                let body = ByteStream::from_path(&path).await?;
                put_object_async(&bucket, &key, &body.collect().await?.into_bytes()).await?;
                log::debug!("finished upload of {:?} → s3://{}/{}", path, bucket, key);
                Ok::<(), anyhow::Error>(())
            }));
        }

        while let Some(join_res) = futs.next().await {
            join_res??;
        }
        Ok(())
    });

    // INFO: result of bulk upload
    match &result {
        Ok(()) => log::info!(
            "Finished upload of {} file(s) to {}",
            paths.len(),
            dest_prefix
        ),
        Err(e) => log::error!("Upload failed: {}", e),
    }
    result
}

/// Download one key or every key under a prefix into `dest_dir/`,
/// creating files with their original basenames, up to `max_in_flight` at once.
pub fn download_objects(
    src_uri: &str,
    dest_dir: &Path,
    max_in_flight: usize,
) -> Result<()> {
    let (bucket, key) = parse_s3_uri(src_uri)?;
    let keys = if key.ends_with('/') || key.is_empty() {
        list_objects(&bucket, &key)?
    } else {
        vec![key]
    };

    // sync—make sure the dir exists
    fs::create_dir_all(dest_dir)?;

    // Cap the number of concurrent tasks to the number of items
    let effective_jobs = std::cmp::min(max_in_flight, keys.len());

    // build a Tokio runtime for our async work
    let rt = Runtime::new()?;

    // INFO: start of bulk download
    log::info!(
        "Starting download of {} object(s) from {} to {:?} (jobs={})",
        keys.len(),
        src_uri,
        dest_dir,
        effective_jobs
    );
    // DEBUG: details of requested jobs
    log::debug!("download_objects debug: requested_jobs={}", max_in_flight);

    let sem = Arc::new(Semaphore::new(effective_jobs));
    let result = rt.block_on(async {
        let mut futs = FuturesUnordered::new();
        for k in keys.clone() {
            let sem = sem.clone();
            let bucket = bucket.clone();
            let out_dir = dest_dir.to_path_buf();

            futs.push(tokio::spawn(async move {
                log::debug!("starting download of s3://{}/{} → {:?}", bucket, k, out_dir);
                let _permit = sem.acquire_owned().await.unwrap();
                let bytes = get_object(&bucket, &k).await?;
                let fname = Path::new(&k)
                    .file_name()
                    .ok_or_else(|| anyhow::anyhow!("Bad key: {}", k))?;
                let out_path = out_dir.join(fname);
                async_fs::write(&out_path, bytes).await?;
                log::debug!("finished download of s3://{}/{} → {:?}", bucket, k, out_path);
                Ok::<(), anyhow::Error>(())
            }));
        }

        while let Some(join_res) = futs.next().await {
            join_res??;
        }
        Ok(())
    });

    // INFO: result of bulk download
    match &result {
        Ok(()) => log::info!(
            "Finished download of {} object(s) to {:?}",
            keys.len(),
            dest_dir
        ),
        Err(e) => log::error!("Download failed: {}", e),
    }
    result
}

