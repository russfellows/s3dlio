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
use tokio::sync::Semaphore;

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
    // Issue #211: a limit of 0 would hang; refuse it before anything else.
    crate::concurrency::require_concurrency_at_least_one("max_in_flight", max_in_flight)?;
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

/// Download every object the `s3://` URI `src_uri` selects (issue #213)
/// into `dest_dir/`, up to `max_in_flight` at once.
///
/// Layout (issue #284): each object is written to `dest_dir/<key relative to the listing
/// base>` (parent directories are created), so `s3://b/dl/` writes `dl/a/x.bin` to
/// `dest_dir/a/x.bin`; a single exact object lands in `dest_dir/<its name>`. The rules are in
/// [`crate::download_layout`]. These library helpers return `Result<()>` and are FAIL-FAST: if
/// ANY selected key is unsafe (empty/absolute/`.`/`..` component, NUL, a drive designator such
/// as `C:`) or two objects map to one local path, the whole call fails with a "Download
/// refused" error that names the first offender, BEFORE `dest_dir` is created or anything is
/// fetched. (`s3-cli` and the Python `download` refuse per object instead and report the counts.)
/// A failure while fetching or writing an object ends the call with that object's error; files
/// already written stay, and so do directories created for a write that then failed.
///
/// The URI is never a regular expression and never a bare prefix (the rule is in
/// [`crate::s3_selection`]): `s3://b/dir/` downloads everything under `dir/`;
/// `s3://b/dir/file.txt` downloads exactly `dir/file.txt` (and, when `recursive`,
/// everything under `dir/file.txt/`) and NEVER `dir/file.txt.bak`; `*` / `**` make an
/// explicit glob over the full key (`s3://b/dir/file.txt*` does select
/// `dir/file.txt.bak`); `.`, `(`, `[` and the like are ordinary characters. To narrow
/// the selection by regular expression use [`download_objects_matching`].
///
/// With `recursive == false` the "directories" among the selected entries are skipped.
pub async fn download_objects(
    src_uri: &str,
    dest_dir: &Path,
    max_in_flight: usize,
    recursive: bool,
    progress_callback: Option<Arc<ProgressCallback>>,
) -> Result<()> {
    // Issue #211: a limit of 0 would hang; refuse it before anything else.
    crate::concurrency::require_concurrency_at_least_one("max_in_flight", max_in_flight)?;
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
/// `src_uri` selects the objects ([`download_objects`]); `pattern`, when given, is matched
/// (unanchored) against each listed key relative to the listing base (see
/// [`crate::list_pattern`]), so `("s3://b/dl/", r"\.npz$")` downloads the `.npz`
/// objects under `dl/`. An invalid pattern fails before anything is listed.
///
/// It lists through a blocking call on s3dlio's own global runtime, so it must not be called
/// from a task running on that runtime (the same constraint as [`download_objects`]).
pub async fn download_objects_matching(
    src_uri: &str,
    pattern: Option<&str>,
    dest_dir: &Path,
    max_in_flight: usize,
    recursive: bool,
    progress_callback: Option<Arc<ProgressCallback>>,
) -> Result<()> {
    // Issue #211: a limit of 0 builds a semaphore with no permits and hangs.  This is the
    // implementation `download_objects` delegates to (and a public entry of its own), so the
    // guard lives here: before the URI is parsed, anything is listed or a directory is made.
    crate::concurrency::require_concurrency_at_least_one("max_in_flight", max_in_flight)?;
    let (bucket, key_prefix) = parse_s3_uri_for_global_client(src_uri)?;

    // List the objects the URI selects, narrowed by the explicit pattern.
    let keys = list_objects_with_pattern(&bucket, &key_prefix, pattern, recursive)?;

    if keys.is_empty() {
        bail!("No objects matched for download");
    }

    // Issue #284: each object keeps its key path relative to the listing base. Refuse the
    // whole download, before anything is created or fetched, when one key is unsafe or two
    // objects map to one local path (an object and the directory of the same name included:
    // the message names the way to download either alone). The plan works on full URIs, the
    // spelling the message quotes and the one the listing base is computed on.
    let uri_prefix = format!("s3://{bucket}/");
    let uris: Vec<String> = keys.iter().map(|k| format!("{uri_prefix}{k}")).collect();
    let mut planned = Vec::new();
    for plan in crate::download_layout::plan_downloads(src_uri, &uris) {
        let comps = match plan.target {
            Ok(comps) => comps,
            Err(e) => bail!("Download refused: {:#}", e),
        };
        let key = plan
            .entry
            .strip_prefix(&uri_prefix)
            .ok_or_else(|| anyhow::anyhow!("internal: planned entry is not under {uri_prefix}"))?
            .to_string();
        planned.push((key, comps));
    }

    // sync—make sure the dir exists
    fs::create_dir_all(dest_dir)?;

    // Cap the number of concurrent tasks to the number of items
    let effective_jobs = std::cmp::min(max_in_flight, keys.len());

    // INFO: start of bulk download
    info!(
        "Starting download of {} object(s) from {} to {:?} (jobs={})",
        keys.len(),
        crate::redact::redact_and_escape(src_uri),
        dest_dir,
        effective_jobs
    );
    // DEBUG: details of requested jobs
    debug!("download_objects debug: requested_jobs={}", max_in_flight);

    let sem = Arc::new(Semaphore::new(effective_jobs));
    let result = {
        let mut futs = FuturesUnordered::new();
        // "Directories" (keys ending in a slash) were left out of `planned`.
        for (k, comps) in planned {
            let sem = sem.clone();
            let bucket = bucket.clone();
            let out_dir = dest_dir.to_path_buf();
            let progress = progress_callback.clone();
            // The object as the messages name it, control characters of the key escaped.
            let shown = crate::redact::escape_control(&format!("s3://{bucket}/{k}"));

            futs.push(tokio::spawn(async move {
                debug!("starting download of {} → {:?}", shown, out_dir);
                let fetched_and_written = async {
                    let _permit = sem.acquire_owned().await?;
                    let bytes = get_object(&bucket, &k).await?;
                    let byte_count = bytes.len() as u64;
                    let out_path =
                        crate::download_layout::write_object(&out_dir, &comps, bytes).await?;
                    debug!("finished download of {} → {:?}", shown, out_path);

                    // Update progress if callback provided
                    if let Some(ref progress) = progress {
                        progress.object_completed(byte_count);
                    }

                    Ok::<(), anyhow::Error>(())
                }
                .await;
                // The failure names the object it belongs to. The store's own error quotes the
                // raw key, so the final text of the whole chain is escaped (a returned error
                // is shown by callers and logged as it stands).
                fetched_and_written.map_err(|e| {
                    anyhow::anyhow!(crate::redact::escape_control(&format!("{shown}: {e:#}")))
                })
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
