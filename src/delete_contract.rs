// src/delete_contract.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! The delete contract shared by `s3-cli delete` (issues #210 / #167) and the
//! Python `s3dlio.delete()` (issue #212).
//!
//! The contract ("contract A"):
//!
//! * **Exact delete.**  `delete(uri)` sends ONE delete request for exactly the
//!   key named by `uri` -- no lookup first, no listing, no parent consulted.
//!   A trailing `/`, `*` or `?` is part of the key, never an instruction.
//!   Deleting an already-missing object SUCCEEDS (S3's DELETE succeeds for a
//!   missing key; for a backend that reports it, exactly its typed not-found
//!   error is mapped to success -- see [`is_not_found`]).  Every other failure
//!   (403, 5xx, network, ...) is returned: a failure is never read as "already
//!   absent".  Success means "the delete request succeeded", not "an existing
//!   object was removed" (S3 does not say whether the key existed; on a
//!   versioned bucket a delete adds a delete marker and erases no version).
//! * **Recursive delete.**  [`delete_objects_under_literal_prefix`] deletes the
//!   REAL objects whose key starts with the literal prefix given
//!   ([`ObjectStore::list_objects_with_prefix`]: no delimiter, objects only,
//!   no `/` appended -- the prefix `foo` also matches `foobar`).
//!
//! The functions here select and delete; the callers decide how to word the
//! outcome (the CLI prints, Python raises).  A recursive delete needs list
//! permission in addition to delete.

use anyhow::{bail, Result};
use futures::stream::StreamExt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::object_store::ObjectStore;

/// Keys per DeleteObjects request (the S3 API limit).
pub const DELETE_REQUEST_KEYS: usize = 1000;
/// DeleteObjects requests kept in flight at once by a recursive delete.
pub const DELETE_REQUESTS_IN_FLIGHT: usize = 10;

/// Delete exactly the object `uri`: one delete request, no lookup.
///
/// An already-missing object is success (see the module docs).  Any other
/// failure is returned as "failed to delete ..." with the cause chained, so the
/// URI is always named.
pub async fn delete_exact_object(store: &dyn ObjectStore, uri: &str) -> Result<()> {
    refuse_directory(uri).await?;
    match store.delete(uri).await {
        Ok(()) => Ok(()),
        Err(e) if is_not_found(&e) => Ok(()),
        Err(e) => Err(e.context(format!("failed to delete {uri}"))),
    }
}

/// A `file://` or `direct://` store's `delete` removes a directory together with
/// everything under it.  An exact delete must never be a recursive delete, so
/// a path that is a directory is refused here (nothing is deleted) and the
/// caller is pointed at the recursive form.  A path that does not exist is left
/// to the store (already gone is success); a regular file is deleted as before.
/// Any other failure to inspect the path is an error.
///
/// The check and the delete are separate steps, so a directory swapped in
/// between is not caught; this guards against the caller's mistake, not against
/// a concurrent writer.
async fn refuse_directory(uri: &str) -> Result<()> {
    let Some(path) = uri
        .strip_prefix("file://")
        .or_else(|| uri.strip_prefix("direct://"))
    else {
        return Ok(());
    };
    match tokio::fs::metadata(path).await {
        Ok(meta) if meta.is_dir() => bail!(
            "{uri} is a directory; use -r / recursive=True to delete everything under it \
             (nothing was deleted)"
        ),
        Ok(_) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => {
            Err(anyhow::Error::new(e).context(format!("cannot inspect {uri}; nothing was deleted")))
        }
    }
}

/// True only when `err` (or a cause in its chain) is a backend's TYPED
/// "object not found" error:
///
/// * an [`std::io::Error`] of kind `NotFound` (file backends);
/// * Azure (`backend-azure`): an HTTP 404 response with service error code
///   `BlobNotFound` (a missing container is a different code and stays an
///   error);
/// * GCS (`backend-gcs`, official client): a NOT_FOUND status / HTTP 404.  The
///   service gives the same status for a missing bucket, so a missing bucket
///   is not told apart from a missing object there.
///
/// Message text is never inspected.  S3 needs no entry: its DELETE succeeds for
/// a missing key.
pub fn is_not_found(err: &anyhow::Error) -> bool {
    err.chain().any(|cause| {
        if let Some(io) = cause.downcast_ref::<std::io::Error>() {
            return io.kind() == std::io::ErrorKind::NotFound;
        }
        #[cfg(feature = "backend-azure")]
        if let Some(az) = cause.downcast_ref::<azure_core::Error>() {
            return matches!(
                az.kind(),
                azure_core::error::ErrorKind::HttpResponse { status, error_code, .. }
                    if *status == azure_core::http::StatusCode::NotFound
                        && error_code.as_deref() == Some("BlobNotFound")
            );
        }
        #[cfg(feature = "backend-gcs")]
        if let Some(g) = cause.downcast_ref::<google_cloud_gax::error::Error>() {
            return g.http_status_code() == Some(404)
                || g.status()
                    .is_some_and(|s| s.code == google_cloud_gax::error::rpc::Code::NotFound);
        }
        false
    })
}

/// What one batch of selected keys came back with.
struct DeleteBatchOutcome {
    /// Keys a DeleteObjects request was sent for.
    requested: usize,
    /// Keys NOT sent because an earlier request or the listing had failed.
    skipped: usize,
    /// `Err` when the request, or any key in it, failed.
    result: Result<()>,
    /// The listing failed while this batch was being filled.
    list_error: Option<anyhow::Error>,
}

/// What [`delete_objects_under_literal_prefix`] did.
///
/// S3 reports a key that does not exist as "deleted", so these are counts of
/// requests sent, never a count of objects actually removed.
#[derive(Debug, Default)]
pub struct PrefixDeleteReport {
    /// Objects the listing returned (counted BEFORE any filter).
    pub listed: u64,
    /// Objects that passed the filter, i.e. were handed to deletion.
    pub selected: u64,
    /// Keys a DeleteObjects request was sent for.
    pub requested: usize,
    /// Selected keys NOT sent because an earlier request or the listing failed.
    pub skipped: usize,
    /// DeleteObjects requests that failed (or reported a per-key error).
    pub failed_requests: usize,
    /// One message per failed delete request.
    pub delete_problems: Vec<String>,
    /// One message per listing error.
    pub list_problems: Vec<String>,
}

impl PrefixDeleteReport {
    /// A delete request failed, or the listing did.
    pub fn failed(&self) -> bool {
        !self.delete_problems.is_empty() || !self.list_problems.is_empty()
    }

    /// The listing failed before any delete request was sent.
    pub fn listing_failed_before_any_send(&self) -> bool {
        self.requested == 0
            && self.skipped == 0
            && self.delete_problems.is_empty()
            && !self.list_problems.is_empty()
    }

    /// `Ok(())` when nothing failed; otherwise the error to report, worded as
    /// the CLI words it.  `uri` is the prefix URI, `pattern` the text of the
    /// filter, if one was applied.
    pub fn into_result(self, uri: &str, pattern: Option<&str>) -> Result<()> {
        // A listing that failed before anything was sent: report the listing
        // error itself, not a delete summary for zero objects.
        if self.listing_failed_before_any_send() {
            bail!(
                "cannot list objects under prefix {uri}; nothing was deleted: {}",
                self.list_problems.join("; ")
            );
        }
        if !self.failed() {
            return Ok(());
        }
        let listed = self.listed;
        let selected = self.selected;
        let scope = match pattern {
            Some(pat) => format!(
                "{selected} selected (pattern '{pat}') of {listed} listed under prefix {uri}"
            ),
            None => format!("{listed} listed under prefix {uri}"),
        };
        let mut msg = format!("Sent delete requests for {} of the {scope}", self.requested);
        if self.failed_requests > 0 {
            msg.push_str(&format!(
                "; {} delete request(s) reported errors",
                self.failed_requests
            ));
        } else {
            msg.push_str("; no delete request reported an error");
        }
        if self.skipped > 0 {
            msg.push_str(&format!(
                "; {} selected objects were NOT attempted because an earlier request \
                 or the listing failed",
                self.skipped
            ));
        }
        if !self.list_problems.is_empty() {
            msg.push_str("; the listing then failed, so later objects were not seen");
        }
        if !self.delete_problems.is_empty() || self.skipped > 0 {
            msg.push_str("; the rest of the prefix was not listed");
        }
        let details: Vec<String> = self
            .delete_problems
            .into_iter()
            .chain(
                self.list_problems
                    .into_iter()
                    .map(|e| format!("listing failed: {e}")),
            )
            .collect();
        bail!("{msg}:\n  {}", details.join("\n  "));
    }
}

/// Send one filled batch: one `delete_batch`, or one `delete` per object.
async fn send(store: &dyn ObjectStore, mode: DeleteMode, uris: &[String]) -> Result<()> {
    match mode {
        DeleteMode::Batched => store.delete_batch(uris).await,
        DeleteMode::PerObject => {
            for uri in uris {
                match store.delete(uri).await {
                    Ok(()) => {}
                    Err(e) if is_not_found(&e) => {}
                    Err(e) => return Err(e.context(format!("failed to delete {uri}"))),
                }
            }
            Ok(())
        }
    }
}

/// A predicate over the listed object URI (`s3://bkt/key`) narrowing the
/// selection.
pub type SelectionFilter<'a> = &'a (dyn Fn(&str) -> bool + Sync);
/// Called after each batch finishes with `(listed so far, delete requests sent
/// for so far)`; for progress display.
pub type ProgressHook<'a> = &'a (dyn Fn(u64, usize) + Sync);

/// How the selected objects are deleted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeleteMode {
    /// Batches of up to [`DELETE_REQUEST_KEYS`] keys per
    /// [`ObjectStore::delete_batch`] (S3 DeleteObjects), `s3-cli delete -r`.
    Batched,
    /// One [`ObjectStore::delete`] per object (a single-key request each, the
    /// op-log records one `DELETE` row per object), Python
    /// `delete(..., recursive=True)`.  An object that has vanished since it was
    /// listed (a typed not-found error) counts as deleted, as for an exact delete.
    PerObject,
}

/// Delete the real objects under the literal prefix `uri`, listed recursively
/// ([`ObjectStore::list_objects_with_prefix`]: no delimiter, objects only, no
/// `/` appended), optionally narrowed by `filter`.  The selection is the same
/// for both [`DeleteMode`]s; only the way each selected object is deleted differs.
///
/// After any failure (a delete request, a per-key error, the listing) no
/// further delete request is SENT: the check is made immediately before each
/// send, so a batch that was only filled after the failure is skipped and
/// counted as not attempted.  A listing error also cancels the batch being
/// assembled when it occurs: its unsent keys are counted as not attempted, not
/// sent.  Requests already in flight finish.  Never panics; every failure is
/// recorded in the returned report.
pub async fn delete_objects_under_literal_prefix(
    store: &dyn ObjectStore,
    uri: &str,
    filter: Option<SelectionFilter<'_>>,
    mode: DeleteMode,
    progress: Option<ProgressHook<'_>>,
) -> PrefixDeleteReport {
    let chunk_keys = match mode {
        DeleteMode::Batched => DELETE_REQUEST_KEYS,
        DeleteMode::PerObject => 1,
    };
    let stop = AtomicBool::new(false);
    // `listed`: objects the listing returned, counted BEFORE the filter.
    // `selected`: objects that passed the filter, i.e. were handed to deletion.
    let listed = AtomicU64::new(0);
    let selected = AtomicU64::new(0);

    let selection = store
        .list_objects_with_prefix(uri)
        // Stop listing once a failure is known; the item that finds the flag
        // set is dropped before it is counted, so the counts add up.
        .take_while(|_| std::future::ready(!stop.load(Ordering::Relaxed)))
        .inspect(|item| {
            if item.is_ok() {
                listed.fetch_add(1, Ordering::Relaxed);
            }
        })
        .filter(|item| {
            let keep = match (item, filter) {
                (Ok(u), Some(f)) => f(u),
                _ => true,
            };
            std::future::ready(keep)
        })
        .inspect(|item| {
            if item.is_ok() {
                selected.fetch_add(1, Ordering::Relaxed);
            }
        });

    let stop_ref = &stop;
    let outcomes = selection
        .chunks(chunk_keys)
        .map(|chunk| async move {
            let mut uris = Vec::with_capacity(chunk.len());
            let mut list_error = None;
            for item in chunk {
                match item {
                    Ok(u) => uris.push(u),
                    Err(e) => {
                        list_error = Some(e);
                        break;
                    }
                }
            }
            // The last look before sending. A batch filled after a failure is
            // not sent, and neither is the batch in which the listing failed:
            // the keys it holds were listed, but the listing that selected
            // them did not finish, so they are counted as not attempted.
            let stopped = stop_ref.load(Ordering::Relaxed) || list_error.is_some();
            if list_error.is_some() {
                stop_ref.store(true, Ordering::Relaxed);
            }
            if stopped || uris.is_empty() {
                return DeleteBatchOutcome {
                    requested: 0,
                    skipped: if stopped { uris.len() } else { 0 },
                    result: Ok(()),
                    list_error,
                };
            }
            let result = send(store, mode, &uris).await;
            if result.is_err() {
                stop_ref.store(true, Ordering::Relaxed);
            }
            DeleteBatchOutcome {
                requested: uris.len(),
                skipped: 0,
                result,
                list_error,
            }
        })
        .buffer_unordered(DELETE_REQUESTS_IN_FLIGHT);
    tokio::pin!(outcomes);

    let mut report = PrefixDeleteReport::default();
    while let Some(outcome) = outcomes.next().await {
        report.requested += outcome.requested;
        report.skipped += outcome.skipped;
        if let Err(e) = outcome.result {
            report.failed_requests += 1;
            report.delete_problems.push(format!("{e:#}"));
        }
        if let Some(e) = outcome.list_error {
            report.list_problems.push(format!("{e:#}"));
        }
        if let Some(hook) = progress {
            hook(listed.load(Ordering::Relaxed), report.requested);
        }
    }
    report.listed = listed.load(Ordering::Relaxed);
    report.selected = selected.load(Ordering::Relaxed);
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object_store::{ObjectMetadata, ObjectWriter};
    use async_trait::async_trait;
    use bytes::Bytes;
    use futures::stream::Stream;
    use std::collections::BTreeSet;
    use std::pin::Pin;
    use std::sync::Mutex;

    /// An in-memory store that records every call. It relies on the trait
    /// DEFAULT `list_objects_with_prefix` (the non-S3 behaviour) unless
    /// `literal_listing` is set, in which case it mimics the S3 override.
    struct Fake {
        keys: Mutex<BTreeSet<String>>,
        /// What `delete` fails with, if anything.
        delete_error: Option<fn() -> anyhow::Error>,
        /// `delete` of exactly this URI fails with the given error.
        delete_error_for: Option<(String, fn() -> anyhow::Error)>,
        literal_listing: bool,
        /// A DeleteObjects request naming this key fails.
        fail_batch_with_key: Option<String>,
        /// The listing fails after yielding this many items.
        list_fails_after: Option<usize>,
        calls: Mutex<Vec<String>>,
        batches: Mutex<Vec<Vec<String>>>,
    }

    impl Fake {
        fn new(keys: &[&str]) -> Fake {
            Fake {
                keys: Mutex::new(keys.iter().map(|k| format!("s3://bkt/{k}")).collect()),
                delete_error: None,
                delete_error_for: None,
                literal_listing: true,
                fail_batch_with_key: None,
                list_fails_after: None,
                calls: Mutex::new(Vec::new()),
                batches: Mutex::new(Vec::new()),
            }
        }
        fn note(&self, call: String) {
            self.calls.lock().unwrap().push(call);
        }
        fn calls(&self) -> Vec<String> {
            self.calls.lock().unwrap().clone()
        }
        fn keys(&self) -> BTreeSet<String> {
            self.keys.lock().unwrap().clone()
        }
        fn listed(&self, prefix: &str) -> Vec<Result<String>> {
            let mut out: Vec<Result<String>> = self
                .keys()
                .into_iter()
                .filter(|k| k.starts_with(prefix))
                .map(Ok)
                .collect();
            if let Some(n) = self.list_fails_after {
                out.truncate(n);
                out.push(Err(anyhow::anyhow!("listing refused")));
            }
            out
        }
    }

    #[async_trait]
    impl ObjectStore for Fake {
        async fn get(&self, _uri: &str) -> Result<Bytes> {
            bail!("unused")
        }
        async fn get_range(&self, _uri: &str, _offset: u64, _length: Option<u64>) -> Result<Bytes> {
            bail!("unused")
        }
        async fn put(&self, _uri: &str, _data: Bytes) -> Result<()> {
            bail!("unused")
        }
        async fn put_multipart(&self, _u: &str, _d: Bytes, _p: Option<usize>) -> Result<()> {
            bail!("unused")
        }
        async fn list(&self, _uri_prefix: &str, _recursive: bool) -> Result<Vec<String>> {
            self.note("LIST(non-literal)".to_string());
            bail!("the contract never calls list()")
        }
        fn list_stream<'a>(
            &'a self,
            uri_prefix: &'a str,
            _recursive: bool,
        ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
            self.note(format!("LIST_STREAM {uri_prefix}"));
            Box::pin(futures::stream::iter(self.listed(uri_prefix)))
        }
        fn list_objects_with_prefix<'a>(
            &'a self,
            uri_prefix: &'a str,
        ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
            if self.literal_listing {
                self.note(format!("LIST_LITERAL {uri_prefix}"));
                Box::pin(futures::stream::iter(self.listed(uri_prefix)))
            } else if uri_prefix.ends_with('/') {
                self.list_stream(uri_prefix, true)
            } else {
                Box::pin(futures::stream::once(async move {
                    Err(anyhow::anyhow!(
                        "recursive listing of '{uri_prefix}' is not supported on this backend \
                         without a trailing '/'"
                    ))
                }))
            }
        }
        async fn stat(&self, _uri: &str) -> Result<ObjectMetadata> {
            bail!("stat is not used by the contract")
        }
        async fn delete(&self, uri: &str) -> Result<()> {
            self.note(format!("DELETE {uri}"));
            if let Some(make) = self.delete_error {
                return Err(make());
            }
            if let Some((bad, make)) = &self.delete_error_for {
                if bad == uri {
                    return Err(make());
                }
            }
            self.keys.lock().unwrap().remove(uri);
            Ok(())
        }
        async fn delete_batch(&self, uris: &[String]) -> Result<()> {
            self.note(format!("DELETE_BATCH {}", uris.len()));
            self.batches.lock().unwrap().push(uris.to_vec());
            if let Some(bad) = &self.fail_batch_with_key {
                if uris.contains(bad) {
                    bail!("AccessDenied for {bad}");
                }
            }
            let mut keys = self.keys.lock().unwrap();
            for u in uris {
                keys.remove(u);
            }
            Ok(())
        }
        async fn delete_prefix(&self, _uri_prefix: &str) -> Result<()> {
            self.note("DELETE_PREFIX".to_string());
            bail!("the contract never calls delete_prefix()")
        }
        async fn create_container(&self, _name: &str) -> Result<()> {
            bail!("unused")
        }
        async fn delete_container(&self, _name: &str) -> Result<()> {
            bail!("unused")
        }
        async fn get_writer(&self, _uri: &str) -> Result<Box<dyn ObjectWriter>> {
            bail!("unused")
        }
    }

    fn set(keys: &[&str]) -> BTreeSet<String> {
        keys.iter().map(|k| format!("s3://bkt/{k}")).collect()
    }

    // ---- exact delete -----------------------------------------------------

    fn io_not_found() -> anyhow::Error {
        anyhow::Error::new(std::io::Error::from(std::io::ErrorKind::NotFound))
            .context("remove_file failed")
    }
    fn io_denied() -> anyhow::Error {
        anyhow::Error::new(std::io::Error::from(std::io::ErrorKind::PermissionDenied))
            .context("remove_file failed")
    }
    fn http_403_text() -> anyhow::Error {
        anyhow::anyhow!("S3 DELETE failed: service error: AccessDenied (HTTP 403)")
    }
    fn untyped_not_found_text() -> anyhow::Error {
        anyhow::anyhow!("object not found (a string, not a typed not-found error)")
    }

    #[tokio::test]
    async fn exact_delete_sends_one_delete_and_no_lookup() {
        let f = Fake::new(&["foo", "foo/", "foo/x", "foobar"]);
        delete_exact_object(&f, "s3://bkt/foo").await.unwrap();
        assert_eq!(f.keys(), set(&["foo/", "foo/x", "foobar"]));
        assert_eq!(f.calls(), ["DELETE s3://bkt/foo"]);
    }

    #[tokio::test]
    async fn exact_delete_trailing_slash_is_the_key_and_never_lists() {
        let f = Fake::new(&["foo/", "foo/x", "foo/sub/y"]);
        delete_exact_object(&f, "s3://bkt/foo/").await.unwrap();
        assert_eq!(f.keys(), set(&["foo/x", "foo/sub/y"]));
        assert_eq!(f.calls(), ["DELETE s3://bkt/foo/"]);
    }

    #[tokio::test]
    async fn exact_delete_of_a_missing_object_succeeds_and_touches_nothing_else() {
        let f = Fake::new(&["foo/x", "other/"]);
        delete_exact_object(&f, "s3://bkt/foo/").await.unwrap();
        assert_eq!(f.keys(), set(&["foo/x", "other/"]));
        assert_eq!(f.calls(), ["DELETE s3://bkt/foo/"]);
    }

    #[tokio::test]
    async fn exact_delete_maps_a_typed_not_found_to_success() {
        let mut f = Fake::new(&["a"]);
        f.delete_error = Some(io_not_found);
        delete_exact_object(&f, "file:///d/a").await.unwrap();
        assert_eq!(f.calls(), ["DELETE file:///d/a"]);
    }

    #[tokio::test]
    async fn exact_delete_propagates_every_other_failure_naming_the_key() {
        for make in [io_denied as fn() -> anyhow::Error, http_403_text] {
            let mut f = Fake::new(&["foo"]);
            f.delete_error = Some(make);
            let err = delete_exact_object(&f, "s3://bkt/foo").await.unwrap_err();
            let text = format!("{err:#}");
            assert!(text.starts_with("failed to delete s3://bkt/foo"), "{text}");
            assert_eq!(f.keys(), set(&["foo"]));
            assert_eq!(f.calls(), ["DELETE s3://bkt/foo"]);
        }
    }

    #[tokio::test]
    async fn exact_delete_never_guesses_not_found_from_message_text() {
        let mut f = Fake::new(&["foo"]);
        f.delete_error = Some(untyped_not_found_text);
        let err = delete_exact_object(&f, "s3://bkt/foo").await.unwrap_err();
        assert!(format!("{err:#}").contains("not found"), "{err:#}");
    }

    #[test]
    fn is_not_found_looks_through_context_but_only_at_typed_errors() {
        assert!(is_not_found(&io_not_found().context("outer")));
        assert!(!is_not_found(&io_denied()));
        assert!(!is_not_found(&http_403_text()));
        assert!(!is_not_found(&untyped_not_found_text()));
    }

    #[cfg(feature = "backend-azure")]
    #[test]
    fn is_not_found_recognises_only_azure_blob_not_found() {
        use azure_core::error::ErrorKind;
        use azure_core::http::StatusCode;
        let http = |status, code: &str| {
            anyhow::Error::new(azure_core::Error::with_message(
                ErrorKind::HttpResponse {
                    status,
                    error_code: Some(code.to_string()),
                    raw_response: None,
                },
                "azure",
            ))
        };
        assert!(is_not_found(&http(StatusCode::NotFound, "BlobNotFound")));
        assert!(!is_not_found(&http(
            StatusCode::NotFound,
            "ContainerNotFound"
        )));
        assert!(!is_not_found(&http(
            StatusCode::Forbidden,
            "AuthorizationFailure"
        )));
    }

    #[cfg(feature = "backend-gcs")]
    #[test]
    fn is_not_found_recognises_only_a_gcs_404() {
        let http = |code: u16| {
            anyhow::Error::new(google_cloud_gax::error::Error::http(
                code,
                http::HeaderMap::new(),
                Bytes::from_static(b"body"),
            ))
        };
        assert!(is_not_found(&http(404)));
        assert!(!is_not_found(&http(403)));
        assert!(!is_not_found(&http(500)));
    }

    #[tokio::test]
    async fn exact_delete_of_a_file_or_direct_directory_is_refused_before_the_store_is_called() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("d");
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("sub").join("g"), b"1").unwrap();
        let plain = tmp.path().join("plain");
        std::fs::write(&plain, b"x").unwrap();
        let f = Fake::new(&[]);

        for scheme in ["file", "direct"] {
            for suffix in ["", "/"] {
                let uri = format!("{scheme}://{}{suffix}", dir.display());
                let err = delete_exact_object(&f, &uri).await.unwrap_err();
                let text = format!("{err:#}");
                assert!(
                    text.contains("is a directory") && text.contains("-r"),
                    "{text}"
                );
            }
        }
        assert!(
            f.calls().is_empty(),
            "the store must not be asked: {:?}",
            f.calls()
        );
        assert!(dir.join("sub").join("g").exists());

        // a regular file and a missing path reach the store, as before
        delete_exact_object(&f, &format!("file://{}", plain.display()))
            .await
            .unwrap();
        delete_exact_object(&f, &format!("file://{}/missing", tmp.path().display()))
            .await
            .unwrap();
        assert_eq!(f.calls().len(), 2);
    }

    #[tokio::test]
    async fn wildcard_characters_in_an_exact_delete_are_part_of_the_key() {
        let f = Fake::new(&["a*", "ab", "a?b", "axb"]);
        delete_exact_object(&f, "s3://bkt/a*").await.unwrap();
        delete_exact_object(&f, "s3://bkt/a?b").await.unwrap();
        assert_eq!(f.keys(), set(&["ab", "axb"]));
        assert_eq!(f.calls(), ["DELETE s3://bkt/a*", "DELETE s3://bkt/a?b"]);
    }

    // ---- recursive delete --------------------------------------------------

    #[tokio::test]
    async fn prefix_delete_deletes_the_listed_objects_under_the_literal_prefix() {
        let f = Fake::new(&["foo", "foo/", "foo/x", "foobar", "other/x"]);
        let report = delete_objects_under_literal_prefix(
            &f,
            "s3://bkt/foo",
            None,
            DeleteMode::Batched,
            None,
        )
        .await;
        assert!(!report.failed());
        assert_eq!(
            (report.listed, report.selected, report.requested),
            (4, 4, 4)
        );
        assert_eq!(f.keys(), set(&["other/x"]));
        // the prefix is passed through exactly as given: no '/' appended
        assert_eq!(f.calls()[0], "LIST_LITERAL s3://bkt/foo");
        report.into_result("s3://bkt/foo", None).unwrap();
    }

    #[tokio::test]
    async fn prefix_delete_filter_narrows_the_selection_and_counts_before_and_after() {
        let f = Fake::new(&["p/a.txt", "p/b.log", "p/c.txt"]);
        let only_txt = |u: &str| u.ends_with(".txt");
        let report = delete_objects_under_literal_prefix(
            &f,
            "s3://bkt/p/",
            Some(&only_txt),
            DeleteMode::Batched,
            None,
        )
        .await;
        assert_eq!(
            (report.listed, report.selected, report.requested),
            (3, 2, 2)
        );
        assert_eq!(f.keys(), set(&["p/b.log"]));
    }

    #[tokio::test]
    async fn prefix_delete_with_nothing_under_the_prefix_sends_nothing() {
        let f = Fake::new(&["other/x"]);
        let report = delete_objects_under_literal_prefix(
            &f,
            "s3://bkt/foo/",
            None,
            DeleteMode::Batched,
            None,
        )
        .await;
        assert!(!report.failed());
        assert_eq!((report.listed, report.requested), (0, 0));
        assert_eq!(f.keys(), set(&["other/x"]));
        assert!(!f.calls().iter().any(|c| c.starts_with("DELETE")));
    }

    #[tokio::test]
    async fn prefix_delete_batches_at_the_s3_limit() {
        let names: Vec<String> = (0..2500).map(|i| format!("big/k{i:05}")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let f = Fake::new(&refs);
        let report = delete_objects_under_literal_prefix(
            &f,
            "s3://bkt/big/",
            None,
            DeleteMode::Batched,
            None,
        )
        .await;
        assert!(!report.failed());
        assert_eq!(report.requested, 2500);
        let mut sizes: Vec<usize> = f.batches.lock().unwrap().iter().map(Vec::len).collect();
        sizes.sort_unstable();
        assert_eq!(sizes, [500, 1000, 1000]);
        assert!(f.keys().is_empty());
    }

    #[tokio::test]
    async fn prefix_delete_sends_nothing_more_after_a_failed_request() {
        let names: Vec<String> = (0..3500).map(|i| format!("big/k{i:05}")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let mut f = Fake::new(&refs);
        // the first batch contains big/k00000 and fails
        f.fail_batch_with_key = Some("s3://bkt/big/k00000".to_string());
        let report = delete_objects_under_literal_prefix(
            &f,
            "s3://bkt/big/",
            None,
            DeleteMode::Batched,
            None,
        )
        .await;
        assert!(report.failed());
        assert_eq!(report.failed_requests, 1);
        assert_eq!(
            f.batches.lock().unwrap().len(),
            1,
            "only the failing request was sent"
        );
        assert_eq!(report.requested, 1000);
        let err = report.into_result("s3://bkt/big/", None).unwrap_err();
        let text = format!("{err:#}");
        assert!(
            text.contains("Sent delete requests for 1000 of the"),
            "{text}"
        );
        assert!(
            text.contains("AccessDenied for s3://bkt/big/k00000"),
            "{text}"
        );
        assert!(
            text.contains("the rest of the prefix was not listed"),
            "{text}"
        );
    }

    #[tokio::test]
    async fn a_listing_error_cancels_the_batch_being_assembled() {
        let names: Vec<String> = (0..1500).map(|i| format!("big/k{i:05}")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let mut f = Fake::new(&refs);
        f.list_fails_after = Some(1500);
        let report = delete_objects_under_literal_prefix(
            &f,
            "s3://bkt/big/",
            None,
            DeleteMode::Batched,
            None,
        )
        .await;
        // the full first batch was sent; the 500 keys listed before the error
        // are NOT sent, they are counted as not attempted
        assert_eq!(f.batches.lock().unwrap().len(), 1);
        assert_eq!((report.requested, report.skipped), (1000, 500));
        assert_eq!(report.list_problems.len(), 1);
        assert_eq!(f.keys().len(), 500);
        let text = format!(
            "{:#}",
            report.into_result("s3://bkt/big/", None).unwrap_err()
        );
        assert!(
            text.contains("Sent delete requests for 1000 of the 1500 listed"),
            "{text}"
        );
        assert!(
            text.contains("500 selected objects were NOT attempted"),
            "{text}"
        );
        assert!(text.contains("listing refused"), "{text}");
    }

    #[tokio::test]
    async fn a_listing_error_in_the_only_batch_sends_nothing() {
        let mut f = Fake::new(&["foo/a", "foo/b", "foo/c"]);
        f.list_fails_after = Some(2);
        let report = delete_objects_under_literal_prefix(
            &f,
            "s3://bkt/foo/",
            None,
            DeleteMode::Batched,
            None,
        )
        .await;
        assert!(f.batches.lock().unwrap().is_empty());
        assert_eq!((report.requested, report.skipped, report.listed), (0, 2, 2));
        assert_eq!(f.keys().len(), 3);
        assert!(report.into_result("s3://bkt/foo/", None).is_err());
    }

    #[tokio::test]
    async fn per_object_mode_sends_one_delete_per_selected_object_and_no_batch() {
        let f = Fake::new(&["foo/a", "foo/b", "foo/sub/c", "foobar", "other/x"]);
        let report = delete_objects_under_literal_prefix(
            &f,
            "s3://bkt/foo/",
            None,
            DeleteMode::PerObject,
            None,
        )
        .await;
        assert!(!report.failed());
        assert_eq!(
            (report.listed, report.selected, report.requested),
            (3, 3, 3)
        );
        assert_eq!(f.keys(), set(&["foobar", "other/x"]));
        let mut calls = f.calls();
        calls.sort();
        assert_eq!(
            calls,
            [
                "DELETE s3://bkt/foo/a",
                "DELETE s3://bkt/foo/b",
                "DELETE s3://bkt/foo/sub/c",
                "LIST_LITERAL s3://bkt/foo/",
            ]
        );
        assert!(f.batches.lock().unwrap().is_empty());
    }

    #[tokio::test]
    async fn per_object_mode_stops_after_a_failed_delete_and_counts_accurately() {
        let names: Vec<String> = (0..60).map(|i| format!("p/k{i:02}")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let mut f = Fake::new(&refs);
        f.delete_error_for = Some(("s3://bkt/p/k00".to_string(), http_403_text));
        let report = delete_objects_under_literal_prefix(
            &f,
            "s3://bkt/p/",
            None,
            DeleteMode::PerObject,
            None,
        )
        .await;
        assert_eq!(report.failed_requests, 1);
        assert!(
            report.requested <= DELETE_REQUESTS_IN_FLIGHT,
            "{}",
            report.requested
        );
        let deletes = f
            .calls()
            .iter()
            .filter(|c| c.starts_with("DELETE "))
            .count();
        assert_eq!(deletes, report.requested);
        // the failed key and everything not sent is still there
        assert!(f.keys().len() >= 60 - (report.requested - 1));
        assert!(f.keys().contains("s3://bkt/p/k00"));
        let text = format!("{:#}", report.into_result("s3://bkt/p/", None).unwrap_err());
        assert!(text.contains("failed to delete s3://bkt/p/k00"), "{text}");
        assert!(
            text.contains("1 delete request(s) reported errors"),
            "{text}"
        );
    }

    #[tokio::test]
    async fn per_object_mode_treats_a_vanished_object_as_deleted_and_honours_the_filter() {
        let mut f = Fake::new(&["p/a.txt", "p/b.log", "p/c.txt"]);
        f.delete_error_for = Some(("s3://bkt/p/a.txt".to_string(), io_not_found));
        let only_txt = |u: &str| u.ends_with(".txt");
        let report = delete_objects_under_literal_prefix(
            &f,
            "s3://bkt/p/",
            Some(&only_txt),
            DeleteMode::PerObject,
            None,
        )
        .await;
        assert!(!report.failed(), "{:?}", report.delete_problems);
        assert_eq!(
            (report.listed, report.selected, report.requested),
            (3, 2, 2)
        );
        assert!(f.keys().contains("s3://bkt/p/b.log"));
    }

    #[tokio::test]
    async fn prefix_delete_listing_failure_before_any_send_names_the_listing() {
        let mut f = Fake::new(&["foo/a", "foo/b"]);
        f.list_fails_after = Some(0);
        let report = delete_objects_under_literal_prefix(
            &f,
            "s3://bkt/foo/",
            None,
            DeleteMode::Batched,
            None,
        )
        .await;
        assert!(report.listing_failed_before_any_send());
        assert_eq!(f.keys(), set(&["foo/a", "foo/b"]));
        let text = format!(
            "{:#}",
            report.into_result("s3://bkt/foo/", None).unwrap_err()
        );
        assert!(
            text.contains("cannot list objects under prefix s3://bkt/foo/; nothing was deleted"),
            "{text}"
        );
        assert!(text.contains("listing refused"), "{text}");
    }

    #[tokio::test]
    async fn prefix_delete_on_a_backend_without_literal_listing_needs_a_trailing_slash() {
        let mut f = Fake::new(&["foo/a", "foobar"]);
        f.literal_listing = false;
        let report = delete_objects_under_literal_prefix(
            &f,
            "s3://bkt/foo",
            None,
            DeleteMode::Batched,
            None,
        )
        .await;
        assert!(report.listing_failed_before_any_send());
        let text = format!(
            "{:#}",
            report.into_result("s3://bkt/foo", None).unwrap_err()
        );
        assert!(text.contains("without a trailing '/'"), "{text}");
        assert_eq!(f.keys(), set(&["foo/a", "foobar"]));

        let report = delete_objects_under_literal_prefix(
            &f,
            "s3://bkt/foo/",
            None,
            DeleteMode::Batched,
            None,
        )
        .await;
        assert!(!report.failed());
        assert_eq!(f.keys(), set(&["foobar"]));
    }
}
