// src/delete_contract.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! The delete contract shared by `s3-cli delete` (issues #210 / #167) and the
//! Python `s3dlio.delete()` (issue #212).
//!
//! The contract ("contract A"):
//!
//! * **Exact delete.**  `delete(uri)` deletes ONLY the object named by `uri`.
//!   The object is looked up first ([`ObjectStore::stat_if_exists`]): an absent
//!   object is a [`NoSuchObject`] error, and a lookup that fails for any other
//!   reason (permission denied, transport error, ...) is an error that deletes
//!   nothing.  Nothing is listed, no parent is consulted, and a trailing `/`,
//!   `*` or `?` is part of the key, never an instruction.
//! * **Recursive delete.**  [`delete_objects_under_literal_prefix`] deletes the
//!   REAL objects whose key starts with the literal prefix given
//!   ([`ObjectStore::list_objects_with_prefix`]: no delimiter, objects only,
//!   no `/` appended -- the prefix `foo` also matches `foobar`).
//!
//! The functions here select and delete; the callers decide how to word the
//! outcome (the CLI prints, Python raises).
//!
//! Both operations need read (HEAD / LIST) permission in addition to delete.

use anyhow::{bail, Context, Result};
use futures::stream::StreamExt;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::object_store::ObjectStore;

/// Keys per DeleteObjects request (the S3 API limit).
pub const DELETE_REQUEST_KEYS: usize = 1000;
/// DeleteObjects requests kept in flight at once by a recursive delete.
pub const DELETE_REQUESTS_IN_FLIGHT: usize = 10;

/// The exact object named for deletion does not exist.
///
/// Callers recover it with `err.downcast_ref::<NoSuchObject>()` (the Python API
/// maps it to `FileNotFoundError`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoSuchObject {
    /// The URI that was looked up.
    pub uri: String,
}

impl std::fmt::Display for NoSuchObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "no such object: {}", self.uri)
    }
}

impl std::error::Error for NoSuchObject {}

/// Delete exactly the object `uri`: look it up, then delete it.
///
/// Errors: [`NoSuchObject`] when the object is absent (nothing is sent); a
/// "cannot confirm that ... exists; nothing was deleted" error when the lookup
/// fails for any other reason (nothing is sent); "failed to delete ..." when
/// the delete itself fails.
pub async fn delete_exact_object(store: &dyn ObjectStore, uri: &str) -> Result<()> {
    let found = store
        .stat_if_exists(uri)
        .await
        .with_context(|| format!("cannot confirm that {uri} exists; nothing was deleted"))?;
    if found.is_none() {
        return Err(NoSuchObject {
            uri: uri.to_string(),
        }
        .into());
    }
    store
        .delete(uri)
        .await
        .with_context(|| format!("failed to delete {uri}"))
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

/// A predicate over the listed object URI (`s3://bkt/key`) narrowing the
/// selection.
pub type SelectionFilter<'a> = &'a (dyn Fn(&str) -> bool + Sync);
/// Called after each batch finishes with `(listed so far, delete requests sent
/// for so far)`; for progress display.
pub type ProgressHook<'a> = &'a (dyn Fn(u64, usize) + Sync);

/// Delete the real objects under the literal prefix `uri`, listed recursively
/// ([`ObjectStore::list_objects_with_prefix`]: no delimiter, objects only, no
/// `/` appended), optionally narrowed by `filter`.
///
/// After any failure (a delete request, a per-key error, the listing) no
/// further DeleteObjects request is SENT: the check is made immediately before
/// each send, so a batch that was only filled after the failure is skipped and
/// counted as not attempted.  Requests already in flight finish.  Never
/// panics; every failure is recorded in the returned report.
pub async fn delete_objects_under_literal_prefix(
    store: &dyn ObjectStore,
    uri: &str,
    filter: Option<SelectionFilter<'_>>,
    progress: Option<ProgressHook<'_>>,
) -> PrefixDeleteReport {
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
        .chunks(DELETE_REQUEST_KEYS)
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
            // The last look before sending: a batch filled after a failure is
            // not sent. (A listing error in this very batch does not stop its
            // own, already listed keys.)
            let stopped = stop_ref.load(Ordering::Relaxed);
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
            let result = store.delete_batch(&uris).await;
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
    use std::collections::{BTreeSet, HashMap};
    use std::pin::Pin;
    use std::sync::Mutex;

    /// How the fake answers `stat_if_exists`.
    #[derive(Clone, Copy, PartialEq)]
    enum Lookup {
        /// Present iff the key is in the fake's key set.
        FromKeys,
        /// Every lookup fails (e.g. HTTP 403).
        Fails,
    }

    /// An in-memory store that records every call. It relies on the trait
    /// DEFAULT `list_objects_with_prefix` (the non-S3 behaviour) unless
    /// `literal_listing` is set, in which case it mimics the S3 override.
    struct Fake {
        keys: Mutex<BTreeSet<String>>,
        lookup: Lookup,
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
                lookup: Lookup::FromKeys,
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

    fn stat_value() -> ObjectMetadata {
        ObjectMetadata {
            size: 1,
            last_modified: None,
            e_tag: None,
            content_type: None,
            content_language: None,
            content_encoding: None,
            cache_control: None,
            content_disposition: None,
            expires: None,
            storage_class: None,
            server_side_encryption: None,
            ssekms_key_id: None,
            sse_customer_algorithm: None,
            version_id: None,
            replication_status: None,
            metadata: HashMap::new(),
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
        async fn stat_if_exists(&self, uri: &str) -> Result<Option<ObjectMetadata>> {
            self.note(format!("HEAD {uri}"));
            match self.lookup {
                Lookup::Fails => bail!("HEAD failed (HTTP 403)"),
                Lookup::FromKeys => Ok(self.keys().contains(uri).then(stat_value)),
            }
        }
        async fn delete(&self, uri: &str) -> Result<()> {
            self.note(format!("DELETE {uri}"));
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

    #[tokio::test]
    async fn exact_delete_removes_only_the_named_object() {
        let f = Fake::new(&["foo", "foo/", "foo/x", "foobar"]);
        delete_exact_object(&f, "s3://bkt/foo").await.unwrap();
        assert_eq!(f.keys(), set(&["foo/", "foo/x", "foobar"]));
        assert_eq!(f.calls(), ["HEAD s3://bkt/foo", "DELETE s3://bkt/foo"]);
    }

    #[tokio::test]
    async fn exact_delete_trailing_slash_names_the_marker_object_and_never_lists() {
        let f = Fake::new(&["foo/", "foo/x", "foo/sub/y"]);
        delete_exact_object(&f, "s3://bkt/foo/").await.unwrap();
        assert_eq!(f.keys(), set(&["foo/x", "foo/sub/y"]));
        assert_eq!(f.calls(), ["HEAD s3://bkt/foo/", "DELETE s3://bkt/foo/"]);
    }

    #[tokio::test]
    async fn exact_delete_of_a_missing_object_is_no_such_object_and_sends_no_delete() {
        let f = Fake::new(&["foo/x", "other/"]);
        let err = delete_exact_object(&f, "s3://bkt/foo/").await.unwrap_err();
        let missing = err.downcast_ref::<NoSuchObject>().expect("typed error");
        assert_eq!(missing.uri, "s3://bkt/foo/");
        assert_eq!(f.keys(), set(&["foo/x", "other/"]));
        assert_eq!(f.calls(), ["HEAD s3://bkt/foo/"]);
    }

    #[tokio::test]
    async fn exact_delete_lookup_failure_is_not_no_such_object_and_sends_no_delete() {
        let mut f = Fake::new(&["foo"]);
        f.lookup = Lookup::Fails;
        let err = delete_exact_object(&f, "s3://bkt/foo").await.unwrap_err();
        assert!(err.downcast_ref::<NoSuchObject>().is_none(), "{err:#}");
        let text = format!("{err:#}");
        assert!(
            text.contains("cannot confirm that s3://bkt/foo exists"),
            "{text}"
        );
        assert!(
            text.contains("nothing was deleted") && text.contains("403"),
            "{text}"
        );
        assert_eq!(f.keys(), set(&["foo"]));
        assert_eq!(f.calls(), ["HEAD s3://bkt/foo"]);
    }

    #[tokio::test]
    async fn wildcard_characters_in_an_exact_delete_are_part_of_the_key() {
        let f = Fake::new(&["a*", "ab", "a?b", "axb"]);
        delete_exact_object(&f, "s3://bkt/a*").await.unwrap();
        delete_exact_object(&f, "s3://bkt/a?b").await.unwrap();
        assert_eq!(f.keys(), set(&["ab", "axb"]));
        assert!(
            f.calls().iter().all(|c| !c.starts_with("LIST")),
            "{:?}",
            f.calls()
        );
    }

    // ---- recursive delete --------------------------------------------------

    #[tokio::test]
    async fn prefix_delete_deletes_the_listed_objects_under_the_literal_prefix() {
        let f = Fake::new(&["foo", "foo/", "foo/x", "foobar", "other/x"]);
        let report = delete_objects_under_literal_prefix(&f, "s3://bkt/foo", None, None).await;
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
        let report =
            delete_objects_under_literal_prefix(&f, "s3://bkt/p/", Some(&only_txt), None).await;
        assert_eq!(
            (report.listed, report.selected, report.requested),
            (3, 2, 2)
        );
        assert_eq!(f.keys(), set(&["p/b.log"]));
    }

    #[tokio::test]
    async fn prefix_delete_with_nothing_under_the_prefix_sends_nothing() {
        let f = Fake::new(&["other/x"]);
        let report = delete_objects_under_literal_prefix(&f, "s3://bkt/foo/", None, None).await;
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
        let report = delete_objects_under_literal_prefix(&f, "s3://bkt/big/", None, None).await;
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
        let report = delete_objects_under_literal_prefix(&f, "s3://bkt/big/", None, None).await;
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
    async fn prefix_delete_listing_failure_before_any_send_names_the_listing() {
        let mut f = Fake::new(&["foo/a", "foo/b"]);
        f.list_fails_after = Some(0);
        let report = delete_objects_under_literal_prefix(&f, "s3://bkt/foo/", None, None).await;
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
        let report = delete_objects_under_literal_prefix(&f, "s3://bkt/foo", None, None).await;
        assert!(report.listing_failed_before_any_send());
        let text = format!(
            "{:#}",
            report.into_result("s3://bkt/foo", None).unwrap_err()
        );
        assert!(text.contains("without a trailing '/'"), "{text}");
        assert_eq!(f.keys(), set(&["foo/a", "foobar"]));

        let report = delete_objects_under_literal_prefix(&f, "s3://bkt/foo/", None, None).await;
        assert!(!report.failed());
        assert_eq!(f.keys(), set(&["foobar"]));
    }
}
