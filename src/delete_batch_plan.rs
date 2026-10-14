// Pure planning helpers for `ObjectStore::delete_batch` (issue #179).
//
// A batch may name objects in several buckets / containers.  Bulk-delete APIs
// (S3 DeleteObjects, GCS per-object deletes, Azure container deletes) address
// exactly one bucket per request, so the caller's URIs must be grouped by
// bucket before any request is issued.  The logic lives here, free of any
// network or SDK types, so it can be unit-tested directly.
//
// SPDX-License-Identifier: Apache-2.0 OR MIT

use anyhow::{anyhow, Result};
use std::collections::HashMap;
use std::hash::Hash;

/// Maximum keys per bulk-delete request (the S3 DeleteObjects limit).
pub(crate) const DELETE_BATCH_MAX_KEYS: usize = 1000;

/// Maximum number of URIs spelled out per bucket / per category in an error
/// message.  The remainder is summarised as a count.
const MAX_URIS_IN_ERROR: usize = 20;

/// The URIs of one batch that live in the same bucket / container.
///
/// `uris` and `keys` are parallel: `keys[i]` is the object key of `uris[i]`.
/// Both preserve the caller's relative order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BucketGroup<K> {
    pub target: K,
    pub uris: Vec<String>,
    pub keys: Vec<String>,
}

impl<K> BucketGroup<K> {
    /// Split into request-sized `(uris, keys)` slices of at most
    /// `DELETE_BATCH_MAX_KEYS` entries each.
    pub fn chunks(&self) -> impl Iterator<Item = (&[String], &[String])> {
        self.uris
            .chunks(DELETE_BATCH_MAX_KEYS)
            .zip(self.keys.chunks(DELETE_BATCH_MAX_KEYS))
    }
}

/// Group `uris` by the bucket / container `parse` extracts from each one.
///
/// Groups appear in order of first occurrence; within a group the caller's
/// order is preserved.  `parse` maps a URI to `(bucket-identity, object-key)`.
///
/// Every URI is validated before anything is returned: if any URI cannot be
/// parsed the whole call fails (naming the offending URIs) so that the caller
/// deletes nothing from a batch it cannot fully interpret.
pub(crate) fn group_uris_by_bucket<K, F>(uris: &[String], parse: F) -> Result<Vec<BucketGroup<K>>>
where
    K: Eq + Hash + Clone,
    F: Fn(&str) -> Result<(K, String)>,
{
    let mut index: HashMap<K, usize> = HashMap::new();
    let mut groups: Vec<BucketGroup<K>> = Vec::new();
    let mut bad: Vec<String> = Vec::new();

    for uri in uris {
        match parse(uri) {
            Ok((target, key)) => {
                let slot = match index.get(&target) {
                    Some(&i) => i,
                    None => {
                        groups.push(BucketGroup {
                            target: target.clone(),
                            uris: Vec::new(),
                            keys: Vec::new(),
                        });
                        let i = groups.len() - 1;
                        index.insert(target, i);
                        i
                    }
                };
                // `slot` was just looked up or pushed, so it is in range; use
                // `get_mut` rather than indexing to stay panic-free.
                if let Some(g) = groups.get_mut(slot) {
                    g.uris.push(uri.clone());
                    g.keys.push(key);
                }
            }
            Err(e) => bad.push(format!("'{uri}' ({e:#})")),
        }
    }

    if !bad.is_empty() {
        return Err(anyhow!(
            "delete_batch: {} of {} URI(s) could not be parsed; nothing was deleted: {}",
            bad.len(),
            uris.len(),
            summarize(&bad)
        ));
    }
    Ok(groups)
}

/// Render at most `MAX_URIS_IN_ERROR` items, then "... and N more".
fn summarize(items: &[String]) -> String {
    let shown = items
        .iter()
        .take(MAX_URIS_IN_ERROR)
        .map(String::as_str)
        .collect::<Vec<_>>()
        .join(", ");
    if items.len() > MAX_URIS_IN_ERROR {
        format!("{shown}, ... and {} more", items.len() - MAX_URIS_IN_ERROR)
    } else {
        shown
    }
}

/// One bucket's failure within a batch delete.
#[derive(Debug)]
pub(crate) struct GroupFailure {
    /// Human-readable bucket / container identity (e.g. `s3://bucket`).
    pub target: String,
    /// The URIs that were not (or may not have been) deleted.
    pub uris: Vec<String>,
    pub error: anyhow::Error,
}

/// Fold per-bucket failures into the batch result.
///
/// `total` is the number of URIs in the whole batch.  With no failures the
/// batch succeeded.  A single failing bucket keeps its original error chain
/// (the batch error is that error with the failed URIs as added context);
/// several failing buckets are combined into one error that names each bucket,
/// its error, and the URIs that were not deleted.
pub(crate) fn aggregate_failures(total: usize, mut failures: Vec<GroupFailure>) -> Result<()> {
    let failed: usize = failures.iter().map(|f| f.uris.len()).sum();
    match failures.len() {
        0 => Ok(()),
        1 => match failures.pop() {
            Some(f) => Err(f.error.context(format!(
                "delete_batch: {failed} of {total} URI(s) not confirmed deleted in {}: {}",
                f.target,
                summarize(&f.uris)
            ))),
            None => Ok(()),
        },
        n => {
            let detail = failures
                .iter()
                .map(|f| format!("{} [{:#}]: {}", f.target, f.error, summarize(&f.uris)))
                .collect::<Vec<_>>()
                .join("; ");
            Err(anyhow!(
                "delete_batch: {failed} of {total} URI(s) not confirmed deleted across {n} bucket(s): {detail}"
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::s3_utils::parse_s3_uri;

    fn s(v: &[&str]) -> Vec<String> {
        v.iter().map(|x| x.to_string()).collect()
    }

    #[test]
    fn mixed_buckets_are_split_and_keys_follow_their_bucket() {
        let g = group_uris_by_bucket(&s(&["s3://a/p/a", "s3://b/p/b"]), parse_s3_uri).unwrap();
        assert_eq!(g.len(), 2);
        assert_eq!(g[0].target, "a");
        assert_eq!(g[0].keys, vec!["p/a"]);
        assert_eq!(g[0].uris, vec!["s3://a/p/a"]);
        assert_eq!(g[1].target, "b");
        assert_eq!(g[1].keys, vec!["p/b"]);
    }

    #[test]
    fn groups_in_first_seen_order_and_caller_order_within_group() {
        let g = group_uris_by_bucket(
            &s(&[
                "s3://b/1", "s3://a/1", "s3://b/2", "s3://a/2", "s3://c/1", "s3://b/3",
            ]),
            parse_s3_uri,
        )
        .unwrap();
        let targets: Vec<&str> = g.iter().map(|x| x.target.as_str()).collect();
        assert_eq!(targets, vec!["b", "a", "c"]);
        assert_eq!(g[0].keys, vec!["1", "2", "3"]);
        assert_eq!(g[1].keys, vec!["1", "2"]);
        assert_eq!(g[2].keys, vec!["1"]);
    }

    #[test]
    fn single_bucket_is_one_group_with_everything() {
        let g =
            group_uris_by_bucket(&s(&["s3://a/x", "s3://a/y", "s3://a/z"]), parse_s3_uri).unwrap();
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].keys, vec!["x", "y", "z"]);
    }

    #[test]
    fn empty_input_yields_no_groups() {
        assert!(group_uris_by_bucket(&[], parse_s3_uri).unwrap().is_empty());
    }

    #[test]
    fn chunks_split_at_1000_and_stay_parallel() {
        let uris: Vec<String> = (0..2500).map(|i| format!("s3://a/k{i}")).collect();
        let g = group_uris_by_bucket(&uris, parse_s3_uri).unwrap();
        assert_eq!(g.len(), 1);
        let sizes: Vec<usize> = g[0].chunks().map(|(u, _)| u.len()).collect();
        assert_eq!(sizes, vec![1000, 1000, 500]);
        for (u, k) in g[0].chunks() {
            assert_eq!(u.len(), k.len());
            for (uri, key) in u.iter().zip(k.iter()) {
                assert!(uri.ends_with(key.as_str()));
            }
        }
        // Exactly 1000 keys is one request, 1001 is two.
        let exact: Vec<String> = (0..1000).map(|i| format!("s3://a/k{i}")).collect();
        let g = group_uris_by_bucket(&exact, parse_s3_uri).unwrap();
        assert_eq!(g[0].chunks().count(), 1);
        let over: Vec<String> = (0..1001).map(|i| format!("s3://a/k{i}")).collect();
        let g = group_uris_by_bucket(&over, parse_s3_uri).unwrap();
        assert_eq!(g[0].chunks().count(), 2);
    }

    #[test]
    fn unparseable_uri_is_an_error_naming_it_wherever_it_sits() {
        for batch in [
            s(&["bogus", "s3://a/x"]),
            s(&["s3://a/x", "bogus", "s3://b/y"]),
            s(&["s3://a/x", "s3://b/y", "bogus"]),
        ] {
            let err = group_uris_by_bucket(&batch, parse_s3_uri).unwrap_err();
            assert!(format!("{err:#}").contains("bogus"), "{err:#}");
        }
    }

    #[test]
    fn every_unparseable_uri_is_reported() {
        let err = group_uris_by_bucket(
            &s(&["s3://a/x", "bad-one", "s3://b/y", "bad-two"]),
            parse_s3_uri,
        )
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("bad-one") && msg.contains("bad-two"), "{msg}");
        assert!(
            !msg.contains("s3://a/x"),
            "valid URIs are not listed: {msg}"
        );
    }

    #[test]
    fn aggregate_with_no_failures_is_ok() {
        assert!(aggregate_failures(3, Vec::new()).is_ok());
    }

    #[test]
    fn aggregate_single_failure_keeps_the_error_chain_and_names_uris() {
        let root = anyhow!("AccessDenied");
        let err = aggregate_failures(
            3,
            vec![GroupFailure {
                target: "s3://bad".into(),
                uris: s(&["s3://bad/one", "s3://bad/two"]),
                error: root,
            }],
        )
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("AccessDenied"), "{msg}");
        assert!(
            msg.contains("s3://bad/one") && msg.contains("s3://bad/two"),
            "{msg}"
        );
        assert!(msg.contains("2 of 3"), "{msg}");
    }

    #[test]
    fn aggregate_multiple_failures_reports_each_bucket() {
        let err = aggregate_failures(
            5,
            vec![
                GroupFailure {
                    target: "s3://x".into(),
                    uris: s(&["s3://x/1"]),
                    error: anyhow!("boom-x"),
                },
                GroupFailure {
                    target: "s3://y".into(),
                    uris: s(&["s3://y/1", "s3://y/2"]),
                    error: anyhow!("boom-y"),
                },
            ],
        )
        .unwrap_err();
        let msg = format!("{err:#}");
        for needle in [
            "boom-x", "boom-y", "s3://x/1", "s3://y/1", "s3://y/2", "3 of 5",
        ] {
            assert!(msg.contains(needle), "missing {needle}: {msg}");
        }
    }

    #[test]
    fn aggregate_truncates_long_uri_lists() {
        let many: Vec<String> = (0..100).map(|i| format!("s3://x/{i}")).collect();
        let err = aggregate_failures(
            100,
            vec![GroupFailure {
                target: "s3://x".into(),
                uris: many,
                error: anyhow!("boom"),
            }],
        )
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("s3://x/0"), "{msg}");
        assert!(!msg.contains("s3://x/99"), "list must be truncated: {msg}");
        assert!(msg.contains("80 more"), "{msg}");
    }
}
