// Pure planning helpers for `ObjectStore::delete_batch` (issue #179).
//
// A batch may name objects in several buckets / containers.  Bulk-delete APIs
// (S3 DeleteObjects, GCS per-object deletes, Azure container deletes) address
// exactly one bucket per request, so the caller's URIs must be grouped by
// bucket before any request is issued.  The logic lives here, free of any
// network or SDK types, so it can be unit-tested directly.
//
// A store also talks to exactly ONE server.  A URI of the host form
// (`s3://host:port/bucket/key`) names a server; a batch containing a URI that
// names a server other than the store's own is rejected up front, because the
// store would otherwise silently apply that URI's bucket/key to ITS server
// (deleting the wrong object, and never the one the caller named).
//
// SPDX-License-Identifier: Apache-2.0 OR MIT

use anyhow::{anyhow, bail, Context, Result};
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
/// Every URI is validated before anything is returned: if `parse` rejects any
/// URI (unparseable, or -- for S3 -- naming a server other than the store's
/// own) the whole call fails, naming each offending URI and why, so that the
/// caller deletes nothing from a batch it cannot fully interpret.
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
            "delete_batch: {} of {} URI(s) rejected (unparseable, or naming a server other \
             than this store's); nothing was deleted: {}",
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
/// batch succeeded.  Otherwise the FIRST failure's original error is the
/// returned error's source (its whole cause chain survives, so callers can
/// still `downcast_ref`/walk `chain()`), with the batch summary as added
/// context.  Any further failing buckets are listed in that summary (as text:
/// only one error can be the source), each with its URIs not deleted.
pub(crate) fn aggregate_failures(total: usize, failures: Vec<GroupFailure>) -> Result<()> {
    let failed: usize = failures.iter().map(|f| f.uris.len()).sum();
    let n = failures.len();
    let mut it = failures.into_iter();
    let Some(first) = it.next() else {
        return Ok(());
    };
    let mut message = format!(
        "delete_batch: {failed} of {total} URI(s) not confirmed deleted in {}: {}",
        first.target,
        summarize(&first.uris)
    );
    if n > 1 {
        let others = it
            .map(|f| format!("{} [{:#}]: {}", f.target, f.error, summarize(&f.uris)))
            .collect::<Vec<_>>()
            .join("; ");
        message.push_str(&format!(
            "; {} other bucket(s) also failed: {others}",
            n - 1
        ));
    }
    Err(first.error.context(message))
}

/// The one server a store talks to, normalised for comparison with the server
/// a URI names: lower-cased host (no brackets, no trailing dot) and a numeric
/// port (the scheme's default when the endpoint URL gives none).
///
/// No DNS resolution is done: `localhost` and `127.0.0.1` are different
/// servers here, and so are two names for one machine.  Refusing to delete is
/// the safe direction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StoreServer {
    host: String,
    port: u16,
    /// Port assumed for a URI authority that carries none.
    default_port: u16,
}

impl StoreServer {
    /// From a store's endpoint URL (`http://host:port`, `https://host`, ...).
    pub(crate) fn from_endpoint_url(url: &str) -> Result<Self> {
        let (default_port, rest) = if let Some(r) = url.strip_prefix("https://") {
            (443u16, r)
        } else if let Some(r) = url.strip_prefix("http://") {
            (80u16, r)
        } else {
            // A bare `host:port` (how multi-endpoint hands one over).
            (80u16, url)
        };
        let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
        let (host, port) = split_authority(authority)?;
        Ok(Self {
            host,
            port: port.unwrap_or(default_port),
            default_port,
        })
    }

    /// Does the server named by a URI authority (`host:port`, `[v6]:port`) equal
    /// this store's server?  An authority without a port is compared at the
    /// store's scheme-default port.
    pub(crate) fn is_named_by(&self, authority: &str) -> Result<bool> {
        let (host, port) = split_authority(authority)?;
        Ok(host == self.host && port.unwrap_or(self.default_port) == self.port)
    }

    fn display(&self) -> String {
        if self.host.contains(':') {
            format!("[{}]:{}", self.host, self.port)
        } else {
            format!("{}:{}", self.host, self.port)
        }
    }
}

/// Split `host`, `host:port` or `[v6]:port` into a normalised host and port.
fn split_authority(authority: &str) -> Result<(String, Option<u16>)> {
    let parse_port = |p: &str| -> Result<u16> {
        p.parse::<u16>()
            .with_context(|| format!("invalid port '{p}' in '{authority}'"))
    };
    let (host, port) = if let Some(inner) = authority.strip_prefix('[') {
        let (host, after) = inner
            .split_once(']')
            .with_context(|| format!("unterminated '[' in '{authority}'"))?;
        let port = match after.strip_prefix(':') {
            Some(p) => Some(parse_port(p)?),
            None if after.is_empty() => None,
            None => bail!("unexpected text after ']' in '{authority}'"),
        };
        (host, port)
    } else {
        match authority.split_once(':') {
            Some((host, p)) => (host, Some(parse_port(p)?)),
            None => (authority, None),
        }
    };
    if host.is_empty() {
        bail!("empty host in '{authority}'");
    }
    Ok((host.trim_end_matches('.').to_ascii_lowercase(), port))
}

/// Parse one S3 URI for a store whose server is `store` (`None` = the store
/// uses the process-wide client, whose server comes from the environment).
///
/// Returns `(bucket, key)`.  A URI that names a server (the host form,
/// `s3://host:port/bucket/key`) is accepted only if that is the store's own
/// server; a bucket-form URI (`s3://bucket/key`) names none and is always
/// accepted.  Whether a given URI "names a server" is decided solely by
/// [`crate::s3_utils::parse_s3_uri_full`].
///
/// Because every accepted URI addresses the store's single server, grouping
/// by `(server, bucket)` is grouping by `bucket`.
pub(crate) fn parse_s3_uri_for_server(
    uri: &str,
    store: Option<&StoreServer>,
) -> Result<(String, String)> {
    let c = crate::s3_utils::parse_s3_uri_full(uri)?;
    if let Some(named) = c.endpoint.as_deref() {
        match store {
            Some(server) => {
                if !server.is_named_by(named)? {
                    bail!(
                        "names server '{named}' but this store talks to {}",
                        server.display()
                    );
                }
            }
            None => bail!(
                "names server '{named}' but this store uses the process-wide S3 client, \
                 which cannot address a server named in a URI"
            ),
        }
    }
    Ok((c.bucket, c.key))
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

    #[test]
    fn aggregate_multiple_failures_keeps_the_first_cause_chain() {
        let root = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "denied-x");
        let first = anyhow::Error::new(root).context("S3 batch DELETE failed for bucket 'x'");
        let err = aggregate_failures(
            3,
            vec![
                GroupFailure {
                    target: "s3://x".into(),
                    uris: s(&["s3://x/1"]),
                    error: first,
                },
                GroupFailure {
                    target: "s3://y".into(),
                    uris: s(&["s3://y/1", "s3://y/2"]),
                    error: anyhow!("boom-y"),
                },
            ],
        )
        .unwrap_err();
        // summary + first's context + io::Error root = a 3-link chain, and the
        // typed root is still reachable (not flattened to a string).
        assert_eq!(err.chain().count(), 3, "{err:#}");
        let io = err
            .root_cause()
            .downcast_ref::<std::io::Error>()
            .expect("typed root cause survives");
        assert_eq!(io.kind(), std::io::ErrorKind::PermissionDenied);
        let top = err.to_string();
        assert!(
            top.contains("s3://x/1") && top.contains("1 other bucket(s)"),
            "{top}"
        );
        assert!(top.contains("boom-y") && top.contains("s3://y/2"), "{top}");
        assert!(top.contains("3 of 3"), "{top}");
    }

    fn server(url: &str) -> StoreServer {
        StoreServer::from_endpoint_url(url).unwrap()
    }

    fn parse_for(url: Option<&str>, uri: &str) -> Result<(String, String)> {
        let store = url.map(server);
        parse_s3_uri_for_server(uri, store.as_ref())
    }

    #[test]
    fn bucket_form_uris_are_always_accepted() {
        for url in [
            None,
            Some("http://127.0.0.1:9000"),
            Some("https://s.example.com"),
        ] {
            assert_eq!(
                parse_for(url, "s3://bkt/dir/k").unwrap(),
                ("bkt".to_string(), "dir/k".to_string())
            );
        }
    }

    #[test]
    fn host_form_uri_for_the_same_server_is_accepted_after_normalising() {
        let ok = |url: &str, uri: &str| {
            assert_eq!(
                parse_for(Some(url), uri).unwrap_or_else(|e| panic!("{uri}: {e:#}")),
                ("bkt".to_string(), "k".to_string()),
                "{uri}"
            )
        };
        ok("http://127.0.0.1:9000", "s3://127.0.0.1:9000/bkt/k");
        ok("http://MinIO.Local:9000/", "s3://minio.local:9000/bkt/k");
        ok("http://minio.local:9000", "s3://MINIO.LOCAL:9000/bkt/k");
        ok("http://minio.local.:9000", "s3://minio.local:9000/bkt/k");
        // https default port is 443, http default 80.
        ok(
            "https://storage.googleapis.com",
            "s3://storage.googleapis.com:443/bkt/k",
        );
        ok(
            "https://storage.googleapis.com:443",
            "s3://storage.googleapis.com:443/bkt/k",
        );
        ok("http://h.example", "s3://h.example:80/bkt/k");
        // IPv6.
        ok("http://[::1]:9000", "s3://[::1]:9000/bkt/k");
        ok("http://[2001:DB8::1]:9000", "s3://[2001:db8::1]:9000/bkt/k");
    }

    #[test]
    fn host_form_uri_for_another_server_is_rejected() {
        let bad = |url: &str, uri: &str| {
            let e = parse_for(Some(url), uri).expect_err(uri);
            assert!(format!("{e:#}").contains("names server"), "{uri}: {e:#}");
        };
        bad("http://127.0.0.1:9000", "s3://127.0.0.1:9/bkt/k");
        bad("http://127.0.0.1:9000", "s3://127.0.0.2:9000/bkt/k");
        bad("http://minio.local:9000", "s3://other.local:9000/bkt/k");
        // no DNS: two names for one machine are different servers
        bad("http://127.0.0.1:9000", "s3://localhost:9000/bkt/k");
        // an http store is not the https server of the same name
        bad("http://h.example:80", "s3://h.example:443/bkt/k");
        bad("https://h.example", "s3://h.example:80/bkt/k");
        bad("http://[::1]:9000", "s3://[::1]:9001/bkt/k");
    }

    #[test]
    fn global_client_store_rejects_every_host_form_uri() {
        let e = parse_for(None, "s3://host:9000/bkt/k").unwrap_err();
        assert!(format!("{e:#}").contains("process-wide"), "{e:#}");
        assert!(parse_for(None, "s3://[::1]:9000/bkt/k").is_err());
    }

    #[test]
    fn malformed_authorities_are_errors_not_matches() {
        assert!(StoreServer::from_endpoint_url("http://host:notaport").is_err());
        assert!(StoreServer::from_endpoint_url("http://[::1").is_err());
        assert!(StoreServer::from_endpoint_url("http://").is_err());
        assert!(parse_for(Some("http://h:1"), "s3://h:99999/bkt/k").is_err());
    }

    #[test]
    fn a_batch_naming_another_server_groups_to_an_error_naming_it() {
        let store = server("http://hostA:9000");
        let err = group_uris_by_bucket(
            &s(&[
                "s3://hostA:9000/data/k",
                "s3://hostB:9000/data/k",
                "s3://data/k",
            ]),
            |u| parse_s3_uri_for_server(u, Some(&store)),
        )
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("s3://hostB:9000/data/k"), "{msg}");
        assert!(!msg.contains("s3://hostA:9000/data/k"), "{msg}");
        assert!(msg.contains("1 of 3"), "{msg}");
    }
}
