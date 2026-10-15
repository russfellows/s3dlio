// tests/test_per_endpoint_listing_two_servers_213.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// The s3:// selection rule (#213) on a PER-ENDPOINT store (#208), with the process-wide
// client pointing at a DIFFERENT server:
//
//   A = the process-wide client's server (`AWS_ENDPOINT_URL`); it holds the same bucket
//       with OTHER keys (`a-only/...`) and records every request it receives
//   B = the server `S3ObjectStore::for_endpoint(B)` talks to
//
// `list`, `list_stream`, `list_objects_with_prefix`, the delete-under-a-selection helper
// behind `s3-cli delete -r`, and `delete_prefix` must all run on B: the keys that come back
// are B's, B's objects are the ones removed, and A receives no request at all. The other
// #213 tests use ONE server for both roles, so a listing that wrongly used the process-wide
// client would still pass there.

#[path = "common/catalog_mock.rs"]
mod catalog_mock;

use catalog_mock::CatalogMock;
use futures::StreamExt;
use s3dlio::delete_contract::{delete_objects_under_literal_prefix, DeleteMode};
use s3dlio::object_store::{ObjectStore, S3ObjectStore};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Once, OnceLock};

static GLOBAL_A: OnceLock<CatalogMock> = OnceLock::new();
static ENV: Once = Once::new();
static N: AtomicU64 = AtomicU64::new(0);

fn server_a() -> &'static CatalogMock {
    let a = GLOBAL_A.get_or_init(|| CatalogMock::start(&[]));
    ENV.call_once(|| {
        std::env::set_var("AWS_ACCESS_KEY_ID", "test-key");
        std::env::set_var("AWS_SECRET_ACCESS_KEY", "test-secret");
        std::env::set_var("AWS_REGION", "us-east-1");
        std::env::set_var("AWS_ENDPOINT_URL", a.url());
        std::env::set_var("S3DLIO_MAX_RETRY_ATTEMPTS", "1");
    });
    a
}

const B_KEYS: &[&str] = &[
    "file.txt",
    "file.txt.bak",
    "dl/a.npz",
    "dl/b.txt",
    "dl/sub/c.npz",
    "dl2/x",
];

/// A fresh bucket on both servers: B holds `B_KEYS`, A holds keys of its own.
async fn setup(tag: &str) -> (&'static CatalogMock, CatalogMock, S3ObjectStore, String) {
    let a = server_a();
    let bucket = format!("t213x-{tag}-{}", N.fetch_add(1, Ordering::SeqCst));
    let b = CatalogMock::start(&[]);
    for key in B_KEYS {
        b.seed(&bucket, key, b"from-b");
    }
    for key in ["a-only/1", "a-only/2", "dl/a-side.npz"] {
        a.seed(&bucket, key, b"from-a");
    }
    // The process-wide client is built lazily from the environment set above.
    let store = S3ObjectStore::for_endpoint(&b.url())
        .await
        .expect("for_endpoint");
    (a, b, store, bucket)
}

fn uris(bucket: &str, keys: &[&str]) -> Vec<String> {
    let mut v: Vec<String> = keys.iter().map(|k| format!("s3://{bucket}/{k}")).collect();
    v.sort();
    v
}

fn assert_a_untouched(a: &CatalogMock, bucket: &str) {
    assert!(
        a.requests_for(bucket).is_empty(),
        "the process-wide server must receive no request: {:?}",
        a.requests_for(bucket)
    );
}

async fn collect(
    mut s: impl futures::Stream<Item = anyhow::Result<String>> + Unpin,
) -> Vec<String> {
    let mut v = Vec::new();
    while let Some(item) = s.next().await {
        v.push(item.expect("listing entry"));
    }
    v.sort();
    v
}

#[tokio::test]
async fn list_runs_on_the_store_server() {
    let (a, _b, store, bucket) = setup("list").await;

    let mut dir = store
        .list(&format!("s3://{bucket}/dl/"), true)
        .await
        .expect("list dir");
    dir.sort();
    assert_eq!(
        dir,
        uris(&bucket, &["dl/a.npz", "dl/b.txt", "dl/sub/c.npz"])
    );

    // exact name: the object, never file.txt.bak
    let exact = store
        .list(&format!("s3://{bucket}/file.txt"), true)
        .await
        .expect("list exact");
    assert_eq!(exact, uris(&bucket, &["file.txt"]));

    // glob
    let mut glob = store
        .list(&format!("s3://{bucket}/dl/*.npz"), true)
        .await
        .expect("list glob");
    glob.sort();
    assert_eq!(glob, uris(&bucket, &["dl/a.npz"]));

    // non-recursive: objects and directory entries of the level
    let mut level = store
        .list(&format!("s3://{bucket}/dl/"), false)
        .await
        .expect("list level");
    level.sort();
    assert_eq!(level, uris(&bucket, &["dl/a.npz", "dl/b.txt", "dl/sub/"]));

    assert_a_untouched(a, &bucket);
}

#[tokio::test]
async fn list_stream_runs_on_the_store_server() {
    let (a, _b, store, bucket) = setup("stream").await;
    let uri = format!("s3://{bucket}/dl/");

    let got = collect(store.list_stream(&uri, true)).await;
    assert_eq!(
        got,
        uris(&bucket, &["dl/a.npz", "dl/b.txt", "dl/sub/c.npz"])
    );

    let exact_uri = format!("s3://{bucket}/file.txt");
    let got = collect(store.list_stream(&exact_uri, true)).await;
    assert_eq!(got, uris(&bucket, &["file.txt"]));

    assert_a_untouched(a, &bucket);
}

#[tokio::test]
async fn list_objects_with_prefix_runs_on_the_store_server() {
    let (a, _b, store, bucket) = setup("owp").await;

    // a name: the object `dl`'s directory, never dl2
    let uri = format!("s3://{bucket}/dl");
    let got = collect(store.list_objects_with_prefix(&uri)).await;
    assert_eq!(
        got,
        uris(&bucket, &["dl/a.npz", "dl/b.txt", "dl/sub/c.npz"])
    );

    let uri = format!("s3://{bucket}/dl/*.npz");
    let got = collect(store.list_objects_with_prefix(&uri)).await;
    assert_eq!(got, uris(&bucket, &["dl/a.npz"]));

    assert_a_untouched(a, &bucket);
}

#[tokio::test]
async fn delete_under_a_selection_runs_on_the_store_server() {
    let (a, b, store, bucket) = setup("del").await;
    let uri = format!("s3://{bucket}/dl");

    let report =
        delete_objects_under_literal_prefix(&store, &uri, None, DeleteMode::Batched, None).await;

    assert!(!report.failed(), "{report:?}");
    assert_eq!(report.selected, 3, "{report:?}");
    let mut left = b.keys(&bucket);
    left.sort();
    assert_eq!(left, vec!["dl2/x", "file.txt", "file.txt.bak"]);
    assert_a_untouched(a, &bucket);
}

#[tokio::test]
async fn delete_prefix_runs_on_the_store_server() {
    let (a, b, store, bucket) = setup("delprefix").await;

    store
        .delete_prefix(&format!("s3://{bucket}/dl"))
        .await
        .expect("delete_prefix");

    let mut left = b.keys(&bucket);
    left.sort();
    assert_eq!(left, vec!["dl2/x", "file.txt", "file.txt.bak"]);
    assert_a_untouched(a, &bucket);
}
