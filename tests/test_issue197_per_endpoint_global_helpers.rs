// tests/test_issue197_per_endpoint_global_helpers.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Regression tests for s3dlio issue #197 (review follow-up): a per-endpoint
// `S3ObjectStore` bound to server B must never hand an operation to a helper that
// uses the process-wide client (server A, `AWS_ENDPOINT_URL`) -- not even for a
// bucket-form URI (`s3://bucket/key`), which names no server and so passes the URI
// check. Per-endpoint multipart / concurrent-range support is issue #208; until then
// those operations are REJECTED, with an error naming the store's server, and no
// server receives a request.
//
// A store bound to the global server (or the default store) is unaffected.
//
//   A = the process-wide (global) client's server (`AWS_ENDPOINT_URL`)
//   B = the server a per-endpoint store talks to
//
// One global mock A for the whole binary (the library's global client is
// process-wide); every test uses a bucket name unique to it.

#[path = "common/recording_s3.rs"]
mod recording_s3;
use recording_s3::MockS3;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Once, OnceLock};

use bytes::Bytes;
use s3dlio::object_store::{CompressionConfig, ObjectStore, S3ObjectStore, WriterOptions};

static GLOBAL_A: OnceLock<MockS3> = OnceLock::new();
static ENV: Once = Once::new();
static COUNTER: AtomicUsize = AtomicUsize::new(0);

/// A, the server `AWS_ENDPOINT_URL` names. The environment is set before anything
/// touches the process-wide client (it is built once from it).
fn global_a() -> &'static MockS3 {
    ENV.call_once(|| {
        let a = MockS3::start(&[]);
        #[allow(deprecated)]
        {
            std::env::set_var("AWS_ACCESS_KEY_ID", "test-key");
            std::env::set_var("AWS_SECRET_ACCESS_KEY", "test-secret");
            std::env::set_var("AWS_REGION", "us-east-1");
            std::env::remove_var("AWS_ENDPOINT_URL_S3");
            std::env::set_var("AWS_ENDPOINT_URL", a.url());
            // Every object counts as "large": get_optimized takes the concurrent path.
            std::env::set_var("S3DLIO_CONCURRENT_THRESHOLD", "0");
        }
        let _ = GLOBAL_A.set(a);
    });
    GLOBAL_A.get().expect("global mock set by call_once")
}

fn unique_bucket(tag: &str) -> String {
    format!(
        "b197d-{tag}-{}",
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

fn mentioning(m: &MockS3, needle: &str) -> Vec<String> {
    m.requests()
        .into_iter()
        .filter(|r| r.contains(needle))
        .collect()
}

async fn store_for(server: &MockS3) -> S3ObjectStore {
    global_a();
    S3ObjectStore::for_endpoint(&server.url())
        .await
        .expect("for_endpoint with fake credentials")
}

/// The error must say what is wrong: whose server, and that it is not supported yet.
fn assert_unsupported_per_endpoint(err: &anyhow::Error, b: &MockS3) {
    let text = format!("{err:#}");
    assert!(
        text.contains(&format!("127.0.0.1:{}", b.port)),
        "the error must name the store's server, got: {text}"
    );
    assert!(
        text.contains("not yet supported") && text.contains("#208"),
        "the error must say the operation is not yet supported per-endpoint (#208), got: {text}"
    );
}

#[tokio::test]
async fn per_endpoint_put_multipart_bucket_form_is_rejected_and_sends_nothing() {
    let a = global_a();
    let b = MockS3::start(&[]);
    let store = store_for(&b).await;
    let bucket = unique_bucket("mp-bucket");

    let err = store
        .put_multipart(
            &format!("s3://{bucket}/key"),
            Bytes::from_static(b"x"),
            None,
        )
        .await
        .expect_err("a per-endpoint store must not run multipart on the global server");

    assert_unsupported_per_endpoint(&err, &b);
    assert_eq!(mentioning(a, &bucket), Vec::<String>::new(), "A got a request");
    assert_eq!(b.requests(), Vec::<String>::new(), "B got a request");
}

#[tokio::test]
async fn per_endpoint_put_multipart_host_form_is_rejected_and_sends_nothing() {
    let a = global_a();
    let b = MockS3::start(&[]);
    let store = store_for(&b).await;
    let bucket = unique_bucket("mp-host");

    let err = store
        .put_multipart(
            &format!("s3://127.0.0.1:{}/{bucket}/key", b.port),
            Bytes::from_static(b"x"),
            None,
        )
        .await
        .expect_err("multipart is not supported per-endpoint");

    assert_unsupported_per_endpoint(&err, &b);
    assert_eq!(mentioning(a, &bucket), Vec::<String>::new(), "A got a request");
    assert_eq!(b.requests(), Vec::<String>::new(), "B got a request");
}

#[tokio::test]
async fn per_endpoint_get_range_optimized_large_is_rejected_and_sends_nothing() {
    let a = global_a();
    let b = MockS3::start(&[]);
    let store = store_for(&b).await;
    let bucket = unique_bucket("rng");

    // An explicit length: no stat, so the concurrent-range path is chosen from the
    // length alone.
    let err = store
        .get_range_optimized(
            &format!("s3://{bucket}/key"),
            0,
            Some(64 * 1024 * 1024),
            None,
            None,
        )
        .await
        .expect_err("concurrent range GET is not supported per-endpoint");

    assert_unsupported_per_endpoint(&err, &b);
    assert_eq!(mentioning(a, &bucket), Vec::<String>::new(), "A got a request");
    assert_eq!(b.requests(), Vec::<String>::new(), "B got a request");
}

#[tokio::test]
async fn per_endpoint_get_optimized_large_is_rejected_and_never_reaches_a() {
    let a = global_a();
    let b = MockS3::start(&[]);
    let store = store_for(&b).await;
    let bucket = unique_bucket("get");

    // S3DLIO_CONCURRENT_THRESHOLD=0: the object is "large". The size lookup (HEAD)
    // legitimately goes to B through the store's own client; the data fetch must not.
    let err = store
        .get_optimized(&format!("s3://{bucket}/key"))
        .await
        .expect_err("concurrent range GET is not supported per-endpoint");

    assert_unsupported_per_endpoint(&err, &b);
    assert_eq!(mentioning(a, &bucket), Vec::<String>::new(), "A got a request");
    assert!(
        b.requests().iter().all(|r| r.starts_with("HEAD ")),
        "B may only see the size lookup, got {:?}",
        b.requests()
    );
}

#[tokio::test]
async fn per_endpoint_writers_are_rejected_when_created_and_send_nothing() {
    let a = global_a();
    let b = MockS3::start(&[]);
    let store = store_for(&b).await;
    let bucket = unique_bucket("wr");
    let uri = format!("s3://{bucket}/key");

    // `Box<dyn ObjectWriter>` is not Debug, so no `expect_err`.
    let results = [
        ("get_writer", store.get_writer(&uri).await),
        (
            "get_writer_with_compression",
            store
                .get_writer_with_compression(&uri, CompressionConfig::zstd_default())
                .await,
        ),
        (
            "create_writer",
            store.create_writer(&uri, WriterOptions::default()).await,
        ),
    ];
    for (name, result) in results {
        match result {
            Ok(_) => panic!("{name}: the writer finalizes through the global client; must be rejected"),
            Err(err) => assert_unsupported_per_endpoint(&err, &b),
        }
    }

    assert_eq!(mentioning(a, &bucket), Vec::<String>::new(), "A got a request");
    assert_eq!(b.requests(), Vec::<String>::new(), "B got a request");
}

#[tokio::test]
async fn store_bound_to_the_global_server_still_uses_it() {
    let a = global_a();
    // A per-endpoint store whose server IS the global server.
    let store = S3ObjectStore::for_endpoint(&a.url())
        .await
        .expect("for_endpoint with fake credentials");
    let bucket = unique_bucket("same");

    // The mock answers CreateMultipartUpload with an empty 200, so the upload itself
    // fails -- but it must get as far as sending the request to A, and the failure
    // must not be the per-endpoint rejection.
    let err = store
        .put_multipart(
            &format!("s3://{bucket}/key"),
            Bytes::from_static(b"x"),
            None,
        )
        .await
        .expect_err("the mock cannot complete a multipart upload");
    assert!(
        !format!("{err:#}").contains("#208"),
        "a store on the global server must not be rejected, got: {err:#}"
    );
    assert!(
        mentioning(a, &bucket)
            .iter()
            .any(|r| r.starts_with("POST ")),
        "A must receive CreateMultipartUpload, got {:?}",
        mentioning(a, &bucket)
    );

    let uri = format!("s3://{bucket}/other");
    assert!(
        store.get_writer(&uri).await.is_ok(),
        "writers on the global server are allowed"
    );
    let bucket2 = unique_bucket("same-rng");
    let _ = store
        .get_range_optimized(
            &format!("s3://{bucket2}/key"),
            0,
            Some(64 * 1024 * 1024),
            None,
            None,
        )
        .await;
    assert!(
        !mentioning(a, &bucket2).is_empty(),
        "the concurrent range GET must reach A"
    );
}

#[tokio::test]
async fn default_store_still_uses_the_global_server() {
    let a = global_a();
    let store = S3ObjectStore::new();
    let bucket = unique_bucket("default");

    let err = store
        .put_multipart(
            &format!("s3://{bucket}/key"),
            Bytes::from_static(b"x"),
            None,
        )
        .await
        .expect_err("the mock cannot complete a multipart upload");
    assert!(
        !format!("{err:#}").contains("#208"),
        "the default store must not be rejected, got: {err:#}"
    );
    assert!(
        mentioning(a, &bucket)
            .iter()
            .any(|r| r.starts_with("POST ")),
        "A must receive CreateMultipartUpload, got {:?}",
        mentioning(a, &bucket)
    );
}
