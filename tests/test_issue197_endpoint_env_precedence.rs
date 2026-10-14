// tests/test_issue197_endpoint_env_precedence.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Integration of #197 (reject a URI naming a server other than the one the
// process-wide client talks to) with #183 (`AWS_ENDPOINT_URL_S3` wins over
// `AWS_ENDPOINT_URL`). With BOTH set, to different servers:
//
//   A = AWS_ENDPOINT_URL        -- NOT the global server (loses to the S3 variable)
//   B = AWS_ENDPOINT_URL_S3     -- the global server
//
// the routing guard must agree with the client about which one is global:
//   * a host-form URI naming B is accepted and reaches B only;
//   * a host-form URI naming A is rejected (A is not the global server), and no
//     server receives a request;
//   * a bucket-form URI goes to B only.
//
// One process per binary: the library's global client is built once from the
// environment, set before anything touches it.

#[path = "common/recording_s3.rs"]
mod recording_s3;
use recording_s3::MockS3;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Once, OnceLock};

use bytes::Bytes;
use s3dlio::object_store::{ObjectStore, S3ObjectStore};

static SERVERS: OnceLock<(MockS3, MockS3)> = OnceLock::new();
static ENV: Once = Once::new();
static COUNTER: AtomicUsize = AtomicUsize::new(0);

/// `(a, b)`: A is `AWS_ENDPOINT_URL`, B is `AWS_ENDPOINT_URL_S3` (the global server).
fn servers() -> &'static (MockS3, MockS3) {
    ENV.call_once(|| {
        let a = MockS3::start(&[]);
        let b = MockS3::start(&[]);
        #[allow(deprecated)]
        {
            std::env::set_var("AWS_ACCESS_KEY_ID", "test-key");
            std::env::set_var("AWS_SECRET_ACCESS_KEY", "test-secret");
            std::env::set_var("AWS_REGION", "us-east-1");
            std::env::set_var("AWS_ENDPOINT_URL", a.url());
            std::env::set_var("AWS_ENDPOINT_URL_S3", b.url());
        }
        let _ = SERVERS.set((a, b));
    });
    SERVERS.get().expect("servers set by call_once")
}

fn unique_bucket(tag: &str) -> String {
    format!("b197e-{tag}-{}", COUNTER.fetch_add(1, Ordering::Relaxed))
}

fn mentioning(m: &MockS3, needle: &str) -> Vec<String> {
    m.requests()
        .into_iter()
        .filter(|r| r.contains(needle))
        .collect()
}

#[tokio::test]
async fn host_form_naming_the_s3_endpoint_is_accepted_and_reaches_only_it() {
    let (a, b) = servers();
    let bucket = unique_bucket("host-b");
    let uri = format!("s3://127.0.0.1:{}/{bucket}/key", b.port);

    S3ObjectStore::new()
        .put(&uri, Bytes::from_static(b"x"))
        .await
        .expect("B is the global server (AWS_ENDPOINT_URL_S3 wins): the URI must be accepted");

    assert_eq!(mentioning(b, &bucket), vec![format!("PUT /{bucket}/key")]);
    assert_eq!(mentioning(a, &bucket), Vec::<String>::new(), "A got a request");
}

#[tokio::test]
async fn host_form_naming_the_losing_endpoint_is_rejected_and_sends_nothing() {
    let (a, b) = servers();
    let bucket = unique_bucket("host-a");
    let uri = format!("s3://127.0.0.1:{}/{bucket}/key", a.port);

    let err = S3ObjectStore::new()
        .put(&uri, Bytes::from_static(b"x"))
        .await
        .expect_err("A is not the global server (AWS_ENDPOINT_URL loses to AWS_ENDPOINT_URL_S3)");
    let text = format!("{err:#}");
    assert!(
        text.contains(&format!("127.0.0.1:{}", a.port)),
        "the error must name the URI's server, got: {text}"
    );
    assert_eq!(a.requests(), Vec::<String>::new(), "A got a request");
    assert_eq!(mentioning(b, &bucket), Vec::<String>::new(), "B got a request");

    // The same rule for the batch-delete guard.
    let err = S3ObjectStore::new()
        .delete_batch(std::slice::from_ref(&uri))
        .await
        .expect_err("A is not the global server");
    assert!(
        format!("{err:#}").contains(&format!("127.0.0.1:{}", a.port)),
        "got: {err:#}"
    );
    assert_eq!(a.requests(), Vec::<String>::new(), "A got a request");
    assert_eq!(mentioning(b, &bucket), Vec::<String>::new(), "B got a request");
}

#[tokio::test]
async fn bucket_form_goes_to_the_s3_endpoint_only() {
    let (a, b) = servers();
    let bucket = unique_bucket("bucket");

    S3ObjectStore::new()
        .put(&format!("s3://{bucket}/key"), Bytes::from_static(b"x"))
        .await
        .expect("bucket-form put to the global server");

    assert_eq!(mentioning(b, &bucket), vec![format!("PUT /{bucket}/key")]);
    assert_eq!(mentioning(a, &bucket), Vec::<String>::new(), "A got a request");
}
