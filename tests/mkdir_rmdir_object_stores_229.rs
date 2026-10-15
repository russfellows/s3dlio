// tests/mkdir_rmdir_object_stores_229.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// s3dlio #229: an object store has no directories -- a prefix
// exists once objects are written under it -- so `mkdir` and `rmdir` on `s3://`
// (and `az://`, `gs://`) are successful NO-OPS. They validate the URI and send ZERO
// requests: no marker object is written, and `rmdir` (recursive or not) deletes
// nothing. Removing the objects under a prefix is `delete(uri, recursive)` /
// `s3-cli delete -r`.
//
// "Zero requests" is checked on the recording mock servers:
//   A = the process-wide (global) client's server (`AWS_ENDPOINT_URL`)
//   B = the server a per-endpoint store talks to
// The pre-fix behaviour, for contrast, was an error ("mkdir not implemented for
// this backend") and no request.

#[path = "common/recording_s3.rs"]
mod recording_s3;
use recording_s3::MockS3;

use std::sync::{Once, OnceLock};

use s3dlio::object_store::{ObjectStore, S3ObjectStore};
use s3dlio::store_for_uri;

static GLOBAL_A: OnceLock<MockS3> = OnceLock::new();
static ENV: Once = Once::new();

/// A, the server `AWS_ENDPOINT_URL` names (set before the global client is built).
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
        }
        let _ = GLOBAL_A.set(a);
    });
    GLOBAL_A.get().expect("global mock set by call_once")
}

/// Every mkdir / rmdir spelling that must be a silent no-op on `store`.
async fn exercise(store: &dyn ObjectStore, dir_uris: &[String]) {
    for uri in dir_uris {
        store
            .mkdir(uri)
            .await
            .unwrap_or_else(|e| panic!("mkdir {uri} must succeed: {e:#}"));
        store
            .mkdir(uri)
            .await
            .unwrap_or_else(|e| panic!("mkdir {uri} again must succeed: {e:#}"));
        for recursive in [true, false] {
            store
                .rmdir(uri, recursive)
                .await
                .unwrap_or_else(|e| panic!("rmdir({uri}, {recursive}) must succeed: {e:#}"));
        }
    }
}

#[tokio::test]
async fn bucket_form_mkdir_rmdir_send_zero_requests() {
    let a = global_a();
    let store = store_for_uri("s3://bkt229/dir").unwrap();
    exercise(
        store.as_ref(),
        &[
            "s3://bkt229/dir".to_string(),
            "s3://bkt229/dir/".to_string(),
            "s3://bkt229/dir/nested/deeper".to_string(),
            "s3://bkt229/".to_string(), // bucket root
        ],
    )
    .await;
    assert_eq!(a.requests(), Vec::<String>::new(), "A got a request");
}

#[tokio::test]
async fn per_endpoint_store_mkdir_rmdir_send_zero_requests_anywhere() {
    let a = global_a();
    let b = MockS3::start(&[]);
    let store = S3ObjectStore::for_endpoint(&b.url()).await.unwrap();
    exercise(
        &store,
        &[
            "s3://bkt229e/dir".to_string(),
            "s3://bkt229e/dir/".to_string(),
            format!("s3://127.0.0.1:{}/bkt229e/dir", b.port),
            format!("s3://127.0.0.1:{}/bkt229e/dir/", b.port),
        ],
    )
    .await;
    assert_eq!(b.requests(), Vec::<String>::new(), "B got a request");
    assert_eq!(a.requests(), Vec::<String>::new(), "A got a request");
}

#[tokio::test]
async fn host_form_factory_store_mkdir_rmdir_send_zero_requests() {
    let a = global_a();
    let b = MockS3::start(&[]);
    let dir = format!("s3://127.0.0.1:{}/bkt229h/dir", b.port);
    let store = store_for_uri(&dir).unwrap();
    exercise(store.as_ref(), &[dir.clone(), format!("{dir}/")]).await;
    assert_eq!(b.requests(), Vec::<String>::new(), "B got a request");
    assert_eq!(a.requests(), Vec::<String>::new(), "A got a request");
}

/// A URI the store cannot use is still an error: the no-op validates first.
#[tokio::test]
async fn unusable_uri_is_still_an_error() {
    let a = global_a();
    let store = S3ObjectStore::new();
    assert!(store.mkdir("gs://bucket/dir").await.is_err());
    assert!(store.rmdir("gs://bucket/dir", true).await.is_err());
    assert!(store.mkdir("not a uri").await.is_err());
    assert_eq!(a.requests(), Vec::<String>::new());
}

/// `MultiEndpointStore` delegates mkdir / rmdir to its endpoint stores (no change
/// needed): with host-form endpoints on two servers B and C, each call is a no-op
/// on whichever endpoint it picks, and no server is contacted.
#[tokio::test]
async fn multi_endpoint_store_mkdir_rmdir_send_zero_requests() {
    use s3dlio::multi_endpoint::{LoadBalanceStrategy, MultiEndpointStore};
    let a = global_a();
    let (b, c) = (MockS3::start(&[]), MockS3::start(&[]));
    let store = MultiEndpointStore::new(
        vec![
            format!("s3://127.0.0.1:{}/bkt229m/mep/", b.port),
            format!("s3://127.0.0.1:{}/bkt229m/mep/", c.port),
        ],
        LoadBalanceStrategy::RoundRobin,
        None,
    )
    .unwrap();
    for i in 0..4 {
        let dir = format!("s3://127.0.0.1:{}/bkt229m/mep/d{i}", b.port);
        store.mkdir(&dir).await.expect("multi-endpoint mkdir");
        store
            .rmdir(&dir, true)
            .await
            .expect("multi-endpoint rmdir -r");
        store
            .rmdir(&dir, false)
            .await
            .expect("multi-endpoint rmdir");
    }
    assert_eq!(b.requests(), Vec::<String>::new(), "B got a request");
    assert_eq!(c.requests(), Vec::<String>::new(), "C got a request");
    assert_eq!(a.requests(), Vec::<String>::new(), "A got a request");
}
