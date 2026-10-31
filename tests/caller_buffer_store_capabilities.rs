// tests/caller_buffer_store_capabilities.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Capability and behaviour agree (plan §4.1, §9 "Capabilities"; #264): for every scheme,
// `object_capabilities(uri).get_into` is true exactly when the store `store_for_uri` builds for
// that URI does NOT answer `get_into` with `Unsupported`, and the same for `head`. The pure
// capability answer and the store wiring are two code paths; this keeps them from drifting.
//
// Local only. s3:// goes to an in-process mock server and az:// to the local fake Blob service.
// gs:// has no local double: the official GCS client has no endpoint override, so the store is
// pointed at a credentials file that does not exist. A supported store then fails building its
// client ("not Unsupported", before any request); an unsupported one answers `Unsupported`
// without reading the environment at all. The provider-level GCS tests are in
// `src/gcs_get_into_tests.rs` and `src/gcs_head_typed_tests.rs`.

#![cfg(all(feature = "native-backends", not(feature = "arrow-backend")))]

#[allow(dead_code)] // shared helpers; each test binary uses a subset
mod s3_get_into_support;

#[cfg(feature = "backend-azure")]
#[path = "common/azure_fake.rs"]
mod azure_fake;
#[cfg(feature = "backend-azure")]
#[path = "caller_buffer_store_support/mod.rs"]
mod support;

use s3_get_into_support::mock::MockObject;
use s3_get_into_support::{init, key, pattern, BUCKET};
use s3dlio::capabilities::object_capabilities;
use s3dlio::object_error::{object_error, ObjectErrorKind};
use s3dlio::object_store::{store_for_uri, ObjectStore};
use s3dlio::receive_progress::ReceiveProgress;
use std::sync::Once;

static ENV: Once = Once::new();

/// The process environment of this binary, set ONCE before any test touches the network or
/// builds a store: the S3 mock (and its environment), the Azure fake (and its environment) and a
/// credentials file for the official GCS client that does not exist. Every test begins here and
/// no test changes the environment itself, so none changes it under another test's I/O.
fn prepare() -> (u16, s3_get_into_support::mock::MockS3State) {
    ENV.call_once(|| {
        init(&[]);
        std::env::set_var(
            "GOOGLE_APPLICATION_CREDENTIALS",
            "/nonexistent/s3dlio-caller-buffer-capabilities.json",
        );
        #[cfg(feature = "backend-azure")]
        support::azure();
    });
    init(&[])
}

fn is_unsupported(result: &anyhow::Result<()>) -> bool {
    result
        .as_ref()
        .err()
        .and_then(object_error)
        .is_some_and(|e| e.kind == ObjectErrorKind::Unsupported)
}

/// Check `uri` on both operations. `None` from the factory means the scheme is compiled out of
/// this build, which the capabilities must also report.
async fn assert_agree(uri: &str) {
    let caps = object_capabilities(uri).unwrap_or_else(|e| panic!("{uri}: {e:#}"));
    let store: Box<dyn ObjectStore> = match store_for_uri(uri) {
        Ok(store) => store,
        Err(e) => {
            assert!(
                !caps.get_into && !caps.head,
                "{uri}: no store can be built ({e:#}) but capabilities say supported: {caps:?}"
            );
            return;
        }
    };

    let mut dst = [0u8; 4];
    let got = store.get_into(uri, &mut dst, &ReceiveProgress::new()).await;
    assert_eq!(
        caps.get_into,
        !is_unsupported(&got),
        "{uri}: capability get_into={} but the store answered {got:?}",
        caps.get_into
    );

    let head = store.head(uri).await.map(|_| ());
    assert_eq!(
        caps.head,
        !is_unsupported(&head),
        "{uri}: capability head={} but the store answered {head:?}",
        caps.head
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn s3_capability_matches_the_store_for_both_uri_forms() {
    let (port, state) = prepare();
    let bucket_form = key("cap-global");
    let host_form = key("cap-host");
    for k in [&bucket_form, &host_form] {
        state.put_object(BUCKET, k, MockObject::new(pattern(4, 1)));
    }
    assert_agree(&format!("s3://{BUCKET}/{bucket_form}")).await;
    assert_agree(&format!("s3://127.0.0.1:{port}/{BUCKET}/{host_form}")).await;
    // The same two, with a length that matches, really succeed: supported means supported.
    for uri in [
        format!("s3://{BUCKET}/{bucket_form}"),
        format!("s3://127.0.0.1:{port}/{BUCKET}/{host_form}"),
    ] {
        let mut dst = [0u8; 4];
        store_for_uri(&uri)
            .expect("store")
            .get_into(&uri, &mut dst, &ReceiveProgress::new())
            .await
            .unwrap_or_else(|e| panic!("{uri}: {e:#}"));
        assert_eq!(dst[..], pattern(4, 1)[..]);
    }
}

#[cfg(feature = "backend-azure")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn azure_capability_matches_the_store() {
    use azure_fake::{pattern, set_blob, uri};
    prepare();
    set_blob(support::azure(), "cap", pattern(4, 2));
    assert_agree(&uri("cap")).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn gcs_capability_matches_the_store() {
    prepare();
    for uri in ["gs://bucket/key", "gcs://bucket/key"] {
        assert_agree(uri).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn file_and_direct_are_supported_in_both_places() {
    prepare();
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("object.bin");
    std::fs::write(&path, [1u8; 4]).expect("write");
    for scheme in ["file", "direct"] {
        let uri = format!("{scheme}://{}", path.display());
        let caps = object_capabilities(&uri).expect("caps");
        assert!(caps.get_into && caps.head, "{uri}: {caps:?}");
        // `assert_agree` reads 4 bytes into a 4-byte destination: the file is exactly that.
        assert_agree(&uri).await;
        let mut dst = [0u8; 4];
        store_for_uri(&uri)
            .expect("store")
            .get_into(&uri, &mut dst, &ReceiveProgress::new())
            .await
            .unwrap_or_else(|e| panic!("{uri}: {e:#}"));
        assert_eq!(dst, [1u8; 4]);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_unknown_scheme_has_no_store_and_no_capability() {
    prepare();
    let caps = object_capabilities("ftp://host/x").expect("caps");
    assert_eq!(caps.provider, "unknown");
    assert!(!caps.get_into && !caps.head);
    assert!(store_for_uri("ftp://host/x").is_err());
}

/// A provider compiled out of this build is reported false and has no store.
#[cfg(not(feature = "backend-azure"))]
#[test]
fn a_compiled_out_azure_is_false_and_has_no_store() {
    prepare();
    let caps = object_capabilities("az://account/container/key").expect("caps");
    assert_eq!(caps.provider, "az");
    assert!(!caps.get_into && !caps.head);
    assert!(store_for_uri("az://account/container/key").is_err());
}

#[cfg(not(feature = "backend-gcs"))]
#[test]
fn a_compiled_out_gcs_is_false_and_has_no_store() {
    prepare();
    let caps = object_capabilities("gs://bucket/key").expect("caps");
    assert!(!caps.get_into && !caps.head);
    assert!(store_for_uri("gs://bucket/key").is_err());
}
