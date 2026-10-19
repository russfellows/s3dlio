// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! `ObjectStore::get_into` and `ObjectStore::head` on az:// URIs through `store_for_uri`
//! (plan v0.10.0 §4.5 A1, A8; #264), against the local fake Blob service. The provider's own
//! behaviour (ranges, ETag pinning, empty blobs) has its tests in `azure_get_into_fake.rs`; this
//! file proves the STORE parses the URI with the shared Azure parser (#224), reaches the
//! provider, writes into the caller's slice, and returns typed errors.

#![cfg(feature = "backend-azure")]

#[path = "common/azure_fake.rs"]
mod azure_fake;
#[path = "caller_buffer_store_support/mod.rs"]
mod support;

use azure_fake::*;
use s3dlio::object_error::{object_error, ObjectErrorKind};
use s3dlio::object_store::{store_for_uri, ObjectStore};
use s3dlio::receive_progress::ReceiveProgress;

const CANARY: u8 = 0xEE;
const PAD: usize = 32;

async fn setup() -> (
    tokio::sync::MutexGuard<'static, ()>,
    State,
    Box<dyn ObjectStore>,
) {
    // The environment and the fake are set up once per binary (see `support`); a test holds the
    // lock because it may change the fake's shared state, and starts from a clean one.
    let guard = ENV_LOCK.lock().await;
    let state = support::azure().clone();
    {
        let mut fake = state.lock().unwrap();
        fake.forbidden_accounts.clear();
        fake.missing_containers.clear();
    }
    let store = store_for_uri("az://acct/ctr/").expect("an Azure store");
    (guard, state, store)
}

fn kind(err: &anyhow::Error) -> ObjectErrorKind {
    object_error(err)
        .unwrap_or_else(|| panic!("an untyped error: {err:#}"))
        .kind
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn get_into_fills_the_callers_slice_and_nothing_else() {
    let (_guard, state, store) = setup().await;
    for len in [1usize, 1000, MIB + 3] {
        let key = format!("store-{len}");
        set_blob(&state, &key, pattern(len, 4));
        let mut buf = vec![CANARY; len + 2 * PAD];
        let progress = ReceiveProgress::new();
        let before = mark(&state);
        store
            .get_into(&uri(&key), &mut buf[PAD..PAD + len], &progress)
            .await
            .unwrap_or_else(|e| panic!("{len}: {e:#}"));
        assert!(buf[PAD..PAD + len] == pattern(len, 4)[..], "{len}");
        assert!(buf[..PAD].iter().all(|&b| b == CANARY), "{len}: before");
        assert!(
            buf[PAD + len..].iter().all(|&b| b == CANARY),
            "{len}: after"
        );
        assert_eq!(progress.filled(), len as u64);
        let seen = requests_since(&state, before);
        assert_eq!(
            count(&seen, "GET"),
            1,
            "{len}: the provider path ran: {seen:?}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn typed_errors_surface_through_the_store() {
    let (_guard, state, store) = setup().await;

    // An absent blob.
    let mut dst = vec![CANARY; 16];
    let err = store
        .get_into(&uri("absent"), &mut dst, &ReceiveProgress::new())
        .await
        .expect_err("absent");
    assert_eq!(kind(&err), ObjectErrorKind::NotFound);
    assert!(dst.iter().all(|&b| b == CANARY), "nothing was written");

    // A blob of another length than the destination.
    set_blob(&state, "len", pattern(40, 1));
    for len in [39usize, 41] {
        let mut dst = vec![CANARY; len];
        let err = store
            .get_into(&uri("len"), &mut dst, &ReceiveProgress::new())
            .await
            .expect_err("length");
        assert!(
            matches!(kind(&err), ObjectErrorKind::LengthMismatch { .. }),
            "{len}: {err:#}"
        );
    }

    // A refusal.
    state
        .lock()
        .unwrap()
        .forbidden_accounts
        .insert("acct".into());
    let err = store
        .get_into(&uri("len"), &mut [0u8; 40], &ReceiveProgress::new())
        .await
        .expect_err("403");
    assert_eq!(kind(&err), ObjectErrorKind::Forbidden);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn head_reports_presence_and_only_a_missing_blob_as_absence() {
    let (_guard, state, store) = setup().await;
    set_blob(&state, "h", pattern(77, 2));
    set_etag(&state, "h", "\"etag-h\"");

    let meta = store.head(&uri("h")).await.expect("head").expect("present");
    assert_eq!(meta.size, 77);
    assert_eq!(meta.etag.as_deref(), Some("\"etag-h\""));

    assert!(store.head(&uri("absent")).await.expect("head").is_none());

    // A missing container is an error, not absence.
    state
        .lock()
        .unwrap()
        .missing_containers
        .insert("acct/ctr".to_string());
    let err = store
        .head(&uri("absent"))
        .await
        .expect_err("a missing container");
    assert_eq!(kind(&err), ObjectErrorKind::Other);

    state.lock().unwrap().missing_containers.clear();
    state
        .lock()
        .unwrap()
        .forbidden_accounts
        .insert("acct".into());
    let err = store.head(&uri("h")).await.expect_err("403");
    assert_eq!(kind(&err), ObjectErrorKind::Forbidden);
}

/// #224: an empty container is refused by the parser before any client is built or any
/// request is sent.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_empty_container_is_refused_before_any_request() {
    let (_guard, state, store) = setup().await;
    let before = mark(&state);
    let bad = "az://acct//key";
    store
        .get_into(bad, &mut [0u8; 4], &ReceiveProgress::new())
        .await
        .expect_err("empty container");
    store.head(bad).await.expect_err("empty container");
    assert!(requests_since(&state, before).is_empty());
}

/// A URI that names a container but no blob is refused with a typed `Other` error before any
/// client is built or request is sent. Without the refusal the request addressed the container
/// path, and a 404 from `head` came back as `Ok(None)`: absence of an object never asked for.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_empty_key_is_refused_before_any_request() {
    let (_guard, state, store) = setup().await;
    let before = mark(&state);
    for bad in [
        "az://acct/ctr",
        "az://acct/ctr/",
        "https://acct.blob.core.windows.net/ctr",
        "https://acct.blob.core.windows.net/ctr/",
    ] {
        let mut dst = [CANARY; 4];
        let err = store
            .get_into(bad, &mut dst, &ReceiveProgress::new())
            .await
            .expect_err(bad);
        assert_eq!(kind(&err), ObjectErrorKind::Other, "{bad}: {err:#}");
        assert!(format!("{err:#}").contains("no key specified"), "{bad}");
        assert_eq!(dst, [CANARY; 4], "{bad}: nothing was written");

        let err = store
            .head(bad)
            .await
            .expect_err("an empty key is an error, never typed absence");
        assert_eq!(kind(&err), ObjectErrorKind::Other, "{bad}: {err:#}");
        assert!(format!("{err:#}").contains("no key specified"), "{bad}");
    }
    assert!(
        requests_since(&state, before).is_empty(),
        "a request reached the fake: {:?}",
        requests_since(&state, before)
    );
}

/// The Azure host form (`https://<account>.blob.core.windows.net/<container>/<key>`) goes
/// through the same parser and reaches the same blob as the `az://` form.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_azure_host_form_reads_and_heads_the_same_blob() {
    let (_guard, state, _store) = setup().await;
    set_blob(&state, "host-form", pattern(300, 9));
    set_etag(&state, "host-form", "\"etag-host\"");
    let host_uri = "https://acct.blob.core.windows.net/ctr/host-form";
    let store = store_for_uri(host_uri).expect("an Azure store for the host form");

    let before = mark(&state);
    let mut dst = vec![CANARY; 300];
    let progress = ReceiveProgress::new();
    store
        .get_into(host_uri, &mut dst, &progress)
        .await
        .expect("get_into");
    assert!(dst == pattern(300, 9));
    assert_eq!(progress.filled(), 300);
    let meta = store.head(host_uri).await.expect("head").expect("present");
    assert_eq!(meta.size, 300);
    assert_eq!(meta.etag.as_deref(), Some("\"etag-host\""));

    let seen = requests_since(&state, before);
    assert!(
        seen.iter().all(|r| r.path == "acct/ctr/host-form"),
        "every request addressed the blob: {seen:?}"
    );
    assert!(
        count(&seen, "GET") >= 1 && count(&seen, "HEAD") == 1,
        "{seen:?}"
    );
}
