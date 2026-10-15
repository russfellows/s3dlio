// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Round-5 review items for the Azure backend, against the local fake Blob service:
//! idempotent deletes (the typed "BlobNotFound" error must survive error wrapping), account-name
//! validation, SAS handling and redaction, and the error text of 401/403 responses.

#![cfg(feature = "backend-azure")]

#[path = "common/azure_fake.rs"]
mod azure_fake;

use azure_fake::*;
use s3dlio::delete_contract::{
    delete_exact_object, delete_objects_under_literal_prefix, DeleteMode,
};
use s3dlio::object_store::{AzureObjectStore, ObjectStore};

async fn setup() -> (tokio::sync::MutexGuard<'static, ()>, State, u16) {
    let guard = ENV_LOCK.lock().await;
    clear_azure_env();
    let (state, port) = start().await;
    std::env::set_var("AZURE_STORAGE_ENDPOINT", format!("http://127.0.0.1:{port}"));
    (guard, state, port)
}

/// Deleting an object that is already gone is success (the idempotent-delete contract), for
/// the exact delete and for both recursive delete modes, and other failures are still failures.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn deleting_a_missing_blob_succeeds() {
    let (_guard, state, _port) = setup().await;
    let store = AzureObjectStore::new();

    // Exact delete of a blob that does not exist.
    delete_exact_object(&store, &uri("never-existed.bin"))
        .await
        .unwrap_or_else(|e| panic!("RED: deleting a missing blob failed: {e:#}"));

    // Recursive delete where listed blobs have vanished before their DELETE (listing ghosts),
    // alongside real ones.
    for mode in [DeleteMode::PerObject, DeleteMode::Batched] {
        set_blob(&state, "tree/real-1", pattern(10, 1));
        set_blob(&state, "tree/real-2", pattern(10, 2));
        state.lock().unwrap().list_ghosts = vec!["tree/ghost-1".into(), "tree/ghost-2".into()];
        let report =
            delete_objects_under_literal_prefix(&store, "az://acct/ctr/tree/", None, mode, None)
                .await;
        assert!(
            !report.failed(),
            "RED: recursive delete ({mode:?}) failed on vanished blobs: {:?} {:?}",
            report.delete_problems,
            report.list_problems
        );
        assert!(state
            .lock()
            .unwrap()
            .blobs
            .keys()
            .all(|k| !k.starts_with("acct/ctr/tree/")));
    }

    // A rejected delete is still an error, and it names the credential mode.
    state
        .lock()
        .unwrap()
        .forbidden_accounts
        .insert("acct".into());
    let err = delete_exact_object(&store, &uri("x.bin"))
        .await
        .unwrap_err();
    let message = format!("{err:#}");
    assert!(
        message.contains("Azure credential mode") && message.contains("rejected"),
        "{message}"
    );
}

/// A SAS token never appears in an error message, even when the service echoes the request URL
/// back in its response body.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_sas_token_is_redacted_from_errors() {
    let (_guard, state, _port) = setup().await;
    std::env::set_var("S3DLIO_AZURE_CREDENTIAL", "sas");
    std::env::set_var(
        "S3DLIO_AZURE_SAS_TOKEN",
        "sv=2024-11-04&sig=SECRETSIGNATURE123",
    );
    state
        .lock()
        .unwrap()
        .forbidden_accounts
        .insert("acct".into());
    let store = AzureObjectStore::new();
    for result in [
        store.stat(&uri("a.bin")).await.map(|_| ()),
        store.get(&uri("a.bin")).await.map(|_| ()),
        store.delete(&uri("a.bin")).await,
    ] {
        let message = format!("{:#}", result.unwrap_err());
        assert!(
            !message.contains("SECRETSIGNATURE123"),
            "RED: the SAS leaked: {message}"
        );
        assert!(message.contains("Azure credential mode sas"), "{message}");
    }
}

/// An `az://` URI whose "account" is not a storage account name is refused before any request
/// or credential use, so it cannot redirect requests (or a bearer token) to another host.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn malformed_account_names_are_refused_before_any_request() {
    let (_guard, state, _port) = setup().await;
    let store = AzureObjectStore::new();
    let before = mark(&state);
    for bad in [
        "az://attacker.example#/c/k",
        "az://a?b/c/k",
        "az://acct@evil/c/k",
        "az://ac\\ct/c/k",
        "az://ac ct/c/k",
    ] {
        for (what, result) in [
            ("stat", store.stat(bad).await.map(|_| ())),
            ("get", store.get(bad).await.map(|_| ())),
            ("put", store.put(bad, bytes::Bytes::from_static(b"x")).await),
            ("delete", store.delete(bad).await),
        ] {
            let message = format!("{:#}", result.expect_err(&format!("{what} {bad}")));
            assert!(
                message.contains("invalid Azure account"),
                "RED: {what} {bad}: {message}"
            );
        }
    }
    assert_eq!(
        mark(&state),
        before,
        "RED: a request was made for a malformed account name"
    );
    // Real account names still work.
    set_blob(&state, "fine.bin", pattern(10, 1));
    assert_eq!(store.get(&uri("fine.bin")).await.unwrap().len(), 10);
}

/// A SAS token must not travel over plain http to a host that is not the local machine.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_sas_token_is_not_sent_over_http_to_a_remote_host() {
    let (_guard, _state, _port) = setup().await;
    std::env::set_var("S3DLIO_AZURE_CREDENTIAL", "sas");
    std::env::set_var("S3DLIO_AZURE_SAS_TOKEN", "sv=2024-11-04&sig=abc");
    std::env::set_var("AZURE_STORAGE_ENDPOINT", "http://203.0.113.9:10000");
    let store = AzureObjectStore::new();
    let err = store.stat(&uri("a.bin")).await.unwrap_err();
    let message = format!("{err:#}");
    assert!(
        message.contains("SAS") && message.contains("http") && !message.contains("abc"),
        "RED: {message}"
    );
}

/// Endpoint and account URLs in error messages carry no query string (a SAS may be in one).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn error_messages_do_not_echo_endpoint_query_strings() {
    let (_guard, _state, _port) = setup().await;
    let err = s3dlio::azure_client::AzureBlob::with_default_credential_from_url(
        "not a url?sig=QUERYSECRET",
        "ctr",
    )
    .err()
    .expect("an invalid URL");
    assert!(!format!("{err:#}").contains("QUERYSECRET"), "RED: {err:#}");

    // A bearer-token mode on an http endpoint names the endpoint, without its query.
    std::env::set_var("S3DLIO_AZURE_CREDENTIAL", "client-secret");
    let err = s3dlio::azure_client::AzureBlob::with_default_credential_from_url(
        "http://127.0.0.1:1/acct?sig=QUERYSECRET",
        "ctr",
    )
    .err()
    .expect("a bearer mode on http");
    let message = format!("{err:#}");
    assert!(
        message.contains("https") && !message.contains("QUERYSECRET"),
        "RED: {message}"
    );
}

/// `list_account_containers` reports a rejected request with the credential mode, like the
/// blob operations.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn listing_containers_names_the_credential_mode_on_403() {
    let (_guard, state, _port) = setup().await;
    state
        .lock()
        .unwrap()
        .forbidden_accounts
        .insert("acct".into());
    let err = s3dlio::azure_client::list_account_containers("acct")
        .await
        .unwrap_err();
    let message = format!("{err:#}");
    assert!(
        message.contains("Azure credential mode") && message.contains("rejected"),
        "RED: {message}"
    );
}
