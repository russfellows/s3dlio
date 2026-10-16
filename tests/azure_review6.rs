// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Round-6 review items for the Azure backend: account names reach every URL builder only after
//! validation (container listing included), and SAS material never survives in error text even
//! when the SDK re-serialises the query or a transport error prints the whole URL.

#![cfg(feature = "backend-azure")]

#[path = "common/azure_fake.rs"]
mod azure_fake;

use azure_fake::*;
use s3dlio::object_store::{AzureObjectStore, ObjectStore};

const SAS: &str =
    "sv=2024-11-04&ss=b&srt=sco&sp=rwdl&se=2099-01-01T00%3A00%3A00Z&sig=SIGNATURE%2BVALUE%2FXYZ%3D";

fn assert_no_sas(err: &anyhow::Error, what: &str) {
    for (form, text) in [
        ("Display", format!("{err}")),
        ("alternate", format!("{err:#}")),
        ("Debug", format!("{err:?}")),
    ] {
        for leak in [
            "SIGNATURE",
            "VALUE%2FXYZ",
            "VALUE/XYZ",
            "2099-01-01",
            "srt=sco",
        ] {
            assert!(
                !text.contains(leak),
                "RED: {what}: {leak:?} leaked in {form}: {text}"
            );
        }
    }
}

/// `list_containers` took the raw text after `az://` and built
/// `https://<text>.blob.core.windows.net`, so `az://attacker.example#` sent the bearer token to
/// attacker.example. Every URL builder now validates the account first.
#[test]
fn container_listing_refuses_malformed_accounts_before_any_credential_use() {
    let _guard = ENV_LOCK.blocking_lock();
    clear_azure_env();
    // A bearer-token mode with no variables: if a credential were built, the error would be
    // about AZURE_TENANT_ID, not about the account.
    std::env::set_var("S3DLIO_AZURE_CREDENTIAL", "client-secret");
    for bad in [
        "az://attacker.example#",
        "az://attacker.example/",
        "az://a?b",
        "az://acct@evil",
        "az://ac\\ct",
        "az://x%2fy",
        "az://ab",
    ] {
        let err = s3dlio::list_containers::list_containers(bad).expect_err(bad);
        let message = format!("{err:#}");
        assert!(
            message.contains("invalid Azure account") && !message.contains("AZURE_TENANT_ID"),
            "RED: {bad}: {message}"
        );
    }
}

#[test]
fn the_cli_refuses_malformed_accounts_in_list_buckets() {
    let _guard = ENV_LOCK.blocking_lock();
    clear_azure_env();
    for bad in ["az://attacker.example#", "az://a?b"] {
        let out = std::process::Command::new(env!("CARGO_BIN_EXE_s3-cli"))
            .args(["list-buckets", bad])
            .env("S3DLIO_AZURE_CREDENTIAL", "client-secret")
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(!out.status.success(), "{bad}: {stderr}");
        assert!(
            stderr.contains("invalid Azure account") && !stderr.contains("AZURE_TENANT_ID"),
            "RED: {bad}: {stderr}"
        );
    }
}

/// Operations that set query pairs make the SDK re-sort and re-encode the SAS; the service
/// here echoes the request URL in a 403 body, as a proxy might.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_sas_does_not_survive_in_errors_whatever_the_operation() {
    let _guard = ENV_LOCK.lock().await;
    clear_azure_env();
    let (state, port) = start().await;
    std::env::set_var("AZURE_STORAGE_ENDPOINT", format!("http://127.0.0.1:{port}"));
    std::env::set_var("S3DLIO_AZURE_CREDENTIAL", "sas");
    std::env::set_var("S3DLIO_AZURE_SAS_TOKEN", SAS);
    state
        .lock()
        .unwrap()
        .forbidden_accounts
        .insert("acct".into());
    let store = AzureObjectStore::new();

    let results: Vec<(&str, anyhow::Result<()>)> = vec![
        ("stat", store.stat(&uri("a")).await.map(|_| ())),
        ("get", store.get(&uri("a")).await.map(|_| ())),
        ("list", store.list("az://acct/ctr/", true).await.map(|_| ())),
        (
            "put_multipart (stage_block / commit_block_list)",
            store
                .put_multipart(&uri("a"), bytes::Bytes::from(vec![1u8; 100]), Some(40))
                .await,
        ),
        ("delete", store.delete(&uri("a")).await),
        (
            "delete_prefix",
            store.delete_prefix("az://acct/ctr/p/").await,
        ),
        (
            "list_account_containers",
            s3dlio::azure_client::list_account_containers("acct")
                .await
                .map(|_| ()),
        ),
    ];
    for (what, result) in results {
        let err = result.expect_err(what);
        assert_no_sas(&err, what);
    }
}

/// A transport error (nothing listening) prints the request URL; the SAS in it must go too.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_sas_does_not_survive_in_transport_errors() {
    let _guard = ENV_LOCK.lock().await;
    clear_azure_env();
    let closed = {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        l.local_addr().unwrap().port()
    };
    std::env::set_var(
        "AZURE_STORAGE_ENDPOINT",
        format!("http://127.0.0.1:{closed}"),
    );
    std::env::set_var("S3DLIO_AZURE_CREDENTIAL", "sas");
    std::env::set_var("S3DLIO_AZURE_SAS_TOKEN", SAS);
    let store = AzureObjectStore::new();
    let started = std::time::Instant::now();
    let err = tokio::time::timeout(std::time::Duration::from_secs(120), store.stat(&uri("a")))
        .await
        .expect("the transport error arrives in time")
        .unwrap_err();
    eprintln!("transport error after {:?}: {err:#}", started.elapsed());
    assert_no_sas(&err, "transport error");
}

/// The error type's Debug output is as clean as its Display.
#[test]
fn the_request_error_debug_output_is_redacted() {
    let _guard = ENV_LOCK.blocking_lock();
    clear_azure_env();
    std::env::set_var("S3DLIO_AZURE_SAS_TOKEN", SAS);
    // Reached through the public typed error: build one from an SDK error that carries the SAS.
    let inner = azure_core::Error::with_message(
        azure_core::error::ErrorKind::Other,
        format!("request to https://h/acct/c/k?{SAS} failed"),
    );
    let err = s3dlio::azure_client::request_error_for_test(inner);
    assert_no_sas(&err, "request error");
    // Debug of the typed error itself (not only the anyhow chain around it).
    let typed = err
        .downcast_ref::<s3dlio::azure_client::AzureRequestError>()
        .expect("the typed error is in the chain");
    let debug = format!("{typed:?}");
    assert!(
        !debug.contains("SIGNATURE") && !debug.contains("2099-01-01"),
        "RED: {debug}"
    );
}
