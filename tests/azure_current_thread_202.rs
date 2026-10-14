// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Issue #202: building the Azure https client used
//! `block_in_place(|| Handle::current().block_on(..))` to obtain the credential, which
//! panics on a current-thread Tokio runtime (`#[tokio::main(flavor = "current_thread")]`,
//! or the default `#[tokio::test]`).
//!
//! The account URL is an https URL on a closed local port, so no external network is
//! used: the credential is created (https path), then the request fails with a connect
//! or credential error. The assertion is only that every path returns a `Result`
//! (an `Err` here) instead of panicking.

#![cfg(feature = "backend-azure")]

use s3dlio::api::store_for_uri;
use std::time::Duration;

#[tokio::test] // current-thread runtime
async fn azure_https_paths_return_err_on_a_current_thread_runtime() {
    // One test function: the endpoint override is process-global environment.
    std::env::set_var("AZURE_STORAGE_ENDPOINT", "https://127.0.0.1:9");

    let store = store_for_uri("az://acct202/cont202/key202").expect("az:// store builds");

    let get = tokio::time::timeout(
        Duration::from_secs(120),
        store.get("az://acct202/cont202/key202"),
    )
    .await
    .expect("get must finish");
    assert!(get.is_err(), "get against a closed port cannot succeed");

    let stat = tokio::time::timeout(
        Duration::from_secs(120),
        store.stat("az://acct202/cont202/key202"),
    )
    .await
    .expect("stat must finish");
    assert!(stat.is_err(), "stat against a closed port cannot succeed");

    let create = tokio::time::timeout(
        Duration::from_secs(120),
        store.create_container("acct202/cont202"),
    )
    .await
    .expect("create_container must finish");
    assert!(
        create.is_err(),
        "create_container against a closed port cannot succeed"
    );

    let delete = tokio::time::timeout(
        Duration::from_secs(120),
        store.delete_container("acct202/cont202"),
    )
    .await
    .expect("delete_container must finish");
    assert!(
        delete.is_err(),
        "delete_container against a closed port cannot succeed"
    );
}
