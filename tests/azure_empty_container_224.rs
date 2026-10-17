// tests/azure_empty_container_224.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// s3dlio #224: `az://account//key` has an EMPTY container. Before the fix the parser accepted it
// and shifted nothing: the empty name reached the client builder. The URI must be refused by pure
// parsing, before any client is built. `AzureObjectStore::mkdir`/`rmdir` validate the URI and then
// send nothing (issue #229), so they observe the parser alone: no credentials, no network.
//
// Valid roots and keys that contain percent signs still parse, and object keys are literal
// (`%2F` stays three characters).
//
// Compiled only with the Azure backend (the wheel's default mode and `full-backends`).

#![cfg(feature = "backend-azure")]

use s3dlio::object_store::{AzureObjectStore, ObjectStore};

/// URIs whose container is empty. Each must fail, whatever the key.
const EMPTY_CONTAINER_URIS: [&str; 5] = [
    "az://account//key",
    "az://account//",
    "az://account/",
    "az://account//a/b/c",
    "az://account///key",
];

#[tokio::test]
async fn empty_container_is_refused_before_any_client_is_built() {
    let store = AzureObjectStore::new();
    for uri in EMPTY_CONTAINER_URIS {
        let err = store
            .mkdir(uri)
            .await
            .expect_err(&format!("{uri}: an empty container must be refused"));
        let msg = format!("{err:#}");
        assert!(
            msg.contains("container"),
            "{uri}: the error must name the missing container, got: {msg}"
        );
        let err = store
            .rmdir(uri, false)
            .await
            .expect_err(&format!("{uri}: rmdir must refuse an empty container too"));
        assert!(format!("{err:#}").contains("container"), "{uri}");
    }
}

#[tokio::test]
async fn valid_roots_and_percent_keys_still_parse() {
    let store = AzureObjectStore::new();
    for uri in [
        "az://account/container",
        "az://account/container/",
        "az://account/container/key",
        "az://account/container/a%2Fb",
        "az://account/container/100%25/x%",
        "az://account/container//leading-slash-key",
        "az://account/container/a=b@c/d.",
    ] {
        store
            .mkdir(uri)
            .await
            .unwrap_or_else(|e| panic!("{uri} must still parse: {e:#}"));
    }
}
