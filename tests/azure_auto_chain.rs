// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! `auto` on a host whose managed-identity endpoint is slow to answer 404 (an Azure VM with no
//! assigned identity: the SDK's VM credential retries 404/410/5xx with back-off) must fall through
//! to the developer tools after a bounded wait, not after the SDK's own retry schedule (about
//! 72 s for the VM credential). A fake identity endpoint stands in for the metadata service.
//!
//! This file has one test: credentials are process-wide, so it needs a process to itself.

#![cfg(feature = "backend-azure")]

use bytes::Bytes;
use http_body_util::Full;
use hyper::service::service_fn;
use hyper::{Response, StatusCode};
use hyper_util::rt::TokioIo;
use s3dlio::azure_credential::{credential_for, CredentialMode};
use std::convert::Infallible;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn auto_falls_through_a_managed_identity_that_answers_404_within_its_deadline() {
    for (name, _) in std::env::vars() {
        if name.starts_with("AZURE_")
            || name.starts_with("S3DLIO_AZURE_")
            || matches!(
                name.as_str(),
                "IDENTITY_ENDPOINT"
                    | "IDENTITY_HEADER"
                    | "IMDS_ENDPOINT"
                    | "MSI_ENDPOINT"
                    | "MSI_SECRET"
            )
        {
            std::env::remove_var(name);
        }
    }
    let hits = Arc::new(AtomicUsize::new(0));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let counter = Arc::clone(&hits);
    tokio::spawn(async move {
        loop {
            let Ok((tcp, _)) = listener.accept().await else {
                continue;
            };
            let counter = Arc::clone(&counter);
            tokio::spawn(async move {
                let service = service_fn(move |_req| {
                    counter.fetch_add(1, Ordering::SeqCst);
                    async move {
                        // The endpoint is slow to give up, as the metadata service is while the
                        // SDK retries 404/5xx with back-off.
                        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                        Ok::<_, Infallible>(
                            Response::builder()
                                .status(StatusCode::NOT_FOUND)
                                .body(Full::new(Bytes::from_static(b"no identity assigned")))
                                .unwrap(),
                        )
                    }
                });
                let _ = hyper::server::conn::http1::Builder::new()
                    .serve_connection(TokioIo::new(tcp), service)
                    .await;
            });
        }
    });

    // A managed-identity endpoint (App Service style) that has no identity for us, and no az/azd.
    std::env::set_var(
        "IDENTITY_ENDPOINT",
        format!("http://127.0.0.1:{port}/msi/token"),
    );
    std::env::set_var("IDENTITY_HEADER", "secret");
    std::env::set_var("S3DLIO_AZURE_AUTO_MI_TIMEOUT_MS", "1500");
    let empty = tempfile::tempdir().unwrap();
    std::env::set_var("PATH", empty.path());

    let auto = credential_for(CredentialMode::Auto).unwrap();
    let started = std::time::Instant::now();
    let err = tokio::time::timeout(
        std::time::Duration::from_secs(60),
        auto.get_token(&["https://storage.azure.com/.default"], None),
    )
    .await
    .expect("auto answers in bounded time")
    .unwrap_err();
    let took = started.elapsed();
    let message = format!(
        "{err:#} / {:?}",
        std::error::Error::source(&err).map(|s| s.to_string())
    );
    eprintln!(
        "auto failed after {took:?}: {message}; identity endpoint hits: {}",
        hits.load(Ordering::SeqCst)
    );

    assert!(
        took < std::time::Duration::from_secs(10),
        "RED: auto waited {took:?} on a managed identity that answers 404"
    );
    assert!(
        hits.load(Ordering::SeqCst) >= 1,
        "the identity endpoint was asked"
    );
    assert!(message.contains("managed-identity"), "{message}");
    assert!(
        message.contains("1500 ms"),
        "the deadline is named: {message}"
    );
    assert!(
        message.contains("cli"),
        "the developer tools were tried after it: {message}"
    );
}
