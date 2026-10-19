// tests/gcs_emulator_endpoint.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// The official GCS client honours `STORAGE_EMULATOR_HOST`: a real `GcsClient`, built from the
// environment the way an application builds it, sends its HTTP requests and its gRPC connection
// to the emulator named there, and does so anonymously.
//
// Local only: a loopback listener plays the emulator.  It answers one HTTP media download and
// records every connection it sees, so the test can tell an HTTP/1.1 request from the HTTP/2
// preface of a gRPC client.  No cloud, no container.
//
// The client is a process-wide singleton (a `OnceCell`), so this binary holds ONE test and sets
// the environment before the first `GcsClient::new()`.

#![cfg(feature = "backend-gcs")]

use s3dlio::google_gcs_client::GcsClient;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

const BODY: &[u8] = b"hello emulator";
const H2_PREFACE: &[u8] = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";

/// What the fake emulator saw on one connection.
#[derive(Clone, Debug)]
enum Seen {
    Http {
        method: String,
        target: String,
        headers: Vec<(String, String)>,
    },
    Http2Preface,
}

impl Seen {
    fn header(&self, name: &str) -> Option<&str> {
        match self {
            Seen::Http { headers, .. } => headers
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(name))
                .map(|(_, v)| v.as_str()),
            _ => None,
        }
    }
}

/// Read until the end of the request head (or the HTTP/2 preface), reply, record.
async fn serve_one(mut socket: tokio::net::TcpStream, seen: Arc<Mutex<Vec<Seen>>>) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        match socket.read(&mut chunk).await {
            Ok(0) | Err(_) => break,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
        if buf.starts_with(H2_PREFACE) || buf.windows(4).any(|w| w == b"\r\n\r\n") {
            break;
        }
        if buf.len() > 64 * 1024 {
            break;
        }
    }
    if buf.starts_with(b"PRI ") {
        if let Ok(mut guard) = seen.lock() {
            guard.push(Seen::Http2Preface);
        }
        // Drop the connection: this fake does not speak gRPC.
        return;
    }
    let text = String::from_utf8_lossy(&buf).to_string();
    let mut lines = text.split("\r\n");
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split(' ');
    let (method, target) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
    let headers: Vec<(String, String)> = lines
        .take_while(|l| !l.is_empty())
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
        .collect();
    if let Ok(mut guard) = seen.lock() {
        guard.push(Seen::Http {
            method: method.to_string(),
            target: target.to_string(),
            headers,
        });
    }
    let head = format!(
        "HTTP/1.1 200 OK\r\ncontent-type: application/octet-stream\r\ncontent-length: {}\r\n\
         x-goog-generation: 7\r\nx-goog-stored-content-length: {}\r\nconnection: close\r\n\r\n",
        BODY.len(),
        BODY.len()
    );
    let _ = socket.write_all(head.as_bytes()).await;
    let _ = socket.write_all(BODY).await;
    let _ = socket.shutdown().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn client_built_from_storage_emulator_host_talks_to_it_anonymously() {
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("address");
    let seen: Arc<Mutex<Vec<Seen>>> = Arc::new(Mutex::new(Vec::new()));
    let accept = {
        let seen = seen.clone();
        tokio::spawn(async move {
            while let Ok((socket, _)) = listener.accept().await {
                tokio::spawn(serve_one(socket, seen.clone()));
            }
        })
    };

    // The emulator convention: host:port without a scheme.  Application Default Credentials are
    // pointed at a file that does not exist: a client that looked for them would fail to build,
    // so a client that builds proves the emulator path does not use them.
    std::env::remove_var("GCS_ENDPOINT_URL");
    // Standard buckets only: Rapid is explicitly off (the existing `false` value of
    // S3DLIO_GCS_RAPID), so no bucket-layout request precedes the read.  Nothing here changes
    // how the library picks the Rapid mode.
    std::env::set_var("S3DLIO_GCS_RAPID", "false");
    std::env::set_var("STORAGE_EMULATOR_HOST", addr.to_string());
    std::env::set_var(
        "GOOGLE_APPLICATION_CREDENTIALS",
        "/nonexistent/s3dlio-test-adc.json",
    );

    let client = GcsClient::new()
        .await
        .map_err(|e| e.to_string())
        .expect("a client for the emulator builds without credentials");

    // HTTP media download (standard bucket).
    let got = client
        .get_object("bkt", "dir/obj")
        .await
        .map_err(|e| e.to_string())
        .expect("read from the emulator");
    assert_eq!(got.as_ref(), BODY);

    let all = seen.lock().expect("lock").clone();
    let prefaces_before = all
        .iter()
        .filter(|s| matches!(s, Seen::Http2Preface))
        .count();
    let requests: Vec<Seen> = all
        .into_iter()
        .filter(|s| !matches!(s, Seen::Http2Preface))
        .collect();
    assert_eq!(requests.len(), 1, "exactly one request, got {requests:?}");
    let Seen::Http { method, target, .. } = &requests[0] else {
        panic!("the read was not a plain HTTP request: {:?}", requests[0]);
    };
    assert_eq!(method, "GET");
    assert!(
        target.starts_with("/storage/v1/b/bkt/o/dir%2Fobj?") && target.contains("alt=media"),
        "unexpected request target {target}"
    );
    // The request reached this listener, so it went to the emulator.  The SDK keeps the default
    // service host in the Host header of a request to a custom endpoint.
    assert_eq!(requests[0].header("host"), Some("storage.googleapis.com"));
    assert_eq!(
        requests[0].header("authorization"),
        None,
        "anonymous: no bearer token is sent to an emulator"
    );

    // The metadata (StorageControl) client is gRPC: its connection must reach the same
    // emulator.  This fake does not speak gRPC, so the call fails; what matters is where the
    // connection went.
    let stat = tokio::time::timeout(
        Duration::from_secs(5),
        client.get_object_metadata("bkt", "dir/obj"),
    )
    .await;
    assert!(
        !matches!(stat, Ok(Ok(_))),
        "the fake emulator cannot answer a gRPC call"
    );
    let prefaces_after = seen
        .lock()
        .expect("lock")
        .iter()
        .filter(|s| matches!(s, Seen::Http2Preface))
        .count();
    assert!(
        prefaces_after > prefaces_before,
        "the StorageControl (gRPC) client did not connect to the emulator \
         (HTTP/2 connections before {prefaces_before}, after {prefaces_after})"
    );

    accept.abort();
}
