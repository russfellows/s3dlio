// tests/pooled_connection_runtime_303_gcs.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! #303 for the GCS backend: the process-wide `GcsClient` keeps pooled connections (and the
//! SDK's own background tasks), and they belong to the runtime that polls the call that creates
//! them. Every GCS operation now runs on s3dlio's global runtime.
//!
//! Each case runs in its own child process (this binary re-executed on one `--exact` test)
//! against a keep-alive local emulator (`STORAGE_EMULATOR_HOST`, standard-bucket HTTP media
//! download). Two live current-thread runtimes reproduce the stall (async on A, then async on
//! B); every step has a 5 s deadline, so a regression fails fast.

#![cfg(feature = "backend-gcs")]

use bytes::Bytes;
use http_body_util::Full;
use hyper::service::service_fn;
use hyper::{Request, Response};
use hyper_util::rt::TokioIo;
use s3dlio::google_gcs_client::GcsClient;
use s3dlio::receive_progress::ReceiveProgress;
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const CASE_VAR: &str = "S3DLIO_TEST_303_GCS_CASE";
const SENTINEL: &str = "GCS-303-OK";
const STEP: Duration = Duration::from_secs(5);
const LEN: usize = 100_000;

fn body() -> Vec<u8> {
    (0..LEN).map(|i| (i % 251) as u8).collect()
}

fn start_emulator() -> u16 {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            tx.send(l.local_addr().unwrap().port()).unwrap();
            loop {
                let (sock, _) = l.accept().await.unwrap();
                tokio::spawn(async move {
                    let svc = service_fn(|_req: Request<hyper::body::Incoming>| async {
                        Ok::<_, std::convert::Infallible>(
                            Response::builder()
                                .status(200)
                                .header("content-type", "application/octet-stream")
                                .header("x-goog-generation", "7")
                                .header("x-goog-stored-content-length", LEN.to_string())
                                .body(Full::new(Bytes::from(body())))
                                .unwrap(),
                        )
                    });
                    let _ = hyper::server::conn::http1::Builder::new()
                        .serve_connection(TokioIo::new(sock), svc)
                        .await;
                });
            }
        });
    });
    rx.recv().unwrap()
}

/// A keep-alive emulator that CUTS the body of the first GET after 40 000 of its 100 000 bytes
/// (the body stream fails, hyper aborts the connection) and answers every later GET fully,
/// honouring `Range` (`206` with `Content-Range`, as the live media API does).
fn start_cutting_emulator() -> (u16, std::sync::Arc<std::sync::atomic::AtomicUsize>) {
    use futures::StreamExt;
    use http_body_util::{BodyExt, StreamBody};
    use hyper::body::Frame;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    let (tx, rx) = std::sync::mpsc::channel();
    let count0 = Arc::new(AtomicUsize::new(0));
    let count_for_server = count0.clone();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            tx.send(l.local_addr().unwrap().port()).unwrap();
            let count = count_for_server;
            loop {
                let (sock, _) = l.accept().await.unwrap();
                let count = count.clone();
                tokio::spawn(async move {
                    let svc = service_fn(move |req: Request<hyper::body::Incoming>| {
                        let count = count.clone();
                        async move {
                            let n = count.fetch_add(1, Ordering::SeqCst);
                            let range = req
                                .headers()
                                .get("range")
                                .and_then(|v| v.to_str().ok())
                                .map(str::to_string);
                            let (start, end) =
                                match range.as_deref().and_then(|v| v.strip_prefix("bytes=")) {
                                    Some(spec) => {
                                        let (a, b) = spec.split_once('-').unwrap();
                                        let a: usize = a.parse().unwrap();
                                        let b: usize = if b.is_empty() {
                                            LEN - 1
                                        } else {
                                            b.parse::<usize>().unwrap().min(LEN - 1)
                                        };
                                        (a, b)
                                    }
                                    None => (0, LEN - 1),
                                };
                            let part = Bytes::from(body()[start..=end].to_vec());
                            let mut b = Response::builder()
                                .status(if range.is_some() { 206 } else { 200 })
                                .header("content-type", "application/octet-stream")
                                .header("content-length", part.len().to_string())
                                .header("x-goog-generation", "7")
                                .header("x-goog-stored-content-length", LEN.to_string());
                            if range.is_some() {
                                b = b.header("content-range", format!("bytes {start}-{end}/{LEN}"));
                            }
                            let body: http_body_util::combinators::BoxBody<Bytes, std::io::Error> =
                                if n == 0 {
                                    let first = part.slice(..40_000.min(part.len()));
                                    let frames =
                                        futures::stream::iter(vec![Ok(Frame::data(first))]).chain(
                                            futures::stream::once(async {
                                                tokio::time::sleep(Duration::from_millis(50)).await;
                                                Err(std::io::Error::new(
                                                    std::io::ErrorKind::ConnectionReset,
                                                    "emulator: connection cut",
                                                ))
                                            }),
                                        );
                                    BodyExt::boxed(StreamBody::new(frames))
                                } else {
                                    Full::new(part).map_err(|never| match never {}).boxed()
                                };
                            Ok::<_, std::convert::Infallible>(b.body(body).unwrap())
                        }
                    });
                    let _ = hyper::server::conn::http1::Builder::new()
                        .serve_connection(TokioIo::new(sock), svc)
                        .await;
                });
            }
        });
    });
    (rx.recv().unwrap(), count0)
}

fn run_child(case: &str, port: u16) -> (bool, Duration, String) {
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["child_entry", "--exact", "--nocapture", "--test-threads=1"])
        .env(CASE_VAR, case)
        .env("STORAGE_EMULATOR_HOST", format!("127.0.0.1:{port}"))
        .env("S3DLIO_GCS_RAPID", "false")
        .env_remove("GCS_ENDPOINT_URL")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut so = child.stdout.take().unwrap();
    let mut se = child.stderr.take().unwrap();
    let a = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = so.read_to_string(&mut s);
        s
    });
    let b = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = se.read_to_string(&mut s);
        s
    });
    let t0 = Instant::now();
    let mut killed = false;
    let st = loop {
        if let Some(st) = child.try_wait().unwrap() {
            break st;
        }
        if t0.elapsed() > Duration::from_secs(40) {
            let _ = child.kill();
            killed = true;
            break child.wait().unwrap();
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let out = format!("{}\n{}", a.join().unwrap(), b.join().unwrap());
    (
        !killed && st.success() && out.contains(SENTINEL),
        t0.elapsed(),
        out,
    )
}

macro_rules! cases {
    ($($name:ident),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                let port = start_emulator();
                let (ok, elapsed, out) = run_child(stringify!($name), port);
                assert!(ok && elapsed < Duration::from_secs(8),
                    "case `{}` regressed: ok={ok} elapsed={elapsed:?}\n{out}", stringify!($name));
            }
        )*
    };
}

#[test]
fn gcs_cut_body_repair_connection_belongs_to_the_io_runtime() {
    let (port, requests) = start_cutting_emulator();
    let (ok, elapsed, out) = run_child("gcs_cut_body_repair", port);
    // cut, repair, and the second runtime's read: the body really was cut and repaired
    assert_eq!(
        requests.load(std::sync::atomic::Ordering::SeqCst),
        3,
        "the first body must have been cut and repaired by one new request"
    );
    assert!(
        ok && elapsed < Duration::from_secs(8),
        "regressed: ok={ok} elapsed={elapsed:?}\n{out}"
    );
}

cases!(
    gcs_two_ct_runtimes_get_a_then_b,
    gcs_ct_then_mt_get,
    gcs_get_into_on_two_ct_runtimes,
    gcs_range_get_on_two_ct_runtimes,
);

fn ct() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

async fn get(c: &GcsClient) {
    let got = tokio::time::timeout(STEP, c.get_object("bkt", "dir/obj"))
        .await
        .expect("get_object stalled (pooled connection owned by a blocked runtime?)")
        .expect("get_object failed");
    assert_eq!(got.as_ref(), &body()[..]);
}

async fn get_range(c: &GcsClient) {
    let got = tokio::time::timeout(STEP, c.get_object_range("bkt", "dir/obj", 0, None))
        .await
        .expect("get_object_range stalled")
        .expect("get_object_range failed");
    assert_eq!(got.len(), LEN);
}

async fn read_into(c: &GcsClient) {
    let mut buf = vec![0u8; LEN];
    let progress = ReceiveProgress::new();
    tokio::time::timeout(
        STEP,
        c.get_object_into("bkt", "dir/obj", &mut buf, &progress),
    )
    .await
    .expect("get_object_into stalled")
    .expect("get_object_into failed");
    assert!(buf == body(), "wrong data");
}

#[test]
fn child_entry() {
    let Ok(case) = std::env::var(CASE_VAR) else {
        return;
    };
    let (a, b) = (ct(), ct());
    // the client itself is built on A (before the fix its background tasks lived there)
    let c = a.block_on(async { GcsClient::new().await.expect("client") });
    match case.as_str() {
        "gcs_two_ct_runtimes_get_a_then_b" => {
            a.block_on(get(&c));
            b.block_on(get(&c));
        }
        "gcs_ct_then_mt_get" => {
            let m = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(2)
                .enable_all()
                .build()
                .unwrap();
            a.block_on(get(&c));
            m.block_on(get(&c));
        }
        "gcs_get_into_on_two_ct_runtimes" => {
            a.block_on(read_into(&c));
            b.block_on(read_into(&c));
        }
        // the first body is cut mid-stream; the receive loop repairs it with a NEW request (a
        // new connection), then a second runtime reads again over that connection
        "gcs_cut_body_repair" => {
            a.block_on(read_into(&c));
            b.block_on(read_into(&c));
        }
        "gcs_range_get_on_two_ct_runtimes" => {
            a.block_on(get_range(&c));
            b.block_on(get_range(&c));
        }
        other => panic!("unknown case {other}"),
    }
    drop(a);
    println!("{SENTINEL}");
}
