// tests/pooled_connection_runtime_303.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! #303: the process-wide S3 client keeps pooled HTTP connections, and a pooled connection can
//! only make progress while the tokio runtime that OPENED it is being polled. A caller runtime
//! that is alive but blocked (a current-thread runtime whose only thread waits inside a sync
//! s3dlio call) therefore stalled every later request that reused its connection, for the whole
//! operation timeout. s3dlio now runs all of its network I/O on its own global runtime.
//!
//! Every case runs in its OWN child process (this test binary re-executed on one `--exact`
//! test) against an in-process keep-alive mock S3 server, with `S3DLIO_OPERATION_TIMEOUT_SECS=5`,
//! so a regression fails in seconds, not after the default 60 s. The assertions are: the case
//! succeeds, and it finishes in well under the operation timeout (a stalled request would have
//! taken at least 5 s). The server runs in the PARENT on its own runtime.
//!
//! The process-wide client and pool exist once per child, so each child exercises exactly one
//! sequence (open a connection on the caller's runtime, then use it again).

use bytes::Bytes;
use http_body_util::Full;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as AutoBuilder;
use s3dlio::s3_utils::{
    delete_objects, get_object_uri, get_object_uri_async, stat_object_uri, stat_object_uri_async,
};
use s3dlio::{MultipartUploadConfig, MultipartUploadSink};
use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const CASE_VAR: &str = "S3DLIO_TEST_303_CASE";
const SENTINEL: &str = "CASE-303-OK";
/// The operation timeout the children run with: a stalled request takes at least this long.
const OP_TIMEOUT_SECS: u64 = 5;
/// A healthy case finishes far below the operation timeout.
const FAST: Duration = Duration::from_secs(3);
const HARD_KILL: Duration = Duration::from_secs(40);

// ---------------------------------------------------------------------------------------------
// Mock server (parent side): keep-alive HTTP/1.1, logs one line per request.
// ---------------------------------------------------------------------------------------------

struct Mock {
    port: u16,
    log: Arc<Mutex<Vec<String>>>,
    /// Set for the https mock: the PEM file of its (self-signed) certificate.
    ca: Option<std::path::PathBuf>,
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    /// http, HTTP/1.1 and h2c prior knowledge
    Auto,
    /// http, HTTP/1.1 only (rejects an h2c prior-knowledge preface)
    H1,
    /// https with ALPN h2 / http/1.1
    Tls,
}

fn start_mock() -> Mock {
    start_mock_kind(Kind::Auto)
}

fn start_mock_kind(kind: Kind) -> Mock {
    let ca_path = (kind == Kind::Tls).then(|| {
        std::env::temp_dir().join(
            format!(
                "s3dlio303-ca-{}-{:?}.pem",
                std::process::id(),
                std::thread::current().id()
            )
            .replace(['(', ')'], ""),
        )
    });
    let acceptor = ca_path.as_ref().map(|path| {
        let rcgen::CertifiedKey { cert, key_pair } =
            rcgen::generate_simple_self_signed(vec!["localhost".to_owned()]).unwrap();
        std::fs::write(path, cert.pem()).unwrap();
        let mut cfg = rustls::ServerConfig::builder_with_provider(Arc::new(
            rustls::crypto::aws_lc_rs::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_no_client_auth()
        .with_single_cert(
            vec![rustls::pki_types::CertificateDer::from(cert.der().to_vec())],
            rustls::pki_types::PrivateKeyDer::Pkcs8(rustls::pki_types::PrivatePkcs8KeyDer::from(
                key_pair.serialize_der(),
            )),
        )
        .unwrap();
        cfg.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];
        tokio_rustls::TlsAcceptor::from(Arc::new(cfg))
    });
    let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let (tx, rx) = std::sync::mpsc::channel();
    let log2 = log.clone();
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
                let log = log2.clone();
                let acceptor = acceptor.clone();
                tokio::spawn(async move {
                    let svc = service_fn(move |req: Request<hyper::body::Incoming>| {
                        let log = log.clone();
                        async move { Ok::<_, std::convert::Infallible>(handle(req, log).await) }
                    });
                    match (kind, acceptor) {
                        (Kind::H1, _) => {
                            let _ = hyper::server::conn::http1::Builder::new()
                                .serve_connection(TokioIo::new(sock), svc)
                                .await;
                        }
                        (_, Some(acc)) => {
                            let Ok(tls) = acc.accept(sock).await else {
                                return;
                            };
                            let _ = AutoBuilder::new(TokioExecutor::new())
                                .serve_connection(TokioIo::new(tls), svc)
                                .await;
                        }
                        _ => {
                            let _ = AutoBuilder::new(TokioExecutor::new())
                                .serve_connection(TokioIo::new(sock), svc)
                                .await;
                        }
                    }
                });
            }
        });
    });
    Mock {
        port: rx.recv().unwrap(),
        log,
        ca: ca_path,
    }
}

async fn handle(
    req: Request<hyper::body::Incoming>,
    log: Arc<Mutex<Vec<String>>>,
) -> Response<Full<Bytes>> {
    use http_body_util::BodyExt;
    let (parts, body) = req.into_parts();
    let _ = body.collect().await;
    let q = parts.uri.query().unwrap_or("").to_string();
    let path = parts.uri.path().to_string();
    let xml = |s: &str| {
        Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "application/xml")
            .body(Full::new(Bytes::from(s.to_string())))
            .unwrap()
    };
    let empty = |code: StatusCode| {
        Response::builder()
            .status(code)
            .body(Full::new(Bytes::new()))
            .unwrap()
    };
    let tag = match (&parts.method, q.as_str()) {
        (&Method::POST, q) if q.starts_with("uploads") => "CREATE",
        (&Method::POST, q) if q.starts_with("delete") => "DELETEMANY",
        (&Method::POST, _) => "COMPLETE",
        (&Method::DELETE, q) if q.contains("uploadId") => "ABORT",
        (&Method::DELETE, _) => "DELETE",
        (&Method::PUT, _) => "PUT",
        (&Method::HEAD, _) => "HEAD",
        (&Method::GET, _) => "GET",
        _ => "OTHER",
    };
    log.lock().unwrap().push(format!("{tag} {path}?{q}"));
    if tag == "HEAD" && path.contains("slow-403") {
        // a long call that ends in an error that is NOT the operation timeout
        tokio::time::sleep(Duration::from_millis(2500)).await;
        return Response::builder()
            .status(StatusCode::FORBIDDEN)
            .body(Full::new(Bytes::new()))
            .unwrap();
    }
    if tag == "ABORT" && path.contains("abort-fails") {
        return Response::builder()
            .status(StatusCode::INTERNAL_SERVER_ERROR)
            .body(Full::new(Bytes::new()))
            .unwrap();
    }
    if tag == "ABORT" && path.contains("slow-abort") {
        // An AbortMultipartUpload that takes a while to answer: a Drop that waits for it is
        // visible whatever else is fixed.
        tokio::time::sleep(Duration::from_millis(1500)).await;
    }
    match tag {
        "CREATE" => xml("<?xml version=\"1.0\"?><InitiateMultipartUploadResult><Bucket>b</Bucket><Key>k</Key><UploadId>UP1</UploadId></InitiateMultipartUploadResult>"),
        "COMPLETE" => xml("<?xml version=\"1.0\"?><CompleteMultipartUploadResult><Bucket>b</Bucket><Key>k</Key><ETag>\"e\"</ETag></CompleteMultipartUploadResult>"),
        "DELETEMANY" => xml("<?xml version=\"1.0\"?><DeleteResult></DeleteResult>"),
        "ABORT" | "DELETE" => empty(StatusCode::NO_CONTENT),
        "PUT" => Response::builder()
            .status(StatusCode::OK)
            .header("etag", "\"p\"")
            .body(Full::new(Bytes::new()))
            .unwrap(),
        "HEAD" | "GET" => Response::builder()
            .status(StatusCode::OK)
            .header("content-length", "5")
            .header("etag", "\"abc\"")
            .header("last-modified", "Wed, 21 Oct 2015 07:28:00 GMT")
            // a HEAD answer announces the length and carries no body (a body on HTTP/2 is a
            // protocol error)
            .body(Full::new(if tag == "HEAD" {
                Bytes::new()
            } else {
                Bytes::from_static(b"hello")
            }))
            .unwrap(),
        _ => empty(StatusCode::NOT_IMPLEMENTED),
    }
}

// ---------------------------------------------------------------------------------------------
// Parent: run one case in a child process.
// ---------------------------------------------------------------------------------------------

struct Outcome {
    elapsed: Duration,
    ok: bool,
    output: String,
}

fn run_child(case: &str, mock: &Mock) -> Outcome {
    run_child_with(case, mock, OP_TIMEOUT_SECS)
}

fn run_child_with(case: &str, mock: &Mock, op_timeout_secs: u64) -> Outcome {
    let exe = std::env::current_exe().unwrap();
    let mut cmd = Command::new(exe);
    cmd.args(["child_entry", "--exact", "--nocapture", "--test-threads=1"])
        .env(CASE_VAR, case)
        .env(
            "AWS_ENDPOINT_URL",
            if mock.ca.is_some() {
                format!("https://localhost:{}", mock.port)
            } else {
                format!("http://127.0.0.1:{}", mock.port)
            },
        )
        .env("AWS_ACCESS_KEY_ID", "x")
        .env("AWS_SECRET_ACCESS_KEY", "y")
        .env("AWS_REGION", "us-east-1")
        .env("S3DLIO_OPERATION_TIMEOUT_SECS", op_timeout_secs.to_string())
        .env_remove("AWS_ENDPOINT_URL_S3")
        .env_remove("AWS_CA_BUNDLE")
        .env_remove("S3DLIO_POOL_MAX_IDLE_PER_HOST")
        .env_remove("S3DLIO_POOL_IDLE_TIMEOUT_SECS")
        .env_remove("S3DLIO_H2C")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(ca) = &mock.ca {
        cmd.env("AWS_CA_BUNDLE", ca);
    }
    let t0 = Instant::now();
    let mut child = cmd.spawn().unwrap();
    let mut so = child.stdout.take().unwrap();
    let mut se = child.stderr.take().unwrap();
    let h1 = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = so.read_to_string(&mut s);
        s
    });
    let h2 = std::thread::spawn(move || {
        let mut s = String::new();
        let _ = se.read_to_string(&mut s);
        s
    });
    let mut killed = false;
    let status = loop {
        if let Some(st) = child.try_wait().unwrap() {
            break st;
        }
        if t0.elapsed() > HARD_KILL {
            let _ = child.kill();
            killed = true;
            break child.wait().unwrap();
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let elapsed = t0.elapsed();
    let out = h1.join().unwrap();
    let err = h2.join().unwrap();
    Outcome {
        elapsed,
        ok: !killed && status.success() && out.contains(SENTINEL),
        output: format!("{out}\n{err}"),
    }
}

/// The case must succeed and finish quickly. `child_budget` is the time the child itself may
/// take, on top of process start-up.
fn expect_fast(case: &str, mock: &Mock) {
    let o = run_child(case, mock);
    assert!(
        o.ok && o.elapsed < FAST + Duration::from_secs(2),
        "case `{case}` regressed: ok={} elapsed={:?} (a stalled request takes >= {OP_TIMEOUT_SECS} s)\n{}",
        o.ok,
        o.elapsed,
        o.output
    );
}

fn wait_for_log(mock: &Mock, needle: &str, within: Duration) -> bool {
    let t0 = Instant::now();
    while t0.elapsed() < within {
        if mock
            .log
            .lock()
            .unwrap()
            .iter()
            .any(|l| l.starts_with(needle))
        {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    false
}

macro_rules! cases {
    ($($name:ident),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                let mock = start_mock();
                expect_fast(stringify!($name), &mock);
            }
        )*
    };
}

cases!(
    // current-thread runtime: an async call opens the pooled connection, then each sync wrapper
    // blocks the runtime's only thread
    ct_async_then_sync_stat,
    ct_async_then_sync_get,
    ct_async_then_sync_delete,
    ct_async_then_sync_sink_new,
    // HTTP/2 (prior-knowledge cleartext) connections have a driver task of their own
    ct_async_then_sync_stat_h2c,
    // a whole multipart upload (create, parts, complete) on a current-thread runtime, then sync
    ct_multipart_async_then_sync_stat,
    // the same inside a LocalSet driven by a current-thread runtime
    localset_ct_async_then_sync,
    // two live current-thread runtimes, async on A then async on B (no sync call at all)
    two_ct_runtimes_async_a_then_b,
    // async on a current-thread runtime, then async on a multi-thread runtime
    ct_async_then_mt_async,
    // an app runtime with ONE worker, custom thread names and no s3dlio marker: async call,
    // then a sync wrapper (it takes the plain wait, so it parks its only worker)
    unmarked_one_worker_custom_name,
    // controls that always worked
    sync_then_async_ct,
    mt_async_then_sync,
);

macro_rules! kind_cases {
    ($kind:expr => $($name:ident),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                let mock = start_mock_kind($kind);
                expect_fast(stringify!($name), &mock);
            }
        )*
    };
}

// `S3DLIO_H2C=auto` against a server that only speaks HTTP/1.1: the first (h2c) attempt is
// rejected and the request is RETRIED on the HTTP/1.1 client (`reqwest_client` fallback), which
// opens its own connection: it must be owned by the I/O runtime too.
kind_cases!(Kind::H1 => ct_async_then_sync_stat_h2c_auto_fallback_retry);
// https with ALPN (h2) and a custom CA bundle, and the redirect-following wrapper over both
kind_cases!(Kind::Tls => ct_async_then_sync_stat_https, ct_async_then_sync_stat_https_follow_redirects);
kind_cases!(Kind::Auto => ct_async_then_sync_stat_follow_redirects, starve_global_workers_do_not_stall_async_io);

/// `warmup_connection_pool` builds its own client and needs no caller runtime: driven by a
/// runtime-less executor on a plain thread it must work (its sockets belong to the I/O
/// runtime) and open every connection it was asked for.
#[test]
fn warmup_connection_pool_needs_no_caller_runtime() {
    let mock = start_mock();
    expect_fast("warmup_no_runtime", &mock);
    let heads = mock
        .log
        .lock()
        .unwrap()
        .iter()
        .filter(|l| l.starts_with("HEAD"))
        .count();
    assert!(
        heads >= 4,
        "warmup sent {heads} HEAD requests, expected 4: {:?}",
        mock.log.lock().unwrap()
    );
}

/// A call that runs long and then fails for another reason must NOT be told that it blocked its
/// runtime (the hint belongs to the operation-timeout error only).
#[test]
fn a_long_call_that_fails_for_another_reason_gets_no_hint() {
    let mock = start_mock();
    // timeout 3 s, failure at 2.5 s: inside the old "elapsed + 1 s >= timeout" window
    let o = run_child_with("long_failure_no_hint", &mock, 3);
    assert!(
        o.ok && o.elapsed < Duration::from_secs(8),
        "ok={} {:?}\n{}",
        o.ok,
        o.elapsed,
        o.output
    );
}

/// The reviewer's process-exit case: a sink dropped just before the process ends. The abort
/// answers slowly (1.5 s); `flush_pending_aborts` is what keeps the process alive until it was
/// delivered, so the mock has logged it by the time the child has exited.
#[test]
fn a_sink_dropped_before_exit_has_its_abort_delivered_by_flush_pending_aborts() {
    let mock = start_mock();
    let o = run_child_with("drop_then_exit_flush", &mock, OP_TIMEOUT_SECS);
    assert!(o.ok, "{}", o.output);
    // no waiting here: the child has exited
    assert!(
        mock.log
            .lock()
            .unwrap()
            .iter()
            .any(|l| l.starts_with("ABORT")),
        "the abort was not delivered before the process exited: {:?}",
        mock.log.lock().unwrap()
    );
}

#[test]
fn drop_unfinished_sink_on_ct_returns_at_once_and_aborts() {
    let mock = start_mock();
    expect_fast("drop_unfinished_ct", &mock);
    assert!(
        wait_for_log(&mock, "ABORT", Duration::from_secs(3)),
        "the mock never saw AbortMultipartUpload: the upload leaked. log: {:?}",
        mock.log.lock().unwrap()
    );
}

/// The one way a synchronous wrapper can still wait for ever: the awaited work needs the very
/// runtime whose thread the wrapper blocks (an in-process server task on a current-thread
/// runtime). That is a self-deadlock no s3dlio design can remove; the error must name it, not
/// say "check network connectivity".
#[test]
fn a_self_deadlock_on_the_callers_runtime_is_reported_by_name() {
    let mock = start_mock(); // unused by the case (the server is in the child), but it sets the env
    let o = run_child_with("self_deadlock_ct", &mock, 2);
    assert!(
        o.ok && o.elapsed < Duration::from_secs(8),
        "ok={} elapsed={:?}\n{}",
        o.ok,
        o.elapsed,
        o.output
    );
}

/// `abort().await` is new API and reports the REAL outcome (unlike `abort_blocking`, whose locked
/// contract is `Ok(())`): a failed AbortMultipartUpload is an `Err` that says the upload may
/// still exist; a repeat is a no-op.
#[test]
fn abort_reports_a_failed_abort_request() {
    let mock = start_mock();
    expect_fast("abort_error_ct", &mock);
}

#[test]
fn explicit_async_abort_aborts_the_upload() {
    let mock = start_mock();
    expect_fast("explicit_abort_ct", &mock);
    assert!(
        wait_for_log(&mock, "ABORT", Duration::from_secs(1)),
        "abort() sent no AbortMultipartUpload"
    );
}

// ---------------------------------------------------------------------------------------------
// Child side.
// ---------------------------------------------------------------------------------------------

fn ct() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn mt(workers: usize) -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .enable_all()
        .build()
        .unwrap()
}

const URI: &str = "s3://bkt/key";

fn cfg() -> MultipartUploadConfig {
    MultipartUploadConfig {
        part_size: 5 << 20,
        ..Default::default()
    }
}

fn check<T>(what: &str, r: anyhow::Result<T>) {
    if let Err(e) = r {
        panic!("{what} failed: {e:#}");
    }
}

#[test]
fn child_entry() {
    let Ok(case) = std::env::var(CASE_VAR) else {
        return; // not a child: a no-op when the whole binary runs
    };
    let t0 = Instant::now();
    match case.as_str() {
        "ct_async_then_sync_stat" => ct().block_on(async {
            check("async stat", stat_object_uri_async(URI).await);
            check("sync stat", stat_object_uri(URI));
        }),
        "ct_async_then_sync_stat_h2c_auto_fallback_retry" => {
            std::env::set_var("S3DLIO_H2C", "auto");
            ct().block_on(async {
                check("async stat", stat_object_uri_async(URI).await);
                check("sync stat", stat_object_uri(URI));
                check("async stat again", stat_object_uri_async(URI).await);
            })
        }
        "ct_async_then_sync_stat_follow_redirects"
        | "ct_async_then_sync_stat_https_follow_redirects" => {
            std::env::set_var("S3DLIO_FOLLOW_REDIRECTS", "1");
            ct().block_on(async {
                check("async stat", stat_object_uri_async(URI).await);
                check("sync stat", stat_object_uri(URI));
            })
        }
        "ct_async_then_sync_stat_https" => ct().block_on(async {
            check("async stat", stat_object_uri_async(URI).await);
            check("sync stat", stat_object_uri(URI));
            check("async stat again", stat_object_uri_async(URI).await);
        }),
        // the global runtime's workers are all busy with blocking user work: async I/O of an
        // independent runtime must not depend on them (the I/O-owner runtime does the I/O)
        "starve_global_workers_do_not_stall_async_io" => {
            std::env::set_var("S3DLIO_RT_THREADS", "2");
            let a = mt(2);
            a.block_on(async { check("warm", stat_object_uri_async(URI).await) });
            for _ in 0..2 {
                s3dlio::s3_client::spawn_on_global_rt(async {
                    std::thread::sleep(Duration::from_secs(20));
                });
            }
            std::thread::sleep(Duration::from_millis(300));
            let b = mt(2);
            b.block_on(async {
                check(
                    "async stat while the global workers are blocked",
                    stat_object_uri_async(URI).await,
                )
            });
        }
        "warmup_no_runtime" => {
            // no tokio runtime on this thread at all
            let endpoint = std::env::var("AWS_ENDPOINT_URL").unwrap();
            futures::executor::block_on(s3dlio::reqwest_client::warmup_connection_pool(
                &endpoint, 4,
            ));
        }
        "long_failure_no_hint" => ct().block_on(async {
            let e = stat_object_uri("s3://bkt/slow-403").expect_err("403");
            let text = format!("{e:#}");
            assert!(
                !text.contains("blocked a thread that belongs to a tokio runtime"),
                "a non-timeout failure got the self-deadlock hint: {text}"
            );
        }),
        "drop_then_exit_flush" => ct().block_on(async {
            let mut s = MultipartUploadSink::new_async("bkt", "slow-abort-key", cfg())
                .await
                .expect("new_async");
            check("write", s.write(vec![7u8; 11 << 20]).await);
            drop(s);
            // the "exit path": wait for the spawned abort, as the CLI and the Python atexit hook do
            let left = s3dlio::flush_pending_aborts(Duration::from_secs(5));
            assert_eq!(left, 0, "{left} abort(s) still pending");
        }),
        "ct_async_then_sync_stat_h2c" => {
            std::env::set_var("S3DLIO_H2C", "1");
            ct().block_on(async {
                check("async stat", stat_object_uri_async(URI).await);
                check("sync stat", stat_object_uri(URI));
                check("async stat again", stat_object_uri_async(URI).await);
            })
        }
        "ct_multipart_async_then_sync_stat" => ct().block_on(async {
            let mut s = MultipartUploadSink::new_async("bkt", "key", cfg())
                .await
                .expect("new_async");
            check("write", s.write(vec![7u8; 11 << 20]).await);
            check("finish", s.finish().await.map(|_| ()));
            check("sync stat", stat_object_uri(URI));
        }),
        "ct_async_then_sync_get" => ct().block_on(async {
            check("async stat", stat_object_uri_async(URI).await);
            check("sync get", get_object_uri(URI));
        }),
        "ct_async_then_sync_delete" => ct().block_on(async {
            check("async stat", stat_object_uri_async(URI).await);
            check("sync delete", delete_objects("bkt", &["key".to_string()]));
        }),
        "ct_async_then_sync_sink_new" => ct().block_on(async {
            check("async stat", stat_object_uri_async(URI).await);
            let s = MultipartUploadSink::new("bkt", "key", cfg());
            check(
                "sink new",
                s.as_ref().map(|_| ()).map_err(|e| anyhow::anyhow!("{e:#}")),
            );
            // an unfinished sink is dropped here: must not stall either
        }),
        "localset_ct_async_then_sync" => {
            let rt = ct();
            let local = tokio::task::LocalSet::new();
            rt.block_on(local.run_until(async {
                check("async get", get_object_uri_async(URI).await);
                check("sync stat", stat_object_uri(URI));
            }));
        }
        "two_ct_runtimes_async_a_then_b" => {
            let a = ct();
            let b = ct();
            a.block_on(async { check("async on A", stat_object_uri_async(URI).await) });
            // A is alive but not being polled from here on
            b.block_on(async { check("async on B", stat_object_uri_async(URI).await) });
            drop(a);
        }
        "ct_async_then_mt_async" => {
            let a = ct();
            let b = mt(2);
            a.block_on(async { check("async on ct", stat_object_uri_async(URI).await) });
            b.block_on(async { check("async on mt", stat_object_uri_async(URI).await) });
            drop(a);
        }
        "unmarked_one_worker_custom_name" => {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .thread_name("my-app-worker")
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async {
                tokio::spawn(async {
                    check("async stat", stat_object_uri_async(URI).await);
                    check("sync stat", stat_object_uri(URI));
                })
                .await
                .unwrap();
            });
        }
        "sync_then_async_ct" => ct().block_on(async {
            check("sync stat", stat_object_uri(URI));
            check("async stat", stat_object_uri_async(URI).await);
        }),
        "mt_async_then_sync" => mt(2).block_on(async {
            tokio::spawn(async {
                check("async stat", stat_object_uri_async(URI).await);
                check("sync stat", stat_object_uri(URI));
            })
            .await
            .unwrap();
        }),
        "drop_unfinished_ct" => ct().block_on(async {
            let mut s = MultipartUploadSink::new_async("bkt", "slow-abort-key", cfg())
                .await
                .expect("new_async");
            check("write", s.write(vec![7u8; 11 << 20]).await);
            let t = Instant::now();
            drop(s); // unfinished: Drop aborts, without blocking this thread
            assert!(
                t.elapsed() < Duration::from_secs(1),
                "Drop blocked for {:?}",
                t.elapsed()
            );
            // give the global runtime a moment to deliver the abort before the process exits
            tokio::time::sleep(Duration::from_millis(500)).await;
        }),
        "self_deadlock_ct" => ct().block_on(async {
            // An in-process "server" that lives on THIS runtime: it can answer only while this
            // thread is free, and the sync call below blocks this thread.
            let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            std::env::set_var(
                "AWS_ENDPOINT_URL",
                format!("http://{}", l.local_addr().unwrap()),
            );
            tokio::spawn(async move {
                loop {
                    let Ok((sock, _)) = l.accept().await else {
                        break;
                    };
                    tokio::spawn(async move {
                        // would answer, if it could run
                        let _sock = sock;
                        std::future::pending::<()>().await;
                    });
                }
            });
            let r = stat_object_uri(URI);
            let e = r.expect_err("the server can never answer");
            let text = format!("{e:#}");
            assert!(
                text.contains("blocked a thread that belongs to a tokio runtime"),
                "the error does not name the cause: {text}"
            );
        }),
        "abort_error_ct" => ct().block_on(async {
            let mut s = MultipartUploadSink::new_async("bkt", "abort-fails-key", cfg())
                .await
                .expect("new_async");
            check("write", s.write(vec![7u8; 11 << 20]).await);
            let e = s.abort().await.expect_err("the abort request fails (500)");
            let text = format!("{e:#}");
            assert!(text.contains("may still exist on the server"), "{text}");
            assert!(
                text.contains("UP1"),
                "the upload id is in the error: {text}"
            );
            check("second abort is a no-op", s.abort().await);
        }),
        "explicit_abort_ct" => ct().block_on(async {
            let mut s = MultipartUploadSink::new_async("bkt", "key", cfg())
                .await
                .expect("new_async");
            check("write", s.write(vec![7u8; 11 << 20]).await);
            check("abort", s.abort().await);
        }),
        other => panic!("unknown case {other}"),
    }
    assert!(
        t0.elapsed() < FAST + Duration::from_secs(1),
        "case took {:?}",
        t0.elapsed()
    );
    println!("{SENTINEL}");
}
