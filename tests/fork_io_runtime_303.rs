// tests/fork_io_runtime_303.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! #297 x #303: a fork child must get its OWN I/O-owner runtime (`IO_RT_HANDLE` is a `ForkCell`).
//! The parent opens pooled S3 connections (sync and current-thread async), then forks; in the
//! child a sync stat and a current-thread async stat must succeed, and `flush_pending_aborts`
//! must start from zero. With a plain `OnceCell` the child enters the parent's runtime, whose
//! threads do not exist after the fork, and every request hangs. The parent waits for the child
//! with a deadline and fails (killing it) instead of hanging.

use bytes::Bytes;
use http_body_util::Full;
use hyper::service::service_fn;
use hyper::{Method, Request, Response};
use hyper_util::rt::TokioIo;
use s3dlio::s3_utils::{stat_object_uri, stat_object_uri_async};
use std::time::{Duration, Instant};

/// A keep-alive HTTP/1.1 mock answering every request with a 5-byte object.
fn start_mock() -> u16 {
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
                    let svc = service_fn(|req: Request<hyper::body::Incoming>| async move {
                        let body = if req.method() == Method::HEAD {
                            Bytes::new()
                        } else {
                            Bytes::from_static(b"hello")
                        };
                        let r = Response::builder()
                            .status(200)
                            .header("content-length", "5")
                            .header("etag", "\"abc\"")
                            .header("last-modified", "Wed, 21 Oct 2015 07:28:00 GMT")
                            .body(Full::new(body))
                            .unwrap();
                        Ok::<_, std::convert::Infallible>(r)
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

fn child_body() -> i32 {
    let sync_ok = stat_object_uri("s3://b/k").is_ok();
    let async_ok = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async { stat_object_uri_async("s3://b/k").await.is_ok() });
    let pending = s3dlio::flush_pending_aborts(Duration::from_millis(200));
    eprintln!("child: sync_ok={sync_ok} async_ok={async_ok} pending_aborts={pending}");
    match (sync_ok, async_ok, pending) {
        (true, true, 0) => 0,
        (false, _, _) => 2,
        (_, false, _) => 3,
        _ => 4,
    }
}

#[test]
fn fork_child_builds_its_own_io_runtime() {
    let port = start_mock();
    std::env::set_var("AWS_ENDPOINT_URL", format!("http://127.0.0.1:{port}"));
    std::env::set_var("AWS_ACCESS_KEY_ID", "x");
    std::env::set_var("AWS_SECRET_ACCESS_KEY", "y");
    std::env::set_var("AWS_REGION", "us-east-1");
    std::env::set_var("S3DLIO_OPERATION_TIMEOUT_SECS", "5");
    std::env::set_var("S3DLIO_FORK_WARN", "0");

    // The parent creates the global and I/O runtimes and pooled connections on both paths.
    stat_object_uri("s3://b/k").expect("parent sync stat");
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(async { stat_object_uri_async("s3://b/k").await })
        .expect("parent async stat");

    // SAFETY: the child only runs `child_body` and leaves with `_exit`.
    let pid = unsafe { libc::fork() };
    assert!(pid >= 0, "fork failed");
    if pid == 0 {
        let code = std::panic::catch_unwind(child_body).unwrap_or(5);
        unsafe { libc::_exit(code) };
    }

    let deadline = Instant::now() + Duration::from_secs(30);
    let mut status = 0;
    loop {
        // SAFETY: waiting for our own child.
        let r = unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG) };
        if r == pid {
            break;
        }
        if Instant::now() > deadline {
            unsafe {
                libc::kill(pid, libc::SIGKILL);
                libc::waitpid(pid, &mut status, 0);
            }
            panic!("fork child hung (> 30 s): it is using the parent's I/O runtime (#297/#303)");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(
        libc::WIFEXITED(status),
        "child did not exit normally: status {status}"
    );
    assert_eq!(
        libc::WEXITSTATUS(status),
        0,
        "child failed (2 = sync stat, 3 = async stat, 4 = inherited abort count, 5 = panic)"
    );
}
