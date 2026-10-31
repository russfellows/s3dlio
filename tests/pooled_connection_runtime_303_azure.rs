// tests/pooled_connection_runtime_303_azure.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! #303 for the Azure backend: the cached service client keeps pooled connections, and a pooled
//! connection only makes progress while the runtime that opened it is polled. Every request now
//! runs on s3dlio's global runtime (`azure_client::GlobalRtHttpClient`).
//!
//! Each case runs in its own child process (this binary re-executed on one `--exact` test)
//! against the local fake Blob service of `common/azure_fake.rs`, which keeps connections alive.
//! Azure has no synchronous Rust wrapper, so the stall is reproduced with two live
//! current-thread runtimes (async on A, then async on B) and with current-thread then
//! multi-thread; every step has a 5 s deadline, so a regression fails fast.

#![cfg(feature = "backend-azure")]

#[path = "common/azure_fake.rs"]
mod azure_fake;

use azure_fake::*;
use s3dlio::azure_client::AzureBlob;
use s3dlio::receive_progress::ReceiveProgress;
use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

const CASE_VAR: &str = "S3DLIO_TEST_303_AZURE_CASE";
const SENTINEL: &str = "AZURE-303-OK";
const STEP: Duration = Duration::from_secs(5);

fn start_fake() -> (State, u16) {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        rt.block_on(async move {
            let (state, port) = start().await;
            tx.send((state, port)).unwrap();
            std::future::pending::<()>().await;
        });
    });
    rx.recv().unwrap()
}

fn run_child(case: &str, port: u16) -> (bool, Duration, String) {
    clear_azure_env();
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["child_entry", "--exact", "--nocapture", "--test-threads=1"])
        .env(CASE_VAR, case)
        .env("AZURE_STORAGE_ENDPOINT", format!("http://127.0.0.1:{port}"))
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
                let (state, port) = start_fake();
                set_blob(&state, "obj", pattern(300_000, 5));
                let (ok, elapsed, out) = run_child(stringify!($name), port);
                assert!(ok && elapsed < Duration::from_secs(8),
                    "case `{}` regressed: ok={ok} elapsed={elapsed:?}\n{out}", stringify!($name));
            }
        )*
    };
}

cases!(
    azure_two_ct_runtimes_a_then_b,
    azure_ct_then_mt,
    azure_get_into_on_ct_then_get_into_on_another_ct,
    azure_pooled_on_ct_used_by_a_plain_thread_runtime,
);

fn ct() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn mt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap()
}

async fn head(c: &AzureBlob) {
    let r = tokio::time::timeout(STEP, c.head_typed("obj"))
        .await
        .expect("head stalled (pooled connection owned by a blocked runtime?)")
        .expect("head failed");
    assert_eq!(r.expect("blob exists").size, 300_000);
}

async fn read(c: &AzureBlob) {
    let mut buf = vec![0u8; 300_000];
    let progress = ReceiveProgress::new();
    tokio::time::timeout(STEP, c.get_into("obj", &mut buf, &progress))
        .await
        .expect("get_into stalled")
        .expect("get_into failed");
    assert!(buf == pattern(300_000, 5), "wrong data");
}

#[test]
fn child_entry() {
    let Ok(case) = std::env::var(CASE_VAR) else {
        return;
    };
    let c = AzureBlob::with_default_credential("acct", "ctr").expect("client");
    match case.as_str() {
        "azure_two_ct_runtimes_a_then_b" => {
            let (a, b) = (ct(), ct());
            a.block_on(head(&c));
            b.block_on(head(&c));
            drop(a);
        }
        "azure_ct_then_mt" => {
            let (a, b) = (ct(), mt());
            a.block_on(head(&c));
            b.block_on(head(&c));
            drop(a);
        }
        "azure_get_into_on_ct_then_get_into_on_another_ct" => {
            let (a, b) = (ct(), ct());
            a.block_on(read(&c));
            b.block_on(read(&c));
        }
        // a plain thread blocks on a current-thread runtime's own future while a second
        // runtime that opened the connection earlier is parked
        "azure_pooled_on_ct_used_by_a_plain_thread_runtime" => {
            let a = ct();
            a.block_on(head(&c));
            let h = std::thread::spawn(move || {
                let b = ct();
                b.block_on(read(&c));
            });
            h.join().unwrap();
            drop(a);
        }
        other => panic!("unknown case {other}"),
    }
    println!("{SENTINEL}");
}
