// tests/run_on_global_rt_contexts_300.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// #300: `s3_client::run_on_global_rt` (and every sync wrapper built on it) must work, and must
// never panic or hang, in EVERY context a caller can be in. `block_in_place` panics when a tokio
// `LocalSet` is driven by a thread that is not a runtime worker, so the call may only be used on
// threads known to be workers (`runtime_support`'s marker).
//
// Each context runs in its own thread under a wall-clock timeout, twice: on runtimes built with
// the s3dlio worker marker and on plain runtimes. The scenario is `run_on_global_rt(async {Ok(42)})`
// (no network I/O at all), plus `stat_object_uri` against an unreachable local port for the
// issue's repro (it reaches `run_on_global_rt` before any network I/O).
//
// The decision rule under test is documented on `runtime_support::blocking_wait`: hand off
// (`block_in_place`) on a marked worker, and on an unmarked worker that bears tokio's default
// thread name; wait plainly everywhere else.
//
// A hang wedges the global runtime, so every row has a SHORT timeout (12 s) and the matrix stops at
// the first hang: a regression fails in about a minute, not an hour. The tests of this binary
// share process-wide state (env vars, the global runtime, the hand-off counter) and are serialized.
//
// `MATRIX_REPORT=1` prints the table and does not assert (used to record RED results).

use s3dlio::s3_client::{run_on_global_rt, spawn_on_global_rt};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Mutex;
use std::time::Duration;

static MARKED: AtomicBool = AtomicBool::new(false);
/// Serializes the tests of this binary.
static SERIAL: Mutex<()> = Mutex::new(());

const ROW_TIMEOUT: Duration = Duration::from_secs(12);

fn call() -> u32 {
    run_on_global_rt(async { Ok::<u32, anyhow::Error>(42) }).expect("run_on_global_rt")
}

/// The issue's repro API: a public sync wrapper. No endpoint is reachable, so Err is expected;
/// what matters is that it returns.
fn stat_call() -> u32 {
    std::env::set_var("AWS_ACCESS_KEY_ID", "x");
    std::env::set_var("AWS_SECRET_ACCESS_KEY", "x");
    std::env::set_var("AWS_REGION", "us-east-1");
    std::env::set_var("AWS_ENDPOINT_URL", "http://127.0.0.1:9");
    let r = s3dlio::s3_utils::stat_object_uri("s3://bucket/key");
    assert!(r.is_err(), "unreachable endpoint must be an Err");
    42
}

fn mt(workers: usize) -> tokio::runtime::Runtime {
    let mut b = tokio::runtime::Builder::new_multi_thread();
    b.worker_threads(workers).enable_all();
    if MARKED.load(Ordering::SeqCst) {
        b.on_thread_start(s3dlio::runtime_support::mark_tokio_worker_thread);
    }
    b.build().unwrap()
}

/// A runtime whose threads have CUSTOM names (the default-name rule does not apply).
fn mt_named(workers: usize) -> tokio::runtime::Runtime {
    let mut b = tokio::runtime::Builder::new_multi_thread();
    b.worker_threads(workers)
        .enable_all()
        .thread_name("embedder-worker");
    if MARKED.load(Ordering::SeqCst) {
        b.on_thread_start(s3dlio::runtime_support::mark_tokio_worker_thread);
    }
    b.build().unwrap()
}

fn ct() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

type Ctx = (&'static str, fn(fn() -> u32) -> u32);

fn contexts() -> Vec<Ctx> {
    vec![
        ("no runtime: plain thread", |f| f()),
        ("current-thread: block_on", |f| {
            ct().block_on(async move { f() })
        }),
        ("current-thread: LocalSet::run_until", |f| {
            let rt = ct();
            let l = tokio::task::LocalSet::new();
            rt.block_on(l.run_until(async move { f() }))
        }),
        ("current-thread: spawn_local", |f| {
            let rt = ct();
            let l = tokio::task::LocalSet::new();
            rt.block_on(l.run_until(async move {
                tokio::task::spawn_local(async move { f() }).await.unwrap()
            }))
        }),
        ("multi-thread: block_on on the main thread", |f| {
            mt(2).block_on(async move { f() })
        }),
        ("multi-thread: spawned task on a worker", |f| {
            mt(2).block_on(async move { tokio::spawn(async move { f() }).await.unwrap() })
        }),
        ("multi-thread, custom thread names: spawned task", |f| {
            let rt = mt_named(2);
            rt.block_on(async move { tokio::spawn(async move { f() }).await.unwrap() })
        }),
        ("multi-thread, custom thread names: spawn_local in LocalSet", |f| {
            let rt = mt_named(2);
            let l = tokio::task::LocalSet::new();
            rt.block_on(l.run_until(async move {
                tokio::task::spawn_local(async move { f() }).await.unwrap()
            }))
        }),
        (
            "multi-thread: LocalSet driven from a worker task (block_in_place + block_on)",
            |f| {
                mt(2).block_on(async move {
                    tokio::spawn(async move {
                        tokio::task::block_in_place(|| {
                            let h = tokio::runtime::Handle::current();
                            let l = tokio::task::LocalSet::new();
                            h.block_on(l.run_until(async move {
                                tokio::task::spawn_local(async move { f() }).await.unwrap()
                            }))
                        })
                    })
                    .await
                    .unwrap()
                })
            },
        ),
        ("multi-thread, 1 worker: spawned task", |f| {
            mt(1).block_on(async move { tokio::spawn(async move { f() }).await.unwrap() })
        }),
        ("multi-thread: LocalSet::run_until in block_on", |f| {
            let rt = mt(2);
            let l = tokio::task::LocalSet::new();
            rt.block_on(l.run_until(async move { f() }))
        }),
        ("multi-thread: LocalSet::block_on", |f| {
            let rt = mt(2);
            let l = tokio::task::LocalSet::new();
            l.block_on(&rt, async move { f() })
        }),
        ("multi-thread: spawn_local in LocalSet", |f| {
            let rt = mt(2);
            let l = tokio::task::LocalSet::new();
            rt.block_on(l.run_until(async move {
                tokio::task::spawn_local(async move { f() }).await.unwrap()
            }))
        }),
        ("multi-thread, 1 worker: LocalSet::run_until", |f| {
            let rt = mt(1);
            let l = tokio::task::LocalSet::new();
            rt.block_on(l.run_until(async move { f() }))
        }),
        (
            "multi-thread: futures::executor::block_on under rt.enter()",
            |f| {
                let rt = mt(2);
                let _g = rt.enter();
                futures::executor::block_on(async move { f() })
            },
        ),
        ("multi-thread: spawn_blocking + Handle::block_on", |f| {
            let rt = mt(2);
            let h = rt.handle().clone();
            rt.block_on(async move {
                tokio::task::spawn_blocking(move || h.block_on(async move { f() }))
                    .await
                    .unwrap()
            })
        }),
        (
            "multi-thread: spawn_blocking + Handle::block_on(LocalSet), no inline scope",
            |f| {
                let rt = mt(2);
                let h = rt.handle().clone();
                rt.block_on(async move {
                    tokio::task::spawn_blocking(move || {
                        let l = tokio::task::LocalSet::new();
                        h.block_on(l.run_until(async move { f() }))
                    })
                    .await
                    .unwrap()
                })
            },
        ),
        ("multi-thread: block_in_place + Handle::block_on", |f| {
            mt(2).block_on(async move {
                tokio::spawn(async move {
                    tokio::task::block_in_place(|| {
                        tokio::runtime::Handle::current().block_on(async move { f() })
                    })
                })
                .await
                .unwrap()
            })
        }),
        (
            "current-thread block_on, multi-thread handle entered",
            |f| {
                let m = mt(2);
                let _g = m.enter();
                ct().block_on(async move { f() })
            },
        ),
        ("global-runtime worker (spawn_on_global_rt)", |f| {
            ct().block_on(async move { spawn_on_global_rt(async move { f() }).await.unwrap() })
        }),
        (
            "global-runtime spawn_blocking + Handle::block_on(LocalSet), no inline scope",
            |f| {
                ct().block_on(async move {
                    spawn_on_global_rt(async move {
                        tokio::task::spawn_blocking(move || {
                            let h = tokio::runtime::Handle::current();
                            let l = tokio::task::LocalSet::new();
                            h.block_on(l.run_until(async move { f() }))
                        })
                        .await
                        .unwrap()
                    })
                    .await
                    .unwrap()
                })
            },
        ),
        (
            "multi-thread, custom thread names: spawn_blocking + Handle::block_on(LocalSet) + spawn_local",
            |f| {
                let rt = mt_named(2);
                let h = rt.handle().clone();
                rt.block_on(async move {
                    tokio::task::spawn_blocking(move || {
                        let l = tokio::task::LocalSet::new();
                        h.block_on(l.run_until(async move {
                            tokio::task::spawn_local(async move { f() }).await.unwrap()
                        }))
                    })
                    .await
                    .unwrap()
                })
            },
        ),
        (
            "multi-thread: spawn_blocking + Handle::block_on(LocalSet), inline scope",
            |f| {
                let rt = mt(2);
                let h = rt.handle().clone();
                rt.block_on(async move {
                    tokio::task::spawn_blocking(move || {
                        let _inline = s3dlio::runtime_support::inline_blocking_io();
                        let l = tokio::task::LocalSet::new();
                        h.block_on(l.run_until(async move { f() }))
                    })
                    .await
                    .unwrap()
                })
            },
        ),
        (
            "global-runtime spawn_blocking + Handle::block_on(LocalSet), inline scope",
            |f| {
                ct().block_on(async move {
                    spawn_on_global_rt(async move {
                        tokio::task::spawn_blocking(move || {
                            let _inline = s3dlio::runtime_support::inline_blocking_io();
                            let h = tokio::runtime::Handle::current();
                            let l = tokio::task::LocalSet::new();
                            h.block_on(l.run_until(async move { f() }))
                        })
                        .await
                        .unwrap()
                    })
                    .await
                    .unwrap()
                })
            },
        ),
        ("global-runtime: 256 workers nested at once", |_f| {
            ct().block_on(async move {
                let mut v = Vec::new();
                for _ in 0..256 {
                    v.push(spawn_on_global_rt(async { call() }));
                }
                let mut s = 0;
                for j in v {
                    s += j.await.unwrap();
                }
                assert_eq!(s, 256 * 42);
                42
            })
        }),
    ]
}

fn run_all(f: fn() -> u32, rounds: &[bool]) -> Vec<String> {
    let report = std::env::var("MATRIX_REPORT").is_ok();
    let mut failures = Vec::new();
    for &marked in rounds {
        MARKED.store(marked, Ordering::SeqCst);
        let round = if marked { "marked" } else { "unmarked" };
        for (name, run) in contexts() {
            let (tx, rx) = mpsc::channel();
            let h = std::thread::Builder::new()
                .name(name.into())
                .spawn(move || {
                    let r = std::panic::catch_unwind(|| run(f));
                    let _ = tx.send(r.map_err(|p| {
                        p.downcast_ref::<String>()
                            .cloned()
                            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                            .unwrap_or_default()
                    }));
                })
                .unwrap();
            let verdict = match rx.recv_timeout(ROW_TIMEOUT) {
                Ok(Ok(42)) => "OK".to_string(),
                Ok(Ok(v)) => format!("WRONG {v}"),
                Ok(Err(msg)) => format!("PANIC: {msg}"),
                Err(_) => "HANG".to_string(),
            };
            if verdict == "OK" {
                let _ = h.join();
            }
            // The documented limit: tokio cannot tell a marked spawn_blocking thread that drives a
            // LocalSet from a worker, so such a closure must use `inline_blocking_io()` (the rows
            // with an inline scope). Reported, not asserted, for the marked round.
            let known_limit = name.ends_with("no inline scope");
            println!(
                "MATRIX|{round}|{name}|{verdict}{}",
                if known_limit && verdict != "OK" {
                    " (known limit)"
                } else {
                    ""
                }
            );
            if verdict != "OK" && !known_limit {
                failures.push(format!("[{round}] {name}: {verdict}"));
            }
            if verdict == "HANG" && !known_limit {
                // The global runtime is wedged now: every later row would burn its whole timeout.
                failures.push("stopped at the first hang (global runtime wedged)".into());
                return if report { Vec::new() } else { failures };
            }
        }
    }
    if report {
        Vec::new()
    } else {
        failures
    }
}

// One test: both scenarios share the process-wide MARKED switch and the global runtime.
#[test]
fn run_on_global_rt_and_its_sync_wrappers_work_in_every_context() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let mut failures = run_all(call, &[true, false]);
    // The issue's reproduction: a public sync wrapper, no real S3.
    if failures.is_empty() {
        failures.extend(run_all(stat_call, &[true, false]));
    }
    assert!(failures.is_empty(), "failing contexts: {failures:#?}");
}

// ---------------------------------------------------------------------------------------------
// The reviewer's HANG case and the hand-off accounting.
// ---------------------------------------------------------------------------------------------

use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// A fake S3 endpoint on the CALLER'S OWN runtime: it answers every request with 404 NoSuchKey.
/// A worker parked without a hand-off cannot serve it.
async fn start_fake_s3() -> u16 {
    let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = l.local_addr().unwrap().port();
    tokio::spawn(async move {
        loop {
            let (mut s, _) = l.accept().await.unwrap();
            tokio::spawn(async move {
                let mut buf = vec![0u8; 8192];
                let _ = s.read(&mut buf).await;
                let body = "<?xml version=\"1.0\"?><Error><Code>NoSuchKey</Code><Message>x</Message></Error>";
                let resp = format!(
                    "HTTP/1.1 404 Not Found\r\nContent-Type: application/xml\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = s.write_all(resp.as_bytes()).await;
                let _ = s.shutdown().await;
            });
        }
    });
    port
}

fn point_endpoint_at(port: u16) {
    std::env::set_var("AWS_ACCESS_KEY_ID", "x");
    std::env::set_var("AWS_SECRET_ACCESS_KEY", "x");
    std::env::set_var("AWS_REGION", "us-east-1");
    std::env::set_var("AWS_ENDPOINT_URL", format!("http://127.0.0.1:{port}"));
}

/// Run `f` on its own thread; panic (test failure) on a panic or after [`ROW_TIMEOUT`].
fn bounded<T: Send + 'static>(what: &str, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)));
    });
    match rx.recv_timeout(ROW_TIMEOUT) {
        Ok(Ok(v)) => v,
        Ok(Err(_)) => panic!("{what}: PANIC"),
        Err(_) => panic!("{what}: HANG (no result in {ROW_TIMEOUT:?})"),
    }
}

fn stat_is_err() -> bool {
    s3dlio::s3_utils::stat_object_uri("s3://bucket/key").is_err()
}

/// The hang the first prototype introduced: an UNMARKED multi-thread runtime (a plain
/// `#[tokio::main]`, `#[tokio::test(flavor = "multi_thread")]`) that serves the endpoint itself,
/// with as many concurrent sync-wrapper callers as it has workers. Without a hand-off every worker
/// parks and nobody serves the listener.
#[test]
fn unmarked_runtime_serving_its_own_endpoint_does_not_hang() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    MARKED.store(false, Ordering::SeqCst);
    for workers in [1usize, 2, 4] {
        let errs = bounded(&format!("unmarked, {workers} worker(s)"), move || {
            let rt = mt(workers);
            rt.block_on(async move {
                let port = start_fake_s3().await;
                point_endpoint_at(port);
                let v: Vec<_> = (0..workers)
                    .map(|_| tokio::spawn(async { stat_is_err() }))
                    .collect();
                let mut n = 0;
                for j in v {
                    n += usize::from(j.await.unwrap());
                }
                n
            })
        });
        assert_eq!(errs, workers, "every stat must return the 404 as an Err");
    }
}

/// The capacity regression: on an unmarked ONE-worker runtime a 50 ms timer task must not be
/// delayed by seconds while a neighbour waits in a sync wrapper for the global runtime (1 s).
#[test]
fn unmarked_one_worker_runtime_keeps_serving_neighbours() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    MARKED.store(false, Ordering::SeqCst);
    let lateness = bounded("latency", || {
        mt(1).block_on(async {
            let t0 = Instant::now();
            let tick = tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(50)).await;
                t0.elapsed()
            });
            let caller = tokio::spawn(async {
                run_on_global_rt(async {
                    tokio::time::sleep(Duration::from_millis(1000)).await;
                    Ok(1u32)
                })
                .unwrap()
            });
            let _ = caller.await;
            tick.await.unwrap()
        })
    });
    assert!(
        lateness < Duration::from_millis(500),
        "the 50 ms timer task woke after {lateness:?}: the worker was parked without a hand-off"
    );
}

/// Where the hand-off happens, counted by `handoff_count()` (an exact delta: the tests of this
/// binary are serialized and each scenario is a single call).
#[test]
fn handoff_accounting() {
    use s3dlio::runtime_support::handoff_count;
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());

    let delta = |marked: bool, scenario: fn() -> u32| -> u64 {
        MARKED.store(marked, Ordering::SeqCst);
        bounded("handoff_accounting", move || {
            let before = handoff_count();
            assert_eq!(scenario(), 42);
            handoff_count() - before
        })
    };

    // A marked worker hands off (#173).
    assert_eq!(
        delta(true, || mt(2).block_on(async {
            tokio::spawn(async { call() }).await.unwrap()
        })),
        1,
        "marked worker"
    );
    // An UNMARKED worker with tokio's default thread names hands off too (the reviewer's hang).
    assert_eq!(
        delta(false, || mt(2).block_on(async {
            tokio::spawn(async { call() }).await.unwrap()
        })),
        1,
        "unmarked default-named worker"
    );
    // A LocalSet on the main thread, marked or not: no hand-off, no panic.
    for marked in [true, false] {
        assert_eq!(
            delta(marked, || {
                let rt = mt(2);
                let l = tokio::task::LocalSet::new();
                rt.block_on(l.run_until(async { call() }))
            }),
            0,
            "LocalSet::run_until on the main thread (marked={marked})"
        );
        assert_eq!(
            delta(marked, || {
                let rt = mt(2);
                let l = tokio::task::LocalSet::new();
                rt.block_on(
                    l.run_until(async {
                        tokio::task::spawn_local(async { call() }).await.unwrap()
                    }),
                )
            }),
            0,
            "spawn_local on the main thread (marked={marked})"
        );
    }
    // An unmarked runtime with custom thread names: plain wait (documented residual a).
    assert_eq!(
        delta(false, || mt_named(2).block_on(async {
            tokio::spawn(async { call() }).await.unwrap()
        })),
        0,
        "unmarked custom-named worker"
    );
    // The same runtime marked: hand-off again (the remedy).
    assert_eq!(
        delta(true, || mt_named(2).block_on(async {
            tokio::spawn(async { call() }).await.unwrap()
        })),
        1,
        "marked custom-named worker"
    );
    // An inline scope turns it off on a marked worker.
    assert_eq!(
        delta(true, || mt(2).block_on(async {
            tokio::spawn(async {
                let _g = s3dlio::runtime_support::inline_blocking_io();
                call()
            })
            .await
            .unwrap()
        })),
        0,
        "inline_blocking_io scope"
    );
    // No runtime, current-thread runtime: none.
    assert_eq!(delta(true, call), 0, "no runtime");
    assert_eq!(
        delta(true, || ct().block_on(async { call() })),
        0,
        "current-thread"
    );
}
