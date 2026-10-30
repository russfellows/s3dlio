// tests/run_on_global_rt_contexts_300.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// #300: `s3_client::run_on_global_rt` (and every sync wrapper built on it) must work, and must
// never panic or hang, in EVERY context a caller can be in. `block_in_place` panics when a tokio
// `LocalSet` is driven by a thread that is not a runtime worker, so the worker hand-off is used
// only where `runtime_support::blocking_wait` can tell that it is valid (the worker marker, or an
// unmarked worker task on a thread with tokio's default worker name).
//
// The documented residual (b) rows (a thread that carries the marker or the default worker name
// and drives a LocalSet with the call inside a task) are ASSERTED to panic: if one stops
// panicking the test fails with "residual fixed? update the docs".
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

/// tokio's default worker thread name, measured here independently of the code under test.
fn default_name() -> String {
    static NAME: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    NAME.get_or_init(|| {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .build()
            .unwrap();
        let (tx, rx) = mpsc::channel();
        rt.spawn(async move {
            let _ = tx.send(std::thread::current().name().unwrap().to_owned());
        });
        let n = rx.recv().unwrap();
        rt.shutdown_background();
        n
    })
    .clone()
}

/// Run `g` on a plain std thread with exactly this name; a panic in it propagates.
fn on_named_thread(name: String, g: impl FnOnce() -> u32 + Send + 'static) -> u32 {
    std::thread::Builder::new()
        .name(name)
        .spawn(g)
        .unwrap()
        .join()
        .unwrap_or_else(|p| std::panic::resume_unwind(p))
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
        (
            "std thread named like the default worker: LocalSet::run_until, call in the MAIN future",
            |f| {
                on_named_thread(default_name(), move || {
                    let rt = mt(2);
                    let l = tokio::task::LocalSet::new();
                    rt.block_on(l.run_until(async move { f() }))
                })
            },
        ),
        (
            "std thread named like the default worker: LocalSet + spawn_local (known limit)",
            |f| {
                on_named_thread(default_name(), move || {
                    let rt = mt(2);
                    let l = tokio::task::LocalSet::new();
                    rt.block_on(l.run_until(async move {
                        tokio::task::spawn_local(async move { f() }).await.unwrap()
                    }))
                })
            },
        ),
        (
            "current-thread runtime on a thread named like the default worker: spawn_local",
            |f| {
                on_named_thread(default_name(), move || {
                    let rt = ct();
                    let l = tokio::task::LocalSet::new();
                    rt.block_on(l.run_until(async move {
                        tokio::task::spawn_local(async move { f() }).await.unwrap()
                    }))
                })
            },
        ),
        (
            "current-thread runtime on a thread named like the default worker: spawned task",
            |f| {
                on_named_thread(default_name(), move || {
                    ct().block_on(async move { tokio::spawn(async move { f() }).await.unwrap() })
                })
            },
        ),
        (
            "thread name merely CONTAINING the default name: LocalSet + spawn_local",
            |f| {
                on_named_thread(format!("my-{}-1", default_name()), move || {
                    let rt = mt(2);
                    let l = tokio::task::LocalSet::new();
                    rt.block_on(l.run_until(async move {
                        tokio::task::spawn_local(async move { f() }).await.unwrap()
                    }))
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
            // The documented limit (b): tokio cannot tell a `spawn_blocking` thread that drives a
            // LocalSet from a worker, so a thread that carries the marker or tokio's default
            // worker name takes the hand-off and panics there. Such a closure must use
            // `inline_blocking_io()` (the rows with an inline scope). Reported, not asserted.
            let known_limit = name.ends_with("no inline scope")
                || name.contains("(known limit)")
                || (marked && name.contains("custom thread names: spawn_blocking"));
            println!(
                "MATRIX|{round}|{name}|{verdict}{}",
                if known_limit { " (known limit)" } else { "" }
            );
            if known_limit {
                // Asserted, not just reported: the docs promise exactly this panic.
                if !verdict.starts_with("PANIC") {
                    failures.push(format!(
                        "[{round}] {name}: expected the documented PANIC, got {verdict} \
                         (residual fixed? update the docs)"
                    ));
                }
            } else if verdict != "OK" {
                failures.push(format!("[{round}] {name}: {verdict}"));
            }
            if verdict == "HANG" {
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
            SERVED.fetch_add(1, Ordering::SeqCst);
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

/// Hits on the fake endpoint.
static SERVED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Runs one hang scenario in a CHILD process: the global S3 client is built once per process and
/// remembers its endpoint, so each scenario needs a process of its own (and a hang costs one
/// child, killed after [`CHILD_TIMEOUT`], not the whole binary).
const CHILD_TIMEOUT: Duration = Duration::from_secs(20);

fn run_child(case: &str) {
    let exe = std::env::current_exe().unwrap();
    let mut child = std::process::Command::new(exe)
        .args([
            "--exact",
            "child_hang_case",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("S3DLIO_300_CHILD_CASE", case)
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .spawn()
        .unwrap();
    let t0 = Instant::now();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "{case}: child failed: {status}");
            return;
        }
        if t0.elapsed() > CHILD_TIMEOUT {
            let _ = child.kill();
            let _ = child.wait();
            panic!("{case}: HANG (child still running after {CHILD_TIMEOUT:?})");
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

/// Child side of [`run_child`]; a no-op unless started by it.
#[test]
fn child_hang_case() {
    let Ok(case) = std::env::var("S3DLIO_300_CHILD_CASE") else {
        return;
    };
    let workers: usize = case.parse().expect("worker count");
    MARKED.store(false, Ordering::SeqCst);
    let rt = mt(workers);
    let errs = rt.block_on(async move {
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
    });
    assert_eq!(errs, workers, "every stat must return the 404 as an Err");
    // The stats really reached the endpoint that lives on the caller's own runtime.
    assert!(
        SERVED.load(Ordering::SeqCst) >= workers,
        "the fake endpoint served {} requests",
        SERVED.load(Ordering::SeqCst)
    );
}

/// The hang the first prototype introduced: an UNMARKED multi-thread runtime (a plain
/// `#[tokio::main]`, `#[tokio::test(flavor = "multi_thread")]`) that serves the endpoint itself,
/// with as many concurrent sync-wrapper callers as it has workers. Without a hand-off every worker
/// parks and nobody serves the listener.
#[test]
fn unmarked_runtime_serving_its_own_endpoint_does_not_hang() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    for workers in ["1", "2", "4"] {
        run_child(workers);
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
    // The name check is strict (equality, not `contains`): a name that merely contains the default
    // takes rule 4, for a worker thread and for a LocalSet on a hand-named thread.
    assert_eq!(
        delta(false, || {
            let mut b = tokio::runtime::Builder::new_multi_thread();
            b.worker_threads(2)
                .enable_all()
                .thread_name(format!("my-{}-1", default_name()));
            b.build()
                .unwrap()
                .block_on(async { tokio::spawn(async { call() }).await.unwrap() })
        }),
        0,
        "worker whose name only contains the default name"
    );
    assert_eq!(
        delta(false, || on_named_thread(
            format!("my-{}-1", default_name()),
            || {
                let l = tokio::task::LocalSet::new();
                mt(2).block_on(
                    l.run_until(async {
                        tokio::task::spawn_local(async { call() }).await.unwrap()
                    }),
                )
            }
        )),
        0,
        "LocalSet + spawn_local on a thread whose name only contains the default name"
    );
    // The try_id guard: a LocalSet MAIN future on a thread hand-named like the default worker has
    // no task id, so it waits plainly (no hand-off, and no panic).
    assert_eq!(
        delta(false, || on_named_thread(default_name(), || {
            let l = tokio::task::LocalSet::new();
            mt(2).block_on(l.run_until(async { call() }))
        })),
        0,
        "LocalSet main future on a default-named std thread"
    );
    // No runtime, current-thread runtime: none.
    assert_eq!(delta(true, call), 0, "no runtime");
    assert_eq!(
        delta(true, || ct().block_on(async { call() })),
        0,
        "current-thread"
    );
}
