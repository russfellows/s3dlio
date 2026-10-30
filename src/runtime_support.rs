// src/runtime_support.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
//! Tokio runtime support for the local file stores (`file://`, `direct://`) (#295, #267).
//!
//! # Why this exists
//!
//! The file stores read into, and write from, a BORROWED slice (Rule #1, zero copy), so their
//! blocking system calls cannot be moved to `spawn_blocking` (the future could be dropped while
//! a thread still touched the slice). They run inside the poll of the returned future instead,
//! one bounded chunk (at most 16 MiB) at a time.
//!
//! A blocking call inside a poll stalls the runtime worker that polls it, and so every task
//! queued behind it. Tokio's answer is [`tokio::task::block_in_place`]: the worker hands its
//! queue to another thread first. Two properties of that call matter here (tokio 1.52,
//! `runtime/scheduler/multi_thread/worker.rs`, `block_in_place`):
//!
//! * it PANICS ("can call blocking only when running on the multi-threaded runtime") when a
//!   [`tokio::task::LocalSet`] is being driven by a thread that is not a runtime worker:
//!   `LocalSet::run_until`/`block_on` disable it (`task/local.rs`, `disallow_block_in_place`)
//!   and a non-worker thread has no worker context to override that. The usual spelling is
//!   `rt.block_on(local.run_until(..))` on the main thread. Tokio has no public query for the
//!   state, and it cannot be probed with `catch_unwind` without a process-wide panic hook
//!   (which every later hook of the application would see), an abort under `panic = "abort"`
//!   and a swallowed genuine panic, so this crate does not try;
//! * everywhere else that is not a worker (no runtime, `Handle::block_on`, a `spawn_blocking`
//!   thread) it only runs the closure inline.
//!
//! # The rule
//!
//! [`blocking_io`] (crate-internal) calls `block_in_place` ONLY when ALL of these hold:
//!
//! 1. the current thread carries the worker marker ([`mark_tokio_worker_thread`]);
//! 2. the current runtime is a multi-thread runtime;
//! 3. the thread is not inside an [`inline_blocking_io`] scope.
//!
//! In every other case the closure runs inline on the calling thread. The decision uses only
//! thread-local state and tokio's public API: no panic is provoked, so no panic hook is
//! installed or consulted, `catch_unwind` is not used, and the crate behaves the same under
//! `panic = "abort"`. A panic raised by the closure itself propagates unchanged.
//!
//! Why this is safe: the one situation that panics is a `LocalSet` driven by a non-worker
//! thread, and the marker is never set on the threads that usually drive one (the application's
//! main thread, a `LocalSet`-per-thread runtime, a current-thread runtime).
//!
//! # Which threads are marked
//!
//! Every multi-thread runtime that s3dlio builds (the global runtime, the Python decode and
//! checkpoint runtimes, the CLI) marks its threads with `Builder::on_thread_start`.
//!
//! Why `on_thread_start` and not the park hooks: tokio's workers are threads of the runtime's
//! blocking pool (`blocking/pool.rs`, `Inner::run`: `after_start` runs once when the pool thread
//! starts, whatever it goes on to do), and when a worker calls `block_in_place` its core moves
//! to ANOTHER pool thread (`spawn_blocking(run(worker))`, `multi_thread/worker.rs`). The park
//! hooks (`on_thread_park` / `on_thread_unpark`) fire only from a worker's own park path, so a
//! replacement worker that stays busy never runs them and would lose the marker, silently
//! turning off the hand-off exactly when the pool is saturated. `on_thread_start` marks every
//! thread a worker can ever be. (There is no per-role hook: a thread is a `spawn_blocking`
//! thread and a worker thread at different times, and the names are shared.)
//!
//! The price: `spawn_blocking` threads of the same runtime carry the marker too. That is
//! harmless for every closure except one: a `spawn_blocking` closure that drives a
//! [`tokio::task::LocalSet`] itself (`handle.block_on(local.run_until(..))`) and runs s3dlio
//! file I/O inside it would reach `block_in_place` on a thread that is not a worker. Tokio gives
//! no way to tell, so such a closure must wrap its body in [`inline_blocking_io`]. (A plain
//! `Handle::block_on` of an ordinary future from a `spawn_blocking` thread is fine: tokio allows
//! `block_in_place` there and it runs inline.)
//!
//! # Embedders
//!
//! A program that owns its multi-thread runtime can opt in with one line:
//!
//! ```no_run
//! let rt = tokio::runtime::Builder::new_multi_thread()
//!     .enable_all()
//!     .on_thread_start(s3dlio::runtime_support::mark_tokio_worker_thread)
//!     .build()
//!     .unwrap();
//! # drop(rt);
//! ```
//!
//! Without the opt-in the file stores still work in every context; they just run each read
//! chunk (at most 16 MiB) inline on the polling worker instead of handing the worker's queue
//! off first, so a neighbouring task can wait for up to one chunk. The trade-off, measured with
//! `examples/runtime_handoff_bench.rs` (release build, 96-core Linux box, 4 workers, 1 GiB
//! file in the warm page cache on ext4, `get_into` into a reused buffer):
//!
//! | | single reader | neighbour wake-up lateness with 4 readers on the 4 workers |
//! |---|---|---|
//! | hand-off (marked) | 3.3 GiB/s | p99 1.8 ms, max 2.1 ms |
//! | inline (unmarked) | 7.4 GiB/s | p99 9.8 ms, max 10.8 ms |
//!
//! (On tmpfs: 2.4 against 4.4 GiB/s, neighbour max 2.0 against 14.9 ms. A heavier, fully
//! saturated pool measured by the reviewer: about 3 ms against 60 ms.) The hand-off costs about
//! 2.6 ms per 16 MiB chunk here, so a single reader is about twice as fast inline; s3dlio's own
//! runtimes keep the hand-off because neighbour latency matters more than one reader's peak.

use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};

thread_local! {
    /// This thread belongs to a multi-thread runtime that was built with the worker marker.
    static MARKED: Cell<bool> = const { Cell::new(false) };
    /// Depth of [`inline_blocking_io`] scopes on this thread.
    static INLINE_SCOPES: Cell<u32> = const { Cell::new(0) };
}

/// Calls to `block_in_place` made by [`blocking_io`] (test instrumentation).
static HANDOFFS: AtomicU64 = AtomicU64::new(0);

/// Mark the current thread as a worker (or worker candidate) of a tokio multi-thread runtime, so
/// that s3dlio's `file://` and `direct://` stores hand the worker's queue off
/// (`tokio::task::block_in_place`) around each blocking read or write chunk instead of
/// stalling the tasks queued behind it.
///
/// It is meant for [`tokio::runtime::Builder::on_thread_start`] of a multi-thread runtime that
/// the embedding program builds itself. It is idempotent and cheap, and safe to call from a
/// larger `on_thread_start` closure of your own (the builder keeps one callback only).
/// s3dlio's own runtimes already do this.
///
/// `on_thread_start` also runs for the runtime\'s `spawn_blocking` threads. That is harmless
/// except for a `spawn_blocking` closure that drives a `LocalSet` itself: see the module
/// documentation and [`inline_blocking_io`].
///
/// ```no_run
/// let rt = tokio::runtime::Builder::new_multi_thread()
///     .enable_all()
///     .on_thread_start(s3dlio::runtime_support::mark_tokio_worker_thread)
///     .build()
///     .unwrap();
/// # drop(rt);
/// ```
pub fn mark_tokio_worker_thread() {
    MARKED.with(|m| m.set(true));
}

/// Guard returned by [`inline_blocking_io`]. Not `Send`: it belongs to the thread that made it.
#[must_use = "the scope ends when the guard is dropped"]
pub struct InlineBlockingIo {
    _not_send: PhantomData<*const ()>,
}

impl Drop for InlineBlockingIo {
    fn drop(&mut self) {
        INLINE_SCOPES.with(|d| d.set(d.get().saturating_sub(1)));
    }
}

/// Make s3dlio's file stores run their blocking chunks inline on this thread (never calling
/// `tokio::task::block_in_place`) until the returned guard is dropped.
///
/// Use it at the top of a `spawn_blocking` closure (on a runtime whose threads carry the worker
/// marker) that drives a `tokio::task::LocalSet` itself, for example
/// `handle.block_on(local.run_until(..))`: `block_in_place` panics there and tokio offers no way
/// to ask beforehand.
///
/// ```no_run
/// # let rt = tokio::runtime::Runtime::new().unwrap();
/// let handle = rt.handle().clone();
/// let join = rt.spawn_blocking(move || {
///     let _inline = s3dlio::runtime_support::inline_blocking_io();
///     let local = tokio::task::LocalSet::new();
///     handle.block_on(local.run_until(async { /* spawn_local + s3dlio file I/O */ }));
/// });
/// # drop(join);
/// ```
pub fn inline_blocking_io() -> InlineBlockingIo {
    INLINE_SCOPES.with(|d| d.set(d.get() + 1));
    InlineBlockingIo {
        _not_send: PhantomData,
    }
}

/// How many times [`blocking_io`] handed a worker off with `block_in_place` in this process.
/// Test instrumentation, not API.
#[doc(hidden)]
pub fn handoff_count() -> u64 {
    HANDOFFS.load(Ordering::Relaxed)
}

/// True when [`blocking_io`] would hand the worker off here and now (test instrumentation).
#[doc(hidden)]
pub fn would_hand_off() -> bool {
    MARKED.with(|m| m.get())
        && INLINE_SCOPES.with(|d| d.get() == 0)
        && matches!(
            tokio::runtime::Handle::try_current().map(|h| h.runtime_flavor()),
            Ok(tokio::runtime::RuntimeFlavor::MultiThread)
        )
}

/// Run blocking file I/O from async code without stalling a runtime worker when that is safe,
/// inline otherwise. See the module documentation for the rule and why it is robust. A panic
/// raised by `f` propagates unchanged on every path.
pub(crate) fn blocking_io<R>(f: impl FnOnce() -> R) -> R {
    if would_hand_off() {
        HANDOFFS.fetch_add(1, Ordering::Relaxed);
        tokio::task::block_in_place(f)
    } else {
        f()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    fn marked_rt(workers: usize) -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(workers)
            .enable_all()
            .on_thread_start(mark_tokio_worker_thread)
            .build()
            .unwrap()
    }

    fn plain_rt(workers: usize) -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(workers)
            .enable_all()
            .build()
            .unwrap()
    }

    fn run_in_thread<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
        std::thread::spawn(f).join().expect("test thread panicked")
    }

    /// The rule, one row per context: the hand-off happens on a marked thread of a multi-thread
    /// runtime outside an inline scope, and nowhere else.
    #[test]
    fn handoff_rule_table() {
        // Marked worker, inside a spawned task: yes.
        let rt = marked_rt(2);
        assert!(rt.block_on(async { tokio::spawn(async { would_hand_off() }).await.unwrap() }));
        // ... but not inside an inline scope.
        assert!(!rt.block_on(async {
            tokio::spawn(async {
                let _g = inline_blocking_io();
                would_hand_off()
            })
            .await
            .unwrap()
        }));
        // ... and the scope ends with the guard.
        assert!(rt.block_on(async {
            tokio::spawn(async {
                drop(inline_blocking_io());
                would_hand_off()
            })
            .await
            .unwrap()
        }));
        // A marked pool thread (spawn_blocking) that drives an ordinary future with
        // Handle::block_on: tokio allows block_in_place there (it runs inline), so the helper
        // may call it, and it must not panic.
        let h = rt.handle().clone();
        rt.block_on(async move {
            tokio::task::spawn_blocking(move || {
                h.block_on(async {
                    assert!(would_hand_off());
                    assert_eq!(blocking_io(|| 7), 7);
                })
            })
            .await
            .unwrap()
        });
        // The main thread that drives the runtime is not marked: inline.
        assert!(!rt.block_on(async { would_hand_off() }));
        drop(rt);

        // A runtime that was not opted in: never.
        let rt = plain_rt(2);
        assert!(!rt.block_on(async { tokio::spawn(async { would_hand_off() }).await.unwrap() }));
        drop(rt);

        // Marked thread, current-thread runtime: never (block_in_place would panic there).
        let ct = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        assert!(!run_in_thread(move || {
            mark_tokio_worker_thread();
            ct.block_on(async { tokio::spawn(async { would_hand_off() }).await.unwrap() })
        }));
        // Marked thread, no runtime: never.
        assert!(!run_in_thread(|| {
            mark_tokio_worker_thread();
            would_hand_off()
        }));
    }

    /// A panic raised by the closure propagates unchanged on a worker (with the hand-off) and
    /// does not strand the runtime.
    #[test]
    fn a_panicking_closure_propagates_on_a_worker() {
        let rt = marked_rt(2);
        rt.block_on(async {
            for _ in 0..4 {
                let r = tokio::spawn(async {
                    assert!(would_hand_off());
                    catch_unwind(AssertUnwindSafe(|| {
                        blocking_io(|| -> u32 { panic!("closure-panic-on-worker") })
                    }))
                })
                .await
                .unwrap();
                let payload = r.expect_err("the panic must reach the caller");
                assert_eq!(
                    payload.downcast_ref::<&str>().copied(),
                    Some("closure-panic-on-worker")
                );
            }
            // Every worker still runs tasks (none was stranded by the unwinding hand-off).
            let done = Arc::new(AtomicUsize::new(0));
            let tasks: Vec<_> = (0..64)
                .map(|_| {
                    let done = Arc::clone(&done);
                    tokio::spawn(async move {
                        blocking_io(|| std::thread::sleep(Duration::from_millis(1)));
                        done.fetch_add(1, Ordering::SeqCst);
                    })
                })
                .collect();
            for t in tasks {
                t.await.unwrap();
            }
            assert_eq!(done.load(Ordering::SeqCst), 64);
        });
    }

    /// The same on every thread that runs inline: no runtime, a plain runtime, a marked
    /// current-thread runtime, a marked thread inside a `spawn_blocking` closure.
    #[test]
    fn a_panicking_closure_propagates_inline() {
        fn check() {
            let r = catch_unwind(AssertUnwindSafe(|| {
                blocking_io(|| -> u32 { panic!("closure-panic-inline") })
            }));
            let payload = r.expect_err("the panic must reach the caller");
            assert_eq!(
                payload.downcast_ref::<&str>().copied(),
                Some("closure-panic-inline")
            );
        }
        run_in_thread(check);
        let rt = plain_rt(2);
        rt.block_on(async { tokio::spawn(async { check() }).await.unwrap() });
        let ct = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        run_in_thread(move || {
            mark_tokio_worker_thread();
            ct.block_on(async { check() })
        });
        let rt = marked_rt(2);
        let h = rt.handle().clone();
        rt.block_on(async move {
            tokio::task::spawn_blocking(move || h.block_on(async { check() }))
                .await
                .unwrap()
        });
    }

    /// The point of the hand-off, observed: with ONE worker, another task makes progress while a
    /// long blocking chunk runs on a marked runtime, and does not on a plain one (the worker is
    /// blocked inline).
    #[test]
    fn handoff_keeps_the_neighbour_running_only_on_a_marked_runtime() {
        fn neighbour_ran_during(rt: &tokio::runtime::Runtime) -> bool {
            rt.block_on(async {
                let started = Arc::new(AtomicBool::new(false));
                let neighbour_ran = Arc::new(AtomicBool::new(false));
                let (s2, n2) = (Arc::clone(&started), Arc::clone(&neighbour_ran));
                // The reader parks the (only) worker inside blocking_io for 400 ms.
                let reader = tokio::spawn(async move {
                    blocking_io(|| {
                        s2.store(true, Ordering::SeqCst);
                        let end = Instant::now() + Duration::from_millis(400);
                        while Instant::now() < end {
                            if n2.load(Ordering::SeqCst) {
                                return true;
                            }
                            std::thread::sleep(Duration::from_millis(2));
                        }
                        n2.load(Ordering::SeqCst)
                    })
                });
                while !started.load(Ordering::SeqCst) {
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
                let n3 = Arc::clone(&neighbour_ran);
                // Spawned while the reader is inside the closure.
                let neighbour = tokio::spawn(async move { n3.store(true, Ordering::SeqCst) });
                let saw = reader.await.unwrap();
                neighbour.await.unwrap();
                saw
            })
        }
        assert!(
            neighbour_ran_during(&marked_rt(1)),
            "marked runtime: the worker's queue was not handed off"
        );
        assert!(
            !neighbour_ran_during(&plain_rt(1)),
            "plain runtime: the read should have run inline on the only worker"
        );
    }

    /// The one case the marker cannot see: a `spawn_blocking` closure (a marked, non-worker
    /// thread) that drives a `LocalSet`. Tokio\'s `block_in_place` panics there (that is WHY the
    /// scope exists; this test pins the behaviour the documentation relies on), and
    /// `inline_blocking_io` makes it work.
    #[test]
    fn a_localset_on_a_marked_pool_thread_needs_the_inline_scope() {
        fn drive(h: tokio::runtime::Handle, scoped: bool) -> std::thread::Result<u32> {
            catch_unwind(AssertUnwindSafe(|| {
                let _inline = scoped.then(inline_blocking_io);
                let local = tokio::task::LocalSet::new();
                h.block_on(local.run_until(async {
                    tokio::task::spawn_local(async { blocking_io(|| 41 + 1) })
                        .await
                        .unwrap()
                }))
            }))
        }
        let rt = marked_rt(2);
        let h = rt.handle().clone();
        let (h1, h2) = (h.clone(), h);
        let scoped = rt.block_on(async move {
            tokio::task::spawn_blocking(move || drive(h1, true))
                .await
                .unwrap()
        });
        assert_eq!(scoped.expect("with the scope it runs inline"), 42);
        let unscoped = rt.block_on(async move {
            tokio::task::spawn_blocking(move || drive(h2, false))
                .await
                .unwrap()
        });
        assert!(
            unscoped.is_err(),
            "tokio no longer panics in this case: the inline_blocking_io documentation (and \
             this test) can be relaxed"
        );
    }
}
