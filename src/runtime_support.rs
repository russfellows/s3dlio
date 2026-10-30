// src/runtime_support.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
//! Tokio runtime support: the blocking steps of the local file stores (`file://`, `direct://`)
//! (#295, #267, `blocking_io`) and the waits of `run_on_global_rt` and the Python store-cache lock
//! `lock_build` (#173, #300, `blocking_wait`). The first sections describe the file stores; the
//! second rule is under "The synchronous wrappers".
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
//! `blocking_io` (crate-internal) calls `block_in_place` ONLY when ALL of these hold:
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
//! # The synchronous wrappers: `blocking_wait` (#173, #300)
//!
//! `run_on_global_rt` (and every sync wrapper built on it, such as `s3_utils::stat_object_uri`) and
//! the Python store-cache lock wait for work that runs on ANOTHER runtime, so they use a second
//! helper, `blocking_wait`, with a rule that is deliberately wider than the one above. The
//! asymmetry: skipping the hand-off here is not only slower, it can HANG, because the awaited
//! work may depend on the caller's own runtime making progress (an in-process mock server on a
//! `#[tokio::main]` runtime, issue #173), so the hand-off is kept wherever it is valid. In order:
//!
//! 1. inside an [`inline_blocking_io`] scope: wait inline;
//! 2. the thread carries the worker marker and the current runtime is multi-thread: hand off
//!    (`block_in_place`), exactly as before;
//! 3. not marked, current runtime multi-thread, inside a task (`tokio::task::try_id()` is `Some`)
//!    and the thread bears tokio's DEFAULT worker thread name (measured once per process, it is
//!    `"tokio-rt-worker"` in tokio 1.52): hand off. That is a real worker of an embedder's runtime
//!    that did not opt in (`#[tokio::main]`, `#[tokio::test(flavor = "multi_thread")]`, axum);
//! 4. anything else: wait inline. That is no runtime, a current-thread runtime, a `LocalSet`
//!    driven by a thread of any other name (the main thread: `rt.block_on(local.run_until(..))`,
//!    `LocalSet::block_on`, and its `spawn_local` tasks), `block_on`'s main future, a thread of
//!    an unmarked runtime that has custom thread names. Safe because the awaited future runs on
//!    the global runtime, which is never the caller's runtime here (its threads are marked).
//!
//! No panic is provoked and none is caught, so a `panic = "abort"` build behaves the same.
//!
//! Residuals, precisely:
//!
//! * (a) an unmarked runtime with CUSTOM worker thread names (`Builder::thread_name`) gets NO
//!   hand-off (rule 4): it cannot panic, but every sync-wrapper call parks a worker for its whole
//!   duration (latency and capacity), and it HANGS if the awaited work needs that same runtime
//!   (an in-process server on it with every worker parked). Remedy, one
//!   line: `Builder::on_thread_start(s3dlio::runtime_support::mark_tokio_worker_thread)`, or use
//!   the async API.
//! * (b) PANIC, on ANY thread that carries the default tokio worker name or the s3dlio marker and
//!   drives a `LocalSet` with the call inside a task. In practice: a `spawn_blocking` thread that
//!   runs `handle.block_on(local.run_until(..))`, even WITHOUT `spawn_local` (the blocking task's
//!   id persists inside `Handle::block_on`, so `try_id()` is `Some` and the thread name is the
//!   worker name), and a thread that was hand-named like tokio's default and runs
//!   `rt.block_on(local.run_until(async { spawn_local(..) }))`. Tokio gives no public way to tell
//!   those from a worker. It is no worse than before #300 (it panicked there too). Remedy: wrap
//!   the closure in [`inline_blocking_io`]. (A plain `handle.block_on(future)` on such a thread is
//!   fine: tokio lets `block_in_place` run inline there; a `LocalSet` main future on a
//!   hand-named thread is fine too, `try_id()` is `None`.) Measured, and asserted so a change is
//!   noticed, by `tests/run_on_global_rt_contexts_300.rs`.
//! * `MultipartUploadSink::write_blocking`, `enqueue_part` and `finish_blocking` use
//!   `Sender::blocking_send`, which panics in ANY tokio context: a separate bug (#301), not
//!   fixed by this rule.
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
//! The marker follows the THREAD, not the runtime. A thread that carries it keeps it whatever it
//! does next: a marked `spawn_blocking` thread of runtime A that drives a `LocalSet` on ANY
//! runtime, even an unmarked runtime B (`b.block_on(local.run_until(read))`), also needs
//! [`inline_blocking_io`] (see `tests::a_marked_thread_driving_a_localset_on_another_runtime_needs_the_inline_scope`).
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
//! Without the opt-in the file stores still work in every context; they just run every blocking
//! step inline on the polling worker: `direct://` and `file://` reads (open, each chunk of at most
//! 16 MiB, finish), `file://` borrowed-slice writes, and the `direct://` writer's O_DIRECT write
//! chunks, per-chunk fsync (`sync_writes`), `sync_all()` and finalize (sync, buffered tail append,
//! fsync). That is every caller of the crate-internal `blocking_io`. Nothing hands the worker's
//! queue off first, so a neighbouring task can wait for up to one chunk. The trade-off, measured with
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
//!
//! The same trade-off for s3dlio's own runtimes as seen from Python (1 GiB `get_into_async`, warm
//! page cache, single reader; 3 repetitions, noisy, 2-socket box): `file://` 3.8-6.9 GiB/s with the
//! hand-off against 5.6-7.7 GiB/s inline, `direct://` 4.4-4.8 against 4.9-5.5; a neighbour stalls
//! about 2 ms with the hand-off against about 10 ms inline (4 readers on 4 workers). Both are far
//! above the allocating `get` (1.5-2.0 GiB/s).

use std::cell::Cell;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

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
/// `on_thread_start` also runs for the runtime's `spawn_blocking` threads. That is harmless
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

/// A multi-thread runtime builder whose threads carry the worker marker: the ONE way s3dlio's own
/// production code builds a multi-thread runtime (`tests::every_production_multi_thread_runtime_is_marked`
/// scans `src/` and fails on any other `new_multi_thread`). Add the usual `enable_all()`,
/// `worker_threads(..)`, `thread_name(..)` and `build()`; do not replace the `on_thread_start` hook
/// (the builder keeps one callback only), call [`mark_tokio_worker_thread`] from yours instead.
///
/// Public only because the `s3-cli` binary is a separate crate. Embedders can use it, or the one-line
/// `on_thread_start` shown in the module documentation; it is not semver-stable API.
#[doc(hidden)]
pub fn marked_multi_thread_builder() -> tokio::runtime::Builder {
    let mut b = tokio::runtime::Builder::new_multi_thread();
    b.on_thread_start(mark_tokio_worker_thread);
    b
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
/// Test instrumentation, not API: hidden, and NOT semver-stable (it may change or go away in any
/// release; the integration tests of this crate need it, nothing else should).
#[doc(hidden)]
pub fn handoff_count() -> u64 {
    HANDOFFS.load(Ordering::Relaxed)
}

/// True when [`blocking_io`] would hand the worker off here and now. Test instrumentation, not API:
/// hidden, and NOT semver-stable (see [`handoff_count`]).
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

/// The thread name tokio gives its worker AND blocking-pool threads when the builder is not given
/// one (`"tokio-rt-worker"` in tokio 1.52, `"tokio-runtime-worker"` in older releases). It is
/// measured, not hard-coded: a throw-away one-worker runtime reports the name of its worker thread.
/// Only a SUCCESSFUL probe is cached; after a failure (no thread available, ...) it is retried on
/// a later call, and the failure is logged once. `None` while no probe has succeeded: rule 3 of
/// [`blocking_wait`] is then off for this call (a plain wait, never a panic).
fn default_tokio_thread_name() -> Option<&'static str> {
    static NAME: OnceLock<String> = OnceLock::new();
    static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if let Some(name) = NAME.get() {
        return Some(name.as_str());
    }
    match probe_default_tokio_thread_name() {
        Some(name) => Some(NAME.get_or_init(|| name).as_str()),
        None => {
            if !WARNED.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    "s3dlio: could not measure tokio's default worker thread name; sync wrappers \
                     called from an unmarked runtime will wait without a worker hand-off until \
                     it succeeds (mark the runtime with mark_tokio_worker_thread)"
                );
            }
            None
        }
    }
}

fn probe_default_tokio_thread_name() -> Option<String> {
    // A separate thread: building and dropping a runtime is not allowed from async context.
    std::thread::Builder::new()
        .name("s3dlio-thread-name-probe".into())
        .spawn(|| {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .worker_threads(1)
                .build()
                .ok()?;
            let (tx, rx) = std::sync::mpsc::channel();
            rt.spawn(async move {
                let _ = tx.send(std::thread::current().name().map(str::to_owned));
            });
            let name = rx.recv().ok().flatten();
            rt.shutdown_background();
            name
        })
        .ok()?
        .join()
        .ok()?
}

/// Rule 3 of [`blocking_wait`]: not marked, but inside a task and on a thread that carries tokio's
/// default worker name.
fn looks_like_default_tokio_worker() -> bool {
    if tokio::task::try_id().is_none() {
        return false;
    }
    match (default_tokio_thread_name(), std::thread::current().name()) {
        (Some(default), Some(name)) => name == default,
        _ => false,
    }
}

/// True when [`blocking_wait`] would hand the worker off here and now (the decision rule in its
/// documentation).
pub(crate) fn would_hand_off_wait() -> bool {
    if INLINE_SCOPES.with(|d| d.get() != 0) {
        return false; // rule 1
    }
    if !matches!(
        tokio::runtime::Handle::try_current().map(|h| h.runtime_flavor()),
        Ok(tokio::runtime::RuntimeFlavor::MultiThread)
    ) {
        return false; // rule 4: no runtime, or a current-thread runtime
    }
    MARKED.with(|m| m.get()) // rule 2
        || looks_like_default_tokio_worker() // rule 3
}

/// Wait for a result produced by ANOTHER runtime (the global runtime) or by another thread, from
/// synchronous code that may be running on any thread: [`crate::s3_client::run_on_global_rt`]
/// and the store-cache `lock_build`. The wait blocks the thread, so where the thread is a worker of
/// a multi-thread runtime its core is handed off first (`tokio::task::block_in_place`), exactly as
/// #173 requires, and everywhere else the closure just runs. The decision, in order:
///
/// 1. inside an [`inline_blocking_io`] scope: run inline;
/// 2. the thread carries the worker marker and the current runtime is multi-thread: hand off;
/// 3. not marked, current runtime is multi-thread, inside a task (`tokio::task::try_id()` is
///    `Some`) AND the thread bears tokio's default worker thread name: hand off. On a real worker
///    that is valid. A `LocalSet` that the main thread (or any thread with another name) drives,
///    and its `spawn_local` tasks, fall through to rule 4;
/// 4. anything else (no runtime, current-thread runtime, a `LocalSet` on a non-worker thread,
///    `block_on`'s main future, a `spawn_blocking` thread, an unmarked runtime with custom
///    thread names): run inline.
///
/// Rule 4 is safe because the awaited future runs on the GLOBAL runtime, whose threads are marked
/// and which is never the caller's runtime here. Only thread-local state, the thread name and
/// tokio's public API are read: no panic is provoked (no `catch_unwind`, no panic hook), so the
/// behaviour is identical under `panic = "abort"`. See the module documentation for the residuals.
pub(crate) fn blocking_wait<R>(f: impl FnOnce() -> R) -> R {
    if would_hand_off_wait() {
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

    /// The rule of `blocking_wait`, one row per context (#300). Rows 3 and 4 are the point:
    /// an UNMARKED worker with the default thread name hands off, a LocalSet on the main thread
    /// and a custom-named worker do not.
    #[test]
    fn wait_rule_table() {
        // Rule 2: marked worker.
        let rt = marked_rt(2);
        assert!(rt.block_on(async { tokio::spawn(async { would_hand_off_wait() }).await.unwrap() }));
        // Rule 1: inline scope wins.
        assert!(!rt.block_on(async {
            tokio::spawn(async {
                let _g = inline_blocking_io();
                would_hand_off_wait()
            })
            .await
            .unwrap()
        }));
        // The main thread of block_on is no worker (no task id).
        assert!(!rt.block_on(async { would_hand_off_wait() }));
        drop(rt);
        // Rule 3: unmarked, default thread names, inside a task.
        let rt = plain_rt(2);
        assert!(rt.block_on(async { tokio::spawn(async { would_hand_off_wait() }).await.unwrap() }));
        // Rule 4: LocalSet on the main thread, main future and spawn_local task.
        let local = tokio::task::LocalSet::new();
        assert!(!rt.block_on(local.run_until(async { would_hand_off_wait() })));
        let local = tokio::task::LocalSet::new();
        assert!(!rt.block_on(local.run_until(async {
            tokio::task::spawn_local(async { would_hand_off_wait() })
                .await
                .unwrap()
        })));
        drop(rt);
        // Rule 4: unmarked runtime with custom thread names (residual a) ...
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("embedder-worker")
            .enable_all()
            .build()
            .unwrap();
        assert!(
            !rt.block_on(async { tokio::spawn(async { would_hand_off_wait() }).await.unwrap() })
        );
        drop(rt);
        // ... and Rule 2 again once it is marked (the remedy).
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("embedder-worker")
            .enable_all()
            .on_thread_start(mark_tokio_worker_thread)
            .build()
            .unwrap();
        assert!(rt.block_on(async { tokio::spawn(async { would_hand_off_wait() }).await.unwrap() }));
        drop(rt);
        // No runtime and a current-thread runtime: never, whatever the thread is called.
        assert!(!run_in_thread(would_hand_off_wait));
        let ct = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        assert!(!run_in_thread(move || {
            mark_tokio_worker_thread();
            ct.block_on(async { tokio::spawn(async { would_hand_off_wait() }).await.unwrap() })
        }));
        // The measured default name is a worker name, not empty.
        assert!(default_tokio_thread_name().is_some_and(|n| n.contains("worker")));
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

    /// The marker follows the THREAD, not the runtime: a marked `spawn_blocking` thread of runtime
    /// A that drives a `LocalSet` on ANY runtime, even an unmarked runtime B, reaches
    /// `block_in_place` and tokio panics there; `inline_blocking_io` avoids it.
    #[test]
    fn a_marked_thread_driving_a_localset_on_another_runtime_needs_the_inline_scope() {
        fn drive(b: tokio::runtime::Handle, scoped: bool) -> std::thread::Result<u32> {
            catch_unwind(AssertUnwindSafe(|| {
                let _inline = scoped.then(inline_blocking_io);
                let local = tokio::task::LocalSet::new();
                b.block_on(local.run_until(async {
                    // The thread is marked (by runtime A) and runtime B is multi-thread.
                    tokio::task::spawn_local(async { blocking_io(|| 41 + 1) })
                        .await
                        .unwrap()
                }))
            }))
        }
        let a = marked_rt(2);
        let b = plain_rt(2); // NOT marked
        let hb = b.handle().clone();
        let (h1, h2) = (hb.clone(), hb);
        let scoped = a.block_on(async move {
            tokio::task::spawn_blocking(move || drive(h1, true))
                .await
                .unwrap()
        });
        assert_eq!(scoped.expect("with the scope it runs inline"), 42);
        let unscoped = a.block_on(async move {
            tokio::task::spawn_blocking(move || drive(h2, false))
                .await
                .unwrap()
        });
        assert!(
            unscoped.is_err(),
            "tokio no longer panics in this case: the documentation (marker follows the thread) \
             and this test can be relaxed"
        );
    }

    /// The builder helper marks every thread of the runtime it builds.
    #[test]
    fn marked_builder_marks_workers() {
        let rt = marked_multi_thread_builder()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();
        assert!(rt.block_on(async { tokio::spawn(async { would_hand_off() }).await.unwrap() }));
    }

    /// Every production `new_multi_thread` in `src/` goes through `marked_multi_thread_builder`
    /// (so no s3dlio runtime can lose the marker unnoticed). Test-only modules are exempt, and so
    /// is this file (it defines the helper).
    #[test]
    fn every_production_multi_thread_runtime_is_marked() {
        fn walk(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            for e in std::fs::read_dir(dir).unwrap() {
                let p = e.unwrap().path();
                if p.is_dir() {
                    walk(&p, out);
                } else if p.extension().is_some_and(|x| x == "rs") {
                    out.push(p);
                }
            }
        }
        let mut files = Vec::new();
        walk(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src"),
            &mut files,
        );
        assert!(
            files.len() > 20,
            "scan found too few files: {}",
            files.len()
        );
        let mut offenders = Vec::new();
        for f in files {
            if f.ends_with("runtime_support.rs") {
                continue;
            }
            let text = std::fs::read_to_string(&f).unwrap();
            // Top-level `#[cfg(test)]` items (module or fn) are test code: from the attribute to
            // the closing `}` in column 0 (or the `;` ending a one-line item).
            let (mut pending, mut in_test) = (false, false);
            for (i, line) in text.lines().enumerate() {
                if in_test {
                    if line == "}" {
                        in_test = false;
                    }
                    continue;
                }
                if line.trim_end() == "#[cfg(test)]" || line.starts_with("#[cfg(all(test") {
                    pending = true;
                    continue;
                }
                if pending && !line.starts_with("#[") && !line.is_empty() {
                    pending = false;
                    if !line.trim_end().ends_with(';') {
                        in_test = line != "}";
                    }
                    continue;
                }
                if line.contains("new_multi_thread") && !line.trim_start().starts_with("//") {
                    offenders.push(format!("{}:{}", f.display(), i + 1));
                }
            }
        }
        assert!(
            offenders.is_empty(),
            "use runtime_support::marked_multi_thread_builder() instead of \
             Builder::new_multi_thread() in production code: {offenders:?}"
        );
    }
}
