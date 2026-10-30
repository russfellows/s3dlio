// src/fork_guard.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! Fork safety for process-global state (issue #297).
//!
//! `fork()` copies only the calling thread. A tokio runtime, a connection pool whose connection
//! tasks run on that runtime, a rayon pool, a helper thread: all of them exist in the child as
//! memory without the threads that made them work, so the first call that waits for one of them
//! hangs for ever. This module gives the crate ONE place that knows about forks.
//!
//! # The rules
//!
//! 1. [`register`] installs a `pthread_atfork` CHILD handler. The handler does exactly one
//!    thing: an atomic increment of the fork epoch. It is async-signal-safe (no allocation, no
//!    lock), so it is safe in a child of a multi-threaded process and between `fork` and `exec`.
//!    It is registered at import and again, idempotently, by every [`ForkCell`] and
//!    [`Guarded`] creation, so a fork after the first use of s3dlio is always seen.
//! 2. **Process-global state** lives in a [`ForkCell`]: a lazily built value that remembers the
//!    epoch it was built in. A later `get_or_init` in another epoch (a fork child) never returns
//!    the inherited value: the inherited value is LEAKED (never dropped, never used) and a fresh
//!    one is built, lazily, on the calling thread, outside the handler. The first time this
//!    abandons something, the process prints ONE warning line to stderr (silence it with
//!    `S3DLIO_FORK_WARN=0`). No inherited lock is ever taken: the cells are replaced, not
//!    locked, so a lock another parent thread held at the instant of the fork cannot wedge the
//!    child.
//! 3. **Objects** that own a runtime and/or connections (a checkpoint store, a writer, a
//!    loader, a multi-endpoint store) hold their state in a [`Guarded`]. It cannot be rebuilt
//!    transparently (the object is the owner of the state), so in a fork child every use is
//!    refused with a [`ForkedObject`] error (a catchable `RuntimeError` in Python) and dropping
//!    the object (`del`, garbage collection, interpreter exit) LEAKS the state instead of
//!    running its destructor: a destructor could join threads that do not exist (a runtime) or,
//!    worse, send a message the parent's operation depends on (abort a multipart upload that
//!    the parent is still writing).
//!
//! The parent is never affected: a child only builds new state or touches its own copy.
//!
//! # The lock rule
//!
//! A lock that a parent thread holds at the instant of the fork stays locked in the child. So:
//! every process-global lock of s3dlio is inside a [`ForkCell`] (a child never touches the
//! inherited one), EXCEPT `OP_LOG_STATE` (and the client-id lock), which the #294 `pthread_atfork`
//! prepare/parent/child handlers hold across the fork; those handlers are registered by
//! `init_op_logger` AND by `global_logger()` (taken on every store lookup), so they exist in any
//! process that did s3dlio I/O.
//!
//! pyo3's own reference-pool mutex is taken off the GIL only by a `Py<..>` dropped on a thread
//! without the GIL; s3dlio never does that (see `put_payload.rs`, `async_bridge.rs`), so no
//! thread can hold it while the forking thread (which holds the GIL) forks.
//!
//! What remains (third party or one-time, none observed as a hang): std's environment lock (only
//! `configure_s3` calls `set_var`, with the GIL held), `OnceLock`/`Lazy` initialisation in
//! progress at the fork, dashmap's parking_lot table, tracing's callsite registry.

use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};

/// Number of fork CHILDREN this process descends from (0 in a process that was never forked).
/// Incremented by the child handler, so it differs between a parent and its child and between
/// a child and a grandchild.
static EPOCH: AtomicU64 = AtomicU64::new(0);
static REGISTERED: AtomicBool = AtomicBool::new(false);
/// `epoch + 1` of the last epoch that printed the warning (0 = never).
static WARNED: AtomicU64 = AtomicU64::new(0);

/// The fork epoch of this process: the number of forks this process descends from.
pub fn epoch() -> u64 {
    EPOCH.load(Ordering::Acquire)
}

extern "C" fn child() {
    // Async-signal-safe: one atomic.
    EPOCH.fetch_add(1, Ordering::AcqRel);
}

/// Install the child handler (once per process; idempotent). Called at import, and by every
/// cell and guard creation.
///
/// An `AtomicBool`, not a `Once`: a fork that lands while another thread is inside the very
/// first registration would leave a `Once` running in the child for ever. Here the worst case
/// is a double registration (the child registers again), which only increments the epoch twice
/// per fork; nothing compares epochs for anything but inequality.
#[cfg(unix)]
pub fn register() {
    if REGISTERED.swap(true, Ordering::AcqRel) {
        return;
    }
    // SAFETY: `child` is an `extern "C" fn()` that lives for the whole process and only does
    // an atomic operation.
    let rc = unsafe { libc::pthread_atfork(None, None, Some(child)) };
    if rc != 0 {
        tracing::warn!("pthread_atfork failed ({rc}): s3dlio is not fork-safe in this process");
    }
}
#[cfg(not(unix))]
pub fn register() {
    REGISTERED.store(true, Ordering::Release);
}

/// `true` unless `S3DLIO_FORK_WARN` is `0`/`false`/`no`/`off`.
///
/// Read with `libc::getenv`, not `std::env::var`: std keeps a process-wide lock around the
/// environment, and a parent thread that was inside `std::env::set_var` at the fork leaves that
/// lock held for ever in the child. This is the first thing a child does after it notices the
/// fork, so it must not be able to hang (found by a stress run of the unit tests, where other
/// tests call `set_var` all the time). A fork child has one thread, so nothing races the read.
#[cfg(unix)]
fn warning_enabled() -> bool {
    // SAFETY: a NUL-terminated name; the returned pointer is read at once.
    let p = unsafe { libc::getenv(c"S3DLIO_FORK_WARN".as_ptr()) };
    if p.is_null() {
        return true;
    }
    // SAFETY: `getenv` returned a NUL-terminated string.
    warn_value_enabled(unsafe { std::ffi::CStr::from_ptr(p) }.to_bytes())
}

/// The value of `S3DLIO_FORK_WARN` says "warn" unless it is `0`/`false`/`no`/`off`.
#[cfg(unix)]
fn warn_value_enabled(v: &[u8]) -> bool {
    let v = v.trim_ascii();
    !(v == b"0"
        || v.eq_ignore_ascii_case(b"false")
        || v.eq_ignore_ascii_case(b"no")
        || v.eq_ignore_ascii_case(b"off"))
}
#[cfg(not(unix))]
fn warning_enabled() -> bool {
    !matches!(
        std::env::var("S3DLIO_FORK_WARN")
            .ok()
            .as_deref()
            .map(str::trim)
            .map(str::to_ascii_lowercase)
            .as_deref(),
        Some("0" | "false" | "no" | "off")
    )
}

const WARNING: &str = "s3dlio: this process was forked after s3dlio had already been used; the \
inherited runtime, thread pools and connections are abandoned and fresh ones are built here \
(create objects such as checkpoint stores and writers in the child; set S3DLIO_FORK_WARN=0 to \
silence this line; the 'spawn' or 'forkserver' start method avoids it)\n";

/// Print the abandonment warning once per process (a grandchild is another process and warns
/// again). Written with a raw `write(2)`: `eprintln!` takes the stderr lock, which another
/// parent thread may have held at the fork.
pub(crate) fn warn_abandoned() {
    let want = epoch() + 1;
    if WARNED.swap(want, Ordering::AcqRel) == want || !warning_enabled() {
        return;
    }
    #[cfg(unix)]
    {
        // SAFETY: a plain write of a static buffer to fd 2; a failure is ignored.
        let _ = unsafe { libc::write(2, WARNING.as_ptr().cast(), WARNING.len()) };
    }
    #[cfg(not(unix))]
    eprint!("{WARNING}");
}

/// Test hook: how many warnings this process would still print for its epoch.
#[cfg(test)]
fn forget_warning() {
    WARNED.store(0, Ordering::Release);
}

// ---------------------------------------------------------------------------------------------
// ForkCell: process-global state, rebuilt in a fork child
// ---------------------------------------------------------------------------------------------

struct Slot<T: 'static> {
    epoch: u64,
    value: T,
}

/// Identity of the calling thread (non-zero, unique within a process, copied into a fork child
/// for the thread that forked). Only used to tell "I am the one building" from "another thread
/// is building".
fn thread_token() -> u64 {
    use std::cell::Cell;
    static NEXT: AtomicU64 = AtomicU64::new(1);
    thread_local! {
        static TOKEN: Cell<u64> = const { Cell::new(0) };
    }
    TOKEN.with(|t| {
        if t.get() == 0 {
            t.set(NEXT.fetch_add(1, Ordering::Relaxed));
        }
        t.get()
    })
}

/// Identity of this process in this epoch (never 0): a claim by any other value belongs to an
/// ancestor whose thread does not exist here.
fn process_token() -> u64 {
    (u64::from(std::process::id()) << 32) | (epoch() & 0xffff_ffff) | (1 << 31)
}

/// A process-global, lazily built value that is rebuilt (not reused) in a fork child.
///
/// Values are leaked: a `ForkCell` never drops what it holds (it lives in a `static`, and an
/// inherited value must not be dropped in the child), so `get_or_init` can hand out
/// `&'static T`.
///
/// If `init` panics the claim is released (the next caller runs `init` again); if `init`
/// re-enters the same cell on the same thread, that is a bug and panics with a clear message
/// instead of waiting for itself.
pub struct ForkCell<T: 'static> {
    slot: AtomicPtr<Slot<T>>,
    /// 0 = nobody is building; else the [`process_token`] of the process whose thread is.
    claim: AtomicU64,
    /// With `claim`: the [`thread_token`] of the building thread (0 when none).
    claim_thread: AtomicU64,
    /// Makes the cell `Sync` only for a `T` that is: a `&'static T` is handed to every thread.
    _marker: std::marker::PhantomData<T>,
}

/// Releases the claim when the builder leaves `init` in any way, including by unwinding.
struct ClaimGuard<'a> {
    claim: &'a AtomicU64,
    claim_thread: &'a AtomicU64,
}

impl Drop for ClaimGuard<'_> {
    fn drop(&mut self) {
        self.claim_thread.store(0, Ordering::Release);
        self.claim.store(0, Ordering::Release);
    }
}

impl<T: 'static> ForkCell<T> {
    pub const fn new() -> Self {
        Self {
            slot: AtomicPtr::new(std::ptr::null_mut()),
            claim: AtomicU64::new(0),
            claim_thread: AtomicU64::new(0),
            _marker: std::marker::PhantomData,
        }
    }

    /// The value, if one was built in THIS process (never an inherited one).
    pub fn get(&self) -> Option<&'static T> {
        let p = self.slot.load(Ordering::Acquire);
        // SAFETY: slots are leaked `Box`es, immutable once published.
        let s = unsafe { p.as_ref() }?;
        (s.epoch == EPOCH.load(Ordering::Acquire)).then_some(&s.value)
    }

    /// `true` if the cell holds a value that this process inherited and may not use.
    pub fn holds_inherited(&self) -> bool {
        let p = self.slot.load(Ordering::Acquire);
        // SAFETY: as in `get`.
        unsafe { p.as_ref() }.is_some_and(|s| s.epoch != EPOCH.load(Ordering::Acquire))
    }

    /// The value of this process, building it with `init` on first use here. In a fork child
    /// the inherited value is abandoned (leaked, with a one-time warning) and `init` runs again.
    pub fn get_or_init(&self, init: impl FnOnce() -> T) -> &'static T {
        match self.get() {
            Some(v) => v,
            None => self.init_slow(init, true),
        }
    }

    /// [`get_or_init`](Self::get_or_init) without the one-time abandonment warning: for state
    /// that only an exit path touches (a child that never used s3dlio must not print a warning
    /// when it merely exits).
    pub fn get_or_init_silent(&self, init: impl FnOnce() -> T) -> &'static T {
        match self.get() {
            Some(v) => v,
            None => self.init_slow(init, false),
        }
    }

    #[cold]
    fn init_slow(&self, init: impl FnOnce() -> T, warn: bool) -> &'static T {
        register();
        let mine = process_token();
        let me = thread_token();
        let (mut spins, mut nap) = (0u32, 50u64);
        loop {
            if let Some(v) = self.get() {
                return v;
            }
            let c = self.claim.load(Ordering::Acquire);
            if c == mine {
                if self.claim_thread.load(Ordering::Acquire) == me {
                    panic!(
                        "s3dlio: a process-global initialiser re-entered itself on the same \
                         thread (recursive ForkCell::get_or_init); this is a bug in s3dlio"
                    );
                }
                // Another thread of THIS process is building: wait for it.
                if spins < 64 {
                    spins += 1;
                    std::thread::yield_now();
                } else {
                    std::thread::sleep(std::time::Duration::from_micros(nap));
                    nap = (nap * 2).min(1000);
                }
                continue;
            }
            // Free, or claimed by an ancestor's thread that does not exist here: take it over.
            if self
                .claim
                .compare_exchange(c, mine, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                continue;
            }
            self.claim_thread.store(me, Ordering::Release);
            // From here on the claim is released on every exit, panics included.
            let _release = ClaimGuard {
                claim: &self.claim,
                claim_thread: &self.claim_thread,
            };
            // The claim was won after a previous builder may have finished: look again.
            if let Some(v) = self.get() {
                return v;
            }
            if warn && self.holds_inherited() {
                warn_abandoned();
            }
            let epoch = EPOCH.load(Ordering::Acquire);
            let value = init();
            let slot = Box::into_raw(Box::new(Slot { epoch, value }));
            // The previous slot (inherited, or none) is leaked on purpose.
            self.slot.store(slot, Ordering::Release);
            // SAFETY: just leaked.
            return unsafe { &(*slot).value };
        }
    }
}

impl<T: 'static> Default for ForkCell<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------------------------
// Guarded: state owned by an OBJECT
// ---------------------------------------------------------------------------------------------

/// The error of using an object in a fork child that was created before the fork.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForkedObject;

impl std::fmt::Display for ForkedObject {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            "this s3dlio object was created before fork(): its runtime, threads and \
             connections belong to the parent process and cannot be used in the forked child; \
             create it in the child process",
        )
    }
}

impl std::error::Error for ForkedObject {}

#[cfg(feature = "extension-module")]
impl From<ForkedObject> for pyo3::PyErr {
    fn from(e: ForkedObject) -> Self {
        pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
    }
}

/// The epoch an object was created in: a lighter form of [`Guarded`] for objects whose state
/// is spread over many fields. `check()` at the top of every method that does work;
/// [`ForkStamp::leak_on_drop`] in the object's `Drop` for fields that must not be dropped.
#[derive(Clone, Copy, Debug)]
pub struct ForkStamp(u64);

impl ForkStamp {
    /// A stamp for an object created now.
    pub fn new() -> Self {
        register();
        Self(epoch())
    }

    /// `true` in a fork child of the process that created the object.
    pub fn is_inherited(self) -> bool {
        self.0 != epoch()
    }

    /// `Err` in a fork child of the process that created the object.
    pub fn check(self) -> Result<(), ForkedObject> {
        if self.is_inherited() {
            Err(ForkedObject)
        } else {
            Ok(())
        }
    }

    /// For an object's `Drop`: if the object is inherited, `forget` the value (it is never
    /// dropped in a fork child); otherwise drop it normally.
    pub fn leak_on_drop<T>(self, value: T) {
        if self.is_inherited() {
            std::mem::forget(value);
        } else {
            drop(value);
        }
    }
}

impl Default for ForkStamp {
    fn default() -> Self {
        Self::new()
    }
}

/// State owned by an object that a fork child inherits but cannot use (it owns a runtime or
/// live connections). Every access goes through [`Guarded::get`]/[`Guarded::get_mut`], which
/// refuse in a child; dropping a guard that was inherited leaks the value (no destructor runs:
/// see the module documentation).
pub struct Guarded<T> {
    /// The epoch the value was created in.
    epoch: u64,
    value: ManuallyDrop<T>,
}

impl<T> Guarded<T> {
    pub fn new(value: T) -> Self {
        register();
        Self {
            epoch: epoch(),
            value: ManuallyDrop::new(value),
        }
    }

    /// `true` in a fork child of the process that created the value.
    pub fn is_inherited(&self) -> bool {
        self.epoch != epoch()
    }

    /// The value; an error in a fork child of the process that created it.
    pub fn get(&self) -> Result<&T, ForkedObject> {
        if self.is_inherited() {
            return Err(ForkedObject);
        }
        Ok(&self.value)
    }

    /// The value, mutably; an error in a fork child of the process that created it.
    pub fn get_mut(&mut self) -> Result<&mut T, ForkedObject> {
        if self.is_inherited() {
            return Err(ForkedObject);
        }
        Ok(&mut self.value)
    }
}

impl<T> Guarded<T> {
    /// Take the value out; an error (the value is leaked, not dropped) in a fork child of the
    /// process that created it.
    pub fn into_inner(self) -> Result<T, ForkedObject> {
        let mut me = ManuallyDrop::new(self);
        if me.is_inherited() {
            return Err(ForkedObject);
        }
        // SAFETY: `me` is never dropped or used again, so the value is taken exactly once.
        Ok(unsafe { ManuallyDrop::take(&mut me.value) })
    }
}

impl<T> Drop for Guarded<T> {
    fn drop(&mut self) {
        if !self.is_inherited() {
            // SAFETY: dropped exactly once, here; never used afterwards.
            unsafe { ManuallyDrop::drop(&mut self.value) };
        }
        // An inherited value is leaked on purpose.
    }
}

impl<T: Clone> Clone for Guarded<T> {
    /// A clone keeps the epoch of its source: a copy of an inherited value stays inherited.
    fn clone(&self) -> Self {
        Self {
            epoch: self.epoch,
            value: ManuallyDrop::new(T::clone(&self.value)),
        }
    }
}

impl<T> std::fmt::Debug for Guarded<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Guarded")
            .field("inherited", &self.is_inherited())
            .finish()
    }
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// A claim left by an ancestor's thread (the fork landed while it was inside `init`) is
    /// taken over by the child instead of waited for.
    #[test]
    fn a_claim_of_an_ancestor_is_taken_over() {
        static CELL: ForkCell<u32> = ForkCell::new();
        // Pretend another process' thread is building: a claim by a different process token.
        CELL.claim
            .store(process_token() ^ (1 << 40), Ordering::Release);
        CELL.claim_thread
            .store(thread_token() + 1000, Ordering::Release);
        let v = *CELL.get_or_init(|| 7);
        assert_eq!(v, 7);
        assert_eq!(CELL.claim.load(Ordering::Acquire), 0, "claim released");
    }

    /// RED before the fix: a panic in `init` kept the claim for ever and every later caller
    /// spun.
    #[test]
    fn a_panic_in_init_releases_the_claim() {
        static CELL: ForkCell<u32> = ForkCell::new();
        let r = std::panic::catch_unwind(|| CELL.get_or_init(|| panic!("init failed")));
        assert!(r.is_err());
        assert_eq!(
            CELL.claim.load(Ordering::Acquire),
            0,
            "claim must be released"
        );
        // A later caller (same thread and another) builds normally.
        assert_eq!(*CELL.get_or_init(|| 3), 3);
        let other = std::thread::spawn(|| *CELL.get_or_init(|| 4))
            .join()
            .unwrap();
        assert_eq!(other, 3);
    }

    /// Recursive initialisation on one thread is a clear panic, not an endless wait; the cell is
    /// usable afterwards.
    #[test]
    fn recursive_init_is_detected() {
        static CELL: ForkCell<u32> = ForkCell::new();
        let r = std::panic::catch_unwind(|| {
            CELL.get_or_init(|| *CELL.get_or_init(|| 1) + 1);
        });
        let payload = r.expect_err("must panic");
        let msg = payload
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .expect("a string payload");
        assert!(msg.contains("re-entered itself"), "{msg}");
        assert_eq!(CELL.claim.load(Ordering::Acquire), 0);
        assert_eq!(*CELL.get_or_init(|| 9), 9);
    }

    /// Several threads race to build: exactly one `init` runs.
    #[test]
    fn concurrent_first_use_builds_once() {
        static CELL: ForkCell<u32> = ForkCell::new();
        static BUILDS: AtomicUsize = AtomicUsize::new(0);
        let hs: Vec<_> = (0..16)
            .map(|_| {
                std::thread::spawn(|| {
                    *CELL.get_or_init(|| {
                        BUILDS.fetch_add(1, Ordering::SeqCst);
                        std::thread::sleep(std::time::Duration::from_millis(20));
                        11
                    })
                })
            })
            .collect();
        for h in hs {
            assert_eq!(h.join().unwrap(), 11);
        }
        assert_eq!(BUILDS.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn s3dlio_fork_warn_values() {
        for off in ["0", "false", "FALSE", " no ", "Off", "off\n"] {
            assert!(!warn_value_enabled(off.as_bytes()), "{off:?}");
        }
        for on in ["1", "true", "yes", "on", "", "anything"] {
            assert!(warn_value_enabled(on.as_bytes()), "{on:?}");
        }
    }

    /// The warning is printed once per process, and not at all with S3DLIO_FORK_WARN=0.
    #[test]
    fn the_warning_is_once_per_epoch() {
        forget_warning();
        let want = epoch() + 1;
        assert_ne!(WARNED.swap(want, Ordering::AcqRel), want);
        assert_eq!(
            WARNED.swap(want, Ordering::AcqRel),
            want,
            "second call: already warned"
        );
        forget_warning();
    }
}
