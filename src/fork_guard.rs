// src/fork_guard.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
//! PROTOTYPE for issue #297: what a process-global resource does when the process forks.
//!
//! `fork()` copies only the calling thread. A tokio runtime, a connection pool whose connection
//! tasks run on that runtime, a helper thread: all of them exist in the child as memory without
//! the threads that made them work. This module gives the crate ONE place that knows about forks:
//!
//! * [`register`] installs a `pthread_atfork` CHILD handler. The handler does exactly one thing,
//!   an atomic increment of [`EPOCH`] (plus one atomic flag): async-signal-safe, no allocation,
//!   no lock, so it is safe in a child of a multi-threaded process and between `fork` and `exec`.
//! * [`ForkCell`] is a lazily built process-global whose value remembers the epoch it was built
//!   in. A later `get_or_init` in another epoch (that is: in a fork child) never returns the
//!   inherited value. The inherited value is LEAKED (never dropped: dropping a runtime would
//!   join threads that do not exist) and a fresh one is built, lazily, on the calling thread.
//! * [`Policy`] chooses what a child does that finds inherited state: `Rebuild` (default in the
//!   prototype) builds fresh state; `Error` refuses with a clear error instead (`check`).
//!
//! The rebuild is done by the first caller in the child, outside the atfork handler, so it may
//! allocate, take locks and spawn threads freely. No lock of the parent is ever touched: the
//! inherited cells are replaced, not locked, so a lock another parent thread held at the instant
//! of the fork cannot wedge the child.

use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicU8, Ordering};

/// Number of fork CHILDREN this process descends from (0 in a process that was never forked).
/// Incremented by the child handler, so it differs between a parent and its child and between
/// a child and a grandchild.
static EPOCH: AtomicU64 = AtomicU64::new(0);
/// A [`ForkCell`] was initialised in this process (or inherited as such): s3dlio threads and
/// pools exist. Narrow "used" state for the fail-fast policy.
static USED: AtomicBool = AtomicBool::new(false);
/// Set by the child handler when [`USED`] was set at the fork: this process holds inherited
/// s3dlio state that cannot work.
static INHERITED_USED: AtomicBool = AtomicBool::new(false);
static REGISTERED: AtomicBool = AtomicBool::new(false);
static POLICY: AtomicU8 = AtomicU8::new(Policy::Rebuild as u8);

/// What a fork child does when it finds inherited s3dlio state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Policy {
    /// Leak the inherited state and build fresh state lazily (default).
    Rebuild = 0,
    /// Refuse every call that needs process-global state with a clear error.
    Error = 1,
}

pub(crate) fn policy() -> Policy {
    if POLICY.load(Ordering::Relaxed) == Policy::Error as u8 {
        Policy::Error
    } else {
        Policy::Rebuild
    }
}

/// Prototype switch: `S3DLIO_FORK_POLICY=error|rebuild` (read once, at [`register`]).
fn policy_from_env() -> Policy {
    match std::env::var("S3DLIO_FORK_POLICY").ok().as_deref() {
        Some("error") | Some("fail") => Policy::Error,
        _ => Policy::Rebuild,
    }
}

const MSG: &str = "s3dlio was used in the parent process before fork(): its runtime threads and \
connections do not exist in the forked child. Create the child with the 'spawn' or 'forkserver' \
start method (multiprocessing.set_start_method('spawn'); DataLoader(multiprocessing_context='spawn')), \
or fork before the first s3dlio call, or set S3DLIO_FORK_POLICY=rebuild";

/// `Err` in a fork child that inherited s3dlio state while the policy is `Error`.
pub(crate) fn check() -> anyhow::Result<()> {
    if policy() == Policy::Error && INHERITED_USED.load(Ordering::Acquire) {
        anyhow::bail!(MSG);
    }
    Ok(())
}

/// [`check`] as a Python exception (`RuntimeError`).
#[cfg(feature = "extension-module")]
pub(crate) fn check_py() -> pyo3::PyResult<()> {
    if policy() == Policy::Error && INHERITED_USED.load(Ordering::Acquire) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(MSG));
    }
    Ok(())
}

extern "C" fn child() {
    // Async-signal-safe: atomics only.
    EPOCH.fetch_add(1, Ordering::AcqRel);
    if USED.load(Ordering::Acquire) {
        INHERITED_USED.store(true, Ordering::Release);
    }
}

/// Install the child handler (once per process; idempotent). Called at import and by the first
/// [`ForkCell`] initialisation, so every fork after the first s3dlio use is seen.
#[cfg(unix)]
pub fn register() {
    if REGISTERED.swap(true, Ordering::AcqRel) {
        return;
    }
    POLICY.store(policy_from_env() as u8, Ordering::Relaxed);
    // SAFETY: `child` is an `extern "C" fn()` that lives for the whole process and only does
    // atomic operations.
    let rc = unsafe { libc::pthread_atfork(None, None, Some(child)) };
    if rc != 0 {
        tracing::warn!("pthread_atfork failed ({rc}): s3dlio is not fork-safe in this process");
    }
}
#[cfg(not(unix))]
pub fn register() {}

struct Slot<T: 'static> {
    epoch: u64,
    value: T,
}

/// A process-global, lazily built value that is rebuilt (not reused) in a fork child.
///
/// Values are leaked: a `ForkCell` never drops what it holds (it is a `static`, and an inherited
/// value must not be dropped in the child), so `get_or_init` can hand out `&'static T`.
pub struct ForkCell<T: 'static> {
    slot: AtomicPtr<Slot<T>>,
    /// 0 = nobody is building; else (process id + 1) of the process whose thread is building.
    claim: AtomicU64,
}

impl<T: 'static> ForkCell<T> {
    pub const fn new() -> Self {
        Self {
            slot: AtomicPtr::new(std::ptr::null_mut()),
            claim: AtomicU64::new(0),
        }
    }

    /// The value, if one was built in THIS process (never an inherited one).
    pub fn get(&self) -> Option<&'static T> {
        let p = self.slot.load(Ordering::Acquire);
        // SAFETY: slots are leaked `Box`es, immutable once published.
        let s = unsafe { p.as_ref() }?;
        (s.epoch == EPOCH.load(Ordering::Acquire)).then_some(&s.value)
    }

    /// The value of this process, building it with `init` on first use here. In a fork child
    /// the inherited value is abandoned (leaked) and `init` runs again.
    pub fn get_or_init(&self, init: impl FnOnce() -> T) -> &'static T {
        if let Some(v) = self.get() {
            return v;
        }
        register();
        if policy() == Policy::Error && INHERITED_USED.load(Ordering::Acquire) {
            // Defence in depth: an entry point that did not call `check` must still never hang.
            panic!("{MSG}");
        }
        let mine = u64::from(std::process::id()) + 1;
        let (mut spins, mut nap) = (0u32, 50u64);
        loop {
            if let Some(v) = self.get() {
                return v;
            }
            let c = self.claim.load(Ordering::Acquire);
            if c == mine {
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
            // Re-read the epoch AFTER the claim (a fork may have landed while waiting).
            let epoch = EPOCH.load(Ordering::Acquire);
            let value = init();
            let slot = Box::into_raw(Box::new(Slot { epoch, value }));
            USED.store(true, Ordering::Release);
            // The previous slot (inherited, or none) is leaked on purpose.
            self.slot.store(slot, Ordering::Release);
            self.claim.store(0, Ordering::Release);
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
