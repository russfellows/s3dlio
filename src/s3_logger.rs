// src/s3_logger.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! I/O logging functionality to trace S3 operations.
//!
//! Creates a zstd-compressed, tab-separated log file with a format
//! identical to the 'warp-replay' tool.

use once_cell::sync::OnceCell;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{
    mpsc::{channel, sync_channel, Receiver, SyncSender, TrySendError},
    Arc, Mutex, MutexGuard, OnceLock, PoisonError,
};
use std::thread;
use std::time::{Duration, SystemTime};
use tracing::info;
use zstd::stream::write::Encoder;

/// Process-wide op-log lifecycle: `Uninitialized` -> `Active` -> `Finalized`.
///
/// `Finalized` is terminal: a process supports exactly one op-log session
/// (issue #170). It keeps the (closed) logger only so that a repeated or
/// concurrent finalize can wait for the first one to finish draining.
enum OpLogState {
    Uninitialized,
    Active(Logger),
    Finalized(Logger),
}

// Global variables
static OP_LOG_STATE: Mutex<OpLogState> = Mutex::new(OpLogState::Uninitialized);
static GLOBAL_CLIENT_ID: OnceCell<Mutex<String>> = OnceCell::new();

/// Top bit of `Shared::gate`: set once finalize has begun. The low bits count
/// `Logger::try_log` calls currently between "gate entered" and "outcome decided".
const GATE_CLOSED: u64 = 1 << 63;
/// Name of the background writer thread. Linux keeps at most 15 bytes of a
/// thread name (longer names are cut), so this stays within 15; it shows in
/// debuggers and in `/proc/<pid>/task/<tid>/comm` (`ps -T -o comm`).
const WRITER_THREAD_NAME: &str = "s3dlio-oplog";
const HEADER: &str = "idx\tthread\top\tclient_id\tn_objects\tbytes\tendpoint\tfile\terror\tstart\tfirst_byte\tend\tduration_ns\n";

/// Test-only scheduling noise injected at the two points of [`Logger::try_log`]
/// where a race with `shutdown` can matter: between the closed-bit hint and the
/// gate increment, and between entering the gate and sending. Enabled per
/// thread, so it never affects other tests; compiled out of non-test builds.
#[cfg(test)]
mod race_hooks {
    use std::cell::Cell;
    use std::time::Duration;

    type Hook = Box<dyn FnOnce()>;

    thread_local! {
        static STATE: Cell<Option<u64>> = const { Cell::new(None) };
        static ENTERED: Cell<u64> = const { Cell::new(0) };
        static BEFORE_GATE: Cell<Option<Hook>> = const { Cell::new(None) };
    }

    /// Run `hook` once, on the current thread, at the next `try_log` between the
    /// closed-bit hint and the gate increment (the window the hint cannot close).
    pub(super) fn run_once_before_gate(hook: impl FnOnce() + 'static) {
        BEFORE_GATE.with(|h| h.set(Some(Box::new(hook))));
    }

    /// Run (and clear) the hook set by [`run_once_before_gate`], if any.
    pub(super) fn before_gate() {
        if let Some(hook) = BEFORE_GATE.with(Cell::take) {
            hook();
        }
    }

    /// Record that the current thread just incremented the in-flight count.
    pub(super) fn count_entry() {
        ENTERED.with(|c| c.set(c.get() + 1));
    }

    /// How many times the current thread has incremented the in-flight count.
    pub(super) fn entered() -> u64 {
        ENTERED.with(Cell::get)
    }

    /// Turn the noise on for the current thread (xorshift state from `seed`).
    pub(super) fn enable(seed: u64) {
        STATE.with(|s| s.set(Some(seed | 1)));
    }

    /// Possibly pause the current thread for a pseudo-random moment.
    pub(super) fn point() {
        STATE.with(|s| {
            let Some(mut x) = s.get() else { return };
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            s.set(Some(x));
            match x % 4 {
                0 => {}
                1 => std::thread::yield_now(),
                2 => (0..(x >> 8) % 3000).for_each(|_| std::hint::spin_loop()),
                _ => std::thread::sleep(Duration::from_micros((x >> 8) % 150)),
            }
        });
    }
}

/// Lock the lifecycle state, recovering from a poisoned lock: every transition
/// is a single assignment, so the state is consistent even if a holder panicked.
fn lock_state() -> MutexGuard<'static, OpLogState> {
    OP_LOG_STATE.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Start the process's op-log session, writing to `path`.
///
/// A process supports **exactly one** op-log session:
/// `Uninitialized` -> `Active` -> `Finalized` (terminal).
///
/// - Uninitialized: resolves `path` (plain local path or `file://` destination,
///   s3dlio's file-store convention, not full RFC 8089), creates
///   the file and becomes Active. A failed init (bad path, unsupported scheme,
///   I/O error) leaves the op-log Uninitialized, so a corrected call can succeed.
/// - Active: a no-op returning `Ok(())`. The new `path` is **not** created or
///   truncated and the session keeps writing to the original file.
/// - Finalized: returns an error and performs no filesystem action. The
///   op-log cannot be restarted after [`finalize_op_logger`].
///
/// Stores capture the logger at construction (`store_for_uri_with_logger(uri,
/// global_logger())`): stores created before init are unlogged, and stores
/// created before finalize stop recording when it runs.
pub fn init_op_logger<P: AsRef<str>>(path: P) -> std::io::Result<()> {
    let path = path.as_ref();
    {
        let mut state = lock_state();
        match &*state {
            OpLogState::Active(_) => return Ok(()),
            // Checked before any filesystem action: must not create or truncate `path`.
            OpLogState::Finalized(_) => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::Unsupported,
                    format!(
                        "op-log was already finalized and cannot be restarted; a process \
                         supports only one op-log session, so '{path}' was not opened"
                    ),
                ))
            }
            OpLogState::Uninitialized => {}
        }
        // Created while the lifecycle lock is held, so concurrent inits cannot
        // both create a file.
        let logger = Logger::new(path)?;
        *state = OpLogState::Active(logger);
    }

    // Logged after the lifecycle lock is released: a tracing subscriber may be
    // slow, or may itself touch the op-log.
    info!("Intialized S3 operation logging to file: {}", path);

    Ok(())
}

/// Fetch the global logger. `Some` only while the op-log is Active: `None`
/// before [`init_op_logger`] and again after [`finalize_op_logger`].
///
/// Stores capture the returned logger when they are constructed. A store built
/// before init is unlogged forever; a store built before finalize keeps its
/// clone, but its entries are rejected (`LogOutcome::Closed`) once finalize runs.
pub fn global_logger() -> Option<Logger> {
    match &*lock_state() {
        OpLogState::Active(logger) => Some(logger.clone()),
        OpLogState::Uninitialized | OpLogState::Finalized(_) => None,
    }
}

/// Counts of what happened to the entries submitted to the process's op-log
/// session, so a caller can detect most ways records are lost.
///
/// Available while the session is Active and after it is Finalized (`None`
/// only before [`init_op_logger`]). The outcome counts cover every `log` /
/// `try_log` call that reached the session's [`Logger`] (through any clone)
/// between init and finalize; `written` and `writer_error` come from the writer
/// thread.
///
/// # What is counted, and what is not
///
/// "Written" means the writer thread handed the row to the zstd encoder without
/// an error, and that the bytes the encoder and the file buffer were accepted by
/// the OS `write(2)` call. There is **no fsync**: written does not mean durable.
///
/// These losses are **not** visible in the stats (a program that needs them
/// covered must handle them itself):
///
/// - **Process exit without [`finalize_op_logger`].** Nothing finalizes the log
///   at exit (there is no `atexit` hook, and the global logger lives in a static
///   that is never dropped): entries still on the queue, the buffered bytes and
///   the zstd frame end are lost, and the stats are never read. Always finalize
///   before the process exits.
/// - **Stores built before [`init_op_logger`]** hold no logger. Their operations
///   are not logged and not counted.
/// - **Errors reported only at `close(2)`** (for example NFS or quota errors that
///   show up when the file is closed). The writer drops the `File` after its last
///   flush, and Rust's `File` drop ignores the result of `close`, so such an
///   error is not observable here. An fsync would not be added: it would cost
///   performance and the op-log mirrors native write semantics.
///
/// # When is the log complete?
///
/// After [`finalize_op_logger`] has returned, every loss that **is** counted is
/// absent exactly when [`OpLogStats::is_complete`] is true:
///
/// ```text
/// accepted == written  &&  dropped_full == 0  &&  writer_failed == 0  &&  writer_error is None
/// ```
///
/// - `accepted == written` catches entries that were queued but never reached
///   the encoder, which is what a writer that dies on an I/O error with entries
///   still queued leaves behind (`writer_failed` only counts calls made *after*
///   the writer died, so on its own it misses them).
/// - `writer_error` is the first error the writer met: a failed write, a failed
///   zstd finish or final flush (the rows may all have been counted as written
///   yet not be on disk), or a writer panic (a defect, not an I/O error).
///
/// `rejected_closed` is deliberately **not** part of "complete": those calls
/// arrived after finalize had begun and were never going to be in the file, but
/// a program that still logs after finalizing has a bug of its own (finalize
/// only after the workers stop), so check `rejected_closed == 0` as well.
///
/// Read the stats **after** finalize returns for the final answer. While the
/// session is Active, `accepted > written` is normal (entries are queued), and
/// the counters are relaxed atomics: a snapshot taken while producers are still
/// logging is a progress report, not a transaction. Once finalize has returned,
/// every call that was accepted, dropped or refused before it began has
/// happened-before the read (see [`Logger::stats`]).
///
/// Not for construction by callers (`#[non_exhaustive]`): new fields may be added.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct OpLogStats {
    /// Entries queued for the writer (`LogOutcome::Accepted`).
    pub accepted: u64,
    /// Entries dropped because the queue was full (`LogOutcome::Full`);
    /// only happens in the default non-lossless mode.
    pub dropped_full: u64,
    /// Entries rejected because finalize had begun (`LogOutcome::Closed`).
    pub rejected_closed: u64,
    /// Entries refused because the writer had already stopped
    /// (`LogOutcome::WriterFailed`). Entries that were queued *before* the
    /// writer stopped are not counted here; they show as `accepted - written`.
    pub writer_failed: u64,
    /// True if the session blocks on a full queue instead of dropping
    /// (`S3DLIO_OPLOG_LOSSLESS=1`).
    pub lossless: bool,
    /// Rows the writer thread handed to the zstd encoder without an error. An
    /// upper bound of the rows in the file: the encoder and the file buffer
    /// them, so a later flush/finish failure can still lose some.
    pub written: u64,
    /// The first error the writer thread met, if any (write, zstd finish,
    /// final flush, or a panic), as text.
    pub writer_error: Option<String>,
}

impl OpLogStats {
    /// True when none of the counted losses happened: `accepted == written &&
    /// dropped_full == 0 && writer_failed == 0 && writer_error.is_none()`.
    /// Meaningful once [`finalize_op_logger`] has returned (see the type docs for
    /// what it cannot see); calls counted in `rejected_closed` are not part of it.
    pub fn is_complete(&self) -> bool {
        self.accepted == self.written
            && self.dropped_full == 0
            && self.writer_failed == 0
            && self.writer_error.is_none()
    }
}

/// What the writer's recorded error says went wrong, which decides the advice:
/// an I/O error points at the destination, a panic or a vanished thread does not.
enum WriterFailure {
    /// A failed write, zstd finish or final flush (or no error recorded at all).
    Io,
    /// The writer thread panicked: a defect in this process, not the destination.
    Panic,
    /// The writer thread ended without completing, with no error of its own.
    Vanished,
}

impl WriterFailure {
    fn of(error: Option<&str>) -> Self {
        match error {
            Some(e) if e.starts_with(WRITER_PANICKED) => Self::Panic,
            Some(e) if e.starts_with(&format!("{WRITER_VANISHED}: {WRITER_VANISHED_WHY}")) => {
                Self::Vanished
            }
            _ => Self::Io,
        }
    }
}

/// The single warning printed when finalize finds that records were lost;
/// `None` when the log is complete. Names only the non-zero losses and gives
/// only the advice that applies to the session's mode and to the kind of writer
/// failure (an I/O error, a panic, or a thread that vanished).
fn incomplete_warning(stats: &OpLogStats) -> Option<String> {
    if stats.is_complete() {
        return None;
    }
    let unwritten = stats.accepted.saturating_sub(stats.written);
    let mut lines = Vec::new();
    let mut advice = Vec::new();
    if stats.dropped_full > 0 {
        lines.push(format!(
            "{} entries were dropped because the queue was full",
            stats.dropped_full
        ));
    }
    if stats.writer_failed > 0 {
        lines.push(format!(
            "{} entries were refused because the writer had already stopped",
            stats.writer_failed
        ));
    }
    if unwritten > 0 {
        lines.push(format!("{unwritten} accepted entries were never written"));
    }
    if stats.written > stats.accepted {
        // The counters are separate relaxed atomics: a snapshot read while
        // producers are still logging can see a row as written before its accept.
        lines.push(format!(
            "{} rows written but only {} accepted (the counters were read while \
             the session was still active)",
            stats.written, stats.accepted
        ));
    }
    if let Some(error) = &stats.writer_error {
        lines.push(format!("the writer failed: {error}"));
    }
    if stats.rejected_closed > 0 {
        lines.push(format!(
            "{} calls arrived after finalize began and are not in the file",
            stats.rejected_closed
        ));
    }
    if stats.dropped_full > 0 && !stats.lossless {
        advice.push(
            "To avoid queue-full drops set S3DLIO_OPLOG_LOSSLESS=1 (callers wait \
             instead of dropping) and/or raise S3DLIO_OPLOG_BUF (queue capacity in entries)."
                .to_string(),
        );
    }
    if unwritten > 0 || stats.writer_failed > 0 || stats.writer_error.is_some() {
        advice.push(match WriterFailure::of(stats.writer_error.as_deref()) {
            WriterFailure::Io => "Check the op-log destination for I/O errors (full disk, \
                 permissions, a failing device) and this process's stderr for the \
                 writer's message."
                .to_string(),
            WriterFailure::Panic => "The writer thread panicked: that is a defect in this \
                 process, not an I/O error. Entries after the panic point were lost. The \
                 panic message is on this process's stderr; please report it."
                .to_string(),
            WriterFailure::Vanished => "The writer thread ended abnormally without \
                 completing, so entries it had not written were lost. Check this \
                 process's stderr for the cause."
                .to_string(),
        });
    }
    let mut text = format!(
        "op-log is INCOMPLETE ({} accepted, {} written):\n  - {}",
        stats.accepted,
        stats.written,
        lines.join("\n  - ")
    );
    for line in advice {
        text.push('\n');
        text.push_str(&line);
    }
    Some(text)
}

/// Statistics of the process's op-log session: `Some` while it is Active and
/// after it is Finalized, `None` only before [`init_op_logger`]. See
/// [`OpLogStats`] for how to use them to detect dropped records.
pub fn op_log_stats() -> Option<OpLogStats> {
    match &*lock_state() {
        OpLogState::Active(logger) | OpLogState::Finalized(logger) => Some(logger.stats()),
        OpLogState::Uninitialized => None,
    }
}

/// Set clock offset for the global logger (convenience function).
///
/// # Parameters
/// - `offset_nanos`: Nanoseconds to subtract from local timestamps for distributed sync
///
/// # Returns
/// - `Ok(())` if the op-log is Active and the offset was set
/// - `Err(NotFound)` if the op-log was never initialised
/// - `Err(Unsupported)` if the op-log was already finalized: the session is over,
///   so there is nothing left to configure (see [`finalize_op_logger`])
///
/// # Example
/// ```ignore
/// // Initialize logger
/// s3dlio::init_op_logger("operations.log.zst")?;
///
/// // Calculate clock offset during agent sync
/// let offset = (local_time - controller_time).as_nanos() as i64;
///
/// // Set offset for all future log entries
/// s3dlio::set_clock_offset(offset)?;
/// ```
pub fn set_clock_offset(offset_nanos: i64) -> std::io::Result<()> {
    with_active_logger("set_clock_offset", |logger| {
        logger.set_clock_offset(offset_nanos)
    })
}

/// Get the current clock offset from the global logger.
///
/// Errors exactly like [`set_clock_offset`]: `NotFound` before
/// [`init_op_logger`], `Unsupported` after [`finalize_op_logger`].
pub fn get_clock_offset() -> std::io::Result<i64> {
    with_active_logger("get_clock_offset", |logger| logger.get_clock_offset())
}

/// Run `f` on the Active global logger, or explain why there is none. The two
/// failure states are told apart because the remedy differs: a logger that was
/// never initialised needs `init_op_logger()`, a finalized one can never come back.
fn with_active_logger<T>(what: &str, f: impl FnOnce(&Logger) -> T) -> std::io::Result<T> {
    match &*lock_state() {
        OpLogState::Active(logger) => Ok(f(logger)),
        OpLogState::Uninitialized => Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("{what}: Logger not initialized - call init_op_logger() first"),
        )),
        OpLogState::Finalized(_) => Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            format!(
                "{what}: the op-log was already finalized (a process supports only one \
                 op-log session), so there is no active logger"
            ),
        )),
    }
}

/// Set the global client ID for operation logging.
///
/// All operations logged after this call will use the specified client_id value.
/// This is useful for identifying which client/agent generated which operations
/// in distributed systems or multi-client scenarios.
///
/// # Parameters
/// - `client_id`: String identifier for this client (e.g., "agent-1", "worker-3")
///
/// # Example
/// ```ignore
/// // Initialize logger
/// s3dlio::init_op_logger("operations.log.zst")?;
///
/// // Set client ID for all subsequent operations
/// s3dlio::set_client_id("agent-1")?;
/// ```
pub fn set_client_id(client_id: &str) -> std::io::Result<()> {
    // Initialize global client_id storage if not already done
    let _ = GLOBAL_CLIENT_ID.get_or_init(|| Mutex::new(String::new()));

    if let Some(global_id) = GLOBAL_CLIENT_ID.get() {
        if let Ok(mut id) = global_id.lock() {
            *id = client_id.to_string();
            Ok(())
        } else {
            Err(std::io::Error::other("Failed to acquire client_id lock"))
        }
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Failed to initialize client_id storage",
        ))
    }
}

/// Get the current global client ID.
pub fn get_client_id() -> String {
    GLOBAL_CLIENT_ID
        .get()
        .and_then(|m| m.lock().ok())
        .map(|id| id.clone())
        .unwrap_or_default()
}

/// Finish the process's op-log session: every record whose [`Logger::try_log`]
/// (or `log`) call was accepted is written (unless the writer failed: an I/O
/// error or a panic in the writer thread, see [`OpLogStats::writer_error`]), and
/// the writer has finished the zstd stream and signalled completion before this
/// returns.
///
/// The process must call this before it exits: nothing finalizes the log
/// automatically, so without it queued entries and the zstd frame end are lost.
///
/// Finalize is terminal and idempotent. Entries submitted once it has begun are
/// rejected (`LogOutcome::Closed`), [`global_logger`] returns `None` and the
/// op-log cannot be initialised again in this process. Called while
/// Uninitialized it is a no-op; called again (or concurrently) after
/// finalization it waits for the first call to finish and returns.
pub fn finalize_op_logger() {
    let logger = {
        let mut state = lock_state();
        match std::mem::replace(&mut *state, OpLogState::Uninitialized) {
            OpLogState::Uninitialized => return,
            OpLogState::Active(logger) | OpLogState::Finalized(logger) => {
                *state = OpLogState::Finalized(logger.clone());
                logger
            }
        }
    };
    // The state lock is released: global_logger() already reports None while we drain.
    logger.shutdown();
}

/// A record of one S3 operation (GET, PUT, LIST, DELETE, HEAD).
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub idx: u64,
    pub thread_id: usize,
    pub operation: String,
    pub client_id: String,
    pub num_objects: u32,
    pub bytes: u64,
    pub endpoint: String,
    pub file: String,
    pub error: Option<String>,
    pub start_time: SystemTime,
    pub first_byte_time: Option<SystemTime>,
    pub end_time: SystemTime,
}

impl LogEntry {
    fn to_log_line(&self, clock_offset_nanos: i64) -> String {
        // Apply clock offset correction for distributed systems
        // Positive offset means local clock is ahead, so we subtract to align with reference time
        // `unsigned_abs` instead of negation: `-i64::MIN` overflows (a panic in
        // the writer thread when overflow checks are on). For every other offset
        // the magnitude is the same value as before.
        let offset_duration = Duration::from_nanos(clock_offset_nanos.unsigned_abs());

        let corrected_start = if clock_offset_nanos >= 0 {
            self.start_time
                .checked_sub(offset_duration)
                .unwrap_or(self.start_time)
        } else {
            self.start_time
                .checked_add(offset_duration)
                .unwrap_or(self.start_time)
        };

        let corrected_end = if clock_offset_nanos >= 0 {
            self.end_time
                .checked_sub(offset_duration)
                .unwrap_or(self.end_time)
        } else {
            self.end_time
                .checked_add(offset_duration)
                .unwrap_or(self.end_time)
        };

        let corrected_first_byte = self.first_byte_time.and_then(|t| {
            if clock_offset_nanos >= 0 {
                t.checked_sub(offset_duration)
            } else {
                t.checked_add(offset_duration)
            }
        });

        let duration_ns = corrected_end
            .duration_since(corrected_start)
            .unwrap_or_default()
            .as_nanos();
        let start_ts = humantime::format_rfc3339_nanos(corrected_start).to_string();
        let end_ts = humantime::format_rfc3339_nanos(corrected_end).to_string();
        let first_byte_ts = corrected_first_byte
            .map(|t| humantime::format_rfc3339_nanos(t).to_string())
            .unwrap_or_default();
        let error_str = self.error.as_deref().unwrap_or_default();
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            self.idx,
            self.thread_id,
            self.operation,
            self.client_id,
            self.num_objects,
            self.bytes,
            self.endpoint,
            self.file,
            error_str,
            start_ts,
            first_byte_ts,
            end_ts,
            duration_ns
        )
    }
}

/// Capacity (in log entries) of the bounded channel that feeds the
/// background writer thread; entries are dropped via `try_send()` once
/// full.
///
/// audit #157 bug 7.3 (F3): this previously defaulted to 256, while
/// `docs/Environment_Variables.md` documented 8192 -- a 32x mismatch. Under
/// a moderately fast burst of ops the 256-slot channel filled and started
/// silently dropping entries at a fraction of the load the docs promised.
/// Corrected the code default to match the documented value rather than
/// weakening the docs to match the smaller, operationally worse default.
fn oplog_channel_capacity() -> usize {
    std::env::var("S3DLIO_OPLOG_BUF")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8192)
}

/// Byte capacity of the `BufWriter` used for the op-log file.
///
/// audit #157 bug 7.3 (F3): `docs/Environment_Variables.md` documented a
/// default of 4096 bytes, but the code default is 256 KiB (262144) -- a
/// buffered-I/O size, not an entry count, chosen to avoid flushing on
/// every write. Corrected the docs to match this code default rather than
/// shrinking the buffer 64x (which would increase flush frequency and hurt
/// write throughput for no benefit).
fn oplog_write_buffer_capacity() -> usize {
    std::env::var("S3DLIO_OPLOG_WBUFCAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256 * 1024)
}

/// Result of [`Logger::try_log`]. Only `Accepted` entries can be in the file once
/// the session is finalized, and they will be unless the writer fails first (an
/// I/O error, or a panic in the writer thread; see [`OpLogStats::writer_error`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum LogOutcome {
    /// Queued for the writer; it will be written, even if finalize runs next,
    /// unless the writer fails (an I/O error, or a panic in the writer thread)
    /// before reaching it.
    Accepted,
    /// The queue was full (non-lossless mode): the entry was dropped.
    Full,
    /// The session is finalizing or finalized: the entry was rejected, not written.
    Closed,
    /// The writer stopped (after an I/O error or a panic): the entry was not written.
    WriterFailed,
}

/// What the writer thread receives. A dedicated variant (rather than a magic
/// operation name) ends the session, so no real entry can be mistaken for it.
#[derive(Debug)]
enum Message {
    Entry(LogEntry),
    Shutdown,
}

/// State shared by every clone of a [`Logger`].
#[derive(Debug)]
struct Shared {
    sender: SyncSender<Message>,
    /// `GATE_CLOSED` bit plus the number of `log()` calls in flight. See
    /// [`Logger::try_log`] and [`Logger::shutdown`] for the drain protocol.
    gate: AtomicU64,
    /// Completion signal from the writer thread; `Some` until a shutdown has seen
    /// the writer finish. It is only taken out AFTER the wait, so a shutdown that
    /// panics before that (poisoning the mutex) leaves it for the next caller.
    /// The mutex is held for the whole shutdown so later callers wait for it.
    done_rx: Mutex<Option<Receiver<()>>>,
    clock_offset_nanos: Arc<AtomicI64>, // Nanoseconds to adjust timestamps for distributed sync
    /// Block on a full queue instead of dropping (`S3DLIO_OPLOG_LOSSLESS`).
    /// Fixed per logger when it is built.
    lossless: bool,
    closed_noted: AtomicBool,
    writer_failure_noted: AtomicBool,
    /// Outcome counters behind [`Logger::stats`]. Each is bumped only on the
    /// path it names, with a relaxed `fetch_add`.
    accepted: AtomicU64,
    dropped_full: AtomicU64,
    rejected_closed: AtomicU64,
    writer_failed: AtomicU64,
    /// What the writer thread reports (rows written, first error); a separate,
    /// cache-line-aligned allocation so its stores never touch the line the
    /// producers' counters and gate live on.
    writer: Arc<WriterReport>,
    /// Set (Release) by the first shutdown as its very last step; read
    /// (Acquire) by [`Logger::stats`]. See there for the ordering argument.
    final_published: AtomicBool,
}

/// What the writer thread tells everyone else. Written only by the writer thread
/// (and, for a vanished writer, by shutdown), so `written` is never contended.
#[derive(Debug, Default)]
#[repr(align(128))]
struct WriterReport {
    /// Rows handed to the zstd encoder without an error.
    written: AtomicU64,
    /// The FIRST error the writer met; later ones are ignored.
    first_error: OnceLock<String>,
}

impl WriterReport {
    /// Remember `what` went wrong, unless an earlier error is already recorded.
    fn record_error(&self, what: &str, error: impl std::fmt::Display) {
        let _ = self.first_error.set(format!("{what}: {error}"));
    }
}

/// How a writer failure is recorded (the start of `WriterReport::first_error`),
/// so the finalize warning can tell a panic or a vanished thread from an I/O error.
const WRITER_PANICKED: &str = "writer thread panicked";
const WRITER_PANICKED_FINISHING: &str = "writer thread panicked while finishing";
const WRITER_VANISHED: &str = "writer thread";
const WRITER_VANISHED_WHY: &str = "ended without completing (it was terminated abnormally)";

/// Wait for the writer thread's completion message. A closed channel with no
/// message means the thread ended without finishing its work, which is an error.
fn wait_for_writer(done: &Receiver<()>, report: &WriterReport) {
    if done.recv().is_err() {
        report.record_error(WRITER_VANISHED, WRITER_VANISHED_WHY);
    }
}

/// Print one diagnostic line to stderr. Never panics: `eprintln!` panics when
/// stderr is closed or full, and these notes run on the caller's own I/O path
/// (`try_log`), where losing a note is far better than failing the caller.
fn stderr_note(message: &str) {
    let _ = writeln!(std::io::stderr(), "{message}");
}

/// Text of a panic payload.
fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_string())
}

/// A zstd encoder that finishes its stream when dropped (so every early exit,
/// such as a failed thread spawn in [`Logger::start`], still leaves a complete
/// stream, as `auto_finish` did) but can also be finished explicitly, which is
/// the only way to see the error.
struct StreamEncoder<W: Write>(Option<Encoder<'static, W>>);

impl<W: Write> StreamEncoder<W> {
    fn new(sink: W, level: i32) -> std::io::Result<Self> {
        Ok(Self(Some(Encoder::new(sink, level)?)))
    }

    fn encoder(&mut self) -> std::io::Result<&mut Encoder<'static, W>> {
        self.0
            .as_mut()
            .ok_or_else(|| std::io::Error::other("the zstd stream is already finished"))
    }

    /// Write the zstd frame end and hand back the sink.
    fn finish(&mut self) -> std::io::Result<W> {
        self.0
            .take()
            .ok_or_else(|| std::io::Error::other("the zstd stream is already finished"))?
            .finish()
    }
}

impl<W: Write> Write for StreamEncoder<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.encoder()?.write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.encoder()?.flush()
    }
}

impl<W: Write> Drop for StreamEncoder<W> {
    fn drop(&mut self) {
        if let Some(encoder) = self.0.take() {
            // Nobody is left to tell: an explicit `finish` is where errors are kept.
            let _ = encoder.finish();
        }
    }
}

/// The writer thread's loop: numbers and writes entries until `Shutdown`, the
/// queue closing, or a write error (recorded, then the loop ends). The
/// `receiver` is consumed, so when the loop ends producers see the queue closed.
fn write_rows<W: Write>(
    receiver: Receiver<Message>,
    encoder: &mut StreamEncoder<W>,
    clock_offset: &AtomicI64,
    report: &WriterReport,
) {
    let mut written = 0_u64;
    for (idx, message) in (0_u64..).zip(receiver) {
        let mut entry = match message {
            Message::Entry(entry) => entry,
            Message::Shutdown => break,
        };
        entry.idx = idx;
        let offset = clock_offset.load(Ordering::Relaxed);
        let line = entry.to_log_line(offset);
        if let Err(e) = encoder.write_all(line.as_bytes()) {
            report.record_error("write failed", &e);
            let _ = writeln!(std::io::stderr(), "Error writing to op-log: {e}");
            break;
        }
        written += 1;
        // The only writer of this counter: a plain store, no read-modify-write.
        report.written.store(written, Ordering::Relaxed);
    }
}

/// Finish the zstd stream and flush the sink, recording (not dropping) errors.
///
/// The last thing that happens to the sink is its drop, which for the production
/// sink (`BufWriter<File>`) closes the file. A failing `close(2)` (reported late
/// by some network filesystems, or on a quota) is not observable: Rust's `File`
/// drop ignores it, and the op-log deliberately adds no fsync. So the flush above
/// it is the last error this writer can see, and `written` means "accepted by
/// `write(2)`", not durable.
fn finish_stream<W: Write>(encoder: &mut StreamEncoder<W>, report: &WriterReport) {
    match encoder.finish() {
        Ok(mut sink) => {
            if let Err(e) = sink.flush() {
                report.record_error("final flush failed", &e);
            }
            drop(sink); // explicit: the close error, if any, is ignored by `File`
        }
        Err(e) => report.record_error("finishing the zstd stream failed", &e),
    }
}

/// Everything the writer thread does: write rows, then finish the stream, then
/// say it is done. A panic in either part is caught and recorded, so shutdown
/// never waits forever and the failure is visible in the stats; the stream is
/// still finished after a panic in the row loop.
fn run_writer<W: Write>(
    receiver: Receiver<Message>,
    mut encoder: StreamEncoder<W>,
    clock_offset: &AtomicI64,
    report: &WriterReport,
    done: &std::sync::mpsc::Sender<()>,
) {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    if let Err(payload) = catch_unwind(AssertUnwindSafe(|| {
        write_rows(receiver, &mut encoder, clock_offset, report);
    })) {
        report.record_error(WRITER_PANICKED, panic_text(&*payload));
    }
    if let Err(payload) = catch_unwind(AssertUnwindSafe(|| finish_stream(&mut encoder, report))) {
        report.record_error(WRITER_PANICKED_FINISHING, panic_text(&*payload));
    }
    drop(encoder);
    let _ = done.send(());
}

/// The body of the writer thread: [`run_writer`] in production.
type WriterBody<W> = fn(
    Receiver<Message>,
    StreamEncoder<W>,
    &AtomicI64,
    &WriterReport,
    &std::sync::mpsc::Sender<()>,
);

impl Shared {
    /// Outcome for a call that found the logger closed; prints the single
    /// "finalized" diagnostic for the whole logger the first time.
    fn reject_closed(&self) -> LogOutcome {
        self.rejected_closed.fetch_add(1, Ordering::Relaxed);
        if !self.closed_noted.swap(true, Ordering::Relaxed) {
            stderr_note(
                "op-log finalized: entries from stores created before finalize are no \
                 longer recorded",
            );
        }
        LogOutcome::Closed
    }
}

/// Handle for sending log entries. Clones share one writer thread.
///
/// Invariant: the op-log never records its own I/O, whatever form the
/// destination takes (plain path or `file://` URI). The writer thread writes
/// with `std::fs` and `zstd` directly and must NEVER go through an
/// `ObjectStore` or any other logged path: doing so would add a row for every
/// row written. Tests `oplog_lifecycle_no_self_log_*` pin this.
#[derive(Debug, Clone)]
pub struct Logger {
    shared: Arc<Shared>,
}

/// Decrements the in-flight count when a `log()` call ends, including by unwinding.
struct InFlight<'a>(&'a AtomicU64);

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        // Release: our send (if any) happens-before shutdown observes the count drop.
        self.0.fetch_sub(1, Ordering::Release);
    }
}

/// True if `s` is a valid RFC 3986 URI scheme: an ASCII letter followed by
/// ASCII letters, digits, `+`, `-` or `.`.
fn is_rfc3986_scheme(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic())
        && chars.all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
}

/// Resolve a user-supplied op-log destination to a local filesystem path.
///
/// This is **s3dlio's `file://` convention**, the same as the `file://` object
/// store (`FileSystemObjectStore::uri_to_path` in `file_store.rs`); it is **not**
/// full RFC 8089:
///
/// - A destination that starts with the lowercase prefix `file://` has that
///   prefix removed and the remainder is the path, used **literally**:
///   `file:///tmp/x` is `/tmp/x`, but `file://localhost/x` is the *relative*
///   path `localhost/x`, `file://relative/x` is the relative path `relative/x`,
///   and percent sequences are not decoded (`file:///tmp/a%20b` is
///   `/tmp/a%20b`). An empty remainder (`file://`) is `InvalidInput`.
/// - Any other destination whose text before the first `://` is a valid
///   RFC 3986 scheme (a letter, then letters, digits, `+`, `-` or `.`) is
///   rejected with `ErrorKind::InvalidInput` and a message naming the
///   destination: `s3://`, `az://`, `gs://`, `direct://` and also `FILE://` or
///   `File://` (the file store requires the lowercase prefix, so the op-log does
///   too). The scheme is shown as written. Without this check such a string
///   would be passed to `File::create` as a bogus relative path (issue #168).
/// - Text before `://` that is not a valid scheme leaves the destination a plain
///   path: `<dir>/a://b.log` is the local path `<dir>/a:/b.log`.
fn resolve_local_oplog_path(path: &str) -> std::io::Result<&str> {
    const FILE_PREFIX: &str = "file://";
    if let Some(local) = path.strip_prefix(FILE_PREFIX) {
        if local.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("op-log path '{path}' has an empty file path after 'file://'"),
            ));
        }
        return Ok(local);
    }
    // Only text that is a syntactically valid RFC 3986 scheme before the first
    // "://" makes this a URI (any case: "FILE://" is not our "file://"). A local
    // path such as "<dir>/a://b.log" (which the OS resolves to "<dir>/a:/b.log")
    // merely contains "://" and stays a path.
    if let Some((scheme, _)) = path
        .split_once("://")
        .filter(|(scheme, _)| is_rfc3986_scheme(scheme))
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "op-log path '{path}' uses unsupported scheme '{scheme}://'; the op-log must be \
                 written to a local filesystem path, e.g. '/tmp/ops.tsv.zst' or \
                 'file:///tmp/ops.tsv.zst' (lowercase 'file://')"
            ),
        ));
    }
    Ok(path)
}

// Logger methods
impl Logger {
    /// Creates a new logger and spawns the background thread to handle writing.
    ///
    /// `log_file_path` may be a plain local path or a `file://` destination
    /// (s3dlio's file-store convention, not full RFC 8089: the text after
    /// `file://` is the path, literally). Queue size,
    /// lossless mode, buffer size and compression level come from the
    /// `S3DLIO_OPLOG_*` environment variables, read once here.
    pub fn new(log_file_path: &str) -> std::io::Result<Self> {
        let lossless = std::env::var("S3DLIO_OPLOG_LOSSLESS")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        Self::open(log_file_path, oplog_channel_capacity(), lossless)
    }

    /// [`Logger::new`] with the queue capacity and lossless mode given explicitly
    /// (the environment is not consulted for these two).
    fn open(log_file_path: &str, cap: usize, lossless: bool) -> std::io::Result<Self> {
        let local_path = resolve_local_oplog_path(log_file_path)?;

        // Tunable buffer sizes via (env) variables:
        let wbuf: usize = oplog_write_buffer_capacity();
        let level: i32 = std::env::var("S3DLIO_OPLOG_LEVEL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1); // bias to fast

        // Prepare compressed writer.
        // Include the path in the error: a bare "No such file or directory
        // (os error 2)" gives the user nothing to act on (issue #168).
        let file = File::create(local_path).map_err(|e| {
            let shown = if local_path == log_file_path {
                format!("'{local_path}'")
            } else {
                format!("'{local_path}' (from '{log_file_path}')")
            };
            std::io::Error::new(
                e.kind(),
                format!("failed to create op-log file {shown}: {e}"),
            )
        })?;

        // Old, we didn't set a buffer capacity
        //let writer = BufWriter::new(file);
        let writer = BufWriter::with_capacity(wbuf, file);
        Self::start(writer, cap, lossless, level, WRITER_THREAD_NAME)
    }

    /// Write the header to `sink`, then start the background writer thread that
    /// compresses entries into it. Fails (instead of panicking) if the header
    /// cannot be written or the OS refuses to start a thread; the file, if any,
    /// has already been created by the caller and is left as it is.
    fn start<W: Write + Send + 'static>(
        sink: W,
        cap: usize,
        lossless: bool,
        level: i32,
        thread_name: &str,
    ) -> std::io::Result<Self> {
        Self::start_with(sink, cap, lossless, level, thread_name, run_writer::<W>)
    }

    /// [`Logger::start`] with the writer thread's body given explicitly. Production
    /// always passes [`run_writer`]; tests pass a body that misbehaves in a way the
    /// real one is built to prevent (for example, ending without its completion
    /// message), to prove the rest of the machinery copes.
    fn start_with<W: Write + Send + 'static>(
        sink: W,
        cap: usize,
        lossless: bool,
        level: i32,
        thread_name: &str,
        run: WriterBody<W>,
    ) -> std::io::Result<Self> {
        // Bounded channel for log entries (drop on overflow in `log()` via try_send).
        let (sender, receiver): (SyncSender<Message>, Receiver<Message>) = sync_channel(cap);

        // One-shot channel to signal that the background thread finished.
        let (done_tx, done_rx) = channel::<()>();

        let mut encoder = StreamEncoder::new(sink, level)?;

        // Write header now so even empty logs have the header.
        encoder.write_all(HEADER.as_bytes())?;
        encoder.flush()?;

        // Background writer thread.
        let clock_offset = Arc::new(AtomicI64::new(0));
        let clock_offset_clone = Arc::clone(&clock_offset);
        let writer = Arc::new(WriterReport::default());
        let writer_clone = Arc::clone(&writer);
        thread::Builder::new()
            .name(thread_name.to_string())
            .spawn(move || {
                run(
                    receiver,
                    encoder,
                    &clock_offset_clone,
                    &writer_clone,
                    &done_tx,
                );
            })
            .map_err(|e| {
                std::io::Error::new(
                    e.kind(),
                    format!("failed to start the op-log writer thread: {e}"),
                )
            })?;

        Ok(Logger {
            shared: Arc::new(Shared {
                sender,
                gate: AtomicU64::new(0),
                done_rx: Mutex::new(Some(done_rx)),
                clock_offset_nanos: clock_offset,
                lossless,
                closed_noted: AtomicBool::new(false),
                writer_failure_noted: AtomicBool::new(false),
                accepted: AtomicU64::new(0),
                dropped_full: AtomicU64::new(0),
                rejected_closed: AtomicU64::new(0),
                writer_failed: AtomicU64::new(0),
                writer,
                final_published: AtomicBool::new(false),
            }),
        })
    }

    /// Set the clock offset for timestamp correction in distributed systems.
    ///
    /// # Parameters
    /// - `offset_nanos`: Nanoseconds to subtract from local timestamps
    ///   - Positive: local clock is ahead of reference (subtract to correct)
    ///   - Negative: local clock is behind reference (add to correct)
    ///   - Zero: no correction needed
    ///
    /// # Example
    /// ```ignore
    /// // Agent calculates offset during synchronization
    /// let clock_offset_nanos = (local_time - controller_time).as_nanos() as i64;
    ///
    /// // Set offset for all future log entries
    /// logger.set_clock_offset(clock_offset_nanos);
    /// ```
    pub fn set_clock_offset(&self, offset_nanos: i64) {
        self.shared
            .clock_offset_nanos
            .store(offset_nanos, Ordering::Relaxed);
    }

    /// Get the current clock offset value.
    pub fn get_clock_offset(&self) -> i64 {
        self.shared.clock_offset_nanos.load(Ordering::Relaxed)
    }

    /// Counts of what happened to the entries submitted through this logger
    /// (and its clones), and what the writer thread reported. See [`OpLogStats`].
    ///
    /// Ordering: the counters are relaxed atomics, so a read while producers are
    /// logging is only a progress report. After the first `shutdown` (what
    /// [`finalize_op_logger`] runs) has returned, a read sees the final values
    /// of every call that began before the close: each such call bumped its
    /// counter before its in-flight guard dropped (Release), shutdown's
    /// wait-for-zero loads the count with Acquire, the writer's last stores
    /// reach shutdown through the completion channel, and shutdown ends by
    /// storing `final_published` with Release. The Acquire load below pairs with
    /// that store, so everything shutdown had observed happens-before the loads
    /// that follow, on any thread, with no help from the caller. Calls that
    /// start after the close (`rejected_closed`) are concurrent with the read
    /// by definition and are only as exact as any relaxed counter.
    pub fn stats(&self) -> OpLogStats {
        let shared = &*self.shared;
        shared.final_published.load(Ordering::Acquire);
        OpLogStats {
            accepted: shared.accepted.load(Ordering::Relaxed),
            dropped_full: shared.dropped_full.load(Ordering::Relaxed),
            rejected_closed: shared.rejected_closed.load(Ordering::Relaxed),
            writer_failed: shared.writer_failed.load(Ordering::Relaxed),
            lossless: shared.lossless,
            written: shared.writer.written.load(Ordering::Relaxed),
            writer_error: shared.writer.first_error.get().cloned(),
        }
    }

    /// Submit a log entry, ignoring what happened to it.
    ///
    /// This is the original, fire-and-forget API and returns `()`, exactly as
    /// before the op-log lifecycle work. It is [`Logger::try_log`] with the
    /// [`LogOutcome`] discarded; use `try_log` to learn whether the entry was
    /// queued, dropped or rejected.
    pub fn log(&self, entry: LogEntry) {
        let _ = self.try_log(entry);
    }

    /// Submit a log entry and report what happened to it.
    ///
    /// Non-blocking by default (`try_send`): when the queue is full the entry is
    /// dropped and [`LogOutcome::Full`] is returned. With `S3DLIO_OPLOG_LOSSLESS=1`
    /// the call blocks instead of dropping.
    ///
    /// `Accepted` means the entry is on the writer's queue, ahead of the shutdown
    /// marker, so [`finalize_op_logger`] will write it unless the writer fails
    /// (an I/O error or a panic in the writer thread). Once finalize has begun, calls return
    /// [`LogOutcome::Closed`] (never `Full`) and write nothing; at most one
    /// diagnostic is printed for the whole closed logger.
    ///
    /// [`Logger::log`] is this method with the outcome discarded; the store
    /// wrappers use it. Tests use the outcome to prove drain exactness.
    pub fn try_log(&self, entry: LogEntry) -> LogOutcome {
        let shared = &*self.shared;

        // Fast path: once the logger is closed, a caller must not touch the
        // in-flight count at all, or a hot loop of post-close callers could keep
        // the count above zero and starve `shutdown`'s wait-for-zero. This load
        // is only a hint and never admits anything: the authoritative check is
        // the value returned by the `fetch_add` below.
        if shared.gate.load(Ordering::Acquire) & GATE_CLOSED != 0 {
            return shared.reject_closed();
        }
        #[cfg(test)]
        {
            race_hooks::point();
            race_hooks::before_gate();
        }

        // Enter the gate. All updates to `gate` are totally ordered. If our add
        // comes before shutdown's `fetch_or(GATE_CLOSED)` in that order, shutdown's
        // wait-for-zero sees our count and cannot send the sentinel until our
        // guard drops, i.e. until our send has finished. If it comes after, the
        // returned `before` carries the closed bit and we send nothing. So a
        // caller that lost the race with the hint above is rejected here, it only
        // holds the count for the instant it takes to notice (at most once per
        // call, and only for calls that read the gate before the close), and
        // nothing can be accepted behind the sentinel.
        let before = shared.gate.fetch_add(1, Ordering::Acquire);
        #[cfg(test)]
        race_hooks::count_entry();
        let _in_flight = InFlight(&shared.gate);
        if before & GATE_CLOSED != 0 {
            return shared.reject_closed();
        }
        #[cfg(test)]
        race_hooks::point();

        let outcome = if shared.lossless {
            // Blocking: guarantees no drops (may stall if writer can’t keep up).
            match shared.sender.send(Message::Entry(entry)) {
                Ok(()) => {
                    shared.accepted.fetch_add(1, Ordering::Relaxed);
                    LogOutcome::Accepted
                }
                Err(_) => {
                    shared.writer_failed.fetch_add(1, Ordering::Relaxed);
                    LogOutcome::WriterFailed
                }
            }
        } else {
            // Non‑blocking: preserves I/O throughput; may drop under burst.
            match shared.sender.try_send(Message::Entry(entry)) {
                Ok(()) => {
                    shared.accepted.fetch_add(1, Ordering::Relaxed);
                    LogOutcome::Accepted
                }
                Err(TrySendError::Full(_)) => {
                    shared.dropped_full.fetch_add(1, Ordering::Relaxed);
                    stderr_note("op-log channel full; dropping entry");
                    LogOutcome::Full
                }
                Err(TrySendError::Disconnected(_)) => {
                    shared.writer_failed.fetch_add(1, Ordering::Relaxed);
                    LogOutcome::WriterFailed
                }
            }
        };
        if outcome == LogOutcome::WriterFailed
            && !shared.writer_failure_noted.swap(true, Ordering::Relaxed)
        {
            stderr_note("op-log writer has stopped after an error; entries are being dropped");
        }
        outcome
    }

    /// Close the logger, drain every accepted entry and wait for the writer.
    ///
    /// Exactness: (1) set `GATE_CLOSED`, so every later `log()` is rejected;
    /// (2) wait until the in-flight count reaches zero, i.e. every `log()` that
    /// entered before the close has finished its send, so everything accepted is
    /// already on the queue (and is written unless the writer fails with an I/O
    /// error or panic); (3) send `Shutdown`, which the FIFO channel delivers
    /// after all of those entries; (4) wait for the writer to finish the zstd
    /// stream. Idempotent; concurrent callers block until the first finishes.
    fn shutdown(&self) {
        let shared = &*self.shared;
        // A poisoned lock (an earlier shutdown panicked) is recovered: the receiver
        // is still in it unless that shutdown had already seen the writer finish.
        let mut done_rx = shared
            .done_rx
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let Some(done) = done_rx.as_ref() else {
            return;
        };

        shared.gate.fetch_or(GATE_CLOSED, Ordering::AcqRel);
        let mut spins = 0_u32;
        while shared.gate.load(Ordering::Acquire) & !GATE_CLOSED != 0 {
            // In-flight calls only wait for the writer (LOSSLESS) or finish at
            // once, so this is short; back off from yielding to sleeping.
            if spins < 100 {
                thread::yield_now();
            } else {
                thread::sleep(Duration::from_micros(50));
            }
            spins = spins.saturating_add(1);
        }

        // Blocking send so the sentinel cannot be lost on a full queue. If the
        // writer already stopped the send fails, which is fine. (After a panic in an
        // earlier shutdown this is a second sentinel, which nothing reads: harmless.)
        let _ = shared.sender.send(Message::Shutdown);

        // Print info if info set. A tracing subscriber may panic here; the receiver
        // is still in the mutex, so the next shutdown waits for the writer itself.
        info!("Shutting down S3 operation logging");

        // Wait for the background thread to finish the zstd stream and flush the
        // sink. A thread that vanished without saying so is recorded as an error.
        wait_for_writer(done, &shared.writer);
        // The writer is done: later calls have nothing left to wait for.
        *done_rx = None;
        drop(done_rx);

        // Publish the final values (see `stats` for the ordering argument).
        shared.final_published.store(true, Ordering::Release);

        // One warning, from the first (draining) finalize, if records were lost.
        if let Some(warning) = incomplete_warning(&self.stats()) {
            // Never panic on a closed stderr (EPIPE): losing the warning is the
            // lesser evil.
            let _ = writeln!(std::io::stderr(), "{warning}");
        }
    }
}

#[cfg(test)]
mod oplog_buffer_default_tests {
    use super::*;
    use std::sync::Mutex;

    // Mutex to serialize tests that manipulate the shared process-wide
    // S3DLIO_OPLOG_BUF / S3DLIO_OPLOG_WBUFCAP env vars.
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    // RED-then-GREEN regression tests for s3dlio issue #157 bug 7.3 (F3).
    //
    // Bug: code and docs disagreed on both buffer defaults. Assert the
    // code defaults match the values documented in
    // docs/Environment_Variables.md (which was corrected in the same
    // commit as this fix).

    #[test]
    fn channel_capacity_default_matches_documented_8192() {
        let _guard = ENV_MUTEX.lock().unwrap();
        #[allow(deprecated)]
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        assert_eq!(
            oplog_channel_capacity(),
            8192,
            "S3DLIO_OPLOG_BUF's code default must match the documented default"
        );
    }

    #[test]
    fn write_buffer_capacity_default_matches_documented_262144() {
        let _guard = ENV_MUTEX.lock().unwrap();
        #[allow(deprecated)]
        std::env::remove_var("S3DLIO_OPLOG_WBUFCAP");
        assert_eq!(
            oplog_write_buffer_capacity(),
            256 * 1024,
            "S3DLIO_OPLOG_WBUFCAP's code default must match the documented default"
        );
    }

    #[test]
    fn channel_capacity_env_override_still_works() {
        let _guard = ENV_MUTEX.lock().unwrap();
        #[allow(deprecated)]
        std::env::set_var("S3DLIO_OPLOG_BUF", "42");
        let result = oplog_channel_capacity();
        #[allow(deprecated)]
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        assert_eq!(result, 42);
    }
}

#[cfg(test)]
mod oplog_path_tests {
    use super::*;

    // RED-then-GREEN regression tests for s3dlio issue #168.
    //
    // Bug: `init_op_log("file:///tmp/operations.log")` (the form shown in
    // docs/PYTHON_API_GUIDE.md) passed the URI straight to `File::create`,
    // which treated it as a relative path under a non-existent `file:`
    // directory and failed with a bare "No such file or directory
    // (os error 2)" that did not name the path.

    #[test]
    fn file_uri_is_accepted_and_written_to_local_path() {
        let dir = tempfile::tempdir().unwrap();
        let local = dir.path().join("operations.log");
        let uri = format!("file://{}", local.display());

        let logger = Logger::new(&uri).expect("file:// URI must be accepted for the op-log path");
        drop(logger);

        assert!(
            local.exists(),
            "op-log must be created at the local path the file:// URI points to"
        );
    }

    #[test]
    fn plain_local_path_still_works() {
        let dir = tempfile::tempdir().unwrap();
        let local = dir.path().join("operations.log");

        let logger = Logger::new(local.to_str().unwrap()).expect("plain path must work");
        drop(logger);

        assert!(local.exists());
    }

    #[test]
    fn create_failure_error_names_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let bad = dir.path().join("no-such-subdir").join("operations.log");
        let bad = bad.to_str().unwrap();

        let err = Logger::new(bad).expect_err("creating a log in a missing directory must fail");

        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
        let msg = err.to_string();
        assert!(
            msg.contains(bad),
            "error must name the op-log path so users can see what went wrong; got: {msg}"
        );
    }

    #[test]
    fn remote_uri_is_rejected_with_clear_error() {
        for uri in [
            "s3://bucket/operations.log",
            "az://container/operations.log",
            "gs://bucket/operations.log",
            "direct:///tmp/operations.log",
        ] {
            let err = Logger::new(uri)
                .err()
                .unwrap_or_else(|| panic!("{uri} must be rejected as an op-log path"));
            assert_eq!(
                err.kind(),
                std::io::ErrorKind::InvalidInput,
                "{uri}: wrong error kind ({err})"
            );
            let msg = err.to_string();
            assert!(
                msg.contains(uri) && msg.contains("local"),
                "{uri}: error must name the path and say it must be local; got: {msg}"
            );
        }
    }

    // Issue #168 follow-up: only a syntactically valid RFC 3986 scheme makes
    // the text before the first "://" a URI scheme. Anything else is a local
    // path -- `<dir>/a://b.log` is the valid POSIX path `<dir>/a:/b.log`.

    #[test]
    fn path_with_colon_slash_slash_inside_a_directory_name_is_a_local_path() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("a:")).unwrap();
        let path = format!("{}/a://b.log", dir.path().display());

        let logger = Logger::new(&path)
            .unwrap_or_else(|e| panic!("'{path}' is a valid local path but was rejected: {e}"));
        drop(logger);

        assert!(
            dir.path().join("a:").join("b.log").exists(),
            "op-log must be created at <dir>/a:/b.log"
        );
    }

    #[test]
    fn text_before_separator_containing_a_slash_is_not_a_scheme() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("x").join("y:")).unwrap();
        let path = format!("{}/x/y://z.log", dir.path().display());

        let logger = Logger::new(&path)
            .unwrap_or_else(|e| panic!("'{path}' is a valid local path but was rejected: {e}"));
        drop(logger);

        assert!(dir.path().join("x").join("y:").join("z.log").exists());
    }

    #[test]
    fn only_rfc3986_schemes_are_treated_as_uris() {
        // Not valid schemes -> plain paths, returned unchanged.
        for path in [
            "dir/a://b.log",     // '/' is not a scheme character
            "my dir://b.log",    // space
            "1abc://b.log",      // must start with a letter
            "://b.log",          // empty scheme
            "-x://b.log",        // must start with a letter
            "a_b://b.log",       // '_' is not allowed
            "caf\u{e9}://b.log", // non-ASCII letter
        ] {
            assert_eq!(
                resolve_local_oplog_path(path).unwrap_or_else(|e| panic!("{path}: {e}")),
                path
            );
        }
        // Valid schemes (including +, -, . and digits after the first char).
        for uri in ["s3://b/k", "az://c/k", "a://b.log", "x+y-z.1://p"] {
            let err = resolve_local_oplog_path(uri)
                .err()
                .unwrap_or_else(|| panic!("{uri} must be rejected"));
            assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput, "{uri}");
        }
    }

    // Pins that held before the file:// convention change too: the text before the
    // FIRST "://" decides, and it must be a valid scheme.

    #[test]
    fn non_ascii_first_char_is_a_path_and_first_separator_decides() {
        // 'é' is not an ASCII letter, so "é a" is not a scheme.
        for path in ["\u{e9} a://b.log", "dir/a://b://c"] {
            assert_eq!(
                resolve_local_oplog_path(path).unwrap_or_else(|e| panic!("{path}: {e}")),
                path
            );
        }
        // The first "://" is preceded by the valid scheme "s3": a URI, rejected,
        // even though a later "://" exists.
        let err =
            resolve_local_oplog_path("s3://bucket/a://b").expect_err("s3:// must be rejected");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("s3://bucket/a://b"));
    }

    // s3dlio's file:// convention (the same as the file:// object store, see
    // FileSystemObjectStore::uri_to_path): a destination that starts with the
    // lowercase prefix "file://" has that prefix stripped and the remainder is
    // the path, LITERALLY. This is not full RFC 8089: "file://localhost/x" is the
    // relative path "localhost/x", not "/x", and nothing is percent-decoded.

    #[test]
    fn file_prefix_is_stripped_and_the_remainder_is_used_literally() {
        for (uri, want) in [
            ("file:///tmp/x", "/tmp/x"),
            ("file://localhost/x", "localhost/x"),
            ("file://localhost/tmp/x", "localhost/tmp/x"),
            ("file://LocalHost/tmp/x", "LocalHost/tmp/x"),
            ("file://relative/x", "relative/x"),
            ("file://otherhost/tmp/x", "otherhost/tmp/x"),
            ("file://a", "a"),
            ("file:////tmp/x", "//tmp/x"),
            ("file://./x", "./x"),
            // Percent sequences are NOT decoded.
            ("file:///tmp/a%20b", "/tmp/a%20b"),
            ("file://rel%2Fx", "rel%2Fx"),
            // Only the FIRST "file://" is a prefix.
            ("file://file:///x", "file:///x"),
        ] {
            assert_eq!(
                resolve_local_oplog_path(uri).unwrap_or_else(|e| panic!("{uri}: {e}")),
                want,
                "{uri}"
            );
        }
    }

    #[test]
    fn empty_remainder_after_file_prefix_is_rejected_and_names_the_input() {
        let err = resolve_local_oplog_path("file://").expect_err("file:// alone must be rejected");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
        let msg = err.to_string();
        assert!(msg.contains("file://") && msg.contains("empty"), "{msg}");
        let err = Logger::new("file://").expect_err("file:// alone must be rejected");
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn only_the_lowercase_file_prefix_is_the_file_scheme() {
        // The file store requires lowercase "file://", so the op-log does too:
        // any other spelling is an unsupported scheme, shown as written.
        for (uri, shown) in [
            ("FILE:///tmp/x", "'FILE://'"),
            ("File:///tmp/x", "'File://'"),
            ("FILE://localhost/tmp/x", "'FILE://'"),
            ("S3://bucket/k", "'S3://'"),
            ("Az://c/k", "'Az://'"),
            ("DIRECT:///tmp/x", "'DIRECT://'"),
        ] {
            let err = resolve_local_oplog_path(uri)
                .err()
                .unwrap_or_else(|| panic!("{uri} must be rejected"));
            assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput, "{uri}");
            assert!(err.to_string().contains(uri), "{uri}: {err}");
            assert!(err.to_string().contains(shown), "{uri}: {err}");
        }
    }

    #[test]
    fn direct_and_s3_schemes_are_rejected_as_unsupported() {
        for (uri, shown) in [
            ("direct:///tmp/x", "'direct://'"),
            ("direct://relative/x", "'direct://'"),
            ("s3://bucket/ops.tsv.zst", "'s3://'"),
        ] {
            let err = resolve_local_oplog_path(uri)
                .err()
                .unwrap_or_else(|| panic!("{uri} must be rejected"));
            assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput, "{uri}");
            let msg = err.to_string();
            assert!(
                msg.contains(uri) && msg.contains("unsupported scheme"),
                "{msg}"
            );
            assert!(msg.contains(shown), "{msg}");
        }
    }

    // The op-log resolver and the file:// object store must agree on the path
    // for every file:// input, because users pass the same string to both. The
    // store's conversion is called directly (it is pub(crate) for this test).
    #[test]
    fn file_uri_resolves_to_the_same_path_as_the_file_store() {
        use crate::file_store::FileSystemObjectStore;
        use std::path::PathBuf;
        for uri in [
            "file:///tmp/x",
            "file:///tmp/a%20b",
            "file://localhost/x",
            "file://localhost/tmp/x",
            "file://LocalHost/tmp/x",
            "file://relative/x",
            "file://otherhost/tmp/x",
            "file://a",
            "file:////tmp/x",
            "file://./x",
            "file://rel%2Fx",
            "file://file:///x",
        ] {
            let store = FileSystemObjectStore::uri_to_path(uri)
                .unwrap_or_else(|e| panic!("file store rejected {uri}: {e}"));
            let oplog = resolve_local_oplog_path(uri).unwrap_or_else(|e| panic!("{uri}: {e}"));
            assert_eq!(PathBuf::from(oplog), store, "{uri}");
        }
        // Where the store refuses a destination (it requires the lowercase
        // "file://" prefix), the op-log refuses it too.
        for uri in ["FILE:///tmp/x", "direct:///tmp/x", "s3://bucket/k"] {
            assert!(FileSystemObjectStore::uri_to_path(uri).is_err(), "{uri}");
            assert!(resolve_local_oplog_path(uri).is_err(), "{uri}");
        }
        // The one deliberate difference: the store turns "file://" into the
        // empty path, the op-log rejects it (it cannot create a file named "").
        assert_eq!(
            FileSystemObjectStore::uri_to_path("file://").unwrap(),
            PathBuf::new()
        );
        assert!(resolve_local_oplog_path("file://").is_err());
    }
}

#[cfg(test)]
mod oplog_shutdown_tests {
    use super::*;
    use std::path::Path;
    use std::sync::Barrier;
    use std::time::Instant;

    const HEADER_LINE: &str = "idx\tthread\top\tclient_id\tn_objects\tbytes\tendpoint\tfile\terror\tstart\tfirst_byte\tend\tduration_ns";

    fn entry(tag: &str) -> LogEntry {
        let now = SystemTime::now();
        LogEntry {
            idx: 0,
            thread_id: 0,
            operation: "GET".to_string(),
            client_id: String::new(),
            num_objects: 1,
            bytes: 1,
            endpoint: String::new(),
            file: tag.to_string(),
            error: None,
            start_time: now,
            first_byte_time: None,
            end_time: now,
        }
    }

    /// Decompress an op-log (it must be a complete zstd stream), check the full
    /// header line, 13 columns and sequential idx on every row, and return the
    /// `file` column of each row.
    fn read_tags(path: &Path) -> Result<Vec<String>, String> {
        let raw = std::fs::read(path).map_err(|e| format!("read {path:?}: {e}"))?;
        let text = zstd::decode_all(raw.as_slice())
            .map_err(|e| format!("not a complete zstd stream: {e}"))?;
        let text = String::from_utf8(text).map_err(|e| format!("not UTF-8: {e}"))?;
        let mut lines = text.lines();
        if lines.next() != Some(HEADER_LINE) {
            return Err("header line differs from the frozen header".to_string());
        }
        let mut tags = Vec::new();
        for (i, line) in lines.enumerate() {
            let cols: Vec<&str> = line.split('\t').collect();
            if cols.len() != 13 {
                return Err(format!("row {i} has {} columns: {line:?}", cols.len()));
            }
            if cols[0] != i.to_string() {
                return Err(format!("row {i} has idx {}", cols[0]));
            }
            tags.push(cols[7].to_string());
        }
        Ok(tags)
    }

    // A standalone Logger (not the process-global one) can be shut down
    // repeatedly; once closed it reports Closed, never Full.
    #[test]
    fn shutdown_drains_is_idempotent_and_rejects_later_entries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ops.tsv.zst");
        // Explicit capacity: independent of S3DLIO_OPLOG_BUF / _LOSSLESS.
        let logger = Logger::open(path.to_str().unwrap(), 64, false).unwrap();

        assert_eq!(logger.try_log(entry("a")), LogOutcome::Accepted);
        assert_eq!(logger.try_log(entry("b")), LogOutcome::Accepted);
        logger.shutdown();
        logger.shutdown();
        assert_eq!(logger.clone().try_log(entry("c")), LogOutcome::Closed);

        assert_eq!(read_tags(&path).unwrap(), ["a", "b"]);
    }

    // The writer thread's name must fit the 15 bytes Linux keeps (a longer name
    // is silently cut), and must really be the name the thread runs under.
    #[cfg(target_os = "linux")]
    #[test]
    fn writer_thread_name_fits_the_linux_limit_and_is_applied() {
        assert!(
            WRITER_THREAD_NAME.len() <= 15,
            "'{WRITER_THREAD_NAME}' is {} bytes; Linux keeps 15",
            WRITER_THREAD_NAME.len()
        );
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ops.tsv.zst");
        let logger = Logger::open(path.to_str().unwrap(), 8, false).unwrap();
        // std names a thread from inside the new thread, so the name can appear
        // after `open` returns: poll until it does, up to a deadline.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        let comms = loop {
            let comms: Vec<String> = std::fs::read_dir("/proc/self/task")
                .unwrap()
                .filter_map(|t| std::fs::read_to_string(t.ok()?.path().join("comm")).ok())
                .map(|c| c.trim_end().to_string())
                .collect();
            if comms.iter().any(|c| c == WRITER_THREAD_NAME)
                || std::time::Instant::now() >= deadline
            {
                break comms;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        };
        logger.shutdown();
        assert!(
            comms.iter().any(|c| c == WRITER_THREAD_NAME),
            "no thread is named '{WRITER_THREAD_NAME}'; thread names: {comms:?}"
        );
    }

    // Finding 1: a caller that finds the logger closed must not touch the
    // in-flight count at all (otherwise a hot loop of post-close callers can keep
    // the count above zero and starve shutdown's wait-for-zero).
    #[test]
    fn post_close_callers_never_enter_the_in_flight_count() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ops.tsv.zst");
        let logger = Logger::open(path.to_str().unwrap(), 8, false).unwrap();
        logger.shutdown();

        let entered_before = race_hooks::entered();
        for _ in 0..1000 {
            assert_eq!(logger.try_log(entry("late")), LogOutcome::Closed);
        }
        assert_eq!(
            race_hooks::entered() - entered_before,
            0,
            "log() on a closed logger incremented the in-flight count"
        );
        assert_eq!(
            logger.shared.gate.load(Ordering::SeqCst),
            GATE_CLOSED,
            "closed logger must have an empty in-flight count"
        );
    }

    // Finding 1, behavioural form: shutdown completes while other threads hammer
    // log() on the closed logger.
    #[test]
    fn shutdown_is_not_starved_by_a_hot_loop_of_post_close_callers() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ops.tsv.zst");
        let logger = Logger::open(path.to_str().unwrap(), 64, false).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let hammers: Vec<_> = (0..8)
            .map(|_| {
                let (logger, stop) = (logger.clone(), Arc::clone(&stop));
                thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        logger.log(entry("hot"));
                    }
                })
            })
            .collect();
        thread::sleep(Duration::from_millis(20));
        let started = Instant::now();
        logger.shutdown();
        let took = started.elapsed();
        // Keep hammering the closed logger for a moment after shutdown returned.
        thread::sleep(Duration::from_millis(50));
        stop.store(true, Ordering::Relaxed);
        for h in hammers {
            h.join().unwrap();
        }
        assert!(
            took < Duration::from_secs(5),
            "shutdown took {took:?} under a hot loop of callers"
        );
        read_tags(&path).unwrap();
    }

    fn xorshift(x: &mut u64) -> u64 {
        *x ^= *x << 13;
        *x ^= *x >> 7;
        *x ^= *x << 17;
        *x
    }

    /// One randomised producers-versus-shutdown round on a standalone logger.
    /// `Err` describes the first exactness violation found.
    fn race_round(round: u64, dir: &Path) -> Result<(), String> {
        let mut rng = (round + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let producers = 2 + (xorshift(&mut rng) % 3) as usize;
        let lossless = round & 1 == 0;
        let cap = if lossless {
            [1_usize, 2, 4][(xorshift(&mut rng) % 3) as usize]
        } else {
            [2_usize, 8, 64][(xorshift(&mut rng) % 3) as usize]
        };
        let delay_us = xorshift(&mut rng) % 700;
        let path = dir.join(format!("race-{round}.tsv.zst"));
        let logger = Logger::open(path.to_str().ok_or("non-UTF-8 temp path")?, cap, lossless)
            .map_err(|e| format!("Logger::open: {e}"))?;

        let barrier = Arc::new(Barrier::new(producers + 1));
        let handles: Vec<_> = (0..producers)
            .map(|t| {
                let logger = logger.clone();
                let barrier = Arc::clone(&barrier);
                let seed = xorshift(&mut rng);
                thread::spawn(move || {
                    race_hooks::enable(seed);
                    barrier.wait();
                    let deadline = Instant::now() + Duration::from_secs(10);
                    let mut accepted = Vec::new();
                    let mut problems = Vec::new();
                    for n in 0_u64.. {
                        let tag = format!("t{t}-{n}");
                        match logger.try_log(entry(&tag)) {
                            LogOutcome::Accepted => accepted.push(tag),
                            LogOutcome::Full => {
                                if lossless {
                                    problems.push("Full in lossless mode".to_string());
                                    break;
                                }
                                thread::sleep(Duration::from_micros(50));
                            }
                            LogOutcome::Closed => {
                                // Closed is final: nothing later may be accepted.
                                for _ in 0..3 {
                                    if logger.try_log(entry("late")) != LogOutcome::Closed {
                                        problems.push("outcome after Closed".to_string());
                                    }
                                }
                                return (accepted, problems);
                            }
                            LogOutcome::WriterFailed => {
                                problems.push(format!(
                                    "WriterFailed for {tag} (sent after the sentinel?)"
                                ));
                                break;
                            }
                        }
                        if Instant::now() > deadline {
                            problems.push("never saw Closed".to_string());
                            break;
                        }
                    }
                    (accepted, problems)
                })
            })
            .collect();

        barrier.wait();
        if delay_us > 0 {
            thread::sleep(Duration::from_micros(delay_us));
        }
        logger.shutdown();

        let mut accepted = std::collections::BTreeSet::new();
        let mut accepted_count = 0;
        let mut problems = Vec::new();
        for h in handles {
            match h.join() {
                Ok((a, p)) => {
                    accepted_count += a.len();
                    accepted.extend(a);
                    problems.extend(p);
                }
                Err(_) => problems.push("producer thread panicked".to_string()),
            }
        }
        if !problems.is_empty() {
            return Err(format!("cap={cap} lossless={lossless}: {problems:?}"));
        }
        let gate = logger.shared.gate.load(Ordering::SeqCst);
        if gate != GATE_CLOSED {
            return Err(format!("gate not empty after shutdown: {gate:#x}"));
        }
        let tags = read_tags(&path)?;
        let written: std::collections::BTreeSet<String> = tags.iter().cloned().collect();
        if written.len() != tags.len() {
            return Err("an entry was written twice".to_string());
        }
        if accepted.len() != accepted_count {
            return Err("duplicate tags among accepted entries".to_string());
        }
        let lost: Vec<_> = accepted.difference(&written).take(3).collect();
        let phantom: Vec<_> = written.difference(&accepted).take(3).collect();
        if !lost.is_empty() || !phantom.is_empty() {
            return Err(format!(
                "cap={cap} lossless={lossless} producers={producers}: accepted but not \
                 written {lost:?}; written but not accepted {phantom:?} \
                 ({accepted_count} accepted, {} written)",
                tags.len()
            ));
        }
        std::fs::remove_file(&path).map_err(|e| format!("cleanup: {e}"))?;
        Ok(())
    }

    // Finding 10: the written set must equal the Accepted set no matter where
    // shutdown lands. Scheduling noise (race_hooks) is injected at the two
    // points where log() and shutdown can interleave, which is what lets this
    // fail reliably when the drain protocol is broken (see the commit message
    // for the mutants it kills).
    #[test]
    fn written_set_equals_accepted_set_across_randomised_shutdown_races() {
        const ROUNDS: u64 = 1500;
        let dir = tempfile::tempdir().unwrap();
        let mut failures = Vec::new();
        for round in 0..ROUNDS {
            if let Err(e) = race_round(round, dir.path()) {
                failures.push(format!("round {round}: {e}"));
            }
        }
        println!("race rounds failed: {}/{ROUNDS}", failures.len());
        assert!(
            failures.is_empty(),
            "{}/{ROUNDS} rounds violated exactness; first: {:?}",
            failures.len(),
            failures.iter().take(3).collect::<Vec<_>>()
        );
    }

    /// A sink that accepts everything until armed, then fails every write.
    struct ArmedFailSink(Arc<AtomicBool>);

    impl Write for ArmedFailSink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if self.0.load(Ordering::SeqCst) {
                Err(std::io::Error::other("injected: no space left on device"))
            } else {
                Ok(buf.len())
            }
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// ~400 KB of incompressible-ish text in the `file` column, so that one
    /// entry forces zstd to hand output to the sink.
    fn big_entry(seed: u64) -> LogEntry {
        let mut x = seed | 1;
        let mut file = String::with_capacity(400_000);
        while file.len() < 400_000 {
            file.push_str(&format!("{:016x}", xorshift(&mut x)));
        }
        entry(&file)
    }

    // The closed-bit hint cannot close every window: a caller can read the gate
    // open, lose the CPU, and find it closed only at its fetch_add. That second
    // reject path must count the call, send nothing and release the gate. The test
    // hook runs the close in exactly that window, so the path is taken every time.
    #[test]
    fn a_close_between_the_hint_and_the_gate_is_rejected_at_the_gate() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ops.tsv.zst");
        let logger = Logger::open(path.to_str().unwrap(), 8, false).unwrap();
        assert_eq!(logger.try_log(entry("a")), LogOutcome::Accepted);

        let closer = logger.clone();
        race_hooks::run_once_before_gate(move || closer.shutdown());
        let entered_before = race_hooks::entered();
        assert_eq!(logger.try_log(entry("late")), LogOutcome::Closed);
        assert_eq!(
            race_hooks::entered() - entered_before,
            1,
            "the call must have got past the hint and been refused by the gate"
        );

        let st = logger.stats();
        assert_eq!((st.accepted, st.rejected_closed), (1, 1), "{st:?}");
        assert_eq!(
            logger.shared.gate.load(Ordering::SeqCst),
            GATE_CLOSED,
            "the refused call must give the gate back"
        );
        assert_eq!(read_tags(&path).unwrap(), ["a"], "nothing after the close");
    }

    // Finding 11: after the writer fails with an I/O error, log() reports
    // WriterFailed -- never Full -- in both queue modes, and shutdown returns.
    #[test]
    fn writer_io_error_surfaces_as_writer_failed_never_full() {
        for lossless in [false, true] {
            let armed = Arc::new(AtomicBool::new(false));
            let logger = Logger::start(
                ArmedFailSink(Arc::clone(&armed)),
                1024,
                lossless,
                1,
                "oplog-test",
            )
            .unwrap();
            assert_eq!(logger.try_log(entry("before")), LogOutcome::Accepted);
            armed.store(true, Ordering::SeqCst);

            let mut seen_failed = false;
            for i in 0..200_u64 {
                match logger.try_log(big_entry(i)) {
                    LogOutcome::Accepted => thread::sleep(Duration::from_millis(20)),
                    LogOutcome::WriterFailed => {
                        seen_failed = true;
                        break;
                    }
                    other => panic!("lossless={lossless}: unexpected {other:?} (never Full)"),
                }
            }
            assert!(
                seen_failed,
                "lossless={lossless}: writer I/O error never surfaced"
            );
            for _ in 0..5 {
                assert_eq!(logger.try_log(entry("after")), LogOutcome::WriterFailed);
            }
            logger.shutdown(); // must return even though the writer is gone
            assert_eq!(logger.try_log(entry("closed")), LogOutcome::Closed);
        }
    }
}

// Drop detection (issue #170): `Logger::stats()` must count every outcome of
// `try_log` exactly once, so a caller can tell whether the log is complete.
#[cfg(test)]
mod oplog_stats_tests {
    use super::*;
    use std::sync::Condvar;

    fn entry(tag: &str) -> LogEntry {
        let now = SystemTime::now();
        LogEntry {
            idx: 0,
            thread_id: 0,
            operation: "GET".to_string(),
            client_id: String::new(),
            num_objects: 1,
            bytes: 1,
            endpoint: String::new(),
            file: tag.to_string(),
            error: None,
            start_time: now,
            first_byte_time: None,
            end_time: now,
        }
    }

    /// ~400 KB in the `file` column so one entry forces zstd to hand output to
    /// the sink (and a blocked sink therefore blocks the writer thread).
    fn big_entry(seed: u64) -> LogEntry {
        let mut x = seed | 1;
        let mut file = String::with_capacity(400_000);
        while file.len() < 400_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            file.push_str(&format!("{x:016x}"));
        }
        entry(&file)
    }

    #[derive(Default)]
    struct SinkState {
        /// A write is waiting for `released`.
        blocked: bool,
        released: bool,
        bytes: Vec<u8>,
    }

    /// A sink that stalls its first write until released, then keeps all bytes.
    #[derive(Clone, Default)]
    struct StallingSink(Arc<(Mutex<SinkState>, Condvar)>);

    impl StallingSink {
        fn lock(&self) -> MutexGuard<'_, SinkState> {
            self.0 .0.lock().unwrap_or_else(PoisonError::into_inner)
        }

        fn wait_until_blocked(&self) {
            let mut st = self.lock();
            let deadline = Instant::now() + Duration::from_secs(20);
            while !st.blocked {
                assert!(Instant::now() < deadline, "writer never reached the sink");
                st = self
                    .0
                     .1
                    .wait_timeout(st, Duration::from_millis(10))
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
            }
        }

        fn release(&self) {
            self.lock().released = true;
            self.0 .1.notify_all();
        }

        /// Data rows (header excluded) of the complete zstd stream written so far.
        fn rows(&self) -> usize {
            let text = zstd::decode_all(self.lock().bytes.as_slice()).unwrap();
            String::from_utf8(text).unwrap().lines().count() - 1
        }
    }

    impl Write for StallingSink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            let mut st = self.lock();
            if !st.released {
                st.blocked = true;
                self.0 .1.notify_all();
                while !st.released {
                    st = self
                        .0
                         .1
                        .wait_timeout(st, Duration::from_millis(10))
                        .unwrap_or_else(PoisonError::into_inner)
                        .0;
                }
            }
            st.bytes.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    use std::time::Instant;

    // Tiny queue + a stalled writer: the entries beyond the queue are dropped and
    // `dropped_full` equals the number of `Full` outcomes; `accepted` equals the
    // rows that reach the file.
    #[test]
    fn dropped_full_matches_full_outcomes_and_accepted_matches_rows_written() {
        const CAP: usize = 4;
        const SENT: usize = 25;
        // The header was flushed before the sink stalled, so only release the
        // sink once the writer is blocked inside a write of the big entry.
        let sink = StallingSink::default();
        {
            // Header write happens in `start`, on this thread: let it through.
            sink.lock().released = true;
        }
        let logger = Logger::start(sink.clone(), CAP, false, 1, "oplog-stats").unwrap();
        {
            let mut st = sink.lock();
            st.released = false;
            st.blocked = false;
        }

        assert_eq!(logger.try_log(big_entry(1)), LogOutcome::Accepted);
        sink.wait_until_blocked(); // writer holds the big entry; queue is empty

        let (mut accepted, mut full) = (1_u64, 0_u64);
        for i in 0..SENT {
            match logger.try_log(entry(&format!("e{i}"))) {
                LogOutcome::Accepted => accepted += 1,
                LogOutcome::Full => full += 1,
                other => panic!("unexpected {other:?}"),
            }
        }
        assert_eq!(accepted, 1 + CAP as u64, "queue holds exactly CAP entries");
        assert_eq!(full, (SENT - CAP) as u64);

        let mid = logger.stats();
        assert_eq!(mid.accepted, accepted);
        assert_eq!(mid.dropped_full, full);
        assert_eq!((mid.rejected_closed, mid.writer_failed), (0, 0));
        assert!(!mid.lossless);

        sink.release();
        logger.shutdown();
        let end = logger.stats();
        assert_eq!(
            (
                end.accepted,
                end.dropped_full,
                end.rejected_closed,
                end.writer_failed
            ),
            (
                mid.accepted,
                mid.dropped_full,
                mid.rejected_closed,
                mid.writer_failed
            ),
            "draining must not change the outcome counts"
        );
        assert_eq!(mid.written, 0, "the writer is still inside its first row");
        assert_eq!(
            end.written, end.accepted,
            "draining writes every accepted row"
        );
        assert_eq!(
            sink.rows() as u64,
            end.accepted,
            "accepted must equal the data rows in the file"
        );
    }

    // Closed outcomes are counted, on both reject paths, and never counted as
    // accepted.
    #[test]
    fn rejected_closed_counts_calls_after_finalize() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ops.tsv.zst");
        let logger = Logger::open(path.to_str().unwrap(), 64, false).unwrap();
        for i in 0..3 {
            assert_eq!(
                logger.try_log(entry(&format!("a{i}"))),
                LogOutcome::Accepted
            );
        }
        assert_eq!(logger.stats().rejected_closed, 0);
        logger.shutdown();
        for _ in 0..7 {
            assert_eq!(logger.try_log(entry("late")), LogOutcome::Closed);
        }
        let st = logger.stats();
        assert_eq!(st.accepted, 3);
        assert_eq!(st.rejected_closed, 7);
        assert_eq!((st.dropped_full, st.writer_failed), (0, 0));
    }

    // Lossless mode blocks instead of dropping: a one-slot queue under four
    // producers shows zero drops and every entry accepted.
    #[test]
    fn lossless_mode_reports_zero_drops() {
        const PRODUCERS: usize = 4;
        const EACH: usize = 500;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ops.tsv.zst");
        let logger = Logger::open(path.to_str().unwrap(), 1, true).unwrap();
        let handles: Vec<_> = (0..PRODUCERS)
            .map(|t| {
                let logger = logger.clone();
                thread::spawn(move || {
                    for n in 0..EACH {
                        assert_eq!(
                            logger.try_log(entry(&format!("t{t}-{n}"))),
                            LogOutcome::Accepted
                        );
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        logger.shutdown();
        let st = logger.stats();
        assert!(st.lossless);
        assert_eq!(st.accepted, (PRODUCERS * EACH) as u64);
        assert_eq!(
            (st.dropped_full, st.rejected_closed, st.writer_failed),
            (0, 0, 0)
        );
        let text = zstd::decode_all(std::fs::read(&path).unwrap().as_slice()).unwrap();
        assert_eq!(text.split(|b| *b == b'\n').count() - 2, PRODUCERS * EACH);
    }

    /// A sink that accepts everything until armed, then fails every write.
    struct ArmedFailSink(Arc<AtomicBool>);

    impl Write for ArmedFailSink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if self.0.load(Ordering::SeqCst) {
                Err(std::io::Error::other("injected: no space left on device"))
            } else {
                Ok(buf.len())
            }
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    // After the writer dies, `writer_failed` equals the number of WriterFailed
    // outcomes in both queue modes (the send-error path in lossless mode, the
    // disconnected path otherwise).
    #[test]
    fn writer_failed_matches_writer_failed_outcomes_in_both_modes() {
        for lossless in [false, true] {
            let armed = Arc::new(AtomicBool::new(false));
            let logger = Logger::start(
                ArmedFailSink(Arc::clone(&armed)),
                1024,
                lossless,
                1,
                "oplog-stats-f",
            )
            .unwrap();
            assert_eq!(logger.try_log(entry("before")), LogOutcome::Accepted);
            armed.store(true, Ordering::SeqCst);

            let (mut accepted, mut failed) = (1_u64, 0_u64);
            for i in 0..200_u64 {
                match logger.try_log(big_entry(i)) {
                    LogOutcome::Accepted => {
                        accepted += 1;
                        thread::sleep(Duration::from_millis(20));
                    }
                    LogOutcome::WriterFailed => {
                        failed += 1;
                        break;
                    }
                    other => panic!("lossless={lossless}: unexpected {other:?}"),
                }
            }
            assert_eq!(failed, 1, "lossless={lossless}: writer failure never seen");
            for _ in 0..5 {
                assert_eq!(logger.try_log(entry("after")), LogOutcome::WriterFailed);
                failed += 1;
            }
            let st = logger.stats();
            assert_eq!(st.writer_failed, failed, "lossless={lossless}");
            assert_eq!(st.accepted, accepted, "lossless={lossless}");
            assert_eq!(st.dropped_full, 0, "lossless={lossless}");
            logger.shutdown();
        }
    }

    // Under a shutdown race each outcome the producers observed is counted
    // exactly once, whichever reject path (closed hint or gate) saw it.
    #[test]
    fn counters_equal_outcomes_observed_by_racing_producers() {
        for round in 0..200_u64 {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("ops.tsv.zst");
            let cap = [1_usize, 2, 8][(round % 3) as usize];
            let logger = Logger::open(path.to_str().unwrap(), cap, false).unwrap();
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    let logger = logger.clone();
                    thread::spawn(move || {
                        let mut seen = [0_u64; 3]; // accepted, full, closed
                        let mut extra_closed = 0;
                        loop {
                            match logger.try_log(entry("x")) {
                                LogOutcome::Accepted => seen[0] += 1,
                                LogOutcome::Full => seen[1] += 1,
                                LogOutcome::Closed => {
                                    seen[2] += 1;
                                    extra_closed += 1;
                                    if extra_closed == 3 {
                                        return seen;
                                    }
                                }
                                LogOutcome::WriterFailed => panic!("writer failed"),
                            }
                        }
                    })
                })
                .collect();
            thread::sleep(Duration::from_micros(100 * (round % 7)));
            logger.shutdown();
            let mut total = [0_u64; 3];
            for h in handles {
                for (t, n) in total.iter_mut().zip(h.join().unwrap()) {
                    *t += n;
                }
            }
            let st = logger.stats();
            assert_eq!(
                [st.accepted, st.dropped_full, st.rejected_closed],
                total,
                "round {round}"
            );
        }
    }

    // ---- the note lines on the log() hot path must never panic (review P3-3) ----

    const CHILD_ENV: &str = "S3DLIO_TEST_DEAD_STDERR_CHILD";
    const CHILD_OK: &str = "dead-stderr child finished";

    /// Run `body` in a child copy of this test binary whose stderr is `/dev/full`,
    /// where every write fails (a closed or broken stderr). `eprintln!` panics on
    /// such a write; the op-log's own notes must not, because they run inside the
    /// caller's I/O path. `test` is the child's test path within this crate.
    #[cfg(target_os = "linux")]
    fn with_dead_stderr(test: &str, body: impl FnOnce()) {
        if std::env::var_os(CHILD_ENV).is_some() {
            body();
            println!("{CHILD_OK}");
            return;
        }
        let dead = File::options().write(true).open("/dev/full").unwrap();
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", test, "--nocapture", "--test-threads=1"])
            .env(CHILD_ENV, "1")
            .stderr(std::process::Stdio::from(dead))
            .output()
            .unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout);
        assert!(
            out.status.success() && stdout.contains(CHILD_OK),
            "the child died or never ran its body ({}): {stdout}",
            out.status
        );
    }

    /// `module_path!()` without the crate name: the path libtest filters on.
    fn test_path(name: &str) -> String {
        let module = module_path!();
        let (_, module) = module.split_once("::").unwrap();
        format!("{module}::{name}")
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn closed_note_survives_a_dead_stderr() {
        with_dead_stderr(&test_path("closed_note_survives_a_dead_stderr"), || {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("ops.tsv.zst");
            let logger = Logger::open(path.to_str().unwrap(), 8, false).unwrap();
            logger.shutdown();
            // The first call after the close prints the one "finalized" note.
            assert_eq!(logger.try_log(entry("late")), LogOutcome::Closed);
            assert_eq!(logger.try_log(entry("later")), LogOutcome::Closed);
            assert_eq!(logger.stats().rejected_closed, 2);
        });
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn full_note_survives_a_dead_stderr() {
        with_dead_stderr(&test_path("full_note_survives_a_dead_stderr"), || {
            let sink = StallingSink::default();
            sink.lock().released = true; // the header is written by `start` itself
            let logger = Logger::start(sink.clone(), 1, false, 1, "oplog-dse-f").unwrap();
            {
                let mut st = sink.lock();
                st.released = false;
                st.blocked = false;
            }
            assert_eq!(logger.try_log(big_entry(1)), LogOutcome::Accepted);
            sink.wait_until_blocked(); // the writer holds the big entry; the queue is empty
            assert_eq!(logger.try_log(entry("queued")), LogOutcome::Accepted);
            // Every one of these prints a "channel full" line.
            for i in 0..3 {
                assert_eq!(logger.try_log(entry(&format!("d{i}"))), LogOutcome::Full);
            }
            sink.release();
            logger.shutdown();
            assert_eq!(logger.stats().dropped_full, 3);
        });
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn writer_stopped_note_survives_a_dead_stderr() {
        with_dead_stderr(
            &test_path("writer_stopped_note_survives_a_dead_stderr"),
            || {
                let armed = Arc::new(AtomicBool::new(false));
                let logger = Logger::start(
                    ArmedFailSink(Arc::clone(&armed)),
                    64,
                    false,
                    1,
                    "oplog-dse-w",
                )
                .unwrap();
                armed.store(true, Ordering::SeqCst);
                let mut seen_failed = false;
                for i in 0..200_u64 {
                    match logger.try_log(big_entry(i)) {
                        LogOutcome::Accepted => thread::sleep(Duration::from_millis(20)),
                        LogOutcome::WriterFailed => {
                            seen_failed = true;
                            break;
                        }
                        other => panic!("unexpected {other:?}"),
                    }
                }
                assert!(seen_failed, "the writer failure never surfaced");
                logger.shutdown();
            },
        );
    }

    // ---- a panic inside shutdown must not let a later finalize return early (P3-2) ----

    /// A tracing subscriber that panics on every event (what a faulty or hostile
    /// `tracing` layer can do inside `info!`). Installed per thread only.
    struct PanickingSubscriber;

    impl tracing::Subscriber for PanickingSubscriber {
        fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
            true
        }
        fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }
        fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
        fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
        fn event(&self, _: &tracing::Event<'_>) {
            panic!("injected: tracing subscriber panicked");
        }
        fn enter(&self, _: &tracing::span::Id) {}
        fn exit(&self, _: &tracing::span::Id) {}
    }

    // The first shutdown panics (in the tracing call it makes) while the writer is
    // still running. Before the fix it had already taken the completion receiver
    // out of the mutex, so the panic lost it and every later finalize returned at
    // once, before the writer was done, and the incomplete-log warning never ran.
    #[test]
    fn a_panic_in_the_first_shutdown_does_not_let_a_later_one_return_early() {
        let sink = StallingSink::default();
        sink.lock().released = true; // the header is written by `start` itself
        let logger = Logger::start(sink.clone(), 8, false, 1, "oplog-ps").unwrap();
        {
            let mut st = sink.lock();
            st.released = false;
            st.blocked = false;
        }
        assert_eq!(logger.try_log(big_entry(1)), LogOutcome::Accepted);
        sink.wait_until_blocked(); // the writer is stuck inside the sink

        let first = logger.clone();
        let outcome = thread::spawn(move || {
            tracing::subscriber::with_default(PanickingSubscriber, || {
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| first.shutdown()))
            })
        })
        .join()
        .unwrap();
        assert!(
            outcome.is_err(),
            "the injected panic never reached shutdown"
        );

        let (tx, rx) = channel::<()>();
        let second = logger.clone();
        thread::spawn(move || {
            second.shutdown();
            let _ = tx.send(());
        });
        assert!(
            rx.recv_timeout(Duration::from_millis(300)).is_err(),
            "a later shutdown returned while the writer was still running"
        );

        sink.release();
        rx.recv_timeout(Duration::from_secs(30))
            .unwrap_or_else(|_| panic!("the later shutdown never returned"));
        let st = logger.stats();
        assert_eq!((st.accepted, st.written), (1, 1), "{st:?}");
        assert!(st.is_complete(), "{st:?}");
        // A third call finds the session already finished.
        logger.shutdown();
        assert_eq!(sink.rows(), 1);
    }
}

// Writer-side loss and end-of-stream errors (issue #170 review, finding F1):
// entries the writer accepted but never wrote, and errors that happen after the
// last row (zstd finish, flush), a writer panic, must all show in the stats.
#[cfg(test)]
mod oplog_writer_error_tests {
    use super::*;
    use std::io::Read;
    use std::sync::Condvar;
    use std::time::Instant;

    fn entry_at(tag: &str, when: SystemTime) -> LogEntry {
        LogEntry {
            idx: 0,
            thread_id: 0,
            operation: "GET".to_string(),
            client_id: String::new(),
            num_objects: 1,
            bytes: 1,
            endpoint: String::new(),
            file: tag.to_string(),
            error: None,
            start_time: when,
            first_byte_time: None,
            end_time: when,
        }
    }

    fn entry(tag: &str) -> LogEntry {
        entry_at(tag, SystemTime::now())
    }

    /// ~400 KB of hex digits in the `file` column: one entry makes zstd hand
    /// several blocks to the sink.
    fn big_entry(seed: u64) -> LogEntry {
        let mut x = seed | 1;
        let mut file = String::with_capacity(400_000);
        while file.len() < 400_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            file.push_str(&format!("{x:016x}"));
        }
        entry(&file)
    }

    #[derive(Default)]
    struct QuotaState {
        /// A write is waiting for `released`.
        blocked: bool,
        released: bool,
        bytes: Vec<u8>,
    }

    /// A sink that stalls its writes until released, then keeps up to `quota`
    /// bytes and fails every write that would exceed it (a disk that fills up).
    #[derive(Clone)]
    struct QuotaSink {
        state: Arc<(Mutex<QuotaState>, Condvar)>,
        quota: usize,
    }

    impl QuotaSink {
        fn new(quota: usize) -> Self {
            Self {
                state: Arc::default(),
                quota,
            }
        }

        fn lock(&self) -> MutexGuard<'_, QuotaState> {
            self.state.0.lock().unwrap_or_else(PoisonError::into_inner)
        }

        fn wait_until_blocked(&self) {
            let mut st = self.lock();
            let deadline = Instant::now() + Duration::from_secs(20);
            while !st.blocked {
                assert!(Instant::now() < deadline, "writer never reached the sink");
                st = self
                    .state
                    .1
                    .wait_timeout(st, Duration::from_millis(10))
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
            }
        }

        fn release(&self) {
            self.lock().released = true;
            self.state.1.notify_all();
        }

        fn bytes(&self) -> Vec<u8> {
            self.lock().bytes.clone()
        }
    }

    impl Write for QuotaSink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            let mut st = self.lock();
            if !st.released {
                st.blocked = true;
                self.state.1.notify_all();
                while !st.released {
                    st = self
                        .state
                        .1
                        .wait_timeout(st, Duration::from_millis(10))
                        .unwrap_or_else(PoisonError::into_inner)
                        .0;
                }
            }
            if st.bytes.len() + buf.len() > self.quota {
                return Err(std::io::Error::other("injected: no space left on device"));
            }
            st.bytes.extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// A sink that accepts everything until a failure is armed.
    #[derive(Clone, Default)]
    struct SwitchSink {
        fail_write: Arc<AtomicBool>,
        fail_flush: Arc<AtomicBool>,
        bytes: Arc<Mutex<Vec<u8>>>,
    }

    impl SwitchSink {
        fn bytes(&self) -> Vec<u8> {
            self.bytes
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone()
        }
    }

    impl Write for SwitchSink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if self.fail_write.load(Ordering::SeqCst) {
                return Err(std::io::Error::other("injected: write failed"));
            }
            self.bytes
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            if self.fail_flush.load(Ordering::SeqCst) {
                Err(std::io::Error::other("injected: flush failed"))
            } else {
                Ok(())
            }
        }
    }

    /// Complete rows of the readable prefix of a possibly truncated zstd stream
    /// (header excluded). Each complete row must have 13 columns and the next idx.
    fn readable_rows(bytes: &[u8]) -> u64 {
        let mut decoder = zstd::stream::read::Decoder::new(bytes).unwrap();
        let (mut text, mut buf) = (Vec::new(), [0_u8; 65536]);
        // Stops at the end of the stream or at the first error (a truncated stream).
        while let Ok(n @ 1..) = decoder.read(&mut buf) {
            text.extend_from_slice(&buf[..n]);
        }
        let text = String::from_utf8_lossy(&text);
        // Only newline-terminated lines are complete rows.
        let complete = text.rfind('\n').map_or("", |last| &text[..=last]);
        let mut lines = complete.lines();
        assert_eq!(lines.next().map(|h| h.starts_with("idx\t")), Some(true));
        let mut rows = 0;
        for line in lines {
            let cols: Vec<&str> = line.split('\t').collect();
            assert_eq!(cols.len(), 13, "row {rows}");
            assert_eq!(cols[0], rows.to_string(), "row {rows} idx");
            rows += 1;
        }
        rows
    }

    fn stats(accepted: u64, written: u64) -> OpLogStats {
        OpLogStats {
            accepted,
            dropped_full: 0,
            rejected_closed: 0,
            writer_failed: 0,
            lossless: false,
            written,
            writer_error: None,
        }
    }

    // THE bug: the writer fails on a full disk with many entries queued. Every
    // one of them was Accepted, no later call is refused, and so the counters
    // that existed before (`dropped_full`, `writer_failed`) all read zero --
    // yet most of the log is gone. `written < accepted` and `writer_error` say so.
    //
    // What the destination holds: a zstd stream cut at the failure point. It has
    // no frame end (a strict decoder reports the truncation), its readable
    // prefix is the header plus R complete rows with sequential idx, and
    // R <= written (the encoder buffers, so rows handed to it are not all out).
    #[test]
    fn writer_failure_with_entries_queued_shows_in_the_stats() {
        const QUEUED: u64 = 20;
        for lossless in [false, true] {
            let sink = QuotaSink::new(600_000);
            sink.lock().released = true; // the header is written by `start` itself
            let logger = Logger::start(sink.clone(), 64, lossless, 1, "oplog-wf").unwrap();
            {
                let mut st = sink.lock();
                st.released = false;
                st.blocked = false;
            }
            assert_eq!(logger.try_log(big_entry(0)), LogOutcome::Accepted);
            sink.wait_until_blocked(); // the writer holds row 0 inside the sink
            for i in 1..QUEUED {
                assert_eq!(logger.try_log(big_entry(i)), LogOutcome::Accepted);
            }
            sink.release();
            logger.shutdown();

            let st = logger.stats();
            let ctx = format!("lossless={lossless}: {st:?}");
            assert_eq!(st.accepted, QUEUED, "{ctx}");
            // The pre-existing counters cannot see this loss:
            assert_eq!(
                (st.dropped_full, st.rejected_closed, st.writer_failed),
                (0, 0, 0),
                "{ctx}"
            );
            assert!(st.written >= 1 && st.written < st.accepted, "{ctx}");
            let error = st.writer_error.as_deref().unwrap_or_default();
            // The FIRST error (the failed write), not the later finish attempt's.
            assert!(
                error.starts_with("write failed") && error.contains("injected: no space left"),
                "{ctx}"
            );
            assert!(!st.is_complete(), "{ctx}");

            let warning = incomplete_warning(&st).unwrap_or_else(|| panic!("no warning: {ctx}"));
            let lost = QUEUED - st.written;
            assert!(
                warning.contains(&format!("{lost} accepted entries were never written")),
                "{warning}"
            );
            assert!(warning.contains("injected: no space left"), "{warning}");

            let bytes = sink.bytes();
            assert!(
                zstd::decode_all(bytes.as_slice()).is_err(),
                "{ctx}: the stream must be visibly truncated"
            );
            assert!(readable_rows(&bytes) <= st.written, "{ctx}");
        }
    }

    // The error only appears when the stream is finished: the rows were all
    // accepted and handed to the encoder (written == accepted), the zstd frame
    // end cannot be written. Before the fix this was swallowed by the encoder's
    // drop.
    #[test]
    fn zstd_finish_failure_is_reported() {
        let sink = SwitchSink::default();
        let logger = Logger::start(sink.clone(), 64, false, 1, "oplog-ff").unwrap();
        for i in 0..5 {
            assert_eq!(
                logger.try_log(entry(&format!("r{i}"))),
                LogOutcome::Accepted
            );
        }
        // Five tiny rows stay inside the encoder: only finishing reaches the sink.
        sink.fail_write.store(true, Ordering::SeqCst);
        logger.shutdown();

        let st = logger.stats();
        assert_eq!((st.accepted, st.written), (5, 5), "{st:?}");
        let error = st.writer_error.as_deref().unwrap_or_default();
        assert!(
            error.contains("finish") && error.contains("injected: write failed"),
            "{st:?}"
        );
        assert!(!st.is_complete());
        assert!(incomplete_warning(&st).is_some());
        assert!(
            zstd::decode_all(sink.bytes().as_slice()).is_err(),
            "no frame end reached the sink"
        );
    }

    // The final flush of the buffered sink (BufWriter in production) failing:
    // every byte was handed over, but it may not be on disk.
    #[test]
    fn final_flush_failure_is_reported() {
        let sink = SwitchSink::default();
        let logger = Logger::start(sink.clone(), 64, false, 1, "oplog-fl").unwrap();
        for i in 0..5 {
            assert_eq!(
                logger.try_log(entry(&format!("r{i}"))),
                LogOutcome::Accepted
            );
        }
        sink.fail_flush.store(true, Ordering::SeqCst);
        logger.shutdown();

        let st = logger.stats();
        assert_eq!((st.accepted, st.written), (5, 5), "{st:?}");
        let error = st.writer_error.as_deref().unwrap_or_default();
        assert!(
            error.contains("flush") && error.contains("injected: flush failed"),
            "{st:?}"
        );
        assert!(!st.is_complete());
        assert!(incomplete_warning(&st).is_some());
        // Only the flush failed: the bytes that reached the sink are a whole stream.
        assert_eq!(readable_rows(&sink.bytes()), 5);
    }

    // A panic in the writer thread (here: a time pushed before the epoch makes
    // humantime's formatter panic, a recorded pre-existing limit) must not hang
    // shutdown, and must show up as an error. The stream is still finished, so
    // the rows before the panic are readable.
    #[test]
    fn writer_panic_is_reported_and_shutdown_does_not_hang() {
        for lossless in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let path = dir.path().join("ops.tsv.zst");
            let logger = Logger::open(path.to_str().unwrap(), 8, lossless).unwrap();
            logger.set_clock_offset(2_000_000_000);
            assert_eq!(logger.try_log(entry("fine")), LogOutcome::Accepted);
            let pre_epoch = entry_at("boom", std::time::UNIX_EPOCH + Duration::from_secs(1));
            assert_eq!(logger.try_log(pre_epoch), LogOutcome::Accepted);
            for _ in 0..50 {
                match logger.try_log(entry("later")) {
                    // Full: the queue can fill before the writer reaches the bad row.
                    LogOutcome::Accepted | LogOutcome::WriterFailed => {}
                    LogOutcome::Full if !lossless => {}
                    other => panic!("lossless={lossless}: unexpected {other:?}"),
                }
            }

            let (done_tx, done_rx) = channel::<()>();
            let closing = logger.clone();
            thread::spawn(move || {
                closing.shutdown();
                let _ = done_tx.send(());
            });
            done_rx
                .recv_timeout(Duration::from_secs(30))
                .unwrap_or_else(|_| panic!("lossless={lossless}: shutdown hung after a panic"));

            let st = logger.stats();
            let error = st.writer_error.as_deref().unwrap_or_default();
            assert!(
                error.contains("panicked") && error.contains("after the epoch"),
                "lossless={lossless}: {st:?}"
            );
            assert_eq!(st.written, 1, "only the row before the panic: {st:?}");
            assert!(st.accepted >= 2, "{st:?}");
            assert!(!st.is_complete());
            assert!(incomplete_warning(&st).is_some());
            let raw = std::fs::read(&path).unwrap();
            assert_eq!(readable_rows(&raw), 1);
            assert!(
                zstd::decode_all(raw.as_slice()).is_ok(),
                "the stream is finished after a writer panic"
            );
        }
    }

    // If the writer thread vanishes without its completion message (a panic that
    // escaped), shutdown records that instead of ignoring the closed channel.
    #[test]
    fn missing_completion_message_is_recorded() {
        let report = WriterReport::default();
        let (done_tx, done_rx) = channel::<()>();
        drop(done_tx);
        wait_for_writer(&done_rx, &report);
        let recorded = report.first_error.get().map(String::as_str);
        assert!(
            recorded.is_some_and(|e| e.contains("without completing")),
            "{recorded:?}"
        );

        // A message from the writer is not an error, and the first error stays.
        let report = WriterReport::default();
        let (done_tx, done_rx) = channel::<()>();
        done_tx.send(()).unwrap();
        wait_for_writer(&done_rx, &report);
        assert_eq!(report.first_error.get(), None);
        report.record_error("write", "first");
        report.record_error("flush", "second");
        assert_eq!(
            report.first_error.get().map(String::as_str),
            Some("write: first")
        );
    }

    /// Call `logger.shutdown()` on another thread and fail the test (instead of
    /// hanging it) if it has not returned within 30 seconds.
    fn shutdown_within_30s(logger: &Logger, what: &str) {
        let (done_tx, done_rx) = channel::<()>();
        let closing = logger.clone();
        thread::spawn(move || {
            closing.shutdown();
            let _ = done_tx.send(());
        });
        done_rx
            .recv_timeout(Duration::from_secs(30))
            .unwrap_or_else(|_| panic!("{what}: shutdown hung"));
    }

    /// A sink whose writes panic once armed: a panic seam that does not depend on
    /// any other crate's behaviour.
    #[derive(Clone, Default)]
    struct PanicSink {
        armed: Arc<AtomicBool>,
    }

    impl Write for PanicSink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            assert!(
                !self.armed.load(Ordering::SeqCst),
                "injected: sink write panicked"
            );
            Ok(buf.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    // A panic while WRITING ROWS (here: the sink panics when the first big row
    // reaches it). Does not depend on humantime or on any time arithmetic. The
    // first error is the row phase's; the stream is still finished afterwards.
    #[test]
    fn a_panic_in_the_row_loop_is_reported_and_shutdown_does_not_hang() {
        for lossless in [false, true] {
            let sink = PanicSink::default();
            let logger = Logger::start(sink.clone(), 8, lossless, 1, "oplog-pr").unwrap();
            sink.armed.store(true, Ordering::SeqCst);
            assert_eq!(logger.try_log(big_entry(1)), LogOutcome::Accepted);
            for _ in 0..20 {
                match logger.try_log(entry("later")) {
                    LogOutcome::Accepted | LogOutcome::WriterFailed => {}
                    LogOutcome::Full if !lossless => {}
                    other => panic!("lossless={lossless}: unexpected {other:?}"),
                }
            }
            shutdown_within_30s(&logger, &format!("lossless={lossless}"));

            let st = logger.stats();
            let error = st.writer_error.as_deref().unwrap_or_default();
            assert!(
                error.starts_with("writer thread panicked:")
                    && error.contains("injected: sink write panicked"),
                "lossless={lossless}: {st:?}"
            );
            assert!(
                !error.contains("while finishing"),
                "the FIRST error is the row phase's: {st:?}"
            );
            assert_eq!(st.written, 0, "{st:?}");
            assert!(!st.is_complete());
            let warning = incomplete_warning(&st).unwrap();
            assert!(
                warning.contains("panicked") && !warning.contains("I/O errors"),
                "a panic is not an I/O error: {warning}"
            );
        }
    }

    // A panic while FINISHING the stream: the rows were all written (they sit in
    // the encoder), the frame end is what reaches the sink and panics. The second
    // catch_unwind in the writer records it; without it the panic would escape the
    // thread and surface only as a vanished writer.
    #[test]
    fn a_panic_while_finishing_the_stream_is_reported_as_such() {
        let sink = PanicSink::default();
        let logger = Logger::start(sink.clone(), 64, false, 1, "oplog-pf").unwrap();
        for i in 0..3 {
            assert_eq!(
                logger.try_log(entry(&format!("r{i}"))),
                LogOutcome::Accepted
            );
        }
        // Three tiny rows stay inside the encoder: only finishing reaches the sink.
        sink.armed.store(true, Ordering::SeqCst);
        shutdown_within_30s(&logger, "finish phase");

        let st = logger.stats();
        assert_eq!((st.accepted, st.written), (3, 3), "{st:?}");
        let error = st.writer_error.as_deref().unwrap_or_default();
        assert!(
            error.starts_with("writer thread panicked while finishing:")
                && error.contains("injected: sink write panicked"),
            "{st:?}"
        );
        assert!(!st.is_complete());
        let warning = incomplete_warning(&st).unwrap();
        assert!(
            warning.contains("panicked") && !warning.contains("I/O errors"),
            "a panic is not an I/O error: {warning}"
        );
    }

    /// A writer body that returns at once without sending its completion message
    /// (as a thread does that dies from a panic nobody caught).
    fn vanishing_writer(
        receiver: Receiver<Message>,
        encoder: StreamEncoder<SwitchSink>,
        _: &AtomicI64,
        _: &WriterReport,
        _: &std::sync::mpsc::Sender<()>,
    ) {
        drop(receiver);
        drop(encoder);
    }

    // The real shutdown, not just `wait_for_writer` on its own: when the writer's
    // completion channel closes without a message, shutdown must record it. A
    // shutdown that ignored the closed channel would leave the stats clean.
    #[test]
    fn shutdown_records_a_writer_that_vanished_without_completing() {
        let logger = Logger::start_with(
            SwitchSink::default(),
            8,
            false,
            1,
            "oplog-vanish",
            vanishing_writer,
        )
        .unwrap();
        shutdown_within_30s(&logger, "vanished writer");

        let st = logger.stats();
        let error = st.writer_error.as_deref().unwrap_or_default();
        assert!(error.contains("without completing"), "{st:?}");
        assert!(!st.is_complete());
        let warning = incomplete_warning(&st).unwrap();
        assert!(
            warning.contains("ended abnormally") && !warning.contains("I/O errors"),
            "{warning}"
        );
    }

    // The counters are separate relaxed atomics, so a snapshot read while the
    // session is still Active can show more rows written than accepted. That is
    // not complete either, and the warning must say what is wrong rather than print
    // an empty bullet.
    #[test]
    fn written_above_accepted_is_named_not_an_empty_bullet() {
        let st = stats(5, 7);
        assert!(!st.is_complete());
        let w = incomplete_warning(&st).unwrap();
        assert!(w.contains("7 rows written but only 5 accepted"), "{w}");
        for bullet in w.lines().filter(|l| l.trim_start().starts_with('-')) {
            assert!(
                !bullet.trim_start_matches([' ', '-']).trim().is_empty(),
                "empty bullet in: {w}"
            );
        }
    }

    // Loss comes from a writer I/O error OR a writer panic; the advice must follow
    // the actual error, not always send the reader to the disk.
    #[test]
    fn warning_advice_follows_the_kind_of_writer_failure() {
        let mut panicked = stats(10, 1);
        panicked.writer_error = Some("writer thread panicked: boom".to_string());
        let w = incomplete_warning(&panicked).unwrap();
        assert!(w.contains("writer thread panicked: boom"), "{w}");
        assert!(w.contains("not an I/O error"), "{w}");
        assert!(!w.contains("I/O errors (full disk"), "{w}");

        let mut finishing = stats(10, 10);
        finishing.writer_error = Some("writer thread panicked while finishing: boom".to_string());
        let w = incomplete_warning(&finishing).unwrap();
        assert!(w.contains("not an I/O error"), "{w}");

        let mut io = stats(10, 1);
        io.writer_error = Some("write failed: no space left on device".to_string());
        let w = incomplete_warning(&io).unwrap();
        assert!(w.contains("I/O errors (full disk"), "{w}");
        assert!(!w.contains("not an I/O error"), "{w}");
    }

    // A healthy session: nothing to report, byte-for-byte what it always wrote.
    #[test]
    fn healthy_session_is_complete() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ops.tsv.zst");
        let logger = Logger::open(path.to_str().unwrap(), 64, false).unwrap();
        for i in 0..7 {
            assert_eq!(
                logger.try_log(entry(&format!("r{i}"))),
                LogOutcome::Accepted
            );
        }
        logger.shutdown();
        let st = logger.stats();
        assert_eq!((st.accepted, st.written), (7, 7));
        assert_eq!(st.writer_error, None);
        assert!(st.is_complete());
        assert_eq!(incomplete_warning(&st), None);
        assert_eq!(
            readable_rows(&std::fs::read(&path).unwrap()),
            7,
            "all rows in the file"
        );
    }

    // "Complete" is exactly: accepted == written, no drops, no refused entries,
    // no writer error. Closed calls are separate.
    #[test]
    fn is_complete_definition() {
        assert!(stats(10, 10).is_complete());
        let mut st = stats(10, 10);
        st.rejected_closed = 99;
        assert!(st.is_complete(), "rejected_closed is not part of complete");
        assert!(!stats(10, 9).is_complete());
        let mut st = stats(10, 10);
        st.dropped_full = 1;
        assert!(!st.is_complete());
        let mut st = stats(10, 10);
        st.writer_failed = 1;
        assert!(!st.is_complete());
        let mut st = stats(10, 10);
        st.writer_error = Some("x".to_string());
        assert!(!st.is_complete());
    }

    // The warning is tailored: only the non-zero counts, and only the advice that
    // applies to the mode.
    #[test]
    fn warning_text_is_tailored_to_mode_and_counts() {
        assert_eq!(incomplete_warning(&stats(100, 100)), None);
        let mut closed_only = stats(100, 100);
        closed_only.rejected_closed = 5;
        assert_eq!(incomplete_warning(&closed_only), None);

        // Lossy mode, queue-full drops only: the remedies, nothing else.
        let mut drops = stats(100, 100);
        drops.dropped_full = 7;
        let w = incomplete_warning(&drops).unwrap();
        assert!(w.contains("op-log is INCOMPLETE"), "{w}");
        assert!(
            w.contains("7 entries were dropped because the queue was full"),
            "{w}"
        );
        assert!(w.contains("S3DLIO_OPLOG_LOSSLESS=1") && w.contains("S3DLIO_OPLOG_BUF"));
        assert!(w.contains("100 accepted, 100 written"), "{w}");
        for absent in ["never written", "writer failed", "refused", "I/O"] {
            assert!(!w.contains(absent), "unexpected '{absent}' in: {w}");
        }

        // Lossless mode, writer error with queued entries lost: no LOSSLESS
        // advice and no queue advice, no "dropped" line, the error is quoted.
        let mut lost = stats(100, 40);
        lost.lossless = true;
        lost.writer_error = Some("write: no space left on device".to_string());
        let w = incomplete_warning(&lost).unwrap();
        assert!(w.contains("60 accepted entries were never written"), "{w}");
        assert!(w.contains("no space left on device"), "{w}");
        assert!(w.contains("destination"), "{w}");
        for absent in ["LOSSLESS", "S3DLIO_OPLOG_BUF", "dropped", "refused"] {
            assert!(!w.contains(absent), "unexpected '{absent}' in: {w}");
        }

        // Refused-after-the-writer-stopped and late (closed) calls are named
        // only when non-zero.
        let mut refused = stats(10, 10);
        refused.writer_failed = 3;
        refused.rejected_closed = 2;
        let w = incomplete_warning(&refused).unwrap();
        assert!(w.contains("3 entries were refused"), "{w}");
        assert!(w.contains("2 calls arrived after finalize began"), "{w}");
        assert!(!w.contains("dropped"), "{w}");
    }
}

// Golden test: pins the serializer's output (header plus rows, written out as
// literal strings, not derived from HEADER or to_log_line) for THREE fixed
// entries at clock offset 0: a GET with a first-byte time (caller idx 777 is
// replaced by 0); a PUT with an error text and no first-byte time; a LIST with
// an empty client id, 25 objects, 123456789012 bytes and a first-byte time 1 ns
// after the start. A renamed or reordered column, a changed timestamp layout or
// a wrong idx fails here. It does NOT establish what any operation puts in its
// fields, when its times are taken, other clock offsets (see
// oplog_clock_offset_tests) or that any consumer parses the output.
#[cfg(test)]
mod oplog_golden_format_tests {
    use super::*;
    use std::time::UNIX_EPOCH;

    /// 2023-11-14T22:13:20.123456789Z
    fn base() -> SystemTime {
        UNIX_EPOCH + Duration::new(1_700_000_000, 123_456_789)
    }

    fn entries() -> Vec<LogEntry> {
        vec![
            // Success row with a first-byte time. The caller-supplied idx (777)
            // must be ignored: the writer numbers rows 0..N-1.
            LogEntry {
                idx: 777,
                thread_id: 7,
                operation: "GET".to_string(),
                client_id: "client-a".to_string(),
                num_objects: 1,
                bytes: 4096,
                endpoint: "s3://bucket".to_string(),
                file: "obj/key0".to_string(),
                error: None,
                start_time: base(),
                first_byte_time: Some(base() + Duration::from_nanos(1_500_000)),
                end_time: base() + Duration::from_nanos(4_000_000),
            },
            // Error row without a first-byte time.
            LogEntry {
                idx: 0,
                thread_id: 8,
                operation: "PUT".to_string(),
                client_id: "client-b".to_string(),
                num_objects: 1,
                bytes: 0,
                endpoint: "file://".to_string(),
                file: "dir/obj1".to_string(),
                error: Some("403 Forbidden: access denied".to_string()),
                start_time: base() + Duration::from_secs(10),
                first_byte_time: None,
                end_time: base() + Duration::from_secs(10) + Duration::from_nanos(250_000),
            },
            // Empty client_id, many objects, first-byte time present.
            LogEntry {
                idx: 5,
                thread_id: 0,
                operation: "LIST".to_string(),
                client_id: String::new(),
                num_objects: 25,
                bytes: 123_456_789_012,
                endpoint: "az://container".to_string(),
                file: "prefix/".to_string(),
                error: None,
                start_time: base() + Duration::from_secs(20),
                first_byte_time: Some(base() + Duration::from_secs(20) + Duration::from_nanos(1)),
                end_time: base() + Duration::from_secs(21),
            },
        ]
    }

    const EXPECTED_HEADER: &str = "idx\tthread\top\tclient_id\tn_objects\tbytes\tendpoint\tfile\terror\tstart\tfirst_byte\tend\tduration_ns\n";
    const EXPECTED_ROWS: [&str; 3] = [
        "0\t7\tGET\tclient-a\t1\t4096\ts3://bucket\tobj/key0\t\t2023-11-14T22:13:20.123456789Z\t2023-11-14T22:13:20.124956789Z\t2023-11-14T22:13:20.127456789Z\t4000000\n",
        "1\t8\tPUT\tclient-b\t1\t0\tfile://\tdir/obj1\t403 Forbidden: access denied\t2023-11-14T22:13:30.123456789Z\t\t2023-11-14T22:13:30.123706789Z\t250000\n",
        "2\t0\tLIST\t\t25\t123456789012\taz://container\tprefix/\t\t2023-11-14T22:13:40.123456789Z\t2023-11-14T22:13:40.123456790Z\t2023-11-14T22:13:41.123456789Z\t1000000000\n",
    ];

    /// Write `entries()` through a standalone Logger at `destination` and return
    /// the raw, decompressed bytes of the resulting file at `local`.
    fn decompressed_after_logging(destination: &str, local: &std::path::Path) -> Vec<u8> {
        // Explicit capacity: the S3DLIO_OPLOG_BUF / S3DLIO_OPLOG_LOSSLESS
        // environment must not be able to change this test.
        let logger = Logger::open(destination, 64, false).unwrap();
        for e in entries() {
            assert_eq!(logger.try_log(e), LogOutcome::Accepted);
        }
        logger.shutdown();
        zstd::decode_all(std::fs::read(local).unwrap().as_slice()).unwrap()
    }

    // A plain path and the equivalent file:// URIs must produce byte-identical
    // header + rows for identical fixed-timestamp entries.
    #[test]
    fn plain_path_and_equivalent_file_uris_write_identical_content() {
        let dir = tempfile::tempdir().unwrap();
        let plain = dir.path().join("plain.tsv.zst");
        let uri = dir.path().join("uri.tsv.zst");
        let slashes = dir.path().join("slashes.tsv.zst");
        let from_plain = decompressed_after_logging(plain.to_str().unwrap(), &plain);
        let from_uri = decompressed_after_logging(&format!("file://{}", uri.display()), &uri);
        // "file:////dir/x" resolves to "//dir/x", the same file on Linux.
        let from_slashes =
            decompressed_after_logging(&format!("file:///{}", slashes.display()), &slashes);
        let mut expected = EXPECTED_HEADER.to_string();
        expected.extend(EXPECTED_ROWS);
        assert_eq!(String::from_utf8(from_plain.clone()).unwrap(), expected);
        assert_eq!(from_plain, from_uri);
        assert_eq!(from_plain, from_slashes);
    }

    #[test]
    fn written_file_matches_the_frozen_format_byte_for_byte() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("golden.tsv.zst");
        // Explicit capacity, as in decompressed_after_logging above.
        let logger = Logger::open(path.to_str().unwrap(), 64, false).unwrap();
        assert_eq!(logger.get_clock_offset(), 0, "golden test needs offset 0");
        for e in entries() {
            assert_eq!(logger.try_log(e), LogOutcome::Accepted);
        }
        logger.shutdown();

        let text =
            String::from_utf8(zstd::decode_all(std::fs::read(&path).unwrap().as_slice()).unwrap())
                .unwrap();

        // Exact full text, then line by line for a readable failure message.
        let mut expected = EXPECTED_HEADER.to_string();
        expected.extend(EXPECTED_ROWS);
        assert_eq!(text, expected, "op-log text differs from the frozen format");

        let lines: Vec<&str> = text.split_inclusive('\n').collect();
        assert_eq!(lines.len(), 1 + EXPECTED_ROWS.len());
        assert_eq!(lines[0], EXPECTED_HEADER);
        for (i, row) in EXPECTED_ROWS.iter().enumerate() {
            assert_eq!(lines[i + 1], *row, "row {i}");
            assert_eq!(
                lines[i + 1].trim_end_matches('\n').split('\t').count(),
                13,
                "row {i} must have 13 tab-separated columns"
            );
        }
    }
}

// Clock-offset arithmetic in `LogEntry::to_log_line`. Expected strings are
// written out literally (computed independently of the code under test).
#[cfg(test)]
mod oplog_clock_offset_tests {
    use super::*;
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::time::UNIX_EPOCH;

    fn entry_at(start: SystemTime) -> LogEntry {
        LogEntry {
            idx: 3,
            thread_id: 1,
            operation: "GET".to_string(),
            client_id: "c".to_string(),
            num_objects: 1,
            bytes: 10,
            endpoint: "e".to_string(),
            file: "f".to_string(),
            error: None,
            start_time: start,
            first_byte_time: Some(start + Duration::from_nanos(1_500_000)),
            end_time: start + Duration::from_nanos(4_000_000),
        }
    }

    /// 2023-11-14T22:13:20.123456789Z
    fn base() -> SystemTime {
        UNIX_EPOCH + Duration::new(1_700_000_000, 123_456_789)
    }

    /// 2286-11-20T17:46:40.123456789Z: far enough from the epoch that even an
    /// offset of i64::MAX nanoseconds (about 292 years) keeps every time after it.
    fn far() -> SystemTime {
        UNIX_EPOCH + Duration::new(10_000_000_000, 123_456_789)
    }

    fn row(idx: u64, start: &str, first_byte: &str, end: &str) -> String {
        format!("{idx}\t1\tGET\tc\t1\t10\te\tf\t\t{start}\t{first_byte}\t{end}\t4000000\n")
    }

    /// The arithmetic of `to_log_line` exactly as it was before the i64::MIN fix
    /// (negation `-clock_offset_nanos`), copied here as the reference.
    fn to_log_line_before_the_fix(e: &LogEntry, clock_offset_nanos: i64) -> String {
        let offset_duration = if clock_offset_nanos >= 0 {
            Duration::from_nanos(clock_offset_nanos as u64)
        } else {
            Duration::from_nanos((-clock_offset_nanos) as u64)
        };
        let corrected_start = if clock_offset_nanos >= 0 {
            e.start_time
                .checked_sub(offset_duration)
                .unwrap_or(e.start_time)
        } else {
            e.start_time
                .checked_add(offset_duration)
                .unwrap_or(e.start_time)
        };
        let corrected_end = if clock_offset_nanos >= 0 {
            e.end_time
                .checked_sub(offset_duration)
                .unwrap_or(e.end_time)
        } else {
            e.end_time
                .checked_add(offset_duration)
                .unwrap_or(e.end_time)
        };
        let corrected_first_byte = e.first_byte_time.and_then(|t| {
            if clock_offset_nanos >= 0 {
                t.checked_sub(offset_duration)
            } else {
                t.checked_add(offset_duration)
            }
        });
        let duration_ns = corrected_end
            .duration_since(corrected_start)
            .unwrap_or_default()
            .as_nanos();
        let start_ts = humantime::format_rfc3339_nanos(corrected_start).to_string();
        let end_ts = humantime::format_rfc3339_nanos(corrected_end).to_string();
        let first_byte_ts = corrected_first_byte
            .map(|t| humantime::format_rfc3339_nanos(t).to_string())
            .unwrap_or_default();
        let error_str = e.error.as_deref().unwrap_or_default();
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            e.idx,
            e.thread_id,
            e.operation,
            e.client_id,
            e.num_objects,
            e.bytes,
            e.endpoint,
            e.file,
            error_str,
            start_ts,
            first_byte_ts,
            end_ts,
            duration_ns
        )
    }

    // Golden rows for the ordinary offsets (positive subtracts, negative adds).
    #[test]
    fn positive_zero_and_negative_offsets_have_these_exact_rows() {
        let e = entry_at(base());
        for (offset, want) in [
            (
                1_500_000_000_i64,
                row(
                    3,
                    "2023-11-14T22:13:18.623456789Z",
                    "2023-11-14T22:13:18.624956789Z",
                    "2023-11-14T22:13:18.627456789Z",
                ),
            ),
            (
                0,
                row(
                    3,
                    "2023-11-14T22:13:20.123456789Z",
                    "2023-11-14T22:13:20.124956789Z",
                    "2023-11-14T22:13:20.127456789Z",
                ),
            ),
            (
                -2_250_000_000,
                row(
                    3,
                    "2023-11-14T22:13:22.373456789Z",
                    "2023-11-14T22:13:22.374956789Z",
                    "2023-11-14T22:13:22.377456789Z",
                ),
            ),
        ] {
            assert_eq!(e.to_log_line(offset), want, "offset {offset}");
        }
    }

    // The extremes. i64::MAX subtracts about 292 years; i64::MIN adds 2^63 ns
    // (about 292 years), which is also what release builds of the old code
    // produced (the old negation wrapped to 2^63); debug builds panicked.
    #[test]
    fn extreme_offsets_do_not_panic_and_have_these_exact_rows() {
        let e = entry_at(far());
        assert_eq!(
            e.to_log_line(i64::MAX),
            row(
                3,
                "1994-08-11T17:59:23.268680982Z",
                "1994-08-11T17:59:23.270180982Z",
                "1994-08-11T17:59:23.272680982Z",
            )
        );
        assert_eq!(
            e.to_log_line(i64::MIN),
            row(
                3,
                "2579-03-01T17:33:56.978232597Z",
                "2579-03-01T17:33:56.979732597Z",
                "2579-03-01T17:33:56.982232597Z",
            )
        );
    }

    // End to end: the writer thread must survive an i64::MIN offset and write
    // the row (before the fix a debug build panicked in the writer thread).
    #[test]
    fn writer_thread_survives_an_i64_min_offset() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("min.tsv.zst");
        let logger = Logger::open(path.to_str().unwrap(), 8, false).unwrap();
        logger.set_clock_offset(i64::MIN);
        assert_eq!(logger.try_log(entry_at(far())), LogOutcome::Accepted);
        logger.shutdown();
        let text =
            String::from_utf8(zstd::decode_all(std::fs::read(&path).unwrap().as_slice()).unwrap())
                .unwrap();
        let mut lines = text.lines();
        assert_eq!(lines.next(), Some(HEADER.trim_end_matches('\n')));
        let want = row(
            0,
            "2579-03-01T17:33:56.978232597Z",
            "2579-03-01T17:33:56.979732597Z",
            "2579-03-01T17:33:56.982232597Z",
        );
        assert_eq!(lines.next().map(|l| format!("{l}\n")), Some(want));
        assert_eq!(lines.next(), None);
    }

    // The fix must not change any output: compare against a copy of the old
    // arithmetic for a sweep of offsets (everything but i64::MIN, where the old
    // code panics in debug builds).
    #[test]
    fn output_is_identical_to_the_old_arithmetic_for_every_offset_but_i64_min() {
        let mut offsets: Vec<i64> = vec![i64::MIN + 1, i64::MAX, i64::MAX - 1, -1, 0, 1];
        for exp in 0..18_u32 {
            let p = 10_i64.pow(exp);
            offsets.extend([p, -p, p + 1, -(p + 1), 3 * p, -3 * p]);
        }
        let mut x = 0x9E37_79B9_7F4A_7C15_u64;
        for _ in 0..4000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            offsets.push(x as i64);
            offsets.push(((x >> 1) as i64) >> (x % 40)); // smaller magnitudes too
        }
        offsets.retain(|o| *o != i64::MIN);

        let mut with_error = entry_at(far());
        with_error.error = Some("boom".to_string());
        with_error.first_byte_time = None;
        let entries = [entry_at(far()), with_error];
        for e in &entries {
            for &o in &offsets {
                assert_eq!(
                    e.to_log_line(o),
                    to_log_line_before_the_fix(e, o),
                    "offset {o}"
                );
            }
        }
        assert!(offsets.len() > 8000);
    }

    // Today's behaviour at the edge of the representable range, pinned (not
    // changed by the overflow fix):
    // - a first_byte that the offset pushes out of SystemTime's range is an
    //   empty first_byte column;
    // - a corrected time BEFORE the epoch panics inside humantime's formatter
    //   ("all times should be after the epoch"), which in the writer thread
    //   ends the log. That is a pre-existing limit, recorded in the report.
    #[test]
    fn unrepresentable_first_byte_is_empty() {
        let mut e = entry_at(base());
        e.first_byte_time = Some(UNIX_EPOCH + Duration::new(i64::MAX as u64, 0));
        assert_eq!(
            e.to_log_line(-1_500_000_000),
            "3\t1\tGET\tc\t1\t10\te\tf\t\t2023-11-14T22:13:21.623456789Z\t\t2023-11-14T22:13:21.627456789Z\t4000000\n"
        );
    }

    fn panic_text(f: impl FnOnce() -> String) -> Option<String> {
        catch_unwind(AssertUnwindSafe(f)).err().map(|p| {
            p.downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_default()
        })
    }

    #[test]
    fn times_pushed_before_the_epoch_panic_in_the_formatter_as_they_always_did() {
        // start/end pushed below the epoch.
        let near_epoch = entry_at(UNIX_EPOCH + Duration::from_secs(1));
        // Only first_byte (earlier than start in this contrived entry) is pushed below.
        let mut only_first_byte = entry_at(UNIX_EPOCH + Duration::from_secs(10));
        only_first_byte.first_byte_time = Some(UNIX_EPOCH + Duration::from_secs(1));
        for (name, e) in [("start", near_epoch), ("first_byte", only_first_byte)] {
            let new = panic_text(|| e.to_log_line(2_000_000_000));
            let old = panic_text(|| to_log_line_before_the_fix(&e, 2_000_000_000));
            let new = new.unwrap_or_else(|| panic!("{name}: expected the formatter to panic"));
            assert!(new.contains("all times should be after the epoch"), "{new}");
            assert_eq!(Some(new), old, "{name}: old and new panic identically");
        }
    }
}
