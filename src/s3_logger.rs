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
    Arc, Mutex, MutexGuard, PoisonError,
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
static LOSSLESS: OnceCell<bool> = OnceCell::new();
static GLOBAL_CLIENT_ID: OnceCell<Mutex<String>> = OnceCell::new();

/// Top bit of `Shared::gate`: set once finalize has begun. The low bits count
/// `Logger::log` calls currently between "gate entered" and "outcome decided".
const GATE_CLOSED: u64 = 1 << 63;
const HEADER: &str = "idx\tthread\top\tclient_id\tn_objects\tbytes\tendpoint\tfile\terror\tstart\tfirst_byte\tend\tduration_ns\n";

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
/// - Uninitialized: resolves `path` (plain local path or `file://` URI), creates
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
    let mut state = lock_state();
    match &*state {
        OpLogState::Active(_) => return Ok(()),
        // Checked before any filesystem action: must not create or truncate `path`.
        OpLogState::Finalized(_) => {
            return Err(std::io::Error::new(
                std::io::ErrorKind::Unsupported,
                format!(
                    "op-log was already finalized and cannot be restarted; a process supports \
                     only one op-log session, so '{path}' was not opened"
                ),
            ))
        }
        OpLogState::Uninitialized => {}
    }
    let logger = Logger::new(path)?;
    *state = OpLogState::Active(logger);

    // Print out some info if info logging set
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

/// Set clock offset for the global logger (convenience function).
///
/// # Parameters
/// - `offset_nanos`: Nanoseconds to subtract from local timestamps for distributed sync
///
/// # Returns
/// - `Ok(())` if logger is initialized and offset was set
/// - `Err` if logger is not initialized
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
    if let Some(logger) = global_logger() {
        logger.set_clock_offset(offset_nanos);
        Ok(())
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Logger not initialized - call init_op_logger() first",
        ))
    }
}

/// Get the current clock offset from the global logger.
pub fn get_clock_offset() -> std::io::Result<i64> {
    if let Some(logger) = global_logger() {
        Ok(logger.get_clock_offset())
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Logger not initialized",
        ))
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

/// Finish the process's op-log session: every record whose
/// [`Logger::log`] call was accepted is written, the zstd stream is finished and
/// the writer thread has exited before this returns.
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
        let offset_duration = if clock_offset_nanos >= 0 {
            Duration::from_nanos(clock_offset_nanos as u64)
        } else {
            Duration::from_nanos((-clock_offset_nanos) as u64)
        };

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

/// Result of [`Logger::log`]. Only `Accepted` entries are guaranteed to be in
/// the file once the session is finalized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogOutcome {
    /// Queued for the writer; it will be written, even if finalize runs next.
    Accepted,
    /// The queue was full (non-lossless mode): the entry was dropped.
    Full,
    /// The session is finalizing or finalized: the entry was rejected, not written.
    Closed,
    /// The writer thread stopped after an I/O error: the entry was not written.
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
    /// [`Logger::log`] and [`Logger::shutdown`] for the drain protocol.
    gate: AtomicU64,
    /// Completion signal from the writer thread; taken by the first shutdown.
    /// The mutex is held for the whole shutdown so later callers wait for it.
    done_rx: Mutex<Option<Receiver<()>>>,
    clock_offset_nanos: Arc<AtomicI64>, // Nanoseconds to adjust timestamps for distributed sync
    closed_noted: AtomicBool,
    writer_failure_noted: AtomicBool,
}

/// Handle for sending log entries. Clones share one writer thread.
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
/// Accepts a plain path (`/tmp/ops.tsv.zst`) or a `file://` URI
/// (`file:///tmp/ops.tsv.zst`). Any other `scheme://` URI is rejected with
/// `ErrorKind::InvalidInput` and a message naming the path, rather than being
/// passed to `File::create` as a bogus relative path (issue #168). The text
/// before the first `://` counts as a scheme only if it is a valid RFC 3986
/// scheme; a local path that merely contains `://` stays a path.
fn resolve_local_oplog_path(path: &str) -> std::io::Result<&str> {
    if let Some(local) = path.strip_prefix("file://") {
        if local.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("op-log path '{path}' has an empty file path after 'file://'"),
            ));
        }
        return Ok(local);
    }
    // Only text that is a syntactically valid RFC 3986 scheme before the first
    // "://" makes this a URI. A local path such as "<dir>/a://b.log" (which the
    // OS resolves to "<dir>/a:/b.log") merely contains "://" and stays a path.
    if let Some((scheme, _)) = path
        .split_once("://")
        .filter(|(scheme, _)| is_rfc3986_scheme(scheme))
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!(
                "op-log path '{path}' uses unsupported scheme '{scheme}://'; the op-log must be \
                 written to a local filesystem path, e.g. '/tmp/ops.tsv.zst' or \
                 'file:///tmp/ops.tsv.zst'"
            ),
        ));
    }
    Ok(path)
}

// Logger methods
impl Logger {
    /// Creates a new logger and spawns the background thread to handle writing.
    ///
    /// `log_file_path` may be a plain local path or a `file://` URI.
    pub fn new(log_file_path: &str) -> std::io::Result<Self> {
        let local_path = resolve_local_oplog_path(log_file_path)?;

        //
        // Check to see if the LOSSLESS env variable is set
        let _ = LOSSLESS.set(
            std::env::var("S3DLIO_OPLOG_LOSSLESS")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
        );

        /*
         * Instead of a Bounded channel with set size, check ENV variables to optionally use different
         * sizes
         *
                // Bounded channel for log entries (drop on overflow in `log()` via try_send).
                let (sender, receiver): (SyncSender<LogEntry>, Receiver<LogEntry>) = sync_channel(256);
        */
        // Tunable buffer sizes via (env) variables:
        let cap: usize = oplog_channel_capacity();
        let wbuf: usize = oplog_write_buffer_capacity();
        let level: i32 = std::env::var("S3DLIO_OPLOG_LEVEL")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1); // bias to fast

        // Bounded channel for log entries (drop on overflow in `log()` via try_send).
        let (sender, receiver): (SyncSender<Message>, Receiver<Message>) = sync_channel(cap);

        // One-shot channel to signal that the background thread finished.
        let (done_tx, done_rx) = channel::<()>();

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

        let mut encoder = Encoder::new(writer, level)?.auto_finish();

        // Write header now so even empty logs have the header.
        encoder.write_all(HEADER.as_bytes())?;
        encoder.flush()?;

        // Background writer thread.
        let clock_offset = Arc::new(AtomicI64::new(0));
        let clock_offset_clone = Arc::clone(&clock_offset);
        thread::spawn(move || {
            for (idx, message) in (0_u64..).zip(receiver) {
                let mut entry = match message {
                    Message::Entry(entry) => entry,
                    Message::Shutdown => break,
                };
                entry.idx = idx;
                let offset = clock_offset_clone.load(Ordering::Relaxed);
                let line = entry.to_log_line(offset);
                if let Err(e) = encoder.write_all(line.as_bytes()) {
                    eprintln!("Error writing to op-log: {e}");
                    break;
                }
            }
            // Drop encoder here to finish the zstd stream.
            drop(encoder);
            let _ = done_tx.send(());
        });

        Ok(Logger {
            shared: Arc::new(Shared {
                sender,
                gate: AtomicU64::new(0),
                done_rx: Mutex::new(Some(done_rx)),
                clock_offset_nanos: clock_offset,
                closed_noted: AtomicBool::new(false),
                writer_failure_noted: AtomicBool::new(false),
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

    /// Submit a log entry and report what happened to it.
    ///
    /// Non-blocking by default (`try_send`): when the queue is full the entry is
    /// dropped and [`LogOutcome::Full`] is returned. With `S3DLIO_OPLOG_LOSSLESS=1`
    /// the call blocks instead of dropping.
    ///
    /// `Accepted` is a promise: the entry is on the writer's queue and
    /// [`finalize_op_logger`] will write it. Once finalize has begun, calls
    /// return [`LogOutcome::Closed`] (never `Full`) and write nothing; at most
    /// one diagnostic is printed for the whole closed logger.
    ///
    /// The return value may be ignored (it is deliberately not `#[must_use]`:
    /// the store wrappers fire-and-forget); tests use it to prove drain exactness.
    pub fn log(&self, entry: LogEntry) -> LogOutcome {
        let shared = &*self.shared;
        // Enter the gate. The increment is visible to shutdown's wait-for-zero
        // even if it loses the race with the close, so shutdown cannot send its
        // sentinel while any call that passed the closed check is still sending.
        let before = shared.gate.fetch_add(1, Ordering::Acquire);
        let _in_flight = InFlight(&shared.gate);
        if before & GATE_CLOSED != 0 {
            if !shared.closed_noted.swap(true, Ordering::Relaxed) {
                eprintln!(
                    "op-log finalized: entries from stores created before finalize are no \
                     longer recorded"
                );
            }
            return LogOutcome::Closed;
        }

        let outcome = if *LOSSLESS.get().unwrap_or(&false) {
            // Blocking: guarantees no drops (may stall if writer can’t keep up).
            match shared.sender.send(Message::Entry(entry)) {
                Ok(()) => LogOutcome::Accepted,
                Err(_) => LogOutcome::WriterFailed,
            }
        } else {
            // Non‑blocking: preserves I/O throughput; may drop under burst.
            match shared.sender.try_send(Message::Entry(entry)) {
                Ok(()) => LogOutcome::Accepted,
                Err(TrySendError::Full(_)) => {
                    eprintln!("op-log channel full; dropping entry");
                    LogOutcome::Full
                }
                Err(TrySendError::Disconnected(_)) => LogOutcome::WriterFailed,
            }
        };
        if outcome == LogOutcome::WriterFailed
            && !shared.writer_failure_noted.swap(true, Ordering::Relaxed)
        {
            eprintln!("op-log writer has stopped after an error; entries are being dropped");
        }
        outcome
    }

    /// Close the logger, drain every accepted entry and wait for the writer.
    ///
    /// Exactness: (1) set `GATE_CLOSED`, so every later `log()` is rejected;
    /// (2) wait until the in-flight count reaches zero, i.e. every `log()` that
    /// entered before the close has finished its send, so everything accepted is
    /// already on the queue; (3) send `Shutdown`, which the FIFO channel delivers
    /// after all of those entries; (4) wait for the writer to finish the zstd
    /// stream. Idempotent; concurrent callers block until the first finishes.
    fn shutdown(&self) {
        let shared = &*self.shared;
        let mut done_rx = shared
            .done_rx
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let Some(done) = done_rx.take() else {
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
        // writer already stopped the send fails, which is fine.
        let _ = shared.sender.send(Message::Shutdown);

        // Print info if info set
        info!("Shutting down S3 operation logging");

        // Wait for the background thread to finish (encoder flushes on drop).
        let _ = done.recv();
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
}

#[cfg(test)]
mod oplog_shutdown_tests {
    use super::*;

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

    // A standalone Logger (not the process-global one) can be shut down
    // repeatedly; once closed it reports Closed, never Full.
    #[test]
    fn shutdown_drains_is_idempotent_and_rejects_later_entries() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ops.tsv.zst");
        let logger = Logger::new(path.to_str().unwrap()).unwrap();

        assert_eq!(logger.log(entry("a")), LogOutcome::Accepted);
        assert_eq!(logger.log(entry("b")), LogOutcome::Accepted);
        logger.shutdown();
        logger.shutdown();
        assert_eq!(logger.clone().log(entry("c")), LogOutcome::Closed);

        let text =
            String::from_utf8(zstd::decode_all(std::fs::read(&path).unwrap().as_slice()).unwrap())
                .unwrap();
        let files: Vec<&str> = text
            .lines()
            .skip(1)
            .map(|l| l.split('\t').nth(7).unwrap())
            .collect();
        assert_eq!(files, ["a", "b"]);
    }
}

// Golden test: pins the EXACT bytes of the op-log format that external tools
// (sai3-bench, polarWarp) parse. Expected strings are written out literally --
// they are NOT derived from HEADER or to_log_line -- so a renamed or reordered
// column, a changed timestamp layout or a wrong idx fails here.
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

    #[test]
    fn written_file_matches_the_frozen_format_byte_for_byte() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("golden.tsv.zst");
        let logger = Logger::new(path.to_str().unwrap()).unwrap();
        assert_eq!(logger.get_clock_offset(), 0, "golden test needs offset 0");
        for e in entries() {
            assert_eq!(logger.log(e), LogOutcome::Accepted);
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
