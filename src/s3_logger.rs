// src/s3_logger.rs
//
// Copyright, 2025. Signal65 / Futurum Group.
//
//! I/O logging functionality to trace S3 operations.
//!
//! Creates a zstd-compressed, tab-separated log file with a format
//! identical to the 'warp-replay' tool.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::SystemTime;
use std::thread;
use std::sync::{Mutex, mpsc::{sync_channel, channel, SyncSender, Receiver}};
use once_cell::sync::OnceCell;
use zstd::stream::write::Encoder;

// Global variables
static GLOBAL_LOGGER: OnceCell<Logger> = OnceCell::new();
static LOSSLESS: OnceCell<bool> = OnceCell::new();

const SHUTDOWN_OP: &str = "__SHUTDOWN__";
const HEADER: &str = "idx\tthread\top\tclient_id\tn_objects\tbytes\tendpoint\tfile\terror\tstart\tfirst_byte\tend\tduration_ns\n";

/// Initialise the op‑log *once per process*.
/// Subsequent calls are ignored (they return `Ok(())`).
pub fn init_op_logger<P: AsRef<str>>(path: P) -> std::io::Result<()> {
    if GLOBAL_LOGGER.get().is_some() {
        return Ok(());
    }
    let logger = Logger::new(path.as_ref())?;
    let _ = GLOBAL_LOGGER.set(logger);
    Ok(())
}

/// Fetch the global logger (if initialised).
pub fn global_logger() -> Option<Logger> {
    GLOBAL_LOGGER.get().cloned()
}

/// Signal the background logger to finish and wait for it to flush.
/// This uses a blocking send of a shutdown sentinel so it cannot be dropped.
pub fn finalize_op_logger() {
    if let Some(logger) = GLOBAL_LOGGER.get() {
        // Send a shutdown entry using blocking `send` so it cannot be lost if the buffer is full.
        let _ = logger.sender.send(LogEntry {
            idx: 0,
            thread_id: 0,
            operation: SHUTDOWN_OP.to_string(),
            client_id: String::new(),
            num_objects: 0,
            bytes: 0,
            endpoint: String::new(),
            file: String::new(),
            error: None,
            start_time: SystemTime::now(),
            first_byte_time: None,
            end_time: SystemTime::now(),
        });

        // Wait for the background thread to finish (encoder flushes on drop).
        if let Some(done_rx) = logger.done_rx.lock().unwrap().take() {
            let _ = done_rx.recv();
        }
    }
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
    fn to_log_line(&self) -> String {
        let duration_ns = self
            .end_time
            .duration_since(self.start_time)
            .unwrap_or_default()
            .as_nanos();
        let start_ts = humantime::format_rfc3339_nanos(self.start_time).to_string();
        let end_ts = humantime::format_rfc3339_nanos(self.end_time).to_string();
        let first_byte_ts = self
            .first_byte_time
            .map(|t| humantime::format_rfc3339_nanos(t).to_string())
            .unwrap_or_default();
        let error_str = self.error.as_deref().unwrap_or_default();
        format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            self.idx, self.thread_id, self.operation, self.client_id, self.num_objects,
            self.bytes, self.endpoint, self.file, error_str, start_ts, first_byte_ts,
            end_ts, duration_ns
        )
    }
}

/// Handle for sending log entries and waiting on shutdown.
#[derive(Debug)]
pub struct Logger {
    sender: SyncSender<LogEntry>,
    done_rx: Mutex<Option<Receiver<()>>>,
}

impl Clone for Logger {
    fn clone(&self) -> Self {
        Logger {
            sender: self.sender.clone(),
            done_rx: Mutex::new(None),
        }
    }
}

// Logger methods
impl Logger {
    /// Creates a new logger and spawns the background thread to handle writing.
    pub fn new(log_file_path: &str) -> std::io::Result<Self> {
        // 
        // Check to see if the LOSSLESS env variable is set 
        let _ = LOSSLESS.set(
            std::env::var("S3DLIO_OPLOG_LOSSLESS")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
        );

/*
 * Instead of a Bounded channel with set size, check ENV variables to optionally use different
 * sizes
 *
        // Bounded channel for log entries (drop on overflow in `log()` via try_send).
        let (sender, receiver): (SyncSender<LogEntry>, Receiver<LogEntry>) = sync_channel(256);
*/
        // Tunable buffer sizes via (env) variables:
        let cap: usize = std::env::var("S3DLIO_OPLOG_BUF").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(256);
        let wbuf: usize = std::env::var("S3DLIO_OPLOG_WBUFCAP").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(256 * 1024);
        let level: i32 = std::env::var("S3DLIO_OPLOG_LEVEL").ok()
            .and_then(|s| s.parse().ok()).unwrap_or(1); // bias to fast

        // Bounded channel for log entries (drop on overflow in `log()` via try_send).
        let (sender, receiver): (SyncSender<LogEntry>, Receiver<LogEntry>) = sync_channel(cap);

        // One-shot channel to signal that the background thread finished.
        let (done_tx, done_rx) = channel::<()>();

        // Prepare compressed writer.
        let file = File::create(log_file_path)?;

        // Old, we didn't set a buffer capacity
        //let writer = BufWriter::new(file);
        let writer = BufWriter::with_capacity(wbuf, file); 

        let mut encoder = Encoder::new(writer, level)?.auto_finish();

        // Write header now so even empty logs have the header.
        encoder.write_all(HEADER.as_bytes())?;
        encoder.flush()?;

        // Background writer thread.
        thread::spawn(move || {
            let mut idx: u64 = 0;
            for mut entry in receiver {
                if entry.operation == SHUTDOWN_OP {
                    break;
                }
                entry.idx = idx;
                let line = entry.to_log_line();
                if let Err(e) = encoder.write_all(line.as_bytes()) {
                    eprintln!("Error writing to op-log: {e}");
                    break;
                }
                idx += 1;
            }
            // Drop encoder here to finish the zstd stream.
            drop(encoder);
            let _ = done_tx.send(());
        });

        Ok(Logger { sender, done_rx: Mutex::new(Some(done_rx)) })
    }

    /// Submit a log entry. Use try_send to avoid blocking; drop if channel is full.
    /// if our LOSSLESS env is set, then we do block instead of dropping 
    pub fn log(&self, entry: LogEntry) {
        if *LOSSLESS.get().unwrap_or(&false) {
            // Blocking: guarantees no drops (may stall if writer can’t keep up).
            let _ = self.sender.send(entry);
        } else {
            // Non‑blocking: preserves I/O throughput; may drop under burst.
            if let Err(e) = self.sender.try_send(entry) {
                eprintln!("op-log channel full; dropping entry: {e}");
            }
        }
    }
}

