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
use std::time::{SystemTime, Duration};
use std::thread;
use std::sync::{Arc, Mutex, mpsc::{sync_channel, channel, SyncSender, Receiver}};
use std::sync::atomic::{AtomicI64, Ordering};
use tracing::info;
use once_cell::sync::OnceCell;
use zstd::stream::write::Encoder;

// Global variables
static GLOBAL_LOGGER: OnceCell<Logger> = OnceCell::new();
static LOSSLESS: OnceCell<bool> = OnceCell::new();
static GLOBAL_CLIENT_ID: OnceCell<Mutex<String>> = OnceCell::new();

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

    // Print out some info if info logging set
    info!("Intialized S3 operation logging to file: {}", path.as_ref());

    Ok(())
}

/// Fetch the global logger (if initialised).
pub fn global_logger() -> Option<Logger> {
    GLOBAL_LOGGER.get().cloned()
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
    if let Some(logger) = GLOBAL_LOGGER.get() {
        logger.set_clock_offset(offset_nanos);
        Ok(())
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Logger not initialized - call init_op_logger() first"
        ))
    }
}

/// Get the current clock offset from the global logger.
pub fn get_clock_offset() -> std::io::Result<i64> {
    if let Some(logger) = GLOBAL_LOGGER.get() {
        Ok(logger.get_clock_offset())
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Logger not initialized"
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
            Err(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Failed to acquire client_id lock"
            ))
        }
    } else {
        Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "Failed to initialize client_id storage"
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

        // Print info if info set
        info!("Shutting down S3 operation logging");

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
    fn to_log_line(&self, clock_offset_nanos: i64) -> String {
        // Apply clock offset correction for distributed systems
        // Positive offset means local clock is ahead, so we subtract to align with reference time
        let offset_duration = if clock_offset_nanos >= 0 {
            Duration::from_nanos(clock_offset_nanos as u64)
        } else {
            Duration::from_nanos((-clock_offset_nanos) as u64)
        };
        
        let corrected_start = if clock_offset_nanos >= 0 {
            self.start_time.checked_sub(offset_duration).unwrap_or(self.start_time)
        } else {
            self.start_time.checked_add(offset_duration).unwrap_or(self.start_time)
        };
        
        let corrected_end = if clock_offset_nanos >= 0 {
            self.end_time.checked_sub(offset_duration).unwrap_or(self.end_time)
        } else {
            self.end_time.checked_add(offset_duration).unwrap_or(self.end_time)
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
    clock_offset_nanos: Arc<AtomicI64>,  // Nanoseconds to adjust timestamps for distributed sync
}

impl Clone for Logger {
    fn clone(&self) -> Self {
        Logger {
            sender: self.sender.clone(),
            done_rx: Mutex::new(None),
            clock_offset_nanos: Arc::clone(&self.clock_offset_nanos),
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
        let clock_offset = Arc::new(AtomicI64::new(0));
        let clock_offset_clone = Arc::clone(&clock_offset);
        thread::spawn(move || {
            let mut idx: u64 = 0;
            for mut entry in receiver {
                if entry.operation == SHUTDOWN_OP {
                    break;
                }
                entry.idx = idx;
                let offset = clock_offset_clone.load(Ordering::Relaxed);
                let line = entry.to_log_line(offset);
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

        Ok(Logger { 
            sender, 
            done_rx: Mutex::new(Some(done_rx)),
            clock_offset_nanos: clock_offset,
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
        self.clock_offset_nanos.store(offset_nanos, Ordering::Relaxed);
    }
    
    /// Get the current clock offset value.
    pub fn get_clock_offset(&self) -> i64 {
        self.clock_offset_nanos.load(Ordering::Relaxed)
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

