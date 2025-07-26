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
use tokio::sync::mpsc;
use zstd::stream::write::Encoder;

// LogEntry struct remains the same as before
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
    // to_log_line() method remains the same as before
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


/// A handle for sending log entries to the background logging task.
/// Cloning this is cheap and allows many threads to submit logs concurrently.
#[derive(Clone)]
pub struct Logger {
    sender: mpsc::Sender<LogEntry>,
}

impl Logger {
    /// Creates a new logger and spawns the background task to handle writing.
    pub fn new(log_file_path: &str) -> Result<Self, std::io::Error> {
        // Create an asynchronous channel with a buffer of 256 messages.
        let (sender, mut receiver) = mpsc::channel::<LogEntry>(256);

        let file = File::create(log_file_path)?;
        let writer = BufWriter::new(file);
        let mut encoder = zstd::stream::write::Encoder::new(writer, 0)?.auto_finish();

        // Write the header synchronously before starting
        let header = "idx\tthread\top\tclient_id\tn_objects\tbytes\tendpoint\tfile\terror\tstart\tfirst_byte\tend\tduration_ns\n";
        encoder.write_all(header.as_bytes())?;
        
        // Spawn a dedicated Tokio task to process log messages
        tokio::spawn(async move {
            let mut idx = 0u64;
            while let Some(mut entry) = receiver.recv().await {
                entry.idx = idx;
                let log_line = entry.to_log_line();
                if encoder.write_all(log_line.as_bytes()).is_err() {
                    // Stop if we can no longer write to the file
                    eprintln!("Error writing to log file, stopping logger task.");
                    break;
                }
                idx += 1;
            }
        });

        Ok(Logger { sender })
    }

    /// Submits a log entry to the channel. This is a non-blocking operation.
    pub fn log(&self, entry: LogEntry) {
        // We use try_send to avoid any possibility of blocking. If the channel
        // buffer is full (meaning the logger task can't keep up), we drop the message.
        // This prioritizes application performance over logging every single event.
        if let Err(e) = self.sender.try_send(entry) {
            eprintln!("Failed to send log message (buffer may be full): {}", e);
        }
    }
}
