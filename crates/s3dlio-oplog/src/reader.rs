// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Operation log reader with multi-format support
//!
//! Provides robust parsing of operation logs in JSONL and TSV formats,
//! with automatic compression detection and header-driven column mapping.
//!
//! The streaming reader (OpLogStreamReader) processes large op-logs with constant memory usage
//! by using background decompression and 1MB chunk buffering.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread::{self, JoinHandle};
use tracing::{debug, info, warn};

use crate::types::{OpLogEntry, OpType};

// Import streaming constants from main crate
const DEFAULT_OPLOG_CHUNK_SIZE: usize = 1024 * 1024; // 1 MB
const DEFAULT_OPLOG_ENTRY_BUFFER: usize = 1024; // 1024 entries
const ENV_OPLOG_READ_BUF: &str = "S3DLIO_OPLOG_READ_BUF";
const ENV_OPLOG_CHUNK_SIZE: &str = "S3DLIO_OPLOG_CHUNK_SIZE";

/// Operation log format detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpLogFormat {
    Jsonl,
    Tsv,
}

impl OpLogFormat {
    /// Detect format from file path extension
    pub fn from_path<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_string_lossy();

        if path_str.ends_with(".jsonl") || path_str.ends_with(".jsonl.zst") {
            Ok(OpLogFormat::Jsonl)
        } else if path_str.ends_with(".tsv") 
            || path_str.ends_with(".tsv.zst")
            || path_str.ends_with(".csv")  // .csv files are often TSV in practice (e.g., Warp)
            || path_str.ends_with(".csv.zst")
        {
            Ok(OpLogFormat::Tsv)
        } else {
            anyhow::bail!(
                "Unsupported file format: {}. Supported: .jsonl[.zst], .tsv[.zst], .csv[.zst]",
                path_str
            )
        }
    }
}

/// Streaming op-log reader with background decompression and constant memory usage
///
/// This reader processes op-log files in 1MB chunks using a background thread for decompression,
/// providing constant memory usage regardless of file size. Entries are streamed via an iterator
/// rather than loaded all at once.
///
/// # Example
/// ```no_run
/// use s3dlio_oplog::OpLogStreamReader;
///
/// let stream = OpLogStreamReader::from_file("large_oplog.tsv.zst")?;
/// for entry in stream {
///     let entry = entry?;
///     println!("Processing: {:?}", entry.op);
/// }
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct OpLogStreamReader {
    receiver: Receiver<Result<OpLogEntry>>,
    _background_handle: Option<JoinHandle<()>>,
}

impl OpLogStreamReader {
    /// Create a streaming reader from a file path
    ///
    /// This spawns a background thread for decompression and parsing.
    /// Environment variables:
    /// - S3DLIO_OPLOG_READ_BUF: Channel buffer size (default: 1024 entries)
    /// - S3DLIO_OPLOG_CHUNK_SIZE: Read chunk size (default: 1MB)
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_buf = path.as_ref().to_path_buf();
        let format = OpLogFormat::from_path(&path_buf)?;
        let is_compressed = path_buf.extension().is_some_and(|ext| ext == "zst");

        info!(
            "Opening streaming op-log reader for {:?} (format: {:?}, compressed: {})",
            path_buf, format, is_compressed
        );

        // Get tunable parameters from environment
        let channel_cap = std::env::var(ENV_OPLOG_READ_BUF)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_OPLOG_ENTRY_BUFFER);

        let chunk_size = std::env::var(ENV_OPLOG_CHUNK_SIZE)
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_OPLOG_CHUNK_SIZE);

        debug!(
            "Streaming reader config: channel_cap={}, chunk_size={}",
            channel_cap, chunk_size
        );

        let (sender, receiver) = sync_channel(channel_cap);

        // Spawn background thread for decompression and parsing
        let error_sender = sender.clone();
        let handle = thread::spawn(move || {
            if let Err(e) =
                Self::parse_in_background(path_buf, format, is_compressed, chunk_size, sender)
            {
                warn!("Background parsing error: {}", e);
                // Deliver it to the consumer too (a bad header, an unreadable file): without
                // this the channel just closes and the log reads as empty.
                let _ = error_sender.send(Err(e));
            }
        });

        Ok(Self {
            receiver,
            _background_handle: Some(handle),
        })
    }

    /// Background parsing thread function
    fn parse_in_background(
        path: PathBuf,
        format: OpLogFormat,
        is_compressed: bool,
        chunk_size: usize,
        sender: SyncSender<Result<OpLogEntry>>,
    ) -> Result<()> {
        let file = File::open(&path)
            .with_context(|| format!("Failed to open file: {}", path.display()))?;

        let reader: Box<dyn BufRead> = if is_compressed {
            let decoder = zstd::stream::read::Decoder::new(file)
                .with_context(|| "Failed to create zstd decoder")?;
            Box::new(BufReader::with_capacity(chunk_size, decoder))
        } else {
            Box::new(BufReader::with_capacity(chunk_size, file))
        };

        match format {
            OpLogFormat::Jsonl => Self::stream_jsonl(reader, sender),
            OpLogFormat::Tsv => Self::stream_tsv(reader, sender),
        }
    }

    /// Stream JSONL format line-by-line
    fn stream_jsonl(
        reader: Box<dyn BufRead>,
        sender: SyncSender<Result<OpLogEntry>>,
    ) -> Result<()> {
        for (line_num, line) in reader.lines().enumerate() {
            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    let err = Err(anyhow::anyhow!(
                        "Failed to read line {}: {}",
                        line_num + 1,
                        e
                    ));
                    if sender.send(err).is_err() {
                        break; // Receiver dropped
                    }
                    continue;
                }
            };

            // Skip empty lines and comments
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            // Parse JSON and send entry
            let result = serde_json::from_str::<serde_json::Value>(&line)
                .with_context(|| format!("JSON parse error on line {}", line_num + 1))
                .and_then(|json| parse_json_entry(&json, line_num));

            if sender.send(result).is_err() {
                break; // Receiver dropped, stop parsing
            }
        }
        Ok(())
    }

    /// Stream TSV format line-by-line
    fn stream_tsv(reader: Box<dyn BufRead>, sender: SyncSender<Result<OpLogEntry>>) -> Result<()> {
        let mut lines = reader.lines();

        // Read and parse header
        let header = lines
            .next()
            .ok_or_else(|| anyhow::anyhow!("Empty TSV file"))??;
        let headers: Vec<&str> = header.split('\t').collect();
        let col_mapping = create_column_mapping(&headers)?;

        // Stream entries
        for (line_num, line) in lines.enumerate() {
            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    let err = Err(anyhow::anyhow!(
                        "Failed to read line {}: {}",
                        line_num + 2,
                        e
                    ));
                    if sender.send(err).is_err() {
                        break;
                    }
                    continue;
                }
            };

            // Skip empty lines and comments
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }

            let fields: Vec<&str> = line.split('\t').collect();
            let result = parse_tsv_entry(&fields, &col_mapping, line_num);

            if sender.send(result).is_err() {
                break; // Receiver dropped
            }
        }
        Ok(())
    }
}

impl Iterator for OpLogStreamReader {
    type Item = Result<OpLogEntry>;

    fn next(&mut self) -> Option<Self::Item> {
        self.receiver.recv().ok()
    }
}

/// Reader for op-log files with automatic format and compression detection
///
/// This is a convenience wrapper that loads all entries into memory.
/// For large files, consider using OpLogStreamReader for constant memory usage.
pub struct OpLogReader {
    entries: Vec<OpLogEntry>,
}

impl OpLogReader {
    /// Load op-log from file with automatic format and compression detection
    ///
    /// Note: This loads all entries into memory. For large files, use OpLogStreamReader instead.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_ref = path.as_ref();
        info!(
            "Loading op-log from {:?} (loading all entries into memory)",
            path_ref
        );

        // Use streaming reader internally and collect all entries
        let stream = OpLogStreamReader::from_file(path_ref)?;
        let entries: Result<Vec<_>> = stream.collect();
        let entries = entries?;

        info!("Loaded {} operations from op-log", entries.len());
        Ok(OpLogReader { entries })
    }

    /// Get all entries
    pub fn entries(&self) -> &[OpLogEntry] {
        &self.entries
    }

    /// Filter entries by operation type
    pub fn filter_operations(&self, operations: &[OpType]) -> Vec<&OpLogEntry> {
        self.entries
            .iter()
            .filter(|entry| operations.contains(&entry.op))
            .collect()
    }

    /// Get number of entries
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// =============================================================================
// Shared parsing functions used by both streaming and batch readers
// =============================================================================

/// Start time for a row from a log that carries no start column or key at all (the minimal
/// dl-driver format): the read time, so a replay of it has no timing. Warned once per process.
fn untimed_start() -> DateTime<Utc> {
    static WARNED: AtomicBool = AtomicBool::new(false);
    if !WARNED.swap(true, Ordering::Relaxed) {
        warn!(
            "op-log rows have no start time (no 'start', 't_start' or 't_start_ns'): using the \
             time they are read, so replay timing and ordering information is lost"
        );
    }
    Utc::now()
}

/// Parse an RFC 3339 `start` value; the error names the (1-based) file line and the text.
fn parse_rfc3339_start(text: &str, column: &str, file_line: usize) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(text)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| {
            anyhow::anyhow!(
                "line {file_line}: invalid {column} value '{text}': {e} (expected an RFC 3339 timestamp such as 2026-01-01T00:00:00Z)"
            )
        })
}

/// Op names the object-store writer emits that `OpType` does not model. A row carrying one is
/// valid input that cannot be replayed exactly: its range, key list or other defining
/// parameters are not recorded (s3dlio issue #176).
const UNMODELLED_WRITER_OPS: &[&str] = &[
    "GET_RANGE",
    "DELETE_BATCH",
    "DELETE_PREFIX",
    "CREATE_CONTAINER",
    "DELETE_CONTAINER",
];

/// Where the explanation of unsupported op names lives.
const UNSUPPORTED_OP_DOC: &str = "docs/supplemental/OPERATION_LOGGING.md (Known gaps)";

/// The outermost error context for a row whose op name `OpType` does not parse. Names the
/// 1-based file line and the 0-based data row index. A known writer op is reported as
/// unsupported for exact replay; any other name as unknown. The error is only ever reported;
/// no row is skipped or filtered.
fn unsupported_op_context(op_str: &str, file_line: usize, row_index: usize) -> String {
    if UNMODELLED_WRITER_OPS
        .iter()
        .any(|name| name.eq_ignore_ascii_case(op_str))
    {
        format!(
            "line {file_line} (row index {row_index}): operation {op_str} is not supported for exact replay (its range/key parameters are not recorded); see {UNSUPPORTED_OP_DOC}"
        )
    } else {
        format!(
            "line {file_line} (row index {row_index}): unknown operation '{op_str}' (the reader models GET, PUT, DELETE, LIST and STAT/HEAD)"
        )
    }
}

/// Parse a JSON value into an OpLogEntry
fn parse_json_entry(json: &serde_json::Value, line_num: usize) -> Result<OpLogEntry> {
    let get_str = |key: &str| -> Option<String> {
        json.get(key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    };

    let get_u64 = |key: &str| -> Option<u64> { json.get(key).and_then(|v| v.as_u64()) };

    // Parse operation type (handle both "operation" and "op" field names)
    let op_str = get_str("operation")
        .or_else(|| get_str("op"))
        .ok_or_else(|| anyhow::anyhow!("Missing 'operation' or 'op' field"))?;

    let op = op_str
        .parse::<OpType>()
        .with_context(|| format!("Invalid operation type: {}", op_str))
        .with_context(|| unsupported_op_context(&op_str, line_num + 1, line_num))?;

    // Parse other fields with sensible defaults
    let idx = get_u64("idx").unwrap_or(line_num as u64);
    let bytes = get_u64("bytes").unwrap_or(0);
    let endpoint = get_str("endpoint").unwrap_or_default();
    let file = get_str("file").unwrap_or_default();
    let duration_ns = get_u64("duration_ns");
    let error = get_str("error");

    // Parse timestamp: `start` / `t_start` (RFC 3339 strings) or `t_start_ns` (integer
    // nanoseconds since the Unix epoch). A row with none of them is untimed (the read time);
    // a present but unparsable value is an error naming the line.
    let file_line = line_num + 1;
    let start = match ["start", "t_start"]
        .iter()
        .find_map(|k| json.get(*k).map(|v| (*k, v)))
    {
        Some((key, value)) => {
            let text = value.as_str().ok_or_else(|| {
                anyhow::anyhow!(
                    "line {file_line}: invalid {key} value {value}: expected an RFC 3339 string"
                )
            })?;
            parse_rfc3339_start(text, key, file_line)?
        }
        None => match json.get("t_start_ns") {
            Some(value) => {
                let ns = value.as_i64().ok_or_else(|| {
                    anyhow::anyhow!(
                        "line {file_line}: invalid t_start_ns value {value}: expected integer nanoseconds since the Unix epoch"
                    )
                })?;
                DateTime::<Utc>::from_timestamp_nanos(ns)
            }
            None => untimed_start(),
        },
    };

    Ok(OpLogEntry {
        idx,
        op,
        bytes,
        endpoint,
        file,
        start,
        duration_ns,
        error,
    })
}

/// Create column name to index mapping (case-insensitive, handles variations)
fn create_column_mapping(headers: &[&str]) -> Result<HashMap<String, usize>> {
    let mut mapping = HashMap::new();

    for (idx, &header) in headers.iter().enumerate() {
        mapping.insert(header.to_lowercase(), idx);
    }

    // Ensure we have operation field (handle both "operation" and "op")
    if !mapping.contains_key("operation") && !mapping.contains_key("op") {
        anyhow::bail!("TSV header must contain 'operation' or 'op' column");
    }

    Ok(mapping)
}

/// Parse a single TSV entry using column mapping
fn parse_tsv_entry(
    fields: &[&str],
    col_mapping: &HashMap<String, usize>,
    line_num: usize,
) -> Result<OpLogEntry> {
    let get_field = |name: &str| -> Option<&str> {
        col_mapping
            .get(name)
            .and_then(|&idx| fields.get(idx))
            .copied()
    };

    let get_u64 = |name: &str| -> Result<Option<u64>> {
        match get_field(name) {
            Some(s) if !s.is_empty() => s
                .parse::<u64>()
                .with_context(|| format!("Invalid {} value: {}", name, s))
                .map(Some),
            _ => Ok(None),
        }
    };

    // Parse operation type (handle both column names)
    let op_str = get_field("operation")
        .or_else(|| get_field("op"))
        .ok_or_else(|| anyhow::anyhow!("Missing 'operation' or 'op' field"))?;

    let op = op_str
        .parse::<OpType>()
        .with_context(|| format!("Invalid operation type: {}", op_str))
        .with_context(|| unsupported_op_context(op_str, line_num + 2, line_num))?;

    // Parse other fields
    let idx = get_u64("idx")?.unwrap_or(line_num as u64);
    let bytes = get_u64("bytes")?.unwrap_or(0);
    let endpoint = get_field("endpoint").unwrap_or("").to_string();
    let file = get_field("file").unwrap_or("").to_string();
    let duration_ns = get_u64("duration_ns")?;
    let error = get_field("error")
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());

    // Parse timestamp: `start` / `t_start` (RFC 3339) or `t_start_ns` (integer nanoseconds
    // since the Unix epoch). A log with none of these columns has no timing at all (the
    // minimal dl-driver format) and takes the read time; a row that has the column but
    // lacks the value or holds an unparsable one is an error naming its line.
    let file_line = line_num + 2; // line_num counts data lines; the header is line 1
    let start = if let Some(key) = ["start", "t_start"]
        .into_iter()
        .find(|k| col_mapping.contains_key(*k))
    {
        let text = get_field(key)
            .ok_or_else(|| anyhow::anyhow!("line {file_line}: missing {key} value"))?;
        parse_rfc3339_start(text, key, file_line)?
    } else if col_mapping.contains_key("t_start_ns") {
        let text = get_field("t_start_ns")
            .ok_or_else(|| anyhow::anyhow!("line {file_line}: missing t_start_ns value"))?;
        let ns = text.parse::<i64>().map_err(|e| {
            anyhow::anyhow!(
                "line {file_line}: invalid t_start_ns value '{text}': {e} (expected integer nanoseconds since the Unix epoch)"
            )
        })?;
        DateTime::<Utc>::from_timestamp_nanos(ns)
    } else {
        untimed_start()
    };

    Ok(OpLogEntry {
        idx,
        op,
        bytes,
        endpoint,
        file,
        start,
        duration_ns,
        error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_format_detection() {
        assert_eq!(
            OpLogFormat::from_path("test.jsonl").unwrap(),
            OpLogFormat::Jsonl
        );
        assert_eq!(
            OpLogFormat::from_path("test.jsonl.zst").unwrap(),
            OpLogFormat::Jsonl
        );
        assert_eq!(
            OpLogFormat::from_path("test.tsv").unwrap(),
            OpLogFormat::Tsv
        );
        assert_eq!(
            OpLogFormat::from_path("test.tsv.zst").unwrap(),
            OpLogFormat::Tsv
        );
        assert_eq!(
            OpLogFormat::from_path("test.csv").unwrap(),
            OpLogFormat::Tsv
        );
        assert_eq!(
            OpLogFormat::from_path("test.csv.zst").unwrap(),
            OpLogFormat::Tsv
        );
        assert!(OpLogFormat::from_path("test.txt").is_err());
    }

    #[test]
    fn test_jsonl_parsing() {
        let mut file = NamedTempFile::with_suffix(".jsonl").unwrap();
        writeln!(file, r#"{{"operation": "GET", "file": "test.dat", "bytes": 1024, "duration_ns": 5000000, "start": "2025-01-01T00:00:00Z"}}"#).unwrap();
        writeln!(file, r#"{{"op": "PUT", "file": "test2.dat", "bytes": 2048, "start": "2025-01-01T00:00:01Z"}}"#).unwrap();
        file.flush().unwrap();

        let reader = OpLogReader::from_file(file.path()).unwrap();
        assert_eq!(reader.len(), 2);

        let entries = reader.entries();
        assert_eq!(entries[0].op, OpType::GET);
        assert_eq!(entries[0].bytes, 1024);
        assert_eq!(entries[1].op, OpType::PUT);
        assert_eq!(entries[1].bytes, 2048);
    }

    #[test]
    fn test_tsv_parsing() {
        let mut file = NamedTempFile::with_suffix(".tsv").unwrap();
        writeln!(file, "operation\tfile\tbytes\tduration_ns\tstart").unwrap();
        writeln!(file, "GET\ttest.dat\t1024\t5000000\t2025-01-01T00:00:00Z").unwrap();
        writeln!(file, "PUT\ttest2.dat\t2048\t\t2025-01-01T00:00:01Z").unwrap();
        file.flush().unwrap();

        let reader = OpLogReader::from_file(file.path()).unwrap();
        assert_eq!(reader.len(), 2);

        let entries = reader.entries();
        assert_eq!(entries[0].op, OpType::GET);
        assert_eq!(entries[0].bytes, 1024);
        assert_eq!(entries[0].duration_ns, Some(5000000));
        assert_eq!(entries[1].op, OpType::PUT);
        assert_eq!(entries[1].bytes, 2048);
        assert_eq!(entries[1].duration_ns, None);
    }

    #[test]
    fn test_filter_operations() {
        let mut file = NamedTempFile::with_suffix(".tsv").unwrap();
        writeln!(file, "op\tfile\tbytes\tstart").unwrap();
        writeln!(file, "GET\ttest1.dat\t100\t2025-01-01T00:00:00Z").unwrap();
        writeln!(file, "PUT\ttest2.dat\t200\t2025-01-01T00:00:01Z").unwrap();
        writeln!(file, "GET\ttest3.dat\t300\t2025-01-01T00:00:02Z").unwrap();
        file.flush().unwrap();

        let reader = OpLogReader::from_file(file.path()).unwrap();
        let gets = reader.filter_operations(&[OpType::GET]);
        assert_eq!(gets.len(), 2);
        assert!(gets.iter().all(|e| e.op == OpType::GET));
    }

    // =============================================================================
    // Streaming Reader Tests
    // =============================================================================

    #[test]
    fn test_streaming_jsonl() {
        let mut file = NamedTempFile::with_suffix(".jsonl").unwrap();
        writeln!(file, r#"{{"operation": "GET", "file": "test.dat", "bytes": 1024, "start": "2025-01-01T00:00:00Z"}}"#).unwrap();
        writeln!(file, r#"{{"op": "PUT", "file": "test2.dat", "bytes": 2048, "start": "2025-01-01T00:00:01Z"}}"#).unwrap();
        file.flush().unwrap();

        let stream = OpLogStreamReader::from_file(file.path()).unwrap();
        let entries: Vec<_> = stream.map(|r| r.unwrap()).collect();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].op, OpType::GET);
        assert_eq!(entries[0].bytes, 1024);
        assert_eq!(entries[1].op, OpType::PUT);
        assert_eq!(entries[1].bytes, 2048);
    }

    #[test]
    fn test_streaming_tsv() {
        let mut file = NamedTempFile::with_suffix(".tsv").unwrap();
        writeln!(file, "operation\tfile\tbytes\tstart").unwrap();
        writeln!(file, "GET\ttest.dat\t1024\t2025-01-01T00:00:00Z").unwrap();
        writeln!(file, "PUT\ttest2.dat\t2048\t2025-01-01T00:00:01Z").unwrap();
        file.flush().unwrap();

        let stream = OpLogStreamReader::from_file(file.path()).unwrap();
        let entries: Vec<_> = stream.map(|r| r.unwrap()).collect();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].op, OpType::GET);
        assert_eq!(entries[1].op, OpType::PUT);
    }

    #[test]
    fn test_streaming_with_errors() {
        let mut file = NamedTempFile::with_suffix(".jsonl").unwrap();
        writeln!(file, r#"{{"operation": "GET", "file": "test.dat", "bytes": 1024, "start": "2025-01-01T00:00:00Z"}}"#).unwrap();
        writeln!(file, r#"{{"invalid json"#).unwrap(); // Invalid JSON
        writeln!(file, r#"{{"op": "PUT", "file": "test2.dat", "bytes": 2048, "start": "2025-01-01T00:00:01Z"}}"#).unwrap();
        file.flush().unwrap();

        let stream = OpLogStreamReader::from_file(file.path()).unwrap();
        let results: Vec<_> = stream.collect();

        assert_eq!(results.len(), 3);
        assert!(results[0].is_ok());
        assert!(results[1].is_err()); // Error for invalid JSON
        assert!(results[2].is_ok());
    }

    #[test]
    fn test_streaming_compressed() {
        use std::io::Write;

        let mut file = NamedTempFile::with_suffix(".jsonl.zst").unwrap();
        let mut encoder = zstd::stream::Encoder::new(file.as_file_mut(), 1).unwrap();
        writeln!(encoder, r#"{{"operation": "GET", "file": "test.dat", "bytes": 1024, "start": "2025-01-01T00:00:00Z"}}"#).unwrap();
        writeln!(encoder, r#"{{"op": "PUT", "file": "test2.dat", "bytes": 2048, "start": "2025-01-01T00:00:01Z"}}"#).unwrap();
        encoder.finish().unwrap();
        file.flush().unwrap();

        let stream = OpLogStreamReader::from_file(file.path()).unwrap();
        let entries: Vec<_> = stream.map(|r| r.unwrap()).collect();

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].op, OpType::GET);
        assert_eq!(entries[1].op, OpType::PUT);
    }

    #[test]
    fn test_streaming_take() {
        let mut file = NamedTempFile::with_suffix(".tsv").unwrap();
        writeln!(file, "op\tfile\tbytes\tstart").unwrap();
        for i in 0..100 {
            writeln!(
                file,
                "GET\ttest{}.dat\t{}\t2025-01-01T00:00:00Z",
                i,
                i * 1000
            )
            .unwrap();
        }
        file.flush().unwrap();

        let stream = OpLogStreamReader::from_file(file.path()).unwrap();
        let entries: Vec<_> = stream.take(10).map(|r| r.unwrap()).collect();

        assert_eq!(entries.len(), 10);
        assert_eq!(entries[0].bytes, 0);
        assert_eq!(entries[9].bytes, 9000);
    }

    #[test]
    fn test_backward_compatibility() {
        // Ensure OpLogReader still works exactly the same
        let mut file = NamedTempFile::with_suffix(".tsv").unwrap();
        writeln!(file, "operation\tfile\tbytes\tstart").unwrap();
        writeln!(file, "GET\ttest.dat\t1024\t2025-01-01T00:00:00Z").unwrap();
        writeln!(file, "PUT\ttest2.dat\t2048\t2025-01-01T00:00:01Z").unwrap();
        file.flush().unwrap();

        let reader = OpLogReader::from_file(file.path()).unwrap();
        assert_eq!(reader.len(), 2);

        let stream = OpLogStreamReader::from_file(file.path()).unwrap();
        let stream_count = stream.count();
        assert_eq!(stream_count, 2);
    }
}
