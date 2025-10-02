//! Operation log reader with multi-format support
//!
//! Provides robust parsing of operation logs in JSONL and TSV formats,
//! with automatic compression detection and header-driven column mapping.

use anyhow::{Context, Result};
use chrono::DateTime;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use tracing::{debug, info, warn};

use crate::types::{OpLogEntry, OpType};

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
            || path_str.ends_with(".csv.zst") {
            Ok(OpLogFormat::Tsv)
        } else {
            anyhow::bail!("Unsupported file format: {}. Supported: .jsonl[.zst], .tsv[.zst], .csv[.zst]", path_str)
        }
    }
}

/// Reader for op-log files with automatic format and compression detection
pub struct OpLogReader {
    entries: Vec<OpLogEntry>,
}

impl OpLogReader {
    /// Load op-log from file with automatic format and compression detection
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let format = OpLogFormat::from_path(path)?;
        let is_compressed = path.extension().map_or(false, |ext| ext == "zst");

        info!("Loading op-log from {:?} (format: {:?}, compressed: {})", path, format, is_compressed);

        let reader = open_reader(path, is_compressed)?;
        let entries = match format {
            OpLogFormat::Jsonl => parse_jsonl(reader)?,
            OpLogFormat::Tsv => parse_tsv(reader)?,
        };

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

/// Open a reader with optional zstd decompression
fn open_reader<P: AsRef<Path>>(path: P, is_compressed: bool) -> Result<Box<dyn BufRead>> {
    let file = File::open(&path)
        .with_context(|| format!("Failed to open file: {}", path.as_ref().display()))?;

    if is_compressed {
        let decoder = zstd::stream::read::Decoder::new(file)
            .with_context(|| "Failed to create zstd decoder")?;
        Ok(Box::new(BufReader::new(decoder)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

/// Parse JSONL format (one JSON object per line)
fn parse_jsonl(reader: Box<dyn BufRead>) -> Result<Vec<OpLogEntry>> {
    let mut entries = Vec::new();
    let mut skipped = 0;
    
    for (line_num, line) in reader.lines().enumerate() {
        let line = line.with_context(|| format!("Failed to read line {}", line_num + 1))?;
        
        // Skip empty lines and comments
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Try to parse as JSON
        match serde_json::from_str::<serde_json::Value>(&line) {
            Ok(json) => {
                match parse_json_entry(&json, line_num) {
                    Ok(entry) => entries.push(entry),
                    Err(e) => {
                        warn!("Failed to parse JSON entry on line {}: {}", line_num + 1, e);
                        skipped += 1;
                    }
                }
            }
            Err(e) => {
                warn!("Failed to parse JSON on line {}: {}", line_num + 1, e);
                skipped += 1;
            }
        }
    }

    if skipped > 0 {
        info!("Skipped {} invalid JSONL entries", skipped);
    }

    Ok(entries)
}

/// Parse a JSON value into an OpLogEntry
fn parse_json_entry(json: &serde_json::Value, line_num: usize) -> Result<OpLogEntry> {
    let get_str = |key: &str| -> Option<String> {
        json.get(key).and_then(|v| v.as_str()).map(|s| s.to_string())
    };

    let get_u64 = |key: &str| -> Option<u64> {
        json.get(key).and_then(|v| v.as_u64())
    };

    // Parse operation type (handle both "operation" and "op" field names)
    let op_str = get_str("operation")
        .or_else(|| get_str("op"))
        .ok_or_else(|| anyhow::anyhow!("Missing 'operation' or 'op' field"))?;
    
    let op = op_str.parse::<OpType>()
        .with_context(|| format!("Invalid operation type: {}", op_str))?;

    // Parse other fields with sensible defaults
    let idx = get_u64("idx").unwrap_or(line_num as u64);
    let bytes = get_u64("bytes").unwrap_or(0);
    let endpoint = get_str("endpoint").unwrap_or_default();
    let file = get_str("file").unwrap_or_default();
    let duration_ns = get_u64("duration_ns");
    let error = get_str("error");

    // Parse timestamp (try multiple field names)
    let start = get_str("start")
        .or_else(|| get_str("t_start"))
        .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
        .map(|dt| dt.into())
        .unwrap_or_else(|| chrono::Utc::now());

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

/// Parse TSV format with header-driven column mapping
fn parse_tsv(reader: Box<dyn BufRead>) -> Result<Vec<OpLogEntry>> {
    let mut lines = reader.lines();
    
    // Read header to determine column mapping
    let header_line = lines.next()
        .ok_or_else(|| anyhow::anyhow!("TSV file is empty"))?
        .with_context(|| "Failed to read TSV header")?;
    
    let headers: Vec<&str> = header_line.split('\t').collect();
    let col_mapping = create_column_mapping(&headers)?;

    debug!("TSV column mapping: {:?}", col_mapping);

    let mut entries = Vec::new();
    let mut skipped = 0;
    
    for (line_num, line) in lines.enumerate() {
        let line = line.with_context(|| format!("Failed to read line {}", line_num + 2))?;
        
        // Skip empty lines and comments
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let fields: Vec<&str> = line.split('\t').collect();
        
        match parse_tsv_entry(&fields, &col_mapping, line_num) {
            Ok(entry) => entries.push(entry),
            Err(e) => {
                warn!("Failed to parse TSV entry on line {}: {}", line_num + 2, e);
                skipped += 1;
            }
        }
    }

    if skipped > 0 {
        info!("Skipped {} invalid TSV entries", skipped);
    }

    Ok(entries)
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
        col_mapping.get(name).and_then(|&idx| fields.get(idx)).copied()
    };

    let get_u64 = |name: &str| -> Result<Option<u64>> {
        match get_field(name) {
            Some(s) if !s.is_empty() => {
                s.parse::<u64>()
                    .with_context(|| format!("Invalid {} value: {}", name, s))
                    .map(Some)
            }
            _ => Ok(None),
        }
    };

    // Parse operation type (handle both column names)
    let op_str = get_field("operation")
        .or_else(|| get_field("op"))
        .ok_or_else(|| anyhow::anyhow!("Missing 'operation' or 'op' field"))?;
    
    let op = op_str.parse::<OpType>()
        .with_context(|| format!("Invalid operation type: {}", op_str))?;

    // Parse other fields
    let idx = get_u64("idx")?.unwrap_or(line_num as u64);
    let bytes = get_u64("bytes")?.unwrap_or(0);
    let endpoint = get_field("endpoint").unwrap_or("").to_string();
    let file = get_field("file").unwrap_or("").to_string();
    let duration_ns = get_u64("duration_ns")?;
    let error = get_field("error").filter(|s| !s.is_empty()).map(|s| s.to_string());

    // Parse timestamp
    let start = get_field("start")
        .or_else(|| get_field("t_start"))
        .and_then(|s| DateTime::parse_from_rfc3339(s).ok())
        .map(|dt| dt.into())
        .unwrap_or_else(|| chrono::Utc::now());

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
        assert_eq!(OpLogFormat::from_path("test.jsonl").unwrap(), OpLogFormat::Jsonl);
        assert_eq!(OpLogFormat::from_path("test.jsonl.zst").unwrap(), OpLogFormat::Jsonl);
        assert_eq!(OpLogFormat::from_path("test.tsv").unwrap(), OpLogFormat::Tsv);
        assert_eq!(OpLogFormat::from_path("test.tsv.zst").unwrap(), OpLogFormat::Tsv);
        assert_eq!(OpLogFormat::from_path("test.csv").unwrap(), OpLogFormat::Tsv);
        assert_eq!(OpLogFormat::from_path("test.csv.zst").unwrap(), OpLogFormat::Tsv);
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
}
