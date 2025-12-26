// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Integration tests for s3dlio-oplog crate
//!
//! These tests verify the public API and end-to-end functionality
//! of operation log replay.

use anyhow::Result;
use async_trait::async_trait;
use s3dlio_oplog::{OpExecutor, OpLogReader, OpType, ReplayConfig, replay_workload, translate_uri};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use tempfile::NamedTempFile;
use std::io::Write;

/// Mock executor for testing without real storage backends
struct TestExecutor {
    operations: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl OpExecutor for TestExecutor {
    async fn get(&self, uri: &str) -> Result<()> {
        self.operations.lock().unwrap().push(format!("GET {}", uri));
        Ok(())
    }

    async fn put(&self, uri: &str, bytes: usize) -> Result<()> {
        self.operations.lock().unwrap().push(format!("PUT {} ({} bytes)", uri, bytes));
        Ok(())
    }

    async fn delete(&self, uri: &str) -> Result<()> {
        self.operations.lock().unwrap().push(format!("DELETE {}", uri));
        Ok(())
    }

    async fn list(&self, uri: &str) -> Result<()> {
        self.operations.lock().unwrap().push(format!("LIST {}", uri));
        Ok(())
    }

    async fn stat(&self, uri: &str) -> Result<()> {
        self.operations.lock().unwrap().push(format!("STAT {}", uri));
        Ok(())
    }
}

#[tokio::test]
async fn test_replay_with_custom_executor() -> Result<()> {
    // Create a temporary TSV file with test operations
    let mut file = NamedTempFile::with_suffix(".tsv")?;
    writeln!(file, "idx\top\tbytes\tendpoint\tfile\tstart\tduration_ns\terror")?;
    writeln!(file, "0\tGET\t1048576\thttp://10.0.0.1:8000\t/data/file1.bin\t2024-01-15T10:30:45.123456Z\t1000000\t")?;
    writeln!(file, "1\tPUT\t2097152\thttp://10.0.0.1:8000\t/data/file2.bin\t2024-01-15T10:30:45.223456Z\t2000000\t")?;
    file.flush()?;

    let operations = Arc::new(Mutex::new(Vec::new()));
    let executor = Arc::new(TestExecutor { operations: operations.clone() });

    let config = ReplayConfig {
        op_log_path: file.path().to_path_buf(),
        target_uri: Some("s3://test-bucket/".to_string()),
        speed: 1000.0,  // Very fast for testing
        continue_on_error: false,
        filter_ops: None,
    };

    replay_workload(config, executor).await?;

    let ops = operations.lock().unwrap();
    assert_eq!(ops.len(), 2);
    assert!(ops[0].contains("GET"));
    assert!(ops[1].contains("PUT"));

    Ok(())
}

#[tokio::test]
async fn test_operation_filtering() -> Result<()> {
    // Create a temporary TSV file with mixed operations
    let mut file = NamedTempFile::with_suffix(".tsv")?;
    writeln!(file, "idx\top\tbytes\tendpoint\tfile\tstart\tduration_ns\terror")?;
    writeln!(file, "0\tGET\t1048576\thttp://10.0.0.1:8000\t/data/file1.bin\t2024-01-15T10:30:45.123456Z\t1000000\t")?;
    writeln!(file, "1\tPUT\t2097152\thttp://10.0.0.1:8000\t/data/file2.bin\t2024-01-15T10:30:45.223456Z\t2000000\t")?;
    writeln!(file, "2\tGET\t524288\thttp://10.0.0.1:8000\t/data/file3.bin\t2024-01-15T10:30:45.323456Z\t500000\t")?;
    file.flush()?;

    let operations = Arc::new(Mutex::new(Vec::new()));
    let executor = Arc::new(TestExecutor { operations: operations.clone() });

    // Filter to only GET operations
    let config = ReplayConfig {
        op_log_path: file.path().to_path_buf(),
        target_uri: Some("s3://test-bucket/".to_string()),
        speed: 1000.0,
        continue_on_error: false,
        filter_ops: Some(vec![OpType::GET]),
    };

    replay_workload(config, executor).await?;

    let ops = operations.lock().unwrap();
    assert_eq!(ops.len(), 2, "Should only replay GET operations");
    assert!(ops[0].contains("GET"));
    assert!(ops[1].contains("GET"));

    Ok(())
}

#[test]
fn test_uri_translation() {
    // Test basic translation
    let result = translate_uri(
        "http://10.0.0.1:8000/data/file.bin",
        "http://10.0.0.1:8000",
        "s3://my-bucket"
    ).unwrap();
    assert_eq!(result, "s3://my-bucket/data/file.bin");

    // Test cross-backend translation
    let result = translate_uri(
        "s3://old-bucket/data/file.bin",
        "s3://old-bucket",
        "az://new-container"
    ).unwrap();
    assert_eq!(result, "az://new-container/data/file.bin");

    // Test with trailing slashes
    let result = translate_uri(
        "file:///local/data/file.bin",
        "file:///local/",
        "direct:///fast-storage/"
    ).unwrap();
    assert_eq!(result, "direct:///fast-storage/data/file.bin");
}

#[test]
fn test_oplog_reader_tsv() -> Result<()> {
    // Create a temporary TSV file
    let mut file = NamedTempFile::with_suffix(".tsv")?;
    writeln!(file, "idx\top\tbytes\tendpoint\tfile\tstart\tduration_ns\terror")?;
    writeln!(file, "0\tGET\t1048576\thttp://10.0.0.1:8000\t/data/file1.bin\t2024-01-15T10:30:45.123456Z\t1000000\t")?;
    writeln!(file, "1\tPUT\t2097152\thttp://10.0.0.1:8000\t/data/file2.bin\t2024-01-15T10:30:45.223456Z\t2000000\t")?;
    file.flush()?;

    let reader = OpLogReader::from_file(file.path().to_path_buf())?;
    let entries = reader.entries();

    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].op, OpType::GET);
    assert_eq!(entries[0].bytes, 1048576);
    assert_eq!(entries[1].op, OpType::PUT);
    assert_eq!(entries[1].bytes, 2097152);

    Ok(())
}

#[test]
fn test_oplog_reader_filtering() -> Result<()> {
    // Create a temporary TSV file
    let mut file = NamedTempFile::with_suffix(".tsv")?;
    writeln!(file, "idx\top\tbytes\tendpoint\tfile\tstart\tduration_ns\terror")?;
    writeln!(file, "0\tGET\t1048576\thttp://10.0.0.1:8000\t/data/file1.bin\t2024-01-15T10:30:45.123456Z\t1000000\t")?;
    writeln!(file, "1\tPUT\t2097152\thttp://10.0.0.1:8000\t/data/file2.bin\t2024-01-15T10:30:45.223456Z\t2000000\t")?;
    writeln!(file, "2\tDELETE\t0\thttp://10.0.0.1:8000\t/data/file3.bin\t2024-01-15T10:30:45.323456Z\t500000\t")?;
    file.flush()?;

    let reader = OpLogReader::from_file(file.path().to_path_buf())?;
    let filtered = reader.filter_operations(&[OpType::GET, OpType::PUT]);

    assert_eq!(filtered.len(), 2);
    assert_eq!(filtered[0].op, OpType::GET);
    assert_eq!(filtered[1].op, OpType::PUT);

    Ok(())
}

#[test]
fn test_optype_parsing() {
    assert_eq!("GET".parse::<OpType>().unwrap(), OpType::GET);
    assert_eq!("get".parse::<OpType>().unwrap(), OpType::GET);
    assert_eq!("Put".parse::<OpType>().unwrap(), OpType::PUT);
    assert_eq!("DELETE".parse::<OpType>().unwrap(), OpType::DELETE);
    assert_eq!("list".parse::<OpType>().unwrap(), OpType::LIST);
    assert_eq!("STAT".parse::<OpType>().unwrap(), OpType::STAT);

    // Test invalid operation
    assert!("INVALID".parse::<OpType>().is_err());
}

#[tokio::test]
async fn test_continue_on_error() -> Result<()> {
    // Create executor that fails on PUT
    struct FailingExecutor {
        operations: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl OpExecutor for FailingExecutor {
        async fn get(&self, uri: &str) -> Result<()> {
            self.operations.lock().unwrap().push(format!("GET {}", uri));
            Ok(())
        }

        async fn put(&self, _uri: &str, _bytes: usize) -> Result<()> {
            Err(anyhow::anyhow!("Simulated PUT failure"))
        }

        async fn delete(&self, uri: &str) -> Result<()> {
            self.operations.lock().unwrap().push(format!("DELETE {}", uri));
            Ok(())
        }

        async fn list(&self, uri: &str) -> Result<()> {
            self.operations.lock().unwrap().push(format!("LIST {}", uri));
            Ok(())
        }

        async fn stat(&self, uri: &str) -> Result<()> {
            self.operations.lock().unwrap().push(format!("STAT {}", uri));
            Ok(())
        }
    }

    let mut file = NamedTempFile::with_suffix(".tsv")?;
    writeln!(file, "idx\top\tbytes\tendpoint\tfile\tstart\tduration_ns\terror")?;
    writeln!(file, "0\tGET\t1048576\thttp://10.0.0.1:8000\t/data/file1.bin\t2024-01-15T10:30:45.123456Z\t1000000\t")?;
    writeln!(file, "1\tPUT\t2097152\thttp://10.0.0.1:8000\t/data/file2.bin\t2024-01-15T10:30:45.223456Z\t2000000\t")?;
    writeln!(file, "2\tGET\t524288\thttp://10.0.0.1:8000\t/data/file3.bin\t2024-01-15T10:30:45.323456Z\t500000\t")?;
    file.flush()?;

    let operations = Arc::new(Mutex::new(Vec::new()));
    let executor = Arc::new(FailingExecutor { operations: operations.clone() });

    let config = ReplayConfig {
        op_log_path: file.path().to_path_buf(),
        target_uri: Some("s3://test-bucket/".to_string()),
        speed: 1000.0,
        continue_on_error: true,  // Continue despite PUT failure
        filter_ops: None,
    };

    // Should succeed despite PUT failure because continue_on_error=true
    replay_workload(config, executor).await?;

    let ops = operations.lock().unwrap();
    assert_eq!(ops.len(), 2, "Should have executed 2 GET operations despite PUT failure");

    Ok(())
}

// ============================================================================
// COMPATIBILITY TESTS FOR s3-bench AND dl-driver MIGRATION
// ============================================================================

/// Test parsing s3-bench's full 13-column TSV format
/// s3-bench format: idx, thread, op, client_id, n_objects, bytes, endpoint, file, error, start, first_byte, end, duration_ns
#[tokio::test]
async fn test_s3_bench_13_column_tsv() -> Result<()> {
    let mut file = NamedTempFile::with_suffix(".tsv")?;
    
    // Write s3-bench format header and data
    writeln!(file, "idx\tthread\top\tclient_id\tn_objects\tbytes\tendpoint\tfile\terror\tstart\tfirst_byte\tend\tduration_ns")?;
    writeln!(file, "0\t1\tGET\tclient-1\t1\t1048576\thttp://s3.local:9000\t/bucket/data/file1.bin\t\t1000000000\t1000500000\t1001000000\t1000000000")?;
    writeln!(file, "1\t2\tPUT\tclient-1\t1\t2097152\thttp://s3.local:9000\t/bucket/data/file2.bin\t\t1002000000\t1002500000\t1003000000\t1000000000")?;
    writeln!(file, "2\t1\tDELETE\tclient-1\t1\t0\thttp://s3.local:9000\t/bucket/data/file3.bin\t\t1004000000\t1004100000\t1004200000\t200000000")?;
    file.flush()?;

    let reader = OpLogReader::from_file(file.path())?;
    let entries = reader.entries();

    assert_eq!(entries.len(), 3, "Should parse all 3 operations from 13-column format");
    assert_eq!(entries[0].op, OpType::GET);
    assert_eq!(&entries[0].file, "/bucket/data/file1.bin");
    assert_eq!(entries[0].bytes, 1048576);
    
    assert_eq!(entries[1].op, OpType::PUT);
    assert_eq!(entries[1].bytes, 2097152);
    
    assert_eq!(entries[2].op, OpType::DELETE);

    Ok(())
}

/// Test s3-bench URI translation pattern: strip endpoint prefix and add target prefix
#[test]
fn test_s3_bench_uri_translation() -> Result<()> {
    // s3-bench strips endpoint prefix and adds target prefix
    // Example: "/bucket/data/file.bin" + "s3://newbucket" = "s3://newbucket/data/file.bin"
    
    let source = "file:///bucket/data/file.bin";
    let target_prefix = "s3://production-bucket";
    
    let result = translate_uri(source, "file://", target_prefix)?;
    assert_eq!(result, "s3://production-bucket/bucket/data/file.bin");

    // Test stripping S3 endpoint
    let source = "s3://oldbucket/prefix/data.bin";
    let result = translate_uri(source, "s3://oldbucket", "az://newstorage/container")?;
    assert_eq!(result, "az://newstorage/container/prefix/data.bin");

    Ok(())
}

/// Test dl-driver base URI joining pattern
#[tokio::test]
async fn test_dl_driver_base_uri_joining() -> Result<()> {
    let mut file = NamedTempFile::with_suffix(".tsv")?;
    
    // dl-driver logs use relative paths
    writeln!(file, "operation\tfile\tbytes\tt_start_ns")?;
    writeln!(file, "GET\ttrain_file_001.npz\t1048576\t1000000000")?;
    writeln!(file, "GET\ttrain_file_002.npz\t1048576\t1001000000")?;
    file.flush()?;

    let reader = OpLogReader::from_file(file.path())?;
    let entries = reader.entries();

    // Simulate dl-driver base URI joining
    let base_uri = "file:///tmp/replay_test";
    for entry in entries {
        let file_path = &entry.file;
        let complete_uri = if file_path.starts_with('/') {
            format!("{}{}", base_uri, file_path)
        } else {
            format!("{}/{}", base_uri.trim_end_matches('/'), file_path)
        };
        
        assert!(complete_uri.starts_with("file:///tmp/replay_test/"));
        assert!(complete_uri.contains("train_file_"));
    }

    Ok(())
}

/// Test HEAD operation as alias for STAT (s3-bench compatibility)
#[tokio::test]
async fn test_head_to_stat_alias() -> Result<()> {
    let mut file = NamedTempFile::with_suffix(".tsv")?;
    
    writeln!(file, "idx\top\tbytes\tendpoint\tfile\tstart\tduration_ns\terror")?;
    writeln!(file, "0\tHEAD\t0\thttp://s3.local\t/bucket/object.bin\t1000000000\t5000000\t")?;
    file.flush()?;

    let operations = Arc::new(Mutex::new(Vec::new()));
    let executor = Arc::new(TestExecutor {
        operations: operations.clone(),
    });

    let config = ReplayConfig {
        op_log_path: file.path().to_path_buf(),
        target_uri: None,
        speed: 1.0,
        continue_on_error: false,
        filter_ops: None,
    };

    replay_workload(config, executor).await?;

    let ops = operations.lock().unwrap();
    // HEAD should be treated as STAT
    assert_eq!(ops.len(), 1);
    assert!(ops[0].starts_with("STAT"), "HEAD should map to STAT operation");

    Ok(())
}

/// Test speed multiplier for timeline scheduling (s3-bench feature)
#[tokio::test]
async fn test_speed_multiplier_timing() -> Result<()> {
    let mut file = NamedTempFile::with_suffix(".tsv")?;
    
    // Operations with 1 second gap - use RFC3339 format
    writeln!(file, "idx\top\tbytes\tendpoint\tfile\tstart\tduration_ns\terror")?;
    writeln!(file, "0\tGET\t1024\thttp://s3\t/file1\t2025-01-01T00:00:00Z\t5000000\t")?;
    writeln!(file, "1\tGET\t1024\thttp://s3\t/file2\t2025-01-01T00:00:01Z\t5000000\t")?;
    file.flush()?;

    let reader = OpLogReader::from_file(file.path())?;
    let entries = reader.entries();

    // Calculate delay with speed multiplier (simulate s3-bench logic)
    let speed = 10.0; // 10x faster
    let gap_ns = entries[1].start.timestamp_nanos_opt().unwrap() - entries[0].start.timestamp_nanos_opt().unwrap();
    let gap_ms = gap_ns / 1_000_000;
    let adjusted_delay_ms = (gap_ms as f64 / speed) as u64;

    // 1000ms gap / 10x speed = 100ms delay
    assert_eq!(adjusted_delay_ms, 100, "Speed multiplier should reduce delay by 10x");

    Ok(())
}

/// Test extra columns are ignored gracefully
#[tokio::test]
async fn test_extra_columns_ignored() -> Result<()> {
    let mut file = NamedTempFile::with_suffix(".tsv")?;
    
    // Include extra columns not in our OpLogEntry struct
    writeln!(file, "idx\top\tbytes\tfile\tcustom_field\tthread_id\tstart\terror")?;
    writeln!(file, "0\tGET\t1024\t/data/file.bin\textra_value\t42\t1000000000\t")?;
    file.flush()?;

    let reader = OpLogReader::from_file(file.path())?;
    let entries = reader.entries();

    // Should parse successfully, ignoring unknown columns
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].op, OpType::GET);
    assert_eq!(entries[0].bytes, 1024);

    Ok(())
}

/// Test minimal 6-field format vs extended 13-field format compatibility
#[tokio::test]
async fn test_minimal_vs_extended_formats() -> Result<()> {
    // Test minimal format (our core fields)
    let mut minimal_file = NamedTempFile::with_suffix(".tsv")?;
    writeln!(minimal_file, "op\tfile\tbytes\tstart\tduration_ns\terror")?;
    writeln!(minimal_file, "GET\t/data/small.bin\t512\t1000000000\t1000000\t")?;
    minimal_file.flush()?;

    let minimal_reader = OpLogReader::from_file(minimal_file.path())?;
    assert_eq!(minimal_reader.entries().len(), 1);

    // Test extended format (s3-bench style)
    let mut extended_file = NamedTempFile::with_suffix(".tsv")?;
    writeln!(extended_file, "idx\tthread\top\tclient_id\tn_objects\tbytes\tendpoint\tfile\terror\tstart\tfirst_byte\tend\tduration_ns")?;
    writeln!(extended_file, "0\t1\tGET\tc1\t1\t512\thttp://s3\t/data/small.bin\t\t1000000000\t1000500000\t1001000000\t1000000")?;
    extended_file.flush()?;

    let extended_reader = OpLogReader::from_file(extended_file.path())?;
    assert_eq!(extended_reader.entries().len(), 1);

    // Both should produce equivalent OpLogEntry
    let minimal_entry = &minimal_reader.entries()[0];
    let extended_entry = &extended_reader.entries()[0];

    assert_eq!(minimal_entry.op, extended_entry.op);
    assert_eq!(minimal_entry.file, extended_entry.file);
    assert_eq!(minimal_entry.bytes, extended_entry.bytes);

    Ok(())
}

/// Test zstd compression auto-detection by file extension
#[tokio::test]
async fn test_zstd_auto_detection() -> Result<()> {
    use std::io::BufWriter;
    use zstd::stream::write::Encoder;

    // Create compressed TSV file
    let temp_file = NamedTempFile::with_suffix(".tsv.zst")?;
    let file = std::fs::File::create(temp_file.path())?;
    let mut encoder = Encoder::new(BufWriter::new(file), 3)?;
    
    writeln!(encoder, "op\tfile\tbytes\tstart\tduration_ns\terror")?;
    writeln!(encoder, "GET\t/compressed/file.bin\t2048\t1000000000\t5000000\t")?;
    encoder.finish()?;

    // Should automatically detect and decompress .zst extension
    let reader = OpLogReader::from_file(temp_file.path())?;
    let entries = reader.entries();

    assert_eq!(entries.len(), 1);
    assert_eq!(&entries[0].file, "/compressed/file.bin");
    assert_eq!(entries[0].bytes, 2048);

    Ok(())
}

/// Test dl-driver JSONL format with flexible field mapping
#[tokio::test]
async fn test_dl_driver_jsonl_format() -> Result<()> {
    let mut file = NamedTempFile::with_suffix(".jsonl")?;
    
    // dl-driver uses "operation" field (our format uses "op")
    writeln!(file, r#"{{"operation": "GET", "file": "/test/file1.dat", "bytes": 1024, "t_start_ns": 1000000000}}"#)?;
    writeln!(file, r#"{{"operation": "PUT", "file": "/test/file2.dat", "bytes": 2048, "t_start_ns": 1500000000}}"#)?;
    file.flush()?;

    let reader = OpLogReader::from_file(file.path())?;
    let entries = reader.entries();

    assert_eq!(entries.len(), 2);
    assert_eq!(entries[0].op, OpType::GET);
    assert_eq!(entries[0].bytes, 1024);
    assert_eq!(entries[1].op, OpType::PUT);
    assert_eq!(entries[1].bytes, 2048);

    Ok(())
}

/// Test dl-driver path remapping pattern
#[tokio::test]
async fn test_dl_driver_path_remapping() -> Result<()> {
    let mut file = NamedTempFile::with_suffix(".tsv")?;
    
    writeln!(file, "operation\tfile\tbytes")?;
    writeln!(file, "GET\t/original/data/file1.bin\t1024")?;
    writeln!(file, "GET\t/original/data/file2.bin\t2048")?;
    file.flush()?;

    let reader = OpLogReader::from_file(file.path())?;
    let entries = reader.entries();

    // Simulate dl-driver path remapping: /original -> /remapped
    let mut remapping = HashMap::new();
    remapping.insert("/original".to_string(), "/remapped".to_string());

    for entry in entries {
        let file_path = &entry.file;
        let remapped = if file_path.starts_with("/original") {
            file_path.replace("/original", "/remapped")
        } else {
            file_path.clone()
        };
        
        assert!(remapped.starts_with("/remapped/data/"));
    }

    Ok(())
}

