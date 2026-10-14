// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Shared fixtures for the reader/replayer regression tests: writer-format rows and a
//! recording executor. Every row has the shape of a row written by `LoggedObjectStore`
//! (`src/object_store_logger.rs`) or by the legacy `S3Ops` layer (`src/s3_ops.rs`).

use anyhow::Result;
use async_trait::async_trait;
use s3dlio_oplog::{replay_workload, OpExecutor, OpType, ReplayConfig};
use std::io::Write;
use std::sync::{Arc, Mutex};
use tempfile::NamedTempFile;

pub const HEADER: &str = "idx\tthread\top\tclient_id\tn_objects\tbytes\tendpoint\tfile\terror\tstart\tfirst_byte\tend\tduration_ns";

/// Records the URI of every operation the replayer hands to the executor.
#[derive(Default)]
struct Recorder {
    seen: Mutex<Vec<String>>,
}

impl Recorder {
    fn push(&self, op: &str, uri: &str) {
        self.seen
            .lock()
            .expect("recorder lock")
            .push(format!("{op} {uri}"));
    }
    fn all(&self) -> Vec<String> {
        self.seen.lock().expect("recorder lock").clone()
    }
}

#[async_trait]
impl OpExecutor for Recorder {
    async fn get(&self, uri: &str) -> Result<()> {
        self.push("GET", uri);
        Ok(())
    }
    async fn put(&self, uri: &str, _bytes: usize) -> Result<()> {
        self.push("PUT", uri);
        Ok(())
    }
    async fn delete(&self, uri: &str) -> Result<()> {
        self.push("DELETE", uri);
        Ok(())
    }
    async fn list(&self, uri: &str) -> Result<()> {
        self.push("LIST", uri);
        Ok(())
    }
    async fn stat(&self, uri: &str) -> Result<()> {
        self.push("STAT", uri);
        Ok(())
    }
}

/// One writer-format row. `idx` doubles as the start-time second so rows replay in order.
pub fn row(idx: u32, op: &str, endpoint: &str, file: &str) -> String {
    format!(
        "{idx}\t1\t{op}\t\t1\t10\t{endpoint}\t{file}\t\t2026-01-01T00:00:{idx:02}Z\t\t2026-01-01T00:00:{idx:02}Z\t1000"
    )
}

pub fn tsv(rows: &[String]) -> NamedTempFile {
    let mut f = NamedTempFile::with_suffix(".tsv").expect("temp file");
    writeln!(f, "{HEADER}").expect("write header");
    for r in rows {
        writeln!(f, "{r}").expect("write row");
    }
    f.flush().expect("flush");
    f
}

pub async fn replay(
    file: &NamedTempFile,
    target: Option<&str>,
    continue_on_error: bool,
    filter_ops: Option<Vec<OpType>>,
) -> (Result<()>, Vec<String>) {
    let rec = Arc::new(Recorder::default());
    let cfg = ReplayConfig {
        op_log_path: file.path().to_path_buf(),
        target_uri: target.map(str::to_string),
        speed: 1000.0,
        continue_on_error,
        filter_ops,
    };
    let res = replay_workload(cfg, rec.clone()).await;
    (res, rec.all())
}
