// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Operation replay engine with timing-faithful execution
//!
//! Provides timeline-based replay of operation logs with microsecond precision,
//! backend retargeting, and pluggable execution via the OpExecutor trait.

use anyhow::{Context, Result};
use async_trait::async_trait;
use futures::stream::FuturesUnordered;
use futures::StreamExt;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tracing::{debug, info, warn};

use crate::reader::OpLogReader;
use crate::types::{OpLogEntry, OpType};
use crate::uri::translate_uri;

/// Replay configuration
#[derive(Debug, Clone)]
pub struct ReplayConfig {
    /// Path to op-log file
    pub op_log_path: PathBuf,
    
    /// Optional target URI for backend retargeting (1:1 remapping)
    pub target_uri: Option<String>,
    
    /// Speed multiplier (1.0 = original speed, 2.0 = 2x faster, 0.5 = half speed)
    pub speed: f64,
    
    /// Continue execution on errors instead of stopping
    pub continue_on_error: bool,
    
    /// Optional operation filter (only replay specific operation types)
    pub filter_ops: Option<Vec<OpType>>,
}

impl Default for ReplayConfig {
    fn default() -> Self {
        Self {
            op_log_path: PathBuf::new(),
            target_uri: None,
            speed: 1.0,
            continue_on_error: false,
            filter_ops: None,
        }
    }
}

/// Executor trait for pluggable operation execution
///
/// Implementations of this trait define how operations are actually executed.
/// The default implementation uses s3dlio's ObjectStore for multi-backend support.
#[async_trait]
pub trait OpExecutor: Send + Sync {
    /// Execute a GET operation
    async fn get(&self, uri: &str) -> Result<()>;
    
    /// Execute a PUT operation
    async fn put(&self, uri: &str, bytes: usize) -> Result<()>;
    
    /// Execute a DELETE operation
    async fn delete(&self, uri: &str) -> Result<()>;
    
    /// Execute a LIST operation
    async fn list(&self, uri: &str) -> Result<()>;
    
    /// Execute a STAT/HEAD operation
    async fn stat(&self, uri: &str) -> Result<()>;
}

/// Default executor using s3dlio ObjectStore
///
/// Supports all s3dlio backends: file://, s3://, az://, direct://
pub struct S3dlioExecutor;

#[async_trait]
impl OpExecutor for S3dlioExecutor {
    async fn get(&self, uri: &str) -> Result<()> {
        let store = s3dlio::api::store_for_uri(uri)
            .with_context(|| format!("Failed to create store for URI: {}", uri))?;
        let _ = store.get(uri).await
            .with_context(|| format!("Failed to GET {}", uri))?;
        Ok(())
    }

    async fn put(&self, uri: &str, bytes: usize) -> Result<()> {
        // Generate random data using s3dlio's data generation
        let data = s3dlio::data_gen::generate_controlled_data(bytes, 1, 1);
        
        let store = s3dlio::api::store_for_uri(uri)
            .with_context(|| format!("Failed to create store for URI: {}", uri))?;
        // s3dlio v0.9.36: put() takes Bytes directly for zero-copy
        store.put(uri, bytes::Bytes::from(data)).await
            .with_context(|| format!("Failed to PUT {} ({} bytes)", uri, bytes))?;
        Ok(())
    }

    async fn delete(&self, uri: &str) -> Result<()> {
        let store = s3dlio::api::store_for_uri(uri)
            .with_context(|| format!("Failed to create store for URI: {}", uri))?;
        store.delete(uri).await
            .with_context(|| format!("Failed to DELETE {}", uri))?;
        Ok(())
    }

    async fn list(&self, uri: &str) -> Result<()> {
        let store = s3dlio::api::store_for_uri(uri)
            .with_context(|| format!("Failed to create store for URI: {}", uri))?;
        let _ = store.list(uri, true).await
            .with_context(|| format!("Failed to LIST {}", uri))?;
        Ok(())
    }

    async fn stat(&self, uri: &str) -> Result<()> {
        let store = s3dlio::api::store_for_uri(uri)
            .with_context(|| format!("Failed to create store for URI: {}", uri))?;
        let _ = store.stat(uri).await
            .with_context(|| format!("Failed to STAT {}", uri))?;
        Ok(())
    }
}

/// Main replay orchestrator with absolute timeline scheduling
///
/// Replays operations from an op-log file with microsecond-precision timing.
/// Uses absolute timestamps to prevent drift accumulation.
///
/// # Arguments
/// * `config` - Replay configuration
/// * `executor` - Implementation of OpExecutor trait for operation execution
///
/// # Examples
/// ```no_run
/// use s3dlio_oplog::replayer::{ReplayConfig, replay_workload, S3dlioExecutor};
/// use std::path::PathBuf;
/// use std::sync::Arc;
///
/// # tokio_test::block_on(async {
/// let config = ReplayConfig {
///     op_log_path: PathBuf::from("operations.tsv.zst"),
///     target_uri: Some("s3://my-bucket/test/".to_string()),
///     speed: 1.0,
///     continue_on_error: true,
///     filter_ops: None,
/// };
///
/// let executor = Arc::new(S3dlioExecutor);
/// replay_workload(config, executor).await.unwrap();
/// # });
/// ```
pub async fn replay_workload<E: OpExecutor + 'static>(config: ReplayConfig, executor: std::sync::Arc<E>) -> Result<()> {
    info!("Starting replay with config: {:?}", config);

    // Load operations from file
    let reader = OpLogReader::from_file(&config.op_log_path)
        .with_context(|| format!("Failed to load op-log from {:?}", config.op_log_path))?;

    let mut operations: Vec<OpLogEntry> = reader.entries().to_vec();

    // Apply operation filter if specified
    if let Some(ref filter_ops) = config.filter_ops {
        let original_count = operations.len();
        operations.retain(|op| filter_ops.contains(&op.op));
        info!("Filtered {} -> {} operations (filter: {:?})", 
              original_count, operations.len(), filter_ops);
    }

    if operations.is_empty() {
        anyhow::bail!("No operations to replay after filtering");
    }

    // Sort by start time for timeline replay
    operations.sort_by_key(|op| op.start);

    let first_time = operations[0].start;
    let last_time = operations[operations.len() - 1].start;
    let duration = last_time.signed_duration_since(first_time);

    info!(
        "Replaying {} operations over {} seconds (speed: {:.1}x)",
        operations.len(),
        duration.num_seconds(),
        config.speed
    );

    let replay_epoch = Instant::now();

    // Spawn all operations with absolute timing
    let mut tasks = FuturesUnordered::new();

    for op in operations {
        let target = config.target_uri.clone();
        let speed = config.speed;
        let executor = executor.clone();

        // Calculate absolute time offset from first operation
        let elapsed = op.start.signed_duration_since(first_time);
        let delay = elapsed.to_std()? / speed as u32; // Adjusted for speed multiplier

        // Clone necessary data for async task
        let op_clone = op.clone();
        let task = tokio::spawn(async move {
            schedule_and_execute(op_clone, replay_epoch, delay, target.as_deref(), &*executor).await
        });

        tasks.push(task);
    }

    // Collect results
    let mut success = 0;
    let mut failed = 0;

    while let Some(result) = tasks.next().await {
        match result {
            Ok(Ok(())) => success += 1,
            Ok(Err(e)) => {
                failed += 1;
                if config.continue_on_error {
                    warn!("Operation failed (continuing): {}", e);
                } else {
                    return Err(e);
                }
            }
            Err(e) => {
                failed += 1;
                if config.continue_on_error {
                    warn!("Task panicked (continuing): {}", e);
                } else {
                    return Err(e.into());
                }
            }
        }
    }

    info!("Replay complete: {} successful, {} failed", success, failed);
    Ok(())
}

/// Schedule operation at absolute time and execute with microsecond precision
async fn schedule_and_execute<E: OpExecutor>(
    op: OpLogEntry,
    replay_epoch: Instant,
    delay: Duration,
    target: Option<&str>,
    executor: &E,
) -> Result<()> {
    let target_time = replay_epoch + delay;
    let now = Instant::now();

    // Sleep until target time (microsecond precision with std::thread::sleep)
    if target_time > now {
        let sleep_dur = target_time - now;
        tokio::task::spawn_blocking(move || {
            std::thread::sleep(sleep_dur);
        })
        .await?;
    }

    // Translate URI if target provided
    let uri = if let Some(tgt) = target {
        translate_uri(&op.file, &op.endpoint, tgt)?
    } else {
        format!("{}{}", op.endpoint, op.file)
    };

    debug!("Executing {:?} on {}", op.op, uri);

    // Execute operation via executor trait
    match op.op {
        OpType::GET => executor.get(&uri).await?,
        OpType::PUT => executor.put(&uri, op.bytes as usize).await?,
        OpType::DELETE => executor.delete(&uri).await?,
        OpType::LIST => executor.list(&uri).await?,
        OpType::STAT => executor.stat(&uri).await?,
    }

    Ok(())
}

/// Convenience function for replay with default s3dlio executor
pub async fn replay_with_s3dlio(config: ReplayConfig) -> Result<()> {
    let executor = std::sync::Arc::new(S3dlioExecutor);
    replay_workload(config, executor).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Mock executor for testing
    struct MockExecutor {
        ops: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl OpExecutor for MockExecutor {
        async fn get(&self, uri: &str) -> Result<()> {
            self.ops.lock().unwrap().push(format!("GET {}", uri));
            Ok(())
        }

        async fn put(&self, uri: &str, bytes: usize) -> Result<()> {
            self.ops.lock().unwrap().push(format!("PUT {} ({})", uri, bytes));
            Ok(())
        }

        async fn delete(&self, uri: &str) -> Result<()> {
            self.ops.lock().unwrap().push(format!("DELETE {}", uri));
            Ok(())
        }

        async fn list(&self, uri: &str) -> Result<()> {
            self.ops.lock().unwrap().push(format!("LIST {}", uri));
            Ok(())
        }

        async fn stat(&self, uri: &str) -> Result<()> {
            self.ops.lock().unwrap().push(format!("STAT {}", uri));
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_replay_with_mock_executor() {
        let mut file = NamedTempFile::with_suffix(".tsv").unwrap();
        writeln!(file, "op\tfile\tendpoint\tbytes\tstart").unwrap();
        writeln!(file, "GET\tdata/test.dat\tfile:///tmp/\t1024\t2025-01-01T00:00:00Z").unwrap();
        writeln!(file, "PUT\tdata/test2.dat\tfile:///tmp/\t2048\t2025-01-01T00:00:01Z").unwrap();
        file.flush().unwrap();

        let ops = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let executor = std::sync::Arc::new(MockExecutor { ops: ops.clone() });

        let config = ReplayConfig {
            op_log_path: file.path().to_path_buf(),
            target_uri: Some("s3://test-bucket/".to_string()),
            speed: 1000.0, // Very fast for testing
            continue_on_error: false,
            filter_ops: None,
        };

        replay_workload(config, executor).await.unwrap();

        let executed_ops = ops.lock().unwrap();
        assert_eq!(executed_ops.len(), 2);
        assert!(executed_ops[0].starts_with("GET s3://test-bucket/data/test.dat") || 
                executed_ops[1].starts_with("GET s3://test-bucket/data/test.dat"));
        assert!(executed_ops[0].starts_with("PUT s3://test-bucket/data/test2.dat") || 
                executed_ops[1].starts_with("PUT s3://test-bucket/data/test2.dat"));
    }
}
