// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! s3dlio-oplog: Shared operation log parsing and replay functionality
//!
//! This library provides unified operation log parsing and replay capabilities
//! for the s3dlio ecosystem, enabling tools like s3-bench (io-bench) and dl-driver
//! to share the same core functionality.
//!
//! # Features
//!
//! - **Multi-format support**: Parse JSONL and TSV operation logs
//! - **Auto-compression detection**: Automatic zstd decompression for `.zst` files
//! - **Format tolerance**: Header-driven TSV parsing handles column variations
//! - **Timing-faithful replay**: Microsecond-precision timeline scheduling
//! - **Backend retargeting**: Simple 1:1 URI translation for cross-backend replay
//! - **Pluggable execution**: OpExecutor trait for custom operation execution
//! - **Multi-backend support**: Works with file://, s3://, az://, direct:// via s3dlio
//!
//! # Known limits of replay
//!
//! Replay is exact only for the five operations [`OpType`] models (GET, PUT, DELETE, LIST,
//! STAT/HEAD). Rows for `GET_RANGE`, `DELETE_BATCH`, `DELETE_PREFIX`, `CREATE_CONTAINER` and
//! `DELETE_CONTAINER` do not record the parameters needed to replay them exactly (s3dlio issue
//! #176), so recognising their names would not make them replayable. The first such row, or any
//! malformed row, makes [`OpLogReader::from_file`] (and therefore [`replay_workload`]) fail at
//! load time, before any operation runs; the error names the file line, the row index and the
//! op. [`ReplayConfig::continue_on_error`] applies only to operations that fail while executing,
//! not to load errors. [`OpLogStreamReader`] reports the error for that row and yields the rows
//! after it. Traces that mix in such rows cannot be replayed exactly yet. Separately, a
//! `MultiEndpointStore` writes rows only for `get`/`put` on bucket-form endpoints (#175), so its
//! op-logs under-count requests. See "Known gaps" in `docs/supplemental/OPERATION_LOGGING.md`.
//!
//! # Usage
//!
//! ## Basic Replay
//!
//! ```no_run
//! use s3dlio_oplog::{ReplayConfig, replay_with_s3dlio};
//! use std::path::PathBuf;
//!
//! # tokio_test::block_on(async {
//! let config = ReplayConfig {
//!     op_log_path: PathBuf::from("operations.tsv.zst"),
//!     target_uri: Some("s3://my-bucket/test/".to_string()),
//!     speed: 1.0,
//!     continue_on_error: true,
//!     filter_ops: None,
//! };
//!
//! replay_with_s3dlio(config).await.unwrap();
//! # });
//! ```
//!
//! ## Custom Executor
//!
//! ```no_run
//! use s3dlio_oplog::{ReplayConfig, OpExecutor, replay_workload};
//! use async_trait::async_trait;
//! use anyhow::Result;
//! use std::path::PathBuf;
//!
//! struct CustomExecutor;
//!
//! #[async_trait]
//! impl OpExecutor for CustomExecutor {
//!     async fn get(&self, uri: &str) -> Result<()> {
//!         println!("Custom GET: {}", uri);
//!         Ok(())
//!     }
//!
//!     async fn put(&self, uri: &str, bytes: usize) -> Result<()> {
//!         println!("Custom PUT: {} ({} bytes)", uri, bytes);
//!         Ok(())
//!     }
//!
//!     async fn delete(&self, uri: &str) -> Result<()> {
//!         println!("Custom DELETE: {}", uri);
//!         Ok(())
//!     }
//!
//!     async fn list(&self, uri: &str) -> Result<()> {
//!         println!("Custom LIST: {}", uri);
//!         Ok(())
//!     }
//!
//!     async fn stat(&self, uri: &str) -> Result<()> {
//!         println!("Custom STAT: {}", uri);
//!         Ok(())
//!     }
//! }
//!
//! # tokio_test::block_on(async {
//! let config = ReplayConfig {
//!     op_log_path: PathBuf::from("ops.tsv.zst"),
//!     target_uri: None,
//!     speed: 1.0,
//!     continue_on_error: false,
//!     filter_ops: None,
//! };
//!
//! let executor = std::sync::Arc::new(CustomExecutor);
//! replay_workload(config, executor).await.unwrap();
//! # });
//! ```
//!
//! ## Parsing and Analysis
//!
//! ```no_run
//! use s3dlio_oplog::{OpLogReader, OpType};
//!
//! let reader = OpLogReader::from_file("operations.tsv.zst").unwrap();
//!
//! println!("Total operations: {}", reader.len());
//!
//! // Filter specific operation types
//! let gets = reader.filter_operations(&[OpType::GET]);
//! println!("GET operations: {}", gets.len());
//!
//! // Access individual entries
//! for entry in reader.entries() {
//!     println!("{:?} {} ({})", entry.op, entry.file, entry.bytes);
//! }
//! ```

pub mod reader;
pub mod replayer;
pub mod types;
pub mod uri;

// Re-export main types and functions for convenience
pub use reader::{OpLogFormat, OpLogReader, OpLogStreamReader};
pub use replayer::{replay_with_s3dlio, replay_workload, OpExecutor, ReplayConfig, S3dlioExecutor};
pub use types::{OpLogEntry, OpType};
pub use uri::{row_uri, translate_uri, RowUriError};
