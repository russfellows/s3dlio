// tests/oplog_caller_buffer_rows.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log rows of the caller-buffer GET (`ObjectStore::get_into`) and the typed HEAD
// (`ObjectStore::head`) through `LoggedObjectStore` (v0.10.0 plan, section 4.5 A6, #264).
//
// Every call that starts is EXACTLY ONE row, whichever way it ends:
//
//   GET   bytes = progress.filled(), also with an error; first_byte = the receive's first
//         byte, or the end time when none was recorded; a dropped future or a panic is
//         `error = "cancelled"` with the count so far.
//   HEAD  success = the size, absence = a successful row with 0 bytes, an error = an error
//         row with 0 bytes, cancelled = `cancelled` with 0 bytes; first_byte is ALWAYS empty.
//
// The inner store is a deterministic fake: no network, no timing. A GET is "cancelled
// mid-flight" by polling it once (the fake records progress, then never finishes) and dropping
// it. A future that was never polled started no native work and writes no row.
//
// After the op-log is finalized the store keeps working and writes nothing more ("logging
// off").
//
// One scenario, one process: the op-log is process-global and a finished log cannot be
// re-initialised, so the scenarios run in order against one log and the rows are checked by
// the `file` column afterwards.

#[allow(dead_code)] // shared test helpers; each test binary uses a subset
mod common;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use bytes::Bytes;
use common::oplog_rows::{read_rows, Row, COL_BYTES, COL_ERROR, COL_FILE, COL_N_OBJECTS, COL_OP};
use futures::stream::{self, Stream};
use futures::FutureExt;
use s3dlio::object_error::{object_error, ObjectErrorKind};
use s3dlio::object_store::{
    HeadMetadata, ObjectMetadata, ObjectStore, ObjectWriter, ReceiveProgress,
};
use s3dlio::receive_progress::SpanMark;
use s3dlio::{finalize_op_logger, global_logger, init_op_logger, LoggedObjectStore};
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::time::Duration;

const COL_START: usize = 9;
const COL_FIRST_BYTE: usize = 10;
const COL_END: usize = 11;

fn uri(name: &str) -> String {
    format!("file:///data/{name}")
}

/// An inner store whose `get_into` and `head` behave as the last path segment of the URI says.
struct ScriptedStore;

impl ScriptedStore {
    /// Record `n` filled positions, as a provider does after copying a chunk.
    fn fill(progress: &ReceiveProgress, n: u64) {
        let mut span = SpanMark::new(progress);
        span.record_copy(0, n).expect("record_copy");
    }
}

#[async_trait]
impl ObjectStore for ScriptedStore {
    async fn get_into(&self, uri: &str, dst: &mut [u8], progress: &ReceiveProgress) -> Result<()> {
        match uri.rsplit('/').next().unwrap_or("") {
            "get-ok" => {
                Self::fill(progress, 4);
                dst.fill(7);
                // The row's end time must be strictly after the first byte's.
                tokio::time::sleep(Duration::from_millis(5)).await;
                Ok(())
            }
            "get-err-partial" => {
                Self::fill(progress, 3);
                Err(anyhow!("receive broke after 3 bytes"))
            }
            "get-err-nodata" => Err(anyhow!("refused before any byte")),
            "get-cancel" => {
                Self::fill(progress, 2);
                std::future::pending::<()>().await;
                Ok(())
            }
            "get-cancel-nodata" => {
                std::future::pending::<()>().await;
                Ok(())
            }
            "get-panic" => {
                Self::fill(progress, 1);
                panic!("fake inner store panics mid-receive");
            }
            "get-never-polled" => Ok(()),
            other => Err(anyhow!("scripted store has no get_into script for {other}")),
        }
    }

    async fn head(&self, uri: &str) -> Result<Option<HeadMetadata>> {
        match uri.rsplit('/').next().unwrap_or("") {
            "head-ok" => Ok(Some(HeadMetadata {
                size: 42,
                etag: Some("\"e1\"".to_string()),
                version_id: Some("v1".to_string()),
            })),
            "head-absent" => Ok(None),
            "head-err" => Err(anyhow!("head refused")),
            "head-cancel" => {
                std::future::pending::<()>().await;
                Ok(None)
            }
            "head-panic" => panic!("fake inner store panics in head"),
            "head-never-polled" => Ok(None),
            other => Err(anyhow!("scripted store has no head script for {other}")),
        }
    }

    // The required methods; none is reached by this test.
    async fn get(&self, _uri: &str) -> Result<Bytes> {
        Err(anyhow!("unused"))
    }
    async fn get_range(&self, _uri: &str, _offset: u64, _length: Option<u64>) -> Result<Bytes> {
        Err(anyhow!("unused"))
    }
    async fn put(&self, _uri: &str, _data: Bytes) -> Result<()> {
        Err(anyhow!("unused"))
    }
    async fn put_multipart(&self, _uri: &str, _data: Bytes, _part: Option<usize>) -> Result<()> {
        Err(anyhow!("unused"))
    }
    async fn list(&self, _prefix: &str, _recursive: bool) -> Result<Vec<String>> {
        Err(anyhow!("unused"))
    }
    fn list_stream<'a>(
        &'a self,
        _prefix: &'a str,
        _recursive: bool,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
        Box::pin(stream::empty())
    }
    async fn stat(&self, _uri: &str) -> Result<ObjectMetadata> {
        Err(anyhow!("unused"))
    }
    async fn delete(&self, _uri: &str) -> Result<()> {
        Err(anyhow!("unused"))
    }
    async fn delete_batch(&self, _uris: &[String]) -> Result<()> {
        Err(anyhow!("unused"))
    }
    async fn delete_prefix(&self, _prefix: &str) -> Result<()> {
        Err(anyhow!("unused"))
    }
    async fn create_container(&self, _name: &str) -> Result<()> {
        Err(anyhow!("unused"))
    }
    async fn delete_container(&self, _name: &str) -> Result<()> {
        Err(anyhow!("unused"))
    }
    async fn get_writer(&self, _uri: &str) -> Result<Box<dyn ObjectWriter>> {
        Err(anyhow!("unused"))
    }
}

/// An inner store that implements neither new method: the wrapper forwards, the default
/// answers `Unsupported`, and that failed call is one error row.
struct DefaultsOnlyStore;

#[async_trait]
impl ObjectStore for DefaultsOnlyStore {
    async fn get(&self, _uri: &str) -> Result<Bytes> {
        Err(anyhow!("unused"))
    }
    async fn get_range(&self, _uri: &str, _offset: u64, _length: Option<u64>) -> Result<Bytes> {
        Err(anyhow!("unused"))
    }
    async fn put(&self, _uri: &str, _data: Bytes) -> Result<()> {
        Err(anyhow!("unused"))
    }
    async fn put_multipart(&self, _uri: &str, _data: Bytes, _part: Option<usize>) -> Result<()> {
        Err(anyhow!("unused"))
    }
    async fn list(&self, _prefix: &str, _recursive: bool) -> Result<Vec<String>> {
        Err(anyhow!("unused"))
    }
    fn list_stream<'a>(
        &'a self,
        _prefix: &'a str,
        _recursive: bool,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
        Box::pin(stream::empty())
    }
    async fn stat(&self, _uri: &str) -> Result<ObjectMetadata> {
        Err(anyhow!("unused"))
    }
    async fn delete(&self, _uri: &str) -> Result<()> {
        Err(anyhow!("unused"))
    }
    async fn delete_batch(&self, _uris: &[String]) -> Result<()> {
        Err(anyhow!("unused"))
    }
    async fn delete_prefix(&self, _prefix: &str) -> Result<()> {
        Err(anyhow!("unused"))
    }
    async fn create_container(&self, _name: &str) -> Result<()> {
        Err(anyhow!("unused"))
    }
    async fn delete_container(&self, _name: &str) -> Result<()> {
        Err(anyhow!("unused"))
    }
    async fn get_writer(&self, _uri: &str) -> Result<Box<dyn ObjectWriter>> {
        Err(anyhow!("unused"))
    }
}

/// The rows whose `file` column is the path of `name`, in file order.
fn rows_of<'r>(rows: &'r [Row], name: &str) -> Vec<&'r Row> {
    let file = format!("/data/{name}");
    rows.iter().filter(|r| r[COL_FILE] == file).collect()
}

/// The one row of `name`, asserting it is the only one.
fn only_row<'r>(rows: &'r [Row], name: &str) -> &'r Row {
    let found = rows_of(rows, name);
    assert_eq!(found.len(), 1, "{name}: exactly one row, got {found:#?}");
    found[0]
}

#[tokio::test]
async fn get_into_and_head_write_exactly_one_row_per_call() {
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let dir = tempfile::tempdir().unwrap();
    let oplog = dir.path().join("ops.tsv.zst");
    init_op_logger(oplog.to_str().unwrap()).unwrap();
    let logger = global_logger().unwrap();
    let scripted = LoggedObjectStore::new(std::sync::Arc::new(ScriptedStore), logger.clone());
    let defaults = LoggedObjectStore::new(std::sync::Arc::new(DefaultsOnlyStore), logger);

    // ---- GET -------------------------------------------------------------------------
    let progress = ReceiveProgress::new();
    let mut dst = [0u8; 4];
    scripted
        .get_into(&uri("get-ok"), &mut dst, &progress)
        .await
        .unwrap();
    assert_eq!(
        dst, [7; 4],
        "the inner store's writes reach the caller's buffer"
    );

    let progress = ReceiveProgress::new();
    let err = scripted
        .get_into(&uri("get-err-partial"), &mut [0u8; 8], &progress)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("receive broke"), "{err:#}");

    let progress = ReceiveProgress::new();
    scripted
        .get_into(&uri("get-err-nodata"), &mut [0u8; 8], &progress)
        .await
        .unwrap_err();

    // Cancelled mid-flight: polled once (the inner store records 2 positions, then waits
    // forever), then dropped.
    let cancel_progress = ReceiveProgress::new();
    {
        let mut buf = [0u8; 8];
        let target = uri("get-cancel");
        let fut = scripted.get_into(&target, &mut buf, &cancel_progress);
        assert!(
            fut.now_or_never().is_none(),
            "the scripted receive must still be pending"
        );
    }
    assert_eq!(cancel_progress.filled(), 2, "progress survives the drop");

    // Cancelled before any byte: polled once (the inner store records nothing, then waits
    // forever), then dropped.
    let nodata_progress = ReceiveProgress::new();
    {
        let mut buf = [0u8; 8];
        let target = uri("get-cancel-nodata");
        let fut = scripted.get_into(&target, &mut buf, &nodata_progress);
        assert!(
            fut.now_or_never().is_none(),
            "the scripted receive must still be pending"
        );
    }
    assert_eq!(nodata_progress.filled(), 0);

    // A panic inside the inner future: one cancelled row, not zero and not two.
    let panic_progress = ReceiveProgress::new();
    {
        let mut buf = [0u8; 8];
        let target = uri("get-panic");
        let fut = scripted.get_into(&target, &mut buf, &panic_progress);
        let outcome = AssertUnwindSafe(fut).catch_unwind().await;
        assert!(outcome.is_err(), "the inner panic must propagate");
    }
    assert_eq!(panic_progress.filled(), 1);

    // Never polled: no native work started, so no row.
    {
        let mut buf = [0u8; 8];
        let progress = ReceiveProgress::new();
        drop(scripted.get_into(&uri("get-never-polled"), &mut buf, &progress));
    }

    // The default `Unsupported`, forwarded through the wrapper: one error row.
    let progress = ReceiveProgress::new();
    let unsupported = defaults
        .get_into(&uri("get-default"), &mut [0u8; 2], &progress)
        .await
        .unwrap_err();
    assert_eq!(
        object_error(&unsupported).map(|e| e.kind),
        Some(ObjectErrorKind::Unsupported)
    );

    // ---- HEAD ------------------------------------------------------------------------
    let found = scripted.head(&uri("head-ok")).await.unwrap();
    assert_eq!(found.map(|m| m.size), Some(42));
    assert_eq!(scripted.head(&uri("head-absent")).await.unwrap(), None);
    scripted.head(&uri("head-err")).await.unwrap_err();
    assert!(scripted.head(&uri("head-cancel")).now_or_never().is_none());
    let target = uri("head-panic");
    let fut = scripted.head(&target);
    assert!(AssertUnwindSafe(fut).catch_unwind().await.is_err());
    drop(scripted.head(&uri("head-never-polled")));
    let unsupported = defaults.head(&uri("head-default")).await.unwrap_err();
    assert_eq!(
        object_error(&unsupported).map(|e| e.kind),
        Some(ObjectErrorKind::Unsupported)
    );

    finalize_op_logger();
    let rows = read_rows(&oplog);

    // ---- check the rows --------------------------------------------------------------
    let row = only_row(&rows, "get-ok");
    assert_eq!(row[COL_OP], "GET");
    assert_eq!(row[COL_BYTES], "4");
    assert_eq!(row[COL_N_OBJECTS], "1");
    assert_eq!(row[COL_ERROR], "", "success has no error");
    // The first byte comes from the receive's progress, not from the end time.
    assert!(!row[COL_FIRST_BYTE].is_empty(), "{row:?}");
    assert!(
        row[COL_START] <= row[COL_FIRST_BYTE] && row[COL_FIRST_BYTE] < row[COL_END],
        "first byte is the receive's own and precedes the end: {row:?}"
    );

    let row = only_row(&rows, "get-err-partial");
    assert_eq!(row[COL_OP], "GET");
    assert_eq!(
        row[COL_BYTES], "3",
        "an error keeps what was filled: {row:?}"
    );
    assert!(row[COL_ERROR].contains("receive broke"), "{row:?}");
    assert!(!row[COL_FIRST_BYTE].is_empty(), "{row:?}");

    let row = only_row(&rows, "get-err-nodata");
    assert_eq!(row[COL_BYTES], "0");
    assert!(
        row[COL_ERROR].contains("refused before any byte"),
        "{row:?}"
    );
    assert_eq!(
        row[COL_FIRST_BYTE], row[COL_END],
        "no first byte recorded: first_byte is the end time, as for every other GET row"
    );

    let row = only_row(&rows, "get-cancel");
    assert_eq!(row[COL_OP], "GET");
    assert_eq!(row[COL_ERROR], "cancelled");
    assert_eq!(
        row[COL_BYTES], "2",
        "the count at the time of the drop: {row:?}"
    );
    assert!(!row[COL_FIRST_BYTE].is_empty(), "bytes arrived: {row:?}");

    let row = only_row(&rows, "get-cancel-nodata");
    assert_eq!(row[COL_OP], "GET");
    assert_eq!(row[COL_ERROR], "cancelled");
    assert_eq!(row[COL_BYTES], "0", "{row:?}");
    assert_eq!(
        row[COL_FIRST_BYTE], row[COL_END],
        "no first byte recorded: first_byte is the end time, as for every other GET row"
    );

    let row = only_row(&rows, "get-panic");
    assert_eq!(row[COL_OP], "GET");
    assert_eq!(
        row[COL_ERROR], "cancelled",
        "a panic is a drop without completion"
    );
    assert_eq!(row[COL_BYTES], "1");

    assert!(
        rows_of(&rows, "get-never-polled").is_empty(),
        "a future that never ran started no native work and writes no row"
    );

    let row = only_row(&rows, "get-default");
    assert_eq!(row[COL_OP], "GET");
    assert_eq!(row[COL_BYTES], "0");
    assert!(row[COL_ERROR].contains("unsupported"), "{row:?}");

    for (name, bytes, error) in [
        ("head-ok", "42", ""),
        ("head-absent", "0", ""),
        ("head-err", "0", "head refused"),
        ("head-cancel", "0", "cancelled"),
        ("head-panic", "0", "cancelled"),
    ] {
        let row = only_row(&rows, name);
        assert_eq!(row[COL_OP], "HEAD", "{name}");
        assert_eq!(row[COL_BYTES], bytes, "{name}: {row:?}");
        assert_eq!(row[COL_N_OBJECTS], "1", "{name}");
        if error.is_empty() {
            assert_eq!(row[COL_ERROR], "", "{name}: {row:?}");
        } else {
            assert!(row[COL_ERROR].contains(error), "{name}: {row:?}");
        }
        assert_eq!(
            row[COL_FIRST_BYTE], "",
            "{name}: a HEAD never has a first byte: {row:?}"
        );
    }
    assert!(rows_of(&rows, "head-never-polled").is_empty());
    let row = only_row(&rows, "head-default");
    assert_eq!(row[COL_OP], "HEAD");
    assert_eq!(row[COL_FIRST_BYTE], "");
    assert!(row[COL_ERROR].contains("unsupported"), "{row:?}");

    // Nothing else was written: 7 GET rows and 6 HEAD rows.
    assert_eq!(rows.len(), 13, "{rows:#?}");

    // ---- logging off -----------------------------------------------------------------
    // The op-log is finalized: both methods keep returning the correct results and write no
    // row.
    let finished = std::fs::read(&oplog).unwrap();
    let progress = ReceiveProgress::new();
    let mut dst = [0u8; 4];
    scripted
        .get_into(&uri("get-ok"), &mut dst, &progress)
        .await
        .unwrap();
    assert_eq!(dst, [7; 4]);
    assert_eq!(progress.filled(), 4);
    scripted
        .get_into(
            &uri("get-err-partial"),
            &mut [0u8; 8],
            &ReceiveProgress::new(),
        )
        .await
        .unwrap_err();
    assert_eq!(
        scripted
            .head(&uri("head-ok"))
            .await
            .unwrap()
            .map(|m| m.size),
        Some(42)
    );
    assert_eq!(scripted.head(&uri("head-absent")).await.unwrap(), None);
    assert_eq!(
        std::fs::read(&oplog).unwrap(),
        finished,
        "the finished op-log changed after finalize"
    );
}
