// tests/oplog_logged_store_forwards_every_method.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// s3dlio #177: LoggedObjectStore must be transparent. Before the fix it
// overrode only part of the `ObjectStore` trait, so mkdir/rmdir/update_*/copy/
// rename (and get_optimized, get_range_optimized, pre_stat_*) fell through to
// the trait defaults: mkdir and rmdir always failed, rename became
// GET+PUT+DELETE, and the inner store's own implementations were never called.
//
// A counting inner store overrides EVERY trait method and notes which one was
// reached. Each call on the wrapper must reach exactly the same method on the
// inner store, and the op-log must hold one row only for the operations that
// already have an op name (the rest are forwarded silently).
//
// A guard test reads the trait's source and fails when a method is added to
// `ObjectStore` without being added here, so a new defaulted method cannot be
// left unforwarded.
//
// Own process (the op-log is process-global).

#[allow(dead_code)] // shared test helpers; each test binary uses a subset
mod common;

use anyhow::Result;
use async_trait::async_trait;
use bytes::Bytes;
use common::oplog_rows::{ops, read_rows};
use futures::stream::{self, Stream};
use s3dlio::object_store::{CompressionConfig, ObjectProperties, ObjectWriter, WriterOptions};
use s3dlio::{
    finalize_op_logger, global_logger, init_op_logger, LoggedObjectStore, ObjectMetadata,
    ObjectStore,
};
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

/// Every method of the `ObjectStore` trait, in the order the test calls them,
/// with the op names of the rows its call must write (empty = forwarded
/// silently) and the inner methods that must be reached, in order.
///
/// `CountingStore` declares native `copy` and `rename` (`has_native_*` = true),
/// as `file://` does for `rename`: such a call is forwarded and not recorded. A
/// backend on the trait default (S3, Azure, GCS) is covered by
/// `oplog_rows_copy_rename_prestat.rs`, where copy and rename are recorded.
/// `pre_stat_objects` and `pre_stat_and_cache` are NOT forwarded: they stat every
/// URI through the wrapper (one recorded HEAD each); `pre_stat_and_cache` then
/// hands the sizes to the inner store's cache (`seed_size_cache`).
type Expectation = (
    &'static str,
    &'static [&'static str],
    &'static [&'static str],
);
const METHODS: &[Expectation] = &[
    ("get", &["GET"], &["get"]),
    ("get_range", &["GET_RANGE"], &["get_range"]),
    ("put", &["PUT"], &["put"]),
    ("put_multipart", &["PUT"], &["put_multipart"]),
    ("list", &["LIST"], &["list"]),
    ("list_stream", &["LIST"], &["list_stream"]),
    ("stat", &["HEAD"], &["stat"]),
    ("stat_if_exists", &["HEAD"], &["stat_if_exists"]),
    (
        "list_objects_with_prefix",
        &["LIST"],
        &["list_objects_with_prefix"],
    ),
    ("delete", &["DELETE"], &["delete"]),
    ("delete_batch", &["DELETE_BATCH"], &["delete_batch"]),
    ("delete_prefix", &["DELETE_PREFIX"], &["delete_prefix"]),
    (
        "create_container",
        &["CREATE_CONTAINER"],
        &["create_container"],
    ),
    (
        "delete_container",
        &["DELETE_CONTAINER"],
        &["delete_container"],
    ),
    // exists is answered through the inner store's stat (see #189).
    ("exists", &["HEAD"], &["stat"]),
    ("get_with_validation", &["GET"], &["get_with_validation"]),
    (
        "get_range_with_validation",
        &["GET_RANGE"],
        &["get_range_with_validation"],
    ),
    (
        "load_checkpoint_with_validation",
        &["GET"],
        &["load_checkpoint_with_validation"],
    ),
    ("get_optimized", &["GET"], &["get_optimized"]),
    (
        "get_range_optimized",
        &["GET_RANGE"],
        &["get_range_optimized"],
    ),
    // A NATIVE copy / rename: no op name describes it, forwarded, not recorded.
    ("copy", &[], &["copy"]),
    ("rename", &[], &["rename"]),
    // Answered by the inner store.
    ("has_native_copy", &[], &[]),
    ("has_native_rename", &[], &[]),
    // pre_stat_*: real HEAD requests, recorded through the wrapper.
    ("pre_stat_objects", &["HEAD", "HEAD"], &["stat", "stat"]),
    (
        "pre_stat_and_cache",
        &["HEAD", "HEAD"],
        &["stat", "stat", "seed_size_cache"],
    ),
    ("seed_size_cache", &[], &["seed_size_cache"]),
    // No op name exists for these: forwarded, not recorded.
    ("mkdir", &[], &["mkdir"]),
    ("rmdir", &[], &["rmdir"]),
    ("update_metadata", &[], &["update_metadata"]),
    ("update_properties", &[], &["update_properties"]),
    ("get_writer", &[], &["get_writer"]),
    (
        "get_writer_with_compression",
        &[],
        &["get_writer_with_compression"],
    ),
    ("create_writer", &[], &["create_writer"]),
];

type CallLog = Arc<Mutex<Vec<&'static str>>>;

struct NullWriter;

#[async_trait]
impl ObjectWriter for NullWriter {
    async fn write_chunk(&mut self, _chunk: &[u8]) -> Result<()> {
        Ok(())
    }
    async fn finalize(self: Box<Self>) -> Result<()> {
        Ok(())
    }
    fn bytes_written(&self) -> u64 {
        0
    }
    fn checksum(&self) -> Option<String> {
        None
    }
}

/// An inner store that overrides every trait method and notes which one ran.
struct CountingStore {
    calls: CallLog,
}

impl CountingStore {
    fn note(&self, method: &'static str) {
        self.calls.lock().unwrap().push(method);
    }
}

fn metadata() -> ObjectMetadata {
    ObjectMetadata {
        size: 3,
        last_modified: None,
        e_tag: None,
        content_type: None,
        content_language: None,
        content_encoding: None,
        cache_control: None,
        content_disposition: None,
        expires: None,
        storage_class: None,
        server_side_encryption: None,
        ssekms_key_id: None,
        sse_customer_algorithm: None,
        version_id: None,
        replication_status: None,
        metadata: HashMap::new(),
    }
}

#[async_trait]
impl ObjectStore for CountingStore {
    async fn get(&self, _uri: &str) -> Result<Bytes> {
        self.note("get");
        Ok(Bytes::from_static(b"abc"))
    }
    async fn get_range(&self, _uri: &str, _offset: u64, _length: Option<u64>) -> Result<Bytes> {
        self.note("get_range");
        Ok(Bytes::from_static(b"ab"))
    }
    async fn put(&self, _uri: &str, _data: Bytes) -> Result<()> {
        self.note("put");
        Ok(())
    }
    async fn put_multipart(&self, _uri: &str, _data: Bytes, _part: Option<usize>) -> Result<()> {
        self.note("put_multipart");
        Ok(())
    }
    async fn list(&self, _prefix: &str, _recursive: bool) -> Result<Vec<String>> {
        self.note("list");
        Ok(vec![
            "file:///data/a".to_string(),
            "file:///data/b".to_string(),
        ])
    }
    fn list_stream<'a>(
        &'a self,
        _prefix: &'a str,
        _recursive: bool,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
        self.note("list_stream");
        Box::pin(stream::iter(vec![Ok("file:///data/a".to_string())]))
    }
    async fn stat(&self, _uri: &str) -> Result<ObjectMetadata> {
        self.note("stat");
        Ok(metadata())
    }
    async fn stat_if_exists(&self, _uri: &str) -> Result<Option<ObjectMetadata>> {
        self.note("stat_if_exists");
        Ok(Some(metadata()))
    }
    fn list_objects_with_prefix<'a>(
        &'a self,
        _prefix: &'a str,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
        self.note("list_objects_with_prefix");
        Box::pin(stream::iter(vec![Ok("file:///data/a".to_string())]))
    }
    async fn delete(&self, _uri: &str) -> Result<()> {
        self.note("delete");
        Ok(())
    }
    async fn delete_batch(&self, _uris: &[String]) -> Result<()> {
        self.note("delete_batch");
        Ok(())
    }
    async fn delete_prefix(&self, _prefix: &str) -> Result<()> {
        self.note("delete_prefix");
        Ok(())
    }
    async fn create_container(&self, _name: &str) -> Result<()> {
        self.note("create_container");
        Ok(())
    }
    async fn delete_container(&self, _name: &str) -> Result<()> {
        self.note("delete_container");
        Ok(())
    }
    async fn exists(&self, _uri: &str) -> Result<bool> {
        self.note("exists");
        Ok(true)
    }
    async fn get_with_validation(&self, _uri: &str, _sum: Option<&str>) -> Result<Bytes> {
        self.note("get_with_validation");
        Ok(Bytes::from_static(b"abc"))
    }
    async fn get_range_with_validation(
        &self,
        _uri: &str,
        _offset: u64,
        _length: Option<u64>,
        _sum: Option<&str>,
    ) -> Result<Bytes> {
        self.note("get_range_with_validation");
        Ok(Bytes::from_static(b"ab"))
    }
    async fn load_checkpoint_with_validation(
        &self,
        _uri: &str,
        _sum: Option<&str>,
    ) -> Result<Vec<u8>> {
        self.note("load_checkpoint_with_validation");
        Ok(b"abc".to_vec())
    }
    async fn copy(&self, _src: &str, _dst: &str) -> Result<()> {
        self.note("copy");
        Ok(())
    }
    async fn rename(&self, _src: &str, _dst: &str) -> Result<()> {
        self.note("rename");
        Ok(())
    }
    async fn get_writer(&self, _uri: &str) -> Result<Box<dyn ObjectWriter>> {
        self.note("get_writer");
        Ok(Box::new(NullWriter))
    }
    async fn get_writer_with_compression(
        &self,
        _uri: &str,
        _compression: CompressionConfig,
    ) -> Result<Box<dyn ObjectWriter>> {
        self.note("get_writer_with_compression");
        Ok(Box::new(NullWriter))
    }
    async fn create_writer(
        &self,
        _uri: &str,
        _options: WriterOptions,
    ) -> Result<Box<dyn ObjectWriter>> {
        self.note("create_writer");
        Ok(Box::new(NullWriter))
    }
    async fn get_optimized(&self, _uri: &str) -> Result<Bytes> {
        self.note("get_optimized");
        Ok(Bytes::from_static(b"abc"))
    }
    async fn get_range_optimized(
        &self,
        _uri: &str,
        _offset: u64,
        _length: Option<u64>,
        _chunk: Option<usize>,
        _concurrency: Option<usize>,
    ) -> Result<Bytes> {
        self.note("get_range_optimized");
        Ok(Bytes::from_static(b"ab"))
    }
    // pre_stat_objects / pre_stat_and_cache are deliberately NOT overridden: the
    // wrapper must not forward them (its own default stats through the wrapper).
    fn has_native_copy(&self) -> bool {
        true
    }
    fn has_native_rename(&self) -> bool {
        true
    }
    async fn seed_size_cache(&self, _sizes: &HashMap<String, u64>) {
        self.note("seed_size_cache");
    }
    async fn mkdir(&self, _uri: &str) -> Result<()> {
        self.note("mkdir");
        Ok(())
    }
    async fn rmdir(&self, _uri: &str, _recursive: bool) -> Result<()> {
        self.note("rmdir");
        Ok(())
    }
    async fn update_metadata(&self, _uri: &str, _metadata: &HashMap<String, String>) -> Result<()> {
        self.note("update_metadata");
        Ok(())
    }
    async fn update_properties(&self, _uri: &str, _properties: &ObjectProperties) -> Result<()> {
        self.note("update_properties");
        Ok(())
    }
}

/// Call `method` once on `store`.
async fn call(store: &dyn ObjectStore, method: &str) {
    use futures::StreamExt;
    let uri = "file:///data/obj";
    let uris = vec![uri.to_string(), "file:///data/other".to_string()];
    match method {
        "get" => drop(store.get(uri).await.unwrap()),
        "get_range" => drop(store.get_range(uri, 0, Some(2)).await.unwrap()),
        "put" => store.put(uri, Bytes::from_static(b"abc")).await.unwrap(),
        "put_multipart" => store
            .put_multipart(uri, Bytes::from_static(b"abc"), None)
            .await
            .unwrap(),
        "list" => drop(store.list("file:///data/", false).await.unwrap()),
        "list_stream" => {
            let mut s = store.list_stream("file:///data/", false);
            while let Some(item) = s.next().await {
                item.unwrap();
            }
        }
        "stat" => drop(store.stat(uri).await.unwrap()),
        "stat_if_exists" => assert!(store.stat_if_exists(uri).await.unwrap().is_some()),
        "list_objects_with_prefix" => {
            let mut s = store.list_objects_with_prefix("file:///data/");
            while let Some(item) = s.next().await {
                item.unwrap();
            }
        }
        "delete" => store.delete(uri).await.unwrap(),
        "delete_batch" => store.delete_batch(&uris).await.unwrap(),
        "delete_prefix" => store.delete_prefix("file:///data/").await.unwrap(),
        "create_container" => store.create_container("/data").await.unwrap(),
        "delete_container" => store.delete_container("/data").await.unwrap(),
        "exists" => assert!(store.exists(uri).await.unwrap()),
        "get_with_validation" => drop(store.get_with_validation(uri, None).await.unwrap()),
        "get_range_with_validation" => drop(
            store
                .get_range_with_validation(uri, 0, Some(2), None)
                .await
                .unwrap(),
        ),
        "load_checkpoint_with_validation" => drop(
            store
                .load_checkpoint_with_validation(uri, None)
                .await
                .unwrap(),
        ),
        "get_optimized" => drop(store.get_optimized(uri).await.unwrap()),
        "get_range_optimized" => drop(
            store
                .get_range_optimized(uri, 0, Some(2), None, None)
                .await
                .unwrap(),
        ),
        "copy" => store.copy(uri, "file:///data/copy").await.unwrap(),
        "rename" => store.rename(uri, "file:///data/renamed").await.unwrap(),
        "mkdir" => store.mkdir("file:///data/dir").await.unwrap(),
        "rmdir" => store.rmdir("file:///data/dir", true).await.unwrap(),
        "update_metadata" => store.update_metadata(uri, &HashMap::new()).await.unwrap(),
        "update_properties" => store
            .update_properties(uri, &ObjectProperties::default())
            .await
            .unwrap(),
        "has_native_copy" => assert!(store.has_native_copy(), "answered by the inner store"),
        "has_native_rename" => assert!(store.has_native_rename(), "answered by the inner store"),
        "seed_size_cache" => {
            store
                .seed_size_cache(&HashMap::from([(uri.to_string(), 3)]))
                .await
        }
        "pre_stat_objects" => drop(store.pre_stat_objects(&uris, 2).await.unwrap()),
        "pre_stat_and_cache" => assert_eq!(store.pre_stat_and_cache(&uris, 2).await.unwrap(), 2),
        "get_writer" => drop(store.get_writer(uri).await.unwrap()),
        "get_writer_with_compression" => drop(
            store
                .get_writer_with_compression(uri, CompressionConfig::None)
                .await
                .unwrap(),
        ),
        "create_writer" => drop(
            store
                .create_writer(uri, WriterOptions::default())
                .await
                .unwrap(),
        ),
        other => panic!("test has no call for ObjectStore::{other}"),
    }
}

/// Fails when a method is added to (or removed from) the `ObjectStore` trait
/// without this test being updated: an unforwarded new method would otherwise
/// fall through to its trait default on the logging wrapper.
#[test]
fn every_trait_method_is_covered_by_this_test() {
    let source = include_str!("../src/object_store.rs");
    let start = source
        .find("pub trait ObjectStore: Send + Sync {")
        .expect("trait ObjectStore");
    let body = &source[start..];
    let end = body.find("\n}\n").expect("end of trait");
    let mut in_trait: Vec<&str> = body[..end]
        .lines()
        .filter_map(|l| {
            let l = l.strip_prefix("    ")?;
            if l.starts_with(' ') {
                return None; // deeper indentation: not a method signature line
            }
            let l = l.strip_prefix("async ").unwrap_or(l);
            let name = l.strip_prefix("fn ")?;
            Some(name.split(['(', '<']).next().unwrap())
        })
        .collect();
    in_trait.sort_unstable();
    let mut covered: Vec<&str> = METHODS.iter().map(|m| m.0).collect();
    covered.sort_unstable();
    assert_eq!(
        in_trait, covered,
        "ObjectStore changed: add the new method to METHODS and to call() in this test, \
         and forward it in LoggedObjectStore"
    );
}

#[tokio::test]
async fn logged_store_reaches_every_inner_method_and_records_only_named_ops() {
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let dir = tempfile::tempdir().unwrap();
    let oplog = dir.path().join("ops.tsv.zst");
    init_op_logger(oplog.to_str().unwrap()).unwrap();

    let calls: CallLog = Arc::new(Mutex::new(Vec::new()));
    let store = LoggedObjectStore::new(
        Arc::new(CountingStore {
            calls: calls.clone(),
        }),
        global_logger().unwrap(),
    );

    let mut expected_rows = Vec::new();
    let mut not_forwarded = Vec::new();
    for (method, rows_of_call, inner) in METHODS {
        calls.lock().unwrap().clear();
        // A default impl on the wrapper may call other wrapper methods and fail
        // (mkdir/rmdir bail): record that instead of stopping at the first one.
        let outcome = std::panic::AssertUnwindSafe(call(&store, method));
        let panicked = futures::FutureExt::catch_unwind(outcome).await.is_err();
        let reached = calls.lock().unwrap().clone();
        if panicked || reached != *inner {
            not_forwarded.push(format!(
                "{method}: reached {reached:?}{}, expected {inner:?}",
                if panicked { " and FAILED" } else { "" }
            ));
        }
        expected_rows.extend_from_slice(rows_of_call);
    }
    assert!(
        not_forwarded.is_empty(),
        "LoggedObjectStore does not forward these methods to the inner store:\n{}",
        not_forwarded.join("\n")
    );

    finalize_op_logger();
    let rows = read_rows(&oplog);
    assert_eq!(ops(&rows), expected_rows, "rows: {rows:#?}");
}
