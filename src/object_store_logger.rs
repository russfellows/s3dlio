// src/object_store_logger.rs
//
// Copyright, 2025. Signal65 / Futurum Group.
//
//! Logging wrapper for ObjectStore trait to enable op-log tracing for all backends.
//!
//! This module provides a decorator pattern wrapper that adds operation logging
//! to any ObjectStore implementation without modifying the underlying implementations.
//! Enables trace logging for file://, s3://, az://, and direct:// backends.

use std::sync::Arc;
use std::time::SystemTime;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use async_trait::async_trait;
use anyhow::Result;
use bytes::Bytes;

use crate::object_store::{ObjectStore, ObjectMetadata, ObjectWriter, WriterOptions};
use crate::s3_logger::{Logger, LogEntry};

/// Extract the scheme and container/bucket from a URI for endpoint logging.
/// Examples:
///   "s3://bucket/key" → "s3://bucket"
///   "file:///path/to/file" → "file://"
///   "az://container/blob" → "az://container"
///   "direct:///mnt/storage/file" → "direct://"
fn extract_endpoint(uri: &str) -> String {
    if let Some(scheme_end) = uri.find("://") {
        let after_scheme = scheme_end + 3;
        
        // For file:// and direct://, just return the scheme
        if uri.starts_with("file://") || uri.starts_with("direct://") {
            return uri[..after_scheme].to_string();
        }
        
        // For s3:// and az://, include the bucket/container name
        if let Some(path_start) = uri[after_scheme..].find('/') {
            return uri[..after_scheme + path_start].to_string();
        }
        
        // If no path separator, return entire URI
        return uri.to_string();
    }
    
    // Fallback for malformed URIs
    "unknown".to_string()
}

/// Strip the URI scheme prefix to get just the path for the "file" column.
/// Examples:
///   "s3://bucket/key.txt" → "bucket/key.txt"
///   "file:///tmp/data/file.dat" → "/tmp/data/file.dat"
///   "az://container/blob" → "container/blob"
///   "direct:///mnt/storage/file" → "/mnt/storage/file"
fn strip_uri_scheme(uri: &str) -> String {
    if let Some(scheme_end) = uri.find("://") {
        return uri[scheme_end + 3..].to_string();
    }
    
    // If no scheme found, return as-is
    uri.to_string()
}

/// Get a numeric thread identifier for logging.
/// Uses a hash of the thread ID to produce a stable usize.
fn get_thread_id() -> usize {
    let mut hasher = DefaultHasher::new();
    std::thread::current().id().hash(&mut hasher);
    hasher.finish() as usize
}

/// Wrapper that adds operation logging to any ObjectStore implementation.
/// 
/// This decorator intercepts all ObjectStore method calls, logs them via the
/// provided Logger, and then delegates to the inner store. No changes are
/// needed to existing ObjectStore implementations.
///
/// # Example
///
/// ```rust,no_run
/// use s3dlio::{store_for_uri, LoggedObjectStore, init_op_logger, global_logger, ObjectStore};
/// use std::sync::Arc;
///
/// # async fn example() -> anyhow::Result<()> {
/// // Initialize logger
/// init_op_logger("trace.tsv.zst")?;
/// let logger = global_logger().unwrap();
///
/// // Create store and wrap with logging
/// let store = store_for_uri("file:///tmp/data")?;
/// let logged_store = LoggedObjectStore::new(Arc::from(store), logger);
///
/// // All operations now logged
/// let data = logged_store.get("file:///tmp/data/test.txt").await?;
/// # Ok(())
/// # }
/// ```
pub struct LoggedObjectStore {
    inner: Arc<dyn ObjectStore>,
    logger: Logger,
}

impl LoggedObjectStore {
    /// Create a new LoggedObjectStore wrapping the given store.
    ///
    /// # Arguments
    /// * `inner` - The underlying ObjectStore implementation to wrap
    /// * `logger` - The Logger instance to use for recording operations
    pub fn new(inner: Arc<dyn ObjectStore>, logger: Logger) -> Self {
        Self { inner, logger }
    }

    /// Helper to log an operation with standard fields.
    fn log_operation(
        &self,
        operation: &str,
        uri: &str,
        bytes: u64,
        num_objects: u32,
        start_time: SystemTime,
        end_time: SystemTime,
        error: Option<String>,
    ) {
        let entry = LogEntry {
            idx: 0, // Set by logger
            thread_id: get_thread_id(),
            operation: operation.to_string(),
            client_id: String::new(), // Empty for ObjectStore operations
            num_objects,
            bytes,
            endpoint: extract_endpoint(uri),
            file: strip_uri_scheme(uri), // Strip scheme prefix from file column
            error,
            start_time,
            first_byte_time: None, // Not tracked at ObjectStore level
            end_time,
        };
        
        self.logger.log(entry);
    }
}

#[async_trait]
impl ObjectStore for LoggedObjectStore {
    async fn get(&self, uri: &str) -> Result<Bytes> {
        let start = SystemTime::now();
        let result = self.inner.get(uri).await;
        let end = SystemTime::now();
        
        let bytes = result.as_ref().map(|d| d.len() as u64).unwrap_or(0);
        let error = result.as_ref().err().map(|e| e.to_string());
        
        self.log_operation("GET", uri, bytes, 1, start, end, error);
        
        result
    }

    async fn get_range(&self, uri: &str, offset: u64, length: Option<u64>) -> Result<Bytes> {
        let start = SystemTime::now();
        let result = self.inner.get_range(uri, offset, length).await;
        let end = SystemTime::now();
        
        let bytes = result.as_ref().map(|d| d.len() as u64).unwrap_or(0);
        let error = result.as_ref().err().map(|e| e.to_string());
        
        self.log_operation("GET_RANGE", uri, bytes, 1, start, end, error);
        
        result
    }

    async fn put(&self, uri: &str, data: &[u8]) -> Result<()> {
        let start = SystemTime::now();
        let bytes = data.len() as u64;
        let result = self.inner.put(uri, data).await;
        let end = SystemTime::now();
        
        let error = result.as_ref().err().map(|e| e.to_string());
        
        self.log_operation("PUT", uri, bytes, 1, start, end, error);
        
        result
    }

    async fn put_multipart(&self, uri: &str, data: &[u8], part_size: Option<usize>) -> Result<()> {
        let start = SystemTime::now();
        let bytes = data.len() as u64;
        let result = self.inner.put_multipart(uri, data, part_size).await;
        let end = SystemTime::now();
        
        let error = result.as_ref().err().map(|e| e.to_string());
        
        // Still log as PUT (multipart is implementation detail)
        self.log_operation("PUT", uri, bytes, 1, start, end, error);
        
        result
    }

    async fn list(&self, uri_prefix: &str, recursive: bool) -> Result<Vec<String>> {
        let start = SystemTime::now();
        let result = self.inner.list(uri_prefix, recursive).await;
        let end = SystemTime::now();
        
        let num_objects = result.as_ref().map(|v| v.len() as u32).unwrap_or(0);
        let error = result.as_ref().err().map(|e| e.to_string());
        
        self.log_operation("LIST", uri_prefix, 0, num_objects, start, end, error);
        
        result
    }

    async fn stat(&self, uri: &str) -> Result<ObjectMetadata> {
        let start = SystemTime::now();
        let result = self.inner.stat(uri).await;
        let end = SystemTime::now();
        
        let bytes = result.as_ref().map(|m| m.size).unwrap_or(0);
        let error = result.as_ref().err().map(|e| e.to_string());
        
        self.log_operation("HEAD", uri, bytes, 1, start, end, error);
        
        result
    }

    async fn delete(&self, uri: &str) -> Result<()> {
        let start = SystemTime::now();
        let result = self.inner.delete(uri).await;
        let end = SystemTime::now();
        
        let error = result.as_ref().err().map(|e| e.to_string());
        
        self.log_operation("DELETE", uri, 0, 1, start, end, error);
        
        result
    }

    async fn delete_batch(&self, uris: &[String]) -> Result<()> {
        let start = SystemTime::now();
        let result = self.inner.delete_batch(uris).await;
        let end = SystemTime::now();
        
        let error = result.as_ref().err().map(|e| e.to_string());
        
        // Log batch delete with object count
        let uri_str = if uris.is_empty() { "(empty)" } else { &uris[0] };
        self.log_operation("DELETE_BATCH", uri_str, 0, uris.len() as u32, start, end, error);
        
        result
    }

    async fn delete_prefix(&self, uri_prefix: &str) -> Result<()> {
        let start = SystemTime::now();
        let result = self.inner.delete_prefix(uri_prefix).await;
        let end = SystemTime::now();
        
        let error = result.as_ref().err().map(|e| e.to_string());
        
        // Don't know exact count, use 0 to indicate prefix operation
        self.log_operation("DELETE_PREFIX", uri_prefix, 0, 0, start, end, error);
        
        result
    }

    async fn create_container(&self, name: &str) -> Result<()> {
        let start = SystemTime::now();
        let result = self.inner.create_container(name).await;
        let end = SystemTime::now();
        
        let error = result.as_ref().err().map(|e| e.to_string());
        
        self.log_operation("CREATE_CONTAINER", name, 0, 1, start, end, error);
        
        result
    }

    async fn delete_container(&self, name: &str) -> Result<()> {
        let start = SystemTime::now();
        let result = self.inner.delete_container(name).await;
        let end = SystemTime::now();
        
        let error = result.as_ref().err().map(|e| e.to_string());
        
        self.log_operation("DELETE_CONTAINER", name, 0, 1, start, end, error);
        
        result
    }

    async fn exists(&self, uri: &str) -> Result<bool> {
        let start = SystemTime::now();
        let result = self.inner.exists(uri).await;
        let end = SystemTime::now();
        
        let error = result.as_ref().err().map(|e| e.to_string());
        
        // Log as HEAD operation (existence check)
        self.log_operation("HEAD", uri, 0, 1, start, end, error);
        
        result
    }

    async fn get_with_validation(&self, uri: &str, expected_checksum: Option<&str>) -> Result<Bytes> {
        let start = SystemTime::now();
        let result = self.inner.get_with_validation(uri, expected_checksum).await;
        let end = SystemTime::now();
        
        let bytes = result.as_ref().map(|d| d.len() as u64).unwrap_or(0);
        let error = result.as_ref().err().map(|e| e.to_string());
        
        // Log as GET with validation
        self.log_operation("GET", uri, bytes, 1, start, end, error);
        
        result
    }

    async fn get_range_with_validation(
        &self,
        uri: &str,
        offset: u64,
        length: Option<u64>,
        expected_checksum: Option<&str>,
    ) -> Result<Bytes> {
        let start = SystemTime::now();
        let result = self.inner.get_range_with_validation(uri, offset, length, expected_checksum).await;
        let end = SystemTime::now();
        
        let bytes = result.as_ref().map(|d| d.len() as u64).unwrap_or(0);
        let error = result.as_ref().err().map(|e| e.to_string());
        
        self.log_operation("GET_RANGE", uri, bytes, 1, start, end, error);
        
        result
    }

    async fn load_checkpoint_with_validation(
        &self,
        checkpoint_uri: &str,
        expected_checksum: Option<&str>,
    ) -> Result<Vec<u8>> {
        let start = SystemTime::now();
        let result = self.inner.load_checkpoint_with_validation(checkpoint_uri, expected_checksum).await;
        let end = SystemTime::now();
        
        let bytes = result.as_ref().map(|d| d.len() as u64).unwrap_or(0);
        let error = result.as_ref().err().map(|e| e.to_string());
        
        self.log_operation("GET", checkpoint_uri, bytes, 1, start, end, error);
        
        result
    }

    async fn create_writer(&self, uri: &str, options: WriterOptions) -> Result<Box<dyn ObjectWriter>> {
        // Note: We log the final write, not the writer creation
        // The writer itself will handle logging when finalized
        self.inner.create_writer(uri, options).await
    }

    async fn get_writer(&self, uri: &str) -> Result<Box<dyn ObjectWriter>> {
        // Delegate to inner store - writer operations logged at write time
        self.inner.get_writer(uri).await
    }

    async fn get_writer_with_compression(&self, uri: &str, compression: crate::object_store::CompressionConfig) -> Result<Box<dyn ObjectWriter>> {
        // Delegate to inner store - writer operations logged at write time
        self.inner.get_writer_with_compression(uri, compression).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_endpoint() {
        assert_eq!(extract_endpoint("s3://my-bucket/path/to/key"), "s3://my-bucket");
        assert_eq!(extract_endpoint("s3://bucket"), "s3://bucket");
        assert_eq!(extract_endpoint("file:///tmp/test.txt"), "file://");
        assert_eq!(extract_endpoint("file:///path/to/file"), "file://");
        assert_eq!(extract_endpoint("az://container/blob/path"), "az://container");
        assert_eq!(extract_endpoint("direct:///mnt/storage/file"), "direct://");
        assert_eq!(extract_endpoint("malformed"), "unknown");
    }

    #[test]
    fn test_strip_uri_scheme() {
        assert_eq!(strip_uri_scheme("s3://bucket/path/to/key.txt"), "bucket/path/to/key.txt");
        assert_eq!(strip_uri_scheme("file:///tmp/test.txt"), "/tmp/test.txt");
        assert_eq!(strip_uri_scheme("file:///path/to/file"), "/path/to/file");
        assert_eq!(strip_uri_scheme("az://container/blob/path"), "container/blob/path");
        assert_eq!(strip_uri_scheme("direct:///mnt/storage/file.dat"), "/mnt/storage/file.dat");
        assert_eq!(strip_uri_scheme("no-scheme-path"), "no-scheme-path");
    }

    #[test]
    fn test_get_thread_id() {
        let id1 = get_thread_id();
        let id2 = get_thread_id();
        // Same thread should produce same ID
        assert_eq!(id1, id2);
    }
}
