// src/s3_ops.rs
//
// Copyright, 2025. Signal65 / Futurum Group.
//
//! High-level, logged S3 operations (GET, PUT, STAT, etc.).
//!
//! This module provides an `S3Ops` struct that wraps the AWS S3 client
//! to provide simplified, blocking I/O operations with detailed tracing.

use crate::s3_logger::{LogEntry, Logger};
use anyhow::Result;
use aws_sdk_s3::{types::Delete, Client};
use bytes::Bytes;
//use std::sync::Arc;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};


use std::time::SystemTime;

/// A wrapper around the S3 client to provide logged, high-level operations.
pub struct S3Ops {
    client: Client,
    logger: Option<Logger>,
    client_id: String,
    endpoint: String,
}

/// Private helper struct to reduce logging boilerplate.
struct LogContext {
    operation: &'static str,
    key: String,
    num_objects: u32,
    start_time: SystemTime,
}

impl S3Ops {
    /// Creates a new S3 operations handler.
    pub fn new(
        client: Client,
        logger: Option<Logger>,
        client_id: &str,
        endpoint: &str,
    ) -> Self {
        Self {
            client,
            logger,
            client_id: client_id.to_string(),
            endpoint: endpoint.to_string(),
        }
    }

    /// Centralized helper function to log s3 operation's results.
    fn log_op(
        &self,
        ctx: LogContext,
        result: &Result<impl Sized, impl std::string::ToString>,
        bytes: u64,
        first_byte_time: Option<SystemTime>,
    ) {
        if let Some(logger) = &self.logger {
            let end_time = SystemTime::now();

            let mut hasher = DefaultHasher::new();
            std::thread::current().id().hash(&mut hasher);
            let thread_id = hasher.finish() as usize;

            let entry = LogEntry {
                idx: 0, // Will be set by the logger
                thread_id,
                operation: ctx.operation.to_string(),
                client_id: self.client_id.clone(),
                num_objects: ctx.num_objects,
                bytes,
                endpoint: self.endpoint.clone(),
                file: ctx.key,
                error: result.as_ref().err().map(|e| e.to_string()),
                start_time: ctx.start_time,
                first_byte_time,
                end_time,
            };
            logger.log(entry);
        }
    }

    /// GET (Download) an object.
    pub async fn get_object(&self, bucket: &str, key: &str) -> Result<Bytes> {
        let ctx = LogContext {
            operation: "GET",
            key: key.to_string(),
            num_objects: 1,
            start_time: SystemTime::now(),
        };

        let result = self.client.get_object().bucket(bucket).key(key).send().await;
        let first_byte_time = SystemTime::now();

        match result {
            Ok(output) => {
                let body = output.body.collect().await?.into_bytes();
                let bytes = body.len() as u64;
                self.log_op(ctx, &Ok::<(), &str>(()), bytes, Some(first_byte_time));
                // Zero-copy: return Bytes directly instead of converting to Vec
                Ok(body)
            }
            Err(e) => {
                self.log_op(ctx, &Err::<(), _>(e.to_string()), 0, None); 
                Err(e.into())
            }
        }
    }

    /// GET (Download) a range of bytes from an object.
    pub async fn get_object_range(&self, bucket: &str, key: &str, start: u64, end: u64) -> Result<Bytes> {
        let ctx = LogContext {
            operation: "GET_RANGE",
            key: format!("{}[{}-{}]", key, start, end),
            num_objects: 1,
            start_time: SystemTime::now(),
        };

        let range = format!("bytes={}-{}", start, end);
        let result = self.client
            .get_object()
            .bucket(bucket)
            .key(key)
            .range(range)
            .send()
            .await;
        let first_byte_time = SystemTime::now();

        match result {
            Ok(output) => {
                let body = output.body.collect().await?.into_bytes();
                let bytes = body.len() as u64;
                self.log_op(ctx, &Ok::<(), &str>(()), bytes, Some(first_byte_time));
                // Zero-copy: return Bytes directly
                Ok(body)
            }
            Err(e) => {
                self.log_op(ctx, &Err::<(), _>(e.to_string()), 0, None); 
                Err(e.into())
            }
        }
    }

    /// PUT (Upload) an object.
    pub async fn put_object(&self, bucket: &str, key: &str, data: Vec<u8>) -> Result<()> {
        let ctx = LogContext {
            operation: "PUT",
            key: key.to_string(),
            num_objects: 1,
            start_time: SystemTime::now(),
        };
        let bytes = data.len() as u64;
        let body = data.into();

        let result = self
            .client
            .put_object()
            .bucket(bucket)
            .key(key)
            .body(body)
            .send()
            .await;
        self.log_op(ctx, &Ok::<(), &str>(()), bytes, None);
        Ok(result.map(|_| ())?)
    }

    /// STAT (Metadata) of an object.
    pub async fn stat_object(&self, bucket: &str, key: &str) -> Result<()> {
        let ctx = LogContext {
            operation: "STAT",
            key: key.to_string(),
            num_objects: 1,
            start_time: SystemTime::now(),
        };

        let result = self.client.head_object().bucket(bucket).key(key).send().await;
        self.log_op(ctx, &Ok::<(), &str>(()), 0, None);
        Ok(result.map(|_| ())?)
    }

    /// DELETE a single object.
    pub async fn delete_object(&self, bucket: &str, key: &str) -> Result<()> {
        let ctx = LogContext {
            operation: "DELETE",
            key: key.to_string(),
            num_objects: 1,
            start_time: SystemTime::now(),
        };

        let result = self
            .client
            .delete_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await;
        self.log_op(ctx, &Ok::<(), &str>(()), 0, None);
        Ok(result.map(|_| ())?)
    }

    /// DELETE multiple objects in a single batch request.
    pub async fn delete_objects(&self, bucket: &str, keys: Vec<String>) -> Result<()> {
        let num_objects = keys.len() as u32;
        let ctx = LogContext {
            operation: "DELETE",
            key: "(batch)".to_string(), // Use a placeholder for batch operations
            num_objects,
            start_time: SystemTime::now(),
        };


        // FIX: Collect into a Result to handle potential build errors.
        let objects_to_delete: Result<Vec<_>, _> = keys
            .into_iter()
            .map(|k| aws_sdk_s3::types::ObjectIdentifier::builder().key(k).build())
            .collect();

        let delete_req = Delete::builder()
            .set_objects(Some(objects_to_delete?))
            .build()?;

        let result = self
            .client
            .delete_objects()
            .bucket(bucket)
            .delete(delete_req)
            .send()
            .await;

        self.log_op(ctx, &result, 0, None);
        Ok(result.map(|_| ())?)

    }


    /// LIST objects with a given prefix.
    pub async fn list_objects(&self, bucket: &str, prefix: &str) -> Result<usize> {
        let ctx = LogContext {
            operation: "LIST",
            key: prefix.to_string(),
            num_objects: 0, // Will be updated after the call completes
            start_time: SystemTime::now(),
        };

        let result = self
            .client
            .list_objects_v2()
            .bucket(bucket)
            .prefix(prefix)
            .send()
            .await;

        match result {
            Ok(output) => {
                // FIX: Get the length directly from the slice.
                let count = output.contents().len();

                let log_ctx_with_count = LogContext {
                    num_objects: count as u32,
                    ..ctx
                };
                self.log_op(log_ctx_with_count, &Ok::<(), &str>(()), 0, None);
                Ok(count)
            }
            Err(e) => {
                self.log_op(ctx, &Err::<(), _>(e.to_string()), 0, None);
                Err(e.into())
            }
        }
    }
}
