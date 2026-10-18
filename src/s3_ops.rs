// src/s3_ops.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! High-level, logged S3 operations (GET, PUT, STAT, etc.).
//!
//! This module provides an `S3Ops` struct that wraps the AWS S3 client
//! to provide simplified, blocking I/O operations with detailed tracing.

use crate::object_store::HeadMetadata;
use crate::receive_progress::ReceiveProgress;
use crate::retry::retry_get_body;
use crate::s3_logger::{LogEntry, Logger};
use crate::s3_utils::{format_sdk_error, sdk_anyhow, SdkResultExt};
use anyhow::{bail, Context, Result};
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

/// The one row of a call that must be written exactly once, whichever way the call ends
/// (the helper-row counterpart of the wrapper's call row).  Created when the call starts and
/// consumed by [`CallRowGuard::finish`]; if it is dropped first, the future was dropped (or the
/// stack unwound), and `Drop` writes the row as `error = "cancelled"` with what was filled.
/// Nothing is built when the handler has no logger.
struct CallRowGuard<'a> {
    ops: &'a S3Ops,
    ctx: Option<LogContext>,
    /// The receive progress of a GET; `None` for a HEAD, which has no payload and no first byte.
    progress: Option<&'a ReceiveProgress>,
}

impl<'a> CallRowGuard<'a> {
    fn start(
        ops: &'a S3Ops,
        operation: &'static str,
        key: &str,
        progress: Option<&'a ReceiveProgress>,
    ) -> Self {
        let ctx = ops.logger.is_some().then(|| LogContext {
            operation,
            key: key.to_string(),
            num_objects: 1,
            start_time: SystemTime::now(),
        });
        Self { ops, ctx, progress }
    }

    /// Write the row of a call that returned.  A GET's byte count is the progress; a HEAD's is
    /// `size_of(value)` on success and 0 on failure.
    fn finish<T>(mut self, result: &Result<T>, size_of: impl Fn(&T) -> u64) {
        let Some(ctx) = self.ctx.take() else { return };
        let end = SystemTime::now();
        let (bytes, first_byte) = match self.progress {
            Some(p) => (p.filled(), Some(p.first_byte().unwrap_or(end))),
            None => (result.as_ref().map_or(0, size_of), None),
        };
        match result {
            Ok(_) => self.ops.log_op(ctx, &Ok::<(), &str>(()), bytes, first_byte),
            Err(e) => self
                .ops
                .log_op(ctx, &Err::<(), _>(format!("{e:#}")), bytes, first_byte),
        }
    }
}

impl Drop for CallRowGuard<'_> {
    fn drop(&mut self) {
        let Some(ctx) = self.ctx.take() else { return };
        let now = SystemTime::now();
        let (bytes, first_byte) = match self.progress {
            Some(p) => (p.filled(), Some(p.first_byte().unwrap_or(now))),
            None => (0, None),
        };
        self.ops
            .log_op(ctx, &Err::<(), _>("cancelled"), bytes, first_byte);
    }
}

impl S3Ops {
    /// Creates a new S3 operations handler.
    pub fn new(client: Client, logger: Option<Logger>, client_id: &str, endpoint: &str) -> Self {
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
    ///
    /// v0.9.108+ (issue #148 Phase 4b): wrapped in `retry_get_body` so
    /// body-transfer failures — which the streaming connector no longer
    /// surfaces to the SDK's smithy-level retry — are retried at this
    /// caller level. GETs are idempotent and each attempt allocates a
    /// fresh `Bytes`, so retries are trivially safe here.
    pub async fn get_object(&self, bucket: &str, key: &str) -> Result<Bytes> {
        let ctx = LogContext {
            operation: "GET",
            key: key.to_string(),
            num_objects: 1,
            start_time: SystemTime::now(),
        };

        // Retry the whole send()+collect() pair — each attempt re-issues
        // the request and drains a fresh body stream. Each closure
        // invocation returns (Bytes, SystemTime) so `first_byte_time`
        // reflects the successful attempt without capturing outer state
        // through the FnMut closure. Errors are converted to anyhow at
        // the closure boundary via `sdk_context` for send() (preserves
        // the SDK's connector chain — see #506) and `context` for the
        // body-collect step.
        let result: Result<(Bytes, SystemTime)> = retry_get_body(|| async {
            let output = self
                .client
                .get_object()
                .bucket(bucket)
                .key(key)
                .send()
                .await
                .sdk_context(format!("S3 GET for s3://{}/{} failed", bucket, key))?;
            let first_byte = SystemTime::now();
            let body = output
                .body
                .collect()
                .await
                .context("S3 GET body stream failed")?
                .into_bytes();
            Ok((body, first_byte))
        })
        .await;

        match result {
            Ok((body, first_byte_time)) => {
                let bytes = body.len() as u64;
                self.log_op(ctx, &Ok::<(), &str>(()), bytes, Some(first_byte_time));
                // Zero-copy: return Bytes directly instead of converting to Vec
                Ok(body)
            }
            Err(e) => {
                // `sdk_context` already flattened the SDK connector
                // chain into the message — Display of the anyhow chain
                // via `{:#}` reproduces the same detail we'd get from
                // format_sdk_error() applied directly.
                let detail = format!("{:#}", e);
                self.log_op(ctx, &Err::<(), _>(detail), 0, None);
                Err(e)
            }
        }
    }

    /// GET (Download) a range of bytes from an object.
    ///
    /// v0.9.108+ (issue #148 Phase 4b): wrapped in `retry_get_body` —
    /// see the get_object doc-comment for the same reasoning. Range
    /// GETs are idempotent, so retrying the whole send()+collect()
    /// pair after a body-transfer failure is trivially safe.
    pub async fn get_object_range(
        &self,
        bucket: &str,
        key: &str,
        start: u64,
        end: u64,
    ) -> Result<Bytes> {
        let ctx = LogContext {
            operation: "GET_RANGE",
            key: format!("{}[{}-{}]", key, start, end),
            num_objects: 1,
            start_time: SystemTime::now(),
        };
        let range = format!("bytes={}-{}", start, end);

        let result: Result<(Bytes, SystemTime)> = retry_get_body(|| async {
            let output = self
                .client
                .get_object()
                .bucket(bucket)
                .key(key)
                .range(&range)
                .send()
                .await
                .sdk_context(format!(
                    "S3 GET_RANGE for s3://{}/{} bytes={}-{} failed",
                    bucket, key, start, end
                ))?;
            let first_byte = SystemTime::now();
            let body = output
                .body
                .collect()
                .await
                .context("S3 GET_RANGE body stream failed")?
                .into_bytes();
            Ok((body, first_byte))
        })
        .await;

        match result {
            Ok((body, first_byte_time)) => {
                let bytes = body.len() as u64;
                self.log_op(ctx, &Ok::<(), &str>(()), bytes, Some(first_byte_time));
                // Zero-copy: return Bytes directly
                Ok(body)
            }
            Err(e) => {
                let detail = format!("{:#}", e);
                self.log_op(ctx, &Err::<(), _>(detail), 0, None);
                Err(e)
            }
        }
    }

    /// GET (Download) an object into the caller's buffer (plan §4.5 A6, A8).
    ///
    /// `dst.len()` is the expected length; `Ok(())` means exactly that many bytes were
    /// written and the whole-object size equals it (see
    /// `s3_utils::get_into_with_client` for the contract and the routes).  The request
    /// goes to THIS handler's client only.
    ///
    /// With a logger the call writes exactly ONE `GET` row, whichever way it ends: the byte
    /// count is `progress.filled()` (which can be non-zero with an error), the first byte is
    /// the receive's own (else the end time), and a call whose future is dropped writes
    /// `error = "cancelled"`.  Internal ranges, retries and the fan-out HEAD write no row.
    /// Without a logger nothing is written.
    pub async fn get_into(
        &self,
        bucket: &str,
        key: &str,
        dst: &mut [u8],
        progress: &ReceiveProgress,
    ) -> Result<()> {
        let row = CallRowGuard::start(self, "GET", key, Some(progress));
        let result =
            crate::s3_utils::get_into_with_client(&self.client, bucket, key, dst, progress).await;
        row.finish(&result, |_| 0);
        result
    }

    /// Typed HEAD of an object: `Ok(None)` only for typed absence (see
    /// `s3_utils::head_typed_with_client`).
    ///
    /// A HEAD 404 has no body, so a missing bucket (`NoSuchBucket`) cannot be told from a
    /// missing key: both are `Ok(None)`.  The service's own answer is not visible here.
    ///
    /// A 403 and every other failure are errors, never `Ok(None)`.
    ///
    /// With a logger the call writes exactly ONE `HEAD` row: `bytes` is the size on success
    /// and 0 for absence, an error and a cancelled call, and the first-byte column is empty on
    /// every outcome.
    pub async fn head_typed(&self, bucket: &str, key: &str) -> Result<Option<HeadMetadata>> {
        let row = CallRowGuard::start(self, "HEAD", key, None);
        let result = crate::s3_utils::head_typed_with_client(&self.client, bucket, key).await;
        row.finish(&result, |m| m.as_ref().map_or(0, |m| m.size));
        result
    }

    /// PUT (Upload) an object.
    ///
    /// Accepts `Bytes` for zero-copy conversion to AWS SDK ByteStream.
    pub async fn put_object(&self, bucket: &str, key: &str, data: Bytes) -> Result<()> {
        let ctx = LogContext {
            operation: "PUT",
            key: key.to_string(),
            num_objects: 1,
            start_time: SystemTime::now(),
        };
        let bytes = data.len() as u64;
        let body = data.into(); // Bytes → ByteStream is zero-copy (refcount bump)

        let result = if crate::s3_client::unsigned_payload_enabled() {
            self.client
                .put_object()
                .bucket(bucket)
                .key(key)
                .body(body)
                .customize()
                .disable_payload_signing()
                .send()
                .await
                .map(|_| ())
        } else {
            self.client
                .put_object()
                .bucket(bucket)
                .key(key)
                .body(body)
                .send()
                .await
                .map(|_| ())
        };
        match result {
            Ok(()) => {
                self.log_op(ctx, &Ok::<(), &str>(()), bytes, None);
                Ok(())
            }
            Err(e) => {
                // Surface the SDK connector chain (e.g. dispatch failure detail)
                // instead of the bare SdkError Display. See note in get_object.
                let detail = format_sdk_error(&e);
                self.log_op(ctx, &Err::<(), _>(detail), 0, None);
                Err(sdk_anyhow(
                    format!("S3 PUT for s3://{}/{} failed", bucket, key),
                    e,
                ))
            }
        }
    }

    /// STAT (Metadata) of an object.
    pub async fn stat_object(&self, bucket: &str, key: &str) -> Result<()> {
        let ctx = LogContext {
            operation: "STAT",
            key: key.to_string(),
            num_objects: 1,
            start_time: SystemTime::now(),
        };

        let result = self
            .client
            .head_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await;
        match result {
            Ok(_) => {
                self.log_op(ctx, &Ok::<(), &str>(()), 0, None);
                Ok(())
            }
            Err(e) => {
                let detail = format_sdk_error(&e);
                self.log_op(ctx, &Err::<(), _>(detail), 0, None);
                Err(sdk_anyhow(
                    format!("S3 HEAD for s3://{}/{} failed", bucket, key),
                    e,
                ))
            }
        }
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
        match result {
            Ok(_) => {
                self.log_op(ctx, &Ok::<(), &str>(()), 0, None);
                Ok(())
            }
            Err(e) => {
                let detail = format_sdk_error(&e);
                self.log_op(ctx, &Err::<(), _>(detail), 0, None);
                Err(sdk_anyhow(
                    format!("S3 DELETE for s3://{}/{} failed", bucket, key),
                    e,
                ))
            }
        }
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
            .map(|k| {
                aws_sdk_s3::types::ObjectIdentifier::builder()
                    .key(k)
                    .build()
            })
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

        match result {
            Ok(output) => {
                // Locked contract (audit #151 f36, docs/implementation-plans/
                // v0.9.109-audit-fix-plan.md bug A4): S3's DeleteObjects
                // returns HTTP 200 even when individual keys failed to
                // delete (ACL denial, object-lock retention, throttling,
                // etc.) — the per-object outcome lives in the response
                // body's `errors` list, not the HTTP status. A bare
                // `Ok(_)` here silently discarded that list, so a caller
                // could believe a batch fully succeeded when some or all
                // of it didn't. Keep the `Result<()>` signature (no API
                // break) but inspect `.errors()` and surface them as Err.
                let errs = output.errors();
                if !errs.is_empty() {
                    let detail = errs
                        .iter()
                        .map(|e| {
                            format!(
                                "{}: {} ({})",
                                e.key().unwrap_or("<unknown key>"),
                                e.code().unwrap_or("<unknown code>"),
                                e.message().unwrap_or("<no message>")
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(", ");
                    self.log_op(ctx, &Err::<(), _>(detail.clone()), 0, None);
                    bail!(
                        "S3 batch DELETE for bucket '{}' partially failed: {} of {} object(s) \
                         not deleted: {}",
                        bucket,
                        errs.len(),
                        num_objects,
                        detail
                    );
                }
                self.log_op(ctx, &Ok::<(), &str>(()), 0, None);
                Ok(())
            }
            Err(e) => {
                let detail = format_sdk_error(&e);
                self.log_op(ctx, &Err::<(), _>(detail), 0, None);
                Err(sdk_anyhow(
                    format!("S3 batch DELETE for bucket '{}' failed", bucket),
                    e,
                ))
            }
        }
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
                let detail = format_sdk_error(&e);
                self.log_op(ctx, &Err::<(), _>(detail), 0, None);
                Err(sdk_anyhow(
                    format!("S3 LIST for bucket '{}' prefix '{}' failed", bucket, prefix),
                    e,
                ))
            }
        }
    }
}
