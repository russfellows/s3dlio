// src/s3_ops.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! High-level S3 operations (GET, PUT, STAT, etc.).
//!
//! This module provides an `S3Ops` struct that wraps the AWS S3 client
//! to provide simplified I/O operations. It writes no op-log rows (see the
//! `S3Ops` docs).

use crate::retry::retry_get_body;
use crate::s3_utils::{sdk_anyhow, SdkResultExt};
use anyhow::{bail, Context, Result};
use aws_sdk_s3::{types::Delete, Client};
use bytes::Bytes;

/// A thin wrapper around the S3 client that provides simplified, high-level
/// object operations (GET, PUT, STAT, ...).
///
/// `S3Ops` does **not** write op-log rows. Recording an operation is the job of
/// the layer the caller uses to reach it (`LoggedObjectStore`, or the public
/// helpers in `s3_utils`), so that one call is exactly one row (issue #169).
pub struct S3Ops {
    client: Client,
}

impl S3Ops {
    /// Creates a new S3 operations handler around `client`.
    pub fn new(client: Client) -> Self {
        Self { client }
    }

    /// GET (Download) an object.
    ///
    /// v0.9.108+ (issue #148 Phase 4b): wrapped in `retry_get_body` so
    /// body-transfer failures — which the streaming connector no longer
    /// surfaces to the SDK's smithy-level retry — are retried at this
    /// caller level. GETs are idempotent and each attempt allocates a
    /// fresh `Bytes`, so retries are trivially safe here.
    pub async fn get_object(&self, bucket: &str, key: &str) -> Result<Bytes> {
        // Retry the whole send()+collect() pair — each attempt re-issues
        // the request and drains a fresh body stream. Errors are converted
        // to anyhow at the closure boundary via `sdk_context` for send()
        // (preserves the SDK's connector chain — see #506) and `context`
        // for the body-collect step.
        // Zero-copy: the collected body is returned as `Bytes`.
        retry_get_body(|| async {
            let output = self
                .client
                .get_object()
                .bucket(bucket)
                .key(key)
                .send()
                .await
                .sdk_context(format!("S3 GET for s3://{}/{} failed", bucket, key))?;
            let body = output
                .body
                .collect()
                .await
                .context("S3 GET body stream failed")?
                .into_bytes();
            Ok(body)
        })
        .await
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
        let range = format!("bytes={}-{}", start, end);

        retry_get_body(|| async {
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
            let body = output
                .body
                .collect()
                .await
                .context("S3 GET_RANGE body stream failed")?
                .into_bytes();
            Ok(body)
        })
        .await
    }

    /// PUT (Upload) an object.
    ///
    /// Accepts `Bytes` for zero-copy conversion to AWS SDK ByteStream.
    pub async fn put_object(&self, bucket: &str, key: &str, data: Bytes) -> Result<()> {
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
        // Surface the SDK connector chain (e.g. dispatch failure detail)
        // instead of the bare SdkError Display. See note in get_object.
        result.map_err(|e| sdk_anyhow(format!("S3 PUT for s3://{}/{} failed", bucket, key), e))
    }

    /// STAT (Metadata) of an object.
    pub async fn stat_object(&self, bucket: &str, key: &str) -> Result<()> {
        self.client
            .head_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .map(|_| ())
            .map_err(|e| sdk_anyhow(format!("S3 HEAD for s3://{}/{} failed", bucket, key), e))
    }

    /// DELETE a single object.
    pub async fn delete_object(&self, bucket: &str, key: &str) -> Result<()> {
        self.client
            .delete_object()
            .bucket(bucket)
            .key(key)
            .send()
            .await
            .map(|_| ())
            .map_err(|e| sdk_anyhow(format!("S3 DELETE for s3://{}/{} failed", bucket, key), e))
    }

    /// DELETE multiple objects in a single batch request.
    pub async fn delete_objects(&self, bucket: &str, keys: Vec<String>) -> Result<()> {
        let num_objects = keys.len() as u32;

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

        let output = self
            .client
            .delete_objects()
            .bucket(bucket)
            .delete(delete_req)
            .send()
            .await
            .map_err(|e| {
                sdk_anyhow(format!("S3 batch DELETE for bucket '{}' failed", bucket), e)
            })?;

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
            bail!(
                "S3 batch DELETE for bucket '{}' partially failed: {} of {} object(s) \
                 not deleted: {}",
                bucket,
                errs.len(),
                num_objects,
                detail
            );
        }
        Ok(())
    }

    /// LIST objects with a given prefix.
    pub async fn list_objects(&self, bucket: &str, prefix: &str) -> Result<usize> {
        let output = self
            .client
            .list_objects_v2()
            .bucket(bucket)
            .prefix(prefix)
            .send()
            .await
            .map_err(|e| {
                sdk_anyhow(
                    format!("S3 LIST for bucket '{}' prefix '{}' failed", bucket, prefix),
                    e,
                )
            })?;
        Ok(output.contents().len())
    }
}
