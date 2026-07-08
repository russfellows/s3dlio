// src/data_loader/s3_bytes.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use crate::data_loader::options::{LoaderOptions, ReaderMode};
use crate::data_loader::parallel_fetch::DropCancel;
use crate::data_loader::{Dataset, DatasetError};
use crate::s3_utils::{
    get_object_range_uri_async, // NEW: implemented in s3_utils.rs next
    get_object_uri_async,
    list_objects as list_objects_rs,
    parse_s3_uri,
    stat_object_uri_async,
};
use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures::stream::{FuturesOrdered, StreamExt};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

#[derive(Clone)]
pub struct S3BytesDataset {
    bucket: String,
    keys: Vec<String>,
    // NEW:
    reader_mode: ReaderMode,
    part_size: usize,
    max_inflight_parts: usize,
}

impl S3BytesDataset {
    pub fn from_prefix(uri: &str) -> Result<Self, DatasetError> {
        Self::from_prefix_with_opts(uri, &LoaderOptions::default())
    }

    /// NEW: honor LoaderOptions (reader strategy + part params)
    pub fn from_prefix_with_opts(uri: &str, opts: &LoaderOptions) -> Result<Self, DatasetError> {
        let (bucket, prefix) = parse_s3_uri(uri).map_err(|e| DatasetError::from(e.to_string()))?;

        // Recursively list keys under prefix
        let keys = list_objects_rs(&bucket, &prefix, true)
            .map_err(|e| DatasetError::from(e.to_string()))?;

        Ok(Self {
            bucket,
            keys,
            reader_mode: opts.reader_mode,
            part_size: opts.part_size,
            max_inflight_parts: opts.max_inflight_parts,
        })
    }

    #[inline]
    pub fn keys(&self) -> &Vec<String> {
        &self.keys
    }
}

#[async_trait]
impl Dataset for S3BytesDataset {
    type Item = Bytes;

    fn len(&self) -> Option<usize> {
        Some(self.keys.len())
    }

    fn keys(&self) -> Option<Vec<String>> {
        Some(self.keys.clone())
    }

    async fn get(&self, idx: usize) -> Result<Self::Item, DatasetError> {
        let key = self
            .keys
            .get(idx)
            .ok_or(DatasetError::IndexOutOfRange(idx))?;

        let uri = format!("s3://{}/{}", self.bucket, key);

        match self.reader_mode {
            ReaderMode::Sequential => {
                let bytes = get_object_uri_async(&uri)
                    .await
                    .map_err(DatasetError::from)?;
                // Return Bytes directly - zero-copy!
                Ok(bytes)
            }
            ReaderMode::Range => {
                // HEAD to learn size
                let meta = stat_object_uri_async(&uri)
                    .await
                    .map_err(DatasetError::from)?;
                let size = meta.size;
                if size == 0 {
                    return Ok(Bytes::new());
                }
                let part = self.part_size.max(1) as u64;
                let n_parts = size.div_ceil(part) as usize;
                let max_inflight = self.max_inflight_parts.max(1);

                // Task-level parallelism (issue #148 site 3.1h): each
                // range GET is `tokio::spawn`'d so tokio can distribute
                // per-range CPU work (signing, header parsing, body
                // assembly) across worker threads. Prior `.buffered(N)`
                // polled every in-flight fetch inside a single task.
                //
                // Uses the same shape as 3.1f (range_engine_generic):
                // FuturesOrdered preserves in-order assembly (running
                // write_offset + short-read semantics), and a bounded
                // spawn pool (prime + refill-on-consume) keeps at most
                // `max_inflight` fetches alive at once so peak memory
                // stays ~size instead of blowing up to size * n_parts.
                // DropCancel guard cancels in-flight fetches if the
                // caller drops this future mid-download.
                let cancel = CancellationToken::new();
                let _drop_cancel = DropCancel(cancel.clone());

                let spawn_part = |i: usize| {
                    let start = (i as u64) * part;
                    let len = (size - start).min(part);
                    let uri = uri.clone();
                    let token = cancel.clone();
                    tokio::spawn(async move {
                        tokio::select! {
                            _ = token.cancelled() => Err(anyhow::anyhow!(
                                "range GET cancelled: part {}", i
                            )),
                            r = get_object_range_uri_async(&uri, start, Some(len)) => r,
                        }
                    })
                };

                let mut pending: FuturesOrdered<JoinHandle<anyhow::Result<Bytes>>> =
                    FuturesOrdered::new();
                let mut next_part = 0usize;
                while next_part < n_parts && pending.len() < max_inflight {
                    pending.push_back(spawn_part(next_part));
                    next_part += 1;
                }

                let mut out = BytesMut::zeroed(size as usize);
                let mut write_offset: usize = 0;
                let mut first_err: Option<DatasetError> = None;

                while let Some(join_res) = pending.next().await {
                    match join_res {
                        Ok(Ok(bytes)) => {
                            if first_err.is_none() {
                                let len = bytes.len();
                                let end = write_offset + len;
                                if end > out.len() {
                                    return Err(DatasetError::from(format!(
                                        "range assembly overflow: writing {}..{} but buffer is {} bytes",
                                        write_offset, end, out.len()
                                    )));
                                }
                                out[write_offset..end].copy_from_slice(&bytes);
                                write_offset = end;
                            }
                            drop(bytes);
                        }
                        Ok(Err(e)) => {
                            if first_err.is_none() {
                                first_err = Some(DatasetError::from(e));
                                cancel.cancel();
                            }
                        }
                        Err(join_err) if join_err.is_panic() => {
                            if first_err.is_none() {
                                first_err = Some(DatasetError::from(format!(
                                    "range GET task panicked: {}",
                                    join_err
                                )));
                                cancel.cancel();
                            }
                        }
                        Err(_cancelled) => {}
                    }

                    if first_err.is_none() && next_part < n_parts {
                        pending.push_back(spawn_part(next_part));
                        next_part += 1;
                    }
                }

                if let Some(e) = first_err {
                    return Err(e);
                }

                if write_offset < out.len() {
                    out.truncate(write_offset);
                }
                Ok(out.freeze())
            }
        }
    }
}
