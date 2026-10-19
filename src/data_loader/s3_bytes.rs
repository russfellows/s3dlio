// src/data_loader/s3_bytes.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use crate::data_loader::options::{LoaderOptions, ReaderMode};
use crate::data_loader::parallel_fetch::DropCancel;
use crate::data_loader::{Dataset, DatasetError};
use crate::delete_batch_plan::parse_s3_uri_for_global_client;
use crate::s3_utils::{
    get_object_range_into_segment_async, get_object_uri_async, list_objects as list_objects_rs,
    stat_object_uri_async,
};
use async_trait::async_trait;
use bytes::{Bytes, BytesMut};
use futures::stream::{FuturesOrdered, StreamExt};
use std::collections::VecDeque;
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
        // The policy comes first: parsing a host-form URI reads AWS_ENDPOINT_URL (#243).
        crate::dotenv::apply();
        let (bucket, prefix) =
            parse_s3_uri_for_global_client(uri).map_err(|e| DatasetError::from(e.to_string()))?;

        // Recursively list keys under prefix
        let keys = list_objects_rs(&bucket, &prefix, true)
            .map_err(|e| DatasetError::from(e.to_string()))?;
        if keys.is_empty() {
            // A bare partial name selects nothing (#213): fail loudly instead of
            // handing back a dataset of zero samples.
            return Err(DatasetError::from(
                crate::s3_selection::empty_selection_error(uri).to_string(),
            ));
        }

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

    fn uris(&self) -> Option<Vec<String>> {
        // Exactly the URI `get` fetches: the parsed bucket plus the listed key.
        Some(
            self.keys
                .iter()
                .map(|k| format!("s3://{}/{}", self.bucket, k))
                .collect(),
        )
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
                // FuturesOrdered preserves in-order assembly (a short or long
                // range is an error, never silently stitched), and a bounded
                // spawn pool (prime + refill-on-consume) keeps at most
                // `max_inflight` fetches in flight at once.
                // DropCancel guard cancels in-flight fetches if the
                // caller drops this future mid-download.
                let cancel = CancellationToken::new();
                let _drop_cancel = DropCancel(cancel.clone());

                // Zero-copy (Rule #1, stage G R4): ONE output buffer, pre-split into one
                // segment per part. Each range GET streams its response straight into its own
                // segment (the network-chunk-to-destination copy is the only copy), and the
                // segments are `unsplit` back, in order, into the original allocation (O(1),
                // no copy). The parts used to be received into their own `Bytes` and copied
                // again into the output.
                let total = usize::try_from(size).map_err(|_| {
                    DatasetError::from(format!("object size {size} does not fit in memory"))
                })?;
                let mut master = BytesMut::zeroed(total);
                let mut segments: VecDeque<BytesMut> = VecDeque::with_capacity(n_parts);
                for i in 0..n_parts {
                    let start = (i as u64) * part;
                    let len = (size - start).min(part) as usize;
                    segments.push_back(master.split_to(len));
                }

                let spawn_part = |i: usize, seg: BytesMut| {
                    let start = (i as u64) * part;
                    let bucket = self.bucket.clone();
                    let key = key.clone();
                    let token = cancel.clone();
                    tokio::spawn(async move {
                        tokio::select! {
                            _ = token.cancelled() => Err(anyhow::anyhow!(
                                "range GET cancelled: part {}", i
                            )),
                            r = get_object_range_into_segment_async(&bucket, &key, start, seg) => r,
                        }
                    })
                };

                let mut pending: FuturesOrdered<JoinHandle<anyhow::Result<BytesMut>>> =
                    FuturesOrdered::new();
                let mut next_part = 0usize;
                while next_part < n_parts && pending.len() < max_inflight {
                    match segments.pop_front() {
                        Some(seg) => pending.push_back(spawn_part(next_part, seg)),
                        None => break,
                    }
                    next_part += 1;
                }

                let mut assembled: Option<BytesMut> = None;
                let mut first_err: Option<DatasetError> = None;

                while let Some(join_res) = pending.next().await {
                    match join_res {
                        Ok(Ok(seg)) => {
                            if first_err.is_none() {
                                // Parts arrive in order (FuturesOrdered) and were split
                                // from one allocation in that order: `unsplit` is O(1).
                                match assembled.as_mut() {
                                    Some(acc) => acc.unsplit(seg),
                                    None => assembled = Some(seg),
                                }
                            }
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
                        if let Some(seg) = segments.pop_front() {
                            pending.push_back(spawn_part(next_part, seg));
                            next_part += 1;
                        }
                    }
                }

                if let Some(e) = first_err {
                    return Err(e);
                }

                let out = assembled.unwrap_or_default();
                if out.len() != total {
                    return Err(DatasetError::from(format!(
                        "range assembly length mismatch: assembled {} of {} bytes",
                        out.len(),
                        total
                    )));
                }
                Ok(out.freeze())
            }
        }
    }
}
