// src/data_loader/s3_bytes.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use crate::data_loader::options::{LoaderOptions, ReaderMode};
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
use futures::stream::{self, StreamExt};

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

                // Pre-allocate the master output buffer once (issue #148,
                // audit §3.3b / Patch 3). `.buffered(N)` returns items in
                // stream order, so we can copy each range into its offset
                // as it completes and drop the source Bytes immediately —
                // keeping peak live memory to ~size instead of ~2 * size.
                let mut out = BytesMut::zeroed(size as usize);
                let mut write_offset: usize = 0;
                let mut chunks = stream::iter(0..n_parts)
                    .map(|i| {
                        let start = (i as u64) * part;
                        let len = (size - start).min(part);
                        let uri = uri.clone();
                        async move { get_object_range_uri_async(&uri, start, Some(len)).await }
                    })
                    .buffered(max_inflight);

                while let Some(res) = chunks.next().await {
                    let bytes = res.map_err(DatasetError::from)?;
                    let len = bytes.len();
                    let end = write_offset + len;
                    if end > out.len() {
                        return Err(DatasetError::from(format!(
                            "range assembly overflow: writing {}..{} but buffer is {} bytes",
                            write_offset,
                            end,
                            out.len()
                        )));
                    }
                    out[write_offset..end].copy_from_slice(&bytes);
                    write_offset = end;
                    drop(bytes);
                }

                if write_offset < out.len() {
                    out.truncate(write_offset);
                }
                Ok(out.freeze())
            }
        }
    }
}
