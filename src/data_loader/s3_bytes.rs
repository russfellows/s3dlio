// src/data_loader/s3_bytes.rs
// 

// src/data_loader/s3_bytes.rs
//
use crate::s3_utils::{
    parse_s3_uri,
    get_object_uri_async,
    get_object_range_uri_async, // NEW: implemented in s3_utils.rs next
    list_objects as list_objects_rs,
    stat_object_uri_async,
};
use crate::data_loader::{Dataset, DatasetError};
use crate::data_loader::options::{LoaderOptions, ReaderMode};
use async_trait::async_trait;
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
        let (bucket, prefix) = parse_s3_uri(uri)
            .map_err(|e| DatasetError::from(e.to_string()))?;

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
    pub fn keys(&self) -> &Vec<String> { &self.keys }
}

#[async_trait]
impl Dataset for S3BytesDataset {
    type Item = Vec<u8>;

    fn len(&self) -> Option<usize> { Some(self.keys.len()) }

    async fn get(&self, idx: usize) -> Result<Self::Item, DatasetError> {
        let key = self.keys.get(idx)
            .ok_or(DatasetError::IndexOutOfRange(idx))?;

        let uri = format!("s3://{}/{}", self.bucket, key);

        match self.reader_mode {
            ReaderMode::Sequential => {
                let bytes = get_object_uri_async(&uri)
                    .await
                    .map_err(DatasetError::from)?;
                // Convert Bytes to Vec<u8> for Dataset API compatibility
                Ok(bytes.to_vec())
            }
            ReaderMode::Range => {
                // HEAD to learn size
                let meta = stat_object_uri_async(&uri).await.map_err(DatasetError::from)?;
                let size = meta.size;
                if size == 0 { return Ok(Vec::new()); }
                let part = self.part_size.max(1) as u64;
                let n_parts = ((size + part - 1) / part) as usize;
                let max_inflight = self.max_inflight_parts.max(1);

                // Fetch parts concurrently, preserving order
                let mut out = Vec::with_capacity(size as usize);
                let mut chunks = stream::iter(0..n_parts)
                    .map(|i| {
                        let start = (i as u64) * part;
                        let len = (size - start).min(part);
                        let uri = uri.clone();
                        async move {
                            get_object_range_uri_async(&uri, start, Some(len)).await
                        }
                    })
                    .buffered(max_inflight);

                while let Some(res) = chunks.next().await {
                    let bytes = res.map_err(DatasetError::from)?;
                    // Extend Vec with bytes from Bytes
                    out.extend_from_slice(&bytes);
                }
                Ok(out)
            }
        }
    }
}


