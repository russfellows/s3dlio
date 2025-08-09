// src/data_loader/s3_bytes.rs
// 
use crate::s3_utils::{parse_s3_uri, get_object_uri_async, list_objects as list_objects_rs};
use crate::data_loader::{Dataset, DatasetError};
use async_trait::async_trait;

#[derive(Clone)]
pub struct S3BytesDataset {
    bucket: String,
    keys: Vec<String>,
}

impl S3BytesDataset {
    pub fn from_prefix(uri: &str) -> Result<Self, DatasetError> {
        let (bucket, prefix) = parse_s3_uri(uri)
            .map_err(|e| DatasetError::from(e.to_string()))?;

        let keys = list_objects_rs(&bucket, &prefix)
            .map_err(|e| DatasetError::from(e.to_string()))?;

        Ok(Self { bucket, keys })
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

        let bytes = get_object_uri_async(&uri)
            .await
            .map_err(DatasetError::from)?;
        /*
        let bytes = get_object_uri_async(&uri)
            .map_err(|e| DatasetError::from(e.to_string()))?;
        */

        Ok(bytes)
    }
}

