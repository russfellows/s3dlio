use anyhow::Result;
use bytes::Bytes;

pub fn build_raw(data: &[u8]) -> Result<Bytes> {
    Ok(Bytes::copy_from_slice(data))
}

