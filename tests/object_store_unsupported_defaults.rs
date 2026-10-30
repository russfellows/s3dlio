// tests/object_store_unsupported_defaults.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// `ObjectStore::get_into` and `ObjectStore::head` are provided methods whose default
// returns an `ObjectError` of kind `Unsupported` (v0.10.0 plan, section 4.5 A1, #264):
//
// * an existing implementer that overrides neither compiles unchanged and answers
//   `Unsupported`, never a silent fallback to a download plus a copy;
// * (the real file:// and direct:// stores used to keep the default; since #295 / #267 they
//   override both methods, see the last test);
// * the destination and the progress are left untouched;
// * the error names the provider the URI's scheme implies, and its text carries nothing from
//   the call but the operation name.

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use bytes::Bytes;
use futures::stream::{self, Stream};
use s3dlio::object_error::{object_error, ObjectErrorKind};
use s3dlio::object_store::{
    store_for_uri, ObjectMetadata, ObjectStore, ObjectWriter, ReceiveProgress,
};
use std::pin::Pin;

/// A store that implements only the required methods: everything else is the trait default.
struct MinimalStore;

#[async_trait]
impl ObjectStore for MinimalStore {
    async fn get(&self, _uri: &str) -> Result<Bytes> {
        Err(anyhow!("unused"))
    }
    async fn get_range(&self, _uri: &str, _offset: u64, _length: Option<u64>) -> Result<Bytes> {
        Err(anyhow!("unused"))
    }
    async fn put(&self, _uri: &str, _data: Bytes) -> Result<()> {
        Err(anyhow!("unused"))
    }
    async fn put_multipart(&self, _uri: &str, _data: Bytes, _part: Option<usize>) -> Result<()> {
        Err(anyhow!("unused"))
    }
    async fn list(&self, _prefix: &str, _recursive: bool) -> Result<Vec<String>> {
        Err(anyhow!("unused"))
    }
    fn list_stream<'a>(
        &'a self,
        _prefix: &'a str,
        _recursive: bool,
    ) -> Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
        Box::pin(stream::empty())
    }
    async fn stat(&self, _uri: &str) -> Result<ObjectMetadata> {
        Err(anyhow!("unused"))
    }
    async fn delete(&self, _uri: &str) -> Result<()> {
        Err(anyhow!("unused"))
    }
    async fn delete_batch(&self, _uris: &[String]) -> Result<()> {
        Err(anyhow!("unused"))
    }
    async fn delete_prefix(&self, _prefix: &str) -> Result<()> {
        Err(anyhow!("unused"))
    }
    async fn create_container(&self, _name: &str) -> Result<()> {
        Err(anyhow!("unused"))
    }
    async fn delete_container(&self, _name: &str) -> Result<()> {
        Err(anyhow!("unused"))
    }
    async fn get_writer(&self, _uri: &str) -> Result<Box<dyn ObjectWriter>> {
        Err(anyhow!("unused"))
    }
}

/// Both provided methods on `store` for `uri` answer `Unsupported` for `provider`.
async fn assert_unsupported(store: &dyn ObjectStore, uri: &str, provider: &str) {
    let progress = ReceiveProgress::new();
    let mut dst = [0xAB_u8; 16];
    let err = store
        .get_into(uri, &mut dst, &progress)
        .await
        .expect_err("get_into must be unsupported");
    let typed =
        object_error(&err).unwrap_or_else(|| panic!("{uri}: typed error expected: {err:#}"));
    assert_eq!(typed.kind, ObjectErrorKind::Unsupported, "{uri}");
    assert_eq!(typed.provider, provider, "{uri}");
    assert!(err.to_string().contains("get_into"), "{uri}: {err}");
    assert!(
        !err.to_string().contains(uri),
        "{uri}: the message carries no URI: {err}"
    );
    assert_eq!(dst, [0xAB; 16], "{uri}: the destination is untouched");
    assert_eq!(progress.filled(), 0, "{uri}: nothing was filled");
    assert!(progress.first_byte().is_none(), "{uri}");

    let err = store.head(uri).await.expect_err("head must be unsupported");
    let typed =
        object_error(&err).unwrap_or_else(|| panic!("{uri}: typed error expected: {err:#}"));
    assert_eq!(typed.kind, ObjectErrorKind::Unsupported, "{uri}");
    assert_eq!(typed.provider, provider, "{uri}");
    assert!(err.to_string().contains("head"), "{uri}: {err}");
}

/// A fake that overrides nothing new gets the default, with the provider taken from the URI.
#[tokio::test]
async fn minimal_store_defaults_to_unsupported_for_every_scheme() {
    let store = MinimalStore;
    for (uri, provider) in [
        ("s3://bucket/key", "s3"),
        ("az://account/container/blob", "az"),
        ("https://account.blob.core.windows.net/container/blob", "az"),
        ("gs://bucket/object", "gs"),
        ("gcs://bucket/object", "gs"),
        ("file:///data/object", "file"),
        ("direct:///data/object", "direct"),
        ("ftp://host/object", "unknown"),
        ("not a uri", "unknown"),
    ] {
        assert_unsupported(&store, uri, provider).await;
    }
}

/// The real file:// and direct:// stores override both methods (#295 / #267): they no longer
/// answer `Unsupported`; a 16-byte file read into a 16-byte destination succeeds.
#[tokio::test]
async fn file_and_direct_stores_no_longer_use_the_unsupported_default() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("object.bin");
    std::fs::write(&path, b"0123456789abcdef").unwrap();

    for scheme in ["file", "direct"] {
        let uri = format!("{scheme}://{}", path.display());
        let store = store_for_uri(&uri).unwrap_or_else(|e| panic!("{uri}: {e:#}"));
        let progress = ReceiveProgress::new();
        let mut dst = [0u8; 16];
        store
            .get_into(&uri, &mut dst, &progress)
            .await
            .unwrap_or_else(|e| panic!("{uri}: {e:#}"));
        assert_eq!(&dst, b"0123456789abcdef", "{uri}");
        assert_eq!(progress.filled(), 16, "{uri}");
        let head = store.head(&uri).await.unwrap().expect("present");
        assert_eq!(head.size, 16, "{uri}");
    }
}
