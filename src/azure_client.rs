// src/azure_client.rs

use anyhow::{anyhow, Result};
use bytes::Bytes;
use futures::{stream::FuturesUnordered, Stream, StreamExt};
use std::sync::Arc;
use tokio::sync::OnceCell;

use azure_core::credentials::TokenCredential;
use azure_core::http::{Body, NoFormat, RequestContent, XmlFormat};
use azure_identity::DefaultAzureCredential;

use azure_storage_blob::clients::{
    BlobClient, BlobClientOptions, BlobContainerClient, BlobContainerClientOptions, BlobServiceClient,
    BlobServiceClientOptions, BlockBlobClient,
};
use azure_storage_blob::models::{
    BlobClientDownloadOptions, BlobClientGetPropertiesOptions, BlobClientGetPropertiesResultHeaders,
    BlobContainerClientListBlobFlatSegmentOptions, BlockBlobClientCommitBlockListOptions,
    BlockBlobClientStageBlockOptions, BlockBlobClientUploadOptions, BlockList, BlockListType,
    BlockLookupList, ListBlobsFlatSegmentResponse,
};

// Global credential cache to avoid repeated authentication
static AZURE_CREDENTIAL: OnceCell<Arc<dyn TokenCredential>> = OnceCell::const_new();

/// Minimal properties surfaced by `stat`.
#[derive(Debug, Clone)]
pub struct AzureBlobProperties {
    pub content_length: u64,
    pub etag: Option<String>,
    pub last_modified: Option<String>,
}

/// High-level client bound to one container.
pub struct AzureBlob {
    account_url: String, // e.g. https://{account}.blob.core.windows.net
    pub container: String,
    credential: Arc<dyn TokenCredential>,
}

impl AzureBlob {
    /// Public Azure endpoint for an account name.
    fn account_url_from_account(account: &str) -> String {
        format!("https://{}.blob.core.windows.net", account)
    }

    /// Azurite helper, e.g. http://127.0.0.1:10000/{account}
    #[allow(dead_code)]
    pub fn azurite_url(host: &str, port: u16, account: &str) -> String {
        format!("http://{}:{}/{}", host, port, account)
    }

    /// Build with Entra ID (AAD) default chain (env, managed identity, etc).
    pub fn with_default_credential(account: &str, container: &str) -> Result<Self> {
        let account_url = Self::account_url_from_account(account);
        
        // Get or initialize the global credential (only authenticates once per process)
        let credential = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                AZURE_CREDENTIAL.get_or_try_init(|| async {
                    let credential_arc = DefaultAzureCredential::new()?;
                    let credential: Arc<dyn TokenCredential> = credential_arc;
                    Ok::<Arc<dyn TokenCredential>, anyhow::Error>(credential)
                }).await
            })
        })?;
        
        Ok(Self { 
            account_url, 
            container: container.to_string(), 
            credential: Arc::clone(credential) 
        })
    }

    /// Same, when a full endpoint URL (possibly emulator) is provided.
    pub fn with_default_credential_from_url(account_url: &str, container: &str) -> Result<Self> {
        // Get or initialize the global credential (only authenticates once per process)
        let credential = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                AZURE_CREDENTIAL.get_or_try_init(|| async {
                    let credential_arc = DefaultAzureCredential::new()?;
                    let credential: Arc<dyn TokenCredential> = credential_arc;
                    Ok::<Arc<dyn TokenCredential>, anyhow::Error>(credential)
                }).await
            })
        })?;
        
        Ok(Self { 
            account_url: account_url.to_string(), 
            container: container.to_string(), 
            credential: Arc::clone(credential) 
        })
    }

    /// Blob service (rarely needed directly).
    #[allow(dead_code)]
    fn service_client(&self) -> Result<BlobServiceClient> {
        BlobServiceClient::new(&self.account_url, self.credential.clone(), Some(BlobServiceClientOptions::default()))
            .map_err(|e| anyhow!(e))
    }

    fn container_client(&self) -> Result<BlobContainerClient> {
        BlobContainerClient::new(
            &self.account_url,
            self.container.clone(),
            self.credential.clone(),
            Some(BlobContainerClientOptions::default()),
        )
        .map_err(|e| anyhow!(e))
    }

    fn blob_client(&self, blob: &str) -> Result<BlobClient> {
        BlobClient::new(
            &self.account_url,
            self.container.clone(),
            blob.to_string(),
            self.credential.clone(),
            Some(BlobClientOptions::default()),
        )
        .map_err(|e| anyhow!(e))
    }

    fn block_blob_client(&self, blob: &str) -> Result<BlockBlobClient> {
        Ok(self.blob_client(blob)?.block_blob_client())
    }

    // ----------------------------------------------------------------------
    // Basic ops (single-shot upload, full/range download, stat, list)
    // ----------------------------------------------------------------------

    /// Simple upload (single request). For large bodies prefer multipart helpers below.
    pub async fn put(&self, key: &str, body: Bytes, overwrite: bool) -> Result<()> {
        let blob = self.blob_client(key)?;
        // Convert Bytes -> Body -> RequestContent<Bytes, NoFormat>
        let content_len = body.len() as u64;
        let data: RequestContent<Bytes, NoFormat> = Body::from(body).into();
        let _resp = blob
            .upload(data, overwrite, content_len, Some(BlockBlobClientUploadOptions::default()))
            .await?;
        Ok(())
    }

    /// Range GET. If `end` is None → open-ended range.
    pub async fn get_range(&self, key: &str, start: u64, end: Option<u64>) -> Result<Bytes> {
        let blob = self.blob_client(key)?;
        let mut opts = BlobClientDownloadOptions::default();
        let range = match end {
            Some(e) => format!("bytes={}-{}", start, e),
            None => format!("bytes={}-", start),
        };
        opts.range = Some(range);
        let resp = blob.download(Some(opts)).await?;
        let body = resp.into_raw_body().collect().await?;
        Ok(body)
    }

    /// Full GET (single buffer).
    pub async fn get(&self, key: &str) -> Result<Bytes> {
        let blob = self.blob_client(key)?;
        let resp = blob.download(Some(BlobClientDownloadOptions::default())).await?;
        let body = resp.into_raw_body().collect().await?;
        Ok(body)
    }

    /// Stat: read size, etag, last-modified from typed response headers.
    pub async fn stat(&self, key: &str) -> Result<AzureBlobProperties> {
        let blob = self.blob_client(key)?;
        let resp = blob.get_properties(Some(BlobClientGetPropertiesOptions::default())).await?;
        let content_length = resp.content_length()?.unwrap_or(0) as u64;
        let etag = resp.etag()?.map(|e| e.to_string());
        let last_modified = resp.last_modified()?.map(|dt| dt.to_string());
        Ok(AzureBlobProperties { content_length, etag, last_modified })
    }

    /// Flat list with optional prefix.
    pub async fn list(&self, prefix: Option<&str>) -> Result<Vec<String>> {
        let container = self.container_client()?;
        let mut opts = BlobContainerClientListBlobFlatSegmentOptions::default();
        if let Some(p) = prefix {
            if !p.is_empty() { opts.prefix = Some(p.to_string()); }
        }
        let mut pager = container.list_blobs(Some(opts))?;
        let mut out = Vec::new();

        while let Some(next) = pager.next().await {
            let resp = next?; // Result<Response<...>>
            let body: ListBlobsFlatSegmentResponse = resp.into_body().await?;
            // In 0.4.0, `blob_items` is a Vec<BlobItemInternal>
            for it in body.segment.blob_items {
                // `name` is Option<BlobName>, and BlobName.content is Option<String>
                if let Some(name) = it.name.and_then(|bn| bn.content) {
                    out.push(name);
                }
            }
        }
        Ok(out)
    }

    // src/azure_client.rs  (inside impl AzureBlob)
    /// Delete multiple blobs (simple loop; batch is possible later).
    pub async fn delete_objects(&self, blobs: &[String]) -> anyhow::Result<()> {
        let container = self.container_client()?;
        for name in blobs {
            let b = container.blob_client(name.clone());
            b.delete(None).await?;
        }
        Ok(())
    }


    // ----------------------------------------------------------------------
    // Multipart (block blob) helpers
    // ----------------------------------------------------------------------

    /// Stage a block (non-committal). `block_id` is raw bytes; SDK base64-encodes on wire.
    pub async fn stage_block(&self, key: &str, block_id: &[u8], chunk: Bytes) -> Result<()> {
        let bb = self.block_blob_client(key)?;
        let content_len = chunk.len() as u64;
        let body: RequestContent<Bytes, NoFormat> = Body::from(chunk).into();
        let _resp = bb
            .stage_block(block_id, content_len, body, Some(BlockBlobClientStageBlockOptions::default()))
            .await?;
        Ok(())
    }

    /// Commit previously staged block IDs (order matters).
    pub async fn commit_block_list(&self, key: &str, committed_block_ids: Vec<Vec<u8>>) -> Result<()> {
        let bb = self.block_blob_client(key)?;
        /*
         * Old
         *
        let lookup = BlockLookupList {
            committed: Some(committed_block_ids),
            latest: None,
            uncommitted: None,
        };
        */
        let lookup = BlockLookupList {
            committed: None,
            latest: Some(committed_block_ids),
            uncommitted: None,
        };
        let body: RequestContent<BlockLookupList, XmlFormat> = lookup.try_into()?; // TryInto is provided by SDK
        let _resp = bb
            .commit_block_list(body, Some(BlockBlobClientCommitBlockListOptions::default()))
            .await?;
        Ok(())
    }

    /// Return committed block IDs (raw bytes that correspond to your passed IDs).
    pub async fn get_block_list_committed(&self, key: &str) -> Result<Vec<Vec<u8>>> {
        let bb = self.block_blob_client(key)?;
        let resp = bb.get_block_list(BlockListType::Committed, None).await?;
        let list: BlockList = resp.into_body().await?;
        let mut out = Vec::new();
        if let Some(blocks) = list.committed_blocks {
            for b in blocks {
                if let Some(id) = b.name {
                    out.push(id);
                }
            }
        }
        Ok(out)
    }

    // ----------------------------------------------------------------------
    // High-throughput multipart upload (bounded concurrency), additive API
    // ----------------------------------------------------------------------

    /// High-throughput uploader: feeds chunks to `stage_block` with bounded concurrency.
    /// - `stream`: yields Bytes chunks (already sized to `part_size`, except final)
    /// - `part_size`: hint for generating stable, fixed-width block IDs
    /// - `max_in_flight`: bounds the number of concurrent `stage_block` calls
    pub async fn upload_multipart_stream<S>(
        &self,
        key: &str,
        mut stream: S,
        part_size: usize,
        max_in_flight: usize,
    ) -> Result<()>
    where
        S: Stream<Item = Bytes> + Unpin + Send + 'static,
    {
        let mut in_flight: FuturesUnordered<_> = FuturesUnordered::new();
        let mut next_idx: u64 = 0;
        let mut committed_ids: Vec<Vec<u8>> = Vec::new();

        while let Some(chunk) = stream.next().await {

            // Fixed-width raw bytes (SDK will base64 on the wire)
            let id_str = format!("{:016x}-{:08x}", next_idx, part_size as u32);
            let id_bytes = id_str.as_bytes().to_vec();

            // Maintain order of IDs to match blob composition.
            committed_ids.push(id_bytes.clone());

            // Backpressure
            if in_flight.len() >= max_in_flight {
                in_flight
                    .next()
                    .await
                    .transpose()?
                    .ok_or_else(|| anyhow!("stage_block task ended unexpectedly"))?;
            }

            let this = self.clone_for_upload();
            let key_owned = key.to_string();
            in_flight.push(async move { this.stage_block(&key_owned, &id_bytes, chunk).await });

            next_idx += 1;
        }

        // Drain remaining tasks
        while let Some(res) = in_flight.next().await {
            res?;
        }

        // Commit in produced order
        self.commit_block_list(key, committed_ids).await
    }

    fn clone_for_upload(&self) -> Self {
        Self {
            account_url: self.account_url.clone(),
            container: self.container.clone(),
            credential: self.credential.clone(),
        }
    }

    // ----------------------------------------------------------------------
    // Container helpers (optional)
    // ----------------------------------------------------------------------

    #[allow(dead_code)]
    pub async fn create_container_if_missing(&self) -> Result<()> {
        let c = self.container_client()?;
        let _ = c.create_container(None).await?;
        Ok(())
    }

    #[allow(dead_code)]
    pub async fn delete_container(&self) -> Result<()> {
        let c = self.container_client()?;
        let _ = c.delete_container(None).await?;
        Ok(())
    }

}



