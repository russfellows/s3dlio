// src/azure_client.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use anyhow::{anyhow, bail, Result};
use bytes::Bytes;
use futures::{stream::FuturesUnordered, Stream, StreamExt};
use std::sync::Arc;
use tokio::sync::OnceCell;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::data_loader::parallel_fetch::DropCancel;

use azure_core::credentials::TokenCredential;
use azure_core::http::{Body, NoFormat, RequestContent, XmlFormat};
use azure_identity::DeveloperToolsCredential;

use azure_storage_blob::clients::{
    BlobClient, BlobClientOptions, BlobContainerClient, BlobContainerClientOptions,
    BlobServiceClient, BlobServiceClientOptions, BlockBlobClient,
};
use azure_storage_blob::models::{
    BlobClientDownloadOptions, BlobClientGetPropertiesOptions,
    BlobClientGetPropertiesResultHeaders, BlobContainerClientListBlobFlatSegmentOptions,
    BlockBlobClientCommitBlockListOptions, BlockBlobClientStageBlockOptions,
    BlockBlobClientUploadOptions, BlockList, BlockListType, BlockLookupList,
};
use tracing::{debug, warn};

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
    ///
    /// Supports custom endpoints via environment variables for local emulators and proxies:
    /// - `AZURE_STORAGE_ENDPOINT`: Primary endpoint URL (e.g., http://localhost:10000)
    /// - `AZURE_BLOB_ENDPOINT_URL`: Alternative endpoint URL
    ///
    /// When a custom endpoint is set, the account name is appended to form the full URL.
    /// Example: AZURE_STORAGE_ENDPOINT=http://localhost:10000 + account="devstoreaccount1"
    ///          → http://localhost:10000/devstoreaccount1
    pub fn with_default_credential(account: &str, container: &str) -> Result<Self> {
        // Check for custom endpoint (for Azurite or other emulators/proxies)
        if let Ok(endpoint) = std::env::var(crate::constants::ENV_AZURE_STORAGE_ENDPOINT)
            .or_else(|_| std::env::var(crate::constants::ENV_AZURE_BLOB_ENDPOINT_URL))
        {
            // Use custom endpoint (e.g., http://localhost:10000/account)
            let account_url = if endpoint.ends_with('/') {
                format!("{}{}", endpoint, account)
            } else {
                format!("{}/{}", endpoint, account)
            };
            tracing::info!("Using custom Azure endpoint: {}", account_url);
            return Self::with_default_credential_from_url(&account_url, container);
        }

        // Default: public Azure endpoint
        let account_url = Self::account_url_from_account(account);

        // Get or initialize the global credential (only authenticates once per process)
        let credential = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                AZURE_CREDENTIAL
                    .get_or_try_init(|| async {
                        let credential_arc = DeveloperToolsCredential::new(None)?;
                        let credential: Arc<dyn TokenCredential> = credential_arc;
                        Ok::<Arc<dyn TokenCredential>, anyhow::Error>(credential)
                    })
                    .await
            })
        })?;

        Ok(Self {
            account_url,
            container: container.to_string(),
            credential: Arc::clone(credential),
        })
    }

    /// Same, when a full endpoint URL (possibly emulator) is provided.
    pub fn with_default_credential_from_url(account_url: &str, container: &str) -> Result<Self> {
        // Get or initialize the global credential (only authenticates once per process)
        let credential = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {
                AZURE_CREDENTIAL
                    .get_or_try_init(|| async {
                        let credential_arc = DeveloperToolsCredential::new(None)?;
                        let credential: Arc<dyn TokenCredential> = credential_arc;
                        Ok::<Arc<dyn TokenCredential>, anyhow::Error>(credential)
                    })
                    .await
            })
        })?;

        Ok(Self {
            account_url: account_url.to_string(),
            container: container.to_string(),
            credential: Arc::clone(credential),
        })
    }

    /// Blob service (rarely needed directly).
    #[allow(dead_code)]
    fn service_client(&self) -> Result<BlobServiceClient> {
        BlobServiceClient::new(
            &self.account_url,
            Some(self.credential.clone()),
            Some(BlobServiceClientOptions::default()),
        )
        .map_err(|e| anyhow!(e))
    }

    fn container_client(&self) -> Result<BlobContainerClient> {
        BlobContainerClient::new(
            &self.account_url,
            &self.container,
            Some(self.credential.clone()),
            Some(BlobContainerClientOptions::default()),
        )
        .map_err(|e| anyhow!(e))
    }

    fn blob_client(&self, blob: &str) -> Result<BlobClient> {
        BlobClient::new(
            &self.account_url,
            &self.container,
            blob,
            Some(self.credential.clone()),
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
        debug!(
            "AzureBlob::put container='{}', key='{}', size={}, overwrite={}",
            self.container,
            key,
            body.len(),
            overwrite
        );
        let blob = self.blob_client(key)?;
        // Convert Bytes -> Body -> RequestContent<Bytes, NoFormat>
        let content_len = body.len() as u64;
        let data: RequestContent<Bytes, NoFormat> = Body::from(body).into();
        let _resp = blob
            .upload(
                data,
                overwrite,
                content_len,
                Some(BlockBlobClientUploadOptions::default()),
            )
            .await?;
        Ok(())
    }

    /// Range GET. If `end` is None → open-ended range.
    pub async fn get_range(&self, key: &str, start: u64, end: Option<u64>) -> Result<Bytes> {
        debug!(
            "AzureBlob::get_range container='{}', key='{}', start={}, end={:?}",
            self.container, key, start, end
        );
        let blob = self.blob_client(key)?;
        let mut opts = BlobClientDownloadOptions::default();
        let range = match end {
            Some(e) => format!("bytes={}-{}", start, e),
            None => format!("bytes={}-", start),
        };
        opts.range = Some(range);
        let resp = blob.download(Some(opts)).await?;
        let body = resp.into_body().collect().await?;
        debug!(
            "AzureBlob::get_range success: key='{}', {} bytes",
            key,
            body.len()
        );
        Ok(body)
    }

    /// Full GET (single buffer).
    pub async fn get(&self, key: &str) -> Result<Bytes> {
        debug!(
            "AzureBlob::get container='{}', key='{}'",
            self.container, key
        );
        let blob = self.blob_client(key)?;
        let resp = blob
            .download(Some(BlobClientDownloadOptions::default()))
            .await?;
        let body = resp.into_body().collect().await?;
        debug!(
            "AzureBlob::get success: key='{}', {} bytes",
            key,
            body.len()
        );
        Ok(body)
    }

    /// Stat: read size, etag, last-modified from typed response headers.
    pub async fn stat(&self, key: &str) -> Result<AzureBlobProperties> {
        debug!(
            "AzureBlob::stat container='{}', key='{}'",
            self.container, key
        );
        let blob = self.blob_client(key)?;
        let resp = blob
            .get_properties(Some(BlobClientGetPropertiesOptions::default()))
            .await?;
        let content_length = resp.content_length()?.unwrap_or(0);
        let etag = resp.etag()?.map(|e| e.to_string());
        let last_modified = resp.last_modified()?.map(|dt| dt.to_string());
        debug!(
            "AzureBlob::stat success: key='{}', content_length={}",
            key, content_length
        );
        Ok(AzureBlobProperties {
            content_length,
            etag,
            last_modified,
        })
    }

    /// Flat list with optional prefix.
    /// In SDK 0.7+, the Pager yields BlobItemInternal directly (not Response pages).
    pub async fn list(&self, prefix: Option<&str>) -> Result<Vec<String>> {
        debug!(
            "AzureBlob::list container='{}', prefix={:?}",
            self.container, prefix
        );
        let container = self.container_client()?;
        let mut opts = BlobContainerClientListBlobFlatSegmentOptions::default();
        if let Some(p) = prefix {
            if !p.is_empty() {
                opts.prefix = Some(p.to_string());
            }
        }
        let mut pager = container.list_blobs(Some(opts))?;
        let mut out = Vec::new();

        // In 0.7.0, pager yields Result<BlobItemInternal> directly
        while let Some(item_result) = pager.next().await {
            let item = item_result?;
            // `name` is Option<BlobName>, and BlobName.content is Option<String>
            if let Some(name) = item.name.and_then(|bn| bn.content) {
                out.push(name);
            }
        }
        debug!("AzureBlob::list success: {} objects", out.len());
        Ok(out)
    }

    /// List blobs as a stream, yielding results one by one.
    /// This is more memory-efficient for large listings and enables progress updates.
    pub fn list_stream<'a>(
        &'a self,
        prefix: Option<&'a str>,
    ) -> std::pin::Pin<Box<dyn Stream<Item = Result<String>> + Send + 'a>> {
        Box::pin(async_stream::stream! {
            let container = match self.container_client() {
                Ok(c) => c,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            let mut opts = BlobContainerClientListBlobFlatSegmentOptions::default();
            if let Some(p) = prefix {
                if !p.is_empty() { opts.prefix = Some(p.to_string()); }
            }

            let mut pager = match container.list_blobs(Some(opts)) {
                Ok(p) => p,
                Err(e) => {
                    yield Err(e.into());
                    return;
                }
            };

            // In 0.7.0, pager yields Result<BlobItemInternal> directly
            while let Some(item_result) = pager.next().await {
                let item = match item_result {
                    Ok(i) => i,
                    Err(e) => {
                        yield Err(e.into());
                        return;
                    }
                };

                if let Some(name) = item.name.and_then(|bn| bn.content) {
                    yield Ok(name);
                }
            }
        })
    }

    // src/azure_client.rs  (inside impl AzureBlob)
    /// Delete multiple blobs (simple loop; batch is possible later).
    pub async fn delete_objects(&self, blobs: &[String]) -> anyhow::Result<()> {
        debug!(
            "AzureBlob::delete_objects container='{}', count={}",
            self.container,
            blobs.len()
        );
        let container = self.container_client()?;
        for name in blobs {
            let b = container.blob_client(name);
            b.delete(None).await?;
        }
        Ok(())
    }

    // ----------------------------------------------------------------------
    // Multipart (block blob) helpers
    // ----------------------------------------------------------------------

    /// Stage a block (non-committal). `block_id` is raw bytes; SDK base64-encodes on wire.
    pub async fn stage_block(&self, key: &str, block_id: &[u8], chunk: Bytes) -> Result<()> {
        debug!(
            "AzureBlob::stage_block container='{}', key='{}', chunk_size={}",
            self.container,
            key,
            chunk.len()
        );
        let bb = self.block_blob_client(key)?;
        let content_len = chunk.len() as u64;
        let body: RequestContent<Bytes, NoFormat> = Body::from(chunk).into();
        let _resp = bb
            .stage_block(
                block_id,
                content_len,
                body,
                Some(BlockBlobClientStageBlockOptions::default()),
            )
            .await?;
        Ok(())
    }

    /// Commit previously staged block IDs (order matters).
    pub async fn commit_block_list(
        &self,
        key: &str,
        committed_block_ids: Vec<Vec<u8>>,
    ) -> Result<()> {
        debug!(
            "AzureBlob::commit_block_list container='{}', key='{}', blocks={}",
            self.container,
            key,
            committed_block_ids.len()
        );
        let bb = self.block_blob_client(key)?;
        let lookup = BlockLookupList {
            committed: None,
            latest: Some(committed_block_ids),
            uncommitted: None,
        };
        let body: RequestContent<BlockLookupList, XmlFormat> = lookup.try_into()?;
        let _resp = bb
            .commit_block_list(body, Some(BlockBlobClientCommitBlockListOptions::default()))
            .await?;
        Ok(())
    }

    /// Return committed block IDs (raw bytes that correspond to your passed IDs).
    pub async fn get_block_list_committed(&self, key: &str) -> Result<Vec<Vec<u8>>> {
        debug!(
            "AzureBlob::get_block_list_committed container='{}', key='{}'",
            self.container, key
        );
        let bb = self.block_blob_client(key)?;
        let resp = bb.get_block_list(BlockListType::Committed, None).await?;
        // In 0.7.0, Response::into_model() deserializes directly (not async)
        let list: BlockList = resp.into_model()?;
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
        debug!("AzureBlob::upload_multipart_stream container='{}', key='{}', part_size={}, max_in_flight={}", self.container, key, part_size, max_in_flight);

        // Task-level parallelism (issue #148 site 3.1e): each stage_block
        // is `tokio::spawn`'d so tokio can distribute request signing +
        // upload work across worker threads instead of funneling every
        // block through this task's polling budget. DropCancel + select!
        // on each spawn honors both early-drop and mid-stream errors:
        // when we detect an error (either from a backpressure-drained
        // task or a final-drain task), we `cancel` explicitly to let
        // remaining in-flight blocks bail quickly, then drain to
        // completion so no JoinHandle is dropped mid-flight.
        let cancel = CancellationToken::new();
        let _drop_cancel = DropCancel(cancel.clone());
        let mut in_flight: FuturesUnordered<JoinHandle<Result<()>>> = FuturesUnordered::new();
        let mut next_idx: u64 = 0;
        let mut committed_ids: Vec<Vec<u8>> = Vec::new();
        let mut first_err: Option<anyhow::Error> = None;

        while let Some(chunk) = stream.next().await {
            // Stop enqueuing new work once we've seen an error — but keep
            // draining in_flight below so no spawned task is left detached.
            if first_err.is_some() {
                break;
            }

            // Fixed-width raw bytes (SDK will base64 on the wire)
            let id_str = format!("{:016x}-{:08x}", next_idx, part_size as u32);
            let id_bytes = id_str.as_bytes().to_vec();

            // Maintain order of IDs to match blob composition.
            committed_ids.push(id_bytes.clone());

            // Backpressure — await the oldest completion when the pool
            // is full. Record (but do not bail on) any error observed so
            // we continue draining rather than leaking the rest.
            if in_flight.len() >= max_in_flight {
                if let Some(join_res) = in_flight.next().await {
                    absorb_stage_block_result(join_res, &mut first_err, &cancel);
                }
            }

            let this = self.clone_for_upload();
            let key_owned = key.to_string();
            let token = cancel.clone();
            in_flight.push(tokio::spawn(async move {
                tokio::select! {
                    _ = token.cancelled() => Err(anyhow!("stage_block cancelled")),
                    r = this.stage_block(&key_owned, &id_bytes, chunk) => r,
                }
            }));

            next_idx += 1;
        }

        // Drain remaining tasks — cancellation may have fired above; any
        // still-running spawn will bail through its select! arm quickly.
        while let Some(join_res) = in_flight.next().await {
            absorb_stage_block_result(join_res, &mut first_err, &cancel);
        }

        if let Some(e) = first_err {
            // audit #151 bug 1.5 (D4): previously returned bare Err with
            // no log at all — an operator watching logs had no way to
            // tell a multipart Azure upload failed mid-stream, or how
            // much was staged before it did. commit_block_list is never
            // called in this branch, so the already-staged blocks stay
            // uncommitted; Azure garbage-collects uncommitted blocks
            // automatically (no explicit delete API exists for "just
            // the uncommitted blocks" short of committing an empty/
            // unrelated list, which risks clobbering any pre-existing
            // committed data at this key — not attempted here).
            warn!(
                "{}",
                staged_blocks_failure_warning(&self.container, key, committed_ids.len(), &e)
            );
            return Err(e);
        }

        // Commit in produced order.
        self.commit_block_list(key, committed_ids).await
    }

    fn clone_for_upload(&self) -> Self {
        Self {
            account_url: self.account_url.clone(),
            container: self.container.clone(),
            credential: self.credential.clone(),
        }
    }
}

/// Build the operator-visibility warning logged when
/// `upload_multipart_stream` fails mid-stream (audit #151 bug 1.5 / D4).
/// Extracted as a pure function — independent of any Azure client or
/// network call — so its content (naming the container, blob key, and
/// how many blocks were staged before the failure) is directly
/// unit-testable without needing an Azure mock server, which this repo
/// does not have (unlike the S3 mock harness in tests/common/).
fn staged_blocks_failure_warning(
    container: &str,
    key: &str,
    staged_count: usize,
    err: &anyhow::Error,
) -> String {
    format!(
        "s3dlio Azure MPU: upload_multipart_stream failed for container='{container}' key='{key}' \
         after staging {staged_count} block(s) — those staged blocks were never committed and will \
         be garbage-collected automatically by Azure: {err}"
    )
}

/// Fold one spawned `stage_block` outcome into the first-error slot,
/// firing the shared cancellation token on the first observed error so
/// still-running blocks can bail quickly through their `select!` arm.
///
/// Extracted from `upload_multipart_stream` (issue #148 site 3.1e) so
/// both the backpressure step and the final drain reuse the same
/// bookkeeping.
fn absorb_stage_block_result(
    join_res: std::result::Result<Result<()>, tokio::task::JoinError>,
    first_err: &mut Option<anyhow::Error>,
    cancel: &CancellationToken,
) {
    match join_res {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            if first_err.is_none() {
                *first_err = Some(e);
                cancel.cancel();
            }
        }
        Err(join_err) if join_err.is_panic() => {
            if first_err.is_none() {
                *first_err = Some(anyhow::anyhow!("stage_block task panicked: {}", join_err));
                cancel.cancel();
            }
        }
        Err(_) => {
            // Task was cancelled (via the select! arm) — expected during
            // shutdown after an earlier error already fired cancel.
        }
    }
}

impl AzureBlob {
    // ----------------------------------------------------------------------
    // Container helpers (optional)
    // ----------------------------------------------------------------------

    /// Container creation not supported in newer Azure SDK versions.
    /// Use Azure CLI, portal, or SDK v0.7 for container management.
    #[allow(dead_code)]
    pub async fn create_container_if_missing(&self) -> Result<()> {
        bail!("Container creation not supported in Azure SDK v0.8+. Use Azure CLI: az storage container create")
    }

    /// Container deletion not supported in newer Azure SDK versions.
    /// Use Azure CLI, portal, or SDK v0.7 for container management.
    #[allow(dead_code)]
    pub async fn delete_container(&self) -> Result<()> {
        bail!("Container deletion not supported in Azure SDK v0.8+. Use Azure CLI: az storage container delete")
    }
}

// ============================================================================
// Helper Functions for Custom Endpoint URL Construction
// ============================================================================

/// Constructs the Azure account URL based on environment variables.
///
/// Returns the custom endpoint URL if `AZURE_STORAGE_ENDPOINT` or `AZURE_BLOB_ENDPOINT_URL`
/// is set, otherwise returns the standard Azure Blob endpoint.
///
/// This is extracted as a pure function for testability.
pub fn resolve_azure_account_url(account: &str) -> String {
    if let Ok(endpoint) = std::env::var(crate::constants::ENV_AZURE_STORAGE_ENDPOINT)
        .or_else(|_| std::env::var(crate::constants::ENV_AZURE_BLOB_ENDPOINT_URL))
    {
        // Use custom endpoint (e.g., http://localhost:10000/account)
        if endpoint.ends_with('/') {
            format!("{}{}", endpoint, account)
        } else {
            format!("{}/{}", endpoint, account)
        }
    } else {
        // Default: public Azure endpoint
        format!("https://{}.blob.core.windows.net", account)
    }
}

// ---------------------------------------------------------------------------
// Service-level helpers
// ---------------------------------------------------------------------------

/// List all containers in an Azure Blob Storage account.
///
/// Returns `Vec<(container_name, Option<last_modified_string>)>`.
///
/// Credentials follow the same chain as [`AzureBlob::with_default_credential`]:
/// environment variables, managed identity, developer tools, etc.
pub async fn list_account_containers(account: &str) -> Result<Vec<(String, Option<String>)>> {
    let account_url = resolve_azure_account_url(account);

    let credential = AZURE_CREDENTIAL
        .get_or_try_init(|| async {
            let cred = DeveloperToolsCredential::new(None)?;
            let cred: Arc<dyn azure_core::credentials::TokenCredential> = cred;
            Ok::<_, anyhow::Error>(cred)
        })
        .await?;

    let service_client = BlobServiceClient::new(
        &account_url,
        Some(Arc::clone(credential)),
        Some(BlobServiceClientOptions::default()),
    )
    .map_err(|e| anyhow!(e))?;

    let mut containers: Vec<(String, Option<String>)> = Vec::new();
    let mut pager = service_client
        .list_containers(None)
        .map_err(|e| anyhow!("Azure list_containers failed: {}", e))?
        .into_pages();

    while let Some(page) = pager.next().await {
        let current_page = page
            .map_err(|e| anyhow!("Azure list_containers page error: {}", e))?
            .into_model()
            .map_err(|e| anyhow!("Azure container model deserialisation: {}", e))?;

        for item in current_page.container_items {
            let name = item.name.unwrap_or_default();
            let date = item
                .properties
                .and_then(|p| p.last_modified)
                .map(|t| t.to_string());
            containers.push((name, date));
        }
    }

    Ok(containers)
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // Mutex to serialize tests that modify environment variables
    static ENV_MUTEX: Mutex<()> = Mutex::new(());

    // RED-then-GREEN regression tests for s3dlio issue #151 bug 1.5 (D4).
    //
    // Bug: upload_multipart_stream's failure branch returned a bare
    // `Err(e)` with no log at all when a stage_block failed mid-stream
    // -- an operator watching logs had no way to tell an Azure multipart
    // upload failed, or how many blocks were staged (and left
    // uncommitted, pending Azure's automatic GC) before it did.
    //
    // staged_blocks_failure_warning() is a pure function extracted from
    // the failure branch specifically so its content is testable without
    // an Azure mock server (this repo has one for S3, not Azure).

    #[test]
    fn staged_blocks_failure_warning_names_container_key_and_count() {
        let err = anyhow::anyhow!("stage_block cancelled");
        let msg = staged_blocks_failure_warning("my-container", "my-blob.bin", 7, &err);

        assert!(
            msg.contains("my-container"),
            "warning must name the container: {msg}"
        );
        assert!(
            msg.contains("my-blob.bin"),
            "warning must name the blob key: {msg}"
        );
        assert!(
            msg.contains('7'),
            "warning must include the staged block count: {msg}"
        );
        assert!(
            msg.contains("stage_block cancelled"),
            "warning must include the underlying error: {msg}"
        );
    }

    #[test]
    fn staged_blocks_failure_warning_handles_zero_staged_blocks() {
        // The very first block failed -- nothing was staged yet.
        let err = anyhow::anyhow!("connection refused");
        let msg = staged_blocks_failure_warning("c", "k", 0, &err);
        assert!(
            msg.contains('0'),
            "warning must report 0 staged blocks: {msg}"
        );
    }

    #[test]
    fn test_account_url_from_account() {
        let url = AzureBlob::account_url_from_account("mystorageaccount");
        assert_eq!(url, "https://mystorageaccount.blob.core.windows.net");
    }

    #[test]
    fn test_azurite_url() {
        let url = AzureBlob::azurite_url("127.0.0.1", 10000, "devstoreaccount1");
        assert_eq!(url, "http://127.0.0.1:10000/devstoreaccount1");
    }

    #[test]
    fn test_resolve_azure_account_url_default() {
        let _guard = ENV_MUTEX.lock().unwrap();

        // Clear any existing endpoint env vars
        std::env::remove_var(crate::constants::ENV_AZURE_STORAGE_ENDPOINT);
        std::env::remove_var(crate::constants::ENV_AZURE_BLOB_ENDPOINT_URL);

        let url = resolve_azure_account_url("mystorageaccount");
        assert_eq!(url, "https://mystorageaccount.blob.core.windows.net");
    }

    #[test]
    fn test_resolve_azure_account_url_with_primary_env_var() {
        let _guard = ENV_MUTEX.lock().unwrap();

        // Set primary env var
        std::env::set_var(
            crate::constants::ENV_AZURE_STORAGE_ENDPOINT,
            "http://localhost:10000",
        );
        std::env::remove_var(crate::constants::ENV_AZURE_BLOB_ENDPOINT_URL);

        let url = resolve_azure_account_url("devstoreaccount1");
        assert_eq!(url, "http://localhost:10000/devstoreaccount1");

        // Cleanup
        std::env::remove_var(crate::constants::ENV_AZURE_STORAGE_ENDPOINT);
    }

    #[test]
    fn test_resolve_azure_account_url_with_alternative_env_var() {
        let _guard = ENV_MUTEX.lock().unwrap();

        // Set alternative env var (primary not set)
        std::env::remove_var(crate::constants::ENV_AZURE_STORAGE_ENDPOINT);
        std::env::set_var(
            crate::constants::ENV_AZURE_BLOB_ENDPOINT_URL,
            "http://127.0.0.1:9001",
        );

        let url = resolve_azure_account_url("testaccount");
        assert_eq!(url, "http://127.0.0.1:9001/testaccount");

        // Cleanup
        std::env::remove_var(crate::constants::ENV_AZURE_BLOB_ENDPOINT_URL);
    }

    #[test]
    fn test_resolve_azure_account_url_with_trailing_slash() {
        let _guard = ENV_MUTEX.lock().unwrap();

        // Set env var with trailing slash
        std::env::set_var(
            crate::constants::ENV_AZURE_STORAGE_ENDPOINT,
            "http://localhost:10000/",
        );
        std::env::remove_var(crate::constants::ENV_AZURE_BLOB_ENDPOINT_URL);

        let url = resolve_azure_account_url("devstoreaccount1");
        assert_eq!(url, "http://localhost:10000/devstoreaccount1");

        // Cleanup
        std::env::remove_var(crate::constants::ENV_AZURE_STORAGE_ENDPOINT);
    }

    #[test]
    fn test_resolve_azure_account_url_primary_takes_precedence() {
        let _guard = ENV_MUTEX.lock().unwrap();

        // Set both env vars - primary should take precedence
        std::env::set_var(
            crate::constants::ENV_AZURE_STORAGE_ENDPOINT,
            "http://primary:10000",
        );
        std::env::set_var(
            crate::constants::ENV_AZURE_BLOB_ENDPOINT_URL,
            "http://alternative:9001",
        );

        let url = resolve_azure_account_url("testaccount");
        assert_eq!(url, "http://primary:10000/testaccount");

        // Cleanup
        std::env::remove_var(crate::constants::ENV_AZURE_STORAGE_ENDPOINT);
        std::env::remove_var(crate::constants::ENV_AZURE_BLOB_ENDPOINT_URL);
    }
}
