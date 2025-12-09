// src/google_gcs_client.rs
//
// Google Cloud Storage client implementation using google-cloud-storage (official Google crate).
// Provides high-level operations for GCS buckets and objects with Application Default Credentials (ADC).
//
// This is the "gcs-official" backend - experimental, use gcs-community for production.

use anyhow::{anyhow, bail, Result};
use bytes::Bytes;
use google_cloud_storage::client::{Storage, StorageControl};
use google_cloud_storage::model_ext::ReadRange;
use google_cloud_gax::paginator::ItemPaginator;
use std::sync::Arc;
use tokio::sync::OnceCell;
use tracing::{debug, info};

// Global cached GCS clients - initialized once and reused across all operations
static GCS_STORAGE: OnceCell<Arc<Storage>> = OnceCell::const_new();
static GCS_CONTROL: OnceCell<Arc<StorageControl>> = OnceCell::const_new();

/// Minimal object metadata for GCS objects.
/// Maps to the provider-neutral ObjectMetadata type in object_store.rs.
/// 
/// NOTE: This struct MUST match the one in gcs_client.rs exactly!
#[derive(Debug, Clone)]
pub struct GcsObjectMetadata {
    pub size: u64,
    pub etag: Option<String>,
    pub updated: Option<String>,
    pub key: String,
}

/// High-level GCS client using official Google google-cloud-storage crate.
/// 
/// Authentication follows the standard ADC chain:
/// 1. GOOGLE_APPLICATION_CREDENTIALS environment variable (service account JSON)
/// 2. GCE/GKE metadata server (automatic for Google Cloud workloads)
/// 3. gcloud CLI credentials (~/.config/gcloud/application_default_credentials.json)
pub struct GcsClient {
    storage: Arc<Storage>,  // For read/write operations
    control: Arc<StorageControl>,  // For metadata/list/delete operations
}

impl GcsClient {
    /// Create a new GCS client using Application Default Credentials.
    /// This uses cached global clients for efficiency - authentication only happens once.
    /// 
    /// The credentials are automatically discovered from:
    /// - GOOGLE_APPLICATION_CREDENTIALS env var (loaded by dotenvy)
    /// - Metadata server (if running on GCP)
    /// - gcloud CLI credentials
    pub async fn new() -> Result<Self> {
        let storage = GCS_STORAGE
            .get_or_try_init(|| async {
                debug!("Initializing GCS Storage client (first time only)");
                
                let client = Storage::builder()
                    .build()
                    .await
                    .map_err(|e| anyhow!("Failed to initialize GCS Storage client: {}", e))?;
                
                info!("GCS Storage client initialized successfully");
                Ok::<Arc<Storage>, anyhow::Error>(Arc::new(client))
            })
            .await?;

        let control = GCS_CONTROL
            .get_or_try_init(|| async {
                debug!("Initializing GCS StorageControl client (first time only)");
                
                let client = StorageControl::builder()
                    .build()
                    .await
                    .map_err(|e| anyhow!("Failed to initialize GCS StorageControl client: {}", e))?;
                
                info!("GCS StorageControl client initialized successfully");
                Ok::<Arc<StorageControl>, anyhow::Error>(Arc::new(client))
            })
            .await?;
        
        Ok(Self {
            storage: Arc::clone(storage),
            control: Arc::clone(control),
        })
    }

    /// Get entire object as bytes.
    pub async fn get_object(&self, bucket: &str, object: &str) -> Result<Bytes> {
        debug!("GCS GET (official): bucket={}, object={}", bucket, object);
        
        // Format bucket name as required by official client
        let bucket_name = format!("projects/_/buckets/{}", bucket);
        
        let mut response = self.storage
            .read_object(&bucket_name, object)
            .send()
            .await
            .map_err(|e| anyhow!("GCS GET failed for gs://{}/{}: {}", bucket, object, e))?;
        
        // Collect all chunks into a single buffer
        let mut data = Vec::new();
        while let Some(chunk) = response.next().await.transpose()
            .map_err(|e| anyhow!("GCS GET stream error for gs://{}/{}: {}", bucket, object, e))? 
        {
            data.extend_from_slice(&chunk);
        }
        
        debug!("GCS GET success: {} bytes", data.len());
        Ok(Bytes::from(data))
    }

    /// Get a byte range from an object.
    pub async fn get_object_range(
        &self,
        bucket: &str,
        object: &str,
        offset: u64,
        length: Option<u64>,
    ) -> Result<Bytes> {
        debug!(
            "GCS GET RANGE (official): bucket={}, object={}, offset={}, length={:?}",
            bucket, object, offset, length
        );

        let bucket_name = format!("projects/_/buckets/{}", bucket);
        
        // Use ReadRange to specify byte range
        let read_range = match length {
            Some(len) => ReadRange::segment(offset, len),
            None => ReadRange::offset(offset),
        };

        let mut response = self.storage
            .read_object(&bucket_name, object)
            .set_read_range(read_range)
            .send()
            .await
            .map_err(|e| anyhow!("GCS GET RANGE failed for gs://{}/{}: {}", bucket, object, e))?;

        // Collect all chunks
        let mut data = Vec::new();
        while let Some(chunk) = response.next().await.transpose()
            .map_err(|e| anyhow!("GCS GET RANGE stream error for gs://{}/{}: {}", bucket, object, e))? 
        {
            data.extend_from_slice(&chunk);
        }

        debug!("GCS GET RANGE success: {} bytes", data.len());
        Ok(Bytes::from(data))
    }

    /// Upload an object.
    pub async fn put_object(&self, bucket: &str, object: &str, data: &[u8]) -> Result<()> {
        debug!(
            "GCS PUT (official): bucket={}, object={}, size={}",
            bucket,
            object,
            data.len()
        );

        let bucket_name = format!("projects/_/buckets/{}", bucket);

        // Convert slice to Bytes for upload
        let bytes = Bytes::copy_from_slice(data);

        self.storage
            .write_object(&bucket_name, object, bytes)
            .send_unbuffered()
            .await
            .map_err(|e| anyhow!("GCS PUT failed for gs://{}/{}: {}", bucket, object, e))?;

        debug!("GCS PUT success");
        Ok(())
    }

    /// Delete an object.
    pub async fn delete_object(&self, bucket: &str, object: &str) -> Result<()> {
        debug!("GCS DELETE (official): bucket={}, object={}", bucket, object);

        // StorageControl requires projects/_/buckets/{bucket} format
        let bucket_name = format!("projects/_/buckets/{}", bucket);

        self.control
            .delete_object()
            .set_bucket(bucket_name)
            .set_object(object.to_string())
            .send()
            .await
            .map_err(|e| anyhow!("GCS DELETE failed for gs://{}/{}: {}", bucket, object, e))?;

        debug!("GCS DELETE success");
        Ok(())
    }

    /// Get object metadata without downloading the content.
    pub async fn get_object_metadata(&self, bucket: &str, object: &str) -> Result<GcsObjectMetadata> {
        debug!("GCS GET METADATA (official): bucket={}, object={}", bucket, object);

        // StorageControl requires projects/_/buckets/{bucket} format
        let bucket_name = format!("projects/_/buckets/{}", bucket);

        let obj = self.control
            .get_object()
            .set_bucket(bucket_name)
            .set_object(object.to_string())
            .send()
            .await
            .map_err(|e| anyhow!("GCS GET METADATA failed for gs://{}/{}: {}", bucket, object, e))?;

        let metadata = GcsObjectMetadata {
            size: obj.size as u64,
            etag: Some(obj.etag.clone()),
            updated: obj.update_time.map(|t| {
                // Timestamp is google_cloud_wkt::timestamp::Timestamp
                // Convert to RFC3339 format
                format!("{:?}", t) // Use debug formatting as fallback
            }),
            key: object.to_string(),
        };

        debug!("GCS GET METADATA success: {} bytes", metadata.size);
        Ok(metadata)
    }

    /// Alias for get_object_metadata (for compatibility with gcs_client.rs).
    pub async fn stat_object(&self, bucket: &str, object: &str) -> Result<GcsObjectMetadata> {
        self.get_object_metadata(bucket, object).await
    }

    /// List objects in a bucket with optional prefix.
    /// 
    /// When recursive=false, returns both files and subdirectory prefixes (ending with "/").
    /// This matches S3 behavior when using delimiter="/".
    pub async fn list_objects(
        &self,
        bucket: &str,
        prefix: Option<&str>,
        recursive: bool,
    ) -> Result<Vec<String>> {
        // Normalize prefix for non-recursive listings
        // GCS requires trailing "/" for delimiter behavior to work correctly
        let normalized_prefix = prefix.map(|p| {
            if !recursive && !p.is_empty() && !p.ends_with('/') {
                format!("{}/", p)
            } else {
                p.to_string()
            }
        });

        debug!(
            "GCS LIST (official): bucket={}, prefix={:?}, recursive={}, normalized_prefix={:?}",
            bucket, prefix, recursive, normalized_prefix
        );

        // StorageControl requires projects/_/buckets/{bucket} format
        let bucket_name = format!("projects/_/buckets/{}", bucket);

        let mut builder = self.control
            .list_objects()
            .set_parent(bucket_name);

        if let Some(p) = normalized_prefix.as_ref() {
            builder = builder.set_prefix(p.clone());
        }

        // Set delimiter for non-recursive listings
        // This makes GCS return common prefixes (subdirectories) separately
        if !recursive {
            builder = builder.set_delimiter("/".to_string());
        }

        let mut results = Vec::new();

        if !recursive {
            // For non-recursive, we need access to both items AND prefixes
            // Use by_page() to get the full response structure with prefixes
            use google_cloud_gax::paginator::Paginator;
            let mut pages_iter = builder.by_page();
            
            while let Some(result) = pages_iter.next().await {
                let page = result.map_err(|e| anyhow!("GCS LIST error for gs://{}: {}", bucket, e))?;
                
                // Collect object names (files)
                results.extend(page.objects.into_iter().map(|obj| obj.name));
                
                // Collect prefixes (subdirectories) - these end with "/"
                results.extend(page.prefixes);
            }
        } else {
            // For recursive, by_item() is more efficient (no need for prefixes)
            let mut objects_iter = builder.by_item();
            
            while let Some(object) = objects_iter.next().await.transpose()
                .map_err(|e| anyhow!("GCS LIST error for gs://{}: {}", bucket, e))? 
            {
                results.push(object.name.clone());
            }
        }

        debug!("GCS LIST success: {} objects", results.len());
        Ok(results)
    }

    /// Delete a bucket (must be empty).
    pub async fn delete_bucket(&self, bucket: &str) -> Result<()> {
        debug!("GCS DELETE BUCKET (official): bucket={}", bucket);

        let bucket_name = format!("projects/_/buckets/{}", bucket);

        self.control
            .delete_bucket()
            .set_name(bucket_name)
            .send()
            .await
            .map_err(|e| anyhow!("GCS DELETE BUCKET failed for {}: {}", bucket, e))?;

        debug!("GCS DELETE BUCKET success");
        Ok(())
    }

    /// Create a bucket.
    pub async fn create_bucket(&self, bucket: &str, _project_id: Option<&str>) -> Result<()> {
        debug!("GCS CREATE BUCKET (official): bucket={}", bucket);

        self.control
            .create_bucket()
            .set_parent("projects/_".to_string())
            .set_bucket_id(bucket.to_string())
            .set_bucket(google_cloud_storage::model::Bucket::new())
            .send()
            .await
            .map_err(|e| anyhow!("GCS CREATE BUCKET failed for {}: {}", bucket, e))?;

        debug!("GCS CREATE BUCKET success");
        Ok(())
    }

    /// Delete multiple objects (batch delete).
    pub async fn delete_objects(&self, bucket: &str, objects: Vec<String>) -> Result<()> {
        debug!("GCS DELETE OBJECTS (official): bucket={}, count={}", bucket, objects.len());

        // Note: google-cloud-storage doesn't have a native batch delete API,
        // so we delete objects one by one
        for object in &objects {
            if let Err(e) = self.delete_object(bucket, object).await {
                tracing::warn!("Failed to delete gs://{}/{}: {}", bucket, object, e);
            }
        }

        debug!("GCS DELETE OBJECTS complete");
        Ok(())
    }

    /// Multipart upload.
    pub async fn put_object_multipart(
        &self,
        bucket: &str,
        object: &str,
        data: &[u8],
        _chunk_size: usize,
    ) -> Result<()> {
        // For now, just use regular put_object
        // Google Cloud Storage handles chunking automatically for large objects
        self.put_object(bucket, object, data).await
    }
}


/// Parse a GCS URI (gs://bucket/path/to/object) into (bucket, object_path).
/// 
/// Bucket-only URIs are also supported (for prefix listings):
/// - gs://bucket/ → ("bucket", "")
/// - gs://bucket  → ("bucket", "") - requires trailing slash for proper parsing
pub fn parse_gcs_uri(uri: &str) -> Result<(String, String)> {
    // Strip gs:// or gcs:// prefix
    let path = uri
        .strip_prefix("gs://")
        .or_else(|| uri.strip_prefix("gcs://"))
        .ok_or_else(|| anyhow!("Invalid GCS URI (expected gs:// or gcs:// prefix): {}", uri))?;

    // Split into bucket and object path
    let mut parts = path.splitn(2, '/');
    let bucket = parts
        .next()
        .ok_or_else(|| anyhow!("Invalid GCS URI (missing bucket): {}", uri))?
        .to_string();

    // Object path is optional (empty for bucket-only URIs like gs://bucket/ for listings)
    let object_path = parts.next().unwrap_or("").to_string();

    if bucket.is_empty() {
        bail!("Invalid GCS URI (empty bucket name): {}", uri);
    }

    Ok((bucket, object_path))
}

