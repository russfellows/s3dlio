// src/gcs_client.rs
//
// Google Cloud Storage client implementation using gcloud-storage crate.
// Provides high-level operations for GCS buckets and objects with Application Default Credentials (ADC).

use anyhow::{anyhow, bail, Result};
use bytes::Bytes;
use gcloud_storage::client::{Client, ClientConfig};
use gcloud_storage::http::objects::download::Range;
use gcloud_storage::http::objects::get::GetObjectRequest;
use gcloud_storage::http::objects::upload::{Media, UploadObjectRequest, UploadType};
use gcloud_storage::http::objects::delete::DeleteObjectRequest;
use gcloud_storage::http::objects::list::ListObjectsRequest;
use gcloud_storage::http::buckets::insert::InsertBucketRequest;
use gcloud_storage::http::buckets::delete::DeleteBucketRequest;
use std::sync::Arc;
use tokio::sync::OnceCell;
use tracing::{debug, info};

// Global cached GCS client - initialized once and reused across all operations
static GCS_CLIENT: OnceCell<Arc<Client>> = OnceCell::const_new();

/// Minimal object metadata for GCS objects.
/// Maps to the provider-neutral ObjectMetadata type in object_store.rs.
#[derive(Debug, Clone)]
pub struct GcsObjectMetadata {
    pub size: u64,
    pub etag: Option<String>,
    pub updated: Option<String>,
    pub key: String,
}

/// High-level GCS client using Application Default Credentials (ADC).
/// 
/// Authentication follows the standard ADC chain:
/// 1. GOOGLE_APPLICATION_CREDENTIALS environment variable (service account JSON)
/// 2. GCE/GKE metadata server (automatic for Google Cloud workloads)
/// 3. gcloud CLI credentials (~/.config/gcloud/application_default_credentials.json)
pub struct GcsClient {
    client: Arc<Client>,
}

impl GcsClient {
    /// Create a new GCS client using Application Default Credentials.
    /// This now uses a cached global client for efficiency - authentication only happens once.
    /// 
    /// The credentials are automatically discovered from:
    /// - GOOGLE_APPLICATION_CREDENTIALS env var (loaded by dotenvy)
    /// - Metadata server (if running on GCP)
    /// - gcloud CLI credentials
    pub async fn new() -> Result<Self> {
        let client = GCS_CLIENT
            .get_or_try_init(|| async {
                debug!("Initializing GCS client with Application Default Credentials (first time only)");
                
                // Use with_auth() to automatically discover credentials
                let config = ClientConfig::default()
                    .with_auth()
                    .await
                    .map_err(|e| anyhow!("Failed to initialize GCS authentication: {}", e))?;
                
                let client = Client::new(config);
                
                info!("GCS client initialized successfully with Application Default Credentials (cached for reuse)");
                Ok::<Arc<Client>, anyhow::Error>(Arc::new(client))
            })
            .await?;
        
        Ok(Self {
            client: Arc::clone(client),
        })
    }

    /// Get entire object as bytes.
    pub async fn get_object(&self, bucket: &str, object: &str) -> Result<Bytes> {
        debug!("GCS GET: bucket={}, object={}", bucket, object);
        
        let data = self.client.download_object(
            &GetObjectRequest {
                bucket: bucket.to_string(),
                object: object.to_string(),
                ..Default::default()
            },
            &Range::default(),
        ).await
        .map_err(|e| anyhow!("GCS GET failed for gs://{}/{}: {}", bucket, object, e))?;
        
        debug!("GCS GET success: {} bytes", data.len());
        // Convert Vec<u8> to Bytes (cheap, just wraps in Arc)
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
            "GCS GET RANGE: bucket={}, object={}, offset={}, length={:?}",
            bucket, object, offset, length
        );

        let range = match length {
            Some(len) => Range(Some(offset), Some(offset + len - 1)),
            None => Range(Some(offset), None),
        };

        let data = self.client.download_object(
            &GetObjectRequest {
                bucket: bucket.to_string(),
                object: object.to_string(),
                ..Default::default()
            },
            &range,
        ).await
        .map_err(|e| anyhow!("GCS GET RANGE failed for gs://{}/{}: {}", bucket, object, e))?;

        debug!("GCS GET RANGE success: {} bytes", data.len());
        // Convert Vec<u8> to Bytes (cheap, just wraps in Arc)
        Ok(Bytes::from(data))
    }

    /// Upload object with simple upload (for small objects).
    pub async fn put_object(&self, bucket: &str, object: &str, data: &[u8]) -> Result<()> {
        debug!("GCS PUT: bucket={}, object={}, size={}", bucket, object, data.len());

        let upload_type = UploadType::Simple(Media::new(object.to_string()));
        
        self.client.upload_object(
            &UploadObjectRequest {
                bucket: bucket.to_string(),
                ..Default::default()
            },
            data.to_vec(),
            &upload_type,
        ).await
        .map_err(|e| anyhow!("GCS PUT failed for gs://{}/{}: {}", bucket, object, e))?;

        debug!("GCS PUT success: {} bytes", data.len());
        Ok(())
    }

    /// Upload large object using resumable upload.
    pub async fn put_object_multipart(
        &self,
        bucket: &str,
        object: &str,
        data: &[u8],
        _chunk_size: usize,
    ) -> Result<()> {
        debug!(
            "GCS PUT MULTIPART: bucket={}, object={}, size={}",
            bucket, object, data.len()
        );

        // For now, use simple upload - will implement resumable upload in GcsObjectWriter
        let upload_type = UploadType::Simple(Media::new(object.to_string()));
        
        self.client.upload_object(
            &UploadObjectRequest {
                bucket: bucket.to_string(),
                ..Default::default()
            },
            data.to_vec(),
            &upload_type,
        ).await
        .map_err(|e| anyhow!("GCS PUT MULTIPART failed for gs://{}/{}: {}", bucket, object, e))?;

        debug!("GCS PUT MULTIPART success: {} bytes", data.len());
        Ok(())
    }

    /// Get object metadata (size, etag, updated timestamp).
    pub async fn stat_object(&self, bucket: &str, object: &str) -> Result<GcsObjectMetadata> {
        debug!("GCS STAT: bucket={}, object={}", bucket, object);

        let obj = self.client.get_object(
            &GetObjectRequest {
                bucket: bucket.to_string(),
                object: object.to_string(),
                ..Default::default()
            },
        ).await
        .map_err(|e| anyhow!("GCS STAT failed for gs://{}/{}: {}", bucket, object, e))?;

        let result = GcsObjectMetadata {
            size: obj.size as u64,
            etag: Some(obj.etag),
            updated: obj.updated.map(|dt| format!("{:?}", dt)),
            key: obj.name,
        };

        debug!("GCS STAT success: {} bytes", result.size);
        Ok(result)
    }

    /// Delete a single object.
    pub async fn delete_object(&self, bucket: &str, object: &str) -> Result<()> {
        debug!("GCS DELETE: bucket={}, object={}", bucket, object);

        self.client.delete_object(
            &DeleteObjectRequest {
                bucket: bucket.to_string(),
                object: object.to_string(),
                ..Default::default()
            },
        ).await
        .map_err(|e| anyhow!("GCS DELETE failed for gs://{}/{}: {}", bucket, object, e))?;

        debug!("GCS DELETE success");
        Ok(())
    }

    /// Delete multiple objects (batch deletion with concurrency).
    pub async fn delete_objects(&self, bucket: &str, objects: Vec<String>) -> Result<()> {
        debug!("GCS DELETE BATCH: bucket={}, count={}", bucket, objects.len());

        use futures::stream::{self, StreamExt};

        let bucket = bucket.to_string();
        let client = self.client.clone();

        let results: Vec<Result<()>> = stream::iter(objects)
            .map(|object| {
                let bucket = bucket.clone();
                let client = client.clone();
                async move {
                    client.delete_object(&DeleteObjectRequest {
                        bucket: bucket.clone(),
                        object: object.clone(),
                        ..Default::default()
                    })
                    .await
                    .map_err(|e| anyhow!("Delete failed for {}: {}", object, e))
                }
            })
            .buffer_unordered(16) // Concurrent deletions
            .collect()
            .await;

        // Check for any errors
        let errors: Vec<_> = results.into_iter().filter_map(|r| r.err()).collect();
        if !errors.is_empty() {
            bail!("GCS batch delete had {} failures: {:?}", errors.len(), errors);
        }

        debug!("GCS DELETE BATCH success");
        Ok(())
    }

    /// List objects with optional prefix filtering.
    /// List objects in a GCS bucket with optional prefix filtering.
    /// 
    /// **Pagination**: This method automatically handles GCS pagination to retrieve
    /// all matching objects, not just the first 1000. GCS returns a maximum of 1000
    /// objects per API call by default. This implementation uses the `next_page_token`
    /// from each response to fetch subsequent pages until all objects are retrieved.
    /// 
    /// # Arguments
    /// * `bucket` - The GCS bucket name
    /// * `prefix` - Optional prefix filter for object names
    /// * `recursive` - If false, uses delimiter "/" for directory-like listing
    /// 
    /// # Returns
    /// Complete list of all matching object names (not URIs)
    pub async fn list_objects(
        &self,
        bucket: &str,
        prefix: Option<&str>,
        recursive: bool,
    ) -> Result<Vec<String>> {
        debug!(
            "GCS LIST: bucket={}, prefix={:?}, recursive={}",
            bucket, prefix, recursive
        );

        let mut all_objects = Vec::new();
        let mut page_token: Option<String> = None;

        // Pagination loop - GCS returns max 1000 objects per page by default
        loop {
            let mut request = ListObjectsRequest {
                bucket: bucket.to_string(),
                prefix: prefix.map(|s| s.to_string()),
                page_token: page_token.clone(),
                ..Default::default()
            };

            // For non-recursive, use delimiter to only list "folders"
            if !recursive {
                request.delimiter = Some("/".to_string());
            }

            debug!(
                "GCS LIST page request: page_token={:?}",
                page_token.as_ref().map(|t| format!("{}...", &t[..t.len().min(20)]))
            );

            let response = self.client.list_objects(&request)
                .await
                .map_err(|e| anyhow!("GCS LIST failed for bucket {}: {}", bucket, e))?;

            // Collect objects from this page
            let page_objects: Vec<String> = response
                .items
                .unwrap_or_default()
                .into_iter()
                .map(|obj| obj.name)  // Return just the object name, not full URI
                .collect();

            debug!("GCS LIST page received: {} objects", page_objects.len());
            all_objects.extend(page_objects);

            // Check for next page token
            if let Some(next_token) = response.next_page_token {
                page_token = Some(next_token);
                debug!("GCS LIST continuing to next page");
            } else {
                debug!("GCS LIST no more pages, breaking");
                break;
            }
        }

        debug!("GCS LIST success: {} total objects", all_objects.len());
        Ok(all_objects)
    }

    /// Create a new GCS bucket.
    pub async fn create_bucket(&self, bucket: &str, _location: Option<&str>) -> Result<()> {
        debug!("GCS CREATE BUCKET: bucket={}", bucket);

        let request = InsertBucketRequest {
            name: bucket.to_string(),
            ..Default::default()
        };

        // Note: InsertBucketRequest requires project parameter which needs to be configured
        // Location should be set via bucket metadata, not directly on the request
        self.client.insert_bucket(&request)
            .await
            .map_err(|e| anyhow!("GCS CREATE BUCKET failed for {}: {}", bucket, e))?;

        debug!("GCS CREATE BUCKET success");
        Ok(())
    }

    /// Delete a GCS bucket.
    pub async fn delete_bucket(&self, bucket: &str) -> Result<()> {
        debug!("GCS DELETE BUCKET: bucket={}", bucket);

        self.client.delete_bucket(
            &DeleteBucketRequest {
                bucket: bucket.to_string(),
                ..Default::default()
            },
        ).await
        .map_err(|e| anyhow!("GCS DELETE BUCKET failed for {}: {}", bucket, e))?;

        debug!("GCS DELETE BUCKET success");
        Ok(())
    }
}

/// Parse GCS URI into (bucket, object_path) components.
///
/// Supports both gs:// and gcs:// schemes.
///
/// # Arguments
/// * `uri` - GCS URI in format: gs://bucket/path/to/object or gcs://bucket/path/to/object
///
/// # Returns
/// * `Ok((bucket, object_path))` - Parsed components
/// * `Err(...)` - Invalid URI format
///
/// # Examples
/// ```
/// use s3dlio::gcs_client::parse_gcs_uri;
///
/// let (bucket, object) = parse_gcs_uri("gs://my-bucket/path/to/file.txt").unwrap();
/// assert_eq!(bucket, "my-bucket");
/// assert_eq!(object, "path/to/file.txt");
///
/// let (bucket, object) = parse_gcs_uri("gcs://my-bucket/path/to/file.txt").unwrap();
/// assert_eq!(bucket, "my-bucket");
/// assert_eq!(object, "path/to/file.txt");
/// ```
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

    let object_path = parts
        .next()
        .ok_or_else(|| anyhow!("Invalid GCS URI (missing object path): {}", uri))?
        .to_string();

    if bucket.is_empty() {
        bail!("Invalid GCS URI (empty bucket name): {}", uri);
    }

    if object_path.is_empty() {
        bail!("Invalid GCS URI (empty object path): {}", uri);
    }

    Ok((bucket, object_path))
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_gcs_uri_basic() {
        let (bucket, object) = parse_gcs_uri("gs://my-bucket/path/to/file.txt").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(object, "path/to/file.txt");
    }

    #[test]
    fn test_parse_gcs_uri_alternate_prefix() {
        let (bucket, object) = parse_gcs_uri("gcs://my-bucket/path/to/file.txt").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(object, "path/to/file.txt");
    }

    #[test]
    fn test_parse_gcs_uri_nested_path() {
        let (bucket, object) = parse_gcs_uri("gs://bucket/a/b/c/file.txt").unwrap();
        assert_eq!(bucket, "bucket");
        assert_eq!(object, "a/b/c/file.txt");
    }

    #[test]
    fn test_parse_gcs_uri_special_chars() {
        let (bucket, object) = parse_gcs_uri("gs://bucket/file.txt").unwrap();
        assert_eq!(bucket, "bucket");
        assert_eq!(object, "file.txt");
    }

    #[test]
    fn test_parse_gcs_uri_invalid_prefix() {
        let result = parse_gcs_uri("s3://bucket/file.txt");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("expected gs:// or gcs:// prefix"));
    }

    #[test]
    fn test_parse_gcs_uri_bucket_only() {
        let result = parse_gcs_uri("gs://bucket");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("missing object path"));
    }

    #[test]
    fn test_parse_gcs_uri_empty_bucket() {
        let result = parse_gcs_uri("gs:///path/to/file.txt");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("empty bucket name"));
    }
}
