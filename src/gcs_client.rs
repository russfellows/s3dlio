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
    /// 
    /// Supports custom endpoints via environment variables for local emulators and proxies:
    /// - `GCS_ENDPOINT_URL`: Full endpoint URL (e.g., http://localhost:4443)
    /// - `STORAGE_EMULATOR_HOST`: GCS emulator convention (host:port, http:// prepended if missing)
    /// 
    /// When a custom endpoint is set, anonymous authentication is used (typical for emulators).
    pub async fn new() -> Result<Self> {
        let client = GCS_CLIENT
            .get_or_try_init(|| async {
                // Check for custom endpoint (for fake-gcs-server or other emulators/proxies)
                let custom_endpoint = std::env::var(crate::constants::ENV_GCS_ENDPOINT_URL).ok()
                    .or_else(|| {
                        // GCS emulator convention: STORAGE_EMULATOR_HOST=host:port
                        std::env::var(crate::constants::ENV_STORAGE_EMULATOR_HOST).ok()
                            .map(|host| {
                                if host.starts_with("http://") || host.starts_with("https://") {
                                    host
                                } else {
                                    format!("http://{}", host)
                                }
                            })
                    });
                
                let config = if let Some(endpoint) = custom_endpoint {
                    info!("Using custom GCS endpoint: {}", endpoint);
                    // Local emulators typically don't need auth
                    // ClientConfig has a public storage_endpoint field
                    ClientConfig {
                        storage_endpoint: endpoint,
                        ..ClientConfig::default()
                    }.anonymous()
                } else {
                    debug!("Initializing GCS client with Application Default Credentials (first time only)");
                    // Use with_auth() to automatically discover credentials
                    ClientConfig::default()
                        .with_auth()
                        .await
                        .map_err(|e| anyhow!("Failed to initialize GCS authentication: {}", e))?
                };
                
                let client = Client::new(config);
                
                info!("GCS client initialized successfully (cached for reuse)");
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

        let mut results = Vec::new();
        let mut page_token: Option<String> = None;

        // Normalize folder-like prefix for non-recursive listings
        // GCS API works best with trailing "/" when using delimiter
        let normalized_prefix = prefix.map(|p| {
            if !recursive && !p.is_empty() && !p.ends_with('/') {
                format!("{}/", p)
            } else {
                p.to_string()
            }
        });

        // Pagination loop - GCS returns max 1000 objects per page by default
        loop {
            let mut request = ListObjectsRequest {
                bucket: bucket.to_string(),
                prefix: normalized_prefix.clone(),
                page_token: page_token.clone(),
                ..Default::default()
            };

            // For non-recursive, use delimiter to split results into files (items) and subdirs (prefixes)
            if !recursive {
                request.delimiter = Some("/".to_string());
            }

            debug!(
                "GCS LIST page request: normalized_prefix={:?}, page_token={:?}",
                normalized_prefix,
                page_token.as_ref().map(|t| format!("{}...", &t[..t.len().min(20)]))
            );

            let response = self.client.list_objects(&request)
                .await
                .map_err(|e| anyhow!("GCS LIST failed for bucket {}: {}", bucket, e))?;

            // Collect files at this level (from items[])
            if let Some(items) = response.items {
                let item_names: Vec<String> = items
                    .into_iter()
                    .map(|obj| obj.name)
                    .collect();
                debug!("GCS LIST page received: {} objects", item_names.len());
                results.extend(item_names);
            }

            // For non-recursive, also collect subdirectory prefixes
            // These represent "folders" and always end with "/"
            if !recursive {
                if let Some(prefixes) = response.prefixes {
                    debug!("GCS LIST page received: {} prefixes (subdirectories)", prefixes.len());
                    results.extend(prefixes);
                }
            }

            // Check for next page token
            if let Some(next_token) = response.next_page_token {
                page_token = Some(next_token);
                debug!("GCS LIST continuing to next page");
            } else {
                debug!("GCS LIST no more pages, breaking");
                break;
            }
        }

        debug!("GCS LIST success: {} total results (files + prefixes)", results.len());
        Ok(results)
    }

    /// List objects in a GCS bucket as a stream, yielding results page by page.
    /// This is more memory-efficient for large listings and enables progress updates.
    /// 
    /// # Arguments
    /// * `bucket` - The GCS bucket name
    /// * `prefix` - Optional prefix filter for object names
    /// * `recursive` - If false, uses delimiter "/" for directory-like listing
    /// 
    /// # Returns
    /// A stream of object names (not URIs)
    pub fn list_objects_stream<'a>(
        &'a self,
        bucket: &'a str,
        prefix: Option<&'a str>,
        recursive: bool,
    ) -> std::pin::Pin<Box<dyn futures::Stream<Item = Result<String>> + Send + 'a>> {
        Box::pin(async_stream::stream! {
            debug!(
                "GCS LIST STREAM: bucket={}, prefix={:?}, recursive={}",
                bucket, prefix, recursive
            );

            let mut page_token: Option<String> = None;

            // Normalize folder-like prefix for non-recursive listings
            let normalized_prefix = prefix.map(|p| {
                if !recursive && !p.is_empty() && !p.ends_with('/') {
                    format!("{}/", p)
                } else {
                    p.to_string()
                }
            });

            // Pagination loop - GCS returns max 1000 objects per page by default
            loop {
                let mut request = ListObjectsRequest {
                    bucket: bucket.to_string(),
                    prefix: normalized_prefix.clone(),
                    page_token: page_token.clone(),
                    ..Default::default()
                };

                if !recursive {
                    request.delimiter = Some("/".to_string());
                }

                let response = match self.client.list_objects(&request).await {
                    Ok(r) => r,
                    Err(e) => {
                        yield Err(anyhow!("GCS LIST failed for bucket {}: {}", bucket, e));
                        return;
                    }
                };

                // Yield files at this level (from items[])
                if let Some(items) = response.items {
                    for obj in items {
                        yield Ok(obj.name);
                    }
                }

                // For non-recursive, also yield subdirectory prefixes
                if !recursive {
                    if let Some(prefixes) = response.prefixes {
                        for prefix in prefixes {
                            yield Ok(prefix);
                        }
                    }
                }

                // Check for next page token
                if let Some(next_token) = response.next_page_token {
                    page_token = Some(next_token);
                } else {
                    break;
                }
            }

            debug!("GCS LIST STREAM complete");
        })
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
/// 
/// // Bucket-only URIs are also supported (for prefix listings)
/// let (bucket, object) = parse_gcs_uri("gs://my-bucket/").unwrap();
/// assert_eq!(bucket, "my-bucket");
/// assert_eq!(object, "");
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

    // Object path is optional (empty for bucket-only URIs like gs://bucket/ for listings)
    let object_path = parts.next().unwrap_or("").to_string();

    if bucket.is_empty() {
        bail!("Invalid GCS URI (empty bucket name): {}", uri);
    }

    Ok((bucket, object_path))
}

// ============================================================================
// Helper Functions for Custom Endpoint URL Construction
// ============================================================================

/// Resolves the GCS storage endpoint based on environment variables.
/// 
/// Returns a custom endpoint URL if `GCS_ENDPOINT_URL` or `STORAGE_EMULATOR_HOST`
/// is set, otherwise returns `None` (indicating default GCS endpoint should be used).
/// 
/// This is extracted as a pure function for testability.
pub fn resolve_gcs_endpoint() -> Option<String> {
    std::env::var(crate::constants::ENV_GCS_ENDPOINT_URL).ok()
        .or_else(|| {
            // GCS emulator convention: STORAGE_EMULATOR_HOST=host:port
            std::env::var(crate::constants::ENV_STORAGE_EMULATOR_HOST).ok()
                .map(|host| {
                    if host.starts_with("http://") || host.starts_with("https://") {
                        host
                    } else {
                        format!("http://{}", host)
                    }
                })
        })
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

    // ========================================================================
    // Custom Endpoint Tests
    // ========================================================================

    #[test]
    fn test_resolve_gcs_endpoint_default() {
        let _guard = ENV_MUTEX.lock().unwrap();
        
        // Clear any existing endpoint env vars
        std::env::remove_var(crate::constants::ENV_GCS_ENDPOINT_URL);
        std::env::remove_var(crate::constants::ENV_STORAGE_EMULATOR_HOST);
        
        let endpoint = resolve_gcs_endpoint();
        assert!(endpoint.is_none(), "Expected None when no env vars set");
    }

    #[test]
    fn test_resolve_gcs_endpoint_with_primary_env_var() {
        let _guard = ENV_MUTEX.lock().unwrap();
        
        // Set primary env var
        std::env::set_var(crate::constants::ENV_GCS_ENDPOINT_URL, "http://localhost:4443");
        std::env::remove_var(crate::constants::ENV_STORAGE_EMULATOR_HOST);
        
        let endpoint = resolve_gcs_endpoint();
        assert_eq!(endpoint, Some("http://localhost:4443".to_string()));
        
        // Cleanup
        std::env::remove_var(crate::constants::ENV_GCS_ENDPOINT_URL);
    }

    #[test]
    fn test_resolve_gcs_endpoint_with_emulator_host_no_scheme() {
        let _guard = ENV_MUTEX.lock().unwrap();
        
        // Set STORAGE_EMULATOR_HOST without http:// prefix (common convention)
        std::env::remove_var(crate::constants::ENV_GCS_ENDPOINT_URL);
        std::env::set_var(crate::constants::ENV_STORAGE_EMULATOR_HOST, "localhost:4443");
        
        let endpoint = resolve_gcs_endpoint();
        assert_eq!(endpoint, Some("http://localhost:4443".to_string()));
        
        // Cleanup
        std::env::remove_var(crate::constants::ENV_STORAGE_EMULATOR_HOST);
    }

    #[test]
    fn test_resolve_gcs_endpoint_with_emulator_host_with_scheme() {
        let _guard = ENV_MUTEX.lock().unwrap();
        
        // Set STORAGE_EMULATOR_HOST with http:// prefix
        std::env::remove_var(crate::constants::ENV_GCS_ENDPOINT_URL);
        std::env::set_var(crate::constants::ENV_STORAGE_EMULATOR_HOST, "http://127.0.0.1:9002");
        
        let endpoint = resolve_gcs_endpoint();
        assert_eq!(endpoint, Some("http://127.0.0.1:9002".to_string()));
        
        // Cleanup
        std::env::remove_var(crate::constants::ENV_STORAGE_EMULATOR_HOST);
    }

    #[test]
    fn test_resolve_gcs_endpoint_with_https_scheme() {
        let _guard = ENV_MUTEX.lock().unwrap();
        
        // Set STORAGE_EMULATOR_HOST with https:// prefix
        std::env::remove_var(crate::constants::ENV_GCS_ENDPOINT_URL);
        std::env::set_var(crate::constants::ENV_STORAGE_EMULATOR_HOST, "https://secure-emulator:4443");
        
        let endpoint = resolve_gcs_endpoint();
        assert_eq!(endpoint, Some("https://secure-emulator:4443".to_string()));
        
        // Cleanup
        std::env::remove_var(crate::constants::ENV_STORAGE_EMULATOR_HOST);
    }

    #[test]
    fn test_resolve_gcs_endpoint_primary_takes_precedence() {
        let _guard = ENV_MUTEX.lock().unwrap();
        
        // Set both env vars - primary should take precedence
        std::env::set_var(crate::constants::ENV_GCS_ENDPOINT_URL, "http://primary:4443");
        std::env::set_var(crate::constants::ENV_STORAGE_EMULATOR_HOST, "emulator:9999");
        
        let endpoint = resolve_gcs_endpoint();
        assert_eq!(endpoint, Some("http://primary:4443".to_string()));
        
        // Cleanup
        std::env::remove_var(crate::constants::ENV_GCS_ENDPOINT_URL);
        std::env::remove_var(crate::constants::ENV_STORAGE_EMULATOR_HOST);
    }
}
