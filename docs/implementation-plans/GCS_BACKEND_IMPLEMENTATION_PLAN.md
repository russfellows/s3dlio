# Google Cloud Storage (GCS) Backend Implementation Plan

## Executive Summary

This document outlines the implementation plan for adding Google Cloud Storage (GCS) as the 5th storage backend to s3dlio, joining the existing S3, Azure, local file, and DirectIO backends. The GCS backend will follow the same architecture patterns and provide equivalent functionality to the other cloud backends.

**Target Version**: v0.9.0  
**Estimated Effort**: Medium (similar to Azure implementation)  
**Status**: Planning Phase

## 1. Background & Motivation

### Current Architecture
s3dlio currently supports 4 storage backends:
1. **S3** (`s3://`) - AWS S3 and S3-compatible storage (MinIO, Vast)
2. **Azure** (`az://`) - Azure Blob Storage
3. **File** (`file://`) - Local filesystem with page cache optimization
4. **DirectIO** (`direct://`) - Local filesystem with O_DIRECT bypass

### Why Add GCS?
- **Market Coverage**: Complete the "big 3" cloud provider support (AWS, Azure, GCP)
- **ML/AI Workloads**: Many AI/ML teams use Google Cloud Platform and Vertex AI
- **Unified Interface**: Enable transparent storage operations across all major cloud providers
- **Performance Testing**: Benchmark GCS performance vs S3/Azure for AI/ML workloads

## 2. Technical Design

### 2.1 URI Scheme
**Primary**: `gs://bucket/path/to/object`  
**Alternative**: `gcs://bucket/path/to/object` (alias)

Examples:
```
gs://my-bucket/training-data/dataset.tfrecord
gs://model-checkpoints/epoch_10.pt
gs://inference-results/predictions.jsonl.zst
```

### 2.2 Rust SDK Selection

**Recommended**: `google-cloud-storage` crate (official Google Cloud Rust SDK)
- **Crate**: `google-cloud-storage = "^0.23"`
- **Supporting crates**:
  - `google-cloud-auth = "^0.19"` - Authentication (Application Default Credentials)
  - `google-cloud-default = "^0.8"` - Common types and utilities
  - `reqwest = "^0.12"` - HTTP client (already in project)

**Alternatives Considered**:
1. `cloud-storage` crate - Less maintained, older API
2. `gcp_auth` + raw REST API - More control but more work
3. Apache Arrow `object_store` - Already available but experimental status

**Decision**: Use official `google-cloud-storage` crate for production quality and long-term support.

### 2.3 Authentication Strategy

GCS authentication will follow Google Cloud's **Application Default Credentials (ADC)** chain:

1. **Environment Variable**: `GOOGLE_APPLICATION_CREDENTIALS` → path to service account JSON
2. **Metadata Server**: For GCE/GKE workloads (automatic)
3. **gcloud CLI**: User credentials from `~/.config/gcloud/application_default_credentials.json`
4. **Workload Identity**: For GKE pods with Workload Identity enabled

**Implementation**:
```rust
use google_cloud_auth::credentials::CredentialsFile;
use google_cloud_storage::client::{Client, ClientConfig};

// ADC will automatically find credentials
let config = ClientConfig::default().with_auth().await?;
let client = Client::new(config);
```

### 2.4 Module Structure

**New File**: `src/gcs_client.rs`
- Mirror the pattern from `azure_client.rs`
- Encapsulate GCS SDK operations
- Provide high-level operations: GET, PUT, LIST, DELETE, STAT

**Integration**: `src/object_store.rs`
- Add `GcsObjectStore` struct implementing `ObjectStore` trait
- Update `Scheme` enum to include `Gcs`
- Update `infer_scheme()` to recognize `gs://` and `gcs://`
- Update `store_for_uri()` factory to instantiate GCS backend

### 2.5 ObjectStore Implementation

```rust
// src/gcs_client.rs
pub struct GcsClient {
    client: google_cloud_storage::client::Client,
}

impl GcsClient {
    pub async fn new() -> Result<Self> {
        let config = ClientConfig::default().with_auth().await?;
        let client = Client::new(config);
        Ok(Self { client })
    }
    
    // Core operations
    pub async fn get_object(&self, bucket: &str, object: &str) -> Result<Vec<u8>>;
    pub async fn get_object_range(&self, bucket: &str, object: &str, offset: u64, length: Option<u64>) -> Result<Vec<u8>>;
    pub async fn put_object(&self, bucket: &str, object: &str, data: &[u8]) -> Result<()>;
    pub async fn put_object_multipart(&self, bucket: &str, object: &str, data: &[u8], chunk_size: usize) -> Result<()>;
    pub async fn list_objects(&self, bucket: &str, prefix: Option<&str>, recursive: bool) -> Result<Vec<String>>;
    pub async fn stat_object(&self, bucket: &str, object: &str) -> Result<GcsObjectMetadata>;
    pub async fn delete_object(&self, bucket: &str, object: &str) -> Result<()>;
    pub async fn delete_objects(&self, bucket: &str, objects: Vec<String>) -> Result<()>;
    pub async fn create_bucket(&self, bucket: &str, location: Option<&str>) -> Result<()>;
    pub async fn delete_bucket(&self, bucket: &str) -> Result<()>;
}

// src/object_store.rs
pub struct GcsObjectStore {
    client: Arc<GcsClient>,
}

#[async_trait]
impl ObjectStore for GcsObjectStore {
    // Implement all required trait methods
}
```

### 2.6 URI Parsing

**Pattern**: `gs://bucket/path/to/object` or `gcs://bucket/path/to/object`

```rust
fn parse_gcs_uri(uri: &str) -> Result<(String, String)> {
    // Remove gs:// or gcs:// prefix
    let path = uri.strip_prefix("gs://")
        .or_else(|| uri.strip_prefix("gcs://"))
        .ok_or_else(|| anyhow!("Invalid GCS URI: {}", uri))?;
    
    // Split bucket/object
    let parts: Vec<&str> = path.splitn(2, '/').collect();
    match parts.as_slice() {
        [bucket] => Ok((bucket.to_string(), String::new())),
        [bucket, object] => Ok((bucket.to_string(), object.to_string())),
        _ => bail!("Invalid GCS URI format: {}", uri),
    }
}
```

### 2.7 Performance Optimizations

#### Multipart Upload Strategy
GCS supports **resumable uploads** for large objects (>5MB recommended):
- Use resumable upload API for objects >32MB (configurable threshold)
- Stream data in chunks (default 8MB per chunk)
- Parallel chunk uploads when beneficial

#### Range Request Strategy
- Use byte-range requests for partial reads
- Concurrent range requests for large files (>32MB)
- Leverage `PageCacheMode::Sequential` for streaming workloads

#### Connection Pooling
- Reuse `reqwest::Client` with connection pooling
- Configure timeout and retry policies
- Match S3/Azure performance targets:
  - **Read (GET)**: Target 5+ GB/s
  - **Write (PUT)**: Target 2.5+ GB/s

### 2.8 Feature Parity Matrix

| Feature | S3 | Azure | File | Direct | **GCS** |
|---------|----|----|------|--------|---------|
| GET (full object) | ✅ | ✅ | ✅ | ✅ | 🔄 |
| GET (range) | ✅ | ✅ | ✅ | ✅ | 🔄 |
| PUT (single-shot) | ✅ | ✅ | ✅ | ✅ | 🔄 |
| PUT (multipart) | ✅ | ✅ | ✅ | ✅ | 🔄 |
| LIST (recursive) | ✅ | ✅ | ✅ | ✅ | 🔄 |
| STAT (metadata) | ✅ | ✅ | ✅ | ✅ | 🔄 |
| DELETE (single) | ✅ | ✅ | ✅ | ✅ | 🔄 |
| DELETE (batch) | ✅ | ✅ | ✅ | ✅ | 🔄 |
| CREATE (bucket) | ✅ | ✅ | ✅ | ✅ | 🔄 |
| DELETE (bucket) | ✅ | ✅ | ✅ | ✅ | 🔄 |
| Op-log tracing | ✅ | ✅ | ✅ | ✅ | 🔄 |
| Compression | ✅ | ✅ | ✅ | ✅ | 🔄 |
| Checksums | ✅ | ✅ | ✅ | ✅ | 🔄 |
| ObjectWriter | ✅ | ✅ | ✅ | ✅ | 🔄 |

Legend: ✅ Implemented | 🔄 To Implement | ❌ Not Applicable

## 3. Implementation Phases

### Phase 1: Core Infrastructure (Day 1-2)
1. Add GCS dependencies to `Cargo.toml`
2. Create `src/gcs_client.rs` module
3. Implement authentication (ADC)
4. Implement URI parsing (`parse_gcs_uri`)
5. Update `Scheme` enum and `infer_scheme()`

### Phase 2: Basic Operations (Day 2-3)
1. Implement `GcsClient::get_object()` - full object read
2. Implement `GcsClient::put_object()` - single-shot write
3. Implement `GcsClient::stat_object()` - metadata
4. Implement `GcsClient::delete_object()` - single delete
5. Implement `GcsClient::list_objects()` - list with prefix

### Phase 3: Advanced Operations (Day 3-4)
1. Implement `GcsClient::get_object_range()` - byte-range reads
2. Implement `GcsClient::put_object_multipart()` - resumable uploads
3. Implement `GcsClient::delete_objects()` - batch delete
4. Implement `GcsClient::create_bucket()` - bucket creation
5. Implement `GcsClient::delete_bucket()` - bucket deletion

### Phase 4: ObjectStore Integration (Day 4-5)
1. Create `GcsObjectStore` struct
2. Implement `ObjectStore` trait for `GcsObjectStore`
3. Update `store_for_uri()` factory
4. Add `GcsObjectWriter` for streaming uploads
5. Ensure op-log integration works

### Phase 5: Testing & Validation (Day 5-7)
1. Unit tests for `gcs_client.rs`
2. Integration tests for `GcsObjectStore`
3. Performance benchmarks (compare to S3/Azure)
4. Python API smoke tests
5. Documentation and examples

### Phase 6: Documentation & Release (Day 7-8)
1. Update README.md with GCS examples
2. Update Changelog.md for v0.9.0
3. Add GCS configuration guide
4. Update API documentation
5. Create migration guide for GCS users

## 4. Dependencies

### New Cargo Dependencies
```toml
[dependencies]
# Google Cloud Storage SDK
google-cloud-storage = "^0.23"
google-cloud-auth = "^0.19"
google-cloud-default = "^0.8"

# Already have these (verify versions)
reqwest = { version = "^0.12", features = ["rustls-tls"] }
tokio = { version = "^1", features = ["full"] }
```

### Environment Variables
- `GOOGLE_APPLICATION_CREDENTIALS` - Path to service account JSON (optional)
- `GOOGLE_CLOUD_PROJECT` - Default GCP project (optional)
- `S3DLIO_GCS_ENDPOINT` - Custom endpoint for emulator testing (optional)

### Testing Requirements
- **Local Emulator**: Use `fake-gcs-server` or Google Cloud Storage emulator
- **Real GCS**: Test account with test bucket
- **CI/CD**: Service account with minimal permissions

## 5. Code Examples

### 5.1 Rust API Usage
```rust
use s3dlio::api::{ObjectStore, store_for_uri};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create GCS store
    let store = store_for_uri("gs://my-bucket/data/")?;
    
    // Read object
    let data = store.get("gs://my-bucket/data/file.txt").await?;
    println!("Read {} bytes", data.len());
    
    // Write object
    store.put("gs://my-bucket/data/output.txt", b"Hello GCS!").await?;
    
    // List objects
    let objects = store.list("gs://my-bucket/data/", true).await?;
    for obj in objects {
        println!("Found: {}", obj);
    }
    
    Ok(())
}
```

### 5.2 Python API Usage
```python
import s3dlio

# Upload to GCS
s3dlio.upload(["./data/*.tfrecord"], "gs://training-data/", max_in_flight=16)

# Download from GCS
s3dlio.download("gs://model-checkpoints/", "./models/", recursive=True)

# Copy GCS to S3
s3dlio.copy("gs://source-bucket/", "s3://dest-bucket/", recursive=True)
```

### 5.3 CLI Usage
```bash
# Copy local to GCS
s3-cli copy ./data/ gs://my-bucket/data/ --recursive

# Copy GCS to Azure
s3-cli copy gs://source/ az://container/dest/ --recursive

# List GCS objects
s3-cli list gs://my-bucket/prefix/

# Delete GCS objects
s3-cli delete gs://my-bucket/old-data/ --recursive
```

## 6. Testing Strategy

### 6.1 Unit Tests
Location: `src/gcs_client.rs` and `src/object_store.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_parse_gcs_uri() {
        let (bucket, object) = parse_gcs_uri("gs://my-bucket/path/to/file.txt").unwrap();
        assert_eq!(bucket, "my-bucket");
        assert_eq!(object, "path/to/file.txt");
    }
    
    #[tokio::test]
    async fn test_gcs_get_put() {
        // Requires GCS credentials
        let client = GcsClient::new().await.unwrap();
        
        // PUT
        client.put_object("test-bucket", "test.txt", b"Hello GCS").await.unwrap();
        
        // GET
        let data = client.get_object("test-bucket", "test.txt").await.unwrap();
        assert_eq!(data, b"Hello GCS");
        
        // DELETE
        client.delete_object("test-bucket", "test.txt").await.unwrap();
    }
}
```

### 6.2 Integration Tests
Location: `tests/test_gcs_backend.rs`

- Test all `ObjectStore` trait methods
- Test cross-backend copy (S3 → GCS, GCS → Azure, etc.)
- Test error handling (missing credentials, 404s, etc.)
- Test large file operations (multipart uploads)

### 6.3 Performance Benchmarks
Location: `benches/gcs_performance.rs`

- Benchmark GET throughput (compare to S3/Azure)
- Benchmark PUT throughput (single-shot vs multipart)
- Benchmark list operations (1K, 10K, 100K objects)
- Benchmark concurrent operations (16, 32, 64 parallel)

### 6.4 Python Integration Tests
Location: `python/tests/test_gcs_backend.py`

```python
def test_gcs_upload_download():
    # Upload to GCS
    s3dlio.upload(["test_data.txt"], "gs://test-bucket/")
    
    # Download from GCS
    s3dlio.download("gs://test-bucket/test_data.txt", "./downloaded/")
    
    # Verify content
    with open("./downloaded/test_data.txt") as f:
        assert f.read() == "test content"
```

## 7. Performance Targets

### Throughput Targets (Match S3/Azure)
- **Read (GET)**: ≥5 GB/s (50 Gbps) sustained
- **Write (PUT)**: ≥2.5 GB/s (25 Gbps) sustained
- **List**: 10K objects/sec

### Latency Targets
- **GET (small object <1MB)**: <50ms p99
- **PUT (small object <1MB)**: <100ms p99
- **STAT**: <20ms p99
- **LIST (1K objects)**: <500ms p99

### Concurrency Targets
- **Concurrent GETs**: 32 (default)
- **Concurrent PUTs**: 16 (default)
- **Multipart chunk size**: 8MB (configurable)

## 8. Configuration Options

### Environment Variables
```bash
# Authentication
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
export GOOGLE_CLOUD_PROJECT=my-project-id

# Performance tuning
export S3DLIO_GCS_CONCURRENT_GETS=32
export S3DLIO_GCS_CONCURRENT_PUTS=16
export S3DLIO_GCS_MULTIPART_THRESHOLD=33554432  # 32MB
export S3DLIO_GCS_MULTIPART_CHUNK_SIZE=8388608   # 8MB

# Endpoint override (for emulator)
export S3DLIO_GCS_ENDPOINT=http://localhost:4443
```

### Rust Configuration
```rust
// Future enhancement: GCS-specific config struct
pub struct GcsConfig {
    pub project_id: Option<String>,
    pub credentials_path: Option<String>,
    pub endpoint: Option<String>,
    pub concurrent_gets: usize,
    pub concurrent_puts: usize,
    pub multipart_threshold: u64,
    pub chunk_size: usize,
}
```

## 9. Error Handling

### GCS-specific Errors
- **401 Unauthorized**: Invalid or missing credentials
- **403 Forbidden**: Insufficient permissions
- **404 Not Found**: Bucket or object doesn't exist
- **409 Conflict**: Bucket name already taken
- **429 Rate Limit**: Too many requests

### Error Mapping
```rust
fn map_gcs_error(err: google_cloud_storage::http::Error) -> anyhow::Error {
    match err.status() {
        Some(StatusCode::NOT_FOUND) => anyhow!("GCS object not found"),
        Some(StatusCode::UNAUTHORIZED) => anyhow!("GCS authentication failed"),
        Some(StatusCode::FORBIDDEN) => anyhow!("GCS permission denied"),
        Some(StatusCode::TOO_MANY_REQUESTS) => anyhow!("GCS rate limit exceeded"),
        _ => anyhow!("GCS error: {}", err),
    }
}
```

## 10. Documentation Updates

### Files to Update
1. **README.md**
   - Add GCS to backend list
   - Update badge count (4 → 5 backends)
   - Add GCS examples

2. **docs/Changelog.md**
   - Add v0.9.0 entry with GCS backend

3. **docs/GCS-QUICKSTART.md** (NEW)
   - Authentication setup
   - Basic usage examples
   - Performance tuning

4. **src/lib.rs**
   - Update module exports
   - Update crate-level docs

5. **Python docstrings**
   - Update `upload()`, `download()`, `copy()` docs with GCS examples

## 11. Migration Considerations

### For Existing Users
- **No Breaking Changes**: GCS is additive, doesn't affect existing backends
- **Backward Compatibility**: All existing code continues to work
- **Opt-in**: GCS only used when `gs://` URIs are provided

### For GCS Users
- **Drop-in Replacement**: Replace `gsutil` or custom GCS code with s3dlio
- **Unified API**: Same API works for GCS, S3, Azure, and local files
- **Performance**: Potentially better performance than `gsutil` for ML workloads

## 12. Security Considerations

### Authentication
- **Service Account**: Recommended for production (JSON key file)
- **Workload Identity**: Best for GKE deployments
- **User Credentials**: OK for development (gcloud CLI)
- **No Hardcoded Keys**: Never commit credentials to git

### Permissions Required
Minimal IAM permissions for s3dlio operations:
- `storage.objects.get` - Read objects
- `storage.objects.create` - Write objects
- `storage.objects.delete` - Delete objects
- `storage.objects.list` - List objects
- `storage.buckets.create` - Create buckets (optional)
- `storage.buckets.delete` - Delete buckets (optional)

### Data Security
- **TLS by Default**: All GCS operations use HTTPS
- **Customer-Managed Keys**: Support for CMEK (future enhancement)
- **Object Versioning**: Preserve object versions (future enhancement)

## 13. Known Limitations

### Initial Release (v0.9.0)
- No support for GCS signed URLs (can add later)
- No support for GCS lifecycle policies (out of scope)
- No support for GCS requester pays (can add later)
- No support for custom metadata (can add later)

### Future Enhancements (v0.10.0+)
- GCS-specific features (CMEK, versioning, lifecycle)
- GCS Transfer Service integration
- Multi-region bucket support
- GCS Analytics integration

## 14. Success Criteria

### Functional Requirements ✅
- [ ] GCS backend implements full `ObjectStore` trait
- [ ] `gs://` URIs work in all APIs (Rust, Python, CLI)
- [ ] All operations logged to op-log
- [ ] Compression support works with GCS
- [ ] Cross-backend copy works (GCS ↔ S3 ↔ Azure)

### Performance Requirements ✅
- [ ] GET throughput ≥5 GB/s (match S3)
- [ ] PUT throughput ≥2.5 GB/s (match S3)
- [ ] Latency p99 <100ms for small objects

### Quality Requirements ✅
- [ ] Zero compiler warnings
- [ ] All tests passing (unit, integration, doc)
- [ ] Python tests passing
- [ ] Documentation complete and accurate

### Release Requirements ✅
- [ ] Changelog.md updated
- [ ] README.md updated
- [ ] Version bumped to v0.9.0
- [ ] Release notes published

## 15. Risk Assessment

### Technical Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GCS SDK API changes | Low | Medium | Pin version, test thoroughly |
| Auth complexity | Medium | High | Use ADC, comprehensive docs |
| Performance below target | Low | High | Benchmark early, optimize |
| Breaking changes to existing backends | Very Low | Critical | Comprehensive regression tests |

### Schedule Risks
| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Unfamiliar with GCS SDK | Medium | Medium | Study Azure impl, read docs |
| Testing delays (need GCS account) | Low | Low | Use emulator for most tests |
| Integration complexity | Low | Medium | Follow proven patterns |

## 16. Open Questions

1. **Emulator Support**: Which GCS emulator to recommend? (fake-gcs-server vs official)
2. **Default Location**: Should we default GCS buckets to a specific region? (US vs EU vs ASIA)
3. **Timeout Values**: What timeouts for GCS operations? (Match S3 or different?)
4. **Retry Logic**: Use SDK retry logic or implement custom? (Probably SDK)
5. **Metrics**: Should we add GCS-specific metrics? (bytes transferred, request count, etc.)

## 17. References

### Documentation
- [Google Cloud Storage Rust SDK](https://docs.rs/google-cloud-storage/latest/google_cloud_storage/)
- [GCS REST API](https://cloud.google.com/storage/docs/json_api)
- [GCS Authentication](https://cloud.google.com/docs/authentication)
- [GCS Best Practices](https://cloud.google.com/storage/docs/best-practices)

### Existing Implementations
- `src/azure_client.rs` - Reference implementation for cloud backend
- `src/s3_client.rs` - Alternative reference for cloud backend
- `src/object_store.rs` - Core trait and factory patterns

### Tools
- [fake-gcs-server](https://github.com/fsouza/fake-gcs-server) - GCS emulator for testing
- [gsutil](https://cloud.google.com/storage/docs/gsutil) - Official GCS CLI (for comparison)

---

**Status**: ✅ Planning Complete - Ready for Implementation  
**Next Step**: Create TODO list and begin Phase 1 implementation  
**Author**: AI Coding Agent  
**Date**: October 2, 2025
