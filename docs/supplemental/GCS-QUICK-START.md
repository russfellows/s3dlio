# GCS Quick Start Guide

## Build & Use (Simple!)

### Default Build - Just Works™
```bash
# Build with GCS support (uses production-ready gcs-community backend)
cargo build --release

# That's it! No complex flags needed.
```

### Run Tests
```bash
# Set up authentication (choose one):
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
# OR: gcloud auth application-default login

# Set test bucket
export GCS_TEST_BUCKET=your-test-bucket

# Run tests (10/10 should pass)
./scripts/test_gcs_community.sh
```

### Use in Code
```rust
use s3dlio::api::store_for_uri;

// Works immediately - no backend selection needed
let store = store_for_uri("gs://my-bucket/prefix/")?;
let objects = store.list("", false)?;
```

### Use CLI
```bash
# List objects
./target/release/s3-cli ls gs://my-bucket/prefix/ -r

# Upload files
./target/release/s3-cli upload /local/path/ gs://my-bucket/prefix/

# Download objects  
./target/release/s3-cli download gs://my-bucket/prefix/ /local/path/
```

## Advanced: Try Experimental Backend

**Warning**: gcs-official has 10-20% transport failure rate. Use gcs-community for production.

```bash
# Build with experimental backend
cargo build --release --no-default-features \
  --features native-backends,s3,gcs-official

# Test (expect 3-9/10 tests to pass due to upstream flakes)
./scripts/test_gcs_official.sh
```

## URI Schemes

Both `gs://` and `gcs://` work identically:
```bash
gs://bucket/prefix/object.txt
gcs://bucket/prefix/object.txt
```

## Authentication

GCS uses Application Default Credentials (ADC) in this order:
1. `GOOGLE_APPLICATION_CREDENTIALS` env var (service account JSON)
2. GCE/GKE metadata server (automatic on Google Cloud)
3. gcloud CLI credentials (`~/.config/gcloud/application_default_credentials.json`)

### Service Account (Recommended for Production)
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

### gcloud CLI (Recommended for Development)
```bash
gcloud auth application-default login
```

## Feature Comparison

| Backend | Build | Reliability | API | Recommendation |
|---------|-------|-------------|-----|----------------|
| `gcs-community` | **Default** | ✅ 100% | REST | **Use this** |
| `gcs-official` | Opt-in | ⚠️ 80-90% | gRPC | Experimental only |

## Troubleshooting

### Build Error: "Enable only one of: gcs-community or gcs-official"
**Problem**: Both backends enabled simultaneously  
**Solution**: Use `--no-default-features` when selecting gcs-official:
```bash
cargo build --no-default-features --features native-backends,s3,gcs-official
```

### Test Failures: "transport reports an error: transport error"
**Problem**: Using gcs-official backend (has upstream flakes)  
**Solution**: Use gcs-community (the default) for production:
```bash
cargo build --release  # Uses gcs-community
```

### Authentication Failed
**Problem**: No credentials found  
**Solution**: Set up ADC (see Authentication section above)

## More Information

- **Backend Comparison**: [docs/GCS-BACKEND-SELECTION.md](GCS-BACKEND-SELECTION.md)
- **Release Notes**: [docs/releases/v0.9.8-GCS-BACKENDS-SUMMARY.md](releases/v0.9.8-GCS-BACKENDS-SUMMARY.md)
- **Changelog**: [docs/Changelog.md](Changelog.md)
