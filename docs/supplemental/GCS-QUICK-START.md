# GCS Quick Start Guide

s3dlio has ONE Google Cloud Storage backend: the official `google-cloud-storage` client from the
patched `russfellows/google-cloud-rust` fork (gRPC, with Rapid / zonal bucket support). It is
selected with the `backend-gcs` feature, which `full-backends` includes and which the full Python
wheel already uses. There is no backend to choose between.

## Build & Use

```bash
# GCS only (plus the default AWS backend)
cargo build --release --features backend-gcs

# AWS + Azure + GCS
cargo build --release --features full-backends
```

### Authentication

Set up credentials (choose one):
```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
# OR: gcloud auth application-default login
```

### Use in Code
```rust
use s3dlio::api::store_for_uri;

let store = store_for_uri("gs://my-bucket/prefix/")?;
let objects = store.list("gs://my-bucket/prefix/", false).await?; // inside an async fn
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

### Live tests

The live GCS tests are `#[ignore]`d and read their bucket and prefix from the environment; see
the header of `tests/live_caller_buffer_gcs.rs` for the variables and the command. To check
`list-buckets` against GCS, see `scripts/test_list_buckets_gcs.sh`.

## URI Schemes

Both `gs://` and `gcs://` work identically:
```bash
gs://bucket/prefix/object.txt
gcs://bucket/prefix/object.txt
```

## Authentication details

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

## Using an emulator

Point the official client at an emulator or test server with an environment variable (read when the
first GCS client is built, after the `.env` policy):

```bash
# Emulator convention: host:port, http:// is added.
export STORAGE_EMULATOR_HOST=localhost:4443
export S3DLIO_GCS_RAPID=0   # standard buckets: skip the bucket-type probe

# Or a full URL; wins over STORAGE_EMULATOR_HOST.
export GCS_ENDPOINT_URL=http://localhost:4443
```

Either variable switches the client to **anonymous credentials** (no Application Default Credentials
are looked up). An empty value counts as unset. Emulators serve standard buckets: also set
`S3DLIO_GCS_RAPID=0` so s3dlio does not probe the bucket type (the existing setting; nothing else
changes). With neither set, nothing changes and the client talks to
`storage.googleapis.com`.

The endpoint applies to **both** halves of the client, and they use different protocols (from the
code): reads (`get`, `get_range`) use HTTP/JSON; `put`, `stat`, `list`, `delete` and bucket operations
use the Storage gRPC API. Limitation: an emulator that serves only the JSON API supports reads only.

What was tested: a local stand-in (not a real emulator) receives the HTTP read and the gRPC connection
of a client built from either variable, and the read carries no `Authorization` header
(`tests/gcs_emulator_endpoint.rs`, `tests/gcs_endpoint_url_credentials.rs`). Nothing was run against a
real emulator. See
[Environment_Variables.md](../Environment_Variables.md) (Google Cloud Storage Configuration) for the
full table.

## Troubleshooting

### Error: GCS backend is not enabled in this build
**Problem**: `gs://` operations fail with "GCS backend is not enabled in this build. Rebuild with feature
'backend-gcs' or 'full-backends'."

**Solution**: build with `--features backend-gcs` (or `full-backends`).

### Build error: unknown feature `gcs-community` (or `gcs-official`)
**Problem**: the legacy community GCS client (`gcs-community`, the `gcloud-storage` crate) was
removed in 0.10.0, and `gcs-official` never existed as a feature name in current releases.

**Solution**: use `backend-gcs`.

### Authentication Failed
**Problem**: No credentials found  
**Solution**: Set up ADC (see Authentication above)

## More Information

- **Changelog**: [docs/Changelog.md](../Changelog.md)
- **Environment variables**: [docs/Environment_Variables.md](../Environment_Variables.md)
- **History** of the two-backend period: [v0.9.8-GCS-BACKENDS-SUMMARY.md](v0.9.8-GCS-BACKENDS-SUMMARY.md)
