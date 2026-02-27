# Using GCS RAPID / Hyperdisk ML Storage with s3dlio

**Applies to:** s3dlio v0.9.50+  
**Backend required:** `gcs-official` (default since v0.9.50)

---

## What is RAPID Storage?

Google Cloud Storage RAPID (also marketed as **Hyperdisk ML**) is a storage tier
designed for ultra-low-latency, high-throughput AI/ML inference workloads — primarily
serving large model weights to inference accelerators.

RAPID buckets impose two hard constraints that differ from ordinary GCS buckets:

1. **All writes must set `appendable=true`** in the gRPC `WriteObjectSpec`.
   Omitting this flag causes every PUT to fail immediately with:
   > `This bucket requires appendable objects. Make sure you use the appendable=true query parameter.`

2. **The JSON API does not work with RAPID objects.** Only the gRPC transport (HTTP/2)
   can read or write these objects. Any client using the JSON API will get
   `HTTP 400 Bad Request` on reads.

This means only the `gcs-official` backend (which uses the official Google gRPC SDK)
is capable of working with RAPID buckets. The `gcs-community` backend (JSON API) is
**incompatible** and cannot be made to work — it is a fundamental protocol limitation.

---

## Prerequisites

### Build and feature flag

`gcs-official` is the **default** Cargo feature as of v0.9.50:

```toml
# Cargo.toml default features (no action needed for standard builds)
default = ["s3", "native-backends", "gcs-official", "thread-pinning"]
```

To confirm which GCS backend your binary was compiled with:

```bash
# Should report "gcs-official" in feature list
cargo build --release
# Or check directly
cargo metadata --no-deps --format-version 1 | grep gcs
```

If you are compiling a custom build with explicit features, ensure `gcs-official` is
included and `gcs-community` is **not** — they are mutually exclusive:

```bash
cargo build --release --no-default-features --features s3,native-backends,gcs-official
```

### Authentication

s3dlio uses **Application Default Credentials (ADC)**, the standard Google Cloud
authentication chain:

1. `GOOGLE_APPLICATION_CREDENTIALS` — path to a service account JSON key file
2. GCE / GKE metadata server — automatic when running on Google Cloud
3. `gcloud` CLI credentials — `~/.config/gcloud/application_default_credentials.json`

For service account authentication (most common in CI/CD and on-premises):

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
```

---

## Environment Variables

### `S3DLIO_GCS_RAPID`

Enables appendable-object writes for RAPID / Hyperdisk ML buckets.

| Value | Effect |
|---|---|
| `true` | RAPID mode **on** — all GCS writes set `appendable=true` |
| `1` | RAPID mode **on** |
| `yes` | RAPID mode **on** |
| `false` | RAPID mode **off** (default) |
| `0` | RAPID mode **off** |
| *(absent)* | RAPID mode **off** |

Values are **case-insensitive** (`TRUE`, `YES`, `True` all work).

```bash
# Enable RAPID mode
export S3DLIO_GCS_RAPID=true

# Or any of these are equivalent
export S3DLIO_GCS_RAPID=1
export S3DLIO_GCS_RAPID=yes
```

> **Important:** `S3DLIO_GCS_RAPID=true` applies to **all GCS writes** in the process.
> Do not set this variable if you also write to standard (non-RAPID) GCS buckets in
> the same invocation — standard buckets do not accept appendable objects.

### `GOOGLE_APPLICATION_CREDENTIALS`

Path to a GCP service account JSON key file. Required unless running on GCP
infrastructure with an attached service account.

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/workspace/secrets/gcp-sa.json
```

### `GOOGLE_CLOUD_PROJECT` (optional)

Project ID used by `list-buckets` when listing GCS buckets natively via
`gs://` URIs. If not set, s3dlio will attempt to resolve the project from the
ADC credentials.

```bash
export GOOGLE_CLOUD_PROJECT=my-gcp-project-id
```

---

## Using a `.env` File

For local development and testing, s3dlio supports loading variables from a `.env`
file via the [`dotenvy`](https://crates.io/crates/dotenvy) crate. The file is read
at startup before the GCS client is initialized.

Create a `.env` file in your working directory:

```dotenv
# .env — GCS RAPID credentials and mode
GOOGLE_APPLICATION_CREDENTIALS=/workspace/secrets/gcp-sa.json
GOOGLE_CLOUD_PROJECT=my-gcp-project-id
S3DLIO_GCS_RAPID=true
```

**Precedence rule:** Shell environment variables always take precedence over `.env`
values. If `S3DLIO_GCS_RAPID` is already set in your shell, the `.env` value is
ignored. This is standard `dotenvy` behavior and prevents accidental overrides in
production environments.

---

## Quick-Start Example

```bash
# 1. Set credentials and enable RAPID mode
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
export S3DLIO_GCS_RAPID=true

# 2. Upload a file to a RAPID bucket
./target/release/s3-cli upload /local/model.safetensors gs://my-rapid-bucket/models/model.safetensors

# 3. Download from a RAPID bucket
./target/release/s3-cli download gs://my-rapid-bucket/models/model.safetensors /local/model-copy.safetensors

# 4. List objects in a RAPID bucket (no RAPID flag needed for listings)
./target/release/s3-cli list gs://my-rapid-bucket/models/

# 5. List all GCS buckets in your project
export GOOGLE_CLOUD_PROJECT=my-gcp-project-id
./target/release/s3-cli list-buckets gs://
```

---

## Using RAPID with the Rust / Python API

### Rust

Set the environment variable before constructing the object store:

```rust
// Enable RAPID mode via environment (can also be set in .env)
std::env::set_var("S3DLIO_GCS_RAPID", "true");

// Store creation picks up RAPID mode automatically
let store = store_for_uri("gs://my-rapid-bucket/").await?;

// Writes now use appendable=true
store.put("models/weights.bin", data).await?;
```

### Python

```python
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/path/to/sa.json"
os.environ["S3DLIO_GCS_RAPID"] = "true"

import s3dlio

# Upload with RAPID mode enabled
s3dlio.put("gs://my-rapid-bucket/models/weights.bin", data)

# Download
data = s3dlio.get("gs://my-rapid-bucket/models/weights.bin")
```

Or source a `.env` file before running:

```bash
# .env
GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa.json
S3DLIO_GCS_RAPID=true

# run
python my_training_script.py
```

---

## Diagnosing Common Errors

### PUT fails: "This bucket requires appendable objects"

```
Error: GCS PUT failed for gs://<bucket>/<object>:
  service error: This bucket requires appendable objects.
  Make sure you use the appendable=true query parameter.
```

**Fix:** Set `S3DLIO_GCS_RAPID=true` before running. This tells s3dlio to include
`appendable=true` in every write request spec.

### GET returns `HTTP 400 Bad Request`

Objects already stored in a RAPID bucket require the gRPC transport to read.
If you are seeing 400 on reads, verify:

1. You are using the `gcs-official` build (default since v0.9.50).
2. You are not using a `gcs-community` build (check your Cargo features).
3. Your service account has `storage.objects.get` on the bucket.

### Authentication error: "Request had insufficient authentication scopes"

Your ADC credentials or service account does not have the required IAM roles.
RAPID / Hyperdisk ML buckets require:

- `roles/storage.objectAdmin` (or `storage.objectCreator` + `storage.objectViewer`)
- `roles/storage.legacyBucketReader` (for listings)

Grant these in the Google Cloud Console under **IAM & Admin → IAM**.

### `GOOGLE_APPLICATION_CREDENTIALS` not found

```
Error: Failed to initialize GCS Storage client: No credentials found
```

The file path in `GOOGLE_APPLICATION_CREDENTIALS` does not exist or is not readable.
Run:

```bash
echo $GOOGLE_APPLICATION_CREDENTIALS
cat "$GOOGLE_APPLICATION_CREDENTIALS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['type'], d.get('client_email',''))"
```

---

## Summary of All Relevant Variables

| Variable | Required | Purpose |
|---|---|---|
| `GOOGLE_APPLICATION_CREDENTIALS` | Yes (outside GCP) | Path to service account JSON |
| `S3DLIO_GCS_RAPID` | Yes (RAPID buckets) | Enable appendable writes |
| `GOOGLE_CLOUD_PROJECT` | For bucket listing | GCP project ID |
| `S3DLIO_LOG` / `RUST_LOG` | No | Logging verbosity (e.g. `s3dlio=debug`) |

---

## Technical Details

s3dlio vendored `google-cloud-storage` v1.8.0 and added a `set_appendable()` method
to the `WriteObject` request builder. The upstream SDK's model already supports the
`appendable` field in `WriteObjectSpec`, but the high-level builder did not expose
it. A PR has been filed upstream; once merged s3dlio will switch back to the
unmodified upstream crate.

The vendor patch is in [`vendor/google-cloud-storage/`](../vendor/google-cloud-storage/)
and is wired via `[patch.crates-io]` in `Cargo.toml` — no changes to user-visible
APIs or build steps are required.

See [GCS-RAPID-ANALYSIS.md](GCS-RAPID-ANALYSIS.md) for the full bug analysis and
implementation history.
