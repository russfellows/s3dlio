# GCS Configuration API Reference — s3dlio v0.9.65+

**Status**: Implemented (v0.9.65)  
**Date**: 2026-03-01  
**See also**: [docs/GCS-gRPC_Fixes.md](GCS-gRPC_Fixes.md) — root-cause analysis and performance results  
**See also**: [docs/GCS-gRPC-Transport.md](GCS-gRPC-Transport.md) — transport architecture

---

## Overview

All GCS/gRPC tuning is governed by a clear two-layer API:

1. **Programmatic API** — Rust functions called before the first GCS operation.
   Intended for library users and tools like sai3-bench that know their
   workload parameters at startup.
2. **Environment variables** — Runtime overrides. Always take precedence over
   the programmatic API, allowing ad-hoc tuning without recompilation.

Each setting has a defined priority order.  The GCS client is a **process-wide
singleton** initialized on first use — all programmatic settings must be applied
before any `gs://` I/O.

---

## Programmatic API

### `s3dlio::set_gcs_channel_count(n: usize)`

Pre-configure the number of gRPC subchannels (TCP connections) the GCS client
will open.

```rust
// Call once at startup, before any GCS operation.
s3dlio::set_gcs_channel_count(64);
```

**Why it matters**: Each subchannel is one HTTP/2 connection with its own
flow-control window.  At the default auto-tune floor of 64, 64 concurrent jobs
each get an uncontested 128 MiB window.  Fewer subchannels than concurrent jobs
means multiple streams share one window and stall waiting for WINDOW_UPDATE
frames.

**Priority order** (highest wins):
1. `S3DLIO_GCS_GRPC_CHANNELS` env var
2. This function call
3. `max(64, cpu_count)` auto fallback

**Best practice for tools**: call with your job/concurrency count:
```rust
s3dlio::set_gcs_channel_count(my_concurrency);
```

---

### `s3dlio::set_gcs_rapid_mode(force: Option<bool>)` *(new in v0.9.65)*

Pre-configure RAPID (Hyperdisk ML / zonal GCS) mode before the first GCS
operation.

```rust
s3dlio::set_gcs_rapid_mode(Some(true));   // force RAPID on all buckets
s3dlio::set_gcs_rapid_mode(Some(false));  // force RAPID off all buckets
s3dlio::set_gcs_rapid_mode(None);         // auto-detect per bucket (default)
```

**Why it matters**: RAPID buckets require the `appendable=true` write API and
use `BidiReadObject` for reads.  Calling the wrong API on a non-RAPID bucket
returns an error; auto-detection adds one `GetStorageLayout` RPC per bucket
(cached for the process lifetime).

**Priority order** (highest wins):
1. `S3DLIO_GCS_RAPID` env var (`true`/`1`/`yes`, `false`/`0`/`no`, `auto`)
2. This function call (`Some(true)` / `Some(false)`)
3. `None` / unset → per-bucket auto-detect via `GetStorageLayout`

**When to use `Some(true)`**: you know all target buckets are RAPID (Hyperdisk
ML) and want to skip the `GetStorageLayout` detection RPC.

**When to use `Some(false)`**: you know no target buckets are RAPID and want to
force the standard HTTP read path.

**When to use `None`** (recommended default): auto-detect is correct for mixed
workloads and adds negligible overhead (one cached RPC per unique bucket name).

---

### `s3dlio::get_gcs_channel_count() -> usize` *(new in v0.9.65)*

Read back the programmatic subchannel count previously set by
`set_gcs_channel_count`.

```rust
let n = s3dlio::get_gcs_channel_count();
// n == 0 means set_gcs_channel_count was never called;
// the client will auto-detect from max(64, cpu_count).
```

Useful for logging the effective configuration at workload startup.  Note that
the `S3DLIO_GCS_GRPC_CHANNELS` env var takes precedence inside `GcsClient::new()`;
this getter returns only the programmatic setting, not the env-var override.

---

### `s3dlio::get_gcs_rapid_mode() -> Option<bool>` *(new in v0.9.65)*

Read back the current effective RAPID mode, combining the env var and the
programmatic setting.

```rust
match s3dlio::get_gcs_rapid_mode() {
    Some(true)  => println!("RAPID: forced on"),
    Some(false) => println!("RAPID: forced off"),
    None        => println!("RAPID: auto-detect per bucket"),
}
```

Resolution order (same as `GcsClient::new()`):
1. `S3DLIO_GCS_RAPID` env var
2. Value set by `set_gcs_rapid_mode()`
3. Unset → `None` (auto-detect)

---

### `s3dlio::query_gcs_rapid_bucket(bucket_or_uri: &str) -> bool` *(new in v0.9.65)*

Async function that returns `true` if a bucket is a RAPID (Hyperdisk ML /
zonal) bucket.

```rust
// Bare bucket name
let rapid = s3dlio::query_gcs_rapid_bucket("my-bucket").await;

// Full URI — scheme and object path are extracted automatically
let rapid = s3dlio::query_gcs_rapid_bucket("gs://my-bucket/checkpoints/").await;
```

**Key properties**:
- Calls `GetStorageLayout` on first access; result is cached per bucket for the
  process lifetime (same cache used by every `get_object` / `put_object` call).
- Safe to call concurrently: duplicate calls for the same bucket are
  deduplicated — only one RPC is ever issued.
- Returns `false` and logs a warning on auth / network errors.
- Respects `ForceOn` / `ForceOff` set by `set_gcs_rapid_mode` or env var
  (returns immediately without an RPC).

**Use case**: log or record whether RAPID is active for a given benchmark run.

---

## Environment Variables

All environment variables take precedence over programmatic API calls.

| Variable | Default | Description |
|----------|---------|-------------|
| `S3DLIO_GCS_RAPID` | `auto` | RAPID mode: `true`/`1`/`yes` force on; `false`/`0`/`no` force off; `auto` or unset = per-bucket detection |
| `S3DLIO_GCS_GRPC_CHANNELS` | auto | Force exact subchannel count; overrides `set_gcs_channel_count()` and auto fallback |
| `S3DLIO_GRPC_INITIAL_WINDOW_MIB` | `128` | HTTP/2 initial stream + connection window in MiB. Set `0` for protocol default (65 KB — not recommended). |
| `S3DLIO_GRPC_WRITE_CHUNK_SIZE` | `2097152` | gRPC write data payload per `BidiWriteObjectRequest` in bytes. Silently clamped to `MAX_GRPC_WRITE_CHUNK_SIZE` (4,128,768 = 4 MiB − 64 KiB). |

### Debug log line

On first client initialization the library emits:
```
GCS gRPC config: subchannels=64 (source=jobs/concurrency, cpus=8), initial_window=128 MiB
```

`source` shows which priority tier was used for subchannel count:
- `env-var` — `S3DLIO_GCS_GRPC_CHANNELS` was set
- `jobs/concurrency` — `set_gcs_channel_count()` was called
- `auto-fallback` — no override; used `max(64, cpu_count)`

---

## Constants Reference

Defined in `vendor/google-cloud-gax-internal/src/gcs_constants.rs` and
re-exported from `src/gcs_constants.rs`.

| Constant | Value | Bytes | Purpose |
|----------|-------|-------|---------|
| `GCS_SERVER_MAX_MESSAGE_SIZE` | `4 * 1024 * 1024` | 4,194,304 | Raw GCS server ceiling — the complete serialised `BidiWriteObjectRequest` must be below this |
| `MAX_GRPC_WRITE_CHUNK_SIZE` | `GCS_SERVER_MAX_MESSAGE_SIZE − (64 * 1024)` | 4,128,768 | Maximum safe data payload (63 × 64 KiB); the 64 KiB guard band absorbs ~89-byte protobuf framing overhead |
| `DEFAULT_GRPC_WRITE_CHUNK_SIZE` | `2 * 1024 * 1024` | 2,097,152 | Default chunk size (32 × 64 KiB); conservative, well below ceiling |
| `GCS_MIN_CHANNELS` | `64` | — | Floor for subchannel auto-tune |
| `GCS_MAX_CONCURRENT_DELETES` | `64` | — | Semaphore limit for batch delete concurrency |
| `ENV_GCS_GRPC_CHANNELS` | `"S3DLIO_GCS_GRPC_CHANNELS"` | — | Env var name constant |
| `ENV_GCS_RAPID` | `"S3DLIO_GCS_RAPID"` | — | Env var name constant |
| `ENV_GRPC_INITIAL_WINDOW_MIB` | `"S3DLIO_GRPC_INITIAL_WINDOW_MIB"` | — | Env var name constant |
| `ENV_GRPC_WRITE_CHUNK_SIZE` | `"S3DLIO_GRPC_WRITE_CHUNK_SIZE"` | — | Env var name constant |

All chunk size values are 64 KiB-aligned.

---

## Typical Startup Sequence (Library Users)

```rust
use s3dlio;

// 1. Set channel count to match your concurrency BEFORE first GCS I/O.
s3dlio::set_gcs_channel_count(my_job_count);

// 2. Optionally override RAPID mode (default auto-detect is usually correct).
//    Only needed if you know all buckets are RAPID and want to skip detection.
s3dlio::set_gcs_rapid_mode(Some(true));

// 3. Optionally log the effective settings for diagnostics / benchmarking.
let chans = s3dlio::get_gcs_channel_count();
let rapid = s3dlio::get_gcs_rapid_mode();
println!("GCS: {chans} subchannels, RAPID mode = {rapid:?}");

// 4. Optionally query a specific bucket before I/O.
let is_rapid = s3dlio::query_gcs_rapid_bucket("gs://my-bucket/").await;
println!("Bucket is RAPID: {is_rapid}");

// 5. Now proceed with GCS operations — client initialises on first use.
let store = s3dlio::store_for_uri("gs://my-bucket/prefix/").await?;
```

---

## Python API

All five GCS tuning functions are exposed in the `s3dlio` Python module.  Call
setters before issuing any `gs://` I/O:

```python
import s3dlio

# Setters — must be called before first gs:// operation
s3dlio.gcs_set_channel_count(64)         # match your concurrency
s3dlio.gcs_set_rapid_mode(True)          # True / False / None (auto)

# Getters — read back the current effective settings
channels = s3dlio.gcs_get_channel_count()  # int, 0 = not set
rapid    = s3dlio.gcs_get_rapid_mode()     # True / False / None

# Query a bucket (blocking; cached for process lifetime)
is_rapid = s3dlio.gcs_query_rapid_bucket("gs://my-bucket/")
```

| Python function | Notes |
|----------------|-------|
| `gcs_set_channel_count(n: int)` | Must call before first I/O; env var overrides |
| `gcs_set_rapid_mode(force: bool \| None)` | Must call before first I/O; env var overrides |
| `gcs_get_channel_count() → int` | Returns `0` if not set programmatically |
| `gcs_get_rapid_mode() → bool \| None` | Resolves env var + programmatic setting |
| `gcs_query_rapid_bucket(uri: str) → bool` | Blocking call; result cached per bucket |

---

## Implementation Notes

### Why `set_gcs_rapid_mode` was added (v0.9.65)

Before v0.9.65, `RapidMode` was `pub(crate)` and the only way to control it was
the `S3DLIO_GCS_RAPID` environment variable.  Tools like sai3-bench configure
settings programmatically from YAML and should not be forced to call
`std::env::set_var` to reach library internals.  The new function follows the
identical pattern to `set_gcs_channel_count` / `DESIRED_GCS_CHANNELS`.

### Internal implementation

```
DESIRED_GCS_RAPID: AtomicU8
  0 = unset  →  defer to env var, then Auto
  1 = ForceOn
  2 = ForceOff

read_rapid_mode() priority:
  if S3DLIO_GCS_RAPID env var is set  →  parse and return
  else if DESIRED_GCS_RAPID != 0      →  return ForceOn / ForceOff
  else                                →  return Auto (per-bucket detection)
```

The bucket RAPID cache (`BUCKET_RAPID_CACHE`) is a `LazyLock<Mutex<HashMap>>` of
`OnceCell<bool>` entries — concurrent callers for the same bucket share a single
detection RPC.
