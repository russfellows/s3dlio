# HTTP/2 (h2c) and Per-Endpoint Client Change Proposal

**Date**: April 17, 2026  
**Target**: s3dlio + sai3-bench  
**Goal**: Sustain 70,000 PUT ops/s per client × 4 clients = 280,000 PUT/s total across 16 VAST S3 endpoints

## Infrastructure

- 4 client nodes × 2×100Gb NICs → switch → 4 VAST CINs × 4 network connections = **16 unique S3 endpoints**

---

## Review of the Proposed Approach (from another agent)

### What's correct in the proposal

1. **The general idea is right**: You do need a custom HTTP client to get h2c (cleartext HTTP/2). The default AWS SDK HTTP client defaults to HTTP/1.1 on cleartext endpoints.

2. **`hyper_util::client::legacy::Client` with `.http2_only(true)` is the correct engine**. That's the hyper 1.x way to force HTTP/2 prior knowledge (h2c).

3. **The concept of injecting via `aws_config::defaults(...).http_client(custom)`** is correct — s3dlio already does exactly this in `aws_s3_client_async()`.

### What's wrong or misleading

**1. The crate names and APIs in the proposal don't match reality**

The proposal references `aws_smithy_runtime::client::http::hyper_014::HyperClientBuilder`. This doesn't exist in the actual dependency tree. s3dlio uses `aws-smithy-http-client` v1.1.12, not the older `aws-smithy-runtime` hyper adapter. The correct builder is `aws_smithy_http_client::Builder`.

**2. You can't just wrap a raw hyper `Client` and hand it to the SDK**

The proposal creates a `hyper_util::client::legacy::Client` directly and calls `HyperClientBuilder::new().build(hyper_client)`. That API doesn't exist in `aws-smithy-http-client` v1.1.12. The SDK requires a `SharedHttpClient`, which is constructed through `Builder::build_http()`, `Builder::build_https()`, or `Builder::build_with_connector_fn()`.

**3. The `Connector::builder().hyper_builder()` method is `pub(crate)` — not public**

This is why s3dlio's existing experimental code has the comment "requires patched aws-smithy-http-client." The `hyper_builder()` and `set_hyper_builder()` methods on `ConnectorBuilder` are `pub(crate)`, meaning external crates cannot call them. The experimental-http-client feature in s3dlio cannot compile against the stock crate.

**4. The proposal completely ignores the two real architectural problems**

It never mentions:
- The global `static CLIENT: OnceCell<Client>` — one S3 client for all endpoints
- The `parse_s3_uri()` function that strips the endpoint before calling `put_object_async()`

These are the actual bottlenecks, independent of HTTP version.

---

## What Actually Needs to Change — The Real Architecture

There are **three layers** of problems, in order of impact:

### Layer 1: Per-Endpoint Clients (Critical — blocks all multi-endpoint work)

**Current state**: Every S3 call goes through `build_ops_async()` which calls `aws_s3_client_async()` which returns the single `static CLIENT: OnceCell<Client>`. This client is configured with one `AWS_ENDPOINT_URL`. Every PUT in the entire process goes to the same endpoint.

When `MultiEndpointStore` is used, it calls `store_for_uri()` per endpoint, which creates an `S3ObjectStore`. But `S3ObjectStore::put()` calls `s3_put_object_uri_async()` → `parse_s3_uri()` → `put_object_async()` → `build_ops_async()` → the single global client. **The endpoint in the URI is parsed out by `parse_s3_uri()` and discarded — only bucket and key are kept.**

**Required fix**: Each `S3ObjectStore` needs to hold its own `aws_sdk_s3::Client` configured with the specific endpoint URL. The `put()` method should use that client directly, not the global singleton.

### Layer 2: h2c Transport (High impact — reduces connections 100×)

**Current state**: The default code path calls `HttpClientBuilder::new().build_http()`. Looking at the published source for `build_http()`:

```rust
pub fn build_http(self) -> SharedHttpClient {
    build_with_conn_fn(self.client_builder, self.pool_idle_timeout,
        move |client_builder, settings, runtime_components| {
            let builder = new_conn_builder(client_builder, settings, runtime_components);
            builder.build_http()
        },
    )
}
```

And `new_tokio_hyper_builder()` is the default `client_builder` — it creates a plain `hyper_util::client::legacy::Builder` with no HTTP/2 settings. The result is HTTP/1.1 on cleartext.

**The actual viable approach** (no patching required):

`build_with_connector_fn` is `#[doc(hidden)]` but **public** on `Builder<TlsUnset>`. Inside the closure, you construct a `Connector` by calling `Connector::builder().build_http()` — which uses the **default** hyper builder (HTTP/1.1). But here's the trick: the `build_with_conn_fn` function already takes `self.client_builder` and passes it through to the inner closure. The problem is that `build_with_connector_fn`'s user closure receives `(settings, runtime_components)` but NOT the builder — it's discarded via `_builder`.

**The correct solution**: Don't use `build_with_connector_fn`. Instead, use `build_http()` but set `self.client_builder` first. The `Builder` struct has `client_builder: Option<hyper_util::client::legacy::Builder>` but no public setter for it.

The only way to set the hyper builder via the public API is through the `build_with_conn_fn` internal function — which `build_http()` uses and passes `self.client_builder` through. So we need a different approach entirely.

**The actual working approach**: Implement `aws_smithy_runtime_api::client::http::HttpClient` trait directly. This bypasses the `pub(crate)` restriction entirely. The `HttpConnector` and `HttpClient` traits are public in `aws-smithy-runtime-api`:

```rust
use aws_smithy_runtime_api::client::http::{
    HttpClient, HttpConnector, HttpConnectorFuture, HttpRequest,
    SharedHttpClient, SharedHttpConnector,
};

struct H2cConnector {
    inner: hyper_util::client::legacy::Client<
        hyper_util::client::legacy::connect::HttpConnector,
        SdkBody,
    >,
}

impl HttpConnector for H2cConnector {
    fn call(&self, request: HttpRequest) -> HttpConnectorFuture {
        // Convert smithy HttpRequest → hyper Request, dispatch, convert back
    }
}
```

The implementation is roughly 80 lines. No patching of aws-smithy-http-client needed.

### Layer 3: Connection Pool Tuning (Important but secondary)

With HTTP/2 multiplexing, connection count matters less (100 streams per connection). But pool settings still matter:
- `pool_max_idle_per_host(16)` — keep 16 connections warm per endpoint
- `pool_idle_timeout(90s)` — don't kill idle connections too aggressively
- `http2_adaptive_window(true)` — let the flow control window grow

---

## Concrete Changes

### Change 1: Per-Endpoint S3 Client (in s3dlio)

**File**: `src/object_store.rs` — modify `S3ObjectStore`

```rust
pub struct S3ObjectStore {
    size_cache: Arc<ObjectSizeCache>,
    client: Arc<aws_sdk_s3::Client>,  // ← ADD: per-instance client
    endpoint_url: Option<String>,      // ← ADD: this instance's endpoint
}
```

**New constructor**:
```rust
impl S3ObjectStore {
    pub async fn for_endpoint(endpoint_url: &str) -> Result<Self> {
        let http_client = create_h2c_http_client()?; // from change 2
        let config = aws_config::defaults(BehaviorVersion::v2026_01_12())
            .endpoint_url(endpoint_url)
            .http_client(http_client)
            .load().await;
        let s3_config = aws_sdk_s3::config::Builder::from(&config)
            .force_path_style(true)
            .build();
        Ok(Self {
            size_cache: Arc::new(ObjectSizeCache::new(...)),
            client: Arc::new(Client::from_conf(s3_config)),
            endpoint_url: Some(endpoint_url.to_string()),
        })
    }
}
```

**Modified `put()`**: Use `self.client` directly instead of going through `build_ops_async()`:
```rust
async fn put(&self, uri: &str, data: Bytes) -> Result<()> {
    let (bucket, key) = parse_s3_uri(uri)?;
    self.client.put_object()
        .bucket(&bucket).key(&key)
        .body(data.into())
        .send().await?;
    Ok(())
}
```

### Change 2: h2c HTTP Client (in s3dlio)

**New file**: `src/h2c_client.rs` — implement `HttpConnector` directly

This bypasses the `pub(crate)` limitation entirely. The `HttpConnector` and `HttpClient` traits are public in `aws-smithy-runtime-api`:

```rust
use aws_smithy_runtime_api::client::http::{
    HttpClient, HttpConnector, HttpConnectorFuture, HttpRequest,
    SharedHttpClient, SharedHttpConnector,
};

struct H2cConnector {
    inner: hyper_util::client::legacy::Client<
        hyper_util::client::legacy::connect::HttpConnector,
        SdkBody,
    >,
}

impl HttpConnector for H2cConnector {
    fn call(&self, request: HttpRequest) -> HttpConnectorFuture {
        // Convert smithy HttpRequest → hyper Request, dispatch, convert back
    }
}
```

The implementation is ~80 lines of Rust. No patching of aws-smithy-http-client needed.

### Change 3: `store_for_uri()` Becomes Endpoint-Aware (in s3dlio)

When `store_for_uri("s3://10.9.0.17:9000/bucket/")` is called, it should detect the endpoint in the URI and create an `S3ObjectStore::for_endpoint("http://10.9.0.17:9000")` rather than using the global singleton.

### Change 4: MultiEndpointStore Creates Per-Endpoint Stores (in s3dlio)

This already happens conceptually — `MultiEndpointStore::from_config()` calls `store_for_uri()` per endpoint. After Change 3, each call creates an `S3ObjectStore` with its own `aws_sdk_s3::Client` pointing at the correct endpoint. No change needed in `MultiEndpointStore` itself — only in the store factory.

### Change 5: sai3-bench Changes (Minimal)

sai3-bench already uses `MultiEndpointStore` via config. Once s3dlio is fixed, sai3-bench just needs:
- YAML config with `multi_endpoint` section listing all 16 endpoints
- `concurrency: 100` (or more) to have enough tasks in flight

No code changes needed in sai3-bench if the s3dlio changes are done correctly.

---

## Summary Table

| Change | Where | Effort | Impact |
|---|---|---|---|
| 1. Per-endpoint `S3ObjectStore` | s3dlio `object_store.rs`, `s3_client.rs` | 1-2 days | **Critical** — enables real multi-endpoint |
| 2. h2c `HttpConnector` impl | s3dlio new `h2c_client.rs` | 1 day | **High** — 100× fewer TCP connections |
| 3. Endpoint-aware `store_for_uri()` | s3dlio `object_store.rs` | 0.5 day | Required for Change 1 |
| 4. MultiEndpointStore wiring | s3dlio `multi_endpoint.rs` | Minimal | Already works after Change 3 |
| 5. sai3-bench config | sai3-bench YAML | Minutes | Just config, no code |

---

## Key Dependency Versions (verified from Cargo.lock)

- `aws-sdk-s3` = 1.124.0
- `aws-config` = 1.8.14
- `aws-smithy-http-client` = 1.1.12
- `hyper` = 1.8.1
- `hyper-util` = 0.1.20
- `h2` = 0.4.13
