# Per-Endpoint S3 Client + h2c Implementation Plan

**Date**: April 17, 2026  
**Target**: s3dlio v0.9.87 → sai3-bench  
**Goal**: 4 clients × 70,000 PUT/s × 1 KiB = 280,000 PUT/s across 16 S3 endpoints  
**Prerequisite**: Read `docs/http2-h2c-change-proposal.md` for full architectural analysis

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Dependency Changes](#2-dependency-changes)
3. [Code Removal](#3-code-removal)
4. [New Module: reqwest-based HTTP Client](#4-new-module-reqwest-based-http-client)
5. [Bug Fix 1: parse_s3_uri Drops Endpoint](#5-bug-fix-1-parse_s3_uri-drops-endpoint)
6. [Bug Fix 2: Global Client Singleton → Per-Endpoint Clients](#6-bug-fix-2-global-client-singleton--per-endpoint-clients)
7. [S3ObjectStore Becomes Endpoint-Aware](#7-s3objectstore-becomes-endpoint-aware)
8. [store_for_uri Factory Changes](#8-store_for_uri-factory-changes)
9. [s3_client.rs Refactor](#9-s3_clientrs-refactor)
10. [sai3-bench Changes](#10-sai3-bench-changes)
11. [Environment Variables](#11-environment-variables)
12. [Testing Plan](#12-testing-plan)
13. [Implementation Order](#13-implementation-order)

---

## 1. Executive Summary

There are **two bugs** and **one feature** blocking 280K PUT/s:

| Problem | Type | Impact |
|---------|------|--------|
| `parse_s3_uri()` discards endpoint from URI | Bug | All PUTs go to global endpoint, ignoring per-endpoint URIs |
| `static CLIENT: OnceCell<Client>` global singleton | Bug | One connection pool for all endpoints; can't do per-endpoint routing |
| No h2c (HTTP/2 cleartext) support | Feature | HTTP/1.1 needs 1 TCP connection per in-flight request; h2c multiplexes ~100 streams per connection |

The fix is architectural: **S3ObjectStore gets its own `aws_sdk_s3::Client`** configured for a specific endpoint, with an optional h2c transport layer built on **reqwest 0.13** (not custom hyper code). ~1,140 lines of dead custom HTTP code are removed.

---

## 2. Dependency Changes

### 2.1 Version Updates in `Cargo.toml`

```toml
# BEFORE → AFTER
aws-config         = "=1.8.14"    →  "=1.8.15"
aws-sdk-s3         = "=1.124.0"   →  "=1.129.0"
```

### 2.2 Add New Dependencies

```toml
# Move reqwest from optional to required, upgrade to 0.13
reqwest = { version = "0.13", features = ["rustls-tls", "http2"] }
```

**Note**: reqwest 0.13 is a major upgrade from the current optional 0.12. The `hickory-dns` feature is dropped — it's not needed for direct IP endpoints (which is our target: `http://10.9.0.x:9000`).

### 2.3 Dependencies That Can Be Removed

After the refactor, these are no longer directly imported anywhere:

```toml
# REMOVE these direct dependencies (they remain as transitive deps):
hyper = ...          # Only used in create_optimized_http_client (being replaced)
hyper-util = ...     # Only used in create_optimized_http_client (being replaced)
hyper-rustls = ...   # Not imported anywhere in src/
rustls = ...         # Not imported anywhere in src/ (only in comments)
webpki-roots = ...   # Not imported anywhere in src/
```

**IMPORTANT**: `aws-smithy-http-client` must be KEPT — it's used for the `AWS_CA_BUNDLE` TLS context path (line 313 of `s3_client.rs`). It provides `tls_context_from_pem()` and `tls::Provider::Rustls(CryptoMode::AwsLc)`. This is the CA bundle code path that must continue working.

**IMPORTANT**: `aws-smithy-runtime-api` must be KEPT — it provides the `HttpClient`, `HttpConnector`, `SharedHttpClient` traits used by `redirect_client.rs` and the new reqwest module.

### 2.4 Feature Flag Changes in `Cargo.toml`

```toml
# REMOVE these features:
experimental-http-client = []   # Dead — requires patched crate, never compiles
enhanced-http = ["dep:reqwest"] # Dead — reqwest becomes a required dependency

# ADD this feature:
h2c = []  # Enable HTTP/2 cleartext (h2c) transport for high-throughput PUT workloads
```

The `enhanced-http` feature guard on `pub mod http;` in `lib.rs` also needs removal (the module itself is deleted).

### 2.5 Final Dependency Block (HTTP-related section only)

```toml
# --- async / net stack ---
tokio = { version = "^1", features = ["full"] }
tokio-stream = { version = "^0.1", features = ["sync"] }
tokio-util = "^0.7"
async-stream = "^0.3"

# HTTP transport
reqwest = { version = "0.13", features = ["rustls-tls", "http2"] }
aws-smithy-http-client = { version = "^1", features = ["rustls-aws-lc"] }  # For AWS_CA_BUNDLE TLS context only
aws-smithy-runtime-api = "^1"

# REMOVED: hyper, hyper-util, hyper-rustls, rustls, webpki-roots
```

---

## 3. Code Removal

### 3.1 Delete Entire Files

| File | Lines | Reason |
|------|-------|--------|
| `src/http/client.rs` | 281 | EnhancedHttpClient, HttpClientConfig, HttpClientFactory — dead code behind `enhanced-http` feature |
| `src/http/mod.rs` | 9 | Module root for deleted http/ directory |
| `src/sharded_client.rs` | 373 | ShardedS3Clients — only used by range_engine.rs GET path, not in PUT path, has TODO comments |
| `src/performance/mod.rs` | 201 | PerformanceOptimizedClient — not used outside performance/ module |
| `src/performance/config.rs` | 276 | PerformanceConfig — not used by any external caller |

**Total removed**: ~1,140 lines

### 3.2 Verify No External Callers

Before deleting, verify these types have NO callers outside their own module:
- `EnhancedHttpClient` — only in `performance/mod.rs` behind `cfg(feature = "enhanced-http")`
- `HttpClientConfig` — only in `http/client.rs`
- `HttpClientFactory` — only in `http/client.rs`
- `ShardedS3Clients` — only in `range_engine.rs` (GET path, not PUT)
- `PerformanceOptimizedClient` — only in `performance/` module
- `PerformanceConfig` — only in `performance/` module

**Confirmed**: sai3-bench has ZERO imports of any of these types.

### 3.3 Update `src/lib.rs`

```rust
// REMOVE these lines:
#[cfg(feature = "enhanced-http")]
pub mod http;

pub mod performance;
pub mod sharded_client;
```

### 3.4 Update `src/s3_client.rs`

Remove the entire `create_optimized_http_client()` function (both `#[cfg]` variants, lines ~214-275). This is replaced by the new reqwest-based client builder in section 4.

Remove these imports:
```rust
// REMOVE:
use aws_smithy_http_client::Connector;  // Only used by experimental path
```

Keep these imports (needed for CA bundle path):
```rust
// KEEP:
use aws_smithy_http_client::{tls, Builder as HttpClientBuilder};
use aws_smithy_http_client::tls::rustls_provider::CryptoMode;
```

### 3.5 Delete `src/performance/` Directory

Delete the entire `src/performance/` directory. If any file in this directory is used by something else (double-check), extract only what's needed.

---

## 4. New Module: reqwest-based HTTP Client

### 4.1 Decision: Implement HttpClient Trait vs. Use Third-Party Crate

**Option A (Recommended): Implement `HttpClient` trait directly (~120 lines)**

Create `src/reqwest_client.rs` that implements `aws_smithy_runtime_api::client::http::HttpClient` by forwarding to a `reqwest::Client`. This is modeled on [aws-smithy-http-client-reqwest](https://crates.io/crates/aws-smithy-http-client-reqwest) v0.1.0 (117 SLoC, Apache-2.0) but with streaming response support.

**Option B: Use aws-smithy-http-client-reqwest crate (0 custom lines)**

Add `aws-smithy-http-client-reqwest = "0.1"` to Cargo.toml. Zero custom code but:
- Only 71 downloads, single author (bzp2010)
- Buffers entire response body in memory via `resp.bytes().await` (fine for 1KB PUTs, problematic for large GETs)
- 22 days old (published April 2026)

**Recommendation**: Option A. The trait implementation is small, we control it, and we can optimize for streaming responses on the GET path later. Net code reduction is still massive (~1,140 lines removed, ~120 added).

### 4.2 New File: `src/reqwest_client.rs`

```rust
//! Reqwest-backed HTTP client for the AWS SDK.
//!
//! Implements `aws_smithy_runtime_api::client::http::HttpClient` by delegating
//! to a `reqwest::Client`. This enables:
//! - h2c (HTTP/2 cleartext) via `reqwest::ClientBuilder::http2_prior_knowledge()`
//! - Custom connection pool tuning via reqwest's builder API
//! - TLS configuration via reqwest's built-in rustls support

use std::fmt;
use std::time::Duration;

use aws_smithy_runtime_api::client::http::{
    HttpClient, HttpConnector, HttpConnectorFuture, HttpConnectorSettings,
    SharedHttpConnector,
};
use aws_smithy_runtime_api::client::orchestrator::{HttpRequest, HttpResponse};
use aws_smithy_runtime_api::client::result::ConnectorError;
use aws_smithy_runtime_api::client::runtime_components::RuntimeComponents;
use aws_smithy_types::body::SdkBody;

/// A Smithy `HttpClient` backed by `reqwest::Client`.
#[derive(Debug, Clone)]
pub struct ReqwestHttpClient {
    client: reqwest::Client,
}

impl ReqwestHttpClient {
    pub fn new(client: reqwest::Client) -> Self {
        Self { client }
    }
}

impl HttpClient for ReqwestHttpClient {
    fn http_connector(
        &self,
        settings: &HttpConnectorSettings,
        _components: &RuntimeComponents,
    ) -> SharedHttpConnector {
        SharedHttpConnector::new(ReqwestHttpConnector {
            client: self.client.clone(),
            read_timeout: settings.read_timeout(),
        })
    }
}

#[derive(Debug)]
struct ReqwestHttpConnector {
    client: reqwest::Client,
    read_timeout: Option<Duration>,
}

impl HttpConnector for ReqwestHttpConnector {
    fn call(&self, req: HttpRequest) -> HttpConnectorFuture {
        let client = self.client.clone();
        let timeout = self.read_timeout;
        HttpConnectorFuture::new(async move {
            // Convert Smithy HttpRequest → reqwest::Request
            let http_req = req.try_into_http1x()
                .map_err(|e| ConnectorError::user(Box::new(e)))?;
            let (parts, body) = http_req.into_parts();

            let url: reqwest::Url = parts.uri.to_string().parse()
                .expect("valid URI from SDK");

            let mut builder = client.request(parts.method, url)
                .headers(parts.headers);

            // Set per-request timeout from SDK settings
            if let Some(t) = timeout {
                builder = builder.timeout(t);
            }

            // Attach body (for PUTs this is the object data)
            builder = builder.body(reqwest::Body::wrap_stream(
                body.into_data_stream()
            ));

            // Execute
            let resp = builder.send().await
                .map_err(|e| ConnectorError::other(Box::new(e), None))?;

            // Convert reqwest::Response → Smithy HttpResponse
            let status = aws_smithy_runtime_api::http::StatusCode::from(resp.status());
            let headers = resp.headers().clone();

            // Read response body (typically small for PUT responses)
            let body_bytes = resp.bytes().await
                .map_err(|e| ConnectorError::other(Box::new(e), None))?;

            let mut http_resp = HttpResponse::new(status, SdkBody::from(body_bytes));
            *http_resp.headers_mut() = aws_smithy_runtime_api::http::Headers::try_from(headers)
                .map_err(|e| ConnectorError::user(Box::new(e)))?;

            Ok(http_resp)
        })
    }
}
```

### 4.3 Client Builder Factory Function

Add to `src/reqwest_client.rs`:

```rust
/// Build a `SharedHttpClient` with the appropriate transport settings.
///
/// - If `S3DLIO_H2C=1`: uses HTTP/2 cleartext (h2c) — required for high PUT throughput
/// - Otherwise: uses HTTP/1.1 with connection pooling (default)
///
/// Connection pool settings are tuned for high-throughput workloads:
/// - 32 idle connections per host (enough for ~3200 concurrent HTTP/2 streams)
/// - 90s idle timeout
/// - TCP_NODELAY enabled
pub fn build_reqwest_http_client() -> reqwest::Client {
    let h2c = matches!(
        std::env::var("S3DLIO_H2C").unwrap_or_default().to_lowercase().as_str(),
        "1" | "true" | "yes" | "on" | "enable"
    );

    let mut builder = reqwest::Client::builder()
        .pool_max_idle_per_host(32)
        .pool_idle_timeout(Duration::from_secs(90))
        .tcp_nodelay(true);

    if h2c {
        tracing::info!("S3DLIO_H2C enabled — using HTTP/2 cleartext (h2c) transport");
        builder = builder.http2_prior_knowledge();
        // HTTP/2 tuning for high-throughput small PUTs
        builder = builder
            .http2_adaptive_window(true)
            .http2_keep_alive_interval(Duration::from_secs(30))
            .http2_keep_alive_while_idle(true);
    }

    builder.build().expect("reqwest client build should not fail")
}
```

### 4.4 Register in `src/lib.rs`

```rust
pub mod reqwest_client;
```

---

## 5. Bug Fix 1: parse_s3_uri Drops Endpoint

### 5.1 Current State

In `src/s3_utils.rs`:

```rust
// Line 219: parse_s3_uri_full() — CORRECT, returns (bucket, key, endpoint)
pub fn parse_s3_uri_full(uri: &str) -> Result<(String, String, Option<String>)>

// Line 279: parse_s3_uri() — BUG, calls parse_s3_uri_full() but discards endpoint
pub fn parse_s3_uri(uri: &str) -> Result<(String, String)> {
    let (bucket, key, _endpoint) = parse_s3_uri_full(uri)?;  // ← endpoint discarded!
    Ok((bucket, key))
}
```

### 5.2 Call Sites of parse_s3_uri()

| File | Line | Function | Context |
|------|------|----------|---------|
| `s3_utils.rs` | 780 | `s3_read_object_async()` | GET path |
| `s3_utils.rs` | 854 | `s3_range_read_object_async()` | Range GET path |
| `s3_utils.rs` | 1100 | `s3_list_keys_async()` | LIST path |
| `s3_utils.rs` | 1349 | `s3_delete_object_async()` | DELETE path |
| `s3_utils.rs` | 1442 | `s3_head_object_async()` | HEAD path |
| `s3_utils.rs` | 1499 | `put_object_uri_async()` | **PUT path** (critical) |

### 5.3 Fix Approach

**This bug becomes moot after the S3ObjectStore refactor** (Section 7). When `S3ObjectStore` holds its own `aws_sdk_s3::Client` configured for a specific endpoint, the PUT/GET/LIST/DELETE methods use `self.client` directly. The URI is parsed only for bucket+key — the endpoint is already baked into `self.client`.

However, the standalone `s3_utils.rs` functions still need fixing for callers that use them directly (outside ObjectStore):

**Option A (Minimal)**: Keep `parse_s3_uri()` as-is. The standalone functions in `s3_utils.rs` continue to use the global client. Only the `S3ObjectStore` path gets per-endpoint clients.

**Option B (Complete)**: Change all 6 call sites to use `parse_s3_uri_full()` and pass the endpoint to `build_ops_async()` which uses it to select the correct client. This requires `build_ops_async()` to accept an optional endpoint parameter.

**Recommendation**: Option A for the initial implementation. The `S3ObjectStore` path (used by sai3-bench via `MultiEndpointStore`) is the critical path for 280K PUT/s. The standalone `s3_utils.rs` functions can be fixed in a follow-up. The global client continues to work for single-endpoint use cases.

---

## 6. Bug Fix 2: Global Client Singleton → Per-Endpoint Clients

### 6.1 Current State

In `src/s3_client.rs` line 39:

```rust
static CLIENT: OnceCell<Client> = OnceCell::const_new();
```

All S3 operations go through `aws_s3_client_async()` which returns this single global `Client`. This client is configured once with `AWS_ENDPOINT_URL` and cannot be changed per-request.

### 6.2 Fix: Keep Global Client + Add Per-Endpoint Constructor

**Do NOT remove the global client.** It's used by:
- All `s3_utils.rs` standalone functions (~20 call sites)
- `multipart.rs` (1 call site)
- Single-endpoint use cases (most existing users)

**Instead, add a new async function** that creates a per-endpoint `aws_sdk_s3::Client`:

Add to `src/s3_client.rs`:

```rust
/// Create an S3 client configured for a specific endpoint URL.
///
/// Unlike `aws_s3_client_async()` which returns the global singleton, this
/// creates a NEW client with its own connection pool targeting the given endpoint.
/// Used by `S3ObjectStore::for_endpoint()` to achieve per-endpoint clients.
///
/// The `http_client` parameter allows injecting a custom HTTP transport
/// (e.g., reqwest-based h2c client). If `None`, uses the default SDK transport.
pub async fn create_s3_client_for_endpoint(
    endpoint_url: &str,
    http_client: Option<SharedHttpClient>,
) -> Result<Client> {
    dotenvy::dotenv().ok();

    if std::env::var("AWS_ACCESS_KEY_ID").is_err()
        || std::env::var("AWS_SECRET_ACCESS_KEY").is_err()
    {
        bail!("Missing AWS_ACCESS_KEY_ID or AWS_SECRET_ACCESS_KEY");
    }

    let region = RegionProviderChain::first_try(
        std::env::var("AWS_REGION").ok().map(Region::new),
    )
    .or_default_provider()
    .or_else(Region::new(DEFAULT_REGION));

    let mut loader = aws_config::defaults(aws_config::BehaviorVersion::v2026_01_12())
        .region(region)
        .endpoint_url(endpoint_url);

    // Apply timeout config
    let op_timeout = get_operation_timeout();
    let timeout_config = TimeoutConfig::builder()
        .connect_timeout(Duration::from_secs(5))
        .operation_timeout(op_timeout)
        .build();
    loader = loader.timeout_config(timeout_config);

    // Inject custom HTTP client if provided
    if let Some(client) = http_client {
        loader = loader.http_client(client);
    }

    let cfg = loader.load().await;
    let s3_config = aws_sdk_s3::config::Builder::from(&cfg)
        .force_path_style(true)
        .build();

    Ok(Client::from_conf(s3_config))
}
```

---

## 7. S3ObjectStore Becomes Endpoint-Aware

### 7.1 Current State (src/object_store.rs)

```rust
// ~Line 950 — stateless, no client, no endpoint
pub struct S3ObjectStore {
    size_cache: Arc<ObjectSizeCache>,
}
```

The `put()` method calls `s3_put_object_uri_async(uri, data)` which goes through the global client.

### 7.2 New S3ObjectStore Structure

```rust
pub struct S3ObjectStore {
    size_cache: Arc<ObjectSizeCache>,
    /// Per-endpoint S3 client. If `None`, falls back to the global client.
    client: Option<Arc<aws_sdk_s3::Client>>,
    /// The endpoint URL this store targets (for logging/debugging).
    endpoint_url: Option<String>,
}
```

### 7.3 New Constructor

```rust
impl S3ObjectStore {
    /// Create an S3ObjectStore with its own client for a specific endpoint.
    ///
    /// This is the key change enabling multi-endpoint PUT distribution.
    /// Each endpoint gets its own connection pool, avoiding the global singleton.
    pub async fn for_endpoint(endpoint_url: &str) -> Result<Self> {
        use crate::reqwest_client::{build_reqwest_http_client, ReqwestHttpClient};
        use aws_smithy_runtime_api::client::http::SharedHttpClient;

        let reqwest_client = build_reqwest_http_client();
        let http_client = SharedHttpClient::new(ReqwestHttpClient::new(reqwest_client));

        // Optionally wrap with redirect following
        let http_client = crate::redirect_client::maybe_wrap_redirecting(http_client);

        let s3_client = crate::s3_client::create_s3_client_for_endpoint(
            endpoint_url,
            Some(http_client),
        ).await?;

        Ok(Self {
            size_cache: Arc::new(ObjectSizeCache::new(/* appropriate size */)),
            client: Some(Arc::new(s3_client)),
            endpoint_url: Some(endpoint_url.to_string()),
        })
    }

    /// Create an S3ObjectStore that uses the global client (backward compatible).
    pub fn new() -> Self {
        Self {
            size_cache: Arc::new(ObjectSizeCache::new(/* appropriate size */)),
            client: None,
            endpoint_url: None,
        }
    }
}
```

### 7.4 Modified PUT Method

The `put()` implementation on `S3ObjectStore` should use `self.client` when available:

```rust
async fn put_impl(&self, uri: &str, data: bytes::Bytes) -> Result<()> {
    let (bucket, key) = parse_s3_uri(uri)?;

    if let Some(client) = &self.client {
        // Per-endpoint path — use our own client directly
        client.put_object()
            .bucket(&bucket)
            .key(&key)
            .body(aws_sdk_s3::primitives::ByteStream::from(data))
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("PUT failed: {e}"))?;
        Ok(())
    } else {
        // Legacy path — use global client via s3_utils
        s3_put_object_uri_async(uri, data).await
    }
}
```

Apply the same pattern for `get()`, `delete()`, `list()`, `head()` — use `self.client` when available, fall back to global functions otherwise.

### 7.5 Backward Compatibility

The existing `S3ObjectStore::boxed()` constructor (used by `store_for_uri()` for single-endpoint use) should continue to work with `client: None`, falling back to the global client. Only `for_endpoint()` creates per-endpoint clients.

---

## 8. store_for_uri Factory Changes

### 8.1 Current State (src/object_store.rs ~line 2566)

```rust
pub fn store_for_uri(uri: &str) -> Result<Box<dyn ObjectStore>> {
    if uri.starts_with("s3://") {
        Ok(S3ObjectStore::boxed())  // ← No endpoint awareness
    }
    // ... other backends
}
```

### 8.2 New store_for_uri with Endpoint Awareness

```rust
/// Create an ObjectStore for the given URI.
///
/// For s3:// URIs with an explicit endpoint (e.g., `s3://10.9.0.17:9000/bucket/`),
/// creates an S3ObjectStore with a per-endpoint client.
/// For plain s3:// URIs (e.g., `s3://bucket/key`), uses the global client.
pub async fn store_for_uri(uri: &str) -> Result<Box<dyn ObjectStore>> {
    if uri.starts_with("s3://") {
        // Check if URI has an explicit endpoint (host:port before bucket)
        let (_bucket, _key, endpoint) = parse_s3_uri_full(uri)?;
        if let Some(endpoint_url) = endpoint {
            // Per-endpoint client
            Ok(Box::new(S3ObjectStore::for_endpoint(&endpoint_url).await?))
        } else {
            // Global client (backward compatible)
            Ok(S3ObjectStore::boxed())
        }
    }
    // ... other backends unchanged
}
```

**IMPORTANT**: `store_for_uri()` becomes `async`. This affects callers — check all call sites and convert as needed. The main caller is `MultiEndpointStore::from_config()` which is already in an async context.

### 8.3 MultiEndpointStore Changes

`multi_endpoint.rs` `from_config()` calls `store_for_uri()` per endpoint. After making `store_for_uri()` async, update the call to `.await`. No other changes needed — the multi-endpoint logic (round-robin, least-connections) already works correctly once each store has its own client.

---

## 9. s3_client.rs Refactor

### 9.1 Summary of Changes

| What | Action |
|------|--------|
| `static CLIENT: OnceCell<Client>` | **KEEP** — used by legacy callers |
| `aws_s3_client_async()` | **KEEP** — used by `s3_utils.rs` standalone functions |
| `create_optimized_http_client()` (both `#[cfg]` variants) | **REMOVE** — replaced by reqwest client |
| `create_s3_client_for_endpoint()` | **ADD** — new per-endpoint factory (Section 6.2) |
| `S3DLIO_USE_OPTIMIZED_HTTP` env var handling | **REMOVE** — replaced by `S3DLIO_H2C` |
| `AWS_CA_BUNDLE` handling | **KEEP** — still needed for custom TLS certificates |
| `S3DLIO_FOLLOW_REDIRECTS` handling | **KEEP** — AIStore redirect support |
| `run_on_global_rt()` and `RT_HANDLE` | **KEEP** — useful infrastructure for sync→async bridge |

### 9.2 Updated aws_s3_client_async()

The global client factory can also use the reqwest-based transport for consistency. Modify the default path (no CA bundle, no optimized HTTP) to use reqwest:

```rust
// Inside aws_s3_client_async(), replace the S3DLIO_USE_OPTIMIZED_HTTP block:
// BEFORE:
match env::var("S3DLIO_USE_OPTIMIZED_HTTP")... {
    "true" | "1" => Some(create_optimized_http_client()?),
    _ => None,
}

// AFTER:
// Use reqwest-based client for all transports (supports h2c via S3DLIO_H2C)
let reqwest_client = crate::reqwest_client::build_reqwest_http_client();
let http_client = SharedHttpClient::new(
    crate::reqwest_client::ReqwestHttpClient::new(reqwest_client)
);
Some(http_client)
```

This unifies the transport layer — both global and per-endpoint clients use the same reqwest-based transport.

---

## 10. sai3-bench Changes

### 10.1 No Code Changes Required

sai3-bench uses `MultiEndpointStore` via YAML config. Once s3dlio is fixed, sai3-bench just needs configuration.

### 10.2 Example YAML Config for 16 Endpoints

```yaml
# 4 storage nodes × 4 connections each = 16 endpoints
multi_endpoint:
  strategy: round_robin
  endpoints:
    - "s3://10.9.0.17:9000/"
    - "s3://10.9.0.17:9001/"
    - "s3://10.9.0.17:9002/"
    - "s3://10.9.0.17:9003/"
    - "s3://10.9.0.18:9000/"
    - "s3://10.9.0.18:9001/"
    - "s3://10.9.0.18:9002/"
    - "s3://10.9.0.18:9003/"
    - "s3://10.9.0.19:9000/"
    - "s3://10.9.0.19:9001/"
    - "s3://10.9.0.19:9002/"
    - "s3://10.9.0.19:9003/"
    - "s3://10.9.0.20:9000/"
    - "s3://10.9.0.20:9001/"
    - "s3://10.9.0.20:9002/"
    - "s3://10.9.0.20:9003/"

concurrency: 256  # Enough tasks to saturate 16 endpoints
object_size: 1KiB
operation: put
```

### 10.3 sai3-bench Cargo.toml Update

After s3dlio is released with these changes (tag the new version), update sai3-bench:

```toml
# Use local path during development:
s3dlio = { path = "../s3dlio", features = ["full-backends"] }

# Use git tag for release:
# s3dlio = { git = "...", tag = "v0.9.87", features = ["full-backends"] }
```

---

## 11. Environment Variables

### 11.1 New Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `S3DLIO_H2C` | `0` | Enable HTTP/2 cleartext (h2c) transport. Values: `1`, `true`, `yes`, `on`, `enable` |

### 11.2 Removed Variables

| Variable | Replacement |
|----------|-------------|
| `S3DLIO_USE_OPTIMIZED_HTTP` | Removed. The reqwest client is always used. `S3DLIO_H2C` controls HTTP/2. |

### 11.3 Unchanged Variables

| Variable | Purpose |
|----------|---------|
| `AWS_ENDPOINT_URL` | Global S3 endpoint for the singleton client (unchanged) |
| `AWS_CA_BUNDLE` | Custom TLS CA certificate (unchanged) |
| `S3DLIO_FOLLOW_REDIRECTS` | AIStore redirect following (unchanged) |
| `S3DLIO_REDIRECT_MAX` | Max redirect hops (unchanged) |
| `AWS_REGION` | AWS region (unchanged) |

---

## 12. Testing Plan

### 12.1 Unit Tests

1. **reqwest_client.rs**: Test `build_reqwest_http_client()` with and without `S3DLIO_H2C`
2. **S3ObjectStore::for_endpoint()**: Test construction (may need mock endpoint)
3. **Existing tests**: Run `cargo test` — all existing tests must pass

### 12.2 Integration Tests

1. **Single-endpoint PUT** (regression):
   ```bash
   AWS_ENDPOINT_URL=http://localhost:9000 \
     sai3-bench run --config single_endpoint_test.yaml
   ```
   Must match existing throughput.

2. **Multi-endpoint PUT** (new):
   ```bash
   S3DLIO_H2C=1 \
     sai3-bench run --config multi_endpoint_16.yaml
   ```
   Each endpoint should show traffic in endpoint stats.

3. **h2c verification**:
   ```bash
   S3DLIO_H2C=1 \
     sai3-bench run --config single_endpoint_test.yaml
   ```
   Verify via `tcpdump` or Wireshark that connections use HTTP/2 (magic bytes `PRI * HTTP/2.0`).

4. **CA bundle** (regression):
   ```bash
   AWS_CA_BUNDLE=/path/to/ca.pem \
     sai3-bench run --config test.yaml
   ```
   Must still work.

5. **Redirect following** (regression):
   ```bash
   S3DLIO_FOLLOW_REDIRECTS=1 \
     sai3-bench run --config aistore_test.yaml
   ```
   Must still follow 307 redirects.

### 12.3 Build Verification

```bash
# Must compile with zero warnings:
cargo build
cargo clippy

# Must compile without default features:
cargo build --no-default-features --features s3,backend-aws

# Must compile with full backends:
cargo build --features full-backends
```

---

## 13. Implementation Order

Execute in this exact order. Each step should compile with zero warnings before proceeding.

### Step 1: Dependency Updates (Cargo.toml)
- Update `aws-config` = `"=1.8.15"`
- Update `aws-sdk-s3` = `"=1.129.0"`
- Change `reqwest` from optional 0.12 to required 0.13 with features `["rustls-tls", "http2"]`
- Remove `enhanced-http` feature (and its `["dep:reqwest"]`)
- Remove `experimental-http-client` feature
- Add `h2c = []` feature (empty for now — just a marker)
- Remove `hyper`, `hyper-util`, `hyper-rustls`, `rustls`, `webpki-roots` from `[dependencies]`
- **Run**: `cargo build` — fix any compile errors from version bumps

### Step 2: Code Removal
- Delete `src/http/` directory
- Delete `src/sharded_client.rs`
- Delete `src/performance/` directory
- Remove `pub mod http;` (with its `#[cfg]` guard), `pub mod performance;`, `pub mod sharded_client;` from `lib.rs`
- Remove `create_optimized_http_client()` (both variants) from `s3_client.rs`
- Remove `use aws_smithy_http_client::Connector;` import from `s3_client.rs`
- Remove `S3DLIO_USE_OPTIMIZED_HTTP` handling from `aws_s3_client_async()`
- **Run**: `cargo build` — must compile with zero warnings

### Step 3: Add reqwest_client.rs
- Create `src/reqwest_client.rs` with `ReqwestHttpClient`, `ReqwestHttpConnector`, and `build_reqwest_http_client()`
- Add `pub mod reqwest_client;` to `lib.rs`
- **Run**: `cargo build` — must compile

### Step 4: Wire reqwest into s3_client.rs
- Modify `aws_s3_client_async()` to use reqwest-based transport as default
- Add `create_s3_client_for_endpoint()` function
- **Run**: `cargo build && cargo test` — must pass

### Step 5: Make S3ObjectStore Endpoint-Aware
- Add `client: Option<Arc<aws_sdk_s3::Client>>` and `endpoint_url: Option<String>` fields
- Add `for_endpoint()` constructor
- Modify `put()`, `get()`, `delete()`, `list()`, `head()` to use `self.client` when available
- Maintain backward compatibility — `client: None` falls back to global
- **Run**: `cargo build && cargo test` — must pass

### Step 6: Update store_for_uri and MultiEndpointStore
- Make `store_for_uri()` async and endpoint-aware
- Update all callers of `store_for_uri()` (check `multi_endpoint.rs`, anywhere else)
- **Run**: `cargo build && cargo test` — must pass

### Step 7: Integration Testing
- Test with local MinIO or VAST endpoints
- Verify multi-endpoint distribution
- Verify h2c with `S3DLIO_H2C=1`
- Verify CA bundle and redirect still work

---

## Appendix A: Files Modified Summary

| File | Action | Lines Changed (est.) |
|------|--------|---------------------|
| `Cargo.toml` | Modify deps, features | ~30 |
| `src/lib.rs` | Remove 3 module declarations, add 1 | ~5 |
| `src/reqwest_client.rs` | **NEW** | ~120 |
| `src/s3_client.rs` | Remove create_optimized_http_client, add create_s3_client_for_endpoint | ~100 |
| `src/object_store.rs` | Add fields to S3ObjectStore, for_endpoint(), modify put/get/etc | ~150 |
| `src/s3_utils.rs` | No changes (parse_s3_uri fix deferred to follow-up) | 0 |
| `src/multi_endpoint.rs` | Make store_for_uri call async | ~10 |
| `src/redirect_client.rs` | No changes | 0 |
| `src/http/` | **DELETE** | -290 |
| `src/sharded_client.rs` | **DELETE** | -373 |
| `src/performance/` | **DELETE** | -477 |

**Net**: ~-955 lines (remove ~1,140, add ~385)

## Appendix B: Key Discovery — aws-smithy-http-client-reqwest

During research, we discovered [aws-smithy-http-client-reqwest](https://crates.io/crates/aws-smithy-http-client-reqwest) v0.1.0 (published ~22 days ago, 117 SLoC, Apache-2.0, by bzp2010). This crate does exactly what our `reqwest_client.rs` does — implements Smithy's `HttpClient` trait by forwarding to `reqwest::Client`.

We chose to implement the trait ourselves (Option A in Section 4.1) rather than take a dependency on this 71-download crate, but it validates the approach. The implementing agent may reconsider if the crate gains traction.

Usage would be:
```toml
aws-smithy-http-client-reqwest = "0.1"
```
```rust
use aws_smithy_http_client_reqwest::ReqwestHttpClient;
let client = reqwest::Client::builder().http2_prior_knowledge().build()?;
let http = ReqwestHttpClient::new(client);
let config = aws_config::defaults(BehaviorVersion::latest()).http_client(http).load();
```

## Appendix C: reqwest 0.13 h2c Confirmation

`reqwest::ClientBuilder::http2_prior_knowledge()` is available with the `http2` feature in reqwest 0.13.2. Its docs state: "Only use HTTP/2." This is h2c — HTTP/2 cleartext without TLS ALPN negotiation. Confirmed at [docs.rs/reqwest/0.13.2](https://docs.rs/reqwest/0.13.2/reqwest/struct.ClientBuilder.html#method.http2_prior_knowledge).
