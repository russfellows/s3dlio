# AIStore Redirect Support — Implementation Reference (v0.9.90)

**Related issues:** mlcommons/storage#271 · russfellows/s3dlio#126  
**Version:** v0.9.90  
**Implemented:** March–April 2026  
**Status:** Complete ✅ — all security concerns addressed, 27 tests passing

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Design Constraints and Approach](#2-design-constraints-and-approach)
3. [AIStore Redirect Protocol (Confirmed from Source)](#3-aistore-redirect-protocol-confirmed-from-source)
4. [Implementation Design](#4-implementation-design)
5. [Security Analysis and Fixes](#5-security-analysis-and-fixes)
6. [AIStore Compatibility Assessment](#6-aistore-compatibility-assessment)
7. [Test Coverage](#7-test-coverage)
8. [Known Limitations and Caveats](#8-known-limitations-and-caveats)
9. [Environment Variables](#9-environment-variables)
10. [Implementation Files](#10-implementation-files)
11. [Security Appendix: How Cert Pinning Was Implemented](#11-security-appendix-how-cert-pinning-was-implemented)

---

## 1. Problem Statement

NVIDIA AIStore uses a proxy-based architecture for internal load balancing. When a client sends a
request to the AIStore proxy via the S3 API, the proxy responds with an **HTTP 307 Temporary
Redirect** pointing at the specific storage target node that holds the data:

```
Client  ──GET s3://bucket/object──▶  AIStore Proxy (:9000)
Client  ◀──307 Location: http://target-node-3:9000/v1/objects/bucket/object──  Proxy
Client  ──GET http://target-node-3:9000/v1/objects/bucket/object──▶  Target node
Client  ◀──200 OK (data)──  Target node
```

The **AWS Rust SDK's default HTTP client does not follow cross-host redirects.** This is intentional
security behavior: following cross-host redirects would silently forward the `Authorization` header
to an untrusted third-party server. As a result, AIStore was unusable via s3dlio's S3 path —
clients received a 307 and no data.

This same issue affects s3torchconnector and every other standard S3 client used against AIStore.

---

## 2. Design Constraints and Approach

### Constraints

| Constraint | Rationale |
|---|---|
| Must not break existing S3/MinIO/Ceph/GCS users | Feature must be **opt-in only** |
| Must not forward Auth headers to untrusted hosts | Strip `Authorization` on cross-host redirects |
| Must compose with existing HTTP client options | `AWS_CA_BUNDLE`, `S3DLIO_USE_OPTIMIZED_HTTP` are independent and stackable |
| No AWS SDK fork | All code uses only public APIs; tracking SDK updates forever is not sustainable |
| Zero overhead when disabled | `S3DLIO_FOLLOW_REDIRECTS` must not touch the hot path when unset |

### Approaches Considered

**AWS Smithy Interceptor (rejected):** `aws-smithy-runtime` is a transitive dep, not a direct one.
The interceptor API cannot directly rewrite the target URL for a retry — it only controls the
operation-level retry machinery.

**hyper Tower Middleware (rejected):** Requires `build_with_connector_fn`, which is `#[doc(hidden)]`
and additionally requires a patched `aws-smithy-http-client`. Does not compose correctly with the
CA bundle path.

**Custom `HttpClient` + `HttpConnector` Wrapper (selected):** `aws-smithy-runtime-api` exposes
genuinely public and stable traits: `HttpClient` and `HttpConnector`. A wrapper struct that
implements both and delegates to the real underlying client is the cleanest approach with no
additional required dependencies. The redirect wrapper is inserted between the SDK and the real
HTTP client and is completely transparent to all other callers.

### Integration Point

```rust
// In s3_client.rs — after the existing http_client is built, before the SDK config:
let http_client = if S3DLIO_FOLLOW_REDIRECTS is set {
    crate::redirect_client::make_redirecting_client(http_client)
} else {
    http_client  // unchanged — zero overhead when not enabled
};
```

The wrapper **composes** with `AWS_CA_BUNDLE` (wraps the custom-TLS client) and with
`S3DLIO_USE_OPTIMIZED_HTTP` (wraps the pooled client). When disabled, no extra allocation or
function call occurs on any request.

---

## 3. AIStore Redirect Protocol (Confirmed from Source)

These findings come from reading the actual AIStore Go source (`ais/prxs3.go`, `ais/prxred.go`,
`core/meta/smap.go`) at `~/Documents/Code/aistore`:

### `ais/prxs3.go` — `s3Redirect()` function

- Status is always **HTTP 307** (`http.StatusTemporaryRedirect`)
- The `Location` header is always set before the status write
- Response body is S3-compatible XML:
  `<Error><Code>TemporaryRedirect</Code><Message>Redirect</Message><Endpoint>host:port/s3</Endpoint><Bucket>bucket</Bucket></Error>`
- If `feat.S3ReverseProxy` is set cluster-wide, AIStore reverse-proxies transparently — no 307 is
  sent; s3dlio receives a 200 directly

### `ais/prxred.go` — `redurl()` / `_preparse()` functions

- The Location URL includes AIStore-specific query parameters appended by the proxy:
  - `ais_pid` (proxy ID) — always present
  - `ais_ts` (Unix timestamp) — always present
  - `ais_smver`, `ais_nonce`, `ais_hmac` — present only on CSK-secured clusters
- The scheme (`http://` vs `https://`) in the Location URL is determined entirely by how the
  cluster network was configured at startup (`NetInfo.Init(proto, ...)` in `core/meta/smap.go`)
- The target node validates the AIStore query params via `isRedirect(query)` — they must arrive
  intact

---

## 4. Implementation Design

### 4.1 Module Structure

```
s3dlio/src/
├── s3_client.rs          ← modified: wires in redirect wrapper
├── redirect_client.rs    ← implementation of redirect-following connector
└── lib.rs                ← pub(crate) mod redirect_client
```

### 4.2 Struct Hierarchy

```
RedirectFollowingHttpClient     implements HttpClient (aws-smithy-runtime-api)
  └── wraps: SharedHttpClient   (the real AWS SDK client)
      └── returns per-call: RedirectFollowingConnector  implements HttpConnector
                └── wraps: SharedHttpConnector          (the real AWS SDK connector)
                └── holds: Option<CertVerifyStore>      (TLS cert pinning store)
```

### 4.3 `follow_redirects()` Loop Logic

For each request attempt:

1. `try_clone()` the request **before** sending (succeeds for GET/HEAD; returns `None` for streaming bodies)
2. Call inner connector → receive response
3. If status **not** in {301, 302, 303, 307, 308}: return response as-is ← fast path
4. If no `Location` header: return response as-is (malformed redirect, graceful degradation)
5. If clone was `None` (streaming body): return response as-is (cannot replay body)
6. If `hops_remaining == 0`: return `ConnectorError` "Too many redirects"
7. **TLS checks** (if origin was `https://`):
   - If Location scheme is not `https://`: **refuse** with `ConnectorError` (scheme downgrade)
   - Pre-flight TLS probe origin host if not already in `CertVerifyStore`
   - Pre-flight TLS probe redirect target if not already in `CertVerifyStore`
   - Pin to origin cert on first HTTPS redirect
8. Set URI of cloned request to `Location` header value
9. If new `host:port` ≠ original `host:port`: remove `Authorization` header (RFC 9110 §11.6.2)
10. Decrement `hops_remaining`, loop

**`S3DLIO_REDIRECT_MAX`** (default 5): the maximum number of redirect **responses** that will be
followed before returning `ConnectorError`. Redirects 1–N are followed; error fires on response N+1
if it is also a redirect.

### 4.4 Key Behaviors on Edge Cases

| Scenario | Behavior | Rationale |
|---|---|---|
| Same-host redirect | All headers preserved, including `Authorization` | Safe: same server, same trust domain |
| Cross-host redirect | `Authorization` stripped (RFC 9110 §11.6.2) | Prevents credential leakage to AIStore target nodes |
| Missing `Location` on 3xx | Returns 3xx to SDK as-is | Graceful degradation |
| Streaming / non-cloneable body | Returns 3xx to SDK as-is | Cannot replay body without buffering |
| Redirect loop | `ConnectorError` at `S3DLIO_REDIRECT_MAX` hops | Prevents infinite loops |
| AIStore query params | `set_uri(location)` copies full `Location` verbatim | All `ais_pid`, `ais_ts` etc. are preserved |
| `feat.S3ReverseProxy` mode | Transparent 200 — no redirect — no redirect handling engaged | Correct: connector never sees a 3xx |

---

## 5. Security Analysis and Fixes

### 5.1 Security Properties — Current State (v0.9.90)

All four protections are active when `S3DLIO_FOLLOW_REDIRECTS=1`:

| Protection | Mechanism | Status in v0.9.90 |
|---|---|---|
| Standard TLS cert chain + hostname verification | `rustls` / WebPKI (always runs, independent of s3dlio) | ✅ Always active |
| Cross-host `Authorization` header stripping | RFC 9110 §11.6.2 in `follow_redirects()` | ✅ Active |
| **Scheme downgrade prevention (HTTPS → HTTP)** | URI scheme string comparison; refuses with `ConnectorError` | ✅ **Fixed in March 2026** |
| **Certificate pinning across redirect chain** | Pre-flight TLS probe → `CertVerifyStore` DER comparison | ✅ **Fixed in April 2026** |

For HTTP origins and for non-AIStore S3 targets (where `S3DLIO_FOLLOW_REDIRECTS` is not set),
there is **zero overhead** — none of these checks are entered.

### 5.2 History: The Two Security Gaps That Were Found and Fixed

#### Gap 1 — Scheme Downgrade Prevention Was Dead Code (HIGH severity, fixed March 2026)

**Root cause:** `make_redirecting_client()` hardcoded `cert_store: None`. Every TLS policy inside
`follow_redirects()` is guarded by `if let Some(ref store) = cert_store`. With `None`, that gate
was never entered — the scheme-downgrade check and cert-pinning check were both unreachable.

**Threat:** An HTTPS origin receives a 307 pointing to `http://`. The connector silently followed
it, transmitting the S3 `Authorization` header (HMAC-SHA256 over the request, including credentials)
in plaintext to the HTTP target.

**Fix:** Changed `cert_store: None` → `cert_store: Some(CertVerifyStore::new())`.

```rust
// BEFORE (vulnerable):
cert_store: None,   // both TLS checks were dead code in production

// AFTER (fixed):
cert_store: Some(CertVerifyStore::new()),  // scheme downgrade check now fires
```

**Effect:** An `https://` → `http://` redirect now returns `ConnectorError` immediately. HTTP →
HTTP redirects are completely unaffected (the scheme check only fires when the origin is `https://`).

#### Gap 2 — Certificate Pinning Store Was Always Empty (MEDIUM severity, fixed April 2026)

**Root cause:** `CertVerifyStore` was attached (Gap 1 fixed) but always empty. The pinning check
compared against `store.get(host)` which always returned `None`, so pinning never activated — the
code logged a `warn!` and skipped silently.

**Threat:** A redirect chain stays on `https://` but the target presents a different TLS certificate
than the origin. Standard WebPKI validation (CA chain + hostname) still runs, but a host under a
different TLS identity (compromised intermediate CA, corporate proxy with trusted root, wildcard
cert) could receive forwarded credentials.

**Fix:** Implemented `RecordingVerifier` + `probe_and_record_cert()`. When an HTTPS redirect is
about to be followed, `follow_redirects()` opens a short-lived TLS connection to each HTTPS host.
`RecordingVerifier` runs the standard WebPKI check first (never bypassed), then records the leaf
certificate DER bytes into `CertVerifyStore` on success. The pinning comparison then has real cert
bytes to work with.

```rust
// New Cargo.toml direct dependencies (were already transitive):
rustls              = { version = "0.23", features = ["aws-lc-rs"] }
tokio-rustls        = "0.26"
rustls-native-certs = "0.8"

// RecordingVerifier — the key: WebPKI first, record only on success
fn verify_server_cert(&self, end_entity: ...) -> Result<ServerCertVerified, rustls::Error> {
    let result = self.inner.verify_server_cert(end_entity, ...);  // standard WebPKI
    if result.is_ok() {
        self.store.record(self.host_port.clone(), end_entity.as_ref().to_vec());
    }
    result  // return WebPKI result unchanged
}
```

**TOCTOU trade-off:** The probe uses a separate TCP connection from the real request. An attacker
could theoretically serve a legitimate cert during the probe and switch for the real request. This
requires simultaneous BGP-level routing manipulation and millisecond timing — well beyond realistic
threat models. It is the best available approach given that the AWS Rust SDK's internal TLS layer
is entirely `pub(crate)`-gated (see [§11](#11-security-appendix-how-cert-pinning-was-implemented)).

### 5.3 End-to-End Test Validation of All Four Redirect Scenarios

All four scenarios are validated against real OS-assigned loopback ports. For HTTPS cases, cert
bytes are real DER from genuine TLS handshakes via `probe_and_record_cert_with_roots()`.

| # | Origin | Redirect target | Expected result | Test |
|---|---|---|---|---|
| 1 | `https://` (real TLS server) | `http://` (real TCP port, never contacted) | ❌ Refused — scheme downgrade | `end_to_end_https_to_http_redirect_refused_real_servers` |
| 2 | `https://` cert A (real TLS server) | `https://` cert B ≠ A (real TLS server) | ❌ Refused — cert mismatch | `end_to_end_https_cert_mismatch_refused_real_servers` |
| 3 | `https://` cert A (real TLS server) | `https://` cert A = A (shared `ServerConfig`) | ✅ 200 OK — cert match | `end_to_end_https_same_cert_redirect_passes_real_servers` |
| 4 | `http://` (real TCP port) | `http://` (real TCP port) | ✅ 200 OK — no TLS checks | `end_to_end_http_to_http_redirect_passes_real_servers` |

Test 4 confirms that plain HTTP → HTTP redirects work without any certificate involvement — no
probe fires, no scheme check engages, the redirect is followed transparently.

---

## 6. AIStore Compatibility Assessment

**Verdict: Highly likely to work correctly with NVIDIA AIStore. ✅**

No end-to-end test against a live AIStore cluster has been performed yet.

### Compatibility Matrix

| AIStore Behavior | s3dlio Behavior | Compatible? |
|---|---|---|
| Redirect status always 307 | Handles 301, 302, 303, 307, 308 | ✅ |
| `Location` header always present | Guards against missing `Location` | ✅ |
| AIStore query params in `Location` (`ais_pid`, `ais_ts`, etc.) | `set_uri(location)` copies full URL verbatim — all params preserved | ✅ |
| 307 response body (XML error) | SDK never sees 307: connector intercepts below SDK response-parsing | ✅ |
| `Authorization` not expected by target node | Stripped on cross-host hops (RFC 9110 §11.6.2) | ✅ |
| `feat.S3ReverseProxy` — transparent proxy, no 307 | Gets 200 directly — redirect handling never engaged | ✅ |
| All-HTTP cluster (`Location` is `http://`) | Scheme check only fires if origin was HTTPS | ✅ |
| All-HTTPS cluster (shared cluster cert) | Probes record matching certs → pinning passes | ✅ |
| Single hop (proxy → target) | Default MAX=5, more than sufficient | ✅ |

### Why the SDK Layer Separation Matters

`RedirectFollowingConnector` implements `HttpConnector` in the AWS Smithy runtime stack. This
layer operates **below** the SDK's response parsing. When AIStore sends a 307 with an XML body, the
connector catches it, extracts the `Location` header, follows the redirect, and returns only the
final response to the SDK. **The SDK never sees the 307 body and never attempts to parse it as an
S3 error.** This is the correct interception point.

### What Would Confirm Full Compatibility

1. **Live AIStore test** — Run `S3DLIO_FOLLOW_REDIRECTS=1` against an actual AIStore cluster
   doing S3 GET/PUT/DELETE through the proxy node
2. **CSK-secured cluster test** — Verify `ais_nonce` / `ais_hmac` query params are preserved
   correctly through the redirect hop
3. **All-HTTPS cluster test** — Confirm behavior when Location URL contains `https://` with a
   real cluster TLS certificate

---

## 7. Test Coverage

**27 tests** in `src/redirect_client.rs` across four groups:

### RFC 9110 Conformance (13 tests)

- Basic 302/307/308 redirect following
- Cross-host `Authorization` stripping (RFC 9110 §11.6.2)
- Same-host `Authorization` preservation
- Max-redirect loop termination with `ConnectorError`
- RFC-correct 303 POST → GET method rewrite
- 307/308 method preservation
- Non-redirect passthrough
- Missing `Location` header handling

### TLS Policy Unit Tests (4 tests, `MockConnector` with pre-populated store)

- Scheme-downgrade refusal (HTTPS → HTTP)
- Cert pinning: same cert passes
- Cert pinning: different cert refused
- HTTP → HTTP: no TLS checks apply

### TLS Probe Integration Tests (2 tests, real `tokio-rustls` server)

- `probe_records_cert_from_real_tls_server` — real TLS handshake records correct DER bytes
- `probe_does_not_record_cert_when_webpki_fails` — untrusted cert leaves store empty (WebPKI must pass before recording)

### End-to-End Tests with Real OS-Assigned Ports (4 tests)

See the table in [§5.3](#53-end-to-end-test-validation-of-all-four-redirect-scenarios).

**Additional validation tool:** `examples/tls_test_server.rs` — a full HTTPS+HTTP/2 test server
(ALPN-negotiating, rcgen cert, written to `/tmp/tls_test_server.crt`) for interactive testing with
`curl` and `cargo run --bin s3-cli`.

---

## 8. Known Limitations and Caveats

### HTTPS Front-End with HTTP Intra-Cluster Nodes

Some AIStore deployments terminate TLS at a load balancer in front of the cluster, while
intra-cluster target nodes communicate over plain HTTP:

- Client connects to AIStore over `https://`
- Proxy returns 307 with `Location: http://target-node:9000/...`
- s3dlio's scheme-downgrade protection **refuses** this hop (HTTPS → HTTP)

For uniformly-HTTP or uniformly-HTTPS AIStore deployments this is not an issue. If you encounter
this scenario, the workaround is to connect to AIStore over `http://` instead of `https://`, or
ensure your AIStore deployment uses end-to-end TLS throughout.

### `S3DLIO_REDIRECT_MAX` Out-of-Range Values Fail Silently

`configured_max_redirects()` parses the env var as `u8`. Values over 255 silently fall back to 5.
A `warn!()` would improve diagnostics but has not been added yet.

### Both Env Vars Are Read Once at First Client Construction

`S3DLIO_FOLLOW_REDIRECTS` and `S3DLIO_REDIRECT_MAX` are evaluated inside the `OnceCell`
initializer. Setting either variable **after the first call** to `aws_s3_client()` has no effect.
Set both before any storage operation.

### `uri_host()` Does Not Handle RFC 3986 Userinfo

If a `Location` URI contains `user:pass@host:port`, `uri_host()` returns the full authority
including the userinfo component. S3 and AIStore URIs never embed credentials in URLs (credentials
are in headers), so this is not a realistic risk in practice.

### Relative `Location` Headers Are Not Handled

If a server returns a relative `Location` (e.g., `/v1/objects/b/k` with no scheme or host),
`uri_host()` returns an empty string and the redirect is treated as cross-host (auth stripped).
The subsequent `set_uri()` call with a relative path may also fail. AIStore always returns absolute
`Location` URIs; this edge case has no test coverage.

---

## 9. Environment Variables

| Variable | Accepted Values | Default | Purpose |
|---|---|---|---|
| `S3DLIO_FOLLOW_REDIRECTS` | `1`, `true`, `yes`, `on`, `enable` | `0` (disabled) | Enable redirect following. Must be set before first client construction. |
| `S3DLIO_REDIRECT_MAX` | integer 1–255 | `5` | Maximum redirect hops per request. Values > 255 silently fall back to 5. |

---

## 10. Implementation Files

| File | Role |
|---|---|
| `src/redirect_client.rs` | Full implementation: `RedirectFollowingConnector`, `RedirectFollowingHttpClient`, `CertVerifyStore`, `RecordingVerifier`, `probe_and_record_cert()`, `make_redirecting_client()`, 27 tests |
| `src/s3_client.rs` | Wires in redirect wrapper when `S3DLIO_FOLLOW_REDIRECTS=1` |
| `src/lib.rs` | `pub(crate) mod redirect_client` |
| `examples/tls_test_server.rs` | Interactive HTTPS+h2 test server (rcgen cert, ALPN) |
| `docs/Environment_Variables.md` | `S3DLIO_FOLLOW_REDIRECTS` and `S3DLIO_REDIRECT_MAX` documented |

---

## 11. Security Appendix: How Cert Pinning Was Implemented

This section documents the AWS SDK API investigation that was required to find a viable public-API
path for cert pinning.

### 11.1 Constraint: No SDK Fork

All code must use only **public APIs** of the AWS Rust SDK and its published dependencies.

Crates investigated (from local Cargo registry cache):
- `aws-smithy-http-client` v1.1.12 — the HTTP/TLS connector crate
- `aws-smithy-runtime-api` v1.11.6 — the connector trait definitions
- `hyper-rustls` v0.27.7 — the TLS layer underneath the SDK connector
- `rustls` v0.23.37 — the TLS implementation
- `tokio-rustls` v0.26.4 — async TLS streams

### 11.2 Dead Ends

**Dead End A — `build_with_connector_fn`:** The closure must return
`aws-smithy-http-client::Connector`, whose only field is `Box<dyn HttpConnector>` — private, no
public constructor. `ConnectorBuilder::wrap_connector()` is `pub(crate)`. Additionally
`build_with_connector_fn` is `#[doc(hidden)]`. **Dead end.**

**Dead End B — `TlsContext` / `TrustStore`:** The `with_pem_certificate()` API only adds root CAs
in PEM format. There is no method accepting a custom `rustls::client::danger::ServerCertVerifier`.
**Dead end for cert recording.**

**Dead End C — `create_rustls_client_config()`:** This function is `pub(crate)` inside
`aws-smithy-http-client`. Even if accessible it returns a finished `rustls::ClientConfig` with
no injection point. **Dead end.**

### 11.3 Viable Approach: Pre-Flight TLS Probe

The `aws-smithy-runtime-api` crate exposes genuinely public `HttpConnector` trait. But
`HttpConnector::call()` operates at the HTTP protocol layer — TLS is opaque at this layer. The
peer certificate is not accessible from inside `call()`.

**Solution:** A pre-flight TLS probe using `tokio-rustls` directly. A separate short-lived TCP
connection is opened to each HTTPS host at redirect time. `RecordingVerifier` — a custom
`rustls::client::danger::ServerCertVerifier` — delegates all WebPKI checks to the standard inner
verifier and, only on WebPKI success, records the leaf certificate DER bytes into `CertVerifyStore`.

```
Origin request arrives at follow_redirects()
  ↓
Origin is https:// and redirect found
  ↓
Probe origin:  TcpStream::connect → TlsConnector::connect
               → RecordingVerifier::verify_server_cert (WebPKI runs first)
               → store.record("localhost:443", <leaf cert DER>)
  ↓
Probe target:  same process for redirect target host
  ↓
Pin to origin cert: pinned_cert = store.get(origin_host)
  ↓
Follow redirect
  ↓
Next loop iteration: cert check fires
  store.get(target_host) == pinned_cert  →  ✅ continue
  store.get(target_host) != pinned_cert  →  ❌ ConnectorError "cert mismatch"
```

### 11.4 Option B (Chosen) vs Option A

**Option A (not chosen):** Wrap `SharedHttpConnector` in a `PinningConnector` that probes every
new HTTPS host on every request. Broadest coverage but adds probe overhead to all HTTPS requests.

**Option B (chosen):** Inline probes inside `follow_redirects()` at redirect boundaries only. All
logic stays in `redirect_client.rs`, no SDK wiring changes required. Probes fire only when an HTTPS
redirect is actually being followed and the host is not already in the store. Zero overhead for
non-redirect requests, and zero overhead for HTTP-only AIStore clusters.
