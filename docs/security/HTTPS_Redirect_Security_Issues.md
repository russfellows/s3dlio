# HTTPS Redirect Security Issues in `redirect_client.rs`

**File:** `src/redirect_client.rs`  
**Component:** `RedirectFollowingConnector` / `make_redirecting_client()`  
**Date:** March 23, 2026  
**Status:** Item 1 fixed. Item 2 pending (see §5).

---

## 1. Background

`RedirectFollowingConnector` was added to support NVIDIA AIStore, which uses HTTP 307
redirects to route clients from a stateless proxy node to the storage node that holds
the object. The connector follows 3xx redirects while stripping the `Authorization`
header on cross-host hops (RFC 9110 §11.6.2).

Two additional TLS security policies were designed and their **policy logic was
implemented and fully tested**:

1. **Scheme downgrade prevention** — an HTTPS → HTTP redirect must be refused; following
   it would retransmit subsequent request headers (including S3 signing credentials) in
   plaintext over an unencrypted channel.
2. **Certificate pinning** — every host in the redirect chain must present the same leaf
   certificate as the first HTTPS host, preventing credential forwarding to a host under
   a different TLS identity (e.g., a compromised intermediate CA or rogue proxy).

Both policies gate on a `CertVerifyStore` that maps `"host:port"` → DER-encoded leaf
certificate bytes. The problem, documented here, is that in production the store was
never attached to the connector — so neither policy was active.

---

## 2. Root Cause

`make_redirecting_client()` is the factory that the rest of the codebase calls when
`S3DLIO_FOLLOW_REDIRECTS=1` is set. As of the initial implementation it hardcoded
`cert_store: None`:

```rust
// BEFORE (vulnerable):
pub(crate) fn make_redirecting_client(inner: SharedHttpClient) -> SharedHttpClient {
    SharedHttpClient::new(RedirectFollowingHttpClient {
        inner,
        max_redirects: configured_max_redirects(),
        cert_store: None,   // ← both TLS checks are dead code in production
    })
}
```

Every TLS security check inside `follow_redirects()` is guarded by:

```rust
if let Some(ref store) = cert_store {   // gate 1: is a store attached?
    ...
    if let Some(ref expected_cert) = pinned_cert {  // gate 2: is a cert pinned?
```

With `cert_store: None`, **gate 1 is never passed**. All TLS security logic is
unreachable in production.

The unit tests worked correctly because they bypass `make_redirecting_client()` entirely —
they call `follow_redirects()` directly with an explicit `Some(pre_populated_store)`,
simulating what a real TLS handshake would provide. The policy logic itself is correct; it
was simply never activated in the production code path.

---

## 3. What Is, and Is Not, Protected

| Protection | Mechanism | Active before fix | Active after fix (Item 1) |
|---|---|---|---|
| Standard TLS cert chain + hostname verification | `rustls` / WebPKI (always runs) | ✅ Yes | ✅ Yes |
| Cross-host `Authorization` header strip | RFC 9110 §11.6.2 in `follow_redirects()` | ✅ Yes | ✅ Yes |
| **Scheme downgrade prevention (HTTPS → HTTP)** | URI scheme string comparison | ❌ **No** | ✅ **Yes (fixed)** |
| **Certificate pinning across redirect chain** | `CertVerifyStore` + DER comparison | ❌ No | ⚠️ Partial (see §5) |

---

## 4. Security Risk Analysis — State Before the Fix

### 4.1 Scheme Downgrade (HIGH severity)

**Threat:** An HTTPS connection to the proxy receives a 307 redirect pointing to an
`http://` URI. The connector follows it silently.

**Impact:** The next request — which carries the S3 `Authorization` header (a full
HMAC-SHA256 signature over the request, including credentials) — is transmitted in
plaintext to the HTTP target. An attacker with network access can:

- Read the `Authorization` header and replay the signed request.
- Read any other sensitive headers (session tokens, custom auth).

**Exploitability:** Requires the attacker to either control the redirect response (MitM
on the HTTPS connection, or a compromised proxy) or influence the server's redirect
`Location` header. In a cloud or cluster environment with shared network fabrics, this
is a realistic threat.

**Severity:** HIGH — direct credential exposure.

### 4.2 Certificate Pinning (MEDIUM severity — defense in depth)

**Threat:** The redirect chain stays on `https://` but the redirect target presents a
different TLS certificate than the origin.

**Impact:** Credentials are forwarded to a host that, while TLS-protected, is under a
different TLS identity. The auth-stripping on cross-host hops mitigates this partially,
but same-host redirects (different paths, same `host:port`) preserve `Authorization`.

**Note:** Standard WebPKI validation still runs — the target's certificate still must
chain to a trusted root and match the hostname. The risk is elevated in environments
with:
- A compromised intermediate CA.
- An internal corporate proxy with a trusted root cert.
- A wildcard certificate covering both legitimate and malicious hosts.

**Severity:** MEDIUM — relevant in high-security environments; WebPKI provides a baseline.

---

## 5. The Two Checks Are Not Equal

### 5.1 Scheme Downgrade (easy — URI string only)

Does **not** require the `CertVerifyStore` to be populated. The check is a pure string
comparison on URI scheme tokens:

```rust
if uri_scheme(&this_uri) == "https" && uri_scheme(&location) != "https" {
    return Err(...); // refuse
}
```

**Fix:** Change `cert_store: None` → `cert_store: Some(CertVerifyStore::new())` in
`make_redirecting_client()`. The store can be empty; the scheme check fires regardless.

**→ This fix is implemented. See §6.**

### 5.2 Certificate Pinning (harder — requires TLS verifier integration)

Requires the `CertVerifyStore` to contain the actual DER-encoded certificate that the
server presented during the TLS handshake. After the easy fix, the store is attached
but **empty**, so pinning never activates (the code logs a `warn!` and skips pinning
gracefully — it does not error, preserving normal HTTPS→HTTPS redirect behavior).

To populate the store in production, a custom `rustls::client::ServerCertVerifier`
must be wired in so that the TLS handshake records the peer certificate into the store.

See §7 for the complete analysis of **which AWS SDK APIs were investigated**, which are
dead ends, and which viable public-API path exists for implementing this without
modifying or forking the AWS Rust SDK.

**This is the "future PR" work.** It is a plumbing task, not a design problem. The
policy logic is already correct and fully tested.

---

## 6. Fix Applied (Item 1 — Scheme Downgrade Prevention)

**Commit:** (this session, March 23, 2026)  
**Change:** `make_redirecting_client()` now attaches an empty `CertVerifyStore` instead
of `None`:

```rust
// AFTER (fixed):
pub(crate) fn make_redirecting_client(inner: SharedHttpClient) -> SharedHttpClient {
    SharedHttpClient::new(RedirectFollowingHttpClient {
        inner,
        max_redirects: configured_max_redirects(),
        cert_store: Some(CertVerifyStore::new()),  // scheme downgrade prevention is now active
    })
}
```

**Effect:**
- An `https://` → `http://` redirect is now refused with a `ConnectorError`.
- An `https://` → `https://` redirect logs a `warn!` that cert pinning cannot be applied
  (store is empty) and proceeds normally — no behavioral regression.
- HTTP → HTTP redirects are completely unaffected (the scheme check only fires when the
  origin is `https://`).

**Regression test added:** `security_gap_scheme_downgrade_not_refused_when_cert_store_none`  
This test was first written to document the gap (asserted error, got success → FAILED).
After the fix it was updated to use `Some(CertVerifyStore::new())` matching the corrected
production path, and now PASSES. Total redirect tests: **21 passed, 0 failed**.

---

## 7. Remaining Issue (Item 2 — Certificate Pinning): Full Analysis

| Item | Status |
|---|---|
| Scheme downgrade prevention (HTTPS → HTTP) | ✅ **Fixed** |
| Certificate pinning across redirect chain | ⚠️ **Pending** — future PR required |

**Risk if not done:** MEDIUM (see §4.2). WebPKI validation remains active. The primary
credential-leakage risk (scheme downgrade) is now blocked. Cert pinning adds defense in
depth for high-security environments with non-standard CA configurations.

---

### 7.1 Constraint: No SDK Fork

The implementation must use **only public APIs** of the AWS Rust SDK and its published
dependencies. Modifying or forking the AWS SDK is not acceptable — tracking SDK updates
forever is not sustainable. All custom code lives in s3dlio.

Crates investigated (from local Cargo registry cache):
- `aws-smithy-http-client` v1.1.12 — the HTTP/TLS connector crate
- `aws-smithy-runtime-api` v1.11.6 — the connector trait definitions
- `hyper-rustls` v0.27.7 — the TLS layer underneath the SDK connector
- `rustls` v0.23.31 — the TLS implementation
- `tokio-rustls` v0.26.2 — async TLS streams

---

### 7.2 Approaches That Were Investigated and Found to Be Dead Ends

#### Dead End A — `aws-smithy-http-client::Builder::build_with_connector_fn`

`aws-smithy-http-client::Builder` exposes:

```rust
// aws-smithy-http-client/src/client.rs line 935
// Note: also marked #[doc(hidden)]
pub fn build_with_connector_fn<F>(self, connector_fn: F) -> SharedHttpClient
where
    F: Fn(Option<&HttpConnectorSettings>, Option<&RuntimeComponents>) -> Connector
        + Send + Sync + 'static,
```

The closure must return `aws-smithy-http-client::Connector`. That type has a single
private field:

```rust
// aws-smithy-http-client/src/client.rs
pub struct Connector {
    adapter: Box<dyn HttpConnector>,  // private field — no public constructor
}
```

The only way to construct a `Connector` externally is via `ConnectorBuilder`. But
`ConnectorBuilder::wrap_connector()` — the method that wraps a custom hyper connector
(including a custom `hyper_rustls::HttpsConnector`) — is `pub(crate)`:

```rust
pub(crate) fn wrap_connector<C>(self, tcp_connector: C) -> Connector { ... }
```

**Result: inaccessible from s3dlio. Dead end.**

Additionally, `build_with_connector_fn` is marked `#[doc(hidden)]`, signaling that
Amazon does not consider it a stable public API — even if the `pub(crate)` barrier were
somehow bypassed, relying on it would be fragile.

#### Dead End B — `TlsContext` / `TrustStore`

`ConnectorBuilder::tls_context(TlsContext)` is a genuinely public API. But `TlsContext`
only allows adding root CA certificates in PEM format:

```rust
// aws-smithy-http-client/src/client/tls.rs
pub struct TrustStore { ... }
impl TrustStore {
    pub fn with_pem_certificate(self, pem: &[u8]) -> Result<Self, ...> { ... }
}
```

There is no method on `TlsContext`, `TrustStore`, or `ConnectorBuilder` that accepts a
custom `rustls::client::danger::ServerCertVerifier`. This path can customize which root
CAs are trusted, but cannot inject a recording verifier.

**Result: adds custom trust anchors only. Dead end for cert pinning.**

#### Dead End C — `create_rustls_client_config()`

Inside `aws-smithy-http-client/src/client/tls/rustls_provider.rs`, the function that
builds the `rustls::ClientConfig` from a `TlsContext` is:

```rust
pub(crate) fn create_rustls_client_config(
    crypto_mode: CryptoMode,
    tls_context: &TlsContext,
) -> rustls::ClientConfig { ... }
```

This is also `pub(crate)`. Even if it were accessible, it returns a finished
`rustls::ClientConfig` with no further injection point.

**Result: `pub(crate)`. Dead end.**

---

### 7.3 Viable Approach — Public APIs Only

The viable path uses `aws-smithy-runtime-api` (a different crate from
`aws-smithy-http-client`), which exposes a genuinely public and stable trait:

```rust
// aws-smithy-runtime-api/src/client/http.rs
pub trait HttpConnector: Send + Sync + fmt::Debug {
    fn call(&self, request: HttpRequest) -> HttpConnectorFuture;
}

pub struct SharedHttpConnector(Arc<dyn HttpConnector>);
impl SharedHttpConnector {
    pub fn new(connection: impl HttpConnector + 'static) -> Self { ... }
}

/// Returns a SharedHttpClient from a closure that selects a connector per request.
pub fn http_client_fn<F>(connector: F) -> SharedHttpClient
where
    F: Fn(&HttpConnectorSettings, &RuntimeComponents) -> SharedHttpConnector
         + Send + Sync + 'static;
```

This lets s3dlio implement `HttpConnector` on its own struct and register it with the
SDK **without touching any `pub(crate)` boundary**.

However, `HttpConnector::call()` operates at the **HTTP protocol layer** — it receives
an already-assembled `HttpRequest` and returns an `HttpResponse`. TLS is opaque at this
layer. Peer certificates are not accessible from inside `call()`.

The only way to read the peer certificate using public APIs is therefore a **pre-flight
TLS probe**: a separate TCP+TLS connection to the target host, used purely to extract
and record its leaf certificate, before the real HTTP request proceeds.

#### The Pre-Flight Probe Pattern

```rust
// Pseudo-code — all APIs used are public:
async fn probe_and_record_cert(
    host: &str,
    port: u16,
    store: &CertVerifyStore,
) -> Result<(), ConnectorError> {
    use tokio::net::TcpStream;
    use tokio_rustls::TlsConnector;
    use rustls::client::danger::ServerCertVerifier;

    // 1. Build a rustls::ClientConfig containing our RecordingVerifier.
    //    RecordingVerifier delegates to the standard WebPKI verifier and
    //    records the leaf cert into 'store' on success.
    let config = Arc::new(
        rustls::ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(
                RecordingVerifier::new(store.clone(), webpki_verifier())
            ))
            .with_no_client_auth()
    );

    // 2. Open a TCP connection and perform the TLS handshake.
    let tcp = TcpStream::connect((host, port)).await?;
    let connector = TlsConnector::from(config);
    let server_name = rustls::pki_types::ServerName::try_from(host)?;
    let tls_stream = connector.connect(server_name, tcp).await?;
    // Handshake complete — RecordingVerifier has already called store.record().

    // 3. Drop the probe connection. Only the cert bytes are retained.
    drop(tls_stream);
    Ok(())
}
```

The `RecordingVerifier::verify_server_cert()` standard signature is:

```rust
impl rustls::client::danger::ServerCertVerifier for RecordingVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        server_name: &ServerName<'_>,
        ocsp_response: &[u8],
        now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        // CRITICAL: standard WebPKI check runs first and MUST not be bypassed.
        let result = self.inner_verifier.verify_server_cert(
            end_entity, intermediates, server_name, ocsp_response, now,
        );
        // Only record if the standard check passed.
        if result.is_ok() {
            let key = format!("{server_name}:{port}");
            self.store.record(key, end_entity.as_ref().to_vec());
        }
        result
    }
    // ... other required methods delegate to self.inner_verifier
}
```

---

### 7.4 Two Placement Options

#### Option A — Custom `HttpConnector` Wrapper (broadest coverage)

Implement `HttpConnector` on a new struct that wraps the real `SharedHttpConnector` and
holds an `Arc<CertVerifyStore>`:

```rust
struct PinningConnector {
    inner: SharedHttpConnector,
    store: Arc<CertVerifyStore>,
}

impl HttpConnector for PinningConnector {
    fn call(&self, request: HttpRequest) -> HttpConnectorFuture {
        // If HTTPS and host not yet in store: run probe, populating the store.
        // Then forward to self.inner.call(request).
        // The redirect policy logic in follow_redirects() checks the store
        // against subsequent redirect targets.
    }
}
```

Register via:
```rust
let http_client = aws_smithy_runtime_api::client::http::http_client_fn(
    move |settings, components| SharedHttpConnector::new(PinningConnector { ... })
);
```

**Pros:** Intercepts all HTTPS requests made by the SDK, not just redirect hops. Populates
the store during the initial request, so it is ready when the first redirect fires.  
**Cons:** More code; adds a probe TCP connection to every new HTTPS host contacted.

#### Option B — Inline Probe in `follow_redirects()` (minimal scope)

Add the probe only at redirect boundaries, where the cert pinning check fires:

```rust
// Inside follow_redirects(), just before forwarding to location:
if uri_scheme(&location) == "https" {
    if let Some(ref store) = cert_store {
        let (host, port) = host_port(&location);
        if store.get(&format!("{host}:{port}")).is_none() {
            probe_and_record_cert(&host, port, store).await?;
        }
        // Now pinned_cert will be Some(...) and the comparison fires.
    }
}
```

**Pros:** Minimal change; all logic stays in `redirect_client.rs`; no changes to SDK
wiring or client construction; `cert_store` is already threaded through `follow_redirects()`.  
**Cons:** Only populates the store at redirect time; does not record the cert presented
by the *first* request in the chain (a probe for that host fires on the first redirect
to it). In practice this is fine because the pinning check compares the *redirect target*
against the already-recorded origin cert.

**Recommendation for the initial PR:** Option B. The scope is small, the diff is
readable, and the existing `CertVerifyStore` + `follow_redirects()` contract is
preserved exactly.

---

### 7.5 TOCTOU Trade-Off

The pre-flight probe opens a **different TCP connection** than the real HTTP request.
There is a theoretical window between probe completion and real request dispatch:
an attacker could briefly serve a legitimate certificate during the probe, then switch
to a malicious certificate for the real request.

Exploiting this requires simultaneous BGP-level routing manipulation, certificate
substitution, and millisecond-precision timing — well beyond typical threat models.

This is the inherent limitation of any approach that cannot intercept the real
request's own TLS handshake. The alternative (intercepting the real handshake) is
blocked by the `pub(crate)` boundaries documented in §7.2. The probe approach is the
correct engineering choice given the constraints and the actual threat model (redirect-
based credential forwarding).

---

### 7.6 Direct-Dependency Changes Needed

These crates are already resolved by Cargo as transitive dependencies via
`aws-smithy-http-client`. Making them direct dependencies locks in the version
constraint and makes the relationship visible in `Cargo.toml`:

```toml
# In s3dlio/Cargo.toml — promote transitive deps to direct:
rustls       = { version = "0.23", default-features = false }
tokio-rustls = { version = "0.26", default-features = false }
```

`hyper-rustls` is **not** needed as a direct dependency — the probe connects at the
`tokio-rustls` level (raw `TlsConnector`), which is lower than hyper.

---

### 7.7 Full Implementation Checklist (for future PR)

1. Add `rustls` v0.23 and `tokio-rustls` v0.26 as direct dependencies.
2. Implement `RecordingVerifier: rustls::client::danger::ServerCertVerifier`:
   - Delegates all standard methods to an inner WebPKI verifier.
   - Calls `store.record(host_port, cert_der)` on `verify_server_cert()` success.
   - **Must not bypass** the inner WebPKI check.
3. Implement `probe_and_record_cert(host, port, store) -> Result<(), ConnectorError>`:
   - Builds a `rustls::ClientConfig` using `RecordingVerifier`.
   - Opens `tokio::net::TcpStream` → `tokio_rustls::TlsConnector::connect()`.
   - Drops the stream; cert bytes are now in `store`.
4. (Option B) In `follow_redirects()`, before the redirect pinning check:
   - If target is `https://` and not yet in `store` → call `probe_and_record_cert()`.
5. Remove the `#[cfg_attr(not(test), allow(dead_code))]` annotations from
   `CertVerifyStore::record()` and `CertVerifyStore::new()` — they will now
   be called in production code paths.
6. Update `make_redirecting_client()` if needed (currently passes `Some(CertVerifyStore::new())`
   — no change required for Option B).
7. Write integration tests that use a real `tokio_rustls` listener to verify:
   - Cert is recorded by the probe before the redirect check fires.
   - A cert mismatch between origin and redirect target is refused.
   - A cert match between origin and redirect target proceeds normally.
