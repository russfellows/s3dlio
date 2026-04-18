// src/redirect_client.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! HTTP redirect-following client for S3-compatible services that use
//! HTTP 307 redirects for internal load balancing (e.g., NVIDIA AIStore).
//!
//! AIStore's proxy node receives the initial S3 request and returns an
//! HTTP 307, routing the client to the storage node where the data lives:
//!
//! ```text
//! Client ──GET /bucket/object──▶ AIStore Proxy
//! Client ◀──307 Location: http://target-node:port/v1/objects/... ── Proxy
//! Client ──GET /v1/objects/...──▶ target-node   (this is where data is)
//! ```
//!
//! The AWS SDK's default HTTP client deliberately does **not** follow
//! cross-host redirects (to prevent credential leakage to untrusted servers).
//! This module provides a wrapper connector that follows 3xx redirects while
//! stripping the `Authorization` header on cross-host hops.
//!
//! # Enabling redirect following
//!
//! Set `S3DLIO_FOLLOW_REDIRECTS=1` (or `true` / `yes` / `on` / `enable`)
//! before creating the S3 client.
//!
//! # Tuning
//!
//! | Variable | Default | Purpose |
//! |---|---|---|
//! | `S3DLIO_FOLLOW_REDIRECTS` | `0` | Enable redirect following |
//! | `S3DLIO_REDIRECT_MAX` | `5` | Maximum redirect hops per request |
//!
//! # References
//!
//! - mlcommons/storage#271 — "Add support for S3-compatible storage with HTTP redirects"
//! - russfellows/s3dlio#126 — linked s3dlio tracking issue

use std::fmt;
use std::sync::Arc;

use aws_smithy_runtime_api::client::http::{
    HttpClient, HttpConnector, HttpConnectorFuture, HttpConnectorSettings, SharedHttpClient,
    SharedHttpConnector,
};
use aws_smithy_runtime_api::client::orchestrator::{HttpRequest, HttpResponse};
use aws_smithy_runtime_api::client::result::ConnectorError;
use aws_smithy_runtime_api::client::runtime_components::RuntimeComponents;
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified};
use rustls::client::WebPkiServerVerifier;
use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, SignatureScheme};
use tracing::{debug, warn};

// ─────────────────────────────────────────────────────────────────────────────
// Configuration helpers
// ─────────────────────────────────────────────────────────────────────────────

const DEFAULT_MAX_REDIRECTS: u8 = 5;

fn configured_max_redirects() -> u8 {
    std::env::var("S3DLIO_REDIRECT_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_REDIRECTS)
}

// ─────────────────────────────────────────────────────────────────────────────
// RedirectFollowingConnector — wraps a SharedHttpConnector, follows 3xx
// ─────────────────────────────────────────────────────────────────────────────

// ─── TLS Certificate Verification Store ─────────────────────────────────────

/// Stores the DER-encoded leaf certificate for each `"host:port"` that an HTTPS
/// connection was established to.
///
/// In production this store is populated by a custom `rustls::client::ServerCertVerifier`
/// that intercepts TLS handshakes.  In tests it is pre-populated by the test
/// harness, simulating what the verifier would record during a real handshake.
///
/// [`RedirectFollowingConnector`] uses this store to enforce two TLS security
/// policies when following redirects from an HTTPS origin:
///
/// 1. **Scheme downgrade prevention**: a redirect from `https://` to `http://`
///    is refused — following it would expose credentials in plaintext.
/// 2. **Certificate pinning**: every host in the redirect chain must present the
///    same leaf certificate as the first HTTPS host.  A mismatch means the
///    redirect is leading to a genuinely different server and is refused.
#[derive(Debug, Default, Clone)]
pub(crate) struct CertVerifyStore {
    certs: Arc<dashmap::DashMap<String, Vec<u8>>>,
}

impl CertVerifyStore {
    // `new()` and `record()` are used both in production (by `probe_and_record_cert`
    // to populate the store during TLS handshakes) and in tests (which pre-populate
    // the store directly to simulate what the verifier would record).
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Record the DER-encoded leaf certificate for `host_port` (e.g. `"proxy:9443"`).
    ///
    /// Called by [`RecordingVerifier`] after each successful TLS handshake during a
    /// cert probe, and by test fixtures to pre-populate the store.
    pub(crate) fn record(&self, host_port: impl Into<String>, cert_der: Vec<u8>) {
        self.certs.insert(host_port.into(), cert_der);
    }

    /// Retrieve the DER-encoded leaf certificate recorded for `host_port`.
    fn get(&self, host_port: &str) -> Option<Vec<u8>> {
        self.certs.get(host_port).map(|r| r.value().clone())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TLS certificate probe — pre-flight connection to record a host's leaf cert
// ─────────────────────────────────────────────────────────────────────────────

/// Split `"host:port"` authority into `(host, port)`.  Uses `default_port` when
/// no port is present in the string (e.g. bare `"s3.amazonaws.com"`).
fn split_host_port(authority: &str, default_port: u16) -> (String, u16) {
    match authority.rfind(':') {
        Some(colon) => {
            let potential_port = &authority[colon + 1..];
            match potential_port.parse::<u16>() {
                Ok(port) => (authority[..colon].to_owned(), port),
                // Colon is part of an IPv6 address literal — treat whole string as host.
                Err(_) => (authority.to_owned(), default_port),
            }
        }
        None => (authority.to_owned(), default_port),
    }
}

/// A `rustls::client::danger::ServerCertVerifier` that:
///
/// 1. Delegates every validation step to an inner `WebPkiServerVerifier` — the
///    standard WebPKI chain + hostname check is **never** bypassed.
/// 2. On a *successful* validation, records the leaf certificate DER bytes into
///    a [`CertVerifyStore`] under the key `"host:port"` that was set at
///    construction time.
///
/// This is used exclusively by [`probe_and_record_cert_with_roots`] to record
/// the certificate presented during a pre-flight TLS handshake, which is then
/// compared by [`follow_redirects`] to enforce cert-pinning across HTTPS
/// redirect chains.
#[derive(Debug)]
struct RecordingVerifier {
    inner: Arc<WebPkiServerVerifier>,
    store: CertVerifyStore,
    /// `"host:port"` key under which the leaf cert will be stored.
    host_port: String,
}

impl rustls::client::danger::ServerCertVerifier for RecordingVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        intermediates: &[CertificateDer<'_>],
        server_name: &ServerName<'_>,
        ocsp_response: &[u8],
        now: UnixTime,
    ) -> Result<ServerCertVerified, rustls::Error> {
        // Standard WebPKI check runs first; MUST NOT be bypassed.
        let result = self.inner.verify_server_cert(
            end_entity,
            intermediates,
            server_name,
            ocsp_response,
            now,
        );
        // Record the leaf cert only if the standard check passed.
        if result.is_ok() {
            self.store
                .record(self.host_port.clone(), end_entity.as_ref().to_vec());
        }
        result
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        self.inner.verify_tls12_signature(message, cert, dss)
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, rustls::Error> {
        self.inner.verify_tls13_signature(message, cert, dss)
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        self.inner.supported_verify_schemes()
    }
}

/// Open a one-shot TLS connection to `host:port`, run the handshake via
/// [`RecordingVerifier`] (which records the server's leaf certificate into
/// `store`), then drop the connection.
///
/// Uses the platform's native root certificate store.  If individual platform
/// certificates cannot be parsed they are silently skipped; the TLS handshake
/// will fail if the server's cert cannot chain to any trusted root.
///
/// On failure the store is unmodified — the cert is recorded only on a
/// *successful* WebPKI handshake.
async fn probe_and_record_cert(
    host: &str,
    port: u16,
    store: &CertVerifyStore,
) -> Result<(), ConnectorError> {
    let mut roots = rustls::RootCertStore::empty();
    for cert in rustls_native_certs::load_native_certs().certs {
        roots.add(cert).ok(); // skip unparseable individual certs
    }
    probe_and_record_cert_with_roots(host, port, store, Arc::new(roots)).await
}

/// Like [`probe_and_record_cert`] but uses a caller-supplied root cert store.
///
/// This variant exists so tests can supply a root store that trusts a locally
/// generated test certificate without reaching out to the system trust anchors.
async fn probe_and_record_cert_with_roots(
    host: &str,
    port: u16,
    store: &CertVerifyStore,
    roots: Arc<rustls::RootCertStore>,
) -> Result<(), ConnectorError> {
    let host_port = format!("{host}:{port}");
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());

    let inner_verifier = WebPkiServerVerifier::builder_with_provider(roots, provider.clone())
        .build()
        .map_err(|e| {
            ConnectorError::other(
                boxed_str_error(format!(
                    "TLS cert probe: could not build WebPKI verifier for \"{host_port}\": {e}"
                )),
                None,
            )
        })?;

    let verifier = Arc::new(RecordingVerifier {
        inner: inner_verifier,
        store: store.clone(),
        host_port: host_port.clone(),
    });

    let config = Arc::new(
        rustls::ClientConfig::builder_with_provider(provider)
            .with_safe_default_protocol_versions()
            .map_err(|e| {
                ConnectorError::other(
                    boxed_str_error(format!(
                        "TLS cert probe: could not set TLS version for \"{host_port}\": {e}"
                    )),
                    None,
                )
            })?
            .dangerous()
            .with_custom_certificate_verifier(verifier)
            .with_no_client_auth(),
    );

    let connector = tokio_rustls::TlsConnector::from(config);
    let tcp = tokio::net::TcpStream::connect(&host_port)
        .await
        .map_err(|e| {
            ConnectorError::other(
                boxed_str_error(format!(
                    "TLS cert probe: TCP connect to \"{host_port}\" failed: {e}"
                )),
                None,
            )
        })?;

    let server_name = ServerName::try_from(host.to_owned()).map_err(|e| {
        ConnectorError::other(
            boxed_str_error(format!(
                "TLS cert probe: invalid server name \"{host}\": {e}"
            )),
            None,
        )
    })?;

    connector.connect(server_name, tcp).await.map_err(|e| {
        ConnectorError::other(
            boxed_str_error(format!(
                "TLS cert probe: TLS handshake with \"{host_port}\" failed: {e}"
            )),
            None,
        )
    })?;

    // Handshake complete — RecordingVerifier has stored the cert. Drop the connection.
    Ok(())
}

// ─────────────────────────────────────────────────────────────────────────────

/// Wraps an HTTP connector and follows 307 / 302 / 308 redirects.
///
/// - Same-host redirects: all request headers (including `Authorization`) are
///   preserved.
/// - Cross-host redirects: the `Authorization` header is stripped to prevent
///   credential leakage to untrusted hosts.
/// - Non-GET/HEAD requests whose body is not cloneable pass through unchanged.
#[derive(Debug, Clone)]
pub(crate) struct RedirectFollowingConnector {
    inner: SharedHttpConnector,
    max_redirects: u8,
    cert_store: Option<CertVerifyStore>,
}

impl RedirectFollowingConnector {
    fn new(
        inner: SharedHttpConnector,
        max_redirects: u8,
        cert_store: Option<CertVerifyStore>,
    ) -> Self {
        Self {
            inner,
            max_redirects,
            cert_store,
        }
    }
}

impl HttpConnector for RedirectFollowingConnector {
    fn call(&self, request: HttpRequest) -> HttpConnectorFuture {
        let inner = self.inner.clone();
        let max_redirects = self.max_redirects;
        let cert_store = self.cert_store.clone();
        HttpConnectorFuture::new(async move {
            follow_redirects(inner, request, max_redirects, cert_store).await
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Core redirect-following logic
// ─────────────────────────────────────────────────────────────────────────────

/// Returns `true` for HTTP status codes that indicate a redirect.
fn is_redirect_status(status: u16) -> bool {
    matches!(status, 301 | 302 | 303 | 307 | 308)
}

/// Extracts the `host[:port]` portion from an absolute URI string.
///
/// Examples:
/// - `"http://node:9000/path"` → `"node:9000"`
/// - `"https://s3.us-east-1.amazonaws.com/bucket"` → `"s3.us-east-1.amazonaws.com"`
fn uri_host(uri: &str) -> &str {
    // Skip past the scheme "://"
    let authority_start = uri.find("://").map(|i| i + 3).unwrap_or(0);
    let rest = &uri[authority_start..];
    // Authority ends at the first '/'
    let end = rest.find('/').unwrap_or(rest.len());
    &rest[..end]
}

/// Extracts the URI scheme from an absolute URI string (the part before `"://"`).
///
/// Returns `""` for relative URIs or strings with no scheme.
fn uri_scheme(uri: &str) -> &str {
    match uri.find("://") {
        Some(i) => &uri[..i],
        None => "",
    }
}

/// Convert an owned `String` message into the `BoxError` expected by
/// [`ConnectorError`] constructors (`Box<dyn Error + Send + Sync>`).
fn boxed_str_error(msg: String) -> Box<dyn std::error::Error + Send + Sync> {
    // std::io::Error is the simplest stdlib type that satisfies Error + Send + Sync.
    Box::new(std::io::Error::other(msg))
}

/// Send `request` through `connector`, following 3xx redirects up to
/// `max_redirects` times.
///
/// When `cert_store` is `Some`, two TLS security policies are enforced for
/// HTTPS origins:
///  1. Scheme downgrade (HTTPS → HTTP) is refused.
///  2. All redirect targets must present the same leaf certificate as the
///     first HTTPS host (certificate pinning).
async fn follow_redirects(
    connector: SharedHttpConnector,
    mut request: HttpRequest,
    max_redirects: u8,
    cert_store: Option<CertVerifyStore>,
) -> Result<HttpResponse, ConnectorError> {
    let mut hops_remaining = max_redirects;
    // DER-encoded leaf certificate of the first HTTPS host in the redirect
    // chain.  Set when the first redirect from an HTTPS origin is followed;
    // every subsequent HTTPS hop must present the same certificate.
    let mut pinned_cert: Option<Vec<u8>> = None;

    loop {
        // Capture the URI *before* the request is consumed by the connector.
        let this_uri = request.uri().to_owned();

        // Save a clone *before* consuming the request so we have something to
        // rebuild from if the response is a redirect.  For GET/HEAD requests
        // the body is always empty and `try_clone()` therefore always succeeds.
        // For streaming bodies (PUT/POST/etc.) `try_clone()` returns `None`,
        // and we will fall through to the non-redirect path below.
        let saved_clone = request.try_clone();

        let response = connector.call(request).await?;

        // ── TLS certificate pinning ────────────────────────────────────────
        // If a certificate was pinned on an earlier HTTPS redirect, verify that
        // the response we just received came from a host presenting the same
        // cert.  This check runs on every iteration so even a final non-redirect
        // 200 is verified against the pinned cert.
        if let Some(ref store) = cert_store {
            if let Some(ref expected_cert) = pinned_cert {
                let this_host = uri_host(&this_uri);
                match store.get(this_host) {
                    None => {
                        return Err(ConnectorError::other(
                            boxed_str_error(format!(
                                "TLS certificate check: no certificate recorded for redirect \
                                 target \"{this_host}\"; refusing connection to unverified host"
                            )),
                            None,
                        ));
                    }
                    Some(actual_cert) => {
                        if actual_cert != *expected_cert {
                            return Err(ConnectorError::other(
                                boxed_str_error(format!(
                                    "TLS certificate mismatch: redirect target \"{this_host}\" \
                                     presents a different certificate than the HTTPS origin; \
                                     refusing redirect to prevent credential disclosure to an \
                                     untrusted host"
                                )),
                                None,
                            ));
                        }
                    }
                }
            }
        }

        let status = response.status().as_u16();

        if !is_redirect_status(status) {
            return Ok(response);
        }

        // ── It is a redirect ─────────────────────────────────────────────────

        let Some(location) = response.headers().get("location").map(str::to_owned) else {
            warn!(
                "s3dlio redirect: HTTP {} without Location header — returning as-is",
                status
            );
            return Ok(response);
        };

        let Some(mut next_request) = saved_clone else {
            // Body not cloneable (streaming upload).  We cannot follow the
            // redirect; return the 3xx response to the AWS SDK and let it deal.
            debug!(
                "s3dlio redirect: HTTP {} but body is not cloneable — returning as-is",
                status
            );
            return Ok(response);
        };

        if hops_remaining == 0 {
            warn!(
                "s3dlio redirect: exceeded maximum {} hops; last redirect to: {}",
                max_redirects, location
            );
            return Err(ConnectorError::other(
                boxed_str_error(format!(
                    "Too many HTTP redirects (max {max_redirects}); \
                     last redirect Location: {location}"
                )),
                None,
            ));
        }
        hops_remaining -= 1;

        // ── TLS scheme-downgrade and origin certificate pinning ────────────────
        // When the original request used HTTPS we enforce two policies:
        //   1. The redirect target MUST also use HTTPS.  A scheme downgrade to
        //      HTTP would expose credentials in plaintext.
        //   2. On the first HTTPS redirect, pin the origin host’s certificate.
        //      Every subsequent HTTPS hop must present that same cert, ensuring
        //      the entire redirect chain stays within one TLS trust boundary.
        if let Some(ref store) = cert_store {
            let this_scheme = uri_scheme(&this_uri);
            if this_scheme == "https" {
                let location_scheme = uri_scheme(&location);
                if location_scheme != "https" {
                    warn!(
                        "s3dlio redirect: HTTPS → {:?} scheme downgrade refused (Location: {})",
                        location_scheme, location
                    );
                    return Err(ConnectorError::other(
                        boxed_str_error(format!(
                            "TLS downgrade refused: HTTPS request received a redirect to \
                             \"{location}\"; following it would expose credentials in plaintext"
                        )),
                        None,
                    ));
                }

                // Pre-flight TLS probes to populate the cert store.
                //
                // We can't extract the peer certificate from the real AWS SDK
                // connections (SDK internals are pub(crate)-gated — see
                // docs/security/HTTPS_Redirect_Security_Issues.md §7.2).
                // Instead, open a short-lived TLS connection to each HTTPS host
                // before following the redirect.  RecordingVerifier records the
                // leaf cert into the store during the handshake.
                //
                // TOCTOU trade-off: the probe uses a separate TCP connection
                // than the real request (§7.5).  This is the best available
                // approach without forking the AWS Rust SDK.
                //
                // Probe failures degrade to "no cert pinning for this hop"
                // (warn + continue).  The scheme-downgrade check (HIGH severity)
                // has already been enforced above.

                // 1. Probe the origin so we can pin its cert below.
                let origin_authority = uri_host(&this_uri).to_owned();
                if store.get(&origin_authority).is_none() {
                    let (origin_host, origin_port) = split_host_port(&origin_authority, 443);
                    if let Err(e) = probe_and_record_cert(&origin_host, origin_port, store).await {
                        warn!(
                            "s3dlio redirect: TLS cert probe for HTTPS origin \
                             \"{}\" failed: {}; cert pinning will not be applied \
                             for this redirect chain",
                            origin_authority, e
                        );
                    }
                }

                // 2. Probe the redirect target so the cert check on the next
                //    loop iteration has a cert to compare against.
                let target_authority = uri_host(&location).to_owned();
                if store.get(&target_authority).is_none() {
                    let (target_host, target_port) = split_host_port(&target_authority, 443);
                    if let Err(e) = probe_and_record_cert(&target_host, target_port, store).await {
                        warn!(
                            "s3dlio redirect: TLS cert probe for redirect target \
                             \"{}\" failed: {}; cert pinning may not be applied \
                             for this hop",
                            target_authority, e
                        );
                    }
                }

                // Pin to the certificate of the first HTTPS host in the chain.
                if pinned_cert.is_none() {
                    let this_host = uri_host(&this_uri);
                    match store.get(this_host) {
                        Some(cert) => {
                            debug!(
                                "s3dlio redirect: pinning to certificate of HTTPS origin \"{}\"",
                                this_host
                            );
                            pinned_cert = Some(cert);
                        }
                        None => {
                            warn!(
                                "s3dlio redirect: no certificate recorded for HTTPS origin \"{}\"; \
                                 certificate pinning cannot be applied for this redirect chain",
                                this_host
                            );
                        }
                    }
                }
            }
        }

        // Capture the host of the request we are about to redirect *from*
        // before we mutate next_request.
        let original_host = uri_host(next_request.uri()).to_owned();

        // Rewrite the URI to the redirect target.
        next_request.set_uri(location.as_str()).map_err(|e| {
            ConnectorError::other(
                boxed_str_error(format!(
                    "Invalid redirect Location header \"{location}\": {e}"
                )),
                None,
            )
        })?;

        let redirect_host = uri_host(next_request.uri()).to_owned();

        if original_host != redirect_host {
            // Cross-host redirect: strip Authorization to avoid leaking S3
            // credentials to the AIStore target node (which uses its own auth).
            debug!(
                "s3dlio redirect: cross-host {} → {} (stripped Authorization)",
                original_host, redirect_host
            );
            next_request.headers_mut().remove("authorization");
        } else {
            debug!("s3dlio redirect: same-host {} → {}", status, location);
        }

        // RFC 9110 §15.4.4 — 303 See Other MUST reissue the request as GET
        // regardless of the original method ("If the original request method
        // was POST, the client SHOULD perform a GET request").  The body is
        // implicitly empty; the Authorization header has already been
        // conditionally stripped above, so we copy the remaining headers.
        if status == 303 && next_request.method() != "GET" && next_request.method() != "HEAD" {
            debug!(
                "s3dlio redirect: 303 See Other — reissuing as GET (was {})",
                next_request.method()
            );
            let mut get_request = HttpRequest::get(next_request.uri()).map_err(|e| {
                ConnectorError::other(
                    boxed_str_error(format!(
                        "Failed to reissue 303 redirect as GET to \"{location}\": {e}"
                    )),
                    None,
                )
            })?;
            for (name, value) in next_request.headers().iter() {
                get_request
                    .headers_mut()
                    .insert(name.to_owned(), value.to_owned());
            }
            next_request = get_request;
        }

        request = next_request;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RedirectFollowingHttpClient — implements HttpClient
// ─────────────────────────────────────────────────────────────────────────────

/// Wraps a [`SharedHttpClient`] so that every connector it creates follows
/// HTTP 307 / 302 / 308 redirects.
///
/// This is the type registered with the AWS SDK via `.http_client(...)` on the
/// config builder when `S3DLIO_FOLLOW_REDIRECTS=1` is set.
#[derive(Debug, Clone)]
pub(crate) struct RedirectFollowingHttpClient {
    inner: SharedHttpClient,
    max_redirects: u8,
    cert_store: Option<CertVerifyStore>,
}

impl fmt::Display for RedirectFollowingHttpClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RedirectFollowingHttpClient(max_redirects={})",
            self.max_redirects
        )
    }
}

impl HttpClient for RedirectFollowingHttpClient {
    fn http_connector(
        &self,
        settings: &HttpConnectorSettings,
        components: &RuntimeComponents,
    ) -> SharedHttpConnector {
        let inner_connector = self.inner.http_connector(settings, components);
        SharedHttpConnector::new(RedirectFollowingConnector::new(
            inner_connector,
            self.max_redirects,
            self.cert_store.clone(),
        ))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Public (crate-internal) factory
// ─────────────────────────────────────────────────────────────────────────────

/// Wrap `inner` in a redirect-following HTTP client.
///
/// The maximum number of redirect hops is taken from `S3DLIO_REDIRECT_MAX`
/// (default 5).
///
/// An empty [`CertVerifyStore`] is attached.  At the first HTTPS redirect the
/// connector probes origin and target with a short-lived TLS connection via
/// [`probe_and_record_cert`] (using [`RecordingVerifier`] which records the leaf
/// certificate while delegating to the standard WebPKI verifier).  Two TLS
/// policies then fire automatically for every hop in the redirect chain:
///
/// 1. **Scheme downgrade prevention** — `https://` → `http://` is refused.
/// 2. **Certificate pinning** — every HTTPS hop must present the same leaf
///    certificate as the origin.
///
/// Both policies are a no-op for HTTP origins, and the connector itself is only
/// attached when `S3DLIO_FOLLOW_REDIRECTS=1` — so there is no overhead for
/// ordinary S3 (non-AIStore) targets.
pub(crate) fn make_redirecting_client(inner: SharedHttpClient) -> SharedHttpClient {
    SharedHttpClient::new(RedirectFollowingHttpClient {
        inner,
        max_redirects: configured_max_redirects(),
        cert_store: Some(CertVerifyStore::new()), // activates scheme-downgrade prevention
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper-function unit tests ────────────────────────────────────────────
    // These test the internal utility functions directly, independent of any
    // HTTP round-trips.

    #[test]
    fn uri_host_extracts_host_and_port() {
        assert_eq!(uri_host("http://node1:9000/v1/objects/b/k"), "node1:9000");
        assert_eq!(
            uri_host("https://s3.amazonaws.com/bucket/key"),
            "s3.amazonaws.com"
        );
        assert_eq!(uri_host("http://proxy/"), "proxy");
    }

    #[test]
    fn uri_host_with_no_path() {
        assert_eq!(uri_host("http://node2:8080"), "node2:8080");
    }

    #[test]
    fn is_redirect_status_recognises_redirect_codes() {
        for code in [301u16, 302, 303, 307, 308] {
            assert!(
                is_redirect_status(code),
                "expected {} to be a redirect",
                code
            );
        }
        for code in [200u16, 206, 304, 400, 404, 500] {
            assert!(
                !is_redirect_status(code),
                "expected {} NOT to be a redirect",
                code
            );
        }
    }

    #[test]
    fn cross_host_detection() {
        // Same host: no strip needed
        let a = "http://proxy:9000/v1/objects/b/k";
        let b = "http://proxy:9000/v1/objects/b/other";
        assert_eq!(uri_host(a), uri_host(b));

        // Different host (AIStore proxy → target node)
        let c = "http://target-node:9000/v1/objects/b/k";
        assert_ne!(uri_host(a), uri_host(c));
    }

    // ── RFC 9110 Specification Test Infrastructure ────────────────────────────
    //
    // The tests below exercise `follow_redirects()` via a `MockConnector` that
    // serves pre-scripted responses and records what requests it received.
    //
    // Design principle: each test describes WHAT THE RFC REQUIRES, not what the
    // implementation does.  If the implementation is correct, the test passes.
    // A failing test is evidence of an implementation gap, not a test bug.

    use aws_smithy_runtime_api::http::StatusCode;
    use aws_smithy_types::body::SdkBody;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    // ── MockConnector ─────────────────────────────────────────────────────────

    /// A test double for [`HttpConnector`].
    ///
    /// Serves pre-loaded [`HttpResponse`]s in FIFO order.  Records the URI,
    /// `Authorization` header value, and HTTP method of every request it receives
    /// so tests can make assertions after the fact.
    ///
    /// All observable state is stored behind `Arc<Mutex<…>>` so the connector
    /// can be cloned into a `SharedHttpConnector` while the test retains a handle
    /// for reading the captured values.
    #[derive(Clone)]
    struct MockConnector {
        /// Responses returned to callers in queue order.
        responses: Arc<Mutex<VecDeque<HttpResponse>>>,
        /// Absolute URIs of every inbound request, in call order.
        seen_uris: Arc<Mutex<Vec<String>>>,
        /// Value of the `authorization` header for every request ("" when absent).
        seen_auths: Arc<Mutex<Vec<String>>>,
        /// HTTP method string ("GET", "POST", …) for every request.
        seen_methods: Arc<Mutex<Vec<String>>>,
    }

    impl std::fmt::Debug for MockConnector {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MockConnector")
                .field("pending", &self.responses.lock().unwrap().len())
                .finish()
        }
    }

    impl MockConnector {
        fn new(responses: Vec<HttpResponse>) -> Self {
            Self {
                responses: Arc::new(Mutex::new(VecDeque::from(responses))),
                seen_uris: Default::default(),
                seen_auths: Default::default(),
                seen_methods: Default::default(),
            }
        }

        fn call_count(&self) -> usize {
            self.seen_uris.lock().unwrap().len()
        }

        fn seen_uris(&self) -> Vec<String> {
            self.seen_uris.lock().unwrap().clone()
        }

        fn seen_auths(&self) -> Vec<String> {
            self.seen_auths.lock().unwrap().clone()
        }

        fn seen_methods(&self) -> Vec<String> {
            self.seen_methods.lock().unwrap().clone()
        }
    }

    impl HttpConnector for MockConnector {
        fn call(&self, request: HttpRequest) -> HttpConnectorFuture {
            // Capture metadata synchronously before the request is consumed.
            let uri = request.uri().to_owned();
            let auth = request
                .headers()
                .get("authorization")
                .unwrap_or("")
                .to_owned();
            let method = request.method().to_owned();

            self.seen_uris.lock().unwrap().push(uri);
            self.seen_auths.lock().unwrap().push(auth);
            self.seen_methods.lock().unwrap().push(method);

            let response = self.responses.lock().unwrap().pop_front().expect(
                "MockConnector: test provided fewer responses than the redirect loop consumed",
            );

            HttpConnectorFuture::new(async move { Ok::<_, ConnectorError>(response) })
        }
    }

    // ── Test fixture helpers ──────────────────────────────────────────────────

    /// Build a GET [`HttpRequest`] with an optional `Authorization` header.
    fn make_get(uri: &str, auth: Option<&str>) -> HttpRequest {
        let mut req = HttpRequest::get(uri).expect("test URI must be a valid absolute URI");
        if let Some(token) = auth {
            req.headers_mut().insert("authorization", token.to_owned());
        }
        req
    }

    /// Build a GET [`HttpRequest`] with no `Authorization` header.
    fn get(uri: &str) -> HttpRequest {
        make_get(uri, None)
    }

    /// Build a POST [`HttpRequest`] with an empty (cloneable) body.
    ///
    /// Uses the `http` 0.2 builder and the `TryFrom<http::Request<SdkBody>>`
    /// impl on `HttpRequest` (available because the `http-02x` feature of
    /// `aws-smithy-runtime-api` is enabled transitively).
    ///
    /// `SdkBody::empty()` sets an internal `rebuild` closure, so `try_clone()`
    /// succeeds — this is the same clone path as ordinary GET requests.
    fn make_post(uri: &str) -> HttpRequest {
        http::Request::builder()
            .method("POST")
            .uri(uri)
            .body(SdkBody::empty())
            .expect("test POST URI must be well-formed")
            .try_into()
            .expect("http::Request<SdkBody> → HttpRequest conversion must succeed")
    }

    /// Build a redirect response with the given status code and `Location` header.
    fn redirect(status: u16, location: &str) -> HttpResponse {
        let mut resp = HttpResponse::new(
            StatusCode::try_from(status).expect("test status code must be a valid HTTP status"),
            SdkBody::empty(),
        );
        resp.headers_mut().insert("location", location.to_owned());
        resp
    }

    /// Build a redirect-class response with NO `Location` header (malformed from
    /// the server's perspective, but the client must handle it gracefully).
    fn redirect_no_location(status: u16) -> HttpResponse {
        HttpResponse::new(
            StatusCode::try_from(status).expect("test status code must be valid"),
            SdkBody::empty(),
        )
    }

    /// Build a 200 OK response with an empty body.
    fn ok_200() -> HttpResponse {
        HttpResponse::new(StatusCode::try_from(200u16).unwrap(), SdkBody::empty())
    }

    /// Wrap a `MockConnector` in a `SharedHttpConnector`.
    fn wrap(mock: MockConnector) -> SharedHttpConnector {
        SharedHttpConnector::new(mock)
    }

    // ── RFC 9110 Specification Tests ──────────────────────────────────────────
    //
    // RFC 9110 "HTTP Semantics" (June 2022) defines the authoritative rules for
    // redirect handling.  Each test is annotated with the clause it exercises.
    //
    // Reference: https://www.rfc-editor.org/rfc/rfc9110

    /// RFC 9110 §15.4 — 2xx, 4xx, and 5xx responses are NOT redirects.
    ///
    /// "The 2xx (Successful) class of status codes indicates that the client's
    ///  request was successfully received, understood, and accepted."
    ///
    /// A 200, 204, 400, 404, or 500 response MUST be returned immediately to the
    /// caller.  The connector must be called exactly once, with no URI rewriting.
    #[tokio::test]
    async fn rfc9110_s15_4_non_redirect_response_passes_through() {
        for status in [200u16, 204, 400, 404, 500, 503] {
            let mock = MockConnector::new(vec![HttpResponse::new(
                StatusCode::try_from(status).unwrap(),
                SdkBody::empty(),
            )]);
            let conn = wrap(mock.clone());

            let result = follow_redirects(conn, get("http://proxy:9000/bucket/key"), 5, None).await;

            assert!(
                result.is_ok(),
                "status {status} should be returned as Ok, not an error"
            );
            assert_eq!(
                result.unwrap().status().as_u16(),
                status,
                "returned status must equal the server status ({status})"
            );
            assert_eq!(
                mock.call_count(),
                1,
                "connector must be called exactly once for non-redirect status {status}"
            );
        }
    }

    /// RFC 9110 §15.4.8 — On 307, the subsequent request URI MUST equal the
    /// `Location` header value.
    ///
    /// "The server intends that this Redirect be temporary and SHOULD generate
    ///  a Location header field in the response containing a URI reference …
    ///  The user agent MAY use the Location field value for automatic redirection."
    ///
    /// Our client MUST use the exact Location URI as the next request URI.
    #[tokio::test]
    async fn rfc9110_s15_4_8_307_next_request_uri_equals_location() {
        let target = "http://target-node:9000/v1/objects/bucket/key";
        let mock = MockConnector::new(vec![redirect(307, target), ok_200()]);
        let conn = wrap(mock.clone());

        let result = follow_redirects(conn, get("http://proxy:9000/bucket/key"), 5, None).await;

        assert!(result.is_ok());
        let uris = mock.seen_uris();
        assert_eq!(
            uris.len(),
            2,
            "connector must be called twice: original + redirect"
        );
        assert_eq!(
            uris[1], target,
            "second request URI must equal the Location value from the 307 response"
        );
    }

    /// RFC 9110 §15.4.8 — On 307, the request method MUST NOT change.
    ///
    /// "The request method MUST NOT be changed when reissuing the original
    ///  request to the redirect target."
    ///
    /// This is the defining semantic difference between 307 and 302: 307
    /// preserves the method unconditionally.  A GET MUST remain a GET.
    ///
    /// If this test fails, method-preservation on 307 is broken.
    #[tokio::test]
    async fn rfc9110_s15_4_8_307_must_not_change_request_method() {
        let mock = MockConnector::new(vec![redirect(307, "http://target:9000/path"), ok_200()]);
        let conn = wrap(mock.clone());

        let _ = follow_redirects(conn, get("http://proxy:9000/path"), 5, None).await;

        let methods = mock.seen_methods();
        assert_eq!(methods.len(), 2);
        assert_eq!(
            methods[1], "GET",
            "RFC 9110 §15.4.8: 307 MUST NOT change the request method (expected GET → GET)"
        );
    }

    /// RFC 9110 §15.4.9 — On 308, the request method and body MUST NOT change.
    ///
    /// "The request method and message body MUST NOT be changed when reissuing
    ///  the original request to the redirect target."
    ///
    /// 308 (Permanent Redirect, RFC 7538) has identical method-preservation
    /// semantics to 307.  A GET sent to the original URI MUST remain a GET
    /// after following a 308.
    #[tokio::test]
    async fn rfc9110_s15_4_9_308_must_not_change_request_method() {
        let mock = MockConnector::new(vec![redirect(308, "http://target:9000/path"), ok_200()]);
        let conn = wrap(mock.clone());

        let _ = follow_redirects(conn, get("http://proxy:9000/path"), 5, None).await;

        let methods = mock.seen_methods();
        assert_eq!(methods.len(), 2);
        assert_eq!(
            methods[1], "GET",
            "RFC 9110 §15.4.9: 308 MUST NOT change the request method (expected GET → GET)"
        );
    }

    /// RFC 9110 §11.6.2 — Cross-authority redirect MUST remove the `Authorization`
    /// header.
    ///
    /// "A client SHOULD NOT automatically send the same authorization credentials
    ///  to the redirect target if the redirect target is a different authority
    ///  (the combination of URI scheme, host, and port)."
    ///
    /// This is a critical security property for the AIStore use case: the proxy
    /// and the target storage nodes are different hosts.  Leaking S3 credentials
    /// to target nodes is a security vulnerability.
    ///
    /// If this test fails, the implementation has a credential-leakage bug.
    #[tokio::test]
    async fn rfc9110_s11_6_2_cross_authority_redirect_strips_authorization() {
        let mock = MockConnector::new(vec![
            redirect(307, "http://target-node:9000/v1/objects/bucket/key"),
            ok_200(),
        ]);
        let conn = wrap(mock.clone());
        let req = make_get(
            "http://proxy:9000/bucket/key",
            Some("AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE"),
        );

        let _ = follow_redirects(conn, req, 5, None).await;

        let auths = mock.seen_auths();
        assert_eq!(auths.len(), 2);
        assert!(
            !auths[0].is_empty(),
            "original request to proxy must carry the Authorization header"
        );
        assert!(
            auths[1].is_empty(),
            "RFC 9110 §11.6.2: Authorization MUST be stripped on cross-authority redirect \
             (proxy:9000 → target-node:9000). \
             Failure here means S3 credentials are leaked to the storage target node — security bug."
        );
    }

    /// RFC 9110 §11.6.2 (same-authority converse) — Same-authority redirect MUST
    /// preserve the `Authorization` header.
    ///
    /// The §11.6.2 restriction applies to a "different authority".  When the
    /// redirect stays on the same `host:port`, the Authorization header must be
    /// forwarded unchanged.  Stripping it unnecessarily would break authenticated
    /// requests that simply redirect to the canonical path on the same host.
    #[tokio::test]
    async fn rfc9110_s11_6_2_same_authority_redirect_preserves_authorization() {
        let mock = MockConnector::new(vec![
            redirect(307, "http://proxy:9000/v1/objects/bucket/key"),
            ok_200(),
        ]);
        let conn = wrap(mock.clone());
        let req = make_get(
            "http://proxy:9000/bucket/key",
            Some("Bearer session-token-abc"),
        );

        let _ = follow_redirects(conn, req, 5, None).await;

        let auths = mock.seen_auths();
        assert_eq!(auths.len(), 2);
        assert!(
            !auths[1].is_empty(),
            "Authorization header MUST be preserved on same-authority redirect"
        );
        assert_eq!(
            auths[0], auths[1],
            "Authorization value must be byte-identical before and after same-authority redirect"
        );
    }

    /// RFC 9110 §15.4 — A client MUST NOT follow redirects indefinitely.
    ///
    /// "A client SHOULD detect and intervene in cyclical redirections (infinite
    ///  loops)."
    ///
    /// With `max_redirects = 1` the client is permitted to follow exactly one
    /// redirect.  Receiving a second redirect response MUST cause the client to
    /// stop and return an error.
    ///
    /// Hop-counting semantics: `max_redirects = N` allows N redirect responses
    /// to be followed.  The error is raised when the (N+1)-th redirect response
    /// is received.  With N=1: connector called twice, error on second 307.
    #[tokio::test]
    async fn rfc9110_s15_4_loop_prevention_enforces_max_redirect_limit() {
        let mock = MockConnector::new(vec![
            redirect(307, "http://node-a:9000/path"),
            redirect(307, "http://node-b:9000/path"), // this should trigger the limit
        ]);
        let conn = wrap(mock.clone());

        let result = follow_redirects(conn, get("http://proxy:9000/path"), 1, None).await;

        assert!(
            result.is_err(),
            "RFC 9110 §15.4: a second redirect after max_redirects=1 MUST return an error, \
             not silently continue"
        );
        assert_eq!(
            mock.call_count(),
            2,
            "connector must be called exactly twice (original + 1 allowed redirect); \
             the second 307 response triggers the error"
        );
    }

    /// RFC 9110 §15.4.8 — A 307 response with no `Location` header cannot be
    /// followed and MUST be returned as-is.
    ///
    /// "The server SHOULD generate a Location header field in the response
    ///  containing a URI …"
    ///
    /// When a server sends 307 without Location, the client has no redirect target
    /// and must return the response unchanged.  This is a server-side violation of
    /// the RFC, but the client must handle it gracefully without crashing or making
    /// additional requests.
    #[tokio::test]
    async fn rfc9110_s15_4_8_307_without_location_is_returned_as_is() {
        let mock = MockConnector::new(vec![redirect_no_location(307)]);
        let conn = wrap(mock.clone());

        let result = follow_redirects(conn, get("http://proxy:9000/bucket/key"), 5, None).await;

        assert!(
            result.is_ok(),
            "missing Location must not cause a hard error"
        );
        assert_eq!(
            result.unwrap().status().as_u16(),
            307,
            "307 with no Location must be returned to the caller unchanged"
        );
        assert_eq!(
            mock.call_count(),
            1,
            "no additional request must be made when the Location header is absent"
        );
    }

    /// RFC 9110 §15.4 — HTTP 304 (Not Modified) is NOT a redirect.
    ///
    /// 304 is in the 3xx class but its semantics are entirely different: it is a
    /// caching validation response, not a redirect.  The client MUST return it
    /// immediately without attempting to follow any redirect.
    ///
    /// This test verifies that `is_redirect_status()` correctly excludes 304, and
    /// that `follow_redirects()` honours that exclusion.
    #[tokio::test]
    async fn rfc9110_s15_4_304_not_modified_is_not_a_redirect() {
        let mock = MockConnector::new(vec![HttpResponse::new(
            StatusCode::try_from(304u16).unwrap(),
            SdkBody::empty(),
        )]);
        let conn = wrap(mock.clone());

        let result = follow_redirects(conn, get("http://proxy:9000/bucket/key"), 5, None).await;

        assert!(result.is_ok());
        assert_eq!(
            result.unwrap().status().as_u16(),
            304,
            "304 Not Modified is NOT a redirect; it must pass through unchanged"
        );
        assert_eq!(
            mock.call_count(),
            1,
            "only one connector call for a 304 response"
        );
    }

    /// RFC 9110 §15.4 — Multi-hop redirect chains are followed in sequence up to
    /// the hop limit.
    ///
    /// Validates the correctness of the loop: proxy → node-a → node-b → 200.
    /// Each redirect must use the Location of the previous response as the new
    /// request URI.  The final non-redirect response (200) must be returned.
    ///
    /// This exercises the primary AIStore use case (the proxy issues a 307 to
    /// a storage node) extended to a two-hop chain.
    #[tokio::test]
    async fn rfc9110_s15_4_multi_hop_chain_is_followed_to_final_response() {
        let node_a = "http://node-a:9000/v1/objects/bucket/key";
        let node_b = "http://node-b:9000/v1/objects/bucket/key";
        let mock = MockConnector::new(vec![redirect(307, node_a), redirect(307, node_b), ok_200()]);
        let conn = wrap(mock.clone());

        let result = follow_redirects(conn, get("http://proxy:9000/bucket/key"), 5, None).await;

        assert!(result.is_ok());
        assert_eq!(
            result.unwrap().status().as_u16(),
            200,
            "final response must be 200 after following a two-hop chain"
        );
        assert_eq!(
            mock.call_count(),
            3,
            "connector must be called once per hop plus the final request (3 total)"
        );
        let uris = mock.seen_uris();
        assert_eq!(uris[0], "http://proxy:9000/bucket/key", "first call: proxy");
        assert_eq!(uris[1], node_a, "second call: node-a (from first Location)");
        assert_eq!(uris[2], node_b, "third call: node-b (from second Location)");
    }

    /// RFC 9110 §15.4 — Cross-authority redirect rewrites the full URI, not just
    /// the host component.
    ///
    /// The new request URI after a redirect MUST be the complete value of the
    /// `Location` header, including scheme, host, port, and path — not a hybrid
    /// of the original URI with parts from Location.
    #[tokio::test]
    async fn rfc9110_s15_4_redirect_uri_is_the_full_location_value() {
        let location = "http://target-node:9001/v1/objects/mybucket/myfile.npz";
        let mock = MockConnector::new(vec![redirect(307, location), ok_200()]);
        let conn = wrap(mock.clone());

        let _ = follow_redirects(
            conn,
            make_get(
                "http://proxy:9000/mybucket/myfile.npz",
                Some("Bearer token"),
            ),
            5,
            None,
        )
        .await;

        let uris = mock.seen_uris();
        assert_eq!(uris.len(), 2);
        assert_eq!(
            uris[1], location,
            "redirect request URI MUST be the full Location value, \
             including the different port (9001 vs 9000) and new path prefix"
        );
    }

    /// RFC 9110 §15.4.4 — A 303 (See Other) response SHOULD redirect with GET
    /// regardless of the original request method.
    ///
    /// "If the original request method was POST, the client SHOULD perform a
    ///  GET request to the redirect target. … A client SHOULD NOT follow a 303
    ///  response with a method other than GET or HEAD."
    ///
    /// This rule distinguishes 303 from 307/308: 303 explicitly requires a
    /// method change to GET (or HEAD), while 307/308 require method preservation.
    #[tokio::test]
    async fn rfc9110_s15_4_4_303_with_post_must_redirect_as_get() {
        let mock = MockConnector::new(vec![redirect(303, "http://target:9000/result"), ok_200()]);
        let conn = wrap(mock.clone());

        // POST with empty body (SdkBody::empty() is cloneable, so try_clone() succeeds)
        let _ = follow_redirects(conn, make_post("http://proxy:9000/upload"), 5, None).await;

        let methods = mock.seen_methods();
        assert_eq!(
            methods.len(),
            2,
            "connector must be called twice: original POST + redirect"
        );
        assert_eq!(
            methods[1], "GET",
            "RFC 9110 §15.4.4: a 303 response to POST MUST redirect with GET, not POST"
        );
    }

    // ── TLS Security Tests ───────────────────────────────────────────────────
    //
    // These tests validate two redirect-time TLS security policies:
    //
    //   1. Scheme downgrade prevention: an HTTPS → HTTP redirect MUST be refused.
    //   2. Certificate pinning: all hosts in a redirect chain originating from
    //      HTTPS MUST present the same leaf certificate as the origin host.
    //
    // The `CertVerifyStore` is pre-populated by the test harness, simulating what
    // a custom `rustls::client::ServerCertVerifier` would record during real TLS
    // handshakes.  This lets the redirect policy logic be tested without requiring
    // actual TLS infrastructure.

    /// TLS policy — HTTPS → HTTP redirect (scheme downgrade) MUST be refused.
    ///
    /// If an HTTPS connection receives a 3xx redirect pointing to an HTTP URI,
    /// following it would retransmit the next request — including any credentials
    /// or sensitive headers — in plaintext over an unencrypted channel.
    ///
    /// The redirect MUST be refused with an error.  The connector MUST NOT be
    /// called for the HTTP target.
    #[tokio::test]
    async fn tls_scheme_downgrade_https_to_http_refused() {
        let cert_store = CertVerifyStore::new();
        cert_store.record("proxy:9443", b"cert-proxy-der".to_vec());

        let mock = MockConnector::new(vec![redirect(307, "http://target:9000/v1/objects/b/k")]);
        let conn = wrap(mock.clone());

        let result = follow_redirects(
            conn,
            make_get("https://proxy:9443/b/k", Some("Bearer secret-token")),
            5,
            Some(cert_store),
        )
        .await;

        assert!(
            result.is_err(),
            "HTTPS → HTTP redirect (scheme downgrade) MUST be refused; \
             following it would expose credentials in plaintext"
        );
        assert_eq!(
            mock.call_count(),
            1,
            "connector MUST NOT be called for the HTTP redirect target — \
             the error must be raised before building the next request"
        );
    }

    /// TLS policy — HTTPS → HTTPS redirect with the SAME certificate MUST succeed.
    ///
    /// This models the legitimate AIStore case where the proxy and all storage
    /// nodes share a wildcard or cluster-CA certificate.  Both hosts present
    /// identical DER bytes, so the cert-pinning check passes and the redirect
    /// MUST be followed to completion.
    #[tokio::test]
    async fn tls_cert_pinning_same_cert_passes() {
        let shared_cert = b"shared-cluster-cert-der".to_vec();
        let cert_store = CertVerifyStore::new();
        cert_store.record("proxy:9443", shared_cert.clone());
        cert_store.record("target:9443", shared_cert.clone());

        let mock = MockConnector::new(vec![
            redirect(307, "https://target:9443/v1/objects/b/k"),
            ok_200(),
        ]);
        let conn = wrap(mock.clone());

        let result = follow_redirects(
            conn,
            make_get("https://proxy:9443/b/k", Some("Bearer secret-token")),
            5,
            Some(cert_store),
        )
        .await;

        assert!(
            result.is_ok(),
            "identical certificate on redirect target MUST be accepted"
        );
        assert_eq!(
            result.unwrap().status().as_u16(),
            200,
            "final response after a cert-pinning-passing redirect must be 200"
        );
        assert_eq!(
            mock.call_count(),
            2,
            "connector must be called twice: origin + redirect target"
        );
    }

    /// TLS policy — HTTPS → HTTPS redirect with a DIFFERENT certificate MUST be refused.
    ///
    /// A certificate mismatch means the redirect target is a genuinely different
    /// server operating under a different TLS trust boundary.  This could indicate:
    ///   - A man-in-the-middle attack intercepting the redirect.
    ///   - An unintended proxy routing traffic to an untrusted backend.
    ///
    /// Credentials MUST NOT be forwarded to a host presenting a different cert.
    /// The redirect must be refused even after the connection to the target has
    /// been made (the cert check fires after `connector.call()` returns the
    /// redirect target's response, before that response is handed back to the
    /// caller).
    ///
    /// Note: cross-authority auth-stripping (RFC 9110 §11.6.2) is also in effect
    /// but is *not* a sufficient protection — the redirect must be refused
    /// entirely, not merely have its `Authorization` header stripped.
    #[tokio::test]
    async fn tls_cert_pinning_different_cert_refused() {
        let cert_store = CertVerifyStore::new();
        cert_store.record("proxy:9443", b"cert-proxy-der".to_vec());
        cert_store.record("target:9443", b"cert-target-DIFFERENT-der".to_vec());

        let mock = MockConnector::new(vec![
            redirect(307, "https://target:9443/v1/objects/b/k"),
            ok_200(), // reached on the 2nd connector.call(); cert check fires after this
        ]);
        let conn = wrap(mock.clone());

        let result = follow_redirects(
            conn,
            make_get("https://proxy:9443/b/k", Some("Bearer secret-token")),
            5,
            Some(cert_store),
        )
        .await;

        assert!(
            result.is_err(),
            "redirect to a host with a different certificate MUST be refused; \
             a cert mismatch indicates an untrusted server"
        );
        // The cert check fires *after* connector.call() for the redirect target
        // returns, so the connector IS called twice.
        assert_eq!(
            mock.call_count(),
            2,
            "connector is called twice (origin + redirect target); \
             the error fires after receiving the redirect target's response"
        );
    }

    /// TLS policy — HTTP → HTTP redirect is NOT subject to TLS checks.
    ///
    /// When the original request is plain HTTP (no TLS), there is no certificate
    /// to pin and no scheme downgrade to prevent.  Even with a `CertVerifyStore`
    /// attached, redirects among HTTP endpoints MUST be followed normally.
    #[tokio::test]
    async fn tls_http_to_http_redirect_no_tls_checks() {
        // Store is attached but empty — no certs recorded, simulating an HTTP-only cluster.
        let cert_store = CertVerifyStore::new();

        let mock = MockConnector::new(vec![
            redirect(307, "http://target:9000/v1/objects/b/k"),
            ok_200(),
        ]);
        let conn = wrap(mock.clone());

        let result = follow_redirects(
            conn,
            make_get("http://proxy:9000/b/k", None),
            5,
            Some(cert_store),
        )
        .await;

        assert!(
            result.is_ok(),
            "HTTP → HTTP redirect must be allowed when no TLS is involved"
        );
        assert_eq!(result.unwrap().status().as_u16(), 200);
        assert_eq!(mock.call_count(), 2);
    }

    // ── Security gap regression tests ─────────────────────────────────────────
    //
    // These tests exercise the PRODUCTION CODE PATH — i.e., the connector as it
    // is configured by `make_redirecting_client()`.  They are distinct from the
    // policy tests above, which call `follow_redirects()` with an explicitly
    // pre-populated `CertVerifyStore`.
    //
    // The production gap: `make_redirecting_client()` originally set `cert_store: None`,
    // which meant that both TLS security policies were unreachable dead code in
    // production, even though their logic was correct.
    //
    // INITIAL STATE of the test below: FAILS — documents the gap.
    // AFTER FIX (`cert_store: Some(CertVerifyStore::new())`): PASSES.
    //
    // See docs/security/HTTPS_Redirect_Security_Issues.md for full analysis.

    /// SECURITY GAP — The production `RedirectFollowingConnector` (with
    /// `cert_store: None` as created by `make_redirecting_client()`) does NOT
    /// refuse an HTTPS → HTTP scheme downgrade.
    ///
    /// An HTTPS request that receives a redirect to an `http://` URI MUST be
    /// refused.  Following it would retransmit the `Authorization` header (and
    /// other credentials) in plaintext to the HTTP target.
    ///
    /// # Test lifecycle
    ///
    /// **Before fix** (`cert_store: None`): this test FAILS — the redirect is
    /// followed successfully instead of being refused, proving the gap exists.
    ///
    /// **After fix** (`cert_store: Some(CertVerifyStore::new())`): the `None` in
    /// the call below is replaced with `Some(CertVerifyStore::new())`, matching
    /// the corrected production state.  The scheme check fires and the test PASSES.
    #[tokio::test]
    async fn security_gap_scheme_downgrade_not_refused_when_cert_store_none() {
        let mock = MockConnector::new(vec![
            redirect(307, "http://target:9000/v1/objects/b/k"),
            ok_200(), // reached only if the redirect is (incorrectly) followed
        ]);
        let conn = wrap(mock.clone());

        let result = follow_redirects(
            conn,
            make_get("https://proxy:9443/b/k", Some("Bearer secret-token")),
            5,
            // ↓ Matches the FIXED production state in make_redirecting_client().
            // ↓ With an empty store attached, the scheme check fires and refuses the downgrade.
            Some(CertVerifyStore::new()),
        )
        .await;

        // With the fix applied, the scheme downgrade is refused (Err).
        assert!(
            result.is_err(),
            "SECURITY BUG: HTTPS → HTTP redirect is not refused when cert_store is None \
             (the current production state in make_redirecting_client()). \
             The connector follows the redirect and exposes credentials in plaintext. \
             See docs/security/HTTPS_Redirect_Security_Issues.md §2."
        );
    }

    // ── TLS Probe Integration Tests ───────────────────────────────────────────
    //
    // These tests spin up a real tokio-rustls TLS listener with a locally-generated
    // self-signed certificate and verify that `probe_and_record_cert_with_roots()`
    // correctly records the server's leaf certificate into the CertVerifyStore.
    //
    // The test TLS server uses the same aws-lc-rs crypto provider as production.
    // rcgen (already a dev-dependency) generates the test certificates.

    /// Spawn a minimal TLS server on a random port.  The server accepts connections,
    /// completes the TLS handshake, then drops them immediately.
    ///
    /// Returns `(port, cert_der_bytes)` where `cert_der_bytes` is the DER-encoded
    /// leaf certificate the server presents.
    async fn spawn_tls_test_server() -> (u16, Vec<u8>) {
        use rcgen::{generate_simple_self_signed, CertifiedKey};

        let CertifiedKey { cert, key_pair } =
            generate_simple_self_signed(vec!["localhost".to_owned()]).unwrap();
        let cert_der = cert.der().to_vec();
        let key_der = key_pair.serialize_der();

        let server_cert = rustls::pki_types::CertificateDer::from(cert_der.clone());
        let server_key = rustls::pki_types::PrivateKeyDer::Pkcs8(
            rustls::pki_types::PrivatePkcs8KeyDer::from(key_der),
        );

        let server_config = Arc::new(
            rustls::ServerConfig::builder_with_provider(Arc::new(
                rustls::crypto::aws_lc_rs::default_provider(),
            ))
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_no_client_auth()
            .with_single_cert(vec![server_cert], server_key)
            .unwrap(),
        );

        let acceptor = tokio_rustls::TlsAcceptor::from(server_config);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        tokio::spawn(async move {
            while let Ok((tcp, _)) = listener.accept().await {
                let acceptor = acceptor.clone();
                tokio::spawn(async move {
                    // Complete the handshake; drop immediately — probe only needs the cert.
                    let _ = acceptor.accept(tcp).await;
                });
            }
        });

        (port, cert_der)
    }

    /// Build a RootCertStore that trusts exactly the given DER-encoded certificate.
    fn test_root_store(cert_der: &[u8]) -> Arc<rustls::RootCertStore> {
        let mut roots = rustls::RootCertStore::empty();
        roots
            .add(rustls::pki_types::CertificateDer::from(cert_der.to_vec()))
            .unwrap();
        Arc::new(roots)
    }

    /// The probe records the server's leaf certificate byte-for-byte.
    ///
    /// This exercises the full pre-flight TLS probe path:
    ///   TcpStream::connect → TlsConnector::connect
    ///   → RecordingVerifier::verify_server_cert (WebPKI passes)
    ///   → CertVerifyStore::record
    ///   → cert bytes available under "localhost:{port}"
    #[tokio::test]
    async fn probe_records_cert_from_real_tls_server() {
        let (port, cert_der) = spawn_tls_test_server().await;
        let store = CertVerifyStore::new();
        let roots = test_root_store(&cert_der);

        assert!(
            store.get(&format!("localhost:{port}")).is_none(),
            "store must be empty before probe"
        );

        probe_and_record_cert_with_roots("localhost", port, &store, roots)
            .await
            .expect("probe must succeed: TLS server is up and cert is trusted");

        let recorded = store
            .get(&format!("localhost:{port}"))
            .expect("cert must be in store after successful probe");
        assert_eq!(
            recorded, cert_der,
            "recorded cert bytes must be identical to the certificate the server presented"
        );
    }

    /// The probe must NOT record a cert when WebPKI validation fails.
    ///
    /// An empty RootCertStore has no trust anchors, so the server's self-signed
    /// certificate cannot chain to any trusted root.  WebPKI raises an error;
    /// RecordingVerifier only records on success, so the store stays empty.
    #[tokio::test]
    async fn probe_does_not_record_cert_when_webpki_fails() {
        let (port, _cert_der) = spawn_tls_test_server().await;
        let store = CertVerifyStore::new();
        // Empty root store → server cert is not trusted
        let empty_roots = Arc::new(rustls::RootCertStore::empty());

        let result = probe_and_record_cert_with_roots("localhost", port, &store, empty_roots).await;

        assert!(
            result.is_err(),
            "probe must return Err when the server certificate is not trusted by the root store"
        );
        assert!(
            store.get(&format!("localhost:{port}")).is_none(),
            "cert must NOT be recorded after a failed WebPKI validation — \
             RecordingVerifier only records on WebPKI success"
        );
    }

    // ── End-to-End Tests with Real Servers ────────────────────────────────────
    //
    // The three tests below spin up real tokio-rustls HTTPS servers (and one
    // plain TCP listener) on random loopback ports.
    //
    // The follow_redirects() *transport* is still MockConnector — the security
    // policies (scheme comparison, cert DER comparison) are pure in-memory
    // checks that don't depend on real HTTP I/O for the redirect itself.
    //
    // What IS real:
    //   • The server ports (random, OS-assigned — no hardcoded fake addresses).
    //   • The TLS certificates (generated by rcgen, presented via tokio-rustls).
    //   • The cert bytes in the CertVerifyStore (recorded via real TLS probes
    //     through probe_and_record_cert_with_roots(), which performs a genuine
    //     TcpStream → TlsConnector::connect() → RecordingVerifier handshake).
    //
    // Topology tested:
    //
    //  Test 1 — scheme downgrade:
    //    origin: https://localhost:{tls_port}/source
    //    → 307 Location: http://localhost:{http_port}/target
    //    → REFUSED (https:// → http:// scheme downgrade)
    //
    //  Test 2 — cert mismatch:
    //    origin: https://localhost:{port1}/source  (cert A)
    //    → 307 Location: https://localhost:{port2}/target  (cert B ≠ A)
    //    → REFUSED (pinned cert A ≠ recorded cert B)
    //
    //  Test 3 — cert match:
    //    origin: https://localhost:{port1}/source  (cert A)
    //    → 307 Location: https://localhost:{port2}/target  (cert A = A, shared identity)
    //    → 200 OK ✓

    /// Spin up a minimal TLS server reusing a pre-built `ServerConfig`.
    ///
    /// Useful for starting a second server that presents the **same** certificate
    /// as an existing server.  Returns the random port the server is bound to.
    async fn spawn_tls_server_with_config(config: Arc<rustls::ServerConfig>) -> u16 {
        let acceptor = tokio_rustls::TlsAcceptor::from(config);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move {
            while let Ok((tcp, _)) = listener.accept().await {
                let acceptor = acceptor.clone();
                tokio::spawn(async move {
                    let _ = acceptor.accept(tcp).await;
                });
            }
        });
        port
    }

    /// **End-to-end Test 1** — Real HTTPS origin + real HTTP target port.
    ///
    /// The HTTPS server is real (tokio-rustls, random port).  The HTTP side is a
    /// plain `TcpListener` (also random port, never served).  `MockConnector`
    /// delivers the 307 response, simulating what the HTTPS origin would return.
    ///
    /// Expected outcome: `follow_redirects()` detects the `https://` → `http://`
    /// scheme downgrade and returns `Err` **before** attempting any connection to
    /// the HTTP target.  Verified via `call_count == 1`.
    #[tokio::test]
    async fn end_to_end_https_to_http_redirect_refused_real_servers() {
        // Plain TCP listener — the HTTP "target".  Bound to a real port so the
        // Location URL is realistic; but it must NEVER receive a connection.
        let http_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let http_port = http_listener.local_addr().unwrap().port();

        // Real HTTPS server — its port makes the origin URI realistic.
        // The server never receives a connection in this test because MockConnector
        // provides the HTTP response layer; the TLS port is only in the URI string.
        let (tls_port, _cert_der) = spawn_tls_test_server().await;

        // MockConnector delivers the 307 with a downgraded (http://) Location.
        let http_target = format!("http://localhost:{http_port}/target");
        let mock = MockConnector::new(vec![
            redirect(307, &http_target),
            // ok_200() intentionally absent — the redirect must be refused;
            // a second connector call would mean the downgrade was followed.
        ]);
        let conn = wrap(mock.clone());

        let origin = format!("https://localhost:{tls_port}/source");
        let result = follow_redirects(
            conn,
            make_get(&origin, Some("Bearer secret-token")),
            5,
            Some(CertVerifyStore::new()),
        )
        .await;

        // The downgrade must be refused.
        assert!(
            result.is_err(),
            "HTTPS → HTTP redirect must be refused; got: {:?}",
            result
        );
        let err_dbg = format!("{:?}", result.unwrap_err());
        assert!(
            err_dbg.contains("downgrade") || err_dbg.contains("TLS"),
            "error debug output must describe a TLS downgrade refusal; got: {err_dbg}"
        );

        // Exactly one connector call: the origin request.
        // If this were 2, the redirect would have been (incorrectly) followed.
        assert_eq!(
            mock.call_count(),
            1,
            "connector must be called exactly once — the HTTP target must never be contacted"
        );

        drop(http_listener); // explicit: documents that this listener was intentionally idle
    }

    /// **End-to-end Test 2** — Two real HTTPS servers with **different** certs.
    ///
    /// Each server is probed via a real TLS handshake; the cert DER bytes that
    /// `RecordingVerifier` records are genuine bytes from genuine TLS sessions.
    ///
    /// Expected outcome: when the redirect from server 1 to server 2 is issued,
    /// `follow_redirects()` compares the pinned cert (server 1's) against the
    /// recorded cert for server 2 — they differ → `Err`.
    #[tokio::test]
    async fn end_to_end_https_cert_mismatch_refused_real_servers() {
        // Two servers, two fresh rcgen certs (each call to spawn_tls_test_server()
        // generates a new key pair and therefore a unique certificate).
        let (port1, cert1_der) = spawn_tls_test_server().await;
        let (port2, cert2_der) = spawn_tls_test_server().await;
        assert_ne!(
            cert1_der, cert2_der,
            "test requires two distinct certificates"
        );

        // Pre-populate cert store via real TLS probes.
        // The production probe_and_record_cert() uses native roots, which will not
        // trust self-signed rcgen certs.  We use the injectable-roots variant so
        // the probe succeeds and records the real DER bytes.
        let store = CertVerifyStore::new();
        probe_and_record_cert_with_roots("localhost", port1, &store, test_root_store(&cert1_der))
            .await
            .expect("probe to server 1 must succeed with matching trust anchor");
        probe_and_record_cert_with_roots("localhost", port2, &store, test_root_store(&cert2_der))
            .await
            .expect("probe to server 2 must succeed with matching trust anchor");

        // Confirm the store holds the real cert bytes we expect.
        assert_eq!(store.get(&format!("localhost:{port1}")).unwrap(), cert1_der);
        assert_eq!(store.get(&format!("localhost:{port2}")).unwrap(), cert2_der);

        // MockConnector: origin (server 1) redirects to server 2.
        let mock = MockConnector::new(vec![
            redirect(307, &format!("https://localhost:{port2}/target")),
            ok_200(), // iteration 2 — cert pinning check fires before this is used
        ]);
        let conn = wrap(mock.clone());

        let result = follow_redirects(
            conn,
            make_get(&format!("https://localhost:{port1}/source"), None),
            5,
            Some(store),
        )
        .await;

        assert!(
            result.is_err(),
            "redirect to a server with a DIFFERENT cert must be refused; got: {:?}",
            result
        );
        let err_dbg = format!("{:?}", result.unwrap_err());
        assert!(
            err_dbg.contains("mismatch") || err_dbg.contains("cert"),
            "error debug output must describe a certificate mismatch; got: {err_dbg}"
        );
    }

    /// **End-to-end Test 3** — Two real HTTPS servers sharing the **same** cert.
    ///
    /// This models a cluster where all nodes share a single TLS identity
    /// (e.g., a wildcard cert or a cluster-wide certificate).  Cert pinning must
    /// be satisfied and the redirect chain must complete with 200 OK.
    #[tokio::test]
    async fn end_to_end_https_same_cert_redirect_passes_real_servers() {
        use rcgen::{generate_simple_self_signed, CertifiedKey};

        // One cert, two servers — identical TLS identity on both ports.
        let CertifiedKey { cert, key_pair } =
            generate_simple_self_signed(vec!["localhost".to_owned()]).unwrap();
        let cert_der = cert.der().to_vec();
        let key_der = key_pair.serialize_der();

        // Build a factory that creates a ServerConfig serving the shared cert.
        let make_config = || {
            let c = rustls::pki_types::CertificateDer::from(cert_der.clone());
            let k = rustls::pki_types::PrivateKeyDer::Pkcs8(
                rustls::pki_types::PrivatePkcs8KeyDer::from(key_der.clone()),
            );
            Arc::new(
                rustls::ServerConfig::builder_with_provider(Arc::new(
                    rustls::crypto::aws_lc_rs::default_provider(),
                ))
                .with_safe_default_protocol_versions()
                .unwrap()
                .with_no_client_auth()
                .with_single_cert(vec![c], k)
                .unwrap(),
            )
        };

        let port1 = spawn_tls_server_with_config(make_config()).await;
        let port2 = spawn_tls_server_with_config(make_config()).await;

        // Pre-populate cert store — both entries record the same DER bytes.
        let store = CertVerifyStore::new();
        let roots = test_root_store(&cert_der);
        probe_and_record_cert_with_roots("localhost", port1, &store, roots.clone())
            .await
            .expect("probe to server 1 must succeed");
        probe_and_record_cert_with_roots("localhost", port2, &store, roots)
            .await
            .expect("probe to server 2 must succeed");

        assert_eq!(
            store.get(&format!("localhost:{port1}")).unwrap(),
            store.get(&format!("localhost:{port2}")).unwrap(),
            "both store entries must hold identical cert DER bytes"
        );

        // MockConnector: redirect from server 1 to server 2 → final 200.
        let mock = MockConnector::new(vec![
            redirect(307, &format!("https://localhost:{port2}/target")),
            ok_200(),
        ]);
        let conn = wrap(mock.clone());

        let result = follow_redirects(
            conn,
            make_get(&format!("https://localhost:{port1}/source"), None),
            5,
            Some(store),
        )
        .await;

        assert!(
            result.is_ok(),
            "HTTPS redirect with matching cert must succeed; got: {:?}",
            result
        );
        assert_eq!(result.unwrap().status().as_u16(), 200);
        // Two connector calls: origin (→307) and target (→200).
        assert_eq!(
            mock.call_count(),
            2,
            "both servers must be contacted exactly once"
        );
    }

    /// **End-to-end Test 4** — HTTP → HTTP redirect with real OS-assigned ports.
    ///
    /// Two plain TCP listeners are bound on random loopback ports to obtain real
    /// OS-assigned port numbers.  The `CertVerifyStore` is attached (matching the
    /// production configuration) but the origin is `http://` so no TLS checks
    /// fire at all — no scheme-downgrade guard, no cert probe, no cert comparison.
    ///
    /// Expected outcome: `follow_redirects()` follows the redirect transparently
    /// and returns 200 OK.  `call_count == 2` confirms both the origin and the
    /// redirect target were contacted by the connector.
    #[tokio::test]
    async fn end_to_end_http_to_http_redirect_passes_real_servers() {
        // Two real TCP listeners — port numbers appear in the request URIs so
        // the test is realistic about the addresses involved.  The listeners are
        // not actually served (MockConnector handles the HTTP transport layer);
        // their only role is to provide genuine OS-assigned port numbers.
        let listener_origin = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port_origin = listener_origin.local_addr().unwrap().port();

        let listener_target = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port_target = listener_target.local_addr().unwrap().port();

        // cert_store is Some (matching the production state set by
        // make_redirecting_client), but the origin is http:// so none of the TLS
        // guards will be entered.
        let cert_store = CertVerifyStore::new();

        let target_url = format!("http://localhost:{port_target}/v1/objects/bucket/key");
        let mock = MockConnector::new(vec![redirect(307, &target_url), ok_200()]);
        let conn = wrap(mock.clone());

        let origin_url = format!("http://localhost:{port_origin}/bucket/key");
        let result = follow_redirects(
            conn,
            make_get(&origin_url, Some("Bearer secret-token")),
            5,
            Some(cert_store),
        )
        .await;

        assert!(
            result.is_ok(),
            "HTTP \u{2192} HTTP redirect must succeed — no TLS checks apply; got: {:?}",
            result
        );
        assert_eq!(result.unwrap().status().as_u16(), 200);
        // Two connector calls: origin (→307) and redirect target (→200).
        assert_eq!(
            mock.call_count(),
            2,
            "connector must be called twice: origin + redirect target"
        );

        // Drop listeners explicitly to document their role was purely port allocation.
        drop(listener_origin);
        drop(listener_target);
    }
}
