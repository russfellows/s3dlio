// src/reqwest_client.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Reqwest-backed HTTP client for the AWS SDK.
//!
//! Implements `aws_smithy_runtime_api::client::http::HttpClient` by delegating
//! to a `reqwest::Client`. This enables:
//! - Automatic HTTP/2 negotiation via TLS ALPN on `https://` endpoints (always on)
//! - Automatic HTTP/2 cleartext (h2c) probe on `http://` endpoints with transparent
//!   fallback to HTTP/1.1 if the server rejects the h2c prior-knowledge preface
//! - Custom connection pool tuning via reqwest's builder API
//! - TLS configuration via reqwest's built-in rustls support, including
//!   custom CA bundles for private-PKI / self-signed endpoints
//!
//! # HTTP/2 behaviour by endpoint scheme
//!
//! | Endpoint scheme | Default behaviour | Override |
//! |---|---|---|
//! | `https://` | ALPN auto-negotiates h2 — **no config needed** | — |
//! | `http://` | Probes h2c once; falls back to HTTP/1.1 if rejected | `S3DLIO_H2C=1` to force h2c with no fallback |
//!
//! The h2c probe fires exactly once per process on the first plain-HTTP connection.
//! If the server responds with anything other than a valid HTTP/2 frame (e.g.
//! an HTTP/1.1 400), the probe is marked as failed, the connection is retried
//! transparently on the HTTP/1.1 client, and all subsequent connections skip the
//! probe.  `https://` endpoints are completely unaffected.
//!
//! # Environment Variables
//! All environment variable names and their defaults are defined in [`crate::constants`].
//!
//! - [`crate::constants::ENV_S3DLIO_H2C`] — h2c mode (force/disable/auto)
//! - [`crate::constants::ENV_POOL_MAX_IDLE_PER_HOST`] — max idle connections per host
//! - [`crate::constants::ENV_POOL_IDLE_TIMEOUT_SECS`] — idle connection timeout
//! - [`crate::constants::ENV_H2_ADAPTIVE_WINDOW`] — enable/disable BDP adaptive window
//! - [`crate::constants::ENV_H2_STREAM_WINDOW_MB`] — per-stream window (static mode)
//! - [`crate::constants::ENV_H2_CONN_WINDOW_MB`] — per-connection window (static mode)

use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::time::Duration;

use aws_smithy_runtime_api::client::http::{
    HttpClient, HttpConnector, HttpConnectorFuture, HttpConnectorSettings, SharedHttpConnector,
};
use aws_smithy_runtime_api::client::orchestrator::{HttpRequest, HttpResponse};
use aws_smithy_runtime_api::client::result::ConnectorError;
use aws_smithy_runtime_api::client::runtime_components::RuntimeComponents;
use aws_smithy_types::body::SdkBody;
use http_body_util::BodyExt;

use crate::constants::{
    H2WindowConfig, DEFAULT_POOL_IDLE_TIMEOUT_SECS, DEFAULT_POOL_MAX_IDLE_PER_HOST,
    ENV_POOL_IDLE_TIMEOUT_SECS, ENV_POOL_MAX_IDLE_PER_HOST, ENV_S3DLIO_H2C,
};

// ─────────────────────────────────────────────────────────────────────────────
// HTTP-version telemetry
// ─────────────────────────────────────────────────────────────────────────────

/// Ensures the "first response" HTTP-version INFO log is emitted exactly once.
static PROTOCOL_LOGGED: AtomicBool = AtomicBool::new(false);

/// Set to `true` when the first response is HTTP/2, `false` for HTTP/1.x.
/// Written once when `PROTOCOL_LOGGED` transitions to `true`.
static OBSERVED_IS_HTTP2: AtomicBool = AtomicBool::new(false);

/// Returns the HTTP protocol version seen on the first S3 response, or `None`
/// if no response has been received yet.
///
/// The per-request INFO log fires inside the connector while the CLI progress
/// bar is redrawing the terminal and can get overwritten.  Call this function
/// after the progress bar finishes to surface the protocol in the summary line.
pub fn observed_http_version_str() -> Option<&'static str> {
    if PROTOCOL_LOGGED.load(Ordering::Relaxed) {
        Some(if OBSERVED_IS_HTTP2.load(Ordering::Relaxed) {
            "HTTP/2"
        } else {
            "HTTP/1.1"
        })
    } else {
        None
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// H2c auto-probe state
// ─────────────────────────────────────────────────────────────────────────────

const H2C_AUTO_UNKNOWN: u8 = 0; // haven't probed yet
const H2C_AUTO_OK: u8 = 1; // probe succeeded — keep using h2c
const H2C_AUTO_FAILED: u8 = 2; // probe failed — use HTTP/1.1 from now on

/// Per-process h2c auto-probe state.  Transitions: UNKNOWN → OK or UNKNOWN → FAILED.
/// Only consulted when `H2cMode::Auto` is active (the default).
static H2C_AUTO_STATE: AtomicU8 = AtomicU8::new(H2C_AUTO_UNKNOWN);

/// Which reqwest client to use for a given request.
///
/// Returned by [`select_client`] so the routing logic can be unit-tested
/// without requiring a live HTTP connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ClientChoice {
    /// Use the h2c (HTTP/2 prior-knowledge cleartext) client.
    H2c,
    /// Use the HTTP/1.1 client (also used for HTTPS with ALPN).
    Http1,
}

/// Controls how HTTP/2 is used on plain `http://` connections.
/// (`https://` always auto-negotiates via TLS ALPN, unaffected by this.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum H2cMode {
    /// Probe h2c on first plain-HTTP connection; fall back transparently if rejected.
    /// This is the **default** when `S3DLIO_H2C` is not set.
    Auto,
    /// Always use h2c prior knowledge; never fall back.  Set via `S3DLIO_H2C=1`.
    ForceH2c,
    /// Always use HTTP/1.1; skip the auto-probe entirely.
    /// Used by the legacy `ReqwestHttpClient::new(single_client)` constructor.
    ForceHttp1,
}

// ─────────────────────────────────────────────────────────────────────────────
// ReqwestHttpClient  (implements HttpClient — creates connectors on demand)
// ─────────────────────────────────────────────────────────────────────────────

/// A Smithy `HttpClient` backed by a pair of `reqwest::Client`s.
///
/// Holds both an h2c client (built with `http2_prior_knowledge`) and an
/// HTTP/1.1 client.  The [`H2cMode`] determines which is used on plain-HTTP
/// connections; `https://` connections always use the http/1.1 client (ALPN
/// handles HTTP/2 negotiation there automatically).
#[derive(Debug, Clone)]
pub struct ReqwestHttpClient {
    h2c_client: reqwest::Client,
    http1_client: reqwest::Client,
    mode: H2cMode,
}

impl ReqwestHttpClient {
    /// Legacy constructor: wraps a single reqwest client in `ForceHttp1` mode.
    /// For auto h2c probing, use [`build_smithy_http_client`] instead.
    pub fn new(client: reqwest::Client) -> Self {
        Self {
            h2c_client: client.clone(),
            http1_client: client,
            mode: H2cMode::ForceHttp1,
        }
    }
}

impl HttpClient for ReqwestHttpClient {
    fn http_connector(
        &self,
        settings: &HttpConnectorSettings,
        _components: &RuntimeComponents,
    ) -> SharedHttpConnector {
        let read_timeout = settings.read_timeout();
        SharedHttpConnector::new(ReqwestHttpConnector {
            h2c_client: self.h2c_client.clone(),
            http1_client: self.http1_client.clone(),
            mode: self.mode,
            read_timeout,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ReqwestHttpConnector  (implements HttpConnector — executes one request)
// ─────────────────────────────────────────────────────────────────────────────

struct ReqwestHttpConnector {
    h2c_client: reqwest::Client,
    http1_client: reqwest::Client,
    mode: H2cMode,
    read_timeout: Option<Duration>,
}

// Required by SharedHttpConnector::new
impl fmt::Debug for ReqwestHttpConnector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReqwestHttpConnector")
            .field("mode", &self.mode)
            .field("read_timeout", &self.read_timeout)
            .finish()
    }
}

impl HttpConnector for ReqwestHttpConnector {
    fn call(&self, req: HttpRequest) -> HttpConnectorFuture {
        let h2c_client = self.h2c_client.clone();
        let http1_client = self.http1_client.clone();
        let mode = self.mode;
        let read_timeout = self.read_timeout;

        HttpConnectorFuture::new(async move {
            // ── Decompose the Smithy request ──────────────────────────────
            let (parts, sdk_body) = req
                .try_into_http1x()
                .map_err(|e| ConnectorError::other(e.into(), None))?
                .into_parts();

            let url = parts.uri.to_string();
            let method = reqwest::Method::from_bytes(parts.method.as_str().as_bytes())
                .map_err(|e| ConnectorError::other(e.into(), None))?;

            // Collect body bytes before the first attempt so we can retry
            // with a different client if the h2c probe fails.
            // bytes::Bytes is cheaply cloneable (Arc-backed reference count).
            let req_body = sdk_body
                .collect()
                .await
                .map_err(|e: Box<dyn std::error::Error + Send + Sync + 'static>| {
                    ConnectorError::io(e)
                })?
                .to_bytes();

            // ── Pick client for first attempt ─────────────────────────────
            // h2c (prior-knowledge HTTP/2 cleartext) is only valid on plain
            // http:// connections.  For https:// the http1_client is always
            // used; rustls ALPN negotiates HTTP/2 (or falls back to HTTP/1.1)
            // transparently during the TLS handshake.
            let is_plain_http = parts.uri.scheme_str() == Some("http");
            let auto_state = H2C_AUTO_STATE.load(Ordering::Relaxed);

            let (choice, is_auto_probe) = select_client(mode, is_plain_http, auto_state);
            let first_client = match choice {
                ClientChoice::H2c => &h2c_client,
                ClientChoice::Http1 => &http1_client,
            };

            // ── First attempt ─────────────────────────────────────────────
            let mut builder = first_client.request(method.clone(), &url);
            for (name, value) in &parts.headers {
                builder = builder.header(name.as_str(), value.as_bytes());
            }
            if let Some(timeout) = read_timeout {
                builder = builder.timeout(timeout);
            }
            builder = builder.body(reqwest::Body::from(req_body.clone()));

            let first_result = builder.send().await;

            // ── Handle auto-probe outcome ─────────────────────────────────
            let resp = match first_result {
                Ok(r) => {
                    if is_auto_probe {
                        H2C_AUTO_STATE.store(H2C_AUTO_OK, Ordering::Relaxed);
                        tracing::info!(
                            "h2c auto-probe succeeded — \
                             HTTP/2 cleartext active for plain-HTTP connections"
                        );
                    }
                    r
                }

                // Probe failed with a protocol error (not connect/timeout):
                // the server doesn't speak h2c.  Fall back once and remember.
                Err(ref e) if is_auto_probe && !e.is_connect() && !e.is_timeout() => {
                    // compare_exchange: only the first racing failure logs + transitions
                    if H2C_AUTO_STATE
                        .compare_exchange(
                            H2C_AUTO_UNKNOWN,
                            H2C_AUTO_FAILED,
                            Ordering::Relaxed,
                            Ordering::Relaxed,
                        )
                        .is_ok()
                    {
                        tracing::info!(
                            "h2c auto-probe: server rejected HTTP/2 prior knowledge — \
                             falling back to HTTP/1.1 for all future connections"
                        );
                    }
                    // Retry the exact same request with the HTTP/1.1 client.
                    let mut rb = http1_client.request(method.clone(), &url);
                    for (name, value) in &parts.headers {
                        rb = rb.header(name.as_str(), value.as_bytes());
                    }
                    if let Some(timeout) = read_timeout {
                        rb = rb.timeout(timeout);
                    }
                    rb = rb.body(reqwest::Body::from(req_body.clone()));
                    rb.send().await.map_err(|e| {
                        if e.is_connect() || e.is_timeout() {
                            ConnectorError::timeout(e.into())
                        } else {
                            ConnectorError::io(e.into())
                        }
                    })?
                }

                Err(e) => {
                    return Err(if e.is_connect() || e.is_timeout() {
                        ConnectorError::timeout(e.into())
                    } else {
                        ConnectorError::io(e.into())
                    })
                }
            };

            // ── Protocol-version telemetry (logged once at INFO) ──────────
            let status = resp.status().as_u16();
            let version = resp.version();
            let headers = resp.headers().clone();

            if !PROTOCOL_LOGGED.swap(true, Ordering::Relaxed) {
                OBSERVED_IS_HTTP2.store(version == reqwest::Version::HTTP_2, Ordering::Relaxed);
                tracing::info!("HTTP protocol (first response): {:?}", version);
            } else {
                tracing::debug!("HTTP protocol: {:?}", version);
            }

            let resp_body = resp
                .bytes()
                .await
                .map_err(|e| ConnectorError::io(e.into()))?;

            // ── Build Smithy response ─────────────────────────────────────
            let mut response = HttpResponse::new(
                http::StatusCode::from_u16(status)
                    .map_err(|e| ConnectorError::other(e.into(), None))?
                    .into(),
                SdkBody::from(resp_body),
            );

            for (name, value) in &headers {
                response.headers_mut().append(
                    http::HeaderName::from_bytes(name.as_str().as_bytes())
                        .map_err(|e| ConnectorError::other(e.into(), None))?,
                    http::HeaderValue::from_bytes(value.as_bytes())
                        .map_err(|e| ConnectorError::other(e.into(), None))?,
                );
            }

            Ok(response)
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory function
// ─────────────────────────────────────────────────────────────────────────────

/// Select which client to use for a request and whether this is an auto-probe attempt.
///
/// This is a pure function extracted from the hot path in `call()` so the
/// routing logic can be unit-tested without network access.
///
/// # Arguments
/// - `mode`          — h2c operating mode (from `S3DLIO_H2C`)
/// - `is_plain_http` — `true` when the request URI scheme is `http://`
/// - `auto_state`    — current value of `H2C_AUTO_STATE` (`H2C_AUTO_UNKNOWN/OK/FAILED`)
///
/// # Returns
/// `(ClientChoice, is_auto_probe)`:
/// - `ClientChoice::H2c`   — use the h2c (prior-knowledge) client
/// - `ClientChoice::Http1` — use the HTTP/1.1 / ALPN client
/// - `is_auto_probe`       — `true` only on the very first h2c probe attempt
pub(crate) fn select_client(
    mode: H2cMode,
    is_plain_http: bool,
    auto_state: u8,
) -> (ClientChoice, bool) {
    match mode {
        // ForceH2c: h2c prior-knowledge only makes sense on http://.  On
        // https:// fall through to Http1 so TLS ALPN handles HTTP/2.
        H2cMode::ForceH2c if is_plain_http => (ClientChoice::H2c, false),
        H2cMode::ForceH2c => (ClientChoice::Http1, false),
        H2cMode::ForceHttp1 => (ClientChoice::Http1, false),
        // Auto: probe h2c on the first plain-HTTP connection, then remember.
        H2cMode::Auto if is_plain_http && auto_state == H2C_AUTO_UNKNOWN => {
            (ClientChoice::H2c, true)
        }
        H2cMode::Auto if is_plain_http && auto_state == H2C_AUTO_OK => (ClientChoice::H2c, false),
        // Auto + https://, or Auto + http:// but probe already failed.
        H2cMode::Auto => (ClientChoice::Http1, false),
    }
}

/// Returns `true` if the given env-var value should enable HTTP/2 cleartext
/// (h2c / prior knowledge) transport for plain HTTP endpoints.
///
/// Recognized truthy values (case-insensitive): `1`, `true`, `yes`, `on`, `enable`.
/// Everything else (including empty string) is falsy.
///
/// Extracted as a pure function so it can be unit-tested without env-var
/// manipulation.  Note: this flag is **only** relevant for plain `http://`
/// endpoints; HTTPS endpoints always auto-negotiate HTTP/2 via TLS ALPN.
pub(crate) fn h2c_enabled_from_val(val: &str) -> bool {
    matches!(
        val.to_lowercase().as_str(),
        "1" | "true" | "yes" | "on" | "enable"
    )
}

/// Determine `H2cMode` from the [`ENV_S3DLIO_H2C`] environment variable.
///
/// | `S3DLIO_H2C` value | Mode |
/// |---|---|
/// | not set | `Auto` — probe h2c once, fall back to HTTP/1.1 if rejected |
/// | truthy (`1`, `true`, …) | `ForceH2c` — always h2c, no fallback |
/// | falsy (`0`, `false`, …) | `ForceHttp1` — skip probe, always HTTP/1.1 |
pub(crate) fn h2c_mode_from_env() -> H2cMode {
    match std::env::var(ENV_S3DLIO_H2C) {
        Err(_) => H2cMode::Auto,
        Ok(v) if h2c_enabled_from_val(&v) => H2cMode::ForceH2c,
        Ok(_) => H2cMode::ForceHttp1,
    }
}

/// Internal: build one reqwest client with or without h2c prior knowledge.
///
/// When `h2c` is `true`, HTTP/2 flow-control window tuning is applied using
/// the adaptive/static strategy resolved from env vars via [`H2WindowConfig::from_env`].
/// See [`crate::constants`] for all tunable environment variable names and defaults.
fn build_reqwest_client_raw(
    ca_bundle_path: Option<&str>,
    h2c: bool,
) -> anyhow::Result<reqwest::Client> {
    let max_idle: usize = std::env::var(ENV_POOL_MAX_IDLE_PER_HOST)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_POOL_MAX_IDLE_PER_HOST);

    let idle_timeout_secs: u64 = std::env::var(ENV_POOL_IDLE_TIMEOUT_SECS)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_POOL_IDLE_TIMEOUT_SECS);

    let mut builder = reqwest::Client::builder()
        .pool_max_idle_per_host(max_idle)
        .pool_idle_timeout(Duration::from_secs(idle_timeout_secs))
        .tcp_nodelay(true);

    // Load custom CA bundle if provided (e.g. self-signed MinIO / private PKI).
    // Fully independent of HTTP version negotiation.
    if let Some(path) = ca_bundle_path {
        let pem = std::fs::read(path)
            .map_err(|e| anyhow::anyhow!("Failed to read CA bundle '{}': {}", path, e))?;
        let cert = reqwest::Certificate::from_pem(&pem)
            .map_err(|e| anyhow::anyhow!("Failed to parse CA bundle '{}': {}", path, e))?;
        builder = builder.add_root_certificate(cert);
    }

    if h2c {
        builder = builder.http2_prior_knowledge();

        // ── HTTP/2 flow-control window tuning ─────────────────────────────
        // Resolved once from env vars; see constants::H2WindowConfig for docs.
        let win_cfg = H2WindowConfig::from_env();

        if win_cfg.adaptive {
            // BDP-based adaptive window: hyper measures RTT via H2 PINGs and
            // issues WINDOW_UPDATE proactively to keep the window ≥ bandwidth×RTT.
            // Self-tuning from 64 KB up to hundreds of MiB — works for all
            // object sizes without manual configuration.
            builder = builder.http2_adaptive_window(true);
            tracing::debug!(
                "HTTP/2 window mode: adaptive (BDP estimator) \
                 — stream/conn windows auto-tune to link bandwidth×RTT"
            );
        } else {
            // Static windows: user has opted out of adaptive mode by setting
            // S3DLIO_H2_ADAPTIVE_WINDOW=0 or by setting explicit window sizes.
            // These fixed values are used for the lifetime of the process.
            let stream_bytes = win_cfg.stream_window_bytes();
            let conn_bytes = win_cfg.conn_window_bytes();
            builder = builder
                .http2_initial_stream_window_size(stream_bytes)
                .http2_initial_connection_window_size(conn_bytes);
            tracing::debug!(
                "HTTP/2 window mode: static  stream={} MiB  connection={} MiB",
                win_cfg.stream_window_mb,
                win_cfg.conn_window_mb,
            );
        }
    }

    Ok(builder
        .build()
        .expect("reqwest client build should not fail with valid settings"))
}

/// Build a `SharedHttpClient` ready for the AWS SDK.
///
/// This is the **preferred** constructor.  It pre-builds both an h2c and an
/// HTTP/1.1 reqwest client and wires them into the auto-probe logic:
///
/// - **`https://` endpoints** — HTTP/2 via TLS ALPN, fully automatic.
/// - **`http://` endpoints** — h2c probed once on the first connection;
///   transparent HTTP/1.1 fallback if the server rejects it.
/// - **`S3DLIO_H2C=1`** — force h2c on plain HTTP, no fallback.
///
/// `ca_bundle_path` adds a custom PEM root certificate (for private-PKI /
/// self-signed endpoints) and is independent of HTTP version negotiation.
pub fn build_smithy_http_client(
    ca_bundle_path: Option<&str>,
) -> anyhow::Result<aws_smithy_runtime_api::client::http::SharedHttpClient> {
    let mode = h2c_mode_from_env();

    match mode {
        H2cMode::Auto => tracing::info!(
            "HTTP version mode: auto \
             (https:// → HTTP/2 via TLS ALPN; http:// → h2c probe once, HTTP/1.1 fallback)"
        ),
        H2cMode::ForceH2c => {
            let win = H2WindowConfig::from_env();
            let win_desc = if win.adaptive {
                "adaptive BDP window".to_string()
            } else {
                format!(
                    "static stream={} MiB conn={} MiB",
                    win.stream_window_mb, win.conn_window_mb
                )
            };
            tracing::info!(
                "HTTP version mode: FORCED HTTP/2 (S3DLIO_H2C=1) — \
                 https:// uses HTTP/2 via ALPN; http:// uses h2c prior-knowledge, no fallback; \
                 h2 window: {win_desc}"
            );
        }
        H2cMode::ForceHttp1 => tracing::info!("HTTP version mode: forced HTTP/1.1 (S3DLIO_H2C=0)"),
    }

    let h2c_client = build_reqwest_client_raw(ca_bundle_path, true)?;
    let http1_client = build_reqwest_client_raw(ca_bundle_path, false)?;
    Ok(aws_smithy_runtime_api::client::http::SharedHttpClient::new(
        ReqwestHttpClient {
            h2c_client,
            http1_client,
            mode,
        },
    ))
}

/// Build a single `reqwest::Client` (no h2c, no auto-probe).
///
/// Prefer [`build_smithy_http_client`] for new code.  This function is kept
/// for backward compatibility and for callers that need a raw reqwest client.
pub fn build_reqwest_http_client_with_ca(
    ca_bundle_path: Option<&str>,
) -> anyhow::Result<reqwest::Client> {
    build_reqwest_client_raw(ca_bundle_path, false)
}

/// Convenience wrapper — no custom CA bundle, HTTP/1.1 only.
pub fn build_reqwest_http_client() -> reqwest::Client {
    build_reqwest_http_client_with_ca(None).expect("reqwest client build (no CA) should not fail")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{
        H2WindowConfig, DEFAULT_H2_CONN_WINDOW_MB, DEFAULT_H2_STREAM_WINDOW_MB,
        ENV_H2_ADAPTIVE_WINDOW, ENV_H2_CONN_WINDOW_MB, ENV_H2_STREAM_WINDOW_MB,
        ENV_POOL_IDLE_TIMEOUT_SECS, ENV_POOL_MAX_IDLE_PER_HOST, ENV_S3DLIO_H2C,
        H2_WINDOW_MB_HARD_CAP,
    };

    /// Mutex to serialize tests that manipulate `S3DLIO_H2C` / pool env vars.
    /// `std::env::set_var` is not thread-safe; holding this lock before calling it
    /// prevents races when tests run in parallel.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    // ── h2c_enabled_from_val (pure logic — no env var manipulation) ──────────

    #[test]
    fn test_h2c_truthy_values() {
        for val in &[
            "1", "true", "yes", "on", "enable", "TRUE", "YES", "ON", "ENABLE",
        ] {
            assert!(
                h2c_enabled_from_val(val),
                "Expected h2c enabled for '{val}'"
            );
        }
    }

    #[test]
    fn test_h2c_falsy_values() {
        for val in &[
            "0",
            "false",
            "no",
            "off",
            "disable",
            "",
            "2",
            "yes-please",
            "disabled",
            "http2",
        ] {
            assert!(
                !h2c_enabled_from_val(val),
                "Expected h2c disabled for '{val}'"
            );
        }
    }

    // ── build_reqwest_http_client() (builds client — env var manipulation) ──

    #[test]
    fn test_build_client_succeeds() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Must not panic
        let _ = build_reqwest_http_client();
    }

    /// Verify that enabling h2c via `S3DLIO_H2C=1` does not panic during client
    /// construction.  The actual HTTP/2 negotiation is only observable at
    /// connection time (requires a live server), so this test only validates the
    /// build path.
    #[test]
    fn test_build_h2c_client_succeeds() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let old_val = std::env::var(ENV_S3DLIO_H2C).ok();
        #[allow(deprecated)]
        std::env::set_var(ENV_S3DLIO_H2C, "1");
        let result = build_smithy_http_client(None);
        match old_val {
            #[allow(deprecated)]
            Some(v) => std::env::set_var(ENV_S3DLIO_H2C, v),
            #[allow(deprecated)]
            None => std::env::remove_var(ENV_S3DLIO_H2C),
        }
        assert!(
            result.is_ok(),
            "build_smithy_http_client() must not panic with S3DLIO_H2C=1"
        );
    }

    /// All truthy `S3DLIO_H2C` variants produce a valid `SharedHttpClient`.
    #[test]
    fn test_build_h2c_client_all_truthy_variants() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for val in &["1", "true", "yes", "on", "enable"] {
            let old_val = std::env::var(ENV_S3DLIO_H2C).ok();
            #[allow(deprecated)]
            std::env::set_var(ENV_S3DLIO_H2C, val);
            let result = build_smithy_http_client(None);
            match old_val {
                #[allow(deprecated)]
                Some(v) => std::env::set_var(ENV_S3DLIO_H2C, v),
                #[allow(deprecated)]
                None => std::env::remove_var(ENV_S3DLIO_H2C),
            }
            assert!(result.is_ok(), "build failed for S3DLIO_H2C={val}");
        }
    }

    /// `h2c_mode_from_env` returns the correct mode for each env var state.
    #[test]
    fn test_h2c_mode_from_env() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        // Not set → Auto
        #[allow(deprecated)]
        std::env::remove_var(ENV_S3DLIO_H2C);
        assert_eq!(h2c_mode_from_env(), H2cMode::Auto);
        // Truthy → ForceH2c
        for val in &["1", "true", "yes", "on", "enable"] {
            #[allow(deprecated)]
            std::env::set_var(ENV_S3DLIO_H2C, val);
            assert_eq!(
                h2c_mode_from_env(),
                H2cMode::ForceH2c,
                "expected ForceH2c for '{val}'"
            );
        }
        // Falsy → ForceHttp1
        for val in &["0", "false", "no", "off", "disable"] {
            #[allow(deprecated)]
            std::env::set_var(ENV_S3DLIO_H2C, val);
            assert_eq!(
                h2c_mode_from_env(),
                H2cMode::ForceHttp1,
                "expected ForceHttp1 for '{val}'"
            );
        }
        // Restore
        #[allow(deprecated)]
        std::env::remove_var(ENV_S3DLIO_H2C);
    }

    /// Verify that `build_smithy_http_client` loads a valid CA bundle correctly.
    /// Uses `configs/aws-root-ca.pem` which ships with the repository.
    #[test]
    fn test_build_client_with_ca_bundle_succeeds() {
        let ca_path = concat!(env!("CARGO_MANIFEST_DIR"), "/configs/aws-root-ca.pem");
        let result = build_smithy_http_client(Some(ca_path));
        assert!(
            result.is_ok(),
            "Expected SharedHttpClient build to succeed with valid CA bundle: {:?}",
            result.err()
        );
    }

    /// Verify that a missing CA bundle path returns an error (not a panic).
    #[test]
    fn test_build_client_with_missing_ca_bundle_returns_error() {
        let result = build_reqwest_http_client_with_ca(Some("/nonexistent/path/ca.pem"));
        assert!(result.is_err(), "Expected error for missing CA bundle path");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("Failed to read CA bundle"),
            "Expected informative error message, got: {msg}"
        );
    }

    #[test]
    fn test_pool_settings_from_env() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let old_idle = std::env::var(ENV_POOL_MAX_IDLE_PER_HOST).ok();
        let old_timeout = std::env::var(ENV_POOL_IDLE_TIMEOUT_SECS).ok();
        #[allow(deprecated)]
        std::env::set_var(ENV_POOL_MAX_IDLE_PER_HOST, "64");
        #[allow(deprecated)]
        std::env::set_var(ENV_POOL_IDLE_TIMEOUT_SECS, "120");
        let _ = build_reqwest_http_client(); // must not panic
        match old_idle {
            #[allow(deprecated)]
            Some(v) => std::env::set_var(ENV_POOL_MAX_IDLE_PER_HOST, v),
            #[allow(deprecated)]
            None => std::env::remove_var(ENV_POOL_MAX_IDLE_PER_HOST),
        }
        match old_timeout {
            #[allow(deprecated)]
            Some(v) => std::env::set_var(ENV_POOL_IDLE_TIMEOUT_SECS, v),
            #[allow(deprecated)]
            None => std::env::remove_var(ENV_POOL_IDLE_TIMEOUT_SECS),
        }
    }

    // ── H2WindowConfig env-var parsing ─────────────────────────────────────
    //
    // These tests verify the from_env() logic. Because they manipulate env
    // vars, they all hold ENV_LOCK for serialisation.

    /// Helper: save + clear all three window env vars; returns old values.
    fn save_window_env() -> (Option<String>, Option<String>, Option<String>) {
        (
            std::env::var(ENV_H2_ADAPTIVE_WINDOW).ok(),
            std::env::var(ENV_H2_STREAM_WINDOW_MB).ok(),
            std::env::var(ENV_H2_CONN_WINDOW_MB).ok(),
        )
    }

    /// Helper: restore window env vars from saved values.
    fn restore_window_env(saved: (Option<String>, Option<String>, Option<String>)) {
        #[allow(deprecated)]
        match saved.0 {
            Some(v) => std::env::set_var(ENV_H2_ADAPTIVE_WINDOW, v),
            None => std::env::remove_var(ENV_H2_ADAPTIVE_WINDOW),
        }
        #[allow(deprecated)]
        match saved.1 {
            Some(v) => std::env::set_var(ENV_H2_STREAM_WINDOW_MB, v),
            None => std::env::remove_var(ENV_H2_STREAM_WINDOW_MB),
        }
        #[allow(deprecated)]
        match saved.2 {
            Some(v) => std::env::set_var(ENV_H2_CONN_WINDOW_MB, v),
            None => std::env::remove_var(ENV_H2_CONN_WINDOW_MB),
        }
    }

    /// Unset env → adaptive ON with default static sizes (though static sizes
    /// are irrelevant when adaptive is true).
    #[test]
    fn test_h2_window_config_unset_is_adaptive() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved = save_window_env();
        #[allow(deprecated)]
        {
            std::env::remove_var(ENV_H2_ADAPTIVE_WINDOW);
            std::env::remove_var(ENV_H2_STREAM_WINDOW_MB);
            std::env::remove_var(ENV_H2_CONN_WINDOW_MB);
        }
        let cfg = H2WindowConfig::from_env();
        restore_window_env(saved);
        assert!(cfg.adaptive, "unset env must give adaptive=true");
        assert_eq!(cfg.stream_window_mb, DEFAULT_H2_STREAM_WINDOW_MB);
        assert_eq!(cfg.conn_window_mb, DEFAULT_H2_CONN_WINDOW_MB);
    }

    /// Explicit truthy values for S3DLIO_H2_ADAPTIVE_WINDOW → adaptive ON.
    #[test]
    fn test_h2_window_config_adaptive_truthy() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved = save_window_env();
        for val in &["1", "true", "yes", "on"] {
            #[allow(deprecated)]
            std::env::set_var(ENV_H2_ADAPTIVE_WINDOW, val);
            let cfg = H2WindowConfig::from_env();
            assert!(cfg.adaptive, "adaptive must be true for '{val}'");
        }
        restore_window_env(saved);
    }

    /// Explicit falsy values for S3DLIO_H2_ADAPTIVE_WINDOW → static mode.
    #[test]
    fn test_h2_window_config_adaptive_off_uses_defaults() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved = save_window_env();
        #[allow(deprecated)]
        {
            std::env::set_var(ENV_H2_ADAPTIVE_WINDOW, "0");
            std::env::remove_var(ENV_H2_STREAM_WINDOW_MB);
            std::env::remove_var(ENV_H2_CONN_WINDOW_MB);
        }
        let cfg = H2WindowConfig::from_env();
        restore_window_env(saved);
        assert!(
            !cfg.adaptive,
            "adaptive must be false when S3DLIO_H2_ADAPTIVE_WINDOW=0"
        );
        assert_eq!(
            cfg.stream_window_mb, DEFAULT_H2_STREAM_WINDOW_MB,
            "stream window should default to DEFAULT_H2_STREAM_WINDOW_MB"
        );
        assert_eq!(
            cfg.conn_window_mb,
            DEFAULT_H2_STREAM_WINDOW_MB * 4,
            "conn window should default to 4× stream window"
        );
    }

    /// Static stream window can be overridden; conn defaults to 4× stream.
    #[test]
    fn test_h2_window_config_static_stream_override() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved = save_window_env();
        #[allow(deprecated)]
        {
            std::env::set_var(ENV_H2_ADAPTIVE_WINDOW, "0");
            std::env::set_var(ENV_H2_STREAM_WINDOW_MB, "16");
            std::env::remove_var(ENV_H2_CONN_WINDOW_MB);
        }
        let cfg = H2WindowConfig::from_env();
        restore_window_env(saved);
        assert!(!cfg.adaptive);
        assert_eq!(cfg.stream_window_mb, 16);
        assert_eq!(cfg.conn_window_mb, 64, "conn should be 4× stream = 64 MiB");
    }

    /// Both stream and conn can be set independently.
    #[test]
    fn test_h2_window_config_static_both_override() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved = save_window_env();
        #[allow(deprecated)]
        {
            std::env::set_var(ENV_H2_ADAPTIVE_WINDOW, "false");
            std::env::set_var(ENV_H2_STREAM_WINDOW_MB, "8");
            std::env::set_var(ENV_H2_CONN_WINDOW_MB, "32");
        }
        let cfg = H2WindowConfig::from_env();
        restore_window_env(saved);
        assert!(!cfg.adaptive);
        assert_eq!(cfg.stream_window_mb, 8);
        assert_eq!(cfg.conn_window_mb, 32);
        assert_eq!(cfg.stream_window_bytes(), 8 * 1024 * 1024);
        assert_eq!(cfg.conn_window_bytes(), 32 * 1024 * 1024);
    }

    /// Values above the hard cap (256 MiB) are clamped.
    #[test]
    fn test_h2_window_config_hard_cap_clamps() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved = save_window_env();
        #[allow(deprecated)]
        {
            std::env::set_var(ENV_H2_ADAPTIVE_WINDOW, "0");
            std::env::set_var(ENV_H2_STREAM_WINDOW_MB, "999");
            std::env::set_var(ENV_H2_CONN_WINDOW_MB, "999");
        }
        let cfg = H2WindowConfig::from_env();
        restore_window_env(saved);
        assert_eq!(
            cfg.stream_window_mb, H2_WINDOW_MB_HARD_CAP,
            "values above hard cap must be clamped to {H2_WINDOW_MB_HARD_CAP}"
        );
        assert_eq!(cfg.conn_window_mb, H2_WINDOW_MB_HARD_CAP);
    }

    /// Zero and invalid values fall back to defaults (not zero).
    #[test]
    fn test_h2_window_config_zero_and_invalid_use_defaults() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved = save_window_env();
        for bad in &["0", "abc", "-1", ""] {
            #[allow(deprecated)]
            {
                std::env::set_var(ENV_H2_ADAPTIVE_WINDOW, "0");
                std::env::set_var(ENV_H2_STREAM_WINDOW_MB, bad);
                std::env::set_var(ENV_H2_CONN_WINDOW_MB, bad);
            }
            let cfg = H2WindowConfig::from_env();
            assert!(
                cfg.stream_window_mb > 0,
                "stream_window_mb must never be 0 (was {}, input '{bad}')",
                cfg.stream_window_mb
            );
            assert!(
                cfg.conn_window_mb > 0,
                "conn_window_mb must never be 0 (was {}, input '{bad}')",
                cfg.conn_window_mb
            );
        }
        restore_window_env(saved);
    }

    /// Build succeeds with h2c + static window configuration (no network needed).
    #[test]
    fn test_build_h2c_client_with_static_window_succeeds() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved_h2c = std::env::var(ENV_S3DLIO_H2C).ok();
        let saved_win = save_window_env();
        #[allow(deprecated)]
        {
            std::env::set_var(ENV_S3DLIO_H2C, "1");
            std::env::set_var(ENV_H2_ADAPTIVE_WINDOW, "0");
            std::env::set_var(ENV_H2_STREAM_WINDOW_MB, "4");
            std::env::set_var(ENV_H2_CONN_WINDOW_MB, "16");
        }
        let result = build_smithy_http_client(None);
        #[allow(deprecated)]
        match saved_h2c {
            Some(v) => std::env::set_var(ENV_S3DLIO_H2C, v),
            None => std::env::remove_var(ENV_S3DLIO_H2C),
        }
        restore_window_env(saved_win);
        assert!(
            result.is_ok(),
            "build with static h2 window must succeed: {:?}",
            result.err()
        );
    }

    /// Build succeeds with h2c + adaptive window (the default path).
    #[test]
    fn test_build_h2c_client_with_adaptive_window_succeeds() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let saved_h2c = std::env::var(ENV_S3DLIO_H2C).ok();
        let saved_win = save_window_env();
        #[allow(deprecated)]
        {
            std::env::set_var(ENV_S3DLIO_H2C, "1");
            std::env::remove_var(ENV_H2_ADAPTIVE_WINDOW); // unset = adaptive ON
        }
        let result = build_smithy_http_client(None);
        #[allow(deprecated)]
        match saved_h2c {
            Some(v) => std::env::set_var(ENV_S3DLIO_H2C, v),
            None => std::env::remove_var(ENV_S3DLIO_H2C),
        }
        restore_window_env(saved_win);
        assert!(
            result.is_ok(),
            "build with adaptive h2 window must succeed: {:?}",
            result.err()
        );
    }

    // ── ReqwestHttpClient (structural — no env var, no network) ─────────────

    #[test]
    fn test_reqwest_http_client_is_cloneable() {
        let client = reqwest::Client::builder()
            .build()
            .expect("default client should build");
        let http_client = ReqwestHttpClient::new(client);
        // Clone is required by SharedHttpClient::new
        let _cloned = http_client.clone();
    }

    #[test]
    fn test_reqwest_http_client_debug_does_not_panic() {
        let client = reqwest::Client::builder()
            .build()
            .expect("default client should build");
        let http_client = ReqwestHttpClient::new(client);
        let _ = format!("{:?}", http_client);
    }

    // ── select_client routing logic (pure — no env var, no network) ──────────
    //
    // These tests cover every branch of the match in select_client():
    //
    // Cleartext (http://) — is_plain_http = true
    //   ForceH2c  → H2c,   probe=false  (forced h2c, no fallback)
    //   ForceHttp1 → Http1, probe=false  (always HTTP/1.1)
    //   Auto + UNKNOWN → H2c, probe=true   (first-connection h2c probe)
    //   Auto + OK     → H2c, probe=false  (probe already succeeded)
    //   Auto + FAILED → Http1, probe=false  (probe failed, stay on HTTP/1.1)
    //
    // TLS (https://) — is_plain_http = false; ALPN negotiates HTTP/2 automatically
    //   ForceH2c  → Http1, probe=false  (h2c prior-knowledge is invalid over TLS)
    //   ForceHttp1 → Http1, probe=false
    //   Auto (any state) → Http1, probe=false  (ALPN handles HTTP/2 in the TLS layer)

    // ── cleartext http:// ─────────────────────────────────────────────────────

    #[test]
    fn test_select_client_force_h2c_plain_http() {
        // S3DLIO_H2C=1 with an http:// endpoint: must use H2c (no probe).
        let (choice, probe) = select_client(H2cMode::ForceH2c, true, H2C_AUTO_UNKNOWN);
        assert_eq!(choice, ClientChoice::H2c);
        assert!(!probe, "ForceH2c should never be a probe attempt");
    }

    #[test]
    fn test_select_client_force_http1_plain_http() {
        // S3DLIO_H2C=0 with an http:// endpoint: must use Http1 (no probe).
        let (choice, probe) = select_client(H2cMode::ForceHttp1, true, H2C_AUTO_UNKNOWN);
        assert_eq!(choice, ClientChoice::Http1);
        assert!(!probe);
    }

    #[test]
    fn test_select_client_auto_plain_http_first_connection() {
        // Auto mode, http://, no probe yet: send via H2c and mark as probe.
        let (choice, probe) = select_client(H2cMode::Auto, true, H2C_AUTO_UNKNOWN);
        assert_eq!(choice, ClientChoice::H2c);
        assert!(
            probe,
            "First Auto+http:// request must be the probe attempt"
        );
    }

    #[test]
    fn test_select_client_auto_plain_http_probe_succeeded() {
        // Auto mode, http://, probe previously succeeded: use H2c without probing.
        let (choice, probe) = select_client(H2cMode::Auto, true, H2C_AUTO_OK);
        assert_eq!(choice, ClientChoice::H2c);
        assert!(
            !probe,
            "After successful probe, is_auto_probe must be false"
        );
    }

    #[test]
    fn test_select_client_auto_plain_http_probe_failed() {
        // Auto mode, http://, probe previously failed: fall back to Http1.
        let (choice, probe) = select_client(H2cMode::Auto, true, H2C_AUTO_FAILED);
        assert_eq!(choice, ClientChoice::Http1);
        assert!(!probe);
    }

    // ── TLS https:// ──────────────────────────────────────────────────────────

    #[test]
    fn test_select_client_force_h2c_tls() {
        // CRITICAL: ForceH2c on https:// must NOT use the h2c client.
        // h2c prior-knowledge sends a plaintext HTTP/2 preface before TLS, which
        // causes "broken pipe" errors.  For TLS endpoints HTTP/2 is negotiated
        // transparently via ALPN — no special client needed.
        let (choice, probe) = select_client(H2cMode::ForceH2c, false, H2C_AUTO_UNKNOWN);
        assert_eq!(
            choice,
            ClientChoice::Http1,
            "ForceH2c on https:// must route to Http1 (ALPN handles HTTP/2 in TLS layer)"
        );
        assert!(!probe);
    }

    #[test]
    fn test_select_client_force_http1_tls() {
        let (choice, probe) = select_client(H2cMode::ForceHttp1, false, H2C_AUTO_UNKNOWN);
        assert_eq!(choice, ClientChoice::Http1);
        assert!(!probe);
    }

    #[test]
    fn test_select_client_auto_tls_unknown() {
        // Auto + https:// regardless of auto_state: always Http1 (ALPN handles it).
        let (choice, probe) = select_client(H2cMode::Auto, false, H2C_AUTO_UNKNOWN);
        assert_eq!(choice, ClientChoice::Http1);
        assert!(!probe);
    }

    #[test]
    fn test_select_client_auto_tls_ok() {
        let (choice, probe) = select_client(H2cMode::Auto, false, H2C_AUTO_OK);
        assert_eq!(choice, ClientChoice::Http1);
        assert!(!probe);
    }

    #[test]
    fn test_select_client_auto_tls_failed() {
        let (choice, probe) = select_client(H2cMode::Auto, false, H2C_AUTO_FAILED);
        assert_eq!(choice, ClientChoice::Http1);
        assert!(!probe);
    }
}
