// src/reqwest_client.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Reqwest-backed HTTP client for the AWS SDK.
//!
//! Implements `aws_smithy_runtime_api::client::http::HttpClient` by delegating
//! to a `reqwest::Client`. This enables:
//! - h2c (HTTP/2 cleartext) via `reqwest::ClientBuilder::http2_prior_knowledge()`
//! - Custom connection pool tuning via reqwest's builder API
//! - TLS configuration via reqwest's built-in rustls support
//!
//! # Environment Variables
//! - `S3DLIO_H2C=1` — enable HTTP/2 cleartext (h2c) transport
//! - `S3DLIO_POOL_MAX_IDLE_PER_HOST` — max idle connections per host (default: 32)
//! - `S3DLIO_POOL_IDLE_TIMEOUT_SECS` — idle connection timeout in seconds (default: 90)

use std::fmt;
use std::time::Duration;

use aws_smithy_runtime_api::client::http::{
    HttpClient, HttpConnector, HttpConnectorFuture, HttpConnectorSettings, SharedHttpConnector,
};
use aws_smithy_runtime_api::client::orchestrator::{HttpRequest, HttpResponse};
use aws_smithy_runtime_api::client::result::ConnectorError;
use aws_smithy_runtime_api::client::runtime_components::RuntimeComponents;
use aws_smithy_types::body::SdkBody;
use http_body_util::BodyExt;

// ─────────────────────────────────────────────────────────────────────────────
// ReqwestHttpClient  (implements HttpClient — creates connectors on demand)
// ─────────────────────────────────────────────────────────────────────────────

/// A Smithy `HttpClient` backed by `reqwest::Client`.
///
/// Implements the [`HttpClient`] trait expected by the AWS SDK so that
/// `reqwest`'s connection pool (with optional HTTP/2 cleartext support)
/// is used for all SDK requests.
#[derive(Debug, Clone)]
pub struct ReqwestHttpClient {
    client: reqwest::Client,
}

impl ReqwestHttpClient {
    /// Wrap an existing `reqwest::Client`.
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
        let read_timeout = settings.read_timeout();
        SharedHttpConnector::new(ReqwestHttpConnector {
            client: self.client.clone(),
            read_timeout,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ReqwestHttpConnector  (implements HttpConnector — executes one request)
// ─────────────────────────────────────────────────────────────────────────────

struct ReqwestHttpConnector {
    client: reqwest::Client,
    read_timeout: Option<Duration>,
}

// Required by SharedHttpConnector::new
impl fmt::Debug for ReqwestHttpConnector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReqwestHttpConnector")
            .field("read_timeout", &self.read_timeout)
            .finish()
    }
}

impl HttpConnector for ReqwestHttpConnector {
    fn call(&self, req: HttpRequest) -> HttpConnectorFuture {
        let client = self.client.clone();
        let read_timeout = self.read_timeout;

        HttpConnectorFuture::new(async move {
            // Convert the Smithy `HttpRequest` to a `reqwest::Request`.
            // `try_into_http1x()` gives us an `http::Request<SdkBody>`.
            let (parts, sdk_body) = req
                .try_into_http1x()
                .map_err(|e| ConnectorError::other(e.into(), None))?
                .into_parts();

            // Build reqwest URL from parts
            let url = parts.uri.to_string();

            let mut builder = client.request(
                reqwest::Method::from_bytes(parts.method.as_str().as_bytes())
                    .map_err(|e| ConnectorError::other(e.into(), None))?,
                &url,
            );

            // Copy headers
            for (name, value) in &parts.headers {
                builder = builder.header(name.as_str(), value.as_bytes());
            }

            // Apply optional per-request read timeout
            if let Some(timeout) = read_timeout {
                builder = builder.timeout(timeout);
            }

            // Stream the body from SdkBody
            let body_bytes = sdk_body
                .collect()
                .await
                .map_err(|e: Box<dyn std::error::Error + Send + Sync + 'static>| ConnectorError::io(e))?
                .to_bytes();
            builder = builder.body(reqwest::Body::from(body_bytes));

            // Execute the request
            let resp = builder
                .send()
                .await
                .map_err(|e| {
                    if e.is_connect() || e.is_timeout() {
                        ConnectorError::timeout(e.into())
                    } else {
                        ConnectorError::io(e.into())
                    }
                })?;

            // Convert the response back to a Smithy `HttpResponse`
            let status = resp.status().as_u16();
            let headers = resp.headers().clone();
            let body_bytes = resp.bytes().await
                .map_err(|e| ConnectorError::io(e.into()))?;

            let mut response = HttpResponse::new(
                http::StatusCode::from_u16(status)
                    .map_err(|e| ConnectorError::other(e.into(), None))?.into(),
                SdkBody::from(body_bytes),
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

/// Returns `true` if the given env-var value should enable HTTP/2 cleartext (h2c) transport.
///
/// Recognized truthy values (case-insensitive): `1`, `true`, `yes`, `on`, `enable`.
/// Everything else (including empty string) is falsy.
///
/// Extracted as a pure function so it can be unit-tested without env-var manipulation.
pub(crate) fn h2c_enabled_from_val(val: &str) -> bool {
    matches!(
        val.to_lowercase().as_str(),
        "1" | "true" | "yes" | "on" | "enable"
    )
}

/// Build a `reqwest::Client` with transport settings tuned for high-throughput
/// S3 workloads.
///
/// # Transport selection
/// | `S3DLIO_H2C` | Protocol |
/// |---|---|
/// | unset / `0` | HTTP/1.1 with connection keep-alive (TLS via rustls) |
/// | `1` / `true` / `yes` / `on` / `enable` | HTTP/2 cleartext (h2c) — no TLS ALPN negotiation |
///
/// # Connection pool defaults
/// | Setting | Default | Override via env |
/// |---|---|---|
/// | Idle connections per host | 32 | `S3DLIO_POOL_MAX_IDLE_PER_HOST` |
/// | Idle connection timeout | 90 s | `S3DLIO_POOL_IDLE_TIMEOUT_SECS` |
/// | TCP_NODELAY | enabled | — |
pub fn build_reqwest_http_client() -> reqwest::Client {
    let h2c = h2c_enabled_from_val(
        &std::env::var("S3DLIO_H2C").unwrap_or_default(),
    );

    let max_idle: usize = std::env::var("S3DLIO_POOL_MAX_IDLE_PER_HOST")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

    let idle_timeout_secs: u64 = std::env::var("S3DLIO_POOL_IDLE_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(90);

    let mut builder = reqwest::Client::builder()
        .pool_max_idle_per_host(max_idle)
        .pool_idle_timeout(Duration::from_secs(idle_timeout_secs))
        .tcp_nodelay(true);

    if h2c {
        tracing::info!(
            "S3DLIO_H2C enabled — using HTTP/2 cleartext (h2c) transport \
             (pool_max_idle_per_host={}, idle_timeout={}s)",
            max_idle, idle_timeout_secs
        );
        builder = builder.http2_prior_knowledge();
    } else {
        tracing::debug!(
            "S3DLIO_H2C not set — using HTTP/1.1 transport \
             (pool_max_idle_per_host={}, idle_timeout={}s)",
            max_idle, idle_timeout_secs
        );
    }

    builder
        .build()
        .expect("reqwest client build should not fail with valid settings")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mutex to serialize tests that manipulate `S3DLIO_H2C` / pool env vars.
    /// `std::env::set_var` is not thread-safe; holding this lock before calling it
    /// prevents races when tests run in parallel.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    // ── h2c_enabled_from_val (pure logic — no env var manipulation) ─────────

    #[test]
    fn test_h2c_truthy_values() {
        for val in &["1", "true", "yes", "on", "enable", "TRUE", "YES", "ON", "ENABLE"] {
            assert!(
                h2c_enabled_from_val(val),
                "Expected h2c enabled for '{val}'"
            );
        }
    }

    #[test]
    fn test_h2c_falsy_values() {
        for val in &["0", "false", "no", "off", "disable", "", "2", "yes-please", "disabled", "http2"] {
            assert!(
                !h2c_enabled_from_val(val),
                "Expected h2c disabled for '{val}'"
            );
        }
    }

    // ── build_reqwest_http_client() (builds client — env var manipulation) ──

    #[test]
    fn test_build_http1_client_succeeds() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        #[allow(deprecated)]
        std::env::remove_var("S3DLIO_H2C");
        // Must not panic
        let _ = build_reqwest_http_client();
    }

    /// Verify that enabling h2c via `S3DLIO_H2C=1` does not panic during client
    /// construction.  The actual HTTP/2 negotiation is only observable at connection
    /// time (requires a live server), so this test only validates the build path.
    #[test]
    fn test_build_h2c_client_succeeds() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let old_val = std::env::var("S3DLIO_H2C").ok();
        #[allow(deprecated)]
        std::env::set_var("S3DLIO_H2C", "1");
        let result = std::panic::catch_unwind(build_reqwest_http_client);
        // Restore previous value
        match old_val {
            #[allow(deprecated)]
            Some(v) => std::env::set_var("S3DLIO_H2C", v),
            #[allow(deprecated)]
            None => std::env::remove_var("S3DLIO_H2C"),
        }
        assert!(result.is_ok(), "build_reqwest_http_client() must not panic with S3DLIO_H2C=1");
    }

    /// All truthy `S3DLIO_H2C` variants produce a valid client.
    #[test]
    fn test_build_h2c_client_all_truthy_variants() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        for val in &["1", "true", "yes", "on", "enable"] {
            let old_val = std::env::var("S3DLIO_H2C").ok();
            #[allow(deprecated)]
            std::env::set_var("S3DLIO_H2C", val);
            let result = std::panic::catch_unwind(build_reqwest_http_client);
            match old_val {
                #[allow(deprecated)]
                Some(v) => std::env::set_var("S3DLIO_H2C", v),
                #[allow(deprecated)]
                None => std::env::remove_var("S3DLIO_H2C"),
            }
            assert!(result.is_ok(), "build failed for S3DLIO_H2C={val}");
        }
    }

    #[test]
    fn test_pool_settings_from_env() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let old_idle = std::env::var("S3DLIO_POOL_MAX_IDLE_PER_HOST").ok();
        let old_timeout = std::env::var("S3DLIO_POOL_IDLE_TIMEOUT_SECS").ok();
        #[allow(deprecated)]
        std::env::set_var("S3DLIO_POOL_MAX_IDLE_PER_HOST", "64");
        #[allow(deprecated)]
        std::env::set_var("S3DLIO_POOL_IDLE_TIMEOUT_SECS", "120");
        #[allow(deprecated)]
        std::env::remove_var("S3DLIO_H2C");
        let _ = build_reqwest_http_client(); // must not panic
        match old_idle {
            #[allow(deprecated)]
            Some(v) => std::env::set_var("S3DLIO_POOL_MAX_IDLE_PER_HOST", v),
            #[allow(deprecated)]
            None => std::env::remove_var("S3DLIO_POOL_MAX_IDLE_PER_HOST"),
        }
        match old_timeout {
            #[allow(deprecated)]
            Some(v) => std::env::set_var("S3DLIO_POOL_IDLE_TIMEOUT_SECS", v),
            #[allow(deprecated)]
            None => std::env::remove_var("S3DLIO_POOL_IDLE_TIMEOUT_SECS"),
        }
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
}
