// src/azure_credential.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Which credential the Azure Blob backend authenticates with (issue #253).
//!
//! `S3DLIO_AZURE_CREDENTIAL` selects the mode; unset means `auto`:
//!
//! | mode                 | what is used                                                         |
//! |----------------------|----------------------------------------------------------------------|
//! | `auto`               | a fallback chain over the rows below (see below)                      |
//! | `workload-identity`  | `WorkloadIdentityCredential` (AKS federated token)                    |
//! | `client-secret`      | `ClientSecretCredential` (service principal secret)                   |
//! | `client-certificate` | `ClientCertificateCredential` (PKCS#12; feature `azure-client-certificate`) |
//! | `managed-identity`   | `ManagedIdentityCredential` (IMDS, App Service, ...)                  |
//! | `cli`                | `DeveloperToolsCredential`: `az login`, then `azd auth login`          |
//! | `sas`                | no bearer token; `S3DLIO_AZURE_SAS_TOKEN` is appended to every URL    |
//! | `none`               | no credential at all (anonymous access, Azurite)                      |
//!
//! `auto` is a real fallback chain, like the Azure SDKs' `DefaultAzureCredential`: it tries, in
//! order, workload identity (`AZURE_FEDERATED_TOKEN_FILE` + `AZURE_CLIENT_ID` +
//! `AZURE_TENANT_ID`), client secret (`AZURE_CLIENT_ID` + `AZURE_TENANT_ID` +
//! `AZURE_CLIENT_SECRET`), client certificate (`AZURE_CLIENT_ID` + `AZURE_TENANT_ID` +
//! `AZURE_CLIENT_CERTIFICATE_PATH`), managed identity (`IDENTITY_ENDPOINT` or `MSI_ENDPOINT`
//! set, or the instance metadata service identifies an Azure VM), then the developer tools, and
//! uses the first that yields a token. A candidate whose variables are not set is skipped; one
//! that fails (a managed identity that is not assigned, a rejected secret) falls through to the
//! next. The winner is remembered for the process; if every candidate fails the error lists
//! each one and why. `auto` never picks `sas` or `none`.
//!
//! Each credential is built once per process and shared, so the SDK's token cache (and, for
//! the non-CLI modes, its refresh) is used and no process is spawned per request.

use anyhow::{anyhow, bail, Result};
use azure_core::credentials::{AccessToken, Secret, TokenCredential, TokenRequestOptions};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

use crate::constants::{
    AZURE_IMDS_PROBE_TIMEOUT_MS, ENV_AZURE_CLIENT_CERTIFICATE_PATH, ENV_AZURE_CLIENT_ID,
    ENV_AZURE_CLIENT_SECRET, ENV_AZURE_CREDENTIAL, ENV_AZURE_FEDERATED_TOKEN_FILE,
    ENV_AZURE_IDENTITY_ENDPOINT, ENV_AZURE_MSI_ENDPOINT, ENV_AZURE_SAS_TOKEN, ENV_AZURE_TENANT_ID,
};

/// The credential modes. See the module documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CredentialMode {
    Auto,
    Cli,
    ManagedIdentity,
    ClientSecret,
    ClientCertificate,
    WorkloadIdentity,
    Sas,
    None,
}

impl CredentialMode {
    /// Every mode, in the order the error message lists them.
    pub const ALL: [CredentialMode; 8] = [
        CredentialMode::Auto,
        CredentialMode::Cli,
        CredentialMode::ManagedIdentity,
        CredentialMode::ClientSecret,
        CredentialMode::ClientCertificate,
        CredentialMode::WorkloadIdentity,
        CredentialMode::Sas,
        CredentialMode::None,
    ];

    /// The name used in `S3DLIO_AZURE_CREDENTIAL` and in error messages.
    pub fn as_str(self) -> &'static str {
        match self {
            CredentialMode::Auto => "auto",
            CredentialMode::Cli => "cli",
            CredentialMode::ManagedIdentity => "managed-identity",
            CredentialMode::ClientSecret => "client-secret",
            CredentialMode::ClientCertificate => "client-certificate",
            CredentialMode::WorkloadIdentity => "workload-identity",
            CredentialMode::Sas => "sas",
            CredentialMode::None => "none",
        }
    }

    /// Parse a mode name, ignoring case and surrounding whitespace.
    pub fn parse(value: &str) -> Result<Self> {
        let wanted = value.trim().to_ascii_lowercase();
        Self::ALL
            .into_iter()
            .find(|mode| mode.as_str() == wanted)
            .ok_or_else(|| {
                anyhow!(
                    "unknown {ENV_AZURE_CREDENTIAL} value '{value}'; valid values: {}",
                    Self::ALL.map(CredentialMode::as_str).join(", ")
                )
            })
    }

    /// Whether this mode authenticates with a bearer token (and so needs an https endpoint).
    pub fn uses_bearer_token(self) -> bool {
        !matches!(self, CredentialMode::Sas | CredentialMode::None)
    }
}

impl fmt::Display for CredentialMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Environment lookup: the process environment in production, a map in tests.
pub type EnvLookup<'a> = &'a dyn Fn(&str) -> Option<String>;

fn process_env(name: &str) -> Option<String> {
    std::env::var(name).ok()
}

fn present(env: EnvLookup<'_>, name: &str) -> bool {
    env(name).is_some_and(|v| !v.trim().is_empty())
}

/// The configured mode: `S3DLIO_AZURE_CREDENTIAL`, or `auto` when unset or empty.
pub fn configured_mode(env: EnvLookup<'_>) -> Result<CredentialMode> {
    match env(ENV_AZURE_CREDENTIAL) {
        Some(value) if !value.trim().is_empty() => CredentialMode::parse(&value),
        _ => Ok(CredentialMode::Auto),
    }
}

/// The `auto` chain, in order.
const AUTO_CHAIN: [CredentialMode; 5] = [
    CredentialMode::WorkloadIdentity,
    CredentialMode::ClientSecret,
    CredentialMode::ClientCertificate,
    CredentialMode::ManagedIdentity,
    CredentialMode::Cli,
];

/// Whether the variables a chain member needs are present. Managed identity is decided by the
/// caller (host markers or the metadata probe); the developer tools need none.
fn auto_member_configured(mode: CredentialMode, env: EnvLookup<'_>) -> bool {
    let client_and_tenant = present(env, ENV_AZURE_CLIENT_ID) && present(env, ENV_AZURE_TENANT_ID);
    match mode {
        CredentialMode::WorkloadIdentity => {
            client_and_tenant && present(env, ENV_AZURE_FEDERATED_TOKEN_FILE)
        }
        CredentialMode::ClientSecret => client_and_tenant && present(env, ENV_AZURE_CLIENT_SECRET),
        CredentialMode::ClientCertificate => {
            client_and_tenant && present(env, ENV_AZURE_CLIENT_CERTIFICATE_PATH)
        }
        CredentialMode::ManagedIdentity => {
            present(env, ENV_AZURE_IDENTITY_ENDPOINT) || present(env, ENV_AZURE_MSI_ENDPOINT)
        }
        _ => true,
    }
}

/// The members of the `auto` chain that apply, in order, given the environment and whether the
/// instance metadata service identifies an Azure VM. (The running chain asks the metadata
/// service lazily; this is the same decision, for inspection and tests.)
pub fn auto_candidates(env: EnvLookup<'_>, imds_is_azure: bool) -> Vec<CredentialMode> {
    AUTO_CHAIN
        .into_iter()
        .filter(|mode| {
            auto_member_configured(*mode, env)
                || (*mode == CredentialMode::ManagedIdentity && imds_is_azure)
        })
        .collect()
}

/// Whether `response` is what the Azure instance metadata service answers to
/// `GET /metadata/instance`: `200` with the instance document (`"compute"`). Other clouds
/// serve their own metadata service on the same address (Google Compute Engine answers 400
/// to this request, AWS 404), so a bare TCP connection proves nothing.
fn is_azure_imds_response(response: &[u8]) -> bool {
    let text = String::from_utf8_lossy(response);
    let status_ok = text
        .lines()
        .next()
        .is_some_and(|line| line.split_whitespace().nth(1) == Some("200"));
    status_ok && text.contains("\"compute\"")
}

/// Ask the instance metadata service at `addr` whether this is an Azure VM.
///
/// Bounded: the connection gets `AZURE_IMDS_PROBE_TIMEOUT_MS` and then the whole read gets the
/// same again as one deadline (a server that drips bytes cannot extend it), so the worst case
/// is twice that. Blocking: the `auto` chain runs it on the blocking thread pool, never on a runtime worker.
fn probe_azure_imds(addr: std::net::SocketAddr) -> bool {
    use std::io::{Read, Write};
    let timeout = std::time::Duration::from_millis(AZURE_IMDS_PROBE_TIMEOUT_MS);
    let Ok(mut stream) = std::net::TcpStream::connect_timeout(&addr, timeout) else {
        return false;
    };
    if stream.set_write_timeout(Some(timeout)).is_err() {
        return false;
    }
    let request = format!(
        "GET /metadata/instance?api-version=2021-02-01 HTTP/1.0\r\nHost: {}\r\nMetadata: true\r\n\r\n",
        addr.ip()
    );
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let deadline = std::time::Instant::now() + timeout;
    let mut response = Vec::new();
    let mut chunk = [0u8; 2048];
    // The document is well under the cap; stop at the cap, EOF or the deadline.
    while response.len() < 16 * 1024 {
        let Some(left) = deadline.checked_duration_since(std::time::Instant::now()) else {
            break;
        };
        if stream
            .set_read_timeout(Some(left.max(std::time::Duration::from_millis(1))))
            .is_err()
        {
            break;
        }
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => response.extend_from_slice(&chunk[..n]),
        }
    }
    is_azure_imds_response(&response)
}

/// The address Azure serves the instance metadata service on.
fn azure_imds_addr() -> std::net::SocketAddr {
    std::net::SocketAddr::from(([169, 254, 169, 254], 80))
}

/// The configured mode for a request to an `https` (or `http`) endpoint, from the process
/// environment. `auto` on an `http` endpoint (an emulator such as Azurite) is `none`: a bearer
/// token cannot be sent over http anyway, so nothing is probed or tried.
pub fn mode_for_endpoint(https: bool) -> Result<CredentialMode> {
    Ok(match configured_mode(&process_env)? {
        CredentialMode::Auto if !https => CredentialMode::None,
        mode => mode,
    })
}

type SharedEnv = Arc<dyn Fn(&str) -> Option<String> + Send + Sync>;

type CredentialSource =
    Arc<dyn Fn(CredentialMode) -> Result<Arc<dyn TokenCredential>> + Send + Sync>;

/// The `auto` credential: tries the chain members that apply, in order, at the first token
/// request, and remembers the first that yields a token.
struct AutoCredential {
    env: SharedEnv,
    /// Does the instance metadata service identify an Azure VM? (blocking)
    imds_is_azure: Arc<dyn Fn() -> bool + Send + Sync>,
    credential_for: CredentialSource,
    winner: Mutex<Option<(CredentialMode, Arc<dyn TokenCredential>)>>,
}

impl fmt::Debug for AutoCredential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AutoCredential").finish_non_exhaustive()
    }
}

impl AutoCredential {
    fn production() -> Self {
        Self {
            env: Arc::new(process_env),
            imds_is_azure: Arc::new(|| probe_azure_imds(azure_imds_addr())),
            credential_for: Arc::new(credential_for),
            winner: Mutex::new(None),
        }
    }

    fn winner(&self) -> Option<(CredentialMode, Arc<dyn TokenCredential>)> {
        self.winner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    async fn try_member(
        &self,
        mode: CredentialMode,
        scopes: &[&str],
        options: &Option<TokenRequestOptions<'_>>,
        tried: &mut Vec<String>,
    ) -> Option<AccessToken> {
        let credential = match (self.credential_for)(mode) {
            Ok(c) => c,
            Err(e) => {
                tried.push(format!("{mode}: {e:#}"));
                return None;
            }
        };
        match credential.get_token(scopes, options.clone()).await {
            Ok(token) => {
                info!("Azure credential mode auto: using {mode}");
                *self.winner.lock().unwrap_or_else(|e| e.into_inner()) = Some((mode, credential));
                Some(token)
            }
            Err(e) => {
                let mut text = e.to_string();
                let mut source = std::error::Error::source(&e);
                while let Some(cause) = source {
                    let cause_text = cause.to_string();
                    if !text.contains(&cause_text) {
                        text.push_str(": ");
                        text.push_str(&cause_text);
                    }
                    source = cause.source();
                }
                tried.push(format!("{mode}: {text}"));
                None
            }
        }
    }

    /// Run the metadata probe on the blocking pool.
    async fn imds_says_azure(&self) -> bool {
        let probe = Arc::clone(&self.imds_is_azure);
        tokio::task::spawn_blocking(move || probe())
            .await
            .unwrap_or(false)
    }
}

#[async_trait::async_trait]
impl TokenCredential for AutoCredential {
    async fn get_token(
        &self,
        scopes: &[&str],
        options: Option<TokenRequestOptions<'_>>,
    ) -> azure_core::Result<AccessToken> {
        if let Some((_, credential)) = self.winner() {
            return credential.get_token(scopes, options).await;
        }
        let env: &(dyn Fn(&str) -> Option<String> + Send + Sync) = &*self.env;
        let mut tried: Vec<String> = Vec::new();
        let mut managed_identity_tried = false;
        for mode in AUTO_CHAIN {
            let applies = if mode == CredentialMode::ManagedIdentity {
                auto_member_configured(mode, env) || self.imds_says_azure().await
            } else {
                auto_member_configured(mode, env)
            };
            if !applies {
                continue;
            }
            managed_identity_tried |= mode == CredentialMode::ManagedIdentity;
            if let Some(token) = self.try_member(mode, scopes, &options, &mut tried).await {
                return Ok(token);
            }
        }
        // Nothing worked and the metadata service did not look like Azure: look once more (the
        // service can come up late on a fresh VM) before giving up.
        if !managed_identity_tried && self.imds_says_azure().await {
            if let Some(token) = self
                .try_member(
                    CredentialMode::ManagedIdentity,
                    scopes,
                    &options,
                    &mut tried,
                )
                .await
            {
                return Ok(token);
            }
        }
        Err(azure_core::Error::with_message(
            azure_core::error::ErrorKind::Credential,
            format!(
                "no credential in the chain yielded a token; tried: {}",
                tried.join("; ")
            ),
        ))
    }
}

/// The SAS token for `sas` mode, without a leading `?`. The error never contains the token.
pub fn sas_token(env: EnvLookup<'_>) -> Result<String> {
    let token = env(ENV_AZURE_SAS_TOKEN)
        .map(|t| t.trim().trim_start_matches('?').to_string())
        .filter(|t| !t.is_empty());
    token.ok_or_else(|| {
        anyhow!("Azure credential mode sas: {ENV_AZURE_SAS_TOKEN} is not set (or is empty)")
    })
}

/// Wraps a credential so that a token failure names the mode that was tried.
struct NamedCredential {
    mode: CredentialMode,
    inner: Arc<dyn TokenCredential>,
}

impl fmt::Debug for NamedCredential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NamedCredential")
            .field("mode", &self.mode)
            .finish_non_exhaustive()
    }
}

#[async_trait::async_trait]
impl TokenCredential for NamedCredential {
    async fn get_token(
        &self,
        scopes: &[&str],
        options: Option<TokenRequestOptions<'_>>,
    ) -> azure_core::Result<AccessToken> {
        self.inner.get_token(scopes, options).await.map_err(|e| {
            azure_core::Error::with_error(
                azure_core::error::ErrorKind::Credential,
                e,
                format!("Azure credential mode {}: could not get a token", self.mode),
            )
        })
    }
}

fn required(env: EnvLookup<'_>, name: &str, mode: CredentialMode) -> Result<String> {
    env(name)
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
        .ok_or_else(|| anyhow!("Azure credential mode {mode}: {name} is not set"))
}

/// Build the credential for a concrete bearer-token `mode` (not `auto`, `sas` or `none`).
fn build(mode: CredentialMode, env: EnvLookup<'_>) -> Result<Arc<dyn TokenCredential>> {
    let ctx = |e: azure_core::Error| anyhow!("Azure credential mode {mode}: {e}");
    let credential: Arc<dyn TokenCredential> = match mode {
        CredentialMode::Cli => azure_identity::DeveloperToolsCredential::new(None).map_err(ctx)?,
        CredentialMode::ManagedIdentity => {
            let user_assigned_id = env(ENV_AZURE_CLIENT_ID)
                .filter(|v| !v.trim().is_empty())
                .map(azure_identity::UserAssignedId::ClientId);
            azure_identity::ManagedIdentityCredential::new(Some(
                azure_identity::ManagedIdentityCredentialOptions {
                    user_assigned_id,
                    ..Default::default()
                },
            ))
            .map_err(ctx)?
        }
        CredentialMode::ClientSecret => azure_identity::ClientSecretCredential::new(
            &required(env, ENV_AZURE_TENANT_ID, mode)?,
            required(env, ENV_AZURE_CLIENT_ID, mode)?,
            Secret::new(required(env, ENV_AZURE_CLIENT_SECRET, mode)?),
            None,
        )
        .map_err(ctx)?,
        CredentialMode::WorkloadIdentity => {
            // The SDK reads AZURE_TENANT_ID, AZURE_CLIENT_ID and AZURE_FEDERATED_TOKEN_FILE
            // itself; check them here only to name the missing one.
            for name in [
                ENV_AZURE_TENANT_ID,
                ENV_AZURE_CLIENT_ID,
                ENV_AZURE_FEDERATED_TOKEN_FILE,
            ] {
                required(env, name, mode)?;
            }
            azure_identity::WorkloadIdentityCredential::new(None).map_err(ctx)?
        }
        CredentialMode::ClientCertificate => build_client_certificate(env)?,
        CredentialMode::Auto => Arc::new(AutoCredential::production()),
        CredentialMode::Sas | CredentialMode::None => {
            bail!("Azure credential mode {mode} has no bearer-token credential")
        }
    };
    Ok(Arc::new(NamedCredential {
        mode,
        inner: credential,
    }))
}

#[cfg(feature = "azure-client-certificate")]
fn build_client_certificate(env: EnvLookup<'_>) -> Result<Arc<dyn TokenCredential>> {
    use azure_core::credentials::SecretBytes;
    let mode = CredentialMode::ClientCertificate;
    let path = required(env, ENV_AZURE_CLIENT_CERTIFICATE_PATH, mode)?;
    let bytes = std::fs::read(&path)
        .map_err(|e| anyhow!("Azure credential mode {mode}: cannot read {path}: {e}"))?;
    let password =
        env(crate::constants::ENV_AZURE_CLIENT_CERTIFICATE_PASSWORD).filter(|p| !p.is_empty());
    azure_identity::ClientCertificateCredential::new(
        required(env, ENV_AZURE_TENANT_ID, mode)?,
        required(env, ENV_AZURE_CLIENT_ID, mode)?,
        SecretBytes::new(bytes),
        Some(azure_identity::ClientCertificateCredentialOptions {
            password: password.map(Secret::new),
            ..Default::default()
        }),
    )
    .map(|c| c as Arc<dyn TokenCredential>)
    .map_err(|e| {
        anyhow!(
            "Azure credential mode {mode}: {e} (the certificate must be a PKCS#12 / .pfx file \
             with an RSA key; set {} if it has a password)",
            crate::constants::ENV_AZURE_CLIENT_CERTIFICATE_PASSWORD
        )
    })
}

#[cfg(not(feature = "azure-client-certificate"))]
fn build_client_certificate(_env: EnvLookup<'_>) -> Result<Arc<dyn TokenCredential>> {
    bail!(
        "Azure credential mode client-certificate is not available in this build: rebuild with \
         the cargo feature `azure-client-certificate` (it links OpenSSL), or use \
         client-secret, workload-identity or managed-identity"
    )
}

/// One credential per mode per process. Nothing is cached when building fails, so a later call
/// retries.
static CREDENTIALS: std::sync::LazyLock<Mutex<HashMap<CredentialMode, Arc<dyn TokenCredential>>>> =
    std::sync::LazyLock::new(Default::default);

/// The process-wide credential for a concrete bearer-token `mode`, built on first use.
pub fn credential_for(mode: CredentialMode) -> Result<Arc<dyn TokenCredential>> {
    let mut credentials = CREDENTIALS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(credential) = credentials.get(&mode) {
        return Ok(Arc::clone(credential));
    }
    let credential = build(mode, &process_env)?;
    credentials.insert(mode, Arc::clone(&credential));
    Ok(credential)
}

/// Warn (once) that a SAS token is configured but the mode will not use it.
pub fn warn_unused_sas(mode: CredentialMode) {
    if mode != CredentialMode::Sas && present(&process_env, ENV_AZURE_SAS_TOKEN) {
        static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        WARNED.get_or_init(|| {
            warn!(
                "{ENV_AZURE_SAS_TOKEN} is set but the Azure credential mode is {mode}; the SAS \
                 token is ignored (set {ENV_AZURE_CREDENTIAL}=sas to use it)"
            )
        });
    }
}

/// Replace every occurrence of the configured SAS token in `text`, so an error message that
/// echoes a request URL cannot leak it.
pub fn redact_sas(text: &str) -> String {
    redact_with(text, sas_token(&process_env).ok().as_deref())
}

/// The SAS query parameters (service and account SAS, user-delegation SAS): their values are
/// credentials or reveal what the credential allows.
const SAS_PARAMS: &str =
    "sig|se|st|sp|spr|sv|sr|srt|ss|skoid|sktid|skt|ske|sks|skv|saoid|suoid|scid";

static SAS_PARAM_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(&format!(r#"(?i)\b({SAS_PARAMS})=[^&\s"'<>)\]\\]*"#)).expect("valid regex")
});

/// A URL in text, up to the first `?` or `#`: what follows (the query) is dropped.
static URL_QUERY_RE: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
    regex::Regex::new(r#"(?i)(https?://[^\s"'<>)\]\\?#]*)[?#][^\s"'<>)\]\\]*"#)
        .expect("valid regex")
});

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 3 <= bytes.len() {
            if let Ok(b) =
                u8::from_str_radix(std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or(""), 16)
            {
                out.push(b);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 2);
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// Remove SAS material from `text`, whatever form it takes: the SDK re-sorts and re-encodes a
/// query when it sets query pairs, and transport errors print the whole request URL, so an exact
/// copy of the configured token is not enough. Three layers:
/// 1. the query (and fragment) of every URL in the text is dropped;
/// 2. the value of every SAS parameter (`sig`, `se`, ... in raw or encoded `key=value` form) is
///    replaced;
/// 3. each value of the configured token is replaced wherever it occurs, as written, decoded,
///    encoded, and encoded twice (values shorter than 6 characters, like `sp=r`, are left to
///    layer 2).
fn redact_with(text: &str, token: Option<&str>) -> String {
    let mut out = URL_QUERY_RE
        .replace_all(text, "${1}?<query redacted>")
        .into_owned();
    out = SAS_PARAM_RE
        .replace_all(&out, "${1}=<redacted>")
        .into_owned();
    if let Some(token) = token {
        let mut secrets: Vec<String> = vec![token.to_string()];
        for pair in token.split('&') {
            let value = pair.split_once('=').map_or(pair, |(_, v)| v);
            let decoded = percent_decode(value);
            let encoded = percent_encode(&decoded);
            secrets.extend([
                value.to_string(),
                decoded.clone(),
                encoded.clone(),
                percent_encode(&encoded),
            ]);
        }
        secrets.retain(|v| v.len() >= 6);
        // Longest first, so a whole token goes before the values inside it.
        secrets.sort_by_key(|v| std::cmp::Reverse(v.len()));
        secrets.dedup();
        for secret in secrets {
            out = out.replace(&secret, "<redacted>");
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_of<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            pairs
                .iter()
                .find(|(k, _)| *k == name)
                .map(|(_, v)| v.to_string())
        }
    }

    #[test]
    fn modes_parse_case_insensitively_and_unknown_names_list_the_valid_ones() {
        for mode in CredentialMode::ALL {
            assert_eq!(CredentialMode::parse(mode.as_str()).unwrap(), mode);
            assert_eq!(
                CredentialMode::parse(&format!("  {}  ", mode.as_str().to_uppercase())).unwrap(),
                mode
            );
        }
        let err = CredentialMode::parse("password").unwrap_err().to_string();
        assert!(
            err.contains("'password'") && err.contains(ENV_AZURE_CREDENTIAL),
            "{err}"
        );
        for mode in CredentialMode::ALL {
            assert!(err.contains(mode.as_str()), "{err} lacks {mode}");
        }
    }

    #[test]
    fn unset_or_empty_selection_is_auto() {
        assert_eq!(configured_mode(&env_of(&[])).unwrap(), CredentialMode::Auto);
        assert_eq!(
            configured_mode(&env_of(&[(ENV_AZURE_CREDENTIAL, "  ")])).unwrap(),
            CredentialMode::Auto
        );
        assert_eq!(
            configured_mode(&env_of(&[(ENV_AZURE_CREDENTIAL, "SAS")])).unwrap(),
            CredentialMode::Sas
        );
        assert!(configured_mode(&env_of(&[(ENV_AZURE_CREDENTIAL, "nope")])).is_err());
    }

    #[test]
    fn auto_candidates_follow_the_documented_order() {
        let id = [(ENV_AZURE_CLIENT_ID, "c"), (ENV_AZURE_TENANT_ID, "t")];
        let with = |extra: &[(&'static str, &'static str)]| -> Vec<(&'static str, &'static str)> {
            let mut v = id.to_vec();
            v.extend_from_slice(extra);
            v
        };
        let all = with(&[
            (ENV_AZURE_FEDERATED_TOKEN_FILE, "/f"),
            (ENV_AZURE_CLIENT_SECRET, "s"),
            (ENV_AZURE_CLIENT_CERTIFICATE_PATH, "/c.pfx"),
            (ENV_AZURE_IDENTITY_ENDPOINT, "http://x"),
        ]);
        use CredentialMode::*;
        // Everything configured: the whole chain, in order.
        assert_eq!(
            auto_candidates(&env_of(&all), false),
            vec![
                WorkloadIdentity,
                ClientSecret,
                ClientCertificate,
                ManagedIdentity,
                Cli
            ]
        );
        // Members whose variables are missing are skipped, the rest keep their order.
        let no_federated: Vec<_> = all
            .iter()
            .copied()
            .filter(|(k, _)| *k != ENV_AZURE_FEDERATED_TOKEN_FILE)
            .collect();
        assert_eq!(
            auto_candidates(&env_of(&no_federated), false),
            vec![ClientSecret, ClientCertificate, ManagedIdentity, Cli]
        );
        // A partial service principal (no tenant) is not one.
        let partial = [(ENV_AZURE_CLIENT_ID, "c"), (ENV_AZURE_CLIENT_SECRET, "s")];
        assert_eq!(auto_candidates(&env_of(&partial), false), vec![Cli]);
        // Managed identity from a host marker, or because the metadata service says Azure.
        assert_eq!(
            auto_candidates(&env_of(&[(ENV_AZURE_MSI_ENDPOINT, "http://m")]), false),
            vec![ManagedIdentity, Cli]
        );
        assert_eq!(
            auto_candidates(&env_of(&[]), true),
            vec![ManagedIdentity, Cli]
        );
        // Nothing at all: the developer tools. `auto` never picks sas or none.
        assert_eq!(auto_candidates(&env_of(&[]), false), vec![Cli]);
        assert_eq!(
            auto_candidates(&env_of(&[(ENV_AZURE_SAS_TOKEN, "sv=1&sig=x")]), false),
            vec![Cli]
        );
    }

    /// A scripted credential: returns a token, or fails, and counts its calls.
    #[derive(Debug)]
    struct Scripted {
        token: Option<&'static str>,
        error: &'static str,
        calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait::async_trait]
    impl TokenCredential for Scripted {
        async fn get_token(
            &self,
            _scopes: &[&str],
            _options: Option<TokenRequestOptions<'_>>,
        ) -> azure_core::Result<AccessToken> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            match self.token {
                Some(t) => Ok(AccessToken::new(
                    t,
                    azure_core::time::OffsetDateTime::now_utc()
                        + azure_core::time::Duration::hours(1),
                )),
                None => Err(azure_core::Error::with_message(
                    azure_core::error::ErrorKind::Credential,
                    self.error,
                )),
            }
        }
    }

    type Calls = std::collections::HashMap<CredentialMode, Arc<std::sync::atomic::AtomicUsize>>;

    /// An `AutoCredential` over scripted members. `script`: (mode, token or error).
    fn scripted_auto(
        env: Vec<(&'static str, &'static str)>,
        imds: Arc<std::sync::atomic::AtomicUsize>,
        imds_answers: Vec<bool>,
        script: Vec<(CredentialMode, Result<&'static str, &'static str>)>,
    ) -> (AutoCredential, Calls) {
        let mut calls = Calls::new();
        let mut members = std::collections::HashMap::new();
        for (mode, outcome) in script {
            let counter: Arc<std::sync::atomic::AtomicUsize> = Arc::default();
            calls.insert(mode, Arc::clone(&counter));
            let credential: Arc<dyn TokenCredential> = Arc::new(Scripted {
                token: outcome.ok(),
                error: outcome.err().unwrap_or(""),
                calls: counter,
            });
            members.insert(mode, credential);
        }
        let env: Vec<(String, String)> = env
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let auto = AutoCredential {
            env: Arc::new(move |name| env.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone())),
            imds_is_azure: Arc::new(move || {
                let n = imds.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                imds_answers.get(n).copied().unwrap_or(false)
            }),
            credential_for: Arc::new(move |mode| {
                members
                    .get(&mode)
                    .cloned()
                    .ok_or_else(|| anyhow!("{mode} is not available"))
            }),
            winner: Mutex::new(None),
        };
        (auto, calls)
    }

    fn called(calls: &Calls, mode: CredentialMode) -> usize {
        calls
            .get(&mode)
            .map_or(0, |c| c.load(std::sync::atomic::Ordering::SeqCst))
    }

    // `auto` is a fallback chain: a managed identity that is not assigned falls through to the
    // developer tools, and the winner is used (alone) from then on.
    #[tokio::test]
    async fn auto_falls_through_failing_members_and_remembers_the_winner() {
        use CredentialMode::{Cli, ManagedIdentity};
        let imds: Arc<std::sync::atomic::AtomicUsize> = Arc::default();
        let (auto, calls) = scripted_auto(
            vec![(ENV_AZURE_IDENTITY_ENDPOINT, "http://x")],
            imds,
            vec![],
            vec![
                (ManagedIdentity, Err("identity not assigned to this VM")),
                (Cli, Ok("cli-token")),
            ],
        );
        let token = auto
            .get_token(&["s"], None)
            .await
            .expect("falls through to the CLI");
        assert_eq!(token.token.secret(), "cli-token");
        assert_eq!(
            (called(&calls, ManagedIdentity), called(&calls, Cli)),
            (1, 1)
        );

        // Later requests go straight to the winner.
        auto.get_token(&["s"], None).await.unwrap();
        assert_eq!(
            (called(&calls, ManagedIdentity), called(&calls, Cli)),
            (1, 2)
        );
    }

    #[tokio::test]
    async fn auto_prefers_the_earlier_member_when_it_works() {
        use CredentialMode::{Cli, ClientSecret};
        let (auto, calls) = scripted_auto(
            vec![
                (ENV_AZURE_CLIENT_ID, "c"),
                (ENV_AZURE_TENANT_ID, "t"),
                (ENV_AZURE_CLIENT_SECRET, "s"),
            ],
            Arc::default(),
            vec![],
            vec![(ClientSecret, Ok("secret-token")), (Cli, Ok("cli-token"))],
        );
        let token = auto.get_token(&["s"], None).await.unwrap();
        assert_eq!(token.token.secret(), "secret-token");
        assert_eq!(
            called(&calls, Cli),
            0,
            "the developer tools were not touched"
        );
    }

    // If every member fails the error lists each one and why; a metadata service that answers
    // late is looked at once more before giving up.
    #[tokio::test]
    async fn auto_error_lists_every_member_tried_and_re_probes_once() {
        use CredentialMode::{Cli, ClientSecret, ManagedIdentity};
        let imds: Arc<std::sync::atomic::AtomicUsize> = Arc::default();
        let (auto, calls) = scripted_auto(
            vec![
                (ENV_AZURE_CLIENT_ID, "c"),
                (ENV_AZURE_TENANT_ID, "t"),
                (ENV_AZURE_CLIENT_SECRET, "s"),
            ],
            Arc::clone(&imds),
            vec![false, true], // not Azure at first, Azure on the second look
            vec![
                (ClientSecret, Err("AADSTS7000215 invalid client secret")),
                (ManagedIdentity, Err("identity not assigned")),
                (Cli, Err("az not found on PATH")),
            ],
        );
        let err = auto.get_token(&["s"], None).await.unwrap_err().to_string();
        for needle in [
            "client-secret: AADSTS7000215",
            "cli: az not found on PATH",
            "managed-identity: identity not assigned",
        ] {
            assert!(err.contains(needle), "{needle} missing from: {err}");
        }
        assert_eq!(
            imds.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "probed twice, no more"
        );
        assert_eq!(
            called(&calls, ManagedIdentity),
            1,
            "tried after the second probe"
        );
    }

    // The probe never blocks longer than its bound, even against a server that drips bytes.
    #[test]
    fn the_metadata_probe_is_bounded_in_time() {
        use std::io::Write;
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            if let Ok((mut conn, _)) = listener.accept() {
                for _ in 0..40 {
                    if conn.write_all(b"x").is_err() {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(50));
                }
            }
        });
        let started = std::time::Instant::now();
        assert!(!probe_azure_imds(addr));
        let took = started.elapsed();
        assert!(
            took < std::time::Duration::from_millis(2 * AZURE_IMDS_PROBE_TIMEOUT_MS + 700),
            "the probe took {took:?}"
        );
        drop(server);
    }

    #[test]
    fn sas_token_is_trimmed_and_never_echoed_in_its_error() {
        assert_eq!(
            sas_token(&env_of(&[(ENV_AZURE_SAS_TOKEN, " ?sv=1&sig=abc ")])).unwrap(),
            "sv=1&sig=abc"
        );
        let err = sas_token(&env_of(&[(ENV_AZURE_SAS_TOKEN, "?")]))
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("mode sas") && err.contains(ENV_AZURE_SAS_TOKEN),
            "{err}"
        );
    }

    #[test]
    fn construction_errors_name_the_mode_and_the_missing_variable() {
        for (mode, missing) in [
            (CredentialMode::ClientSecret, ENV_AZURE_TENANT_ID),
            (CredentialMode::WorkloadIdentity, ENV_AZURE_TENANT_ID),
            (
                CredentialMode::ClientCertificate,
                ENV_AZURE_CLIENT_CERTIFICATE_PATH,
            ),
        ] {
            let err = build(mode, &env_of(&[]))
                .expect_err("must fail")
                .to_string();
            assert!(
                err.contains(&format!("Azure credential mode {mode}")),
                "{err}"
            );
            #[cfg(not(feature = "azure-client-certificate"))]
            if mode == CredentialMode::ClientCertificate {
                assert!(err.contains("azure-client-certificate"), "{err}");
                continue;
            }
            assert!(err.contains(missing), "{err}");
        }
    }

    #[derive(Debug)]
    struct Failing;

    #[async_trait::async_trait]
    impl TokenCredential for Failing {
        async fn get_token(
            &self,
            _scopes: &[&str],
            _options: Option<TokenRequestOptions<'_>>,
        ) -> azure_core::Result<AccessToken> {
            Err(azure_core::Error::with_message(
                azure_core::error::ErrorKind::Credential,
                "the identity server answered 400",
            ))
        }
    }

    // A failed token request names the mode that was tried and keeps the cause.
    #[tokio::test]
    async fn a_token_failure_names_the_mode_and_keeps_the_cause() {
        let named = NamedCredential {
            mode: CredentialMode::WorkloadIdentity,
            inner: Arc::new(Failing),
        };
        let err = named
            .get_token(&["https://storage.azure.com/.default"], None)
            .await
            .unwrap_err();
        let text = format!(
            "{err} / {:?}",
            std::error::Error::source(&err).map(|s| s.to_string())
        );
        assert!(
            text.contains("Azure credential mode workload-identity"),
            "{text}"
        );
        assert!(text.contains("the identity server answered 400"), "{text}");
        assert_eq!(*err.kind(), azure_core::error::ErrorKind::Credential);
    }

    #[test]
    fn only_an_azure_instance_document_counts_as_azure() {
        assert!(is_azure_imds_response(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"compute\":{\"vmId\":\"x\"}}"
        ));
        // Google Compute Engine answers 400 to this request, AWS 404; neither is Azure.
        assert!(!is_azure_imds_response(
            b"HTTP/1.0 400 Bad Request\r\nMetadata-Flavor: Google\r\n\r\n"
        ));
        assert!(!is_azure_imds_response(b"HTTP/1.1 404 Not Found\r\n\r\n"));
        // A 200 that is not an Azure instance document.
        assert!(!is_azure_imds_response(b"HTTP/1.1 200 OK\r\n\r\nhello"));
        assert!(!is_azure_imds_response(b""));
    }

    /// Serve one canned response on a local port and probe it.
    fn probe_against(canned: &'static [u8]) -> bool {
        use std::io::{Read, Write};
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            if let Ok((mut conn, _)) = listener.accept() {
                let mut request = [0u8; 1024];
                let n = conn.read(&mut request).unwrap_or(0);
                let request = String::from_utf8_lossy(&request[..n]).to_lowercase();
                assert!(request.contains("get /metadata/instance"), "{request}");
                assert!(request.contains("metadata: true"), "{request}");
                let _ = conn.write_all(canned);
            }
        });
        let answer = probe_azure_imds(addr);
        server.join().unwrap();
        answer
    }

    #[test]
    fn the_probe_accepts_azure_and_rejects_other_clouds_and_closed_ports() {
        assert!(probe_against(
            b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n{\"compute\":{}}"
        ));
        assert!(!probe_against(
            b"HTTP/1.0 400 Bad Request\r\nMetadata-Flavor: Google\r\n\r\n"
        ));
        // Nothing listening: the connection is refused at once.
        let closed = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap()
        };
        assert!(!probe_azure_imds(closed));
    }

    const TOKEN: &str =
        "sv=2024-11-04&ss=b&srt=sco&sp=rwdl&se=2099-01-01T00%3A00%3A00Z&sig=abc%2Bdef%2Fghij%3D";

    // The SDK re-sorts and re-encodes a query when it sets query pairs (list, container
    // operations, stage_block, commit_block_list), and transport errors print the whole URL.
    #[test]
    fn redaction_does_not_depend_on_the_exact_token_text() {
        let texts = [
            // The token as configured.
            format!("GET https://h/acct/c/k?{TOKEN} failed"),
            // Re-sorted, with the query pairs the SDK adds, and the signature double-encoded.
            "error sending request for url (https://h/acct/c?comp=list&restype=container&sig=abc%252Bdef%252Fghij%253D&sv=2024-11-04&se=2099-01-01T00%253A00%253A00Z)".to_string(),
            // Decoded forms of the values, with no URL around them.
            "the signature abc+def/ghij= was refused".to_string(),
            // Debug-ish rendering.
            r#"Url { query: Some("restype=container&sig=abc%2Bdef%2Fghij%3D&comp=list") }"#.to_string(),
        ];
        for text in texts {
            let out = redact_with(&text, Some(TOKEN));
            for leak in [
                "abc%2Bdef",
                "abc%252Bdef",
                "abc+def/ghij=",
                "ghij",
                "2099-01-01",
            ] {
                assert!(
                    !out.contains(leak),
                    "RED: {leak:?} survives in {out:?} (from {text:?})"
                );
            }
        }
        // Text that merely mentions common words is left alone.
        let plain = "request failed: connection refused (os error 111) for https://h/acct/c/k";
        assert_eq!(redact_with(plain, Some(TOKEN)), plain);
    }

    #[test]
    fn a_url_loses_its_query_even_without_a_configured_token() {
        let out = redact_with(
            "error sending request for url (https://h:1/a/b?x=1&sig=zzz) boom",
            None,
        );
        assert!(!out.contains("zzz") && !out.contains("x=1"), "{out}");
        assert!(out.contains("https://h:1/a/b"), "{out}");
        assert!(out.contains("boom"), "text after the URL is kept: {out}");
    }

    #[test]
    fn only_bearer_token_modes_need_https() {
        for mode in CredentialMode::ALL {
            let expected = !matches!(mode, CredentialMode::Sas | CredentialMode::None);
            assert_eq!(mode.uses_bearer_token(), expected, "{mode}");
        }
    }
}
