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
//! | `auto`               | the first of the rows below whose variables are present (see below)  |
//! | `workload-identity`  | `WorkloadIdentityCredential` (AKS federated token)                    |
//! | `client-secret`      | `ClientSecretCredential` (service principal secret)                   |
//! | `client-certificate` | `ClientCertificateCredential` (PKCS#12; feature `azure-client-certificate`) |
//! | `managed-identity`   | `ManagedIdentityCredential` (IMDS, App Service, ...)                  |
//! | `cli`                | `DeveloperToolsCredential`: `az login`, then `azd auth login`          |
//! | `sas`                | no bearer token; `S3DLIO_AZURE_SAS_TOKEN` is appended to every URL    |
//! | `none`               | no credential at all (anonymous access, Azurite)                      |
//!
//! `auto` order: workload identity (`AZURE_FEDERATED_TOKEN_FILE` + `AZURE_CLIENT_ID` +
//! `AZURE_TENANT_ID`), client secret (`AZURE_CLIENT_ID` + `AZURE_TENANT_ID` +
//! `AZURE_CLIENT_SECRET`), client certificate (`AZURE_CLIENT_ID` + `AZURE_TENANT_ID` +
//! `AZURE_CLIENT_CERTIFICATE_PATH`), managed identity (`IDENTITY_ENDPOINT` or `MSI_ENDPOINT`
//! set, or the instance metadata service identifies an Azure VM), then the developer tools. `auto` never
//! picks `sas` or `none`.
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
        !matches!(
            self,
            CredentialMode::Auto | CredentialMode::Sas | CredentialMode::None
        )
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

/// What `auto` resolves to. `imds_reachable` is asked only when no other mode's variables are
/// present and no managed-identity endpoint variable is set.
pub fn resolve_auto(env: EnvLookup<'_>, imds_reachable: &dyn Fn() -> bool) -> CredentialMode {
    let client_and_tenant = present(env, ENV_AZURE_CLIENT_ID) && present(env, ENV_AZURE_TENANT_ID);
    if client_and_tenant && present(env, ENV_AZURE_FEDERATED_TOKEN_FILE) {
        return CredentialMode::WorkloadIdentity;
    }
    if client_and_tenant && present(env, ENV_AZURE_CLIENT_SECRET) {
        return CredentialMode::ClientSecret;
    }
    if client_and_tenant && present(env, ENV_AZURE_CLIENT_CERTIFICATE_PATH) {
        return CredentialMode::ClientCertificate;
    }
    if present(env, ENV_AZURE_IDENTITY_ENDPOINT)
        || present(env, ENV_AZURE_MSI_ENDPOINT)
        || imds_reachable()
    {
        return CredentialMode::ManagedIdentity;
    }
    CredentialMode::Cli
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

/// Ask the instance metadata service at `addr` whether this is an Azure VM. Bounded by
/// `AZURE_IMDS_PROBE_TIMEOUT_MS` for the connection and for each read.
fn probe_azure_imds(addr: std::net::SocketAddr) -> bool {
    use std::io::{Read, Write};
    let timeout = std::time::Duration::from_millis(AZURE_IMDS_PROBE_TIMEOUT_MS);
    let Ok(mut stream) = std::net::TcpStream::connect_timeout(&addr, timeout) else {
        return false;
    };
    if stream.set_read_timeout(Some(timeout)).is_err()
        || stream.set_write_timeout(Some(timeout)).is_err()
    {
        return false;
    }
    let request = format!(
        "GET /metadata/instance?api-version=2021-02-01 HTTP/1.0\r\nHost: {}\r\nMetadata: true\r\n\r\n",
        addr.ip()
    );
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = Vec::new();
    // The document is well under the cap; stop at the cap, EOF or the read timeout.
    let _ = stream.take(16 * 1024).read_to_end(&mut response);
    is_azure_imds_response(&response)
}

/// The address Azure serves the instance metadata service on.
fn azure_imds_addr() -> std::net::SocketAddr {
    std::net::SocketAddr::from(([169, 254, 169, 254], 80))
}

fn imds_reachable() -> bool {
    probe_azure_imds(azure_imds_addr())
}

/// The concrete mode (never `Auto`) for a request to an `https` (or `http`) endpoint, from the
/// process environment.
///
/// `auto` is resolved on every call from the variables, except that the instance-metadata probe
/// runs once per process. `auto` on an `http` endpoint (an emulator such as Azurite) is `none`,
/// without probing: a bearer token cannot be sent over http anyway.
pub fn mode_for_endpoint(https: bool) -> Result<CredentialMode> {
    static IMDS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let probe = || *IMDS.get_or_init(imds_reachable);
    Ok(match configured_mode(&process_env)? {
        CredentialMode::Auto if !https => CredentialMode::None,
        CredentialMode::Auto => {
            let chosen = resolve_auto(&process_env, &probe);
            static LOGGED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
            LOGGED.get_or_init(|| info!("Azure credential mode auto resolved to {chosen}"));
            chosen
        }
        explicit => explicit,
    })
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
        CredentialMode::Auto | CredentialMode::Sas | CredentialMode::None => {
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
    match sas_token(&process_env) {
        Ok(token) => text.replace(&token, "<SAS token redacted>"),
        Err(_) => text.to_string(),
    }
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
    fn auto_follows_the_documented_order() {
        let never = || false;
        let always = || true;
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
        // Workload identity beats everything.
        assert_eq!(
            resolve_auto(&env_of(&all), &always),
            CredentialMode::WorkloadIdentity
        );
        // Without the token file: client secret, then certificate, then managed identity.
        let no_federated: Vec<_> = all
            .iter()
            .copied()
            .filter(|(k, _)| *k != ENV_AZURE_FEDERATED_TOKEN_FILE)
            .collect();
        assert_eq!(
            resolve_auto(&env_of(&no_federated), &always),
            CredentialMode::ClientSecret
        );
        let no_secret: Vec<_> = no_federated
            .iter()
            .copied()
            .filter(|(k, _)| *k != ENV_AZURE_CLIENT_SECRET)
            .collect();
        assert_eq!(
            resolve_auto(&env_of(&no_secret), &always),
            CredentialMode::ClientCertificate
        );
        let no_cert: Vec<_> = no_secret
            .iter()
            .copied()
            .filter(|(k, _)| *k != ENV_AZURE_CLIENT_CERTIFICATE_PATH)
            .collect();
        assert_eq!(
            resolve_auto(&env_of(&no_cert), &never),
            CredentialMode::ManagedIdentity
        );
        // A partial service principal (no tenant) is not one.
        let partial = [(ENV_AZURE_CLIENT_ID, "c"), (ENV_AZURE_CLIENT_SECRET, "s")];
        assert_eq!(resolve_auto(&env_of(&partial), &never), CredentialMode::Cli);
        // Managed identity from the host markers or the metadata service; the probe is only
        // consulted when nothing else decides.
        assert_eq!(
            resolve_auto(&env_of(&[(ENV_AZURE_MSI_ENDPOINT, "http://m")]), &|| {
                panic!("the probe must not run when MSI_ENDPOINT is set")
            }),
            CredentialMode::ManagedIdentity
        );
        assert_eq!(
            resolve_auto(&env_of(&[]), &always),
            CredentialMode::ManagedIdentity
        );
        // Nothing at all: the developer tools.
        assert_eq!(resolve_auto(&env_of(&[]), &never), CredentialMode::Cli);
        // `auto` never picks sas or none, even with a SAS token present.
        assert_eq!(
            resolve_auto(&env_of(&[(ENV_AZURE_SAS_TOKEN, "sv=1&sig=x")]), &never),
            CredentialMode::Cli
        );
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

    #[test]
    fn only_bearer_token_modes_need_https() {
        for mode in CredentialMode::ALL {
            let expected = !matches!(
                mode,
                CredentialMode::Auto | CredentialMode::Sas | CredentialMode::None
            );
            assert_eq!(mode.uses_bearer_token(), expected, "{mode}");
        }
    }
}
