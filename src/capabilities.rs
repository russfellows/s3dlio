// src/capabilities.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Pure, environment-free capability answers (v0.10.0 plan §4.5 A3, issue #264).
//!
//! [`object_capabilities`] says whether this build can run the new caller-buffer GET and typed
//! HEAD on a URI. It is computed from the URI text and the compiled feature set ONLY:
//!
//! - no `store_for_uri*`, no client, no runtime, no network;
//! - no credential discovery, no environment read, no `.env` handling (`dotenv::apply`);
//! - no probe of whether a GCS bucket is an ordinary or a Rapid bucket: that is resolved during
//!   real I/O, never here.
//!
//! The URI is checked with the structural parsers in this module ([`parse_s3_uri_pure`],
//! [`parse_azure_uri_pure`], [`parse_gs_uri_pure`]), which read nothing but their argument. A
//! malformed URI is an error; an unknown scheme is not (it reports `"unknown"` with no support).
//!
//! # Structural validation, not full parity
//!
//! These parsers validate what can be known from the URI alone. Anything that depends on the
//! environment stays with the gated parser that builds a store: for Azure, the account name is
//! canonicalised (lower-cased and length-checked, or kept verbatim for an emulator) according to
//! the configured endpoint. A URI accepted here can therefore still be refused when a store is
//! built for it; a URI refused here is malformed in every environment.

use anyhow::{anyhow, bail, Context, Result};

use crate::object_store::{infer_scheme, Scheme};
use crate::s3_utils::{S3FirstSegment, S3UriComponents};

/// What this build can do for an object URI. See [`object_capabilities`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capabilities {
    /// The provider named by the URI: `"s3"`, `"az"`, `"gs"` (for `gs://` and `gcs://`),
    /// `"file"`, `"direct"` or `"unknown"`.
    pub provider: &'static str,
    /// The caller-buffer GET (`get_into`) is available.
    pub get_into: bool,
    /// The typed HEAD is available.
    pub head: bool,
}

impl Capabilities {
    const fn new(provider: &'static str, get_into: bool, head: bool) -> Self {
        Self {
            provider,
            get_into,
            head,
        }
    }
}

/// True when the native object-store implementation (not the Arrow one) is compiled.
const NATIVE_BACKENDS: bool = cfg!(all(
    feature = "native-backends",
    not(feature = "arrow-backend")
));

/// True when native Azure is compiled in.
const AZURE_COMPILED: bool = cfg!(feature = "backend-azure");

/// True when native GCS on the official pinned SDK is compiled in. The legacy community
/// implementation (`gcs-community`) does not acquire the new operations in this round.
const GCS_OFFICIAL_COMPILED: bool =
    cfg!(all(feature = "backend-gcs", not(feature = "gcs-community")));

/// Report what this build supports for `uri`, without credentials, a client or a runtime.
///
/// | scheme | `get_into` / `head` |
/// |---|---|
/// | `s3://` | true (native-backend builds) |
/// | `az://`, Azure blob host | true only when `backend-azure` is compiled |
/// | `gs://`, `gcs://` | true only with `backend-gcs` on the official SDK (not `gcs-community`) |
/// | `file://`, `direct://` | false |
/// | unknown scheme | provider `"unknown"`, false |
/// | any `arrow-backend` build | false |
///
/// # Errors
///
/// A malformed URI for a known object scheme (for example `az://account//key`, which has an empty
/// container). An unknown scheme is not an error.
pub fn object_capabilities(uri: &str) -> Result<Capabilities> {
    match infer_scheme(uri) {
        Scheme::S3 => {
            parse_s3_uri_pure(uri)?;
            Ok(Capabilities::new("s3", NATIVE_BACKENDS, NATIVE_BACKENDS))
        }
        Scheme::Azure => {
            parse_azure_uri_pure(uri)?;
            let on = NATIVE_BACKENDS && AZURE_COMPILED;
            Ok(Capabilities::new("az", on, on))
        }
        Scheme::Gcs => {
            parse_gs_uri_pure(uri)?;
            let on = NATIVE_BACKENDS && GCS_OFFICIAL_COMPILED;
            Ok(Capabilities::new("gs", on, on))
        }
        Scheme::File => Ok(Capabilities::new("file", false, false)),
        Scheme::Direct => Ok(Capabilities::new("direct", false, false)),
        Scheme::Unknown => Ok(Capabilities::new("unknown", false, false)),
    }
}

// ============================================================================
// S3
// ============================================================================

/// True if `s` is a dotted-quad IPv4 address (four `0..=255` octets).
fn looks_like_ipv4(s: &str) -> bool {
    let parts: Vec<&str> = s.split('.').collect();
    parts.len() == 4
        && parts.iter().all(|p| {
            !p.is_empty()
                && p.len() <= 3
                && p.chars().all(|c| c.is_ascii_digit())
                && p.parse::<u16>().is_ok_and(|n| n <= 255)
        })
}

/// Validate the port of `host:port` / `[ipv6]:port`: plain ASCII digits, no leading zero,
/// `1..=65535`.
fn validate_s3_server_port(port: &str, segment: &str) -> Result<()> {
    if port.is_empty() {
        bail!(
            "empty port in '{segment}': a server must be host:port with a port from 1 to 65535 \
             (e.g. s3://host:9000/bucket/key)"
        );
    }
    if !port.bytes().all(|b| b.is_ascii_digit()) {
        bail!(
            "invalid port '{port}' in '{segment}': a port is 1 to 65535 written as plain decimal \
             digits (no sign or other characters), e.g. s3://host:9000/bucket/key"
        );
    }
    if port.len() > 1 && port.starts_with('0') {
        bail!(
            "invalid port '{port}' in '{segment}': leading zeros are not accepted; write the \
             port without them (e.g. :9000)"
        );
    }
    match port.parse::<u16>() {
        Ok(n) if n >= 1 => Ok(()),
        _ => bail!(
            "invalid port '{port}' in '{segment}': a port must be from 1 to 65535 \
             (e.g. s3://host:9000/bucket/key)"
        ),
    }
}

/// Longest host the `host:port` grammar accepts (the DNS name limit).
const S3_SERVER_HOST_MAX_LEN: usize = 253;

/// The supported host grammar, quoted in every host error.
const S3_SERVER_HOST_GRAMMAR: &str = "a host is 1 to 253 ASCII letters, digits, '-', '.' and '_', \
     with no empty labels (no leading or trailing '.', no '..')";

/// Validate the host of an unbracketed `host:port`.
fn validate_s3_server_host(host: &str, segment: &str) -> Result<()> {
    if host.is_empty() {
        bail!(
            "missing host before ':' in '{segment}': use host:port, e.g. \
             s3://host:9000/bucket/key"
        );
    }
    if let Some(bad) = host
        .chars()
        .find(|c| !(c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_')))
    {
        bail!(
            "invalid character {bad:?} in host '{host}' of '{segment}': {S3_SERVER_HOST_GRAMMAR}"
        );
    }
    // Every character is ASCII here, so byte length is character length.
    if host.len() > S3_SERVER_HOST_MAX_LEN {
        bail!(
            "host of '{segment}' is {} characters, longer than the {S3_SERVER_HOST_MAX_LEN} \
             allowed: {S3_SERVER_HOST_GRAMMAR}",
            host.len()
        );
    }
    if host.split('.').any(str::is_empty) {
        bail!("empty label (a leading or trailing '.', or '..') in host '{host}' of '{segment}': {S3_SERVER_HOST_GRAMMAR}");
    }
    Ok(())
}

/// The environment-free form of `s3_utils::classify_s3_first_segment`: decides whether the first
/// path segment of an `s3://` URI is a server (`host:port`, `[ipv6]:port`) or a bucket, by
/// structure alone. The rules and error messages are the same; the only difference is that this
/// function never reads the environment (the existing wrapper additionally logs a one-shot
/// warning when a retired variable is set).
///
/// # Errors
///
/// An empty segment, a URL, userinfo, an ARN, a portless IP literal, or a malformed host, port
/// or bracket.
pub fn classify_s3_first_segment_pure(segment: &str) -> Result<S3FirstSegment<'_>> {
    if segment.is_empty() {
        bail!(
            "empty first segment in the S3 URI: expected a bucket name (s3://bucket/key) or a \
             server with a port (s3://host:9000/bucket/key)"
        );
    }
    if segment.contains("://") {
        bail!(
            "'{segment}' looks like a URL; a server is written host:port without a scheme \
             or path (e.g. host:9000)"
        );
    }
    if segment.contains('@') {
        bail!(
            "'{segment}' contains userinfo ('@'), which is not supported; write the server as \
             host:port and supply credentials through the environment"
        );
    }
    if segment.starts_with("arn:") && segment.matches(':').count() >= 2 {
        bail!(
            "'{segment}' looks like an ARN; ARNs are not supported here — use s3://bucket/key, \
             or s3://host:port/bucket/key to name a server"
        );
    }

    if let Some(rest) = segment.strip_prefix('[') {
        // Bracketed IPv6: "[addr]:port"
        let Some((addr, after)) = rest.split_once(']') else {
            bail!("unterminated '[' in '{segment}': an IPv6 server is written [addr]:port");
        };
        if addr.is_empty() {
            bail!("empty IPv6 address in '{segment}'");
        }
        if addr.parse::<std::net::Ipv6Addr>().is_err() {
            if addr.parse::<std::net::Ipv4Addr>().is_ok() {
                bail!(
                    "'{segment}' puts an IPv4 address inside brackets; brackets are for IPv6 \
                     only — write {addr}:port"
                );
            }
            bail!("invalid IPv6 address '{addr}' in '{segment}'");
        }
        if after.is_empty() {
            bail!(
                "'{segment}' names an IPv6 address without a port; add :port, \
                 e.g. s3://{segment}:9000/bucket/key"
            );
        }
        let Some(port) = after.strip_prefix(':') else {
            bail!(
                "unexpected text '{after}' after ']' in '{segment}': an IPv6 server is written \
                 [addr]:port"
            );
        };
        validate_s3_server_port(port, segment)?;
        return Ok(S3FirstSegment::Server(segment));
    }
    if segment.contains(['[', ']']) {
        bail!(
            "unexpected bracket in '{segment}': brackets are only for an IPv6 server, [addr]:port"
        );
    }

    match segment.matches(':').count() {
        0 => {
            if looks_like_ipv4(segment) {
                bail!(
                    "'{segment}' is an IP address with no port; add :port to name a server, \
                     e.g. s3://{segment}:9000/bucket/key (an IP address cannot be an S3 bucket name)"
                );
            }
            Ok(S3FirstSegment::Bucket(segment))
        }
        1 => {
            let (host, port) = segment.split_once(':').unwrap_or((segment, ""));
            validate_s3_server_host(host, segment)?;
            validate_s3_server_port(port, segment)?;
            Ok(S3FirstSegment::Server(segment))
        }
        _ if segment.parse::<std::net::Ipv6Addr>().is_ok() => bail!(
            "'{segment}' is an unbracketed IPv6 address; write it in brackets with a \
             port, e.g. s3://[{segment}]:9000/bucket/key (add :port)"
        ),
        _ => bail!(
            "'{segment}' contains more than one ':' and is not an IPv6 address; a server is \
             written host:port or [ipv6]:port"
        ),
    }
}

/// The error for `s3://host:9000`, `s3://host:9000/` and `s3://host:9000//key`.
fn missing_bucket_after_server(endpoint: &str, uri: &str) -> anyhow::Error {
    anyhow!(
        "missing bucket after server '{endpoint}' in S3 URI '{uri}': the bucket name cannot be \
         empty; write s3://{endpoint}/bucket/key"
    )
}

/// The environment-free form of `s3_utils::parse_s3_uri_full`: split `s3://bucket/key` or
/// `s3://host:port/bucket/key` into its parts. The key is exactly as written (`%2F` stays three
/// characters). Reads nothing but `uri`.
///
/// # Errors
///
/// A URI that does not start with `s3://`, has no `/` after the bucket, or whose first segment
/// is rejected by [`classify_s3_first_segment_pure`]; a server with no bucket after it.
pub fn parse_s3_uri_pure(uri: &str) -> Result<S3UriComponents> {
    let trimmed = uri
        .strip_prefix("s3://")
        .context("URI must start with s3://")?;

    let Some(slash_pos) = trimmed.find('/') else {
        // `s3://host:9000` is a valid server with no bucket, not a bucket with a missing slash.
        if let Ok(S3FirstSegment::Server(endpoint)) = classify_s3_first_segment_pure(trimmed) {
            return Err(missing_bucket_after_server(endpoint, uri));
        }
        bail!("URI must contain '/' after bucket");
    };
    let before_slash = &trimmed[..slash_pos];
    let after_slash = &trimmed[slash_pos + 1..];

    match classify_s3_first_segment_pure(before_slash)
        .with_context(|| format!("invalid S3 URI '{uri}'"))?
    {
        S3FirstSegment::Server(endpoint) => {
            // Format: s3://host:port/bucket/key
            let slash_pos2 = after_slash.find('/').unwrap_or(after_slash.len());
            let bucket = &after_slash[..slash_pos2];
            let key = after_slash.get(slash_pos2 + 1..).unwrap_or("");

            if bucket.is_empty() {
                return Err(missing_bucket_after_server(endpoint, uri));
            }

            Ok(S3UriComponents {
                endpoint: Some(endpoint.to_string()),
                bucket: bucket.to_string(),
                key: key.to_string(),
            })
        }
        S3FirstSegment::Bucket(bucket) => Ok(S3UriComponents {
            // Format: s3://bucket/key
            endpoint: None,
            bucket: bucket.to_string(),
            key: after_slash.to_string(),
        }),
    }
}

// ============================================================================
// Azure
// ============================================================================

/// True when the URI's authority (`host[:port]`, userinfo stripped) is, or ends with, an Azure
/// Blob Storage host name (`*.blob.core.windows.net`, case-insensitive). Only the authority is
/// examined, so a key that mentions the host name does not match.
///
/// This mirrors the private `object_store::is_azure_blob_host`; the two are meant to be merged
/// once that file is next opened for this work.
fn is_azure_blob_host_pure(uri: &str) -> bool {
    let Some((_, after_scheme)) = uri.split_once("://") else {
        return false;
    };
    let authority = after_scheme.split('/').next().unwrap_or("");
    let host = authority.rsplit('@').next().unwrap_or(authority);
    let host = host.split(':').next().unwrap_or(host);
    host.to_ascii_lowercase()
        .ends_with(".blob.core.windows.net")
}

/// Why an Azure account name could redirect a request or is not a usable name, whatever the
/// configured endpoint. The environment-dependent rules (lower-casing, the 3 to 24 character
/// length of a real account) stay in the gated canonicalisation.
fn azure_account_problem_pure(account: &str) -> Option<&'static str> {
    if account.is_empty() {
        return Some("it is empty");
    }
    if account.chars().any(|c| {
        matches!(c, '/' | '?' | '#' | '@' | '\\' | '%') || c.is_whitespace() || c.is_control()
    }) {
        return Some(
            "it contains a character that could redirect the request (/ ? # @ \\ % or whitespace)",
        );
    }
    (account == "." || account == "..").then_some("it is a path segment")
}

fn check_azure_account_pure(account: &str) -> Result<()> {
    azure_account_problem_pure(account).map_or(Ok(()), |why| {
        Err(anyhow!("invalid Azure account name {account:?}: {why}"))
    })
}

/// Split an Azure URI into `(account, container, key)` by structure alone.
///
/// Accepted forms:
/// - `az://{account}/{container}/{key...}`
/// - `http(s)://{account}.blob.core.windows.net/{container}/{key...}`
///
/// The key is exactly as written after the single `/` that follows the container (`%2F` stays
/// three characters; `c//k` is the container `c` and the key `/k`). The account is returned as
/// written: lower-casing and endpoint-dependent checks belong to the gated
/// `object_store::parse_azure_uri`, which delegates its structural part to this function.
///
/// **This is structural validation, not full parity with store construction.** It refuses what
/// is malformed in every environment: a missing or empty account, an account with a character
/// that could redirect the request, and an EMPTY CONTAINER (issue #224: `az://account//key`).
///
/// # Errors
///
/// As above, and a URI that is neither an `az://` URI nor an Azure blob host URL.
pub fn parse_azure_uri_pure(uri: &str) -> Result<(String, String, String)> {
    if let Some(rest) = uri.strip_prefix("az://") {
        let mut it = rest.splitn(3, '/');
        let account = it.next().unwrap_or("");
        let container = it
            .next()
            .ok_or_else(|| anyhow!("missing container in az:// URI"))?;
        let key = it.next().unwrap_or("");
        check_azure_account_pure(account)?;
        if container.is_empty() {
            bail!("empty container in az:// URI (expected az://account/container/key)");
        }
        return Ok((account.to_string(), container.to_string(), key.to_string()));
    }

    if is_azure_blob_host_pure(uri) {
        // The scheme (http or https) does not change which account, container and key the URL
        // names.
        let after_scheme = uri
            .split_once("://")
            .map(|(_, rest)| rest)
            .ok_or_else(|| anyhow!("expected a scheme (http:// or https://) for Azure URL"))?;
        let (host, path) = after_scheme.split_once('/').unwrap_or((after_scheme, ""));
        let account = host.split('.').next().unwrap_or("");
        // The first path segment is the container; the single `/` after it is the separator,
        // and the rest is the key exactly as written (issue #198).
        let (container, key) = path.split_once('/').unwrap_or((path, ""));
        check_azure_account_pure(account)?;
        if container.is_empty() {
            bail!("missing container in URL path");
        }
        return Ok((account.to_string(), container.to_string(), key.to_string()));
    }

    bail!("not a recognized Azure URI: {}", uri)
}

// ============================================================================
// GCS
// ============================================================================

/// Split `gs://bucket/object` or `gcs://bucket/object` into `(bucket, object)`.
///
/// A bucket-only URI (`gs://bucket/`) gives an empty object path. The object path is exactly as
/// written. This is the environment-free form of both gated `parse_gcs_uri` functions
/// (`google_gcs_client` and the community `gcs_client`).
///
/// # Errors
///
/// A URI without a `gs://` or `gcs://` prefix, or with an empty bucket.
pub fn parse_gs_uri_pure(uri: &str) -> Result<(String, String)> {
    let path = uri
        .strip_prefix("gs://")
        .or_else(|| uri.strip_prefix("gcs://"))
        .ok_or_else(|| anyhow!("Invalid GCS URI (expected gs:// or gcs:// prefix): {}", uri))?;

    let (bucket, object_path) = path.split_once('/').unwrap_or((path, ""));
    if bucket.is_empty() {
        bail!("Invalid GCS URI (empty bucket name): {}", uri);
    }
    Ok((bucket.to_string(), object_path.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_env::{lock, EnvRestore};

    /// Every environment variable that credential discovery, endpoint routing or `.env` handling
    /// could read for any provider.
    const CREDENTIAL_AND_ENDPOINT_VARS: &[&str] = &[
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_PROFILE",
        "AWS_ENDPOINT_URL",
        "AWS_REGION",
        "AZURE_STORAGE_ACCOUNT",
        "AZURE_STORAGE_KEY",
        "AZURE_STORAGE_ENDPOINT",
        "AZURE_BLOB_ENDPOINT_URL",
        "AZURE_CLIENT_ID",
        "AZURE_CLIENT_SECRET",
        "AZURE_TENANT_ID",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GCS_ENDPOINT_URL",
        "STORAGE_EMULATOR_HOST",
    ];

    // ---- results per build -------------------------------------------------------------

    fn expect_azure() -> bool {
        cfg!(all(
            feature = "backend-azure",
            feature = "native-backends",
            not(feature = "arrow-backend")
        ))
    }
    fn expect_gcs() -> bool {
        cfg!(all(
            feature = "backend-gcs",
            not(feature = "gcs-community"),
            feature = "native-backends",
            not(feature = "arrow-backend")
        ))
    }
    fn expect_s3() -> bool {
        cfg!(all(
            feature = "native-backends",
            not(feature = "arrow-backend")
        ))
    }

    fn caps(uri: &str) -> Capabilities {
        object_capabilities(uri).unwrap_or_else(|e| panic!("{uri} must be accepted: {e:#}"))
    }

    #[test]
    fn s3_reports_per_build() {
        for uri in [
            "s3://bucket/key",
            "s3://bucket/",
            "s3://host:9000/bucket/key",
            "s3://[::1]:9000/bucket/key",
        ] {
            assert_eq!(
                caps(uri),
                Capabilities {
                    provider: "s3",
                    get_into: expect_s3(),
                    head: expect_s3()
                },
                "{uri}"
            );
        }
    }

    #[test]
    fn azure_reports_per_build() {
        for uri in [
            "az://account/container/key",
            "az://account/container",
            "https://account.blob.core.windows.net/container/key",
            "http://account.blob.core.windows.net/container/key",
        ] {
            assert_eq!(
                caps(uri),
                Capabilities {
                    provider: "az",
                    get_into: expect_azure(),
                    head: expect_azure()
                },
                "{uri}"
            );
        }
    }

    #[test]
    fn gcs_reports_per_build() {
        for uri in ["gs://bucket/key", "gcs://bucket/key", "gs://bucket/"] {
            assert_eq!(
                caps(uri),
                Capabilities {
                    provider: "gs",
                    get_into: expect_gcs(),
                    head: expect_gcs()
                },
                "{uri}"
            );
        }
    }

    #[test]
    fn file_and_direct_never_support_it() {
        for (uri, provider) in [
            ("file:///tmp/x", "file"),
            ("file://relative/x", "file"),
            ("direct:///tmp/x", "direct"),
        ] {
            assert_eq!(
                caps(uri),
                Capabilities {
                    provider,
                    get_into: false,
                    head: false
                },
                "{uri}"
            );
        }
    }

    #[test]
    fn unknown_scheme_is_reported_not_rejected() {
        for uri in [
            "ftp://host/x",
            "/plain/path",
            "",
            "s4://bucket/key",
            "az:/x",
        ] {
            assert_eq!(
                caps(uri),
                Capabilities {
                    provider: "unknown",
                    get_into: false,
                    head: false
                },
                "{uri:?}"
            );
        }
    }

    #[test]
    fn malformed_uris_are_errors() {
        for uri in [
            // #224: an empty container.
            "az://account//key",
            "az://account/",
            "az://account",
            "az://account//",
            "az:///container/key",
            "az://acc%ount/container/key",
            "az://acc@ount/container/key",
            "az://../container/key",
            "https://account.blob.core.windows.net",
            "https://account.blob.core.windows.net/",
            "https://account.blob.core.windows.net//k",
            "https://.blob.core.windows.net/container/k",
            // S3.
            "s3://",
            "s3:///key",
            "s3://bucket",
            "s3://10.0.0.5/bucket/key",
            "s3://host:9000",
            "s3://host:9000/",
            "s3://host:0/bucket/key",
            "s3://user@host:9000/bucket/key",
            // GCS.
            "gs://",
            "gs:///object",
            "gcs://",
        ] {
            assert!(
                object_capabilities(uri).is_err(),
                "{uri:?} is malformed and must be an error"
            );
        }
    }

    #[test]
    fn keys_are_literal_and_not_decoded() {
        let c = parse_s3_uri_pure("s3://bucket/a%2Fb/c%20d").expect("s3");
        assert_eq!(
            (c.bucket.as_str(), c.key.as_str()),
            ("bucket", "a%2Fb/c%20d")
        );
        assert_eq!(c.key.len(), "a%2Fb/c%20d".len());

        let (_, container, key) = parse_azure_uri_pure("az://acct/cont/a%2Fb").expect("az");
        assert_eq!((container.as_str(), key.as_str()), ("cont", "a%2Fb"));
        assert_eq!(key.chars().count(), 5, "%2F stays three characters");

        let (_, container, key) =
            parse_azure_uri_pure("https://acct.blob.core.windows.net/cont/a%2Fb").expect("https");
        assert_eq!((container.as_str(), key.as_str()), ("cont", "a%2Fb"));

        let (bucket, object) = parse_gs_uri_pure("gs://bkt/a%2Fb").expect("gs");
        assert_eq!((bucket.as_str(), object.as_str()), ("bkt", "a%2Fb"));
    }

    // ---- Azure structure (#224) ---------------------------------------------------------

    #[test]
    fn azure_empty_container_is_refused_and_valid_roots_parse() {
        for uri in [
            "az://account//key",
            "az://account//",
            "az://account/",
            "az://account///key",
            "https://account.blob.core.windows.net//key",
        ] {
            let err = parse_azure_uri_pure(uri).expect_err(uri);
            assert!(format!("{err:#}").contains("container"), "{uri}: {err:#}");
        }
        for (uri, want) in [
            ("az://account/container", ("account", "container", "")),
            ("az://account/container/", ("account", "container", "")),
            (
                "az://account/container/key",
                ("account", "container", "key"),
            ),
            ("az://account/container//k", ("account", "container", "/k")),
            (
                "az://account/container/100%25/x%",
                ("account", "container", "100%25/x%"),
            ),
            (
                "https://acct.blob.core.windows.net/cont/k/n",
                ("acct", "cont", "k/n"),
            ),
        ] {
            let (a, c, k) = parse_azure_uri_pure(uri).expect(uri);
            assert_eq!((a.as_str(), c.as_str(), k.as_str()), want, "{uri}");
        }
    }

    // A `/` cannot appear in an `az://` account: it ends the account segment.
    #[test]
    fn azure_unsafe_accounts_are_refused() {
        for account in [
            "", "a?b", "a#b", "a@b", "a\\b", "a%b", "a b", "a\tb", ".", "..",
        ] {
            assert!(
                parse_azure_uri_pure(&format!("az://{account}/container/key")).is_err(),
                "account {account:?}"
            );
        }
    }

    #[test]
    fn azure_key_mentioning_the_host_does_not_make_an_s3_uri_azure() {
        let c = caps("s3://backups/mirror/acct.blob.core.windows.net/file.dat");
        assert_eq!(c.provider, "s3");
    }

    // ---- parity with the existing parsers ----------------------------------------------

    const S3_CORPUS: &[&str] = &[
        "s3://bucket/key",
        "s3://bucket/",
        "s3://bucket",
        "s3://bucket//k",
        "s3://bucket/a%2Fb/c",
        "s3://data.example.com/dir/file.bin",
        "s3://localhost/key",
        "s3://host:9000/bucket/key",
        "s3://host:9000/bucket",
        "s3://host:9000/bucket/",
        "s3://host:9000/",
        "s3://host:9000",
        "s3://host:9000//key",
        "s3://minio_1:9000/b/k/l",
        "s3://[::1]:9000/bucket/key",
        "s3://[::1]/bucket/key",
        "s3://[::1]:9000",
        "s3://[::1]x/bucket/key",
        "s3://[1.2.3.4]:80/bucket/key",
        "s3://[nope]:80/b/k",
        "s3://[/b/k",
        "s3://]/b/k",
        "s3://::1/bucket/key",
        "s3://a:b:c/bucket/key",
        "s3://10.0.0.5/bucket/key",
        "s3://10.0.0.5:9000/bucket/key",
        "s3://host:/bucket/key",
        "s3://host:09000/bucket/key",
        "s3://host:+80/bucket/key",
        "s3://host:65536/bucket/key",
        "s3://host:65535/bucket/key",
        "s3://:9000/bucket/key",
        "s3://ho/st:9000/bucket/key",
        "s3://ho st:9000/bucket/key",
        "s3://.host:9000/bucket/key",
        "s3://host..x:9000/bucket/key",
        "s3://user@host:9000/bucket/key",
        "s3://arn:aws:s3:::bucket/key",
        "s3://http://host:9000/b/k",
        "s3:///key",
        "s3://",
        "gs://bucket/key",
        "",
    ];

    #[test]
    fn s3_pure_parser_matches_the_existing_one() {
        for uri in S3_CORPUS {
            let pure = parse_s3_uri_pure(uri);
            let existing = crate::s3_utils::parse_s3_uri_full(uri);
            match (pure, existing) {
                (Ok(p), Ok(e)) => assert_eq!(
                    (p.endpoint, p.bucket, p.key),
                    (e.endpoint, e.bucket, e.key),
                    "{uri:?}"
                ),
                (Err(p), Err(e)) => assert_eq!(format!("{p:#}"), format!("{e:#}"), "{uri:?}"),
                (p, e) => panic!("{uri:?}: pure={p:?} existing={e:?}"),
            }
        }
    }

    #[test]
    fn s3_first_segment_classifier_matches_the_existing_one() {
        for uri in S3_CORPUS {
            let segment = uri.strip_prefix("s3://").unwrap_or(uri);
            let segment = segment.split('/').next().unwrap_or("");
            let pure = classify_s3_first_segment_pure(segment);
            let existing = crate::s3_utils::classify_s3_first_segment(segment);
            match (pure, existing) {
                (Ok(p), Ok(e)) => assert_eq!(p, e, "{segment:?}"),
                (Err(p), Err(e)) => assert_eq!(format!("{p:#}"), format!("{e:#}"), "{segment:?}"),
                (p, e) => panic!("{segment:?}: pure={p:?} existing={e:?}"),
            }
        }
    }

    #[cfg(all(feature = "backend-gcs", not(feature = "gcs-community")))]
    #[test]
    fn gs_pure_parser_matches_the_official_gated_one() {
        for uri in [
            "gs://bucket/path/to/object",
            "gcs://bucket/o",
            "gs://bucket/",
            "gs://bucket",
            "gs://bucket//o",
            "gs://",
            "gs:///o",
            "gcs://",
            "s3://bucket/o",
            "bucket/o",
            "",
        ] {
            let pure = parse_gs_uri_pure(uri);
            let existing = crate::google_gcs_client::parse_gcs_uri(uri);
            match (pure, existing) {
                (Ok(p), Ok(e)) => assert_eq!(p, e, "{uri:?}"),
                (Err(p), Err(e)) => assert_eq!(format!("{p:#}"), format!("{e:#}"), "{uri:?}"),
                (p, e) => panic!("{uri:?}: pure={p:?} existing={e:?}"),
            }
        }
    }

    #[cfg(feature = "gcs-community")]
    #[test]
    fn gs_pure_parser_matches_the_community_gated_one() {
        for uri in [
            "gs://bucket/path/to/object",
            "gcs://bucket/o",
            "gs://bucket/",
            "gs://bucket",
            "gs://",
            "gs:///o",
            "s3://bucket/o",
            "",
        ] {
            let pure = parse_gs_uri_pure(uri);
            let existing = crate::gcs_client::parse_gcs_uri(uri);
            match (pure, existing) {
                (Ok(p), Ok(e)) => assert_eq!(p, e, "{uri:?}"),
                (Err(p), Err(e)) => assert_eq!(format!("{p:#}"), format!("{e:#}"), "{uri:?}"),
                (p, e) => panic!("{uri:?}: pure={p:?} existing={e:?}"),
            }
        }
    }

    // ---- purity ------------------------------------------------------------------------

    /// `object_capabilities` answers with every credential and endpoint variable absent and
    /// outside any async runtime. It is a plain synchronous function whose only callees are the
    /// pure parsers above and `infer_scheme` (string matching): none of them can construct a
    /// store, spawn a runtime, or read the environment, so there is nothing for an absent
    /// credential to fail.
    #[test]
    fn answers_with_no_credentials_and_no_runtime() {
        let _guard = lock();
        let _env = EnvRestore::capture(CREDENTIAL_AND_ENDPOINT_VARS);
        for var in CREDENTIAL_AND_ENDPOINT_VARS {
            std::env::remove_var(var);
        }
        assert!(
            tokio::runtime::Handle::try_current().is_err(),
            "this test must run outside any runtime"
        );
        for uri in [
            "s3://bucket/key",
            "az://account/container/key",
            "gs://bucket/key",
            "file:///tmp/x",
            "direct:///tmp/x",
            "ftp://x/y",
        ] {
            object_capabilities(uri).unwrap_or_else(|e| panic!("{uri}: {e:#}"));
        }
        assert!(
            tokio::runtime::Handle::try_current().is_err(),
            "object_capabilities must not leave a runtime entered"
        );
    }

    /// The answer does not depend on the environment: setting every variable (to values that
    /// would make a client build fail or route elsewhere) changes nothing, and the pure Azure
    /// parser neither lower-cases the account nor applies the real-account length rule that the
    /// gated canonicalisation applies for a default endpoint.
    #[test]
    fn answer_is_independent_of_the_environment() {
        let _guard = lock();
        let _env = EnvRestore::capture(CREDENTIAL_AND_ENDPOINT_VARS);
        let uris = [
            "s3://bucket/key",
            "s3://host:9000/bucket/key",
            "az://MyAcct/container/key",
            "az://ab/container/key",
            "gs://bucket/key",
            "file:///tmp/x",
        ];
        for var in CREDENTIAL_AND_ENDPOINT_VARS {
            std::env::remove_var(var);
        }
        let absent: Vec<_> = uris.iter().map(|u| caps(u)).collect();
        let account_absent = parse_azure_uri_pure("az://MyAcct/c/k").expect("az").0;

        for var in CREDENTIAL_AND_ENDPOINT_VARS {
            std::env::set_var(var, "http://127.0.0.1:1");
        }
        let present: Vec<_> = uris.iter().map(|u| caps(u)).collect();
        let account_present = parse_azure_uri_pure("az://MyAcct/c/k").expect("az").0;

        assert_eq!(absent, present);
        assert_eq!(account_absent, "MyAcct");
        assert_eq!(account_present, "MyAcct");
    }
}
