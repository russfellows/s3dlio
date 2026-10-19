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
//! The URI is checked with the crate-internal structural parsers in this module
//! (`parse_s3_uri_pure`, `parse_azure_uri_pure`, `parse_gs_uri_pure`), which read nothing but
//! their argument. They are the single structural implementation, and every gated parser
//! delegates to them: `s3_utils::parse_s3_uri_full` and `classify_s3_first_segment` (which add
//! only the one-shot warning about a retired environment variable), `parse_gcs_uri`
//! (`google_gcs_client`), and `parse_azure_uri` in
//! `object_store` (which adds the endpoint-dependent account canonicalisation). The Azure
//! account rules that every environment shares are used by `azure_client` as well. So the
//! parsers cannot drift from what a store accepts. A malformed URI is an error; an unknown
//! scheme is not (it reports `"unknown"` with no support).
//!
//! # Purity is structural
//!
//! Nothing here takes anything but `&str`. The call graph is the parsers above and
//! `object_store::infer_scheme` (string matching): there is no environment read, no `.env`
//! handling, no store or client construction and no async runtime anywhere in it. The unit tests
//! therefore need not (and do not) manipulate the process environment to show it.
//!
//! # Structural validation, not full parity
//!
//! These parsers validate what can be known from the URI alone. Anything that depends on the
//! environment stays with the gated parser that builds a store: for Azure, the account name is
//! canonicalised (lower-cased and length-checked, or kept verbatim for an emulator) according to
//! the configured endpoint. A URI accepted here can therefore still be refused when a store is
//! built for it; a URI refused here is malformed in every environment.

use anyhow::{anyhow, bail, Context, Result};

use crate::object_store::{infer_scheme, is_azure_blob_host, Scheme};
use crate::s3_utils::{S3FirstSegment, S3UriComponents};

/// What this build can do for an object URI. See [`object_capabilities`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
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

/// True when native GCS on the official pinned SDK is compiled in.
const GCS_OFFICIAL_COMPILED: bool = cfg!(feature = "backend-gcs");

/// Report what this build supports for `uri`, without credentials, a client or a runtime.
///
/// | scheme | `get_into` / `head` |
/// |---|---|
/// | `s3://` | true (native-backend builds) |
/// | `az://`, Azure blob host | true only when `backend-azure` is compiled |
/// | `gs://`, `gcs://` | true only with `backend-gcs` (the official SDK) |
/// | `file://`, `direct://` | false |
/// | unknown scheme | provider `"unknown"`, false |
/// | any `arrow-backend` build | false |
///
/// # Errors
///
/// `Err` means the URI is **malformed** for a known object scheme (for example `az://account//key`,
/// which has an empty container): a structural fault that no environment can repair. An unknown
/// scheme is not an error.
///
/// # What `Ok` does not promise
///
/// This is structural validation only. `Ok` with `true` says that this build has the operations
/// for the scheme and that the URI is well formed; it does **not** guarantee that a store will
/// accept the URI. Rules that depend on the environment are applied only when a store is built
/// (for example Azure account-name canonicalisation and length, which depend on the configured
/// endpoint), so a store can still refuse a URI this function accepts.
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

/// True if `s` is a syntactically valid dotted-quad IPv4 address (four `0..=255` octets
/// separated by dots). S3 bucket names are explicitly forbidden from being formatted as an IP
/// address, so this check is unambiguous: it can never misclassify a legal bucket name.
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

/// Validate the port part of `host:port` / `[ipv6]:port`.
///
/// Accepted: plain ASCII decimal digits, no leading zeros, value `1..=65535`.
/// `u16::from_str` alone is too lax (it accepts a leading `+` and `0`); leading zeros are
/// rejected because `host:09000` is ambiguous (octal habits) and would otherwise leave two
/// spellings of one server in configs, URLs and logs.
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

/// Validate the host part of an unbracketed `host:port` (a DNS name, IPv4 literal or
/// container-style name such as `minio_1`).
///
/// Grammar: 1..=253 characters, each an ASCII letter, digit, `-`, `.` or `_` (`_` is kept for
/// container-style names), with no empty label (no leading or trailing `.`, no `..`). Anything
/// else - `/`, `\`, whitespace, control characters, `%`, `?`, `#`, non-ASCII, ... - is an error
/// naming the character. Enforcing this before any endpoint URI is built means a host can never
/// change how `s3://host:port/bucket/key` is split (`node/prefix:9000` would otherwise read back
/// as bucket `node`, routed to the global client).
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

/// Decides whether the first path segment of an `s3://` URI is a server (`host:port`,
/// `[ipv6]:port`) or a bucket, by structure alone (issues #178, #193, #194).
///
/// This is THE single implementation of the rule: `s3_utils::classify_s3_first_segment` calls
/// it, adding only the one-shot warning about a retired environment variable. This function
/// never reads the environment.
///
/// - `host:port` or `[ipv6]:port` (explicit port, plain decimal `1..=65535`, no leading zeros)
///   is [`S3FirstSegment::Server`]. The bracketed content must be a valid IPv6 address; an
///   unbracketed host must follow the host grammar (1 to 253 ASCII letters, digits, `-`, `.`
///   and `_`, no empty label).
/// - A dotted-quad IPv4 literal with no port is an **error**: S3 forbids IP-shaped bucket names,
///   so it is not a bucket; the caller must add `:port`.
/// - An IPv6 literal without brackets or without a port is an **error**.
/// - Anything else, including dotted DNS-looking names such as `data.example.com` and
///   `localhost`, is [`S3FirstSegment::Bucket`]. Bucket names are not otherwise validated.
///
/// # Errors
///
/// An empty segment, a URL, userinfo, an ARN, a portless IP literal, or a malformed host, port
/// or bracket.
pub(crate) fn classify_s3_first_segment_pure(segment: &str) -> Result<S3FirstSegment<'_>> {
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

/// The error for `s3://host:9000`, `s3://host:9000/` and `s3://host:9000//key`: the server is
/// fine, the bucket after it is missing. Shared with `s3_utils::s3_bucket_target`.
pub(crate) fn missing_bucket_after_server(endpoint: &str, uri: &str) -> anyhow::Error {
    anyhow!(
        "missing bucket after server '{endpoint}' in S3 URI '{uri}': the bucket name cannot be \
         empty; write s3://{endpoint}/bucket/key"
    )
}

/// Split `s3://bucket/key` or `s3://host:port/bucket/key` into its parts. The key is exactly as
/// written (`%2F` stays three characters). Reads nothing but `uri`.
///
/// This is THE single implementation: `s3_utils::parse_s3_uri_full` calls it, adding only the
/// one-shot warning about a retired environment variable.
///
/// # Errors
///
/// A URI that does not start with `s3://`, has no `/` after the bucket, or whose first segment
/// is rejected by [`classify_s3_first_segment_pure`]; a server with no bucket after it.
pub(crate) fn parse_s3_uri_pure(uri: &str) -> Result<S3UriComponents> {
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

/// Why an Azure account name is empty or could redirect a request, whatever the configured
/// endpoint: the part of the account rules that is shared by every environment. The
/// environment-dependent rules (lower-casing, the 3 to 24 character length of a real account)
/// stay with the canonicalisation in `azure_client`, which starts from this function.
pub(crate) fn azure_account_redirect_problem(account: &str) -> Option<&'static str> {
    if account.is_empty() {
        return Some("it is empty");
    }
    account
        .chars()
        .any(|c| {
            matches!(c, '/' | '?' | '#' | '@' | '\\' | '%') || c.is_whitespace() || c.is_control()
        })
        .then_some(
            "it contains a character that could redirect the request (/ ? # @ \\ % or whitespace)",
        )
}

/// Why an Azure account name is not usable in any environment: the redirect problems of
/// [`azure_account_redirect_problem`], and a name that is a path segment (`.` or `..`).
pub(crate) fn azure_account_problem_pure(account: &str) -> Option<&'static str> {
    azure_account_redirect_problem(account)
        .or_else(|| (account == "." || account == "..").then_some("it is a path segment"))
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
pub(crate) fn parse_azure_uri_pure(uri: &str) -> Result<(String, String, String)> {
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

    if is_azure_blob_host(uri) {
        // The scheme (http or https) does not change which account, container and key the URL
        // names.
        let after_scheme = uri
            .split_once("://")
            .map(|(_, rest)| rest)
            .ok_or_else(|| anyhow!("expected a scheme (http:// or https://) for Azure URL"))?;
        let (authority, path) = after_scheme.split_once('/').unwrap_or((after_scheme, ""));
        // The account comes from the host and from nothing else, so the authority must be a
        // plain `host[:port]`. Userinfo (`@`) lets `abc.def@real.blob...` name the account `abc`
        // while the request goes to `real.blob...`; a backslash (read as `/` by URL parsers for
        // http and https), `?`, `#`, `%` and whitespace end or rewrite the authority so the host
        // we would read is not the host a request reaches.
        if authority.chars().any(|c| {
            matches!(c, '@' | '\\' | '?' | '#' | '%') || c.is_whitespace() || c.is_control()
        }) {
            bail!(
                "invalid Azure URL authority {authority:?}: it must be a plain host[:port] \
                 (no userinfo '@', backslash, '?', '#', '%' or whitespace)"
            );
        }
        let host = authority.split(':').next().unwrap_or(authority);
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
/// written. This is the environment-free form of the gated `parse_gcs_uri` in
/// `google_gcs_client`.
///
/// # Errors
///
/// A URI without a `gs://` or `gcs://` prefix, or with an empty bucket.
pub(crate) fn parse_gs_uri_pure(uri: &str) -> Result<(String, String)> {
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

    // ---- Azure authority smuggling (review finding) ----------------------------------------

    /// In the http(s) form the account must come from the host and nothing else. Userinfo,
    /// a backslash (which URL parsers treat as a `/` for http and https), and the `?` and `#`
    /// delimiters all end or rewrite the authority, so the "host" this parser sees would not be
    /// the host a request goes to. Every one of these is refused, never reduced to an account.
    #[test]
    fn azure_https_authority_cannot_smuggle_an_account() {
        for uri in [
            "https://abc.def@real.blob.core.windows.net/c/k",
            "https://abc@real.blob.core.windows.net/c/k",
            "http://user:pw@real.blob.core.windows.net/c/k",
            "https://evil.com\\@real.blob.core.windows.net/c/k",
            "https://evil.com\\.blob.core.windows.net/c/k",
            "https://evil.com?.blob.core.windows.net/c/k",
            "https://evil.com#.blob.core.windows.net/c/k",
            "https://evil.com%2f.blob.core.windows.net/c/k",
            "https://evil.com .blob.core.windows.net/c/k",
        ] {
            let err = parse_azure_uri_pure(uri)
                .expect_err(&format!("{uri}: must be refused, not parsed to an account"));
            assert!(format!("{err:#}").contains("authority"), "{uri}: {err:#}");
            // The same refusal reaches the capability answer: it is not "unknown".
            assert!(object_capabilities(uri).is_err(), "{uri}");
        }
    }

    #[test]
    fn azure_https_plain_hosts_still_parse() {
        for (uri, want) in [
            ("https://acct.blob.core.windows.net/c/k", ("acct", "c", "k")),
            (
                "http://acct.blob.core.windows.net:10000/c/k/n",
                ("acct", "c", "k/n"),
            ),
            (
                "https://MyAcct.Blob.Core.Windows.NET/c/k",
                ("MyAcct", "c", "k"),
            ),
        ] {
            let (a, c, k) = parse_azure_uri_pure(uri).expect(uri);
            assert_eq!((a.as_str(), c.as_str(), k.as_str()), want, "{uri}");
        }
        // A key may contain the characters the authority may not.
        let (_, _, key) =
            parse_azure_uri_pure("https://acct.blob.core.windows.net/c/a@b\\c?d#e").expect("key");
        assert_eq!(key, "a@b\\c?d#e");
    }

    #[test]
    fn azure_key_mentioning_the_host_does_not_make_an_s3_uri_azure() {
        let c = caps("s3://backups/mirror/acct.blob.core.windows.net/file.dat");
        assert_eq!(c.provider, "s3");
    }

    // ---- literal results of the single implementation ----------------------------------
    //
    // `s3_utils::parse_s3_uri_full` / `classify_s3_first_segment` and both `parse_gcs_uri`
    // functions delegate to the parsers here, so comparing them with the parsers would compare a
    // function with itself. These tables pin what the parsers return, written out literally.

    #[test]
    fn s3_uris_parse_to_these_literal_parts() {
        for (uri, endpoint, bucket, key) in [
            ("s3://bucket/key", None, "bucket", "key"),
            ("s3://bucket/", None, "bucket", ""),
            ("s3://bucket//k", None, "bucket", "/k"),
            ("s3://bucket/a%2Fb/c", None, "bucket", "a%2Fb/c"),
            // A ':' after the first '/' is key text, not a port: the bucket is just "ho".
            (
                "s3://ho/st:9000/bucket/key",
                None,
                "ho",
                "st:9000/bucket/key",
            ),
            (
                "s3://data.example.com/dir/file.bin",
                None,
                "data.example.com",
                "dir/file.bin",
            ),
            ("s3://localhost/key", None, "localhost", "key"),
            (
                "s3://host:9000/bucket/key",
                Some("host:9000"),
                "bucket",
                "key",
            ),
            ("s3://host:9000/bucket", Some("host:9000"), "bucket", ""),
            ("s3://host:9000/bucket/", Some("host:9000"), "bucket", ""),
            ("s3://minio_1:9000/b/k/l", Some("minio_1:9000"), "b", "k/l"),
            (
                "s3://[::1]:9000/bucket/key",
                Some("[::1]:9000"),
                "bucket",
                "key",
            ),
            (
                "s3://10.0.0.5:9000/bucket/key",
                Some("10.0.0.5:9000"),
                "bucket",
                "key",
            ),
            (
                "s3://host:65535/bucket/key",
                Some("host:65535"),
                "bucket",
                "key",
            ),
        ] {
            let parts = parse_s3_uri_pure(uri).unwrap_or_else(|e| panic!("{uri:?}: {e:#}"));
            assert_eq!(
                (
                    parts.endpoint.as_deref(),
                    parts.bucket.as_str(),
                    parts.key.as_str()
                ),
                (endpoint, bucket, key),
                "{uri:?}"
            );
        }
    }

    /// Every rejected shape fails, with a message that names what was wrong.
    #[test]
    fn malformed_s3_uris_fail_with_these_messages() {
        for (uri, fragment) in [
            ("", "URI must start with s3://"),
            ("gs://bucket/key", "URI must start with s3://"),
            ("s3://bucket", "URI must contain '/' after bucket"),
            ("s3://", "URI must contain '/' after bucket"),
            ("s3:///key", "empty first segment"),
            ("s3://host:9000", "missing bucket after server 'host:9000'"),
            // An IPv6 server with no slash after it is a server with no bucket, like
            // `s3://host:9000` (not the single-segment `s3://bucket` refusal).
            (
                "s3://[::1]:9000",
                "missing bucket after server '[::1]:9000'",
            ),
            ("s3://host:9000/", "missing bucket after server 'host:9000'"),
            (
                "s3://host:9000//key",
                "missing bucket after server 'host:9000'",
            ),
            ("s3://10.0.0.5/bucket/key", "IP address with no port"),
            ("s3://[::1]/bucket/key", "IPv6 address without a port"),
            ("s3://[::1]x/bucket/key", "unexpected text 'x' after ']'"),
            (
                "s3://[1.2.3.4]:80/bucket/key",
                "IPv4 address inside brackets",
            ),
            ("s3://[nope]:80/b/k", "invalid IPv6 address 'nope'"),
            ("s3://[/b/k", "unterminated '['"),
            ("s3://]/b/k", "unexpected bracket"),
            ("s3://::1/bucket/key", "unbracketed IPv6 address"),
            ("s3://a:b:c/bucket/key", "more than one ':'"),
            ("s3://host:/bucket/key", "empty port"),
            ("s3://host:09000/bucket/key", "leading zeros"),
            ("s3://host:+80/bucket/key", "plain decimal digits"),
            ("s3://host:65536/bucket/key", "from 1 to 65535"),
            ("s3://:9000/bucket/key", "missing host before ':'"),
            (
                "s3://ho st:9000/bucket/key",
                "invalid character ' ' in host",
            ),
            ("s3://.host:9000/bucket/key", "empty label"),
            ("s3://host..x:9000/bucket/key", "empty label"),
            ("s3://user@host:9000/bucket/key", "userinfo"),
            ("s3://arn:aws:s3:::bucket/key", "looks like an ARN"),
            ("s3://http://host:9000/b/k", "empty port in 'http:'"),
        ] {
            let err = match parse_s3_uri_pure(uri) {
                Ok(parts) => panic!("{uri:?} must be rejected, got {parts:?}"),
                Err(e) => format!("{e:#}"),
            };
            assert!(
                err.contains(fragment),
                "{uri:?}: {fragment:?} not in {err:?}"
            );
        }
        // The whole message of the IPv6-server-without-slash shape, not just a fragment.
        assert_eq!(
            format!("{:#}", parse_s3_uri_pure("s3://[::1]:9000").unwrap_err()),
            "missing bucket after server '[::1]:9000' in S3 URI 's3://[::1]:9000': the bucket name \
             cannot be empty; write s3://[::1]:9000/bucket/key"
        );
    }

    #[test]
    fn s3_first_segments_classify_to_these_literal_results() {
        for (segment, server) in [
            ("bucket", false),
            ("data.example.com", false),
            ("localhost", false),
            ("host:9000", true),
            ("[::1]:9000", true),
            ("10.0.0.5:9000", true),
        ] {
            let want = if server {
                S3FirstSegment::Server(segment)
            } else {
                S3FirstSegment::Bucket(segment)
            };
            assert_eq!(
                classify_s3_first_segment_pure(segment).unwrap(),
                want,
                "{segment:?}"
            );
        }
        for bad in [
            "",
            "10.0.0.5",
            "::1",
            "user@host:9000",
            "http://h:1",
            "host:0",
        ] {
            assert!(classify_s3_first_segment_pure(bad).is_err(), "{bad:?}");
        }
        let url = classify_s3_first_segment_pure("http://h:1").unwrap_err();
        assert!(format!("{url:#}").contains("looks like a URL"), "{url:#}");
    }

    /// The literal results of the `gs://` parser and of the gated `parse_gcs_uri` that
    /// delegates to it (the official client).
    #[test]
    fn gs_uris_parse_to_these_literal_parts() {
        for (uri, want) in [
            (
                "gs://bucket/path/to/object",
                Some(("bucket", "path/to/object")),
            ),
            ("gcs://bucket/o", Some(("bucket", "o"))),
            ("gs://bucket/", Some(("bucket", ""))),
            ("gs://bucket", Some(("bucket", ""))),
            ("gs://bucket//o", Some(("bucket", "/o"))),
            ("gs://", None),
            ("gs:///o", None),
            ("gcs://", None),
            ("s3://bucket/o", None),
            ("bucket/o", None),
            ("", None),
        ] {
            let pure = parse_gs_uri_pure(uri);
            match (&pure, want) {
                (Ok((b, o)), Some((wb, wo))) => {
                    assert_eq!((b.as_str(), o.as_str()), (wb, wo), "{uri:?}")
                }
                (Err(_), None) => {}
                _ => panic!("{uri:?}: got {pure:?}, want {want:?}"),
            }
            // The gated parser returns exactly what the pure one does, error text included.
            #[cfg(feature = "backend-gcs")]
            assert_eq!(
                format!("{:?}", crate::google_gcs_client::parse_gcs_uri(uri)),
                format!("{pure:?}"),
                "{uri:?}"
            );
        }
        let err = format!("{:#}", parse_gs_uri_pure("s3://bucket/o").unwrap_err());
        assert!(err.contains("expected gs:// or gcs:// prefix"), "{err}");
        let err = format!("{:#}", parse_gs_uri_pure("gs:///o").unwrap_err());
        assert!(err.contains("empty bucket name"), "{err}");
    }

    /// The account rules shared with `azure_client`: what every environment refuses.
    #[test]
    fn azure_account_rules_have_these_literal_results() {
        assert_eq!(azure_account_redirect_problem("myacct"), None);
        assert_eq!(azure_account_redirect_problem("My_Acct-1"), None);
        assert_eq!(azure_account_redirect_problem(""), Some("it is empty"));
        const REDIRECT: &str =
            "it contains a character that could redirect the request (/ ? # @ \\ % or whitespace)";
        // U+00A0 is Unicode whitespace and U+0007 (BEL) is a control character: both refused.
        for bad in [
            "a/b", "a?b", "a#b", "a@b", "a\\b", "a%b", "a b", "a\tb", "a\nb", "a\u{a0}b", "a\u{7}b",
        ] {
            assert_eq!(
                azure_account_redirect_problem(bad),
                Some(REDIRECT),
                "{bad:?}"
            );
            assert_eq!(azure_account_problem_pure(bad), Some(REDIRECT), "{bad:?}");
        }
        // A path segment is not a redirect character, but is refused in every environment.
        assert_eq!(azure_account_redirect_problem(".."), None);
        assert_eq!(
            azure_account_problem_pure(".."),
            Some("it is a path segment")
        );
        assert_eq!(
            azure_account_problem_pure("."),
            Some("it is a path segment")
        );
        assert_eq!(azure_account_problem_pure("a.b"), None);
        assert_eq!(
            azure_account_problem_pure("a/b"),
            azure_account_redirect_problem("a/b")
        );
    }

    // ---- purity ------------------------------------------------------------------------
    //
    // Purity is structural, not demonstrated by manipulating the environment: the functions
    // take only `&str`, and their call graph (the parsers in this module and
    // `object_store::infer_scheme`) has no environment read, `.env` handling, store, client or
    // runtime in it. These tests therefore never set or remove a variable (which would also race
    // other tests in the process).

    /// `object_capabilities` is a plain synchronous function: it answers outside any async
    /// runtime and leaves none entered.
    #[test]
    fn answers_with_no_runtime() {
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

    /// The pure Azure parser returns the account as written: it neither lower-cases it nor
    /// applies the real-account length rule that the gated canonicalisation applies for a
    /// default endpoint (both depend on the configured endpoint).
    #[test]
    fn pure_azure_parser_keeps_the_account_as_written() {
        assert_eq!(
            parse_azure_uri_pure("az://MyAcct/c/k").expect("az").0,
            "MyAcct"
        );
        assert_eq!(parse_azure_uri_pure("az://ab/c/k").expect("az").0, "ab");
        assert_eq!(caps("az://MyAcct/container/key").provider, "az");
    }

    // ---- literal values in the default build -------------------------------------------

    /// The literal table for the build with no Azure, no GCS and no Arrow backend (Cargo's
    /// default features), written without reference to the `cfg!` constants the
    /// implementation uses, so a wrong constant cannot hide behind a mirrored expectation.
    #[cfg(all(
        feature = "native-backends",
        not(feature = "backend-azure"),
        not(feature = "backend-gcs"),
        not(feature = "arrow-backend")
    ))]
    #[test]
    fn default_build_reports_these_literal_values() {
        for (uri, provider, supported) in [
            ("s3://bucket/key", "s3", true),
            ("az://account/container/key", "az", false),
            ("gs://bucket/key", "gs", false),
            ("file:///tmp/x", "file", false),
            ("ftp://host/x", "unknown", false),
        ] {
            let c = caps(uri);
            assert_eq!(c.provider, provider, "{uri}");
            assert_eq!(c.get_into, supported, "{uri}");
            assert_eq!(c.head, supported, "{uri}");
        }
    }
}
