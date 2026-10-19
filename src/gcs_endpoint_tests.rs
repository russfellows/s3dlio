// src/gcs_endpoint_tests.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Tests of the custom-endpoint / emulator resolution for the official GCS client:
// `GCS_ENDPOINT_URL` and `STORAGE_EMULATOR_HOST`.  The resolver takes a lookup closure, so none of
// these tests touches the process environment.  The on-the-wire proof (a real `GcsClient` built
// from the environment sends its requests to the named endpoint, anonymously) lives in
// `tests/gcs_emulator_endpoint.rs`.

use super::*;
use std::collections::HashMap;

/// A lookup closure over a fixed set of variables.
fn lookup_of(vars: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
    let map: HashMap<String, String> = vars
        .iter()
        .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
        .collect();
    move |name| map.get(name).cloned()
}

fn resolve(vars: &[(&str, &str)]) -> Option<GcsEndpoint> {
    resolve_gcs_endpoint_with(lookup_of(vars))
}

#[test]
fn neither_variable_set_means_no_endpoint() {
    assert_eq!(resolve(&[]), None);
}

#[test]
fn endpoint_url_is_used_as_given() {
    let got = resolve(&[("GCS_ENDPOINT_URL", "http://localhost:4443")]).expect("endpoint");
    assert_eq!(got.url, "http://localhost:4443");
    assert_eq!(got.source, GcsEndpointSource::EndpointUrl);
}

#[test]
fn emulator_host_without_a_scheme_gets_http() {
    let got = resolve(&[("STORAGE_EMULATOR_HOST", "localhost:4443")]).expect("endpoint");
    assert_eq!(got.url, "http://localhost:4443");
    assert_eq!(got.source, GcsEndpointSource::EmulatorHost);
}

#[test]
fn emulator_host_with_a_scheme_is_kept() {
    let http = resolve(&[("STORAGE_EMULATOR_HOST", "http://emu:9023")]).expect("endpoint");
    assert_eq!(http.url, "http://emu:9023");
    let https = resolve(&[("STORAGE_EMULATOR_HOST", "https://emu:9023")]).expect("endpoint");
    assert_eq!(https.url, "https://emu:9023");
    assert_eq!(https.source, GcsEndpointSource::EmulatorHost);
}

#[test]
fn endpoint_url_wins_over_emulator_host() {
    let got = resolve(&[
        ("GCS_ENDPOINT_URL", "http://primary:1"),
        ("STORAGE_EMULATOR_HOST", "secondary:2"),
    ])
    .expect("endpoint");
    assert_eq!(got.url, "http://primary:1");
    assert_eq!(got.source, GcsEndpointSource::EndpointUrl);
}

/// An empty (or blank) value is the same as unset: it must not win the precedence and must not
/// become the empty endpoint "http://".
#[test]
fn empty_and_blank_values_count_as_unset() {
    assert_eq!(resolve(&[("GCS_ENDPOINT_URL", "")]), None);
    assert_eq!(resolve(&[("STORAGE_EMULATOR_HOST", "")]), None);
    assert_eq!(resolve(&[("GCS_ENDPOINT_URL", "   ")]), None);
    assert_eq!(resolve(&[("STORAGE_EMULATOR_HOST", " \t")]), None);

    // An empty primary falls through to the emulator variable.
    let got = resolve(&[
        ("GCS_ENDPOINT_URL", ""),
        ("STORAGE_EMULATOR_HOST", "emu:9023"),
    ])
    .expect("endpoint");
    assert_eq!(got.url, "http://emu:9023");
    assert_eq!(got.source, GcsEndpointSource::EmulatorHost);
}

#[test]
fn surrounding_whitespace_is_trimmed() {
    let got = resolve(&[("GCS_ENDPOINT_URL", "  http://localhost:4443\n")]).expect("endpoint");
    assert_eq!(got.url, "http://localhost:4443");
    let got = resolve(&[("STORAGE_EMULATOR_HOST", " localhost:4443 ")]).expect("endpoint");
    assert_eq!(got.url, "http://localhost:4443");
}

#[test]
fn the_constants_name_the_documented_variables() {
    assert_eq!(crate::constants::ENV_GCS_ENDPOINT_URL, "GCS_ENDPOINT_URL");
    assert_eq!(
        crate::constants::ENV_STORAGE_EMULATOR_HOST,
        "STORAGE_EMULATOR_HOST"
    );
}

/// The logged form never carries a user name or password embedded in the URL.
#[test]
fn the_logged_endpoint_drops_embedded_credentials() {
    assert_eq!(
        redact_endpoint_url("http://user:secret@host:4443/path"),
        "http://***@host:4443/path"
    );
    assert_eq!(
        redact_endpoint_url("http://u:p@host/x"),
        "http://***@host/x"
    );
    assert_eq!(
        redact_endpoint_url("https://token@host"),
        "https://***@host"
    );
    // The authority ends at the first `/`, `?` or `#`; the userinfo is up to the LAST `@` in it.
    assert_eq!(
        redact_endpoint_url("http://u:p@host?x=1"),
        "http://***@host?x=1"
    );
    assert_eq!(
        redact_endpoint_url("http://u:p@host#f"),
        "http://***@host#f"
    );
    assert_eq!(
        redact_endpoint_url("http://a@b:c@host/p"),
        "http://***@host/p"
    );
    // No scheme: a userinfo is still dropped.
    assert_eq!(redact_endpoint_url("u:p@host:1"), "***@host:1");
}

#[test]
fn plain_endpoints_are_logged_unchanged() {
    for url in [
        "http://localhost:4443",
        "https://private.googleapis.com",
        "http://127.0.0.1:9/storage/v1",
        "http://[::1]:4443/x",
        "host:1",
        "localhost",
    ] {
        assert_eq!(redact_endpoint_url(url), url);
    }
}

/// An `@` after the authority, or credentials-like text that is not a host, is masked
/// conservatively instead of printed raw.
#[test]
fn malformed_looking_endpoints_are_masked_not_printed() {
    // A query `@` must not hide the host.
    assert_eq!(redact_endpoint_url("http://host?x=a@b"), "http://host/***");
    assert_eq!(
        redact_endpoint_url("http://host:4443/a@b"),
        "http://host:4443/***"
    );
    assert_eq!(redact_endpoint_url("http://host#a@b"), "http://host/***");
    // `u:pa` is not a host:port, so the `pa` of a password that holds a `/` must not appear.
    let shown = redact_endpoint_url("http://u:pa/ss@host");
    assert!(!shown.contains("pa") && !shown.contains("ss"), "{shown}");
    let shown = redact_endpoint_url("u:pa/ss@host");
    assert!(!shown.contains("pa") && !shown.contains("ss"), "{shown}");
}

/// `Debug` of an endpoint (it ends up in test failures and logs) shows the redacted URL.
#[test]
fn debug_of_an_endpoint_does_not_print_credentials() {
    let endpoint = GcsEndpoint {
        url: "http://user:secret@host:1/p".to_string(),
        source: GcsEndpointSource::EndpointUrl,
    };
    let shown = format!("{endpoint:?}");
    assert!(
        !shown.contains("secret") && !shown.contains("user"),
        "{shown}"
    );
    assert!(shown.contains("http://***@host:1/p"), "{shown}");
    assert!(shown.contains("EndpointUrl"), "{shown}");
    // The real URL is still what the SDK receives.
    assert_eq!(endpoint.url, "http://user:secret@host:1/p");
}
