// tests/object_capabilities.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// s3dlio #264 / plan A3: `object_capabilities` through the public API. It is pure: no
// credentials, no client, no runtime, no network. Expectations follow the compiled features
// (Cargo's default build has no Azure; the wheel's default mode has it).

use s3dlio::capabilities::{object_capabilities, Capabilities};

const NATIVE: bool = cfg!(all(
    feature = "native-backends",
    not(feature = "arrow-backend")
));

fn caps(uri: &str) -> Capabilities {
    object_capabilities(uri).unwrap_or_else(|e| panic!("{uri}: {e:#}"))
}

#[test]
fn every_scheme_reports_what_this_build_compiled() {
    let azure = NATIVE && cfg!(feature = "backend-azure");
    let gcs = NATIVE && cfg!(feature = "backend-gcs");
    let table = [
        ("s3://bucket/key", "s3", NATIVE),
        ("az://account/container/key", "az", azure),
        (
            "https://account.blob.core.windows.net/container/key",
            "az",
            azure,
        ),
        ("gs://bucket/key", "gs", gcs),
        ("gcs://bucket/key", "gs", gcs),
        ("file:///tmp/x", "file", false),
        ("direct:///tmp/x", "direct", false),
        ("ftp://host/x", "unknown", false),
    ];
    for (uri, provider, supported) in table {
        let c = caps(uri);
        assert_eq!(c.provider, provider, "{uri}");
        assert_eq!(c.get_into, supported, "{uri}");
        assert_eq!(c.head, supported, "{uri}");
    }
}

#[test]
fn malformed_uris_are_errors_in_every_build() {
    // #224: an empty Azure container is refused even where Azure is compiled out.
    for uri in [
        "az://account//key",
        "az://account/",
        "s3://bucket",
        "s3://10.0.0.5/bucket/key",
        "gs://",
    ] {
        assert!(object_capabilities(uri).is_err(), "{uri}");
    }
}

#[test]
fn percent_containing_keys_are_accepted_and_literal() {
    for uri in [
        "s3://bucket/a%2Fb",
        "az://account/container/a%2Fb",
        "gs://bucket/a%2Fb",
    ] {
        assert!(object_capabilities(uri).is_ok(), "{uri}");
    }
}

#[test]
fn azure_authority_with_userinfo_or_backslash_is_malformed() {
    // The account must come from the host alone; these name one host and read as another.
    for uri in [
        "https://abc.def@real.blob.core.windows.net/c/k",
        "https://evil.example\\@real.blob.core.windows.net/c/k",
    ] {
        assert!(object_capabilities(uri).is_err(), "{uri}");
    }
}

// Literal expectations for Cargo's default build (no Azure, no GCS, no Arrow backend), written
// without the `cfg!` constants above.
#[cfg(all(
    feature = "native-backends",
    not(feature = "backend-azure"),
    not(feature = "backend-gcs"),
    not(feature = "arrow-backend")
))]
#[test]
fn default_build_literal_values() {
    for (uri, provider, supported) in [
        ("s3://bucket/key", "s3", true),
        ("az://account/container/key", "az", false),
        ("gs://bucket/key", "gs", false),
        ("file:///tmp/x", "file", false),
        ("ftp://host/x", "unknown", false),
    ] {
        let c = caps(uri);
        assert_eq!(c.provider, provider, "{uri}");
        assert_eq!((c.get_into, c.head), (supported, supported), "{uri}");
    }
}
