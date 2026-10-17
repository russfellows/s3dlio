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
    let gcs = NATIVE && cfg!(all(feature = "backend-gcs", not(feature = "gcs-community")));
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
        assert_eq!(
            caps(uri),
            Capabilities {
                provider,
                get_into: supported,
                head: supported
            },
            "{uri}"
        );
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
