// tests/s3_endpoint_precedence.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Regression tests for s3dlio issue #183: `AWS_ENDPOINT_URL_S3` (the service-specific
// variable) takes precedence over `AWS_ENDPOINT_URL`, as in the AWS SDK. Before the
// fix s3dlio applied `AWS_ENDPOINT_URL` as an explicit endpoint, which silently
// overrode the service-specific one. An explicit `--endpoint-url` still wins over
// both, and a server named in the URI (`s3://host:port/bkt`) wins over everything.
//
// Each test runs the real `s3-cli` against recording in-process mock S3 servers.

#[path = "common/recording_s3.rs"]
mod recording_s3;
use recording_s3::{run_s3_cli, text, MockS3};

#[test]
fn cli_service_specific_endpoint_wins_over_global_endpoint() {
    let a = MockS3::start(&[]); // AWS_ENDPOINT_URL
    let b = MockS3::start(&[]); // AWS_ENDPOINT_URL_S3

    let out = run_s3_cli(
        &["create-bucket", "s3://bkt183both"],
        &[
            ("AWS_ENDPOINT_URL", &a.url()),
            ("AWS_ENDPOINT_URL_S3", &b.url()),
        ],
    );

    assert_eq!(
        a.requests(),
        Vec::<String>::new(),
        "AWS_ENDPOINT_URL must lose to AWS_ENDPOINT_URL_S3 ({})",
        text(&out)
    );
    assert_eq!(
        b.requests(),
        vec!["PUT /bkt183both".to_string()],
        "{}",
        text(&out)
    );
    assert!(out.status.success(), "{}", text(&out));
}

#[test]
fn cli_service_specific_endpoint_alone_is_used() {
    let b = MockS3::start(&[]);

    let out = run_s3_cli(
        &["create-bucket", "s3://bkt183only"],
        &[("AWS_ENDPOINT_URL_S3", &b.url())],
    );

    assert_eq!(
        b.requests(),
        vec!["PUT /bkt183only".to_string()],
        "{}",
        text(&out)
    );
    assert!(out.status.success(), "{}", text(&out));
}

#[test]
fn cli_empty_service_specific_endpoint_is_unset() {
    let a = MockS3::start(&[]);

    let out = run_s3_cli(
        &["create-bucket", "s3://bkt183empty"],
        &[("AWS_ENDPOINT_URL", &a.url()), ("AWS_ENDPOINT_URL_S3", "")],
    );

    assert_eq!(
        a.requests(),
        vec!["PUT /bkt183empty".to_string()],
        "{}",
        text(&out)
    );
    assert!(out.status.success(), "{}", text(&out));
}

#[test]
fn cli_endpoint_url_flag_wins_over_service_specific_endpoint() {
    let b = MockS3::start(&[]); // AWS_ENDPOINT_URL_S3
    let c = MockS3::start(&[]); // --endpoint-url

    let out = run_s3_cli(
        &[
            "--endpoint-url",
            &c.url(),
            "create-bucket",
            "s3://bkt183flag",
        ],
        &[("AWS_ENDPOINT_URL_S3", &b.url())],
    );

    assert_eq!(
        b.requests(),
        Vec::<String>::new(),
        "--endpoint-url must beat AWS_ENDPOINT_URL_S3 ({})",
        text(&out)
    );
    assert_eq!(
        c.requests(),
        vec!["PUT /bkt183flag".to_string()],
        "{}",
        text(&out)
    );
}

#[test]
fn cli_host_form_uri_beats_service_specific_endpoint() {
    let b = MockS3::start(&[]); // AWS_ENDPOINT_URL_S3
    let c = MockS3::start(&[]); // named in the URI
    let uri = format!("s3://127.0.0.1:{}/bkt183uri", c.port);

    let out = run_s3_cli(
        &["create-bucket", &uri],
        &[("AWS_ENDPOINT_URL_S3", &b.url())],
    );

    assert_eq!(b.requests(), Vec::<String>::new(), "{}", text(&out));
    assert_eq!(
        c.requests(),
        vec!["PUT /bkt183uri".to_string()],
        "{}",
        text(&out)
    );
}
