// tests/s3_list_buckets_routing.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Regression tests for s3dlio issue #200: `list_containers` and `s3-cli list-buckets`
// for `s3://host:port/` must list THAT server's buckets (per-endpoint client), not
// those of the `AWS_ENDPOINT_URL` server; `s3://10.0.0.5/` (a portless IP) is the #178
// error; every other form (`s3://`, `s3://bucket/`, a dotted name without a port)
// lists the global server.
//
// CLI tests run the real `s3-cli`; library tests call `s3dlio::list_containers`
// (what the Python `list_containers` calls) against a single global mock for the
// whole binary, because the library's global client is process-wide.

#[path = "common/recording_s3.rs"]
mod recording_s3;
use recording_s3::{run_s3_cli, text, MockS3};
use std::sync::{Mutex, OnceLock};

#[test]
fn cli_list_buckets_host_form_lists_that_servers_buckets() {
    let a = MockS3::start(&["bucket-of-a"]); // AWS_ENDPOINT_URL: the wrong server
    let b = MockS3::start(&["bucket-of-b"]); // the server named in the URI
    let uri = format!("s3://127.0.0.1:{}/", b.port);

    let out = run_s3_cli(&["list-buckets", &uri], &[("AWS_ENDPOINT_URL", &a.url())]);
    let stdout = String::from_utf8_lossy(&out.stdout);

    assert_eq!(
        a.requests(),
        Vec::<String>::new(),
        "list-buckets {uri}: the AWS_ENDPOINT_URL server must receive no request ({})",
        text(&out)
    );
    assert_eq!(b.requests(), vec!["GET /".to_string()], "{}", text(&out));
    assert!(out.status.success(), "{}", text(&out));
    assert!(stdout.contains("bucket-of-b"), "{}", text(&out));
    assert!(!stdout.contains("bucket-of-a"), "{}", text(&out));
    // The listed URI must be usable as an argument to other s3dlio commands:
    // it names the server it was listed from.
    assert!(
        stdout.contains(&format!("s3://127.0.0.1:{}/bucket-of-b", b.port)),
        "{}",
        text(&out)
    );
}

#[test]
fn cli_list_buckets_portless_ip_is_an_error_and_sends_nothing() {
    let a = MockS3::start(&["bucket-of-a"]);

    let out = run_s3_cli(
        &["list-buckets", "s3://10.0.0.5/"],
        &[("AWS_ENDPOINT_URL", &a.url())],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);

    assert!(!out.status.success(), "{}", text(&out));
    assert!(
        stderr.contains("IP address with no port"),
        "expected the #178 error ({})",
        text(&out)
    );
    assert_eq!(a.requests(), Vec::<String>::new(), "{}", text(&out));
}

#[test]
fn cli_list_buckets_other_forms_list_the_global_server() {
    // `s3://`, `s3://some-bucket/` and a portless dotted name are all "global".
    for uri in ["s3://", "s3://some-bucket/", "s3://data.example.com/"] {
        let a = MockS3::start(&["bucket-of-a"]);
        let out = run_s3_cli(&["list-buckets", uri], &[("AWS_ENDPOINT_URL", &a.url())]);
        assert!(out.status.success(), "{uri}: {}", text(&out));
        assert!(
            String::from_utf8_lossy(&out.stdout).contains("bucket-of-a"),
            "{uri}: {}",
            text(&out)
        );
        assert_eq!(a.requests(), vec!["GET /".to_string()], "{uri}");
    }
}

// ----------------------------------------------------------------------------
// Library entry point (what the Python bindings call)
// ----------------------------------------------------------------------------

/// The single global-endpoint mock (A) for this test binary. Installed as
/// `AWS_ENDPOINT_URL` before the library's global client is first built; every
/// library test calls this first, before doing anything else.
fn global_a() -> &'static MockS3 {
    static A: OnceLock<MockS3> = OnceLock::new();
    A.get_or_init(|| {
        let a = MockS3::start(&["bucket-of-a"]);
        // SAFETY: runs once, at the start of the first library test, before any
        // s3dlio client exists; the other library tests block on this OnceLock.
        unsafe {
            std::env::set_var("AWS_ENDPOINT_URL", a.url());
            std::env::remove_var("AWS_ENDPOINT_URL_S3");
            std::env::set_var("AWS_ACCESS_KEY_ID", "test-access-key");
            std::env::set_var("AWS_SECRET_ACCESS_KEY", "test-secret-key");
            std::env::set_var("AWS_REGION", "us-east-1");
        }
        a
    })
}

/// The two list tests count `GET /` on the shared global mock, so they must not overlap.
static LIST_LOCK: Mutex<()> = Mutex::new(());

/// Number of `GET /` (ListBuckets) requests the global mock has received.
fn root_gets(a: &MockS3) -> usize {
    a.requests()
        .iter()
        .filter(|r| r.as_str() == "GET /")
        .count()
}

#[test]
fn library_list_containers_host_form_lists_that_servers_buckets() {
    let a = global_a();
    let _serial = LIST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let b = MockS3::start(&["bucket-of-b1", "bucket-of-b2"]);

    let before = root_gets(a);
    let listed = s3dlio::list_containers(&format!("s3://127.0.0.1:{}/", b.port))
        .expect("list the named server");

    let names: Vec<&str> = listed.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(names, vec!["bucket-of-b1", "bucket-of-b2"]);
    assert_eq!(
        listed[0].uri,
        format!("s3://127.0.0.1:{}/bucket-of-b1", b.port)
    );
    assert_eq!(b.requests(), vec!["GET /".to_string()]);
    assert_eq!(
        root_gets(a),
        before,
        "the global server must receive no request"
    );
}

#[test]
fn library_list_containers_other_forms_and_errors() {
    let a = global_a();
    let _serial = LIST_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    // `s3://` and bucket-shaped first segments list the global server.
    for uri in ["s3://", "s3://some-bucket/", "s3://data.example.com/"] {
        let listed = s3dlio::list_containers(uri).expect("list the global server");
        let names: Vec<&str> = listed.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(names, vec!["bucket-of-a"], "{uri}");
        assert_eq!(listed[0].uri, "s3://bucket-of-a", "{uri}");
    }

    // A portless IP is an error and sends nothing.
    let before = root_gets(a);
    let err = s3dlio::list_containers("s3://10.0.0.5/").expect_err("portless IP");
    assert!(
        format!("{err:#}").contains("IP address with no port"),
        "got: {err:#}"
    );
    assert_eq!(root_gets(a), before);
}
