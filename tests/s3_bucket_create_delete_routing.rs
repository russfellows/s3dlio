// tests/s3_bucket_create_delete_routing.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Regression tests for s3dlio issue #199: bucket create/delete for a host-form URI
// (`s3://host:port/bkt`) must go to the NAMED server and act on the bucket `bkt` --
// not create a bucket called `host:port` (or `host:port/bkt`) on the
// `AWS_ENDPOINT_URL` server. Covers the library entry points behind the Python
// `create_bucket` / `delete_bucket` / `put(should_create_bucket=True)`, the CLI
// `put --create-bucket`, and `s3_copy::upload_files`. A portless IP is the #178 error.
//
// CLI tests run the real `s3-cli`; library tests use a single global mock for the
// whole binary (the library's global client is process-wide) and unique bucket names.

#[path = "common/recording_s3.rs"]
mod recording_s3;
use recording_s3::{run_s3_cli, text, MockS3};
use std::sync::OnceLock;

/// Requests the mock received whose recorded text mentions `needle`.
fn mentioning(m: &MockS3, needle: &str) -> Vec<String> {
    m.requests()
        .into_iter()
        .filter(|r| r.contains(needle))
        .collect()
}

#[test]
fn cli_put_create_bucket_goes_to_the_server_named_in_the_uri() {
    let a = MockS3::start(&[]); // AWS_ENDPOINT_URL: the wrong server
    let b = MockS3::start(&[]); // the server named in the URI
    let uri = format!("s3://127.0.0.1:{}/bkt199put/pfx/", b.port);

    let out = run_s3_cli(
        &["put", &uri, "--create-bucket", "-n", "1", "-s", "1024"],
        &[("AWS_ENDPOINT_URL", &a.url())],
    );

    assert_eq!(
        a.requests(),
        Vec::<String>::new(),
        "put {uri} --create-bucket: the AWS_ENDPOINT_URL server must receive no request ({})",
        text(&out)
    );
    assert!(
        b.requests().contains(&"PUT /bkt199put".to_string()),
        "the named server must receive CreateBucket for bkt199put, got {:?} ({})",
        b.requests(),
        text(&out)
    );
    assert!(
        mentioning(&b, "127.0.0.1").is_empty(),
        "no bucket named after the server may be requested: {:?}",
        b.requests()
    );
    assert!(out.status.success(), "{}", text(&out));
}

// ----------------------------------------------------------------------------
// Library entry points (what the Python bindings call)
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

#[test]
fn library_create_and_delete_bucket_for_host_form_uri_hit_the_named_server() {
    let a = global_a();
    let b = MockS3::start(&[]);

    for form in [
        format!("s3://127.0.0.1:{}/bkt199lib", b.port),
        format!("s3://127.0.0.1:{}/bkt199lib/", b.port),
    ] {
        s3dlio::s3_utils::create_bucket_for_uri(&form).expect("create on the named server");
        s3dlio::s3_utils::delete_bucket_for_uri(&form).expect("delete on the named server");
    }

    assert_eq!(
        b.requests(),
        vec![
            "PUT /bkt199lib".to_string(),
            "DELETE /bkt199lib".to_string(),
            "PUT /bkt199lib".to_string(),
            "DELETE /bkt199lib".to_string(),
        ]
    );
    assert_eq!(
        mentioning(a, "bkt199lib"),
        Vec::<String>::new(),
        "the global server must not see the bucket"
    );
    assert_eq!(
        mentioning(a, "127.0.0.1"),
        Vec::<String>::new(),
        "no bucket named after the server may be requested"
    );
}

#[test]
fn library_create_and_delete_bucket_plain_forms_use_the_global_server() {
    let a = global_a();

    for (name, bucket) in [
        ("bkt199plain1", "bkt199plain1"),
        ("s3://bkt199plain2", "bkt199plain2"),
        ("s3://bkt199plain3/", "bkt199plain3"),
        (
            "s3://data.bkt199plain4.example.com/",
            "data.bkt199plain4.example.com",
        ),
    ] {
        s3dlio::s3_utils::create_bucket_for_uri(name).expect("create on the global server");
        s3dlio::s3_utils::delete_bucket_for_uri(name).expect("delete on the global server");
        assert_eq!(
            mentioning(a, bucket),
            vec![format!("PUT /{bucket}"), format!("DELETE /{bucket}")],
            "{name}"
        );
    }
}

#[test]
fn library_create_bucket_portless_ip_is_the_178_error_and_sends_nothing() {
    let a = global_a();

    let err = s3dlio::s3_utils::create_bucket_for_uri("s3://10.0.0.5/bkt199ip")
        .expect_err("a portless IP is not a bucket and not a server");
    assert!(
        format!("{err:#}").contains("IP address with no port"),
        "got: {err:#}"
    );
    let err = s3dlio::s3_utils::delete_bucket_for_uri("s3://10.0.0.5/bkt199ip")
        .expect_err("a portless IP is not a bucket and not a server");
    assert!(
        format!("{err:#}").contains("IP address with no port"),
        "got: {err:#}"
    );
    assert_eq!(mentioning(a, "bkt199ip"), Vec::<String>::new());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn s3_copy_upload_files_creates_the_bucket_and_uploads_on_the_named_server() {
    let a = global_a();
    let b = MockS3::start(&[]);
    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"));
    let file = dir.join("s3copy199.dat");
    std::fs::write(&file, b"payload").expect("write the local file to upload");
    let dest = format!("s3://127.0.0.1:{}/bkt199copy/pfx/", b.port);

    s3dlio::s3_copy::upload_files(&dest, &[file], 1, true, None)
        .await
        .expect("upload_files to the named server");

    assert_eq!(
        b.requests(),
        vec![
            "PUT /bkt199copy".to_string(),
            "PUT /bkt199copy/pfx/s3copy199.dat".to_string()
        ]
    );
    assert_eq!(
        mentioning(a, "bkt199copy"),
        Vec::<String>::new(),
        "the global server must not see the bucket or the object"
    );
    assert_eq!(mentioning(a, "127.0.0.1"), Vec::<String>::new());

    let err = s3dlio::s3_copy::upload_files("s3://10.0.0.5/bkt199copy/pfx/", &[dir], 1, true, None)
        .await
        .expect_err("a portless IP is not a bucket and not a server");
    assert!(
        format!("{err:#}").contains("IP address with no port"),
        "got: {err:#}"
    );
}
