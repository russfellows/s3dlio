// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Regression tests for s3dlio issue #184: the replayer rebuilt a URI as `endpoint + file`,
//! which is malformed for the rows s3dlio actually writes (`endpoint = s3://bkt,
//! file = bkt/key`, and the legacy inner row `endpoint = <AWS_ENDPOINT_URL>, file = key`).

mod oplog_support;

use oplog_support::{replay, row, tsv};

// ----------------------------------------------------------------------------
// #184: URI reconstruction
// ----------------------------------------------------------------------------

/// (endpoint, file, exact URI the original call used, key relative to its container/bucket)
const WRITER_ROWS: &[(&str, &str, &str, &str)] = &[
    // LoggedObjectStore, S3
    (
        "s3://srcbkt",
        "srcbkt/dir/obj1.bin",
        "s3://srcbkt/dir/obj1.bin",
        "dir/obj1.bin",
    ),
    // host-form S3: the server is part of the URI, the bucket follows it
    (
        "s3://127.0.0.1:9000",
        "127.0.0.1:9000/bkt/k/x.bin",
        "s3://127.0.0.1:9000/bkt/k/x.bin",
        "k/x.bin",
    ),
    // Azure (account/container/key)
    (
        "az://acct",
        "acct/cont/dir/blob.bin",
        "az://acct/cont/dir/blob.bin",
        "dir/blob.bin",
    ),
    // Azure https URL form
    (
        "https://acct.blob.core.windows.net",
        "acct.blob.core.windows.net/cont/dir/blob.bin",
        "https://acct.blob.core.windows.net/cont/dir/blob.bin",
        "dir/blob.bin",
    ),
    // GCS
    ("gs://gbkt", "gbkt/a/b.bin", "gs://gbkt/a/b.bin", "a/b.bin"),
    // local backends
    (
        "file://",
        "/tmp/data/x.dat",
        "file:///tmp/data/x.dat",
        "tmp/data/x.dat",
    ),
    (
        "direct://",
        "/mnt/nvme/y.dat",
        "direct:///mnt/nvme/y.dat",
        "mnt/nvme/y.dat",
    ),
];

#[tokio::test]
async fn writer_rows_rebuild_the_exact_original_uri() {
    for (endpoint, file, uri, _) in WRITER_ROWS {
        let f = tsv(&[row(0, "GET", endpoint, file)]);
        let (res, seen) = replay(&f, None, false, None).await;
        assert!(res.is_ok(), "{endpoint} {file}: {res:?}");
        assert_eq!(seen, vec![format!("GET {uri}")], "row {endpoint} {file}");
    }
}

#[tokio::test]
async fn writer_rows_translate_to_the_target_without_the_old_bucket() {
    for (endpoint, file, _, relative) in WRITER_ROWS {
        let f = tsv(&[row(0, "PUT", endpoint, file)]);
        let (res, seen) = replay(&f, Some("s3://replaybkt/"), false, None).await;
        assert!(res.is_ok(), "{endpoint} {file}: {res:?}");
        assert_eq!(
            seen,
            vec![format!("PUT s3://replaybkt/{relative}")],
            "row {endpoint} {file}"
        );
    }
}

#[tokio::test]
async fn bucket_level_rows_rebuild() {
    // LIST of a prefix keeps its trailing slash; a bare bucket has no path at all;
    // a recorded leading '/' in the key (issue #172) is preserved.
    let f = tsv(&[
        row(0, "LIST", "s3://bkt", "bkt/dir/"),
        row(1, "LIST", "s3://bkt", "bkt"),
        row(2, "GET", "s3://bkt", "bkt//data/x"),
    ]);
    let (res, seen) = replay(&f, None, false, None).await;
    assert!(res.is_ok(), "{res:?}");
    assert_eq!(
        seen,
        vec![
            "LIST s3://bkt/dir/".to_string(),
            "LIST s3://bkt".to_string(),
            "GET s3://bkt//data/x".to_string(),
        ]
    );
}

#[tokio::test]
async fn inner_row_without_bucket_is_an_error_and_is_never_executed() {
    // The legacy S3Ops row carries the server and the bare key, but not the bucket:
    // the URI cannot be reconstructed, so the row must fail, not be guessed or dropped.
    let rows = [
        row(0, "PUT", "http://127.0.0.1:19750", "dir/obj1.bin"),
        row(1, "PUT", "s3://srcbkt", "srcbkt/dir/obj1.bin"),
    ];
    let f = tsv(&rows);

    let (res, seen) = replay(&f, None, true, None).await;
    assert!(res.is_ok(), "continue_on_error: {res:?}");
    assert_eq!(seen, vec!["PUT s3://srcbkt/dir/obj1.bin".to_string()]);

    let (res, seen) = replay(&f, Some("s3://replaybkt/"), true, None).await;
    assert!(res.is_ok(), "continue_on_error with target: {res:?}");
    assert_eq!(seen, vec!["PUT s3://replaybkt/dir/obj1.bin".to_string()]);

    let (res, _) = replay(&f, None, false, None).await;
    let err = format!("{:#}", res.expect_err("an inner row must fail the replay"));
    assert!(
        err.contains("bucket") && err.contains("http://127.0.0.1:19750"),
        "error must say what is missing: {err}"
    );
}

#[tokio::test]
async fn inconsistent_or_unrecognised_rows_are_errors() {
    for (endpoint, file) in [
        ("s3://bkt", "other/key"), // file does not start with the endpoint's authority
        ("s3://bkt", "bkt2/key"),  // prefix of the name only
        ("unknown", "mybucket"),   // no scheme (CREATE_CONTAINER style row)
        ("", "key"),               // AWS_ENDPOINT_URL unset on an inner row
        ("ftp://host", "host/key"), // scheme s3dlio has no store for
    ] {
        let f = tsv(&[row(0, "GET", endpoint, file)]);
        let (res, seen) = replay(&f, None, false, None).await;
        assert!(res.is_err(), "{endpoint:?} {file:?} must be an error");
        assert!(
            seen.is_empty(),
            "{endpoint:?} {file:?} must not execute: {seen:?}"
        );
        let (res, seen) = replay(&f, Some("s3://t/"), false, None).await;
        assert!(
            res.is_err(),
            "{endpoint:?} {file:?} with target must be an error"
        );
        assert!(seen.is_empty());
    }
}
