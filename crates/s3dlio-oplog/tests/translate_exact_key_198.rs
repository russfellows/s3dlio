// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Regression tests for s3dlio issue #198 (op-log replay): translating a row onto a replay
//! target must keep the object key exactly as the writer logged it. `translate_uri` used to
//! strip EVERY leading `/` of the key (`trim_start_matches('/')`), so a row for the object
//! `/x` (logged as `bkt//x`) replayed against `s3://t/` addressed `s3://t/x`, a different
//! object. Only the structural separator between the container and the key may be removed.

mod oplog_support;

use oplog_support::{replay, row, tsv};
use s3dlio_oplog::translate_uri;

/// Keys that are legal object keys and must survive replay untouched.
const KEYS: &[&str] = &["/x", "a//b", "x", "//x", "a/b/", "dir/"];

/// How the writer composes the `file` column from a key.
type ComposeFile = Box<dyn Fn(&str) -> String>;

/// (endpoint, how the writer composes `file` from the key)
fn writer_shapes() -> Vec<(&'static str, ComposeFile)> {
    vec![
        ("s3://srcbkt", Box::new(|k| format!("srcbkt/{k}"))),
        (
            "s3://127.0.0.1:9000",
            Box::new(|k| format!("127.0.0.1:9000/bkt/{k}")),
        ),
        ("az://acct", Box::new(|k| format!("acct/cont/{k}"))),
        (
            "https://acct.blob.core.windows.net",
            Box::new(|k| format!("acct.blob.core.windows.net/cont/{k}")),
        ),
        ("gs://gbkt", Box::new(|k| format!("gbkt/{k}"))),
    ]
}

#[test]
fn writer_rows_translate_with_the_exact_key() {
    for (endpoint, compose) in writer_shapes() {
        for key in KEYS {
            let file = compose(key);
            for (target, expected) in [
                ("s3://t/", format!("s3://t/{key}")),
                ("s3://t", format!("s3://t/{key}")),
                ("az://acct/cont/p", format!("az://acct/cont/p/{key}")),
            ] {
                assert_eq!(
                    translate_uri(&file, endpoint, target).expect("translates"),
                    expected,
                    "{endpoint} {file:?} -> {target}"
                );
            }
        }
    }
}

#[tokio::test]
async fn replayed_rows_reach_the_executor_with_the_exact_key() {
    for (endpoint, compose) in writer_shapes() {
        for key in KEYS {
            let file = compose(key);
            let f = tsv(&[row(0, "PUT", endpoint, &file)]);
            let (res, seen) = replay(&f, Some("s3://replaybkt/"), false, None).await;
            assert!(res.is_ok(), "{endpoint} {file:?}: {res:?}");
            assert_eq!(
                seen,
                vec![format!("PUT s3://replaybkt/{key}")],
                "{endpoint} {file:?}"
            );
        }
    }
}

#[test]
fn local_rows_drop_only_the_root_separator() {
    // `file` is an absolute path: its first `/` is the filesystem root, a separator.
    assert_eq!(
        translate_uri("/tmp/data/x.dat", "file://", "s3://t/").unwrap(),
        "s3://t/tmp/data/x.dat"
    );
    assert_eq!(
        translate_uri("/tmp//data/x.dat", "direct://", "s3://t/").unwrap(),
        "s3://t/tmp//data/x.dat"
    );
}

#[test]
fn hand_written_prefix_rows_drop_exactly_one_separator() {
    // The endpoint is a literal prefix of `file`; one `/` after it is the separator.
    assert_eq!(
        translate_uri("/bucket/data/file.bin", "/bucket/", "s3://t").unwrap(),
        "s3://t/data/file.bin"
    );
    assert_eq!(
        translate_uri("/bucket/data/file.bin", "/bucket", "s3://t").unwrap(),
        "s3://t/data/file.bin"
    );
    // A second `/` is part of the key.
    assert_eq!(
        translate_uri("/bucket//data/file.bin", "/bucket", "s3://t").unwrap(),
        "s3://t//data/file.bin"
    );
    assert_eq!(
        translate_uri("/bucket//data/file.bin", "/bucket/", "s3://t").unwrap(),
        "s3://t//data/file.bin"
    );
}

#[test]
fn unrecognised_rows_drop_only_one_leading_separator() {
    // Rows that are not a writer shape keep the pre-#184 result for a single leading '/'
    // (pinned by reader_uri_184.rs); only a second '/' is no longer discarded.
    assert_eq!(
        translate_uri("/data/file1.bin", "http://10.0.0.1:8000", "s3://t").unwrap(),
        "s3://t/data/file1.bin"
    );
    assert_eq!(
        translate_uri("//data/file1.bin", "http://10.0.0.1:8000", "s3://t").unwrap(),
        "s3://t//data/file1.bin"
    );
}
