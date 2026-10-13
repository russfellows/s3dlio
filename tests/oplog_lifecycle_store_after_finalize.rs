// tests/oplog_lifecycle_store_after_finalize.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log lifecycle test for s3dlio issue #170 (real store after finalize).
//
// Scenario: a REAL store built with store_for_uri_with_logger(file://...,
// global_logger()) while the op-log is Active keeps working after finalize: its
// put/get still succeed, but they add no row to the finished op-log, do not
// panic, and print no "channel full" (only the single "op-log finalized" note).
//
// stderr is only observable from outside the process, so the test re-runs
// itself as a child (marker env var + --nocapture) and the parent asserts the
// child really ran before it inspects the child's stderr. Own process.

use bytes::Bytes;
use s3dlio::{finalize_op_logger, global_logger, init_op_logger, store_for_uri_with_logger};
use std::process::Command;

const CHILD_ENV: &str = "S3DLIO_TEST_STORE_AFTER_FINALIZE_CHILD";
const RAN_MARKER: &str = "CHILD-SCENARIO-RAN";
const TEST_NAME: &str = "store_built_before_finalize_is_inert_after_it";
const HEADER_LINE: &str = "idx\tthread\top\tclient_id\tn_objects\tbytes\tendpoint\tfile\terror\tstart\tfirst_byte\tend\tduration_ns";

async fn child_scenario() {
    let dir = tempfile::tempdir().unwrap();
    let oplog = dir.path().join("ops.tsv.zst");
    init_op_logger(oplog.to_str().unwrap()).unwrap();
    let base = format!("file://{}", dir.path().display());
    let store = store_for_uri_with_logger(&base, global_logger()).unwrap();

    // Active: one PUT, one row.
    store
        .put(&format!("{base}/before.bin"), Bytes::from_static(b"before"))
        .await
        .unwrap();
    finalize_op_logger();
    let finished = std::fs::read(&oplog).unwrap();

    // After finalize the store still works but is no longer recorded.
    for i in 0..50 {
        store
            .put(
                &format!("{base}/after-{i}.bin"),
                Bytes::from_static(b"after"),
            )
            .await
            .unwrap();
    }
    assert_eq!(
        &store.get(&format!("{base}/after-0.bin")).await.unwrap()[..],
        b"after"
    );
    assert_eq!(
        std::fs::read(&oplog).unwrap(),
        finished,
        "the finished op-log changed after finalize"
    );

    let text = String::from_utf8(zstd::decode_all(finished.as_slice()).unwrap()).unwrap();
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some(HEADER_LINE));
    let rows: Vec<Vec<&str>> = lines.map(|l| l.split('\t').collect()).collect();
    assert_eq!(rows.len(), 1, "only the pre-finalize PUT is recorded");
    assert_eq!(rows[0].len(), 13);
    assert_eq!(rows[0][2], "PUT");
    assert!(rows[0][7].ends_with("before.bin"), "{:?}", rows[0]);
    println!("{RAN_MARKER}: 51 post-finalize store calls added no row");
}

#[test]
fn store_built_before_finalize_is_inert_after_it() {
    if std::env::var_os(CHILD_ENV).is_some() {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(child_scenario());
        return;
    }

    let out = Command::new(std::env::current_exe().expect("current_exe"))
        .args(["--exact", TEST_NAME, "--nocapture", "--test-threads=1"])
        .env(CHILD_ENV, "1")
        .env_remove("S3DLIO_OPLOG_BUF")
        .env_remove("S3DLIO_OPLOG_LOSSLESS")
        .output()
        .expect("spawn child test process");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    assert!(
        out.status.success(),
        "child failed: status={:?}\nstdout={stdout}\nstderr={stderr}",
        out.status
    );
    assert!(
        stdout.contains(RAN_MARKER),
        "child did not run the scenario\nstdout={stdout}\nstderr={stderr}"
    );
    assert!(stdout.contains("1 passed"), "child stdout: {stdout}");
    for unwanted in ["channel full", "writer has stopped", "panicked"] {
        assert_eq!(
            stderr.lines().filter(|l| l.contains(unwanted)).count(),
            0,
            "unexpected '{unwanted}' on stderr\nstderr={stderr}"
        );
    }
    assert_eq!(
        stderr
            .lines()
            .filter(|l| l.contains("op-log finalized"))
            .count(),
        1,
        "exactly one finalized note (the store's first post-finalize call prints it)\nstderr={stderr}"
    );
}
