// tests/oplog_file_get_into_295.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// #295 / #267: op-log rows of a REAL file:// and direct:// `get_into` / `head`, through the
// stores `store_for_uri_with_logger` builds. Every call is exactly one row: a GET row carries
// the number of destination positions filled (also on an error), a HEAD row the size (0 for
// an absent file), and the inner store writes no row of its own.
//
// One process, one op-log (it is process-global and cannot be re-initialised).

#[allow(dead_code)] // shared test helpers; each test binary uses a subset
mod common;

use common::oplog_rows::{read_rows, Row, COL_BYTES, COL_ERROR, COL_FILE, COL_N_OBJECTS, COL_OP};
use s3dlio::object_store::{store_for_uri_with_logger, ReceiveProgress};
use s3dlio::{finalize_op_logger, global_logger, init_op_logger};

const COL_FIRST_BYTE: usize = 10;
const COL_END: usize = 11;

fn rows_of<'r>(rows: &'r [Row], name: &str) -> Vec<&'r Row> {
    rows.iter()
        .filter(|r| r[COL_FILE].ends_with(&format!("/{name}")))
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn file_and_direct_get_into_and_head_write_one_row_per_call() {
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let dir = tempfile::tempdir().unwrap();
    let oplog = dir.path().join("ops.tsv.zst");
    init_op_logger(oplog.to_str().unwrap()).unwrap();

    let data = dir.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    std::fs::write(data.join("ok-file"), vec![7u8; 1_048_576]).unwrap();
    std::fs::write(data.join("ok-direct"), vec![9u8; 1_048_576]).unwrap();
    std::fs::write(data.join("short-file"), vec![7u8; 100]).unwrap();
    std::fs::write(data.join("empty-file"), b"").unwrap();

    for (scheme, ok, short) in [
        ("file", "ok-file", "short-file"),
        ("direct", "ok-direct", "short-file"),
    ] {
        let uri = |name: &str| format!("{scheme}://{}/{name}", data.display());
        let store = store_for_uri_with_logger(&uri(ok), global_logger()).unwrap();

        // GET success.
        let mut dst = vec![0u8; 1_048_576];
        let progress = ReceiveProgress::new();
        store.get_into(&uri(ok), &mut dst, &progress).await.unwrap();
        assert_eq!(progress.filled(), 1_048_576);
        assert!(dst
            .iter()
            .all(|b| *b == if scheme == "file" { 7 } else { 9 }));

        // GET length mismatch: an error row with nothing filled.
        let mut dst = vec![0u8; 200];
        store
            .get_into(&uri(short), &mut dst, &ReceiveProgress::new())
            .await
            .unwrap_err();

        // GET of a missing file: an error row.
        store
            .get_into(&uri("gone"), &mut dst, &ReceiveProgress::new())
            .await
            .unwrap_err();

        // HEAD: present, empty, absent.
        assert_eq!(store.head(&uri(ok)).await.unwrap().unwrap().size, 1_048_576);
        assert_eq!(
            store.head(&uri("empty-file")).await.unwrap().unwrap().size,
            0
        );
        assert_eq!(store.head(&uri("gone-too")).await.unwrap(), None);
    }

    finalize_op_logger();
    let rows = read_rows(&oplog);

    for ok in ["ok-file", "ok-direct"] {
        let get_rows: Vec<_> = rows_of(&rows, ok)
            .into_iter()
            .filter(|r| r[COL_OP] == "GET")
            .collect();
        assert_eq!(get_rows.len(), 1, "{ok}: one GET row, got {get_rows:#?}");
        let row = get_rows[0];
        assert_eq!(row[COL_BYTES], "1048576", "{row:?}");
        assert_eq!(row[COL_N_OBJECTS], "1");
        assert_eq!(row[COL_ERROR], "", "success has no error: {row:?}");
        assert!(!row[COL_FIRST_BYTE].is_empty(), "{row:?}");
        let head_rows: Vec<_> = rows_of(&rows, ok)
            .into_iter()
            .filter(|r| r[COL_OP] == "HEAD")
            .collect();
        assert_eq!(head_rows.len(), 1, "{ok}: one HEAD row, got {head_rows:#?}");
        assert_eq!(head_rows[0][COL_BYTES], "1048576");
        assert_eq!(head_rows[0][COL_ERROR], "");
        assert_eq!(rows_of(&rows, ok).len(), 2, "{ok}: GET and HEAD only");
    }

    // short-file is read through both schemes: two GET rows, both errors with 0 bytes.
    let short: Vec<_> = rows_of(&rows, "short-file");
    assert_eq!(short.len(), 2, "{short:#?}");
    for row in short {
        assert_eq!(row[COL_OP], "GET");
        assert_eq!(row[COL_BYTES], "0");
        assert!(row[COL_ERROR].contains("length mismatch"), "{row:?}");
        assert_eq!(
            row[COL_FIRST_BYTE], row[COL_END],
            "no first byte: first_byte is the end time"
        );
    }

    let gone = rows_of(&rows, "gone");
    assert_eq!(gone.len(), 2, "{gone:#?}");
    for row in gone {
        assert_eq!(row[COL_OP], "GET");
        assert_eq!(row[COL_BYTES], "0");
        assert!(row[COL_ERROR].contains("not found"), "{row:?}");
    }

    let empty = rows_of(&rows, "empty-file");
    assert_eq!(empty.len(), 2, "{empty:#?}");
    for row in empty {
        assert_eq!(row[COL_OP], "HEAD");
        assert_eq!(row[COL_BYTES], "0");
        assert_eq!(row[COL_ERROR], "");
    }

    // An absent file is a successful HEAD row with 0 bytes.
    let absent = rows_of(&rows, "gone-too");
    assert_eq!(absent.len(), 2, "{absent:#?}");
    for row in absent {
        assert_eq!(row[COL_OP], "HEAD");
        assert_eq!(row[COL_BYTES], "0");
        assert_eq!(row[COL_ERROR], "");
    }
}
