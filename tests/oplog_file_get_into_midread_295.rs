// tests/oplog_file_get_into_midread_295.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// #295 / #267: the op-log row of a REAL file:// / direct:// `get_into` that does not run to
// completion, through the store `store_for_uri_with_logger` builds. The rows are the ones the
// network stores write (tests/oplog_caller_buffer_rows.rs):
//
//   * a read that FAILS part-way is one GET row with `bytes` = the destination positions filled
//     so far and the error text;
//   * a future that is DROPPED part-way (cancellation) is one GET row with `error = "cancelled"`
//     and the positions filled so far;
//   * a future that was never polled started no work and writes no row.
//
// The interleavings are deterministic (the future is polled by hand once: that reads the first
// 16 MiB chunk). One process, one op-log (it is process-global).

#[allow(dead_code)] // shared test helpers; each test binary uses a subset
mod common;
#[path = "common/odirect.rs"]
mod odirect;
#[path = "common/step.rs"]
mod step;

use common::oplog_rows::{read_rows, Row, COL_BYTES, COL_ERROR, COL_FILE, COL_N_OBJECTS, COL_OP};
use s3dlio::object_store::{store_for_uri_with_logger, ReceiveProgress};
use s3dlio::{finalize_op_logger, global_logger, init_op_logger};
use std::pin::pin;
use std::time::Duration;

const MIB: usize = 1024 * 1024;
const CHUNK: usize = 16 * MIB;
const LEN: usize = 40 * MIB;
const COL_FIRST_BYTE: usize = 10;

fn rows_of<'r>(rows: &'r [Row], name: &str) -> Vec<&'r Row> {
    rows.iter()
        .filter(|r| r[COL_FILE].ends_with(&format!("/{name}")))
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failed_dropped_or_unstarted_get_into_writes_the_row_the_network_stores_write() {
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let Some(scratch) = odirect::odirect_scratch_dir("oplog-midread-295-") else {
        return;
    };
    let dir = scratch.path();
    let oplog = dir.join("ops.tsv.zst");
    init_op_logger(oplog.to_str().unwrap()).unwrap();
    let data = dir.join("data");
    std::fs::create_dir_all(&data).unwrap();

    let make = |name: &str| {
        let path = data.join(name);
        let bytes: Vec<u8> = (0..LEN).map(|i| (i % 160) as u8).collect();
        std::fs::write(&path, bytes).unwrap();
        path
    };

    for scheme in ["file", "direct"] {
        let uri = |name: &str| format!("{scheme}://{}", data.join(name).display());
        let store = store_for_uri_with_logger(&uri("x"), global_logger()).unwrap();
        let tick = || std::thread::sleep(Duration::from_millis(30));

        // Shrinks to 20 MiB after the first chunk: fails part-way.
        let name = format!("fail-{scheme}");
        let target = uri(&name);
        let path = make(&name);
        let mut dst = vec![0u8; LEN];
        let progress = ReceiveProgress::new();
        {
            let mut fut = pin!(store.get_into(&target, &mut dst, &progress));
            assert!(step::poll_once(fut.as_mut()).is_pending());
            std::fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .unwrap()
                .set_len(20 * MIB as u64)
                .unwrap();
            let err = step::run_to_end(fut.as_mut()).expect_err("shrunk: an error");
            assert!(format!("{err:#}").contains("length mismatch"), "{err:#}");
        }
        assert!(
            progress.filled() >= CHUNK as u64 && progress.filled() <= 20 * MIB as u64,
            "{scheme}: filled {}",
            progress.filled()
        );

        // Overwritten in place (same size): fails at the end with everything read.
        let name = format!("changed-{scheme}");
        let target = uri(&name);
        let path = make(&name);
        let mut dst = vec![0u8; LEN];
        let progress = ReceiveProgress::new();
        {
            let mut fut = pin!(store.get_into(&target, &mut dst, &progress));
            assert!(step::poll_once(fut.as_mut()).is_pending());
            tick();
            use std::io::{Seek, SeekFrom, Write};
            let mut f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            f.seek(SeekFrom::Start(30 * MIB as u64)).unwrap();
            f.write_all(&[0xEE; 4096]).unwrap();
            drop(f);
            let err = step::run_to_end(fut.as_mut()).expect_err("modified: an error");
            assert!(format!("{err:#}").contains("version changed"), "{err:#}");
        }
        assert_eq!(progress.filled(), LEN as u64);

        // Dropped after the first chunk: cancelled, with the count at the time of the drop.
        let name = format!("drop-{scheme}");
        let target = uri(&name);
        make(&name);
        let mut dst = vec![0u8; LEN];
        let progress = ReceiveProgress::new();
        {
            let mut fut = pin!(store.get_into(&target, &mut dst, &progress));
            assert!(step::poll_once(fut.as_mut()).is_pending());
        } // dropped here
        assert_eq!(
            progress.filled(),
            CHUNK as u64,
            "{scheme}: progress survives the drop"
        );

        // Never polled: no row.
        let name = format!("never-{scheme}");
        let target = uri(&name);
        make(&name);
        let mut dst = vec![0u8; LEN];
        let progress = ReceiveProgress::new();
        drop(store.get_into(&target, &mut dst, &progress));
        assert_eq!(progress.filled(), 0);
    }

    finalize_op_logger();
    let rows = read_rows(&oplog);

    for scheme in ["file", "direct"] {
        let fail = rows_of(&rows, &format!("fail-{scheme}"));
        assert_eq!(
            fail.len(),
            1,
            "{scheme}: one row for the failed read: {fail:#?}"
        );
        let row = fail[0];
        assert_eq!(row[COL_OP], "GET");
        let bytes: u64 = row[COL_BYTES].parse().unwrap();
        assert!(
            (CHUNK as u64..=20 * MIB as u64).contains(&bytes),
            "{scheme}: a failed read keeps what it filled: {row:?}"
        );
        assert!(row[COL_ERROR].contains("length mismatch"), "{row:?}");
        assert_eq!(row[COL_N_OBJECTS], "1");
        assert!(!row[COL_FIRST_BYTE].is_empty(), "bytes arrived: {row:?}");

        let changed = rows_of(&rows, &format!("changed-{scheme}"));
        assert_eq!(changed.len(), 1, "{changed:#?}");
        assert_eq!(changed[0][COL_BYTES], LEN.to_string(), "{:?}", changed[0]);
        assert!(
            changed[0][COL_ERROR].contains("version changed"),
            "{:?}",
            changed[0]
        );

        let dropped = rows_of(&rows, &format!("drop-{scheme}"));
        assert_eq!(
            dropped.len(),
            1,
            "{scheme}: one row for the dropped read: {dropped:#?}"
        );
        assert_eq!(dropped[0][COL_OP], "GET");
        assert_eq!(dropped[0][COL_ERROR], "cancelled", "{:?}", dropped[0]);
        assert_eq!(
            dropped[0][COL_BYTES],
            CHUNK.to_string(),
            "the count at the time of the drop: {:?}",
            dropped[0]
        );
        assert!(!dropped[0][COL_FIRST_BYTE].is_empty());

        assert!(
            rows_of(&rows, &format!("never-{scheme}")).is_empty(),
            "{scheme}: a future that was never polled writes no row"
        );
    }
}
