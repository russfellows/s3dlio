// tests/oplog_explicit_callers_golden.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Op-log golden test (s3dlio #169, W2 round 2): the library entry points that
// build their own store -- `CheckpointStore::open*` (save and load),
// `Reader::read_all_shards_concurrent*`, `MultiBackendDataset::from_uris` /
// `from_prefix` -- produce EXACTLY the rows they produced before W2.
//
// Before W2 they used `store_for_uri` (an unlogged store). On a bucket-form S3
// URI the S3 backend of that store wrote the helper row (`S3Ops` shape:
// endpoint = AWS_ENDPOINT_URL, file = the bare key, client_id = access key id)
// whenever the op-log was active. On file://, direct://, az:// and gs:// they
// wrote NO row. Round 1 made these entry points use a logged store, which
// changed the S3 rows to the store shape and added rows on every other backend;
// that is reverted here.
//
// The expected rows were CAPTURED FROM THE PARENT COMMIT (9473d2c) by running
// this same test there with OPLOG_GOLDEN_OUT=<file> (capture mode) into
// `tests/fixtures/oplog_explicit_callers_pre_w2.golden`.
//
// Own process (the op-log and the S3 client are process-global).

#[allow(dead_code)] // shared test helpers; each test binary uses a subset
mod common;

use common::mock_s3_server::ensure_mock_s3_server;
use common::oplog_rows::{read_rows, Row};
use s3dlio::checkpoint::manifest::{Manifest, ShardMeta};
use s3dlio::checkpoint::CheckpointStore;
use s3dlio::data_loader::async_pool_dataloader::MultiBackendDataset;
use s3dlio::{finalize_op_logger, init_op_logger, set_client_id};

const BUCKET: &str = "mock-bucket";
const GOLDEN: &str = include_str!("fixtures/oplog_explicit_callers_pre_w2.golden");

fn canonical(row: &Row, mock: &str) -> String {
    // The checkpoint writer stamps keys with the wall clock (`ckpt-<step>-<date>-<time>`),
    // and its manifest / latest pointer embed the time, so their byte counts can vary
    // by a few bytes between runs: both are normalized. Every other field is exact.
    let ts = regex::Regex::new(r"-\d{8}-\d{6}").unwrap();
    let scrub = |s: &str| {
        ts.replace_all(&s.replace(mock, "MOCK"), "-<TS>")
            .to_string()
    };
    let volatile_size = row[7].contains("/manifests/") || row[7].ends_with("latest.json");
    let bytes = if volatile_size {
        "<n>"
    } else {
        row[5].as_str()
    };
    let (start, first_byte, end) = (&row[9], &row[10], &row[11]);
    let fb = if first_byte.is_empty() {
        "none"
    } else if first_byte == start {
        "start"
    } else if first_byte == end {
        "end"
    } else {
        "mid"
    };
    format!(
        "op={} client_id={} n_objects={} bytes={} endpoint={} file={} error=[{}] first_byte={}",
        row[2],
        row[3],
        row[4],
        bytes,
        scrub(&row[6]),
        scrub(&row[7]),
        scrub(&row[8]),
        fb
    )
}

#[test]
fn explicit_callers_write_their_pre_w2_rows() {
    #[allow(deprecated)]
    {
        std::env::remove_var("S3DLIO_OPLOG_BUF");
        std::env::remove_var("S3DLIO_OPLOG_LOSSLESS");
    }
    let (port, _server) = ensure_mock_s3_server();
    let mock = format!("127.0.0.1:{port}");
    let dir = tempfile::tempdir().unwrap();
    let oplog = dir.path().join("ops.tsv.zst");
    init_op_logger(oplog.to_str().unwrap()).unwrap();
    // Neither the old S3 rows nor the rows of any other backend use this id.
    set_client_id("client-explicit").unwrap();

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();
    let mut outcomes: Vec<String> = Vec::new();

    // ------------------------------------------------------------ S3 (mock)
    let s3_ckpt = format!("s3://{BUCKET}/golden-explicit-ckpt");
    let ckpt = CheckpointStore::open(&s3_ckpt).unwrap();
    let saved = rt.block_on(ckpt.save(1, 0, "test", b"checkpoint-bytes", None));
    outcomes.push(format!("s3 save ok={}", saved.is_ok()));
    // The mock serves a fixed 4-byte body for every GET, so the manifest read
    // fails to parse; the GET rows are written regardless.
    let loaded = rt.block_on(ckpt.load_latest());
    outcomes.push(format!("s3 load_latest ok={}", loaded.is_ok()));
    // The concurrent shard reader, with a hand-made manifest.
    let mut manifest = Manifest::new("test".to_string(), 1, 0, 1);
    manifest.add_shard(ShardMeta {
        rank: 0,
        key: "shards/golden-shard-0.bin".to_string(),
        size: 4,
        etag: None,
        checksum: None,
    });
    manifest.mark_complete();
    let shards = rt.block_on(ckpt.reader().read_all_shards_concurrent(&manifest));
    outcomes.push(format!(
        "s3 read_all_shards_concurrent ok={} n={}",
        shards.is_ok(),
        shards.as_ref().map(|s| s.len()).unwrap_or(0)
    ));
    // MultiBackendDataset: from_uris (two gets through the dataset's store).
    let s3_uris: Vec<String> = ["golden-explicit/a.bin", "golden-explicit/b.bin"]
        .iter()
        .map(|k| format!("s3://{BUCKET}/{k}"))
        .collect();
    let ds = MultiBackendDataset::from_uris(s3_uris.clone()).unwrap();
    for u in &s3_uris {
        outcomes.push(format!(
            "s3 dataset get ok={}",
            rt.block_on(ds.store.get(u)).is_ok()
        ));
    }
    // from_prefix lists (no row) and the gets below read what it listed.
    let ds_prefix = rt
        .block_on(MultiBackendDataset::from_prefix(&format!(
            "s3://{BUCKET}/golden-explicit-list/"
        )))
        .unwrap();
    outcomes.push(format!("s3 from_prefix n={}", ds_prefix.uris.len()));
    for u in &ds_prefix.uris {
        outcomes.push(format!(
            "s3 prefix dataset get ok={}",
            rt.block_on(ds_prefix.store.get(u)).is_ok()
        ));
    }

    // ------------------------------------------------------------ file:// (no rows)
    let fdir = tempfile::tempdir().unwrap();
    let data_dir = fdir.path().join("data");
    std::fs::create_dir(&data_dir).unwrap();
    std::fs::write(data_dir.join("a.bin"), b"aaaa").unwrap();
    std::fs::write(data_dir.join("b.bin"), b"bbbbbb").unwrap();
    let furis: Vec<String> = ["a.bin", "b.bin"]
        .iter()
        .map(|n| format!("file://{}/{n}", data_dir.display()))
        .collect();
    let fds = MultiBackendDataset::from_uris(furis.clone()).unwrap();
    for u in &furis {
        outcomes.push(format!(
            "file dataset get ok={}",
            rt.block_on(fds.store.get(u)).is_ok()
        ));
    }
    std::fs::create_dir(fdir.path().join("ckpt")).unwrap();
    let fckpt = CheckpointStore::open(&format!("file://{}/ckpt", fdir.path().display())).unwrap();
    let saved = rt.block_on(fckpt.save(1, 0, "test", b"checkpoint-bytes", None));
    outcomes.push(format!("file save ok={}", saved.is_ok()));
    let loaded = rt.block_on(fckpt.load_latest());
    outcomes.push(format!(
        "file load_latest ok={} bytes={}",
        loaded.is_ok(),
        loaded
            .as_ref()
            .ok()
            .and_then(|o| o.as_ref())
            .map(|b| b.len())
            .unwrap_or(0)
    ));
    let fmanifest = rt
        .block_on(fckpt.reader().load_latest_manifest())
        .unwrap()
        .expect("file checkpoint manifest");
    let fshards = rt.block_on(fckpt.reader().read_all_shards_concurrent(&fmanifest));
    outcomes.push(format!(
        "file read_all_shards_concurrent ok={} n={}",
        fshards.is_ok(),
        fshards.as_ref().map(|s| s.len()).unwrap_or(0)
    ));

    finalize_op_logger();

    let rows = read_rows(&oplog);
    let mut lines: Vec<String> = rows.iter().map(|r| canonical(r, &mock)).collect();
    lines.sort();
    let actual = format!(
        "# outcomes\n{}\n# rows\n{}\n",
        outcomes.join("\n"),
        lines.join("\n")
    );

    if let Ok(out) = std::env::var("OPLOG_GOLDEN_OUT") {
        std::fs::write(&out, &actual).unwrap();
        eprintln!("captured {} rows into {out}", lines.len());
        return;
    }
    assert_eq!(
        actual, GOLDEN,
        "rows of the explicit callers differ from the rows captured from the parent (pre-W2)"
    );
    // No file:// row of any kind, as before W2.
    assert!(
        rows.iter().all(|r| !r[7].starts_with('/')),
        "file:// calls write no row: {rows:#?}"
    );
}
