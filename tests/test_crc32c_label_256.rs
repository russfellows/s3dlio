// tests/test_crc32c_label_256.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Regression tests for s3dlio issue #256: every writer labelled its checksum
// `crc32c:<hex>` but computed CRC-32 (IEEE) with `crc32fast`. The label is now true:
// the value is CRC-32C (Castagnoli), the one a GCS `crc32c` or an S3
// `x-amz-checksum-crc32c` carries.
//
// Known answers for the 9 bytes "123456789" (the standard check input):
//   CRC-32C (Castagnoli) = e3069283        CRC-32 (IEEE) = cbf43926

use s3dlio::object_store::store_for_uri;
use std::path::Path;

const CHECK_INPUT: &[u8] = b"123456789";
const CRC32C: &str = "crc32c:e3069283";
const CRC32_IEEE: &str = "crc32c:cbf43926"; // what the old code reported under the CRC-32C label

fn scratch(name: &str) -> std::path::PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("crc256-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

async fn writer_checksum(scheme: &str, dir: &Path) -> String {
    let base = format!("{scheme}://{}", dir.display());
    let store = store_for_uri(&base).unwrap();
    let uri = format!("{base}/obj.bin");
    let mut writer = store.get_writer(&uri).await.unwrap();
    writer.write_chunk(CHECK_INPUT).await.unwrap();
    let checksum = writer.checksum().expect("writers report a checksum");
    writer.finalize().await.unwrap();
    checksum
}

#[tokio::test]
async fn file_writer_reports_a_real_crc32c() {
    let dir = scratch("file");
    let got = writer_checksum("file", &dir).await;
    assert_ne!(
        got, CRC32_IEEE,
        "that is CRC-32 (IEEE) under a crc32c label"
    );
    assert_eq!(got, CRC32C);
}

#[tokio::test]
async fn direct_writer_reports_a_real_crc32c() {
    let dir = scratch("direct");
    let got = writer_checksum("direct", &dir).await;
    assert_ne!(
        got, CRC32_IEEE,
        "that is CRC-32 (IEEE) under a crc32c label"
    );
    assert_eq!(got, CRC32C);
}

#[tokio::test]
async fn chunked_writes_give_the_same_crc32c_as_one_write() {
    let dir = scratch("chunked");
    let base = format!("file://{}", dir.display());
    let store = store_for_uri(&base).unwrap();
    let mut writer = store.get_writer(&format!("{base}/obj.bin")).await.unwrap();
    for chunk in CHECK_INPUT.chunks(4) {
        writer.write_chunk(chunk).await.unwrap();
    }
    assert_eq!(writer.checksum().unwrap(), CRC32C);
    writer.finalize().await.unwrap();
}

/// `get_with_validation` compares against the bare hex of the same algorithm, so a
/// value a writer reported (minus the label) validates the object it wrote.
#[tokio::test]
async fn validation_accepts_the_crc32c_a_writer_reported() {
    let dir = scratch("validate");
    let base = format!("file://{}", dir.display());
    let store = store_for_uri(&base).unwrap();
    let uri = format!("{base}/obj.bin");
    store
        .put(&uri, bytes::Bytes::from_static(CHECK_INPUT))
        .await
        .unwrap();

    let ok = store.get_with_validation(&uri, Some("e3069283")).await;
    assert!(ok.is_ok(), "{ok:?}");
    let ieee = store.get_with_validation(&uri, Some("cbf43926")).await;
    assert!(
        ieee.is_err(),
        "a CRC-32 (IEEE) value is not the CRC-32C of the object"
    );
}
