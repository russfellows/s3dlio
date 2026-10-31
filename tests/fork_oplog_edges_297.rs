#![cfg(unix)]
// tests/fork_oplog_edges_297.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! #297 round 3: a `finalize_op_logger()` that races an init that is still creating its file (the
//! op-log `Starting` state, blocked here on a FIFO without a reader) is not lost: it waits and
//! finishes the session that was published. Own test binary.
use std::time::Duration;

static ONE: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn mkfifo(dir: &std::path::Path, name: &str) -> std::path::PathBuf {
    let f = dir.join(name);
    let c = std::ffi::CString::new(f.to_str().unwrap()).unwrap();
    assert_eq!(unsafe { libc::mkfifo(c.as_ptr(), 0o600) }, 0);
    f
}

#[test]
fn a_finalize_during_starting_is_not_lost() {
    let _g = ONE.lock().unwrap_or_else(|e| e.into_inner());
    let dir = std::env::temp_dir().join(format!("r2f_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let fifo = mkfifo(&dir, "log.fifo");
    let fs = fifo.to_str().unwrap().to_string();
    let starter = std::thread::spawn(move || s3dlio::s3_logger::init_op_logger(&fs));
    std::thread::sleep(Duration::from_millis(300));
    let fin = std::thread::spawn(|| s3dlio::s3_logger::finalize_op_logger());
    std::thread::sleep(Duration::from_millis(300));
    let fin_returned_while_starting = fin.is_finished();
    let _reader = std::fs::File::open(&fifo);
    starter.join().unwrap().unwrap();
    fin.join().unwrap();
    let active = s3dlio::s3_logger::global_logger().is_some();
    s3dlio::s3_logger::finalize_op_logger();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        !active,
        "finalize (returned immediately: {fin_returned_while_starting}) was lost: the session that was starting is Active"
    );
}
