// tests/fork_globals_297.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Fork safety of the crate's process-global state (#297): each test builds the state in the
//! parent, forks, and proves the CHILD can still use it (a child that waits for threads of the
//! parent hangs: `in_child` reports where it hangs with gdb, kills it after 20 s and fails).
//!
//! Own test binary on purpose (like `oplog_lifecycle_fork_*`): `fork()` in a process whose other
//! threads are running unrelated tests can leave a lock of THIS process held in the child
//! (an unrelated test inside `std::env::set_var` holds std's env lock, for example), which is a
//! property of the test harness, not of the code under test. The tests here run one at a time.
#![cfg(unix)]

use rayon::prelude::*;
use std::sync::Mutex;

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// Run `f` in a forked child; 0 = it returned true, -9 = it hung, other = failed/signalled.
fn in_child(f: impl FnOnce() -> bool) -> i32 {
    // SAFETY: the child runs the closure and `_exit`s.
    let pid = unsafe { libc::fork() };
    if pid == 0 {
        // Let the parent attach gdb to a hung child (ptrace_scope=1 forbids it otherwise).
        unsafe { libc::prctl(0x5961_6d61, libc::c_ulong::MAX, 0, 0, 0) };
        let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_or(false);
        unsafe { libc::_exit(if ok { 0 } else { 1 }) };
    }
    let mut status = 0;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        if unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG) } == pid {
            break;
        }
        if std::time::Instant::now() > deadline {
            if let Ok(out) = std::process::Command::new("gdb")
                .args([
                    "-p",
                    &pid.to_string(),
                    "-batch",
                    "-ex",
                    "thread apply all bt 25",
                ])
                .output()
            {
                eprintln!(
                    "HUNG fork child {pid}:\n{}",
                    String::from_utf8_lossy(&out.stdout)
                );
            }
            unsafe { libc::kill(pid, libc::SIGKILL) };
            unsafe { libc::waitpid(pid, &mut status, 0) };
            return -9;
        }
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    if libc::WIFEXITED(status) {
        libc::WEXITSTATUS(status)
    } else {
        -(libc::WTERMSIG(status))
    }
}

/// A parallel fill that must run on s3dlio's pool; returns (checksum << 1) | on_pool.
fn par_sum() -> u64 {
    s3dlio::s3_client::with_rayon_pool(|| {
        let on_pool = rayon::current_thread_index().is_some();
        let mut v = vec![1u8; 8 << 20];
        v.par_chunks_mut(1 << 20).for_each(|c| c[0] = 2);
        (v.iter().map(|&b| u64::from(b)).sum::<u64>() << 1) | u64::from(on_pool)
    })
}

#[test]
fn the_global_runtime_is_rebuilt_in_a_fork_child() {
    let _g = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let run =
        |v: i32| s3dlio::s3_client::run_on_global_rt(async move { Ok::<_, anyhow::Error>(v) });
    assert_eq!(run(42).expect("parent"), 42);
    let status = in_child(|| run(7).ok() == Some(7));
    assert_eq!(
        status, 0,
        "child status {status}: the global runtime must work in a fork child"
    );
    assert_eq!(run(1).expect("the parent is unaffected"), 1);
}

#[test]
fn the_rayon_pool_is_rebuilt_in_a_fork_child_with_the_size_configured_there() {
    let _g = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // The parent's pool has whatever size it was first built with (this process runs several
    // tests, in any order); it is not 2.
    s3dlio::s3_client::configure_thread_pools(3);
    let parent = par_sum();
    assert_eq!(parent & 1, 1, "runs on s3dlio's pool");
    let parent_threads = s3dlio::s3_client::rayon_pool_threads();
    assert_ne!(parent_threads, 2);
    let status = in_child(|| {
        // Configuring builds nothing: the child's pool is built by its first parallel call, with
        // the size configured IN THE CHILD, so it is a different pool from the inherited one.
        s3dlio::s3_client::configure_thread_pools(2);
        let r = par_sum();
        r == parent && s3dlio::s3_client::rayon_pool_threads() == 2
    });
    assert_eq!(status, 0, "child status {status}");
    assert_eq!(
        s3dlio::s3_client::rayon_pool_threads(),
        parent_threads,
        "the parent keeps its pool"
    );
    assert_eq!(par_sum(), parent);
}

#[test]
fn one_shot_data_generation_works_in_a_fork_child() {
    let _g = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // dgen-data's one-shot generator uses rayon's pool of the calling context: ours.
    let n = |size| s3dlio::data_gen_alt::generate_controlled_data_alt(size, 1, 1, None).len();
    assert_eq!(n(24 << 20), 24 << 20);
    let status = in_child(|| n(24 << 20) == 24 << 20);
    assert_eq!(status, 0, "child status {status}");
}

#[test]
fn the_in_place_fill_and_the_npz_builder_work_in_a_fork_child() {
    let _g = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let fill = || {
        let mut b = vec![0u8; 16 << 20];
        s3dlio::data_gen::fill_controlled_data(&mut b, 1, 1);
        b.len()
    };
    let npz = || {
        s3dlio::data_formats::npz::generate_npz_bytes_raw(&[1024, 1024, 8], "<f4", 1)
            .map(|v| v.len())
            .unwrap_or(0)
    };
    assert_eq!(fill(), 16 << 20);
    assert!(npz() > 8 << 20);
    let status = in_child(|| fill() == 16 << 20 && npz() > 8 << 20);
    assert_eq!(status, 0, "child status {status}");
}

#[test]
fn a_streaming_generator_works_in_a_fork_child_of_a_process_that_used_one() {
    let _g = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // dgen-data keeps its OWN global rayon pool (a private OnceLock, not fork-aware): the parent
    // builds it with a first generator; the child's generator must not wait on it.
    let fill_all = || {
        let mut g = s3dlio::data_gen_alt::ObjectGenAlt::new(24 << 20, 1, 1);
        let mut buf = vec![0u8; 24 << 20];
        g.fill_chunk(&mut buf)
    };
    assert_eq!(fill_all(), 24 << 20);
    let status = in_child(|| fill_all() == 24 << 20);
    assert_eq!(status, 0, "child status {status}");
}
