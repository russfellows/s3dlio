// src/fork_tests.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Fork safety of the crate's process-global state (#297): each test builds the state in the
//! parent, forks, and proves that the CHILD can still use it (a child that waits for threads
//! of the parent hangs: `in_child` kills it after 20 s and fails the test).

use crate::fork_guard::test_support::in_child;
use rayon::prelude::*;

/// A parallel sum that must run on s3dlio's pool.
fn par_sum() -> u64 {
    crate::s3_client::with_rayon_pool(|| {
        let on_pool = rayon::current_thread_index().is_some();
        let mut v = vec![1u8; 8 << 20];
        v.par_chunks_mut(1 << 20).for_each(|c| c[0] = 2);
        (v.iter().map(|&b| u64::from(b)).sum::<u64>() << 1) | u64::from(on_pool)
    })
}

#[test]
fn the_global_runtime_is_rebuilt_in_a_fork_child() {
    // The parent builds and uses the global runtime.
    let ok = crate::s3_client::run_on_global_rt(async { Ok::<_, anyhow::Error>(41 + 1) })
        .expect("parent run_on_global_rt");
    assert_eq!(ok, 42);
    let status = in_child(|| {
        crate::s3_client::run_on_global_rt(async { Ok::<_, anyhow::Error>(7) }).ok() == Some(7)
    });
    assert_eq!(
        status, 0,
        "child status {status}: the global runtime must work in a fork child"
    );
    // The parent is unaffected.
    let again = crate::s3_client::run_on_global_rt(async { Ok::<_, anyhow::Error>(1) }).unwrap();
    assert_eq!(again, 1);
}

#[test]
fn the_rayon_pool_is_rebuilt_in_a_fork_child_and_never_built_at_import() {
    let parent = par_sum();
    assert_eq!(parent & 1, 1, "runs on s3dlio's pool");
    assert!(crate::s3_client::rayon_pool_built_here());
    let status = in_child(|| {
        // The inherited pool does not count as built here; configuring does not build one.
        let none_yet = !crate::s3_client::rayon_pool_built_here();
        crate::s3_client::configure_thread_pools(2);
        let still_none = !crate::s3_client::rayon_pool_built_here();
        let r = par_sum();
        none_yet && still_none && crate::s3_client::rayon_pool_built_here() && r == parent
    });
    assert_eq!(status, 0, "child status {status}");
    assert_eq!(par_sum(), parent, "the parent's pool still works");
}

#[test]
fn one_shot_data_generation_works_in_a_fork_child() {
    // dgen-data's one-shot generator uses rayon's pool of the calling context: ours.
    let n = |size| crate::data_gen_alt::generate_controlled_data_alt(size, 1, 1, None).len();
    assert_eq!(n(24 << 20), 24 << 20);
    let status = in_child(|| n(24 << 20) == 24 << 20);
    assert_eq!(status, 0, "child status {status}");
}

#[test]
fn the_in_place_fill_and_the_npz_builder_work_in_a_fork_child() {
    let fill = || {
        let mut b = vec![0u8; 16 << 20];
        crate::data_gen::fill_controlled_data(&mut b, 1, 1);
        b.len()
    };
    let npz = || {
        crate::data_formats::npz::generate_npz_bytes_raw(&[1024, 1024, 8], "<f4", 1)
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
    // dgen-data keeps its OWN global rayon pool (a private OnceLock, not fork-aware): the
    // parent builds it with a first generator, the child's generator must not wait on it.
    let fill_all = || {
        let mut g = crate::data_gen_alt::ObjectGenAlt::new(24 << 20, 1, 1);
        let mut buf = vec![0u8; 24 << 20];
        g.fill_chunk(&mut buf)
    };
    assert_eq!(fill_all(), 24 << 20);
    let status = in_child(|| fill_all() == 24 << 20);
    assert_eq!(status, 0, "child status {status}");
}
