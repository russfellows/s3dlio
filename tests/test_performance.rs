// tests/test_performance.rs
//
use std::time::Instant;
//use s3dlio::data_gen::{generate_random_data, generate_controlled_data};
use s3dlio::data_gen::generate_controlled_data;

/// Measure generate_random_data() over 100 runs of 1 GiB each.
/// Run with: `cargo test -- --ignored --nocapture perf_random`
/*
#[test]
#[ignore]
fn perf_random() {
    let size = 1024 * 1024 * 1024; // 1 GiB
    let runs = 100;

    let t0 = Instant::now();
    for _ in 0..runs {
        let _buf = generate_random_data(size);
    }
    let elapsed = t0.elapsed();
    println!(
        "generate_random_data: {} × {:.1} MB → {:?}",
        runs,
        size as f64 / (1024.0*1024.0),
        elapsed
    );
}
*/

/// Measure generate_controlled_data() over 100 runs of 1 GiB each.
/// Run with: `cargo test -- --ignored --nocapture perf_controlled`
#[test]
#[ignore]
fn perf_controll1() {
    let size = 1024 * 1024 * 1024; // 1 GiB
    let runs = 50;
    let dedup = 3;
    let compress = 2;

    let t0 = Instant::now();
    for _ in 0..runs {
        let _buf = generate_controlled_data(size, dedup, compress);
    }
    let elapsed = t0.elapsed();
    println!(
        "generate_controlled_data: {} × {:.1} MB (dedup={}, compress={}) → {:?}",
        runs,
        size as f64 / (1024.0*1024.0),
        dedup,
        compress,
        elapsed
    );
}

/// Measure generate_controlled_data() over 100 runs of 1 GiB each.
/// Run with: `cargo test -- --ignored --nocapture perf_controlled`
#[test]
#[ignore]
fn perf_controll2() {
    let size = 1024 * 1024 * 1024; // 1 GiB
    let runs = 50;
    let dedup = 1;
    let compress = 1;

    let t0 = Instant::now();
    for _ in 0..runs {
        let _buf = generate_controlled_data(size, dedup, compress);
    }
    let elapsed = t0.elapsed();
    println!(
        "generate_controlled_data: {} × {:.1} MB (dedup={}, compress={}) → {:?}",
        runs,
        size as f64 / (1024.0*1024.0),
        dedup,
        compress,
        elapsed
    );
}
