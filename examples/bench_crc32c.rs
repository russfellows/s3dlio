// examples/bench_crc32c.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Throughput of the CRC implementations behind the writers' checksum (issue #256): the
// old CRC-32 (`crc32fast`, the cost before #256), the `crc32c` crate, `crc-fast`'s
// CRC-32/ISCSI (CRC-32C) streaming and one-shot, and `s3dlio::checksum::Crc32c`, the type
// the writers actually use. No build flag is set: this measures
// what a default `cargo build --release` (and a wheel) gets.
//
//   cargo build --release --example bench_crc32c
//   target/release/examples/bench_crc32c
//
// Run it on a quiet host (no other build or heavy process): the numbers are GB/s (1e9
// bytes/s) for ONE core, the median of several repeats of a fixed-time loop.

use std::hint::black_box;
use std::time::{Duration, Instant};

const SIZES: &[(&str, usize)] = &[
    ("4 KiB", 4 << 10),
    ("64 KiB", 64 << 10),
    ("1 MiB", 1 << 20),
    ("8 MiB", 8 << 20),
];
const REPEATS: usize = 9;
const TARGET: Duration = Duration::from_millis(250);

/// Median GB/s of `f` over `bytes`-sized calls.
fn gbps(bytes: usize, mut f: impl FnMut()) -> f64 {
    // Calibrate: how many calls fill TARGET.
    let mut n = 1usize;
    loop {
        let t = Instant::now();
        for _ in 0..n {
            f();
        }
        if t.elapsed() >= TARGET / 4 || n >= 1 << 24 {
            break;
        }
        n *= 2;
    }
    let mut samples = Vec::with_capacity(REPEATS);
    for _ in 0..REPEATS {
        let t = Instant::now();
        for _ in 0..n {
            f();
        }
        let secs = t.elapsed().as_secs_f64();
        samples.push((bytes * n) as f64 / secs / 1e9);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn main() {
    // Known answers first: a fast wrong answer is worthless.
    assert_eq!(crc32c::crc32c(b"123456789"), 0xe306_9283);
    assert_eq!(crc_fast::crc32_iscsi(b"123456789"), 0xe306_9283);
    assert_eq!(crc32fast::hash(b"123456789"), 0xcbf4_3926);
    assert_eq!(s3dlio::checksum::crc32c_of(b"123456789"), 0xe306_9283);
    println!(
        "crc-fast calculator: {}",
        crc_fast::get_calculator_target(crc_fast::CrcAlgorithm::Crc32Iscsi)
    );
    println!(
        "{:<8} {:>14} {:>14} {:>16} {:>16} {:>16}",
        "buffer",
        "crc32fast",
        "crc32c crate",
        "crc-fast stream",
        "crc-fast oneshot",
        "s3dlio Crc32c"
    );
    println!(
        "{:<8} {:>14} {:>14} {:>16} {:>16} {:>16}",
        "", "(CRC-32,old)", "(CRC-32C)", "(CRC-32C)", "(CRC-32C)", "(CRC-32C,shipped)"
    );

    // Pseudo-random, not zeros.
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let big: Vec<u8> = (0..8usize << 20)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 24) as u8
        })
        .collect();

    for &(label, size) in SIZES {
        let buf = &big[..size];
        let a = gbps(size, || {
            let mut h = crc32fast::Hasher::new();
            h.update(black_box(buf));
            black_box(h.finalize());
        });
        let b = gbps(size, || {
            black_box(crc32c::crc32c_append(0, black_box(buf)));
        });
        let c = gbps(size, || {
            let mut d = crc_fast::Digest::new(crc_fast::CrcAlgorithm::Crc32Iscsi);
            d.update(black_box(buf));
            black_box(d.finalize());
        });
        let d = gbps(size, || {
            black_box(crc_fast::crc32_iscsi(black_box(buf)));
        });
        let e = gbps(size, || {
            let mut h = s3dlio::checksum::Crc32c::new();
            h.update(black_box(buf));
            black_box(h.finalize());
        });
        println!("{label:<8} {a:>14.2} {b:>14.2} {c:>16.2} {d:>16.2} {e:>16.2}");
    }
}
