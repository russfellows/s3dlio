// examples/direct_bounce_bench.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Benchmark of the `direct://` (O_DIRECT) bounce-buffer strategies, PUT, GET and MIXED.
//!
//! One process measures ONE arm at one concurrency / object size / read mix, time-based
//! (default 5 s ramp + 15 s measurement window). The driver script
//! `scripts/direct_bounce_bench.sh` builds the rig, runs every arm N times in alternating order
//! and takes medians. Run it by hand with:
//!
//! ```text
//! cargo run --release --example direct_bounce_bench -- \
//!     --dir /mnt/rig/bench --arm C64 --conc 4 --size-mib 1024 --read-pct 70
//! ```
//!
//! Arms (`--arm`):
//!
//! * `A`: PUT with a private buffer per call of `min(size, 1 GiB)` (one copy, one write); GET
//!   into one fresh page-aligned buffer, read directly (zero copies, fresh page faults).
//! * `B`: PUT with a private 64 MiB buffer per call, reused in steps (copy, write, repeat);
//!   GET as `A`.
//! * `C4`, `C16`, `C64`: the shared per-node ping-pong pool with 4/16/64 MiB buffers for both
//!   directions (PUT copies chunk k+1 while chunk k is written; GET reads chunk k+1 while
//!   chunk k is copied out).
//! * `aligned`: PUT from a page-aligned source (one write, no copy: the reference); GET as `A`.
//!
//! PUT sources are shared by all workers: `--source unaligned` (default; a `Vec` offset by 16
//! bytes, so O_DIRECT cannot use it) or `aligned`. Reports, as one JSON line on stdout: per
//! direction GiB/s and ops, CPU time (getrusage user+sys), minor page faults, peak RSS, and the
//! pool's wait time / fallbacks. It exits non-zero if the measured window is shorter than 90 % of
//! what was asked or no I/O completed, so a run that "finished inside its own ramp" cannot pass.

use bytes::Bytes;
use s3dlio::direct_bounce::{
    __bench_configure, bounce_memory_cap, bounce_stats, BounceConfig, BusyPolicy, GetStrategy,
    PutStrategy,
};
use s3dlio::file_store_direct::{ConfigurableFileSystemObjectStore, FileSystemConfig};
use s3dlio::memory::AlignedBuf;
use s3dlio::object_store::ObjectStore;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const MIB: usize = 1024 * 1024;
const GIB: usize = 1024 * MIB;
const ALIGN: usize = 4096;

struct Args(HashMap<String, String>);

impl Args {
    fn parse() -> Self {
        let mut map = HashMap::new();
        let mut it = std::env::args().skip(1);
        while let Some(k) = it.next() {
            let Some(key) = k.strip_prefix("--") else {
                eprintln!("unexpected argument {k}");
                std::process::exit(2);
            };
            let Some(v) = it.next() else {
                eprintln!("--{key} needs a value");
                std::process::exit(2);
            };
            map.insert(key.to_string(), v);
        }
        Self(map)
    }
    fn get<T: std::str::FromStr>(&self, key: &str, default: T) -> T {
        match self.0.get(key) {
            None => default,
            Some(v) => v.parse().unwrap_or_else(|_| {
                eprintln!("bad value for --{key}: {v}");
                std::process::exit(2)
            }),
        }
    }
    fn string(&self, key: &str, default: &str) -> String {
        self.0
            .get(key)
            .cloned()
            .unwrap_or_else(|| default.to_string())
    }
}

struct AlignedOwner(AlignedBuf, usize);
impl AsRef<[u8]> for AlignedOwner {
    fn as_ref(&self) -> &[u8] {
        &self.0.as_slice()[..self.1]
    }
}

/// The PUT payload: unaligned (a `Vec` sliced 16 bytes in) or page-aligned.
fn make_source(size: usize, aligned: bool) -> Bytes {
    if aligned {
        let mut b = AlignedBuf::new(size, ALIGN).expect("aligned source");
        for (i, x) in b.as_mut_slice().iter_mut().enumerate() {
            *x = (i.wrapping_mul(131) >> 3) as u8;
        }
        let bytes = Bytes::from_owner(AlignedOwner(b, size));
        assert!((bytes.as_ptr() as usize).is_multiple_of(ALIGN));
        bytes
    } else {
        let mut v = vec![0u8; size + 32];
        for (i, x) in v.iter_mut().enumerate() {
            *x = (i.wrapping_mul(131) >> 3) as u8;
        }
        let mut b = Bytes::from(v).slice(16..16 + size);
        if (b.as_ptr() as usize).is_multiple_of(ALIGN) {
            // (cannot happen for a 16-byte offset into a page-aligned block; be explicit)
            b = b.slice(0..size);
        }
        assert!(!(b.as_ptr() as usize).is_multiple_of(ALIGN));
        b
    }
}

fn rusage() -> (f64, f64, i64, i64, i64) {
    // SAFETY: getrusage fills a plain struct.
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    let tv = |t: libc::timeval| t.tv_sec as f64 + t.tv_usec as f64 / 1e6;
    (
        tv(ru.ru_utime),
        tv(ru.ru_stime),
        ru.ru_minflt,
        ru.ru_majflt,
        ru.ru_maxrss,
    )
}

#[derive(Default)]
struct Dir {
    ops: AtomicU64,
    bytes: AtomicU64,
    lat_ns: AtomicU64,
}

struct Shared {
    stop: AtomicBool,
    measuring: AtomicBool,
    put: Dir,
    get: Dir,
    errors: AtomicU64,
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn main() {
    let a = Args::parse();
    let dir = a.string("dir", "");
    if dir.is_empty() {
        eprintln!("usage: direct_bounce_bench --dir <path on the rig> --arm A|B|C4|C16|C64|aligned [--conc N] [--size-mib N] [--read-pct 0..100] [--ramp-secs 5] [--window-secs 15] [--source unaligned|aligned] [--busy wait|fallback] [--pairs N]");
        std::process::exit(2);
    }
    let arm = a.string("arm", "C64");
    let conc: usize = a.get("conc", 1);
    let size = a.get::<usize>("size-mib", 256) * MIB;
    let read_pct: u64 = a.get("read-pct", 0);
    let ramp: f64 = a.get("ramp-secs", 5.0);
    let window: f64 = a.get("window-secs", 15.0);
    let busy = match a.string("busy", "wait").as_str() {
        "fallback" => BusyPolicy::Fallback,
        _ => BusyPolicy::Wait,
    };
    let pairs: usize = a.get("pairs", 1);
    let rep: u32 = a.get("rep", 0);
    let source_aligned = arm == "aligned" || a.string("source", "unaligned") == "aligned";

    let (put, get, buffer_size) = match arm.as_str() {
        "A" => (
            PutStrategy::PerCall { cap: GIB },
            GetStrategy::Direct,
            64 * MIB,
        ),
        "B" => (
            PutStrategy::PerCall { cap: 64 * MIB },
            GetStrategy::Direct,
            64 * MIB,
        ),
        "aligned" => (PutStrategy::Pooled, GetStrategy::Direct, 64 * MIB),
        c if c.starts_with('C') => {
            let mib: usize = c[1..].parse().unwrap_or_else(|_| {
                eprintln!("bad arm {c}");
                std::process::exit(2)
            });
            (PutStrategy::Pooled, GetStrategy::Pooled, mib * MIB)
        }
        other => {
            eprintln!("unknown arm {other}");
            std::process::exit(2);
        }
    };
    assert!(
        __bench_configure(BounceConfig {
            buffer_size,
            pairs_per_node: pairs,
            busy,
            put,
            get,
        }),
        "bounce configuration must be set before the first transfer"
    );

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("runtime");
    rt.block_on(async move {
        let store = Arc::new(ConfigurableFileSystemObjectStore::new(FileSystemConfig {
            direct_io: true,
            ..FileSystemConfig::default()
        }));
        std::fs::create_dir_all(&dir).expect("bench dir");
        let source = make_source(size, source_aligned);
        let uris: Vec<String> = (0..conc)
            .map(|i| format!("direct://{dir}/obj-{i}.bin"))
            .collect();
        // Seed every object (not measured).
        for u in &uris {
            store.put(u, source.clone()).await.expect("seed put");
        }

        let shared = Arc::new(Shared {
            stop: AtomicBool::new(false),
            measuring: AtomicBool::new(false),
            put: Dir::default(),
            get: Dir::default(),
            errors: AtomicU64::new(0),
        });
        let mut workers = Vec::new();
        for (i, uri) in uris.iter().cloned().enumerate() {
            let (store, shared, source) = (store.clone(), shared.clone(), source.clone());
            workers.push(tokio::spawn(async move {
                let mut rng = 0xC0FFEEu64 ^ (i as u64).wrapping_mul(0x1234_5678_9ABC_DEF1);
                while !shared.stop.load(Ordering::Relaxed) {
                    let is_read = splitmix(&mut rng) % 100 < read_pct;
                    let t0 = Instant::now();
                    let ok = if is_read {
                        match store.get(&uri).await {
                            Ok(b) => b.len() == size,
                            Err(_) => false,
                        }
                    } else {
                        store.put(&uri, source.clone()).await.is_ok()
                    };
                    if !ok {
                        shared.errors.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    if shared.measuring.load(Ordering::Relaxed) {
                        let d = if is_read { &shared.get } else { &shared.put };
                        d.ops.fetch_add(1, Ordering::Relaxed);
                        d.bytes.fetch_add(size as u64, Ordering::Relaxed);
                        d.lat_ns
                            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
                    }
                }
            }));
        }

        tokio::time::sleep(Duration::from_secs_f64(ramp)).await;
        let (u0, s0, f0, mf0, _) = rusage();
        let stats0 = bounce_stats();
        let t0 = Instant::now();
        shared.measuring.store(true, Ordering::SeqCst);
        tokio::time::sleep(Duration::from_secs_f64(window)).await;
        shared.measuring.store(false, Ordering::SeqCst);
        let elapsed = t0.elapsed().as_secs_f64();
        let (u1, s1, f1, mf1, rss_kib) = rusage();
        let stats1 = bounce_stats();
        shared.stop.store(true, Ordering::SeqCst);
        for w in workers {
            let _ = w.await;
        }

        let gib = |b: u64| b as f64 / GIB as f64;
        let (pb, gb) = (
            shared.put.bytes.load(Ordering::Relaxed),
            shared.get.bytes.load(Ordering::Relaxed),
        );
        let (po, go) = (
            shared.put.ops.load(Ordering::Relaxed),
            shared.get.ops.load(Ordering::Relaxed),
        );
        let lat = |d: &Dir, ops: u64| {
            if ops == 0 {
                0.0
            } else {
                d.lat_ns.load(Ordering::Relaxed) as f64 / ops as f64 / 1e6
            }
        };
        let d = |a: [u64; 2], b: [u64; 2], i: usize| b[i] - a[i];
        println!(
            "{{\"arm\":\"{arm}\",\"rep\":{rep},\"conc\":{conc},\"size_mib\":{},\"read_pct\":{read_pct},\
\"source\":\"{}\",\"busy\":\"{}\",\"pairs\":{pairs},\"buffer_mib\":{},\"window_secs\":{elapsed:.3},\
\"put_ops\":{po},\"get_ops\":{go},\"put_gibps\":{:.3},\"get_gibps\":{:.3},\"total_gibps\":{:.3},\
\"put_lat_ms\":{:.1},\"get_lat_ms\":{:.1},\
\"cpu_user_s\":{:.3},\"cpu_sys_s\":{:.3},\"cpu_total_s\":{:.3},\"minflt\":{},\"majflt\":{},\
\"peak_rss_mib\":{},\"errors\":{},\
\"pool\":{{\"memory_cap_mib\":{},\"put_leases\":{},\"get_leases\":{},\"put_waited\":{},\"get_waited\":{},\
\"put_wait_ms\":{},\"get_wait_ms\":{},\"put_fallbacks\":{},\"get_fallbacks\":{},\"pools_created\":{},\
\"implicit_put\":{},\"implicit_get\":{},\"implicit_bytes\":{},\"pool_unavailable\":{},\"alignment_mismatches\":{}}}}}",
            size / MIB,
            if source_aligned { "aligned" } else { "unaligned" },
            if busy == BusyPolicy::Wait { "wait" } else { "fallback" },
            buffer_size / MIB,
            gib(pb) / elapsed,
            gib(gb) / elapsed,
            gib(pb + gb) / elapsed,
            lat(&shared.put, po),
            lat(&shared.get, go),
            u1 - u0,
            s1 - s0,
            (u1 - u0) + (s1 - s0),
            f1 - f0,
            mf1 - mf0,
            rss_kib / 1024,
            shared.errors.load(Ordering::Relaxed),
            bounce_memory_cap() / MIB,
            d(stats0.leases, stats1.leases, 0),
            d(stats0.leases, stats1.leases, 1),
            d(stats0.waited, stats1.waited, 0),
            d(stats0.waited, stats1.waited, 1),
            d(stats0.wait_ns, stats1.wait_ns, 0) / 1_000_000,
            d(stats0.wait_ns, stats1.wait_ns, 1) / 1_000_000,
            d(stats0.fallbacks, stats1.fallbacks, 0),
            d(stats0.fallbacks, stats1.fallbacks, 1),
            stats1.pools_created,
            d(stats0.implicit_fallbacks, stats1.implicit_fallbacks, 0),
            d(stats0.implicit_fallbacks, stats1.implicit_fallbacks, 1),
            stats1.implicit_fallback_bytes - stats0.implicit_fallback_bytes,
            stats1.pool_unavailable - stats0.pool_unavailable,
            stats1.alignment_mismatches - stats0.alignment_mismatches,
        );
        // A run that did not measure what was asked is not a result.
        let ok = elapsed >= window * 0.9 && po + go > 0 && shared.errors.load(Ordering::Relaxed) == 0;
        if !ok {
            eprintln!(
                "INVALID RUN: window {elapsed:.2}s of {window}s asked, {po} puts + {go} gets, {} errors",
                shared.errors.load(Ordering::Relaxed)
            );
            std::process::exit(3);
        }
    });
}
