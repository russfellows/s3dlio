// examples/get_into_local_bench.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Benchmark of the caller-buffer GET (`get_into`, #295 / #267) against the allocating `get`
//! for `file://` and `direct://` (O_DIRECT).
//!
//! ```text
//! cargo run --release --example get_into_local_bench -- \
//!     --dir /mnt/nvme/bench --sizes-mib 1,64,1024 --reps 5 [--drop-caches]
//! ```
//!
//! Variants per size (GiB/s = size / wall time of one call, median of `--reps` after one warm-up):
//!
//! * `file get` / `direct get`: the allocating GET (a fresh buffer every call).
//! * `file get_into reused`: `get_into` into one buffer allocated once (what an adapter that
//!   recycles its buffers does).
//! * `file get_into fresh`: `get_into` into a buffer allocated (and zero-filled) inside the
//!   timed call, the fair comparison with `get`.
//! * `direct get_into aligned`: a page-aligned destination, O_DIRECT straight into it.
//! * `direct get_into bounce`: a destination at an odd address, through the bounce pool.
//!
//! `file://` reads are served from the page cache unless `--drop-caches` is given (it runs
//! `sync` and `sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'` before EVERY call, so every
//! `file://` call reads the device); `direct://` bypasses the cache either way. Sizes below
//! 1 MiB are not meaningful for the direct bounce pool; the default sizes are 1, 64 and 1024 MiB.

use s3dlio::file_store_direct::ConfigurableFileSystemObjectStore;
use s3dlio::object_store::{ObjectStore, ReceiveProgress};
use s3dlio::FileSystemObjectStore;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

const MIB: usize = 1024 * 1024;
const PAGE: usize = 4096;

fn arg(name: &str) -> Option<String> {
    let mut it = std::env::args();
    while let Some(a) = it.next() {
        if a == name {
            return it.next();
        }
    }
    None
}

fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

/// `sync` and drop the page cache (needs passwordless sudo); false when that is not possible.
fn drop_caches() -> bool {
    let synced = Command::new("sync").status().is_ok_and(|s| s.success());
    let dropped = Command::new("sudo")
        .args(["-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"])
        .status()
        .is_ok_and(|s| s.success());
    synced && dropped
}

/// Write `size` bytes of incompressible content to `path`.
fn make_file(path: &Path, size: usize) -> std::io::Result<()> {
    let mut chunk = vec![0u8; MIB];
    let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ size as u64;
    for word in chunk.chunks_mut(8) {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        word.copy_from_slice(&x.to_le_bytes());
    }
    let mut f = std::fs::File::create(path)?;
    let mut left = size;
    while left > 0 {
        let n = left.min(MIB);
        f.write_all(&chunk[..n])?;
        left -= n;
    }
    f.sync_all()
}

struct Stats {
    median: f64,
    min: f64,
    max: f64,
}

fn stats(size: usize, mut secs: Vec<f64>) -> Stats {
    secs.sort_by(|a, b| a.total_cmp(b));
    let gib = |s: f64| size as f64 / (1024.0 * MIB as f64) / s;
    Stats {
        median: gib(secs[secs.len() / 2]),
        min: gib(secs[secs.len() - 1]),
        max: gib(secs[0]),
    }
}

/// Time `$body` (it may `.await`) `$reps` times after one untimed warm-up; `$pre` runs before
/// every timed call, outside the timing.
macro_rules! measure {
    ($reps:expr, $pre:expr, $body:expr) => {{
        let mut secs = Vec::new();
        for i in 0..=$reps {
            $pre;
            let start = Instant::now();
            $body;
            if i > 0 {
                secs.push(start.elapsed().as_secs_f64());
            }
        }
        secs
    }};
}

/// `len` bytes at the first page-aligned address inside `backing` (which must be `len + PAGE`).
fn aligned_window(backing: &mut [u8], len: usize) -> &mut [u8] {
    let skew = backing.as_ptr().align_offset(PAGE);
    &mut backing[skew..skew + len]
}

fn report(scheme: &str, size: usize, variant: &str, s: Stats) {
    println!(
        "{scheme:<7} {:>5} MiB  {variant:<26} {:>7.2} GiB/s  (min {:.2}, max {:.2})",
        size / MIB,
        s.median,
        s.min,
        s.max
    );
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> anyhow::Result<()> {
    let dir = arg("--dir").unwrap_or_else(|| "/tmp/get_into_local_bench".to_string());
    let sizes: Vec<usize> = arg("--sizes-mib")
        .unwrap_or_else(|| "1,64,1024".to_string())
        .split(',')
        .map(|s| s.trim().parse::<usize>().map(|m| m * MIB))
        .collect::<Result<_, _>>()?;
    let reps: usize = arg("--reps").map_or(Ok(5), |s| s.parse())?;
    let cold_cache = flag("--drop-caches");
    if cold_cache && !drop_caches() {
        eprintln!("--drop-caches needs passwordless sudo; the file:// numbers are cached reads");
    }
    std::fs::create_dir_all(&dir)?;

    let file_store = FileSystemObjectStore::new();
    let direct_store = ConfigurableFileSystemObjectStore::with_direct_io();
    println!(
        "dir={dir} reps={reps} drop_caches={cold_cache} (file:// is {})",
        if cold_cache {
            "cold"
        } else {
            "warm: page cache"
        }
    );

    for &size in &sizes {
        let path = Path::new(&dir).join(format!("obj-{size}"));
        make_file(&path, size)?;
        let file_uri = format!("file://{}", path.display());
        let direct_uri = format!("direct://{}", path.display());
        let progress = ReceiveProgress::new();
        let nothing = || ();
        let cold = || {
            if cold_cache {
                drop_caches();
            }
        };

        // ---- file:// ----
        let t = measure!(reps, cold(), {
            let b = file_store.get(&file_uri).await?;
            assert_eq!(b.len(), size);
        });
        report("file", size, "get (allocating)", stats(size, t));

        let mut reused = vec![0u8; size];
        let t = measure!(reps, cold(), {
            file_store
                .get_into(&file_uri, &mut reused, &progress)
                .await?;
        });
        report("file", size, "get_into reused buffer", stats(size, t));
        drop(reused);

        let t = measure!(reps, cold(), {
            let mut fresh = vec![0u8; size];
            file_store
                .get_into(&file_uri, &mut fresh, &progress)
                .await?;
        });
        report("file", size, "get_into fresh buffer", stats(size, t));

        // ---- direct:// ----
        let t = measure!(reps, nothing(), {
            let b = direct_store.get(&direct_uri).await?;
            assert_eq!(b.len(), size);
        });
        report("direct", size, "get (allocating)", stats(size, t));

        // A page-aligned window inside a plain zeroed allocation (calloc: untouched pages are
        // not written until the read faults them in), the way an adapter gets an aligned buffer
        // without a memset pass.
        let mut reused_backing = vec![0u8; size + PAGE];
        let reused_window = aligned_window(&mut reused_backing, size);
        let t = measure!(reps, nothing(), {
            direct_store
                .get_into(&direct_uri, reused_window, &progress)
                .await?;
        });
        report("direct", size, "get_into aligned reused", stats(size, t));
        drop(reused_backing);

        let t = measure!(reps, nothing(), {
            let mut fresh_backing = vec![0u8; size + PAGE];
            let fresh = aligned_window(&mut fresh_backing, size);
            direct_store.get_into(&direct_uri, fresh, &progress).await?;
        });
        report("direct", size, "get_into aligned fresh", stats(size, t));

        let mut odd = vec![0u8; size + 8];
        let t = measure!(reps, nothing(), {
            direct_store
                .get_into(&direct_uri, &mut odd[8..], &progress)
                .await?;
        });
        report("direct", size, "get_into bounce (odd addr)", stats(size, t));
        drop(odd);

        std::fs::remove_file(&path)?;
    }
    Ok(())
}
