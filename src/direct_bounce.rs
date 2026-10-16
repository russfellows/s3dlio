// src/direct_bounce.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! The shared bounce-buffer pool of `direct://` (O_DIRECT) PUT and GET.
//!
//! O_DIRECT needs aligned memory. A PUT whose source is page-aligned is written straight from
//! it (no copy, no pool). Every other transfer must be copied through aligned memory, and
//! allocating a fresh multi-hundred-MiB aligned buffer per call pays page faults and zeroing for
//! every call. This module instead keeps WARM buffers and reuses them.
//!
//! # Design
//!
//! * **One pool per NUMA node, shared by both directions.** A request uses the pool of the node
//!   its thread runs on (`sched_getcpu` mapped through `/sys/devices/system/node/node*/cpulist`).
//!   A host with one node (or no sysfs, or not Linux) has one pool. A node's pool is created
//!   lazily on its first use: its buffers are allocated and first touched on that node (an
//!   `mbind` to the node, best effort, plus first touch from a thread pinned to the node's CPUs),
//!   then kept for the life of the process.
//! * **Ping-pong pairs.** A pool holds `pairs` pairs of buffers (`S3DLIO_DIRECT_IO_BOUNCE_PAIRS_PER_NODE`,
//!   default 1) of `buffer_size` bytes (`S3DLIO_DIRECT_IO_BOUNCE_SIZE_MB`, 4..=64, default 64).
//!   A transfer of more than one buffer leases a PAIR for its whole duration (a smaller one leases a single buffer, see below):
//!   * PUT: chunk `k` is copied into buffer A and written while chunk `k+1` is copied into buffer
//!     B; then the buffers swap. The write runs on a helper thread, the copy on the caller's.
//!   * GET: chunk `k+1` is read into buffer A by the helper thread while chunk `k` is copied out
//!     of buffer B into the destination; then they swap.
//! * **Bounded POOLED memory.** The pool buffers themselves total exactly `nodes x pairs x 2 x
//!   size` (default 128 MiB per node): every buffer a request leases comes from there, so the bytes
//!   LEASED at any instant never exceed [`bounce_memory_cap`] under the default `wait` policy.
//!   That bound covers the pooled path only. Other direct-I/O memory is outside it (see below).
//! * **Busy policy** (`S3DLIO_DIRECT_IO_BOUNCE_WHEN_BUSY`): when every pair of the node is
//!   leased, `wait` (default) queues the request in strict FIFO order shared by PUTs and GETs, so
//!   neither direction can starve the other; `fallback` instead gives the request a private
//!   buffer of one bounce size (serial copy-then-I/O, no pipelining; extra memory is one buffer
//!   per concurrent request beyond the pool, counted in [`BounceStats::fallback_live_bytes`]).
//! * **Small objects** lease ONE buffer (not a pair) of the same pool; a transfer of more than
//!   one buffer leases two atomically.
//! * **Availability first when the pool cannot serve a request.** If a node's pool cannot be
//!   created (out of memory; retried every 30 s) or cannot satisfy the requested alignment, the
//!   transfer is NOT failed: it takes a private per-call bounce buffer of up to one bounce size.
//!   Those implicit allocations are not covered by the cap. Each one is counted
//!   ([`BounceStats::implicit_fallbacks`], [`BounceStats::implicit_fallback_bytes`], with the
//!   cause in [`BounceStats::pool_unavailable`] / [`BounceStats::alignment_mismatches`]) and the
//!   first one is logged. Their number is bounded only by the number of concurrent requests.
//! * **Outside the pooled bound, by design:** the opt-in `fallback` buffers and the implicit
//!   ones above, the page-sized zero-padded tail block of an aligned PUT (one 4 KiB buffer per
//!   concurrent aligned PUT), the object-sized destination a default-mode GET returns, the
//!   private buffers of `direct://` range reads (one per read, sized to the range; a range larger
//!   than the range pool's buffer is admitted by count, `capacity` at a time, and a store with no
//!   range pool allocates without admission), and the up-to-64 MiB staging of every unaligned
//!   streaming writer. None of these is counted except the implicit and `fallback` buffers.
//! * **Leases are RAII and owned by the blocking closure that does the I/O**: a cancelled
//!   future can never return a buffer to the pool while a blocking write or read still uses it.
//!   Buffers are acquired all-or-nothing in strict FIFO order (no hold-one/wait-for-one
//!   deadlock); a waiting request polls its cancel flag and abandons the wait before any I/O.
//! * The POOLED bound is observable: [`BounceStats::peak_leased_bytes`] is updated at the moment
//!   of every lease and never exceeds [`bounce_memory_cap`] with the default `wait` policy. It
//!   says nothing about the allocations listed above as outside the pooled bound.
//! * The GET destination is a raw pointer, so the same code fills the buffer a GET returns (a
//!   fresh page-aligned allocation) or, later, a buffer supplied by the caller (`get_into`),
//!   which may be unaligned.
//!
//! The per-call strategies (`PutStrategy::PerCall`, `GetStrategy::Direct`) are kept selectable
//! through [`__bench_configure`] so the benchmark can compare them with the pool.

use crate::constants::{
    direct_io_bounce_size, parse_direct_io_bounce_pairs, BOUNCE_CANCEL_POLL, BOUNCE_POOL_RETRY,
    DEFAULT_PAGE_SIZE, ENV_DIRECT_IO_BOUNCE_PAIRS, ENV_DIRECT_IO_BOUNCE_WHEN_BUSY,
    ENV_DIRECT_IO_GET_MODE,
};
use crate::memory::{AlignedAllocError, AlignedBuf};
use std::collections::VecDeque;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Condvar, Mutex, MutexGuard, OnceLock, PoisonError};
use std::time::Instant;
use tracing::warn;

// ---------------------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------------------

/// What a request does when every buffer pair of its node is in use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BusyPolicy {
    /// Wait in strict FIFO order (shared by PUTs and GETs). Default.
    Wait,
    /// Use a private buffer of one bounce size for this request.
    Fallback,
}

/// How an unaligned PUT gets its aligned buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PutStrategy {
    /// The shared ping-pong pool.
    Pooled,
    /// A private buffer per call of `min(round_up(len), cap)` bytes, filled and written serially
    /// (the pre-pool behaviour; kept for the benchmark: cap 1 GiB = one copy and one write).
    PerCall {
        /// Largest private buffer in bytes.
        cap: usize,
    },
}

/// How a `direct://` GET reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GetStrategy {
    /// Read straight into one fresh page-aligned buffer of the object's size (zero copies, fresh
    /// page faults per call).
    Direct,
    /// Read through the shared ping-pong pool and copy into the destination.
    Pooled,
}

/// Process-wide bounce configuration (environment, or [`__bench_configure`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BounceConfig {
    /// Size of one bounce buffer in bytes.
    pub buffer_size: usize,
    /// Buffer pairs per NUMA node.
    pub pairs_per_node: usize,
    /// Behaviour when all pairs are busy.
    pub busy: BusyPolicy,
    /// PUT strategy.
    pub put: PutStrategy,
    /// GET strategy.
    pub get: GetStrategy,
}

/// Parse [`ENV_DIRECT_IO_BOUNCE_WHEN_BUSY`]: `wait` (default) or `fallback`.
pub fn parse_busy_policy(raw: Option<&str>) -> BusyPolicy {
    match raw.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("fallback") => BusyPolicy::Fallback,
        _ => BusyPolicy::Wait,
    }
}

/// Parse [`ENV_DIRECT_IO_GET_MODE`]: `direct` (default) or `pooled`.
pub fn parse_get_strategy(raw: Option<&str>) -> GetStrategy {
    match raw.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("pooled") => GetStrategy::Pooled,
        _ => GetStrategy::Direct,
    }
}

impl BounceConfig {
    /// The configuration described by the environment.
    pub fn from_env() -> Self {
        // The `.env` policy first (#243, Blue integration item 5): `config()` latches this once,
        // so a setting that only the opted-in file defines must be in place before the read.
        crate::dotenv::apply();
        Self {
            buffer_size: direct_io_bounce_size(),
            pairs_per_node: parse_direct_io_bounce_pairs(
                std::env::var(ENV_DIRECT_IO_BOUNCE_PAIRS).ok().as_deref(),
            ),
            busy: parse_busy_policy(
                std::env::var(ENV_DIRECT_IO_BOUNCE_WHEN_BUSY)
                    .ok()
                    .as_deref(),
            ),
            put: PutStrategy::Pooled,
            get: parse_get_strategy(std::env::var(ENV_DIRECT_IO_GET_MODE).ok().as_deref()),
        }
    }
}

static CONFIG: OnceLock<BounceConfig> = OnceLock::new();

/// The effective configuration (fixed at first use).
pub fn config() -> &'static BounceConfig {
    CONFIG.get_or_init(BounceConfig::from_env)
}

/// Benchmark hook: fix the configuration before the first `direct://` transfer of the process.
/// Returns `false` (and changes nothing) when it was already fixed.
#[doc(hidden)]
pub fn __bench_configure(cfg: BounceConfig) -> bool {
    CONFIG.set(cfg).is_ok()
}

// ---------------------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------------------

/// Direction of a transfer, for fairness accounting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    /// PUT (write)
    Put,
    /// GET (read)
    Get,
}

impl Direction {
    fn idx(self) -> usize {
        match self {
            Direction::Put => 0,
            Direction::Get => 1,
        }
    }
}

#[derive(Default)]
struct Counters {
    leases: [AtomicU64; 2],
    waited: [AtomicU64; 2],
    wait_ns: [AtomicU64; 2],
    fallbacks: [AtomicU64; 2],
    pools_created: AtomicU64,
    /// Bytes of pool buffers leased right now / at the peak.
    leased_bytes: AtomicU64,
    peak_leased_bytes: AtomicU64,
    /// Bytes of pool buffers allocated (resident) so far.
    allocated_bytes: AtomicU64,
    /// Bytes of private fallback buffers alive right now / at the peak (opt-in policy; outside the cap).
    fallback_live_bytes: AtomicU64,
    peak_fallback_bytes: AtomicU64,
    /// Waits abandoned because the request was cancelled.
    cancelled_waits: AtomicU64,
    /// Implicit per-call bounce allocations (pool unavailable / alignment mismatch), by direction.
    implicit_fallbacks: [AtomicU64; 2],
    implicit_fallback_bytes: AtomicU64,
    pool_unavailable: AtomicU64,
    alignment_mismatches: AtomicU64,
}

static COUNTERS: Counters = Counters {
    leases: [AtomicU64::new(0), AtomicU64::new(0)],
    waited: [AtomicU64::new(0), AtomicU64::new(0)],
    wait_ns: [AtomicU64::new(0), AtomicU64::new(0)],
    fallbacks: [AtomicU64::new(0), AtomicU64::new(0)],
    pools_created: AtomicU64::new(0),
    leased_bytes: AtomicU64::new(0),
    peak_leased_bytes: AtomicU64::new(0),
    allocated_bytes: AtomicU64::new(0),
    fallback_live_bytes: AtomicU64::new(0),
    peak_fallback_bytes: AtomicU64::new(0),
    cancelled_waits: AtomicU64::new(0),
    implicit_fallbacks: [AtomicU64::new(0), AtomicU64::new(0)],
    implicit_fallback_bytes: AtomicU64::new(0),
    pool_unavailable: AtomicU64::new(0),
    alignment_mismatches: AtomicU64::new(0),
};

fn add_gauge(live: &AtomicU64, peak: &AtomicU64, n: u64) {
    let now = live.fetch_add(n, Ordering::AcqRel) + n;
    peak.fetch_max(now, Ordering::AcqRel);
}

/// Snapshot of the pool's use, per direction (`[put, get]`). `#[non_exhaustive]`: fields are added
/// as the accounting grows, so external code reads it and cannot construct it with a literal.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct BounceStats {
    /// Pairs leased from the pool.
    pub leases: [u64; 2],
    /// Of those, how many had to wait for a free pair.
    pub waited: [u64; 2],
    /// Total nanoseconds spent waiting for a pair.
    pub wait_ns: [u64; 2],
    /// Requests served from a private fallback buffer because every pair was busy.
    pub fallbacks: [u64; 2],
    /// Node pools created so far (each allocates `pairs x 2 x size` bytes).
    pub pools_created: u64,
    /// Bytes of pool buffers leased by requests right now.
    pub leased_bytes: u64,
    /// The highest `leased_bytes` ever seen: with the default `wait` policy it can never exceed
    /// [`bounce_memory_cap`] (a hard bound, counted at the moment of the lease, independent of
    /// throughput).
    pub peak_leased_bytes: u64,
    /// Bytes of pool buffers allocated so far (idle + leased): at most the cap.
    pub allocated_bytes: u64,
    /// Bytes of private fallback buffers alive now (`BusyPolicy::Fallback` only: deliberately
    /// outside the cap, one buffer per such request).
    pub fallback_live_bytes: u64,
    /// Highest `fallback_live_bytes` seen.
    pub peak_fallback_bytes: u64,
    /// Waits for a buffer abandoned because the request was cancelled.
    pub cancelled_waits: u64,
    /// IMPLICIT private bounce allocations made because the pool could not serve the request
    /// (availability first), per direction `[put, get]`, cumulative. They are outside the pooled
    /// cap, are not in `fallback_live_bytes`, and are not the opt-in `fallback` policy.
    pub implicit_fallbacks: [u64; 2],
    /// Cumulative bytes allocated by those implicit private bounce buffers.
    pub implicit_fallback_bytes: u64,
    /// Transfers for which no pooled (or busy-fallback) buffer could be provided: the node's pool
    /// could not be created, or the fallback buffer could not be allocated. Not a subset of
    /// `implicit_fallbacks`: it also ticks for a pooled-mode GET that then reads directly into
    /// its (already allocated) destination, which allocates no bounce buffer.
    pub pool_unavailable: u64,
    /// Transfers whose alignment the pool's buffers cannot satisfy.
    pub alignment_mismatches: u64,
}

/// Current pool statistics.
pub fn bounce_stats() -> BounceStats {
    let l = |a: &[AtomicU64; 2]| [a[0].load(Ordering::Relaxed), a[1].load(Ordering::Relaxed)];
    BounceStats {
        leases: l(&COUNTERS.leases),
        waited: l(&COUNTERS.waited),
        wait_ns: l(&COUNTERS.wait_ns),
        fallbacks: l(&COUNTERS.fallbacks),
        pools_created: COUNTERS.pools_created.load(Ordering::Relaxed),
        leased_bytes: COUNTERS.leased_bytes.load(Ordering::Relaxed),
        peak_leased_bytes: COUNTERS.peak_leased_bytes.load(Ordering::Relaxed),
        allocated_bytes: COUNTERS.allocated_bytes.load(Ordering::Relaxed),
        fallback_live_bytes: COUNTERS.fallback_live_bytes.load(Ordering::Relaxed),
        peak_fallback_bytes: COUNTERS.peak_fallback_bytes.load(Ordering::Relaxed),
        cancelled_waits: COUNTERS.cancelled_waits.load(Ordering::Relaxed),
        implicit_fallbacks: l(&COUNTERS.implicit_fallbacks),
        implicit_fallback_bytes: COUNTERS.implicit_fallback_bytes.load(Ordering::Relaxed),
        pool_unavailable: COUNTERS.pool_unavailable.load(Ordering::Relaxed),
        alignment_mismatches: COUNTERS.alignment_mismatches.load(Ordering::Relaxed),
    }
}

/// Record one IMPLICIT private bounce allocation of `bytes` (the pool could not serve the
/// request). Counted always; the first one of the process is logged.
pub(crate) fn note_implicit_bounce(dir: Direction, bytes: usize) {
    static LOGGED: AtomicBool = AtomicBool::new(false);
    COUNTERS.implicit_fallbacks[dir.idx()].fetch_add(1, Ordering::Relaxed);
    COUNTERS
        .implicit_fallback_bytes
        .fetch_add(bytes as u64, Ordering::Relaxed);
    if !LOGGED.swap(true, Ordering::Relaxed) {
        warn!(
            "s3dlio: a direct:// transfer used a private per-call bounce buffer of {bytes} bytes \
             because the shared bounce pool could not serve it (pool not creatable, or an \
             alignment it cannot satisfy). The transfer proceeds (availability first) but this \
             memory is NOT covered by the pooled cap; further occurrences are only counted \
             (see direct_bounce::bounce_stats())"
        );
    }
}

/// The cap on POOLED bounce memory in bytes: `nodes x pairs x 2 x size`, shared by every
/// `direct://` request in both directions (pools are created lazily, so the memory actually held
/// is the part of this that has been used). It bounds the buffers the pool leases, nothing else:
/// the opt-in `fallback` buffers, the implicit per-call buffers made when the pool cannot serve a
/// request ([`BounceStats::implicit_fallbacks`]), aligned-PUT tail blocks, GET destinations and
/// streaming-writer staging are outside it.
#[doc(hidden)]
pub fn bounce_memory_cap() -> usize {
    let cfg = config();
    topology().nodes() * cfg.pairs_per_node * 2 * cfg.buffer_size
}

/// A request was cancelled while it waited for bounce buffers: no I/O was started.
#[derive(Debug)]
pub struct BounceCancelled;

impl std::fmt::Display for BounceCancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "direct:// transfer cancelled while waiting for a bounce buffer"
        )
    }
}

impl std::error::Error for BounceCancelled {}

/// Whether `e` (or anything in its chain) is a [`BounceCancelled`] wrapped in an `io::Error`.
pub(crate) fn is_cancelled(e: &anyhow::Error) -> bool {
    e.chain().any(|c| {
        c.downcast_ref::<BounceCancelled>().is_some()
            || c.downcast_ref::<io::Error>()
                .and_then(|io| io.get_ref())
                .is_some_and(|inner| inner.is::<BounceCancelled>())
    })
}

/// A cancel flag that is never set (callers that cannot be cancelled).
#[cfg(test)]
pub(crate) static NEVER_CANCELLED: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

// ---------------------------------------------------------------------------------------
// NUMA topology
// ---------------------------------------------------------------------------------------

/// Parse a sysfs cpulist such as `0-3,8,10-11`.
pub(crate) fn parse_cpulist(s: &str) -> Option<Vec<usize>> {
    let mut cpus = Vec::new();
    for part in s.trim().split(',').filter(|p| !p.is_empty()) {
        match part.split_once('-') {
            Some((a, b)) => {
                let (a, b): (usize, usize) = (a.trim().parse().ok()?, b.trim().parse().ok()?);
                if b < a || b > MAX_CPU_INDEX {
                    return None;
                }
                cpus.extend(a..=b);
            }
            None => {
                let cpu: usize = part.trim().parse().ok()?;
                if cpu > MAX_CPU_INDEX {
                    return None;
                }
                cpus.push(cpu);
            }
        }
    }
    Some(cpus)
}

/// Largest CPU index accepted from a sysfs cpulist: the kernel supports at most 8192 CPUs, so a
/// larger number is a corrupt file, not a machine (it would size the CPU->node table).
const MAX_CPU_INDEX: usize = 65_535;

/// CPUs of each NUMA node, and the reverse map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Topology {
    /// Kernel node ids, in order (index = pool index).
    node_ids: Vec<usize>,
    /// CPUs of each node.
    node_cpus: Vec<Vec<usize>>,
    /// `cpu_to_node[cpu]` = pool index.
    cpu_to_node: Vec<usize>,
}

impl Topology {
    pub(crate) fn single() -> Self {
        Self {
            node_ids: vec![0],
            node_cpus: vec![Vec::new()],
            cpu_to_node: Vec::new(),
        }
    }

    /// Read `root/node*/cpulist`; any problem yields a single-node topology.
    pub(crate) fn from_sysfs(root: &Path) -> Self {
        let Ok(dir) = std::fs::read_dir(root) else {
            return Self::single();
        };
        let mut nodes: Vec<(usize, Vec<usize>)> = Vec::new();
        for entry in dir.flatten() {
            let name = entry.file_name();
            let Some(id) = name
                .to_str()
                .and_then(|n| n.strip_prefix("node"))
                .and_then(|n| n.parse::<usize>().ok())
            else {
                continue;
            };
            let Some(cpus) = std::fs::read_to_string(entry.path().join("cpulist"))
                .ok()
                .and_then(|s| parse_cpulist(&s))
            else {
                return Self::single();
            };
            // A memory-only node has no CPUs: no thread runs there, so it gets no pool.
            if !cpus.is_empty() {
                nodes.push((id, cpus));
            }
        }
        if nodes.len() < 2 {
            return Self::single();
        }
        nodes.sort_by_key(|(id, _)| *id);
        let max_cpu = nodes
            .iter()
            .flat_map(|(_, c)| c)
            .copied()
            .max()
            .unwrap_or(0);
        let mut cpu_to_node = vec![0; max_cpu + 1];
        for (idx, (_, cpus)) in nodes.iter().enumerate() {
            for &c in cpus {
                cpu_to_node[c] = idx;
            }
        }
        Self {
            node_ids: nodes.iter().map(|(id, _)| *id).collect(),
            node_cpus: nodes.into_iter().map(|(_, c)| c).collect(),
            cpu_to_node,
        }
    }

    pub(crate) fn nodes(&self) -> usize {
        self.node_ids.len()
    }

    /// Pool index of `cpu` (0 when unknown).
    pub(crate) fn node_of_cpu(&self, cpu: usize) -> usize {
        self.cpu_to_node.get(cpu).copied().unwrap_or(0)
    }
}

fn current_cpu() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        // SAFETY: sched_getcpu has no preconditions.
        let cpu = unsafe { libc::sched_getcpu() };
        usize::try_from(cpu).ok()
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

/// Pin the calling thread to `cpus` (best effort).
#[cfg(target_os = "linux")]
fn pin_current_thread(cpus: &[usize]) {
    // SAFETY: cpu_set_t is plain data; the libc macros only touch it.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &c in cpus {
            if c < libc::CPU_SETSIZE as usize {
                libc::CPU_SET(c, &mut set);
            }
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

/// Ask the kernel to place the pages of `[ptr, ptr+len)` on `node` (best effort: may be refused
/// without privileges or on kernels without NUMA, in which case first touch decides).
#[cfg(target_os = "linux")]
fn mbind_to_node(ptr: *mut u8, len: usize, node: usize) {
    // PREFERRED, not BIND: when the node is short of memory the kernel falls back to another node
    // instead of failing the page fault or invoking the OOM killer.
    const MPOL_PREFERRED: libc::c_long = 1;
    let words = node / 64 + 1;
    let mut mask = vec![0 as libc::c_ulong; words];
    mask[node / 64] |= 1 << (node % 64);
    // SAFETY: a plain mbind(2) call over memory we own; failure is ignored by design.
    unsafe {
        libc::syscall(
            libc::SYS_mbind,
            ptr as libc::c_long,
            len as libc::c_long,
            MPOL_PREFERRED,
            mask.as_ptr() as libc::c_long,
            (words * 64) as libc::c_long,
            0 as libc::c_long,
        );
    }
}

fn topology() -> &'static Topology {
    static TOPO: OnceLock<Topology> = OnceLock::new();
    TOPO.get_or_init(|| {
        if cfg!(target_os = "linux") {
            Topology::from_sysfs(Path::new("/sys/devices/system/node"))
        } else {
            Topology::single()
        }
    })
}

// ---------------------------------------------------------------------------------------
// The pool
// ---------------------------------------------------------------------------------------

struct PoolState {
    /// Indices of idle buffers.
    free: Vec<usize>,
    /// FIFO of waiting requests `(ticket, buffers needed)`, shared by both directions.
    waiters: VecDeque<(u64, usize)>,
    next_ticket: u64,
}

/// A raw buffer pointer that may cross threads (exclusively owned by the leaseholder).
#[derive(Clone, Copy)]
pub(crate) struct SendPtr(pub(crate) *mut u8);
// SAFETY: a lease gives one request exclusive use of its buffers; the pointers are only used
// while the lease (or the private buffer) is alive.
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

/// The warm buffers of one NUMA node.
pub(crate) struct NodePool {
    /// `2 x pairs` buffers.
    bufs: Vec<AlignedBuf>,
    size: usize,
    align: usize,
    state: Mutex<PoolState>,
    cv: Condvar,
    busy: BusyPolicy,
}

/// Exclusive use of one or two pool buffers, acquired atomically; returned (and the bytes
/// un-counted) on drop, however the request ends: success, error, panic, or its future being
/// cancelled (the lease is owned by the blocking closure that does the I/O, so it outlives a
/// cancelled future and is dropped only after the last native use).
pub(crate) struct Lease<'a> {
    pool: &'a NodePool,
    idx: [usize; 2],
    n: usize,
}

impl Lease<'_> {
    /// The leased buffers (one or two).
    pub(crate) fn ptrs(&self) -> ([SendPtr; 2], usize) {
        let p = |i: usize| SendPtr(self.pool.bufs[self.idx[i]].as_ptr() as *mut u8);
        ([p(0), p(if self.n > 1 { 1 } else { 0 })], self.n)
    }
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        COUNTERS
            .leased_bytes
            .fetch_sub((self.n * self.pool.size) as u64, Ordering::AcqRel);
        let mut st = self.pool.lock();
        st.free.extend_from_slice(&self.idx[..self.n]);
        drop(st);
        self.pool.cv.notify_all();
    }
}

/// A private fallback buffer (opt-in `fallback` policy): counted, outside the cap.
struct FallbackBuf(AlignedBuf);

impl FallbackBuf {
    fn new(len: usize, align: usize) -> Option<Self> {
        let b = AlignedBuf::new(len, align).ok()?;
        add_gauge(
            &COUNTERS.fallback_live_bytes,
            &COUNTERS.peak_fallback_bytes,
            len as u64,
        );
        Some(Self(b))
    }
}

impl Drop for FallbackBuf {
    fn drop(&mut self) {
        COUNTERS
            .fallback_live_bytes
            .fetch_sub(self.0.len() as u64, Ordering::AcqRel);
    }
}

impl NodePool {
    fn lock(&self) -> MutexGuard<'_, PoolState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Allocate and first-touch `2 x pairs` buffers.
    fn build_buffers(
        pairs: usize,
        size: usize,
        align: usize,
        mbind_node: Option<usize>,
    ) -> Result<Vec<AlignedBuf>, AlignedAllocError> {
        let mut bufs = Vec::with_capacity(2 * pairs);
        for _ in 0..2 * pairs {
            let mut b = AlignedBuf::new(size, align)?;
            #[cfg(target_os = "linux")]
            if let Some(node) = mbind_node {
                mbind_to_node(b.as_mut_ptr(), size, node);
            }
            #[cfg(not(target_os = "linux"))]
            let _ = mbind_node;
            // First touch (on the node, because this runs in a thread pinned to it).
            // SAFETY: the buffer owns `size` bytes; after this every byte is initialised.
            unsafe {
                std::ptr::write_bytes(b.as_mut_ptr(), 0, size);
                b.assume_init();
            }
            bufs.push(b);
        }
        Ok(bufs)
    }

    /// Create the pool of pool-index `idx`: buffers placed on that node when the host has
    /// several, created on the calling thread otherwise.
    pub(crate) fn create(
        topo: &Topology,
        idx: usize,
        cfg: &BounceConfig,
        align: usize,
    ) -> Result<Self, AlignedAllocError> {
        let bufs = if topo.nodes() > 1 {
            let node_id = topo.node_ids[idx];
            let cpus = topo.node_cpus[idx].clone();
            std::thread::scope(|s| -> Result<Vec<AlignedBuf>, AlignedAllocError> {
                let fail = AlignedAllocError {
                    len: cfg.buffer_size,
                    align,
                };
                let (tx, rx) = mpsc::channel();
                spawn_helper(s, move || {
                    #[cfg(target_os = "linux")]
                    pin_current_thread(&cpus);
                    #[cfg(not(target_os = "linux"))]
                    let _ = &cpus;
                    let _ = tx.send(Self::build_buffers(
                        cfg.pairs_per_node,
                        cfg.buffer_size,
                        align,
                        Some(node_id),
                    ));
                })
                .map_err(|e| {
                    warn!("s3dlio: cannot start the bounce-pool allocation thread: {e}");
                    fail
                })?;
                rx.recv().unwrap_or(Err(fail))
            })?
        } else {
            Self::build_buffers(cfg.pairs_per_node, cfg.buffer_size, align, None)?
        };
        COUNTERS.pools_created.fetch_add(1, Ordering::Relaxed);
        COUNTERS
            .allocated_bytes
            .fetch_add((bufs.len() * cfg.buffer_size) as u64, Ordering::Relaxed);
        Ok(Self::from_buffers(bufs, cfg.buffer_size, align, cfg.busy))
    }

    pub(crate) fn from_buffers(
        bufs: Vec<AlignedBuf>,
        size: usize,
        align: usize,
        busy: BusyPolicy,
    ) -> Self {
        let n = bufs.len();
        Self {
            bufs,
            size,
            align,
            state: Mutex::new(PoolState {
                free: (0..n).collect(),
                waiters: VecDeque::new(),
                next_ticket: 0,
            }),
            cv: Condvar::new(),
            busy,
        }
    }

    /// A new pool with fresh buffers of the same shape (tests).
    #[cfg(test)]
    fn clone_for_test(&self) -> NodePool {
        let bufs = (0..self.bufs.len())
            .map(|_| AlignedBuf::new_zeroed(self.size, self.align).unwrap())
            .collect();
        NodePool::from_buffers(bufs, self.size, self.align, self.busy)
    }

    /// Bytes this pool holds (its share of the pooled memory cap).
    #[cfg(test)]
    pub(crate) fn memory_bytes(&self) -> usize {
        self.bufs.len() * self.size
    }

    /// Lease `need` (1 or 2) buffers ATOMICALLY: either all of them at once or none, so two
    /// requests can never each hold one buffer while waiting for the other (no hold-one /
    /// wait-for-one deadlock). With the `Wait` policy requests of both directions queue in one
    /// strict FIFO (a request only takes buffers when it is at the head and enough are free), so
    /// neither direction can starve the other and a large request is not overtaken forever by
    /// small ones. The wait polls `cancel` and gives up with [`BounceCancelled`] before any I/O
    /// has started. `Ok(None)`: every buffer is busy and the policy is `Fallback`.
    pub(crate) fn acquire(
        &self,
        dir: Direction,
        need: usize,
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<Option<Lease<'_>>, BounceCancelled> {
        let need = need.clamp(1, self.bufs.len());
        let started = Instant::now();
        let d = dir.idx();
        let mut st = self.lock();
        if cancel.load(Ordering::Acquire) {
            COUNTERS.cancelled_waits.fetch_add(1, Ordering::Relaxed);
            return Err(BounceCancelled);
        }
        let mut waited = false;
        if st.free.len() < need || !st.waiters.is_empty() {
            if self.busy == BusyPolicy::Fallback {
                COUNTERS.fallbacks[d].fetch_add(1, Ordering::Relaxed);
                return Ok(None);
            }
            let ticket = st.next_ticket;
            st.next_ticket += 1;
            st.waiters.push_back((ticket, need));
            waited = true;
            loop {
                if cancel.load(Ordering::Acquire) {
                    st.waiters.retain(|(t, _)| *t != ticket);
                    drop(st);
                    self.cv.notify_all(); // the next waiter may now be at the head
                    COUNTERS.cancelled_waits.fetch_add(1, Ordering::Relaxed);
                    return Err(BounceCancelled);
                }
                if st.waiters.front().map(|(t, _)| *t) == Some(ticket) && st.free.len() >= need {
                    break;
                }
                st = self
                    .cv
                    .wait_timeout(st, BOUNCE_CANCEL_POLL)
                    .unwrap_or_else(PoisonError::into_inner)
                    .0;
            }
            st.waiters.pop_front();
        }
        let mut idx = [0usize; 2];
        for slot in idx.iter_mut().take(need) {
            // `free.len() >= need` was checked above; an empty pop cannot happen.
            *slot = st.free.pop().unwrap_or(0);
        }
        let more = !st.waiters.is_empty() && !st.free.is_empty();
        drop(st);
        if more {
            self.cv.notify_all(); // the next waiter may be able to take what is left
        }
        add_gauge(
            &COUNTERS.leased_bytes,
            &COUNTERS.peak_leased_bytes,
            (need * self.size) as u64,
        );
        COUNTERS.leases[d].fetch_add(1, Ordering::Relaxed);
        if waited {
            COUNTERS.waited[d].fetch_add(1, Ordering::Relaxed);
            COUNTERS.wait_ns[d].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        Ok(Some(Lease {
            pool: self,
            idx,
            n: need,
        }))
    }
}

/// One node's pool and the state of creating it. A failed creation is NOT remembered forever:
/// it is retried after [`BOUNCE_POOL_RETRY`] (memory may have been freed meanwhile).
struct Slot {
    pool: OnceLock<NodePool>,
    last_failure: Mutex<Option<Instant>>,
}

impl Slot {
    fn new() -> Self {
        Self {
            pool: OnceLock::new(),
            last_failure: Mutex::new(None),
        }
    }

    /// The pool, creating it with `create` if absent and the last failure is old enough.
    fn get_or_create(
        &self,
        retry: std::time::Duration,
        create: impl FnOnce() -> Result<NodePool, AlignedAllocError>,
    ) -> Option<&NodePool> {
        if let Some(p) = self.pool.get() {
            return Some(p);
        }
        // Serialises creation: concurrent first users wait for one creator.
        let mut last = self
            .last_failure
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        if let Some(p) = self.pool.get() {
            return Some(p);
        }
        if last.is_some_and(|t| t.elapsed() < retry) {
            return None;
        }
        match create() {
            Ok(p) => {
                let _ = self.pool.set(p);
                self.pool.get()
            }
            Err(e) => {
                warn!(
                    "s3dlio: cannot create the direct:// bounce pool ({e}); using per-call buffers, \
                     retrying in {retry:?}"
                );
                *last = Some(Instant::now());
                None
            }
        }
    }
}

/// The pool of the node the calling thread runs on, creating it on first use. `None` when it
/// could not be created (the caller then uses a private buffer).
fn pool_for_current_node() -> Option<&'static NodePool> {
    #[cfg(test)]
    if test_seam::pool_unavailable() {
        return None; // injected: as if the node's pool could not be created
    }
    static POOLS: OnceLock<Vec<Slot>> = OnceLock::new();
    let topo = topology();
    let pools = POOLS.get_or_init(|| (0..topo.nodes()).map(|_| Slot::new()).collect());
    let idx = current_cpu().map_or(0, |c| topo.node_of_cpu(c));
    let idx = idx.min(pools.len() - 1);
    let cfg = config();
    pools[idx].get_or_create(BOUNCE_POOL_RETRY, || {
        let align = DEFAULT_PAGE_SIZE.max(crate::file_store_direct::system_page_size());
        NodePool::create(topo, idx, cfg, align)
    })
}

/// Test seam: make the CALLING THREAD see no usable pool, as after a failed pool creation.
#[cfg(test)]
pub(crate) mod test_seam {
    use std::cell::Cell;

    thread_local! {
        static NO_POOL: Cell<bool> = const { Cell::new(false) };
    }

    pub(crate) fn pool_unavailable() -> bool {
        NO_POOL.with(Cell::get)
    }

    /// While the guard lives, pool lookups on this thread find no pool.
    pub(crate) struct NoPool(());

    impl NoPool {
        pub(crate) fn inject() -> NoPool {
            NO_POOL.with(|c| c.set(true));
            NoPool(())
        }
    }

    impl Drop for NoPool {
        fn drop(&mut self) {
            NO_POOL.with(|c| c.set(false));
        }
    }
}

// ---------------------------------------------------------------------------------------
// Ping-pong engines
// ---------------------------------------------------------------------------------------

/// One I/O handed to the helper thread.
pub(crate) struct Job {
    pub(crate) buf: SendPtr,
    pub(crate) len: usize,
    pub(crate) offset: u64,
}

/// The helper thread of one transfer: runs `op` for each submitted job.
struct Worker {
    jobs: mpsc::Sender<Job>,
    done: mpsc::Receiver<io::Result<usize>>,
}

impl Worker {
    fn submit(&self, job: Job) -> io::Result<()> {
        self.jobs
            .send(job)
            .map_err(|_| io::Error::other("bounce I/O thread is gone"))
    }
    fn wait(&self) -> io::Result<usize> {
        self.done
            .recv()
            .unwrap_or_else(|_| Err(io::Error::other("bounce I/O thread is gone")))
    }
}

/// Run `body` with a helper thread executing `op` for every submitted job.
fn with_worker<R>(
    op: &(dyn Fn(&Job) -> io::Result<usize> + Sync),
    body: impl FnOnce(&Worker) -> R,
) -> io::Result<R> {
    std::thread::scope(|s| {
        let (jobs, job_rx) = mpsc::channel::<Job>();
        let (done_tx, done) = mpsc::channel::<io::Result<usize>>();
        spawn_helper(s, move || {
            for job in job_rx {
                if done_tx.send(op(&job)).is_err() {
                    break;
                }
            }
        })?;
        let worker = Worker { jobs, done };
        let out = body(&worker);
        drop(worker); // closes the job channel: the helper exits
        Ok(out)
    })
}

// Test hook: make helper-thread creation fail (per thread, so parallel tests cannot interfere).
#[cfg(test)]
thread_local! {
    static FAIL_SPAWN: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Spawn a scoped helper thread; failure (EAGAIN, a thread limit) is an `io::Error`, never a
/// panic like `Scope::spawn`.
fn spawn_helper<'scope, 'env, F>(
    scope: &'scope std::thread::Scope<'scope, 'env>,
    f: F,
) -> io::Result<()>
where
    F: FnOnce() + Send + 'scope,
{
    #[cfg(test)]
    if FAIL_SPAWN.with(|f| f.get()) {
        return Err(io::Error::other("injected thread spawn failure"));
    }
    std::thread::Builder::new()
        .name("s3dlio-bounce-io".into())
        .spawn_scoped(scope, f)
        .map(|_| ())
}

/// One chunk of a PUT through a bounce buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PutChunk {
    /// Offset in the payload and in the file.
    pub(crate) offset: usize,
    /// Payload bytes of this chunk.
    pub(crate) data_len: usize,
    /// Bytes written (`data_len` rounded up to the alignment; the tail is zero-filled).
    pub(crate) io_len: usize,
}

/// Copy `data[chunk]` into `buf` and zero-fill its padding.
///
/// # Safety
/// `buf` must have room for `chunk.io_len` bytes and not be used by another thread.
unsafe fn fill_chunk(buf: SendPtr, data: &[u8], c: &PutChunk) {
    std::ptr::copy_nonoverlapping(data.as_ptr().add(c.offset), buf.0, c.data_len);
    std::ptr::write_bytes(buf.0.add(c.data_len), 0, c.io_len - c.data_len);
}

/// PUT `data` chunk by chunk through `bufs` (one buffer: serial; two: ping-pong, the copy of
/// chunk `k+1` overlapping the write of chunk `k`). `write_at(offset, bytes)` performs a write;
/// `on_copy(k)` is called just before chunk `k` is copied (test hook).
pub(crate) fn pingpong_put(
    chunks: &[PutChunk],
    data: &[u8],
    bufs: &[SendPtr],
    write_at: &(dyn Fn(u64, &[u8]) -> io::Result<()> + Sync),
    on_copy: &(dyn Fn(usize) + Sync),
) -> io::Result<()> {
    if chunks.is_empty() {
        return Ok(());
    }
    if bufs.len() < 2 || chunks.len() < 2 {
        for (k, c) in chunks.iter().enumerate() {
            on_copy(k);
            // SAFETY: bufs[0] is exclusively ours and holds at least `io_len` bytes.
            unsafe { fill_chunk(bufs[0], data, c) };
            let out = unsafe { std::slice::from_raw_parts(bufs[0].0 as *const u8, c.io_len) };
            write_at(c.offset as u64, out)?;
        }
        return Ok(());
    }
    let op = |job: &Job| -> io::Result<usize> {
        // SAFETY: the job's buffer was filled before submission and is not touched by the
        // submitting thread until the job's completion is received.
        let out = unsafe { std::slice::from_raw_parts(job.buf.0 as *const u8, job.len) };
        write_at(job.offset, out).map(|()| job.len)
    };
    with_worker(&op, |w| -> io::Result<()> {
        on_copy(0);
        // SAFETY: nothing else uses bufs[0] yet.
        unsafe { fill_chunk(bufs[0], data, &chunks[0]) };
        let mut result = Ok(());
        for (k, c) in chunks.iter().enumerate() {
            w.submit(Job {
                buf: bufs[k % 2],
                len: c.io_len,
                offset: c.offset as u64,
            })?;
            if let Some(next) = chunks.get(k + 1) {
                on_copy(k + 1);
                // SAFETY: bufs[(k+1)%2] was last used by chunk k-1, whose completion was
                // received at the end of the previous iteration.
                unsafe { fill_chunk(bufs[(k + 1) % 2], data, next) };
            }
            if let Err(e) = w.wait() {
                result = Err(e);
                break;
            }
        }
        result
    })?
}

/// GET `file_size` bytes chunk by chunk through `bufs` into `dest` (one buffer: serial; two:
/// ping-pong, the read of chunk `k+1` overlapping the copy-out of chunk `k`). `read_at(buf, len,
/// offset)` reads into `buf` and returns the bytes read. Every read length and offset is a
/// multiple of `align` (the last read may come back short at EOF).
///
/// # Safety
/// `dest` must be valid for `file_size` bytes of writes and not used by anything else meanwhile
/// (it may have any alignment).
pub(crate) unsafe fn pingpong_get(
    file_size: usize,
    align: usize,
    buf_size: usize,
    dest: SendPtr,
    bufs: &[SendPtr],
    read_at: &(dyn Fn(SendPtr, usize, u64) -> io::Result<usize> + Sync),
    on_copy: &(dyn Fn(usize) + Sync),
) -> io::Result<()> {
    let align = align.max(1);
    let size = (buf_size / align * align).max(align);
    let nchunks = file_size.div_ceil(size);
    let take = |k: usize| size.min(file_size - k * size);
    let want = |k: usize| take(k).div_ceil(align) * align;
    let short = |k: usize, got: usize| {
        io::Error::new(
            io::ErrorKind::UnexpectedEof,
            format!(
                "direct read of chunk {k} returned {got} bytes, expected {} (file changed size?)",
                take(k)
            ),
        )
    };
    if nchunks == 0 {
        return Ok(());
    }
    if bufs.len() < 2 || nchunks < 2 {
        for k in 0..nchunks {
            let got = read_at(bufs[0], want(k), (k * size) as u64)?;
            if got < take(k) {
                return Err(short(k, got));
            }
            on_copy(k);
            std::ptr::copy_nonoverlapping(bufs[0].0, dest.0.add(k * size), take(k));
        }
        return Ok(());
    }
    let op = |job: &Job| read_at(job.buf, job.len, job.offset);
    with_worker(&op, |w| -> io::Result<()> {
        w.submit(Job {
            buf: bufs[0],
            len: want(0),
            offset: 0,
        })?;
        for k in 0..nchunks {
            let got = w.wait()?;
            if got < take(k) {
                return Err(short(k, got));
            }
            if k + 1 < nchunks {
                // bufs[(k+1)%2] was copied out during iteration k-1: free to be read into.
                w.submit(Job {
                    buf: bufs[(k + 1) % 2],
                    len: want(k + 1),
                    offset: ((k + 1) * size) as u64,
                })?;
            }
            on_copy(k);
            std::ptr::copy_nonoverlapping(bufs[k % 2].0, dest.0.add(k * size), take(k));
        }
        Ok(())
    })?
}

// ---------------------------------------------------------------------------------------
// Entry points used by file_store_direct
// ---------------------------------------------------------------------------------------

/// Where a bounce transfer's buffers come from.
enum Source<'a> {
    Leased(Lease<'a>),
    Private(FallbackBuf),
}

/// Lease the buffers a transfer of `bytes` needs from the current node's pool (two for a
/// transfer of more than one buffer, so the copy and the I/O can overlap; one otherwise), or a
/// private one-buffer fallback when the pool is busy with the `Fallback` policy.
/// `Ok(None)`: no pool is usable (use a per-call buffer).
fn source<'a>(
    dir: Direction,
    bytes: usize,
    align: usize,
    cancel: &AtomicBool,
) -> Result<Option<Source<'a>>, BounceCancelled> {
    let Some(pool) = pool_for_current_node() else {
        COUNTERS.pool_unavailable.fetch_add(1, Ordering::Relaxed);
        return Ok(None);
    };
    if !pool.align.is_multiple_of(align.max(1)) {
        COUNTERS
            .alignment_mismatches
            .fetch_add(1, Ordering::Relaxed);
        return Ok(None);
    }
    let need = if bytes > pool.size { 2 } else { 1 };
    Ok(match pool.acquire(dir, need, cancel)? {
        Some(lease) => Some(Source::Leased(lease)),
        None => {
            let fallback = FallbackBuf::new(pool.size, pool.align).map(Source::Private);
            if fallback.is_none() {
                COUNTERS.pool_unavailable.fetch_add(1, Ordering::Relaxed);
            }
            fallback
        }
    })
}

fn cancelled_io(e: BounceCancelled) -> io::Error {
    io::Error::new(io::ErrorKind::Interrupted, e)
}

/// PUT `chunks` of `data` through the shared pool. Returns `Ok(false)` when the pool cannot be
/// used (the caller falls back to a per-call buffer). Every chunk is at most one bounce buffer.
/// The lease is held (and dropped) here, inside the caller's blocking closure, so it outlives a
/// cancelled future and is returned only after the last write; `cancel` only abandons the WAIT
/// for buffers. `after_lease` runs once the buffers are leased and before the first write (it
/// opens/truncates the target), so a request cancelled while waiting never touches the object.
pub(crate) fn pooled_put(
    chunks: &[PutChunk],
    data: &[u8],
    align: usize,
    cancel: &AtomicBool,
    after_lease: &(dyn Fn() -> io::Result<()> + Sync),
    write_at: &(dyn Fn(u64, &[u8]) -> io::Result<()> + Sync),
) -> io::Result<bool> {
    let total: usize = chunks.iter().map(|c| c.data_len).sum();
    let Some(src) = source(Direction::Put, total, align, cancel).map_err(cancelled_io)? else {
        return Ok(false);
    };
    // The buffers are ours: only now create/truncate the target (see `write_all_direct_with`).
    after_lease()?;
    let arr;
    let one;
    let bufs: &[SendPtr] = match &src {
        Source::Leased(lease) => {
            let (p, n) = lease.ptrs();
            arr = p;
            &arr[..n]
        }
        Source::Private(b) => {
            one = [SendPtr(b.0.as_ptr() as *mut u8)];
            &one
        }
    };
    pingpong_put(chunks, data, bufs, write_at, &|_| {})?;
    Ok(true)
}

/// GET `file_size` bytes into `dest` through the shared pool. `Ok(false)`: the pool cannot be
/// used (the caller reads directly).
///
/// # Safety
/// `dest` must be valid for `file_size` bytes of writes (any alignment).
pub(crate) unsafe fn pooled_get(
    file_size: usize,
    align: usize,
    dest: SendPtr,
    cancel: &AtomicBool,
    read_at: &(dyn Fn(SendPtr, usize, u64) -> io::Result<usize> + Sync),
) -> io::Result<bool> {
    let Some(src) = source(Direction::Get, file_size, align, cancel).map_err(cancelled_io)? else {
        return Ok(false);
    };
    let arr;
    let one;
    let (bufs, size): (&[SendPtr], usize) = match &src {
        Source::Leased(lease) => {
            let (p, n) = lease.ptrs();
            arr = p;
            (&arr[..n], lease.pool.size)
        }
        Source::Private(b) => {
            one = [SendPtr(b.0.as_ptr() as *mut u8)];
            (&one, b.0.len())
        }
    };
    pingpong_get(file_size, align, size, dest, bufs, read_at, &|_| {})?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::Arc;
    use std::time::Duration;

    const A: usize = 4096;

    fn pattern(len: usize) -> Vec<u8> {
        (0..len)
            .map(|i| (i.wrapping_mul(131) ^ (i >> 8).wrapping_mul(7)) as u8)
            .collect()
    }

    fn chunks_for(len: usize, size: usize) -> Vec<PutChunk> {
        let mut v = Vec::new();
        let mut off = 0;
        while off < len {
            let d = (len - off).min(size);
            v.push(PutChunk {
                offset: off,
                data_len: d,
                io_len: d.div_ceil(A) * A,
            });
            off += d;
        }
        v
    }

    fn bufs(n: usize, size: usize) -> (Vec<AlignedBuf>, Vec<SendPtr>) {
        let b: Vec<AlignedBuf> = (0..n)
            .map(|_| AlignedBuf::new_zeroed(size, A).unwrap())
            .collect();
        let p = b.iter().map(|x| SendPtr(x.as_ptr() as *mut u8)).collect();
        (b, p)
    }

    // ---- topology / parsing ----

    #[test]
    fn cpulists_parse() {
        assert_eq!(
            parse_cpulist("0-3,8,10-11\n"),
            Some(vec![0, 1, 2, 3, 8, 10, 11])
        );
        assert_eq!(parse_cpulist(""), Some(vec![]));
        assert_eq!(parse_cpulist("3-1"), None);
        assert_eq!(parse_cpulist("a"), None);
    }

    #[test]
    fn topology_from_a_fake_two_node_sysfs() {
        let d = tempfile::tempdir().unwrap();
        for (n, cpus) in [(0, "0-3"), (1, "4-7")] {
            let nd = d.path().join(format!("node{n}"));
            std::fs::create_dir(&nd).unwrap();
            std::fs::write(nd.join("cpulist"), cpus).unwrap();
        }
        std::fs::create_dir(d.path().join("power")).unwrap(); // not a node
        let t = Topology::from_sysfs(d.path());
        assert_eq!(t.nodes(), 2);
        assert_eq!(t.node_of_cpu(0), 0);
        assert_eq!(t.node_of_cpu(5), 1);
        assert_eq!(t.node_of_cpu(99), 0, "unknown CPU");
    }

    #[test]
    fn single_node_hosts_and_broken_sysfs_get_one_pool() {
        assert_eq!(Topology::from_sysfs(Path::new("/nonexistent")).nodes(), 1);
        let d = tempfile::tempdir().unwrap();
        let nd = d.path().join("node0");
        std::fs::create_dir(&nd).unwrap();
        std::fs::write(nd.join("cpulist"), "0-15").unwrap();
        assert_eq!(Topology::from_sysfs(d.path()).nodes(), 1);
        // a node with an unreadable cpulist: give up on NUMA
        let nd1 = d.path().join("node1");
        std::fs::create_dir(&nd1).unwrap();
        assert_eq!(Topology::from_sysfs(d.path()).nodes(), 1);
    }

    #[test]
    fn policy_and_mode_parsing() {
        assert_eq!(parse_busy_policy(None), BusyPolicy::Wait);
        assert_eq!(parse_busy_policy(Some("Fallback")), BusyPolicy::Fallback);
        assert_eq!(parse_busy_policy(Some("garbage")), BusyPolicy::Wait);
        assert_eq!(parse_get_strategy(None), GetStrategy::Direct);
        assert_eq!(parse_get_strategy(Some("POOLED")), GetStrategy::Pooled);
    }

    // ---- ping-pong correctness ----

    #[test]
    fn put_pingpong_writes_every_chunk_in_order_and_pads_the_tail() {
        for len in [1, A, 3 * A + 5, 8 * A, 8 * A + 1, 40 * A + 17] {
            let data = pattern(len);
            let chunks = chunks_for(len, 4 * A);
            let (_keep, p) = bufs(2, 4 * A);
            let out = Mutex::new(vec![
                0xEEu8;
                chunks.last().map_or(0, |c| c.offset + c.io_len)
            ]);
            let write = |off: u64, b: &[u8]| {
                out.lock().unwrap()[off as usize..off as usize + b.len()].copy_from_slice(b);
                Ok(())
            };
            pingpong_put(&chunks, &data, &p, &write, &|_| {}).unwrap();
            let out = out.into_inner().unwrap();
            assert_eq!(&out[..len], data.as_slice(), "len {len}");
            assert!(
                out[len..].iter().all(|&b| b == 0),
                "padding of len {len} not zero"
            );
        }
    }

    /// The structural proof of overlap: write k blocks until chunk k+1 has STARTED copying.
    /// A serial implementation would deadlock here and hit the timeout.
    #[test]
    fn put_overlaps_the_copy_of_the_next_chunk_with_the_write_of_the_current_one() {
        let len = 6 * 4 * A;
        let data = pattern(len);
        let chunks = chunks_for(len, 4 * A);
        let (_keep, p) = bufs(2, 4 * A);
        let started = Arc::new((Mutex::new(vec![false; chunks.len()]), Condvar::new()));
        let s2 = started.clone();
        let on_copy = move |k: usize| {
            let (m, cv) = &*s2;
            m.lock().unwrap()[k] = true;
            cv.notify_all();
        };
        let s3 = started.clone();
        let n = chunks.len();
        let write = move |off: u64, _b: &[u8]| {
            let k = off as usize / (4 * A);
            if k + 1 < n {
                let (m, cv) = &*s3;
                let (g, t) = cv
                    .wait_timeout_while(m.lock().unwrap(), Duration::from_secs(10), |v| !v[k + 1])
                    .unwrap();
                drop(g);
                if t.timed_out() {
                    return Err(io::Error::other(
                        "copy of the next chunk never overlapped the write",
                    ));
                }
            }
            Ok(())
        };
        pingpong_put(&chunks, &data, &p, &write, &on_copy).unwrap();
    }

    #[test]
    fn put_write_error_stops_the_transfer_and_is_reported() {
        let len = 10 * 4 * A;
        let data = pattern(len);
        let chunks = chunks_for(len, 4 * A);
        let (_keep, p) = bufs(2, 4 * A);
        let calls = AtomicUsize::new(0);
        let write = |_o: u64, _b: &[u8]| {
            if calls.fetch_add(1, Ordering::SeqCst) == 2 {
                Err(io::Error::other("disk full"))
            } else {
                Ok(())
            }
        };
        let e = pingpong_put(&chunks, &data, &p, &write, &|_| {}).unwrap_err();
        assert!(e.to_string().contains("disk full"));
        assert_eq!(
            calls.load(Ordering::SeqCst),
            3,
            "no write after the failed one"
        );
    }

    #[test]
    fn put_with_one_buffer_is_serial_and_correct() {
        let len = 9 * A + 3;
        let data = pattern(len);
        let chunks = chunks_for(len, 2 * A);
        let (_keep, p) = bufs(1, 2 * A);
        let out = Mutex::new(vec![0u8; 12 * A]);
        let write = |off: u64, b: &[u8]| {
            out.lock().unwrap()[off as usize..off as usize + b.len()].copy_from_slice(b);
            Ok(())
        };
        pingpong_put(&chunks, &data, &p, &write, &|_| {}).unwrap();
        assert_eq!(&out.lock().unwrap()[..len], data.as_slice());
    }

    fn fake_read(file: &[u8]) -> impl Fn(SendPtr, usize, u64) -> io::Result<usize> + Sync + '_ {
        move |buf, len, off| {
            let off = off as usize;
            if off >= file.len() {
                return Ok(0);
            }
            let n = len.min(file.len() - off);
            // SAFETY: the engine gives `len` bytes of exclusive buffer.
            unsafe { std::ptr::copy_nonoverlapping(file.as_ptr().add(off), buf.0, n) };
            Ok(n)
        }
    }

    #[test]
    fn get_pingpong_fills_any_destination_alignment() {
        for len in [1, A, 3 * A + 5, 8 * A, 8 * A + 1, 40 * A + 17] {
            let file = pattern(len);
            for dest_off in [0usize, 3, 16, A] {
                let (_keep, p) = bufs(2, 4 * A);
                let mut dest = vec![0xAAu8; len + dest_off + 8];
                let read = fake_read(&file);
                // SAFETY: dest has room for len bytes at dest_off.
                unsafe {
                    pingpong_get(
                        len,
                        A,
                        4 * A,
                        SendPtr(dest.as_mut_ptr().add(dest_off)),
                        &p,
                        &read,
                        &|_| {},
                    )
                }
                .unwrap();
                assert_eq!(
                    &dest[dest_off..dest_off + len],
                    file.as_slice(),
                    "len {len} off {dest_off}"
                );
                assert!(
                    dest[..dest_off].iter().all(|&b| b == 0xAA),
                    "wrote before dest"
                );
                assert!(
                    dest[dest_off + len..].iter().all(|&b| b == 0xAA),
                    "wrote past dest"
                );
            }
        }
    }

    /// The read of chunk k+1 must be in flight while chunk k is copied out: the read of chunk
    /// k+1 blocks until the copy-out of chunk k has started.
    #[test]
    fn get_overlaps_the_read_of_the_next_chunk_with_the_copy_out_of_the_current_one() {
        let len = 6 * 4 * A;
        let file = pattern(len);
        let (_keep, p) = bufs(2, 4 * A);
        let copying = Arc::new((Mutex::new(vec![false; 6]), Condvar::new()));
        let c2 = copying.clone();
        let on_copy = move |k: usize| {
            let (m, cv) = &*c2;
            m.lock().unwrap()[k] = true;
            cv.notify_all();
        };
        let c3 = copying.clone();
        let inner = fake_read(&file);
        let read = move |buf: SendPtr, l: usize, off: u64| {
            let k = off as usize / (4 * A);
            if k >= 1 {
                let (m, cv) = &*c3;
                let (g, t) = cv
                    .wait_timeout_while(m.lock().unwrap(), Duration::from_secs(10), |v| !v[k - 1])
                    .unwrap();
                drop(g);
                if t.timed_out() {
                    return Err(io::Error::other(
                        "read of the next chunk never overlapped the copy-out",
                    ));
                }
            }
            inner(buf, l, off)
        };
        let mut dest = vec![0u8; len];
        unsafe {
            pingpong_get(
                len,
                A,
                4 * A,
                SendPtr(dest.as_mut_ptr()),
                &p,
                &read,
                &on_copy,
            )
        }
        .unwrap();
        assert_eq!(dest, file);
    }

    #[test]
    fn get_of_a_file_that_shrank_is_an_error() {
        let file = pattern(5 * A);
        let (_keep, p) = bufs(2, 2 * A);
        let mut dest = vec![0u8; 12 * A];
        let read = fake_read(&file);
        let e = unsafe {
            pingpong_get(
                12 * A,
                A,
                2 * A,
                SendPtr(dest.as_mut_ptr()),
                &p,
                &read,
                &|_| {},
            )
        }
        .unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::UnexpectedEof);
    }

    // ---- the pool: fairness, cap, busy policy, atomic acquisition, cancellation ----

    fn pool(pairs: usize, busy: BusyPolicy) -> Arc<NodePool> {
        let bufs = (0..2 * pairs)
            .map(|_| AlignedBuf::new_zeroed(4 * A, A).unwrap())
            .collect();
        Arc::new(NodePool::from_buffers(bufs, 4 * A, A, busy))
    }

    /// Wait for `cond` for at most 10 s - a regression fails the test instead of hanging CI.
    fn wait_until(what: &str, cond: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !cond() {
            assert!(Instant::now() < deadline, "timed out waiting for: {what}");
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    fn lease(p: &NodePool, dir: Direction, need: usize) -> Lease<'_> {
        p.acquire(dir, need, &NEVER_CANCELLED)
            .expect("not cancelled")
            .expect("wait policy always leases")
    }

    #[test]
    fn pool_memory_is_exactly_pairs_times_two_times_size() {
        assert_eq!(pool(3, BusyPolicy::Wait).memory_bytes(), 3 * 2 * 4 * A);
    }

    #[test]
    fn never_more_buffers_leased_than_the_pool_has() {
        let p = pool(2, BusyPolicy::Wait); // 4 buffers
        let live = Arc::new(AtomicUsize::new(0));
        let max = Arc::new(AtomicUsize::new(0));
        let hs: Vec<_> = (0..12)
            .map(|i| {
                let (p, live, max) = (p.clone(), live.clone(), max.clone());
                std::thread::spawn(move || {
                    let dir = if i % 2 == 0 {
                        Direction::Put
                    } else {
                        Direction::Get
                    };
                    let need = 1 + i % 2;
                    for _ in 0..20 {
                        let l = lease(&p, dir, need);
                        let now = live.fetch_add(need, Ordering::SeqCst) + need;
                        max.fetch_max(now, Ordering::SeqCst);
                        std::thread::sleep(Duration::from_micros(200));
                        live.fetch_sub(need, Ordering::SeqCst);
                        drop(l);
                    }
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
        assert!(
            max.load(Ordering::SeqCst) <= 4,
            "{} buffers in use",
            max.load(Ordering::SeqCst)
        );
    }

    /// Strict FIFO across directions: with every buffer held, a PUT and then a GET queue up;
    /// they get the buffers in that order.
    #[test]
    fn waiters_are_served_in_arrival_order_across_directions() {
        let p = pool(1, BusyPolicy::Wait);
        let held = lease(&p, Direction::Put, 2);
        let order = Arc::new(Mutex::new(Vec::new()));
        let mut hs = Vec::new();
        for (i, dir) in [
            Direction::Put,
            Direction::Get,
            Direction::Put,
            Direction::Get,
        ]
        .into_iter()
        .enumerate()
        {
            let (p2, order) = (p.clone(), order.clone());
            hs.push(std::thread::spawn(move || {
                let l = lease(&p2, dir, 2);
                order.lock().unwrap().push(i);
                drop(l);
            }));
            // let thread i enqueue before thread i+1 starts
            wait_until("the waiter to enqueue", || p.lock().waiters.len() > i);
        }
        drop(held);
        for h in hs {
            h.join().unwrap();
        }
        assert_eq!(*order.lock().unwrap(), vec![0, 1, 2, 3]);
    }

    /// A request for both buffers is not overtaken forever by single-buffer requests: with one
    /// buffer free and a two-buffer request at the head, a later single-buffer request waits.
    #[test]
    fn a_large_request_is_not_overtaken_by_later_small_ones() {
        let p = pool(1, BusyPolicy::Wait); // 2 buffers
        let one = lease(&p, Direction::Get, 1); // 1 buffer free
        let order = Arc::new(Mutex::new(Vec::new()));
        let (p2, o2) = (p.clone(), order.clone());
        let big = std::thread::spawn(move || {
            let l = lease(&p2, Direction::Put, 2);
            o2.lock().unwrap().push("big");
            drop(l);
        });
        wait_until("the first waiter", || !p.lock().waiters.is_empty());
        let (p3, o3) = (p.clone(), order.clone());
        let small = std::thread::spawn(move || {
            let l = lease(&p3, Direction::Get, 1);
            o3.lock().unwrap().push("small");
            drop(l);
        });
        wait_until("the second waiter", || p.lock().waiters.len() >= 2);
        std::thread::sleep(Duration::from_millis(60));
        assert!(
            order.lock().unwrap().is_empty(),
            "the small request overtook the head of the queue"
        );
        drop(one);
        big.join().unwrap();
        small.join().unwrap();
        assert_eq!(*order.lock().unwrap(), vec!["big", "small"]);
    }

    /// Atomic acquisition: requests for one AND for two buffers, many threads, one pair - the
    /// classic hold-one/wait-for-one deadlock cannot happen.
    #[test]
    fn mixed_one_and_two_buffer_requests_never_deadlock() {
        let p = pool(1, BusyPolicy::Wait);
        let done = Arc::new(AtomicBool::new(false));
        let (d2, p2) = (done.clone(), p.clone());
        let t = std::thread::spawn(move || {
            let hs: Vec<_> = (0..16)
                .map(|i| {
                    let p = p2.clone();
                    std::thread::spawn(move || {
                        for _ in 0..25 {
                            let dir = if i % 3 == 0 {
                                Direction::Get
                            } else {
                                Direction::Put
                            };
                            drop(lease(&p, dir, 1 + i % 2));
                        }
                    })
                })
                .collect();
            for h in hs {
                h.join().unwrap();
            }
            d2.store(true, Ordering::SeqCst);
        });
        let deadline = Instant::now() + Duration::from_secs(30);
        while !done.load(Ordering::SeqCst) {
            assert!(Instant::now() < deadline, "mixed traffic deadlocked");
            std::thread::sleep(Duration::from_millis(5));
        }
        t.join().unwrap();
    }

    /// A waiting request whose cancel flag is set gives up (no I/O started), is removed from the
    /// queue, and does not block the requests behind it.
    #[test]
    fn a_cancelled_waiter_abandons_the_wait_and_unblocks_the_queue() {
        let p = pool(1, BusyPolicy::Wait);
        let held = lease(&p, Direction::Put, 2);
        let cancel = Arc::new(AtomicBool::new(false));
        let (p2, c2) = (p.clone(), cancel.clone());
        let victim = std::thread::spawn(move || p2.acquire(Direction::Get, 2, &c2).map(|_| ()));
        wait_until("the first waiter", || !p.lock().waiters.is_empty());
        let (p3, got) = (p.clone(), Arc::new(AtomicBool::new(false)));
        let g2 = got.clone();
        let behind = std::thread::spawn(move || {
            drop(lease(&p3, Direction::Put, 2));
            g2.store(true, Ordering::SeqCst);
        });
        wait_until("the second waiter", || p.lock().waiters.len() >= 2);
        let before = bounce_stats().cancelled_waits;
        cancel.store(true, Ordering::Release);
        let r = victim.join().unwrap();
        assert!(r.is_err(), "the cancelled waiter must not get a lease");
        assert!(bounce_stats().cancelled_waits > before);
        assert_eq!(
            p.lock().waiters.len(),
            1,
            "the cancelled ticket must leave the queue"
        );
        drop(held);
        behind.join().unwrap();
        assert!(got.load(Ordering::SeqCst));
    }

    #[test]
    fn a_request_already_cancelled_never_starts() {
        let p = pool(1, BusyPolicy::Wait);
        let cancelled = AtomicBool::new(true);
        assert!(p.acquire(Direction::Put, 2, &cancelled).is_err());
        // and it took nothing
        assert_eq!(p.lock().free.len(), 2);
    }

    /// RAII on every exit: a lease dropped while unwinding from a panic frees its buffers.
    #[test]
    fn a_panicking_holder_returns_its_buffers() {
        let p = pool(1, BusyPolicy::Wait);
        let p2 = p.clone();
        let r = std::thread::spawn(move || {
            let _l = lease(&p2, Direction::Put, 2);
            panic!("holder panics while leasing");
        })
        .join();
        assert!(r.is_err());
        assert_eq!(p.lock().free.len(), 2);
        drop(lease(&p, Direction::Get, 2));
    }

    #[test]
    fn fallback_policy_returns_none_when_every_buffer_is_busy_and_counts_it() {
        let p = pool(1, BusyPolicy::Fallback);
        let before = bounce_stats().fallbacks;
        let held = lease_f(&p, Direction::Get, 2).expect("free pool leases");
        assert!(p
            .acquire(Direction::Put, 2, &NEVER_CANCELLED)
            .unwrap()
            .is_none());
        assert!(p
            .acquire(Direction::Get, 1, &NEVER_CANCELLED)
            .unwrap()
            .is_none());
        let after = bounce_stats().fallbacks;
        assert!(after[0] > before[0] && after[1] > before[1]);
        drop(held);
        assert!(lease_f(&p, Direction::Put, 2).is_some(), "free again");
    }

    fn lease_f(p: &NodePool, dir: Direction, need: usize) -> Option<Lease<'_>> {
        p.acquire(dir, need, &NEVER_CANCELLED).unwrap()
    }

    // ---- round 5 --------------------------------------------------------------------------

    /// Helper-thread creation failure is an error, not a panic, in both engines.
    #[test]
    fn a_failed_helper_thread_spawn_is_an_error_not_a_panic() {
        let len = 4 * 4 * A;
        let data = pattern(len);
        let chunks = chunks_for(len, 4 * A);
        let (_keep, p) = bufs(2, 4 * A);
        FAIL_SPAWN.with(|f| f.set(true));
        let put = pingpong_put(&chunks, &data, &p, &|_, _| Ok(()), &|_| {});
        let file = pattern(len);
        let mut dest = vec![0u8; len];
        let read = fake_read(&file);
        let get = unsafe {
            pingpong_get(
                len,
                A,
                4 * A,
                SendPtr(dest.as_mut_ptr()),
                &p,
                &read,
                &|_| {},
            )
        };
        FAIL_SPAWN.with(|f| f.set(false));
        assert!(put.unwrap_err().to_string().contains("injected"));
        assert!(get.unwrap_err().to_string().contains("injected"));
        // and a single-chunk transfer needs no helper at all
        let one = chunks_for(A, 4 * A);
        FAIL_SPAWN.with(|f| f.set(true));
        let r = pingpong_put(&one, &data, &p, &|_, _| Ok(()), &|_| {});
        FAIL_SPAWN.with(|f| f.set(false));
        r.unwrap();
    }

    #[test]
    fn cpulist_indexes_are_capped() {
        assert_eq!(parse_cpulist("0-3"), Some(vec![0, 1, 2, 3]));
        assert_eq!(parse_cpulist("65535"), Some(vec![65_535]));
        assert_eq!(parse_cpulist("65536"), None);
        assert_eq!(parse_cpulist("0-4294967295"), None);
        assert_eq!(parse_cpulist("999999999999"), None);
    }

    /// A failed pool creation is retried after the retry interval, not remembered forever.
    #[test]
    fn a_failed_pool_creation_is_retried_later() {
        let slot = Slot::new();
        let calls = AtomicUsize::new(0);
        let fail = || -> Result<NodePool, AlignedAllocError> {
            calls.fetch_add(1, Ordering::SeqCst);
            Err(AlignedAllocError { len: 1, align: 1 })
        };
        let retry = Duration::from_millis(80);
        assert!(slot.get_or_create(retry, fail).is_none());
        assert!(
            slot.get_or_create(retry, fail).is_none(),
            "within the retry interval"
        );
        assert_eq!(calls.load(Ordering::SeqCst), 1, "not retried too early");
        std::thread::sleep(Duration::from_millis(120));
        let ok = || -> Result<NodePool, AlignedAllocError> {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(pool(1, BusyPolicy::Wait).clone_for_test())
        };
        assert!(slot.get_or_create(retry, ok).is_some());
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        // and once it exists it is never created again
        assert!(slot.get_or_create(retry, fail).is_some());
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }
}
