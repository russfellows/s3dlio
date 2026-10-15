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
//!   A request leases ONE pair for its whole duration:
//!   * PUT: chunk `k` is copied into buffer A and written while chunk `k+1` is copied into buffer
//!     B; then the buffers swap. The write runs on a helper thread, the copy on the caller's.
//!   * GET: chunk `k+1` is read into buffer A by the helper thread while chunk `k` is copied out
//!     of buffer B into the destination; then they swap.
//! * **Bounded memory.** Total bounce memory is exactly `nodes x pairs x 2 x size` (default
//!   128 MiB per node) for ALL requests in both directions: that is the global cap.
//! * **Busy policy** (`S3DLIO_DIRECT_IO_BOUNCE_WHEN_BUSY`): when every pair of the node is
//!   leased, `wait` (default) queues the request in strict FIFO order shared by PUTs and GETs, so
//!   neither direction can starve the other; `fallback` instead gives the request a private
//!   buffer of one bounce size (serial copy-then-I/O, no pipelining; extra memory is bounded by
//!   one buffer per concurrent request beyond the pool). Both are counted in [`bounce_stats`].
//! * **Small objects** (bounce need at most [`DIRECT_IO_POOL_MIN_OBJECT`]) skip the pool and use
//!   a private buffer of their own size: waiting for a shared pair would cost more than the copy.
//! * The GET destination is a raw pointer, so the same code fills the buffer a GET returns (a
//!   fresh page-aligned allocation) or, later, a buffer supplied by the caller (`get_into`),
//!   which may be unaligned.
//!
//! The per-call strategies (`PutStrategy::PerCall`, `GetStrategy::Direct`) are kept selectable
//! through [`__bench_configure`] so the benchmark can compare them with the pool.

use crate::constants::{
    direct_io_bounce_size, parse_direct_io_bounce_pairs, DEFAULT_PAGE_SIZE,
    DIRECT_IO_POOL_MIN_OBJECT, ENV_DIRECT_IO_BOUNCE_PAIRS, ENV_DIRECT_IO_BOUNCE_WHEN_BUSY,
    ENV_DIRECT_IO_GET_MODE,
};
use crate::memory::{AlignedAllocError, AlignedBuf};
use std::collections::VecDeque;
use std::io;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
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
}

static COUNTERS: Counters = Counters {
    leases: [AtomicU64::new(0), AtomicU64::new(0)],
    waited: [AtomicU64::new(0), AtomicU64::new(0)],
    wait_ns: [AtomicU64::new(0), AtomicU64::new(0)],
    fallbacks: [AtomicU64::new(0), AtomicU64::new(0)],
    pools_created: AtomicU64::new(0),
};

/// Snapshot of the pool's use, per direction (`[put, get]`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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
}

/// Current pool statistics.
#[doc(hidden)]
pub fn bounce_stats() -> BounceStats {
    let l = |a: &[AtomicU64; 2]| [a[0].load(Ordering::Relaxed), a[1].load(Ordering::Relaxed)];
    BounceStats {
        leases: l(&COUNTERS.leases),
        waited: l(&COUNTERS.waited),
        wait_ns: l(&COUNTERS.wait_ns),
        fallbacks: l(&COUNTERS.fallbacks),
        pools_created: COUNTERS.pools_created.load(Ordering::Relaxed),
    }
}

/// The global cap on bounce memory in bytes: `nodes x pairs x 2 x size`, shared by every
/// `direct://` request in both directions (pools are created lazily, so the memory actually held
/// is the part of this that has been used).
#[doc(hidden)]
pub fn bounce_memory_cap() -> usize {
    let cfg = config();
    topology().nodes() * cfg.pairs_per_node * 2 * cfg.buffer_size
}

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
                if b < a || b - a > 1 << 20 {
                    return None;
                }
                cpus.extend(a..=b);
            }
            None => cpus.push(part.trim().parse().ok()?),
        }
    }
    Some(cpus)
}

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
    const MPOL_BIND: libc::c_long = 2;
    let words = node / 64 + 1;
    let mut mask = vec![0 as libc::c_ulong; words];
    mask[node / 64] |= 1 << (node % 64);
    // SAFETY: a plain mbind(2) call over memory we own; failure is ignored by design.
    unsafe {
        libc::syscall(
            libc::SYS_mbind,
            ptr as libc::c_long,
            len as libc::c_long,
            MPOL_BIND,
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
    /// Indices of idle pairs.
    free: Vec<usize>,
    /// FIFO of waiting requests (ticket numbers), shared by both directions.
    waiters: VecDeque<u64>,
    next_ticket: u64,
}

/// A raw buffer pointer that may cross threads (exclusively owned by the leaseholder).
#[derive(Clone, Copy)]
pub(crate) struct SendPtr(pub(crate) *mut u8);
// SAFETY: a lease gives one request exclusive use of the pair; the pointers are only used
// while the lease (or the private buffer) is alive.
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}

/// The warm buffers of one NUMA node.
pub(crate) struct NodePool {
    /// `2 x pairs` buffers; pair `i` is buffers `2i` and `2i+1`.
    bufs: Vec<AlignedBuf>,
    size: usize,
    align: usize,
    state: Mutex<PoolState>,
    cv: Condvar,
    busy: BusyPolicy,
}

/// Exclusive use of one buffer pair; returned to the pool on drop.
pub(crate) struct Lease<'a> {
    pool: &'a NodePool,
    pair: usize,
}

impl Lease<'_> {
    pub(crate) fn ptrs(&self) -> [SendPtr; 2] {
        let p = |i: usize| SendPtr(self.pool.bufs[2 * self.pair + i].as_ptr() as *mut u8);
        [p(0), p(1)]
    }
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        let mut st = self.pool.lock();
        st.free.push(self.pair);
        drop(st);
        self.pool.cv.notify_all();
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
            std::thread::scope(|s| {
                s.spawn(|| {
                    #[cfg(target_os = "linux")]
                    pin_current_thread(&cpus);
                    #[cfg(not(target_os = "linux"))]
                    let _ = &cpus;
                    Self::build_buffers(cfg.pairs_per_node, cfg.buffer_size, align, Some(node_id))
                })
                .join()
                .unwrap_or(Err(AlignedAllocError {
                    len: cfg.buffer_size,
                    align,
                }))
            })?
        } else {
            Self::build_buffers(cfg.pairs_per_node, cfg.buffer_size, align, None)?
        };
        COUNTERS.pools_created.fetch_add(1, Ordering::Relaxed);
        Ok(Self::from_buffers(bufs, cfg.buffer_size, align, cfg.busy))
    }

    pub(crate) fn from_buffers(
        bufs: Vec<AlignedBuf>,
        size: usize,
        align: usize,
        busy: BusyPolicy,
    ) -> Self {
        let pairs = bufs.len() / 2;
        Self {
            bufs,
            size,
            align,
            state: Mutex::new(PoolState {
                free: (0..pairs).collect(),
                waiters: VecDeque::new(),
                next_ticket: 0,
            }),
            cv: Condvar::new(),
            busy,
        }
    }

    /// Bytes this pool holds (its share of the global cap).
    #[cfg(test)]
    pub(crate) fn memory_bytes(&self) -> usize {
        self.bufs.len() * self.size
    }

    /// Lease a pair, or `None` when every pair is busy and the policy is `Fallback`.
    ///
    /// With `Wait`, requests of both directions queue in one FIFO (a request only takes a
    /// pair when it is at the head), so neither direction can starve the other.
    pub(crate) fn acquire(&self, dir: Direction) -> Option<Lease<'_>> {
        let started = Instant::now();
        let d = dir.idx();
        let mut st = self.lock();
        let mut waited = false;
        if st.free.is_empty() || !st.waiters.is_empty() {
            if self.busy == BusyPolicy::Fallback {
                COUNTERS.fallbacks[d].fetch_add(1, Ordering::Relaxed);
                return None;
            }
            let ticket = st.next_ticket;
            st.next_ticket += 1;
            st.waiters.push_back(ticket);
            waited = true;
            while !(st.waiters.front() == Some(&ticket) && !st.free.is_empty()) {
                st = self.cv.wait(st).unwrap_or_else(PoisonError::into_inner);
            }
            st.waiters.pop_front();
        }
        let Some(pair) = st.free.pop() else {
            return None; // unreachable: checked above
        };
        let more = !st.free.is_empty() && !st.waiters.is_empty();
        drop(st);
        if more {
            self.cv.notify_all(); // the next waiter can take the next free pair
        }
        COUNTERS.leases[d].fetch_add(1, Ordering::Relaxed);
        if waited {
            COUNTERS.waited[d].fetch_add(1, Ordering::Relaxed);
            COUNTERS.wait_ns[d].fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        Some(Lease { pool: self, pair })
    }
}

/// The pool of the node the calling thread runs on, creating it on first use. `None` when it
/// could not be created (the caller then uses a private buffer).
fn pool_for_current_node() -> Option<&'static NodePool> {
    static POOLS: OnceLock<Vec<OnceLock<Option<NodePool>>>> = OnceLock::new();
    let topo = topology();
    let pools = POOLS.get_or_init(|| (0..topo.nodes()).map(|_| OnceLock::new()).collect());
    let idx = current_cpu().map_or(0, |c| topo.node_of_cpu(c));
    let cfg = config();
    pools[idx.min(pools.len() - 1)]
        .get_or_init(|| {
            let align = DEFAULT_PAGE_SIZE.max(crate::file_store_direct::system_page_size());
            match NodePool::create(topo, idx, cfg, align) {
                Ok(p) => Some(p),
                Err(e) => {
                    warn!("s3dlio: cannot create the direct:// bounce pool ({e}); using per-call buffers");
                    None
                }
            }
        })
        .as_ref()
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
) -> R {
    std::thread::scope(|s| {
        let (jobs, job_rx) = mpsc::channel::<Job>();
        let (done_tx, done) = mpsc::channel::<io::Result<usize>>();
        s.spawn(move || {
            for job in job_rx {
                if done_tx.send(op(&job)).is_err() {
                    break;
                }
            }
        });
        let worker = Worker { jobs, done };
        let out = body(&worker);
        drop(worker); // closes the job channel: the helper exits
        out
    })
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
    })
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
    })
}

// ---------------------------------------------------------------------------------------
// Entry points used by file_store_direct
// ---------------------------------------------------------------------------------------

/// Where a bounce transfer's buffers come from.
enum Source<'a> {
    Pair(Lease<'a>),
    Private(AlignedBuf),
}

/// Lease a pair of the current node's pool, or make a private buffer of one bounce size when the
/// pool is busy with the `Fallback` policy. `None`: no pool is usable (use a per-call buffer).
fn source(dir: Direction, align: usize) -> Option<Source<'static>> {
    let pool = pool_for_current_node()?;
    if !pool.align.is_multiple_of(align.max(1)) {
        return None;
    }
    match pool.acquire(dir) {
        Some(lease) => Some(Source::Pair(lease)),
        None => AlignedBuf::new(pool.size, pool.align)
            .ok()
            .map(Source::Private),
    }
}

/// PUT `chunks` of `data` through the shared pool. Returns `Ok(false)` when the pool cannot be
/// used (the caller falls back to a per-call buffer). Every chunk is at most one bounce buffer.
pub(crate) fn pooled_put(
    chunks: &[PutChunk],
    data: &[u8],
    align: usize,
    write_at: &(dyn Fn(u64, &[u8]) -> io::Result<()> + Sync),
) -> io::Result<bool> {
    let Some(src) = source(Direction::Put, align) else {
        return Ok(false);
    };
    let ptrs;
    let one;
    let bufs: &[SendPtr] = match &src {
        Source::Pair(lease) => {
            ptrs = lease.ptrs();
            &ptrs
        }
        Source::Private(b) => {
            one = [SendPtr(b.as_ptr() as *mut u8)];
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
    read_at: &(dyn Fn(SendPtr, usize, u64) -> io::Result<usize> + Sync),
) -> io::Result<bool> {
    let Some(src) = source(Direction::Get, align) else {
        return Ok(false);
    };
    let ptrs;
    let one;
    let (bufs, size): (&[SendPtr], usize) = match &src {
        Source::Pair(lease) => {
            ptrs = lease.ptrs();
            (&ptrs, lease.pool.size)
        }
        Source::Private(b) => {
            one = [SendPtr(b.as_ptr() as *mut u8)];
            (&one, b.len())
        }
    };
    pingpong_get(file_size, align, size, dest, bufs, read_at, &|_| {})?;
    Ok(true)
}

/// Whether a transfer needing `bytes` of bounce space should use the shared pool.
pub(crate) fn use_pool_for(bytes: usize) -> bool {
    bytes > DIRECT_IO_POOL_MIN_OBJECT
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

    // ---- the pool: fairness, cap, busy policy ----

    fn pool(pairs: usize, busy: BusyPolicy) -> Arc<NodePool> {
        let bufs = (0..2 * pairs)
            .map(|_| AlignedBuf::new_zeroed(4 * A, A).unwrap())
            .collect();
        Arc::new(NodePool::from_buffers(bufs, 4 * A, A, busy))
    }

    #[test]
    fn pool_memory_is_exactly_pairs_times_two_times_size() {
        assert_eq!(pool(3, BusyPolicy::Wait).memory_bytes(), 3 * 2 * 4 * A);
    }

    #[test]
    fn never_more_leases_than_pairs() {
        let p = pool(2, BusyPolicy::Wait);
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
                    for _ in 0..20 {
                        let l = p.acquire(dir).expect("wait policy always leases");
                        let now = live.fetch_add(1, Ordering::SeqCst) + 1;
                        max.fetch_max(now, Ordering::SeqCst);
                        std::thread::sleep(Duration::from_micros(200));
                        live.fetch_sub(1, Ordering::SeqCst);
                        drop(l);
                    }
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
        assert!(
            max.load(Ordering::SeqCst) <= 2,
            "{} pairs in use at once",
            max.load(Ordering::SeqCst)
        );
    }

    /// Strict FIFO across directions: with one pair held, a PUT and then a GET queue up; they
    /// must get the pair in that order (and a stream of PUTs cannot overtake the waiting GET).
    #[test]
    fn waiters_are_served_in_arrival_order_across_directions() {
        let p = pool(1, BusyPolicy::Wait);
        let held = p.acquire(Direction::Put).unwrap();
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
                let l = p2.acquire(dir).unwrap();
                order.lock().unwrap().push(i);
                drop(l);
            }));
            // let thread i enqueue before thread i+1 starts
            while p.lock().waiters.len() < i + 1 {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
        drop(held);
        for h in hs {
            h.join().unwrap();
        }
        assert_eq!(*order.lock().unwrap(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn fallback_policy_returns_none_when_every_pair_is_busy_and_counts_it() {
        let p = pool(1, BusyPolicy::Fallback);
        let before = bounce_stats().fallbacks;
        let held = p.acquire(Direction::Get).unwrap();
        assert!(p.acquire(Direction::Put).is_none());
        assert!(p.acquire(Direction::Get).is_none());
        let after = bounce_stats().fallbacks;
        assert!(after[0] > before[0] && after[1] > before[1]);
        drop(held);
        assert!(p.acquire(Direction::Put).is_some(), "free again");
    }

    #[test]
    fn mixed_traffic_completes_under_one_pair_without_deadlock() {
        let p = pool(1, BusyPolicy::Wait);
        let done = Arc::new(AtomicBool::new(false));
        let d2 = done.clone();
        let p2 = p.clone();
        let t = std::thread::spawn(move || {
            let hs: Vec<_> = (0..16)
                .map(|i| {
                    let p = p2.clone();
                    std::thread::spawn(move || {
                        for _ in 0..10 {
                            let dir = if i % 3 == 0 {
                                Direction::Get
                            } else {
                                Direction::Put
                            };
                            drop(p.acquire(dir).unwrap());
                        }
                    })
                })
                .collect();
            for h in hs {
                h.join().unwrap();
            }
            d2.store(true, Ordering::SeqCst);
        });
        let deadline = Instant::now() + Duration::from_secs(20);
        while !done.load(Ordering::SeqCst) {
            assert!(Instant::now() < deadline, "mixed traffic deadlocked");
            std::thread::sleep(Duration::from_millis(5));
        }
        t.join().unwrap();
    }

    #[test]
    fn small_objects_skip_the_pool() {
        assert!(!use_pool_for(DIRECT_IO_POOL_MIN_OBJECT));
        assert!(use_pool_for(DIRECT_IO_POOL_MIN_OBJECT + 1));
    }
}
