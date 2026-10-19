// tests/common/counting_alloc.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// A counting global allocator for the stage G zero-copy tests. A test binary includes it with
// `#[path]` and declares the `#[global_allocator]` itself:
//
//     #[path = "common/counting_alloc.rs"]
//     mod counting_alloc;
//     #[global_allocator]
//     static ALLOC: counting_alloc::CountingAlloc =
//         counting_alloc::CountingAlloc::new(common::mock_s3_server::on_mock_server_thread);
//
// While armed it counts the bytes allocated, the peak of live bytes, and the number of "big"
// allocations (at least `big_threshold` bytes) on every thread for which `exclude` is false (the
// mock server's own threads are excluded: it buffers request bodies, which is not the code under
// test). The allocator is process-wide: tests that use it must run one after another.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicIsize, AtomicUsize, Ordering};

pub struct CountingAlloc {
    exclude: fn() -> bool,
    armed: AtomicBool,
    total: AtomicUsize,
    live: AtomicIsize,
    peak: AtomicIsize,
    big_threshold: AtomicUsize,
    big_allocs: AtomicUsize,
}

/// What one measured window allocated.
pub struct Measured {
    /// Bytes allocated (growth of a realloc counts).
    pub total: usize,
    /// Peak of (bytes allocated - bytes freed) while armed.
    pub peak: usize,
    /// How many single allocations were at least `big_threshold` bytes.
    pub big_allocs: usize,
}

impl CountingAlloc {
    pub const fn new(exclude: fn() -> bool) -> Self {
        Self {
            exclude,
            armed: AtomicBool::new(false),
            total: AtomicUsize::new(0),
            live: AtomicIsize::new(0),
            peak: AtomicIsize::new(0),
            big_threshold: AtomicUsize::new(usize::MAX),
            big_allocs: AtomicUsize::new(0),
        }
    }

    /// Start a window; allocations of at least `big_threshold` bytes are counted separately.
    pub fn arm(&self, big_threshold: usize) {
        self.total.store(0, Ordering::SeqCst);
        self.live.store(0, Ordering::SeqCst);
        self.peak.store(0, Ordering::SeqCst);
        self.big_allocs.store(0, Ordering::SeqCst);
        self.big_threshold.store(big_threshold, Ordering::SeqCst);
        self.armed.store(true, Ordering::SeqCst);
    }

    pub fn disarm(&self) -> Measured {
        self.armed.store(false, Ordering::SeqCst);
        Measured {
            total: self.total.load(Ordering::SeqCst),
            peak: self.peak.load(Ordering::SeqCst).max(0) as usize,
            big_allocs: self.big_allocs.load(Ordering::SeqCst),
        }
    }

    fn counting(&self) -> bool {
        self.armed.load(Ordering::Relaxed) && !(self.exclude)()
    }

    fn add(&self, n: usize, new_allocation: bool) {
        if self.counting() {
            self.total.fetch_add(n, Ordering::Relaxed);
            let now = self.live.fetch_add(n as isize, Ordering::Relaxed) + n as isize;
            self.peak.fetch_max(now, Ordering::Relaxed);
            if new_allocation && n >= self.big_threshold.load(Ordering::Relaxed) {
                self.big_allocs.fetch_add(1, Ordering::Relaxed);
            }
        }
    }

    fn sub(&self, n: usize) {
        if self.counting() {
            self.live.fetch_sub(n as isize, Ordering::Relaxed);
        }
    }
}

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.add(layout.size(), true);
        System.alloc(layout)
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        self.add(layout.size(), true);
        System.alloc_zeroed(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.sub(layout.size());
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if new_size > layout.size() {
            // A growing realloc may move (copy) the block: count the new size as an allocation.
            self.add(
                new_size - layout.size(),
                new_size >= self.big_threshold.load(Ordering::Relaxed)
                    && layout.size() < self.big_threshold.load(Ordering::Relaxed),
            );
        } else {
            self.sub(layout.size() - new_size);
        }
        System.realloc(ptr, layout, new_size)
    }
}
