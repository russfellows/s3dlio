// tests/common/peak_alloc.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Peak-tracking global allocator used by Phase 1 (issue #148) tests to assert
// that the zero-copy / capacity-hint fixes actually reduce peak memory
// usage during range-download assembly.
//
// stats_alloc only exposes cumulative counters (bytes_allocated,
// bytes_deallocated), which cannot distinguish a temporal peak from a
// steady-state footprint of the same size. This wrapper tracks the running
// live allocation total and the peak observed, so a test can bound the peak
// memory a code path is allowed to use.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

pub struct PeakAlloc {
    live: AtomicUsize,
    peak: AtomicUsize,
    total_allocs: AtomicUsize,
    total_reallocs: AtomicUsize,
}

impl PeakAlloc {
    pub const fn new() -> Self {
        Self {
            live: AtomicUsize::new(0),
            peak: AtomicUsize::new(0),
            total_allocs: AtomicUsize::new(0),
            total_reallocs: AtomicUsize::new(0),
        }
    }

    /// Current live-allocation total (bytes currently allocated but not
    /// deallocated through this allocator).
    pub fn live(&self) -> usize {
        self.live.load(Ordering::Relaxed)
    }

    /// Peak live-allocation seen since the last reset.
    pub fn peak(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }

    /// Total allocations counted since the last reset (excludes reallocs).
    pub fn allocations(&self) -> usize {
        self.total_allocs.load(Ordering::Relaxed)
    }

    /// Total reallocs counted since the last reset.
    pub fn reallocations(&self) -> usize {
        self.total_reallocs.load(Ordering::Relaxed)
    }

    /// Reset peak to the current live total and zero the allocation/realloc
    /// counters. Call this at the start of the region you want to measure,
    /// after any warmup has finished.
    pub fn reset(&self) {
        let current = self.live.load(Ordering::Relaxed);
        self.peak.store(current, Ordering::Relaxed);
        self.total_allocs.store(0, Ordering::Relaxed);
        self.total_reallocs.store(0, Ordering::Relaxed);
    }

    fn on_alloc(&self, size: usize) {
        self.total_allocs.fetch_add(1, Ordering::Relaxed);
        let new_live = self.live.fetch_add(size, Ordering::Relaxed) + size;
        let mut peak = self.peak.load(Ordering::Relaxed);
        while new_live > peak {
            match self.peak.compare_exchange_weak(
                peak,
                new_live,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => peak = observed,
            }
        }
    }

    fn on_dealloc(&self, size: usize) {
        self.live.fetch_sub(size, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for PeakAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            self.on_alloc(layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        self.on_dealloc(layout.size());
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new_ptr = System.realloc(ptr, layout, new_size);
        if !new_ptr.is_null() {
            self.total_reallocs.fetch_add(1, Ordering::Relaxed);
            let old_size = layout.size();
            if new_size > old_size {
                let delta = new_size - old_size;
                let new_live = self.live.fetch_add(delta, Ordering::Relaxed) + delta;
                let mut peak = self.peak.load(Ordering::Relaxed);
                while new_live > peak {
                    match self.peak.compare_exchange_weak(
                        peak,
                        new_live,
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    ) {
                        Ok(_) => break,
                        Err(observed) => peak = observed,
                    }
                }
            } else if new_size < old_size {
                self.live.fetch_sub(old_size - new_size, Ordering::Relaxed);
            }
        }
        new_ptr
    }
}
