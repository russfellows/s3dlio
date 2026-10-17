// src/receive_progress.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Progress of one caller-buffer receive (plan §4.5 A1 and A7).
//
// The logical byte count of a GET is the number of DISTINCT destination
// positions filled -- not wire bytes, not retries, not "valid data" (§4.4).
// Each fixed, disjoint destination span (the whole object, or one range part,
// fixed when the read starts) keeps ONE high-water mark in a [`SpanMark`].
// The mark persists across every retry and repair of its span: a retry
// restarts its cursor at the span start and adds nothing until it passes the
// mark; a tail repair writes the suffix of the same span, resuming at the
// cursor (at or below the mark), so it is never a new, separately counted span.
//
// The counter survives errors and cancellation, so a failed read still says
// which positions were touched.  It never says the data is valid.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::SystemTime;

/// Progress of one receive, readable while it runs and after it fails or is
/// cancelled.
#[derive(Debug, Default)]
pub struct ReceiveProgress {
    /// Distinct destination positions filled (plan §4.4, A7).
    pub filled: AtomicU64,
    /// When the first byte was written, for the op-log first-byte column.
    pub first_byte: OnceLock<SystemTime>,
}

impl ReceiveProgress {
    /// A progress record with nothing filled and no first byte.
    pub fn new() -> Self {
        Self::default()
    }

    /// The distinct destination positions filled so far.
    pub fn filled(&self) -> u64 {
        self.filled.load(Ordering::Relaxed)
    }
}

/// The high-water mark of ONE fixed, disjoint destination span.
///
/// Create it once per span, when the read starts, and keep it for every retry
/// and repair of that span.  Positions are offsets WITHIN the span.  The span's
/// own task owns the mark, so recording needs no atomic read-modify-write on
/// the mark itself; only the shared `filled` total is atomic.
#[derive(Debug)]
pub struct SpanMark<'p> {
    progress: &'p ReceiveProgress,
    mark: u64,
}

impl<'p> SpanMark<'p> {
    /// A span with nothing filled yet.
    pub fn new(progress: &'p ReceiveProgress) -> Self {
        Self { progress, mark: 0 }
    }

    /// The highest position (exclusive) any attempt has filled in this span.
    pub fn mark(&self) -> u64 {
        self.mark
    }

    /// Record one copy of `len` bytes into the span at offset `start`
    /// (call it AFTER the copy).
    ///
    /// * A copy of zero bytes records nothing.
    /// * The first non-empty copy sets `first_byte`, whatever span or attempt.
    /// * If the copy's end passes the span's mark, the difference is added to
    ///   `filled` and the mark moves; otherwise nothing is added.
    ///
    /// Returns the number of positions newly counted.
    pub fn record_copy(&mut self, start: u64, len: u64) -> u64 {
        if len == 0 {
            return 0;
        }
        self.progress.first_byte.get_or_init(SystemTime::now);
        let end = start.saturating_add(len);
        if end <= self.mark {
            return 0;
        }
        let delta = end - self.mark;
        self.mark = end;
        self.progress.filled.fetch_add(delta, Ordering::Relaxed);
        delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A retry that restarts at the span start and runs past the old mark counts
    /// only the positions beyond it.
    #[test]
    fn retry_crossing_the_old_mark_counts_each_position_once() {
        let p = ReceiveProgress::new();
        let mut span = SpanMark::new(&p);
        // Attempt 1 fills 0..60 in chunks, then fails.
        assert_eq!(span.record_copy(0, 40), 40);
        assert_eq!(span.record_copy(40, 20), 20);
        assert_eq!(p.filled(), 60);
        // Attempt 2 restarts at 0: nothing new until it passes 60.
        assert_eq!(span.record_copy(0, 30), 0);
        assert_eq!(span.record_copy(30, 30), 0);
        assert_eq!(p.filled(), 60);
        // It crosses the old mark: only 60..100 is new.
        assert_eq!(span.record_copy(60, 40), 40);
        assert_eq!(p.filled(), 100);
        // A third attempt over the whole span adds nothing.
        assert_eq!(span.record_copy(0, 100), 0);
        assert_eq!(p.filled(), 100);
    }

    /// A copy straddling the mark counts only its part beyond it.
    #[test]
    fn copy_straddling_the_mark_counts_the_excess_only() {
        let p = ReceiveProgress::new();
        let mut span = SpanMark::new(&p);
        assert_eq!(span.record_copy(0, 50), 50);
        assert_eq!(span.record_copy(20, 50), 20); // 20..70 passes 50
        assert_eq!(span.mark(), 70);
        assert_eq!(p.filled(), 70);
    }

    /// Several tail repairs resume at the cursor of the SAME span: positions are
    /// counted once, however many repairs rewrite them.
    #[test]
    fn multiple_tail_repairs_count_once_per_position() {
        let p = ReceiveProgress::new();
        let mut span = SpanMark::new(&p);
        // Initial read fills 0..70 of a 100-byte span, then the stream breaks.
        assert_eq!(span.record_copy(0, 70), 70);
        // Repair 1 resumes at 50 (below the mark) and reaches 85, then breaks.
        assert_eq!(span.record_copy(50, 35), 15);
        // Repair 2 resumes at 80 (below the mark again) and finishes the span.
        assert_eq!(span.record_copy(80, 20), 15);
        // Repair 3, a redundant rewrite of the tail, adds nothing.
        assert_eq!(span.record_copy(90, 10), 0);
        assert_eq!(p.filled(), 100);
    }

    /// Disjoint spans each keep their own mark, and the totals sum.
    #[test]
    fn disjoint_spans_have_independent_marks_that_sum() {
        let p = ReceiveProgress::new();
        let mut a = SpanMark::new(&p);
        let mut b = SpanMark::new(&p);
        a.record_copy(0, 10);
        b.record_copy(0, 4);
        a.record_copy(0, 10); // a retry of span a
        b.record_copy(0, 8);
        assert_eq!(p.filled(), 18);
    }

    /// Only a non-empty copy sets `first_byte`, and only the first one does.
    #[test]
    fn first_non_empty_copy_sets_first_byte_once() {
        let p = ReceiveProgress::new();
        let mut span = SpanMark::new(&p);
        assert_eq!(span.record_copy(0, 0), 0);
        assert!(
            p.first_byte.get().is_none(),
            "an empty copy is no first byte"
        );
        span.record_copy(0, 5);
        let first = *p.first_byte.get().expect("first byte set");
        span.record_copy(5, 5);
        span.record_copy(0, 10);
        assert_eq!(*p.first_byte.get().expect("still set"), first);
    }

    /// A position past `u64::MAX` saturates instead of panicking.
    #[test]
    fn an_end_that_would_overflow_saturates_instead_of_panicking() {
        let p = ReceiveProgress::new();
        let mut span = SpanMark::new(&p);
        assert_eq!(span.record_copy(u64::MAX - 1, 10), u64::MAX);
        assert_eq!(p.filled(), u64::MAX);
    }
}
