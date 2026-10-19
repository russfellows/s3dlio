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
// A copy that starts beyond the mark would leave a hole of positions nobody
// filled; counting it would claim bytes that were never written, so it is
// refused (an error) and records nothing.
//
// The counter survives errors and cancellation, so a failed read still says
// which positions were touched.  It never says the data is valid.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::SystemTime;

/// Progress of one receive, readable while it runs and after it fails or is
/// cancelled.
///
/// The fields are private: [`SpanMark::record_copy`] is the only writer, so the
/// total can only grow by counted positions.  Readers use [`filled`] and
/// [`first_byte`].
///
/// [`filled`]: ReceiveProgress::filled
/// [`first_byte`]: ReceiveProgress::first_byte
#[derive(Debug, Default)]
pub struct ReceiveProgress {
    /// Distinct destination positions filled (plan §4.4, A7).
    filled: AtomicU64,
    /// When the first byte was written, for the op-log first-byte column.
    first_byte: OnceLock<SystemTime>,
}

impl ReceiveProgress {
    /// A progress record with nothing filled and no first byte.
    pub fn new() -> Self {
        Self::default()
    }

    /// The distinct destination positions filled so far.
    ///
    /// Saturates at `u64::MAX`; it never wraps.  The load pairs with the
    /// release in [`SpanMark::record_copy`], so a reader that sees a count also
    /// sees the first-byte time recorded before it.
    pub fn filled(&self) -> u64 {
        self.filled.load(Ordering::Acquire)
    }

    /// When the first non-empty copy was recorded, if any.
    pub fn first_byte(&self) -> Option<SystemTime> {
        self.first_byte.get().copied()
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
    /// * A copy that starts beyond the span's mark leaves a hole of positions
    ///   no attempt has filled; it is an error and nothing is recorded.
    /// * The first non-empty copy sets `first_byte`, whatever span or attempt.
    /// * If the copy's end passes the span's mark, the difference is added to
    ///   `filled` (saturating at `u64::MAX`) and the mark moves; otherwise
    ///   nothing is added.
    ///
    /// Returns the number of positions newly counted.
    pub fn record_copy(&mut self, start: u64, len: u64) -> anyhow::Result<u64> {
        if len == 0 {
            return Ok(0);
        }
        if start > self.mark {
            anyhow::bail!(
                "non-contiguous copy: start {start} beyond mark {}",
                self.mark
            );
        }
        // Set first_byte before the release add below, so a reader that sees
        // the count also sees the time.
        self.progress.first_byte.get_or_init(SystemTime::now);
        let end = start.saturating_add(len);
        if end <= self.mark {
            return Ok(0);
        }
        let delta = end - self.mark;
        self.mark = end;
        // A CAS loop, so the total saturates instead of wrapping.  (Written out
        // rather than with `fetch_update`, which newer toolchains deprecate and
        // this crate's minimum version does not yet have a replacement for.)
        let total = &self.progress.filled;
        let mut current = total.load(Ordering::Relaxed);
        loop {
            match total.compare_exchange_weak(
                current,
                current.saturating_add(delta),
                Ordering::Release,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(seen) => current = seen,
            }
        }
        Ok(delta)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Record a copy that the test expects to be contiguous.
    fn rec(span: &mut SpanMark<'_>, start: u64, len: u64) -> u64 {
        span.record_copy(start, len).expect("contiguous copy")
    }

    /// A retry that restarts at the span start and runs past the old mark counts
    /// only the positions beyond it.
    #[test]
    fn retry_crossing_the_old_mark_counts_each_position_once() {
        let p = ReceiveProgress::new();
        let mut span = SpanMark::new(&p);
        // Attempt 1 fills 0..60 in chunks, then fails.
        assert_eq!(rec(&mut span, 0, 40), 40);
        assert_eq!(rec(&mut span, 40, 20), 20);
        assert_eq!(p.filled(), 60);
        // Attempt 2 restarts at 0: nothing new until it passes 60.
        assert_eq!(rec(&mut span, 0, 30), 0);
        assert_eq!(rec(&mut span, 30, 30), 0);
        assert_eq!(p.filled(), 60);
        // It crosses the old mark: only 60..100 is new.
        assert_eq!(rec(&mut span, 60, 40), 40);
        assert_eq!(p.filled(), 100);
        // A third attempt over the whole span adds nothing.
        assert_eq!(rec(&mut span, 0, 100), 0);
        assert_eq!(p.filled(), 100);
    }

    /// A copy straddling the mark counts only its part beyond it.
    #[test]
    fn copy_straddling_the_mark_counts_the_excess_only() {
        let p = ReceiveProgress::new();
        let mut span = SpanMark::new(&p);
        assert_eq!(rec(&mut span, 0, 50), 50);
        assert_eq!(rec(&mut span, 20, 50), 20); // 20..70 passes 50
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
        assert_eq!(rec(&mut span, 0, 70), 70);
        // Repair 1 resumes at 50 (below the mark) and reaches 85, then breaks.
        assert_eq!(rec(&mut span, 50, 35), 15);
        // Repair 2 resumes at 80 (below the mark again) and finishes the span.
        assert_eq!(rec(&mut span, 80, 20), 15);
        // Repair 3, a redundant rewrite of the tail, adds nothing.
        assert_eq!(rec(&mut span, 90, 10), 0);
        assert_eq!(p.filled(), 100);
    }

    /// Disjoint spans each keep their own mark, and the totals sum.
    #[test]
    fn disjoint_spans_have_independent_marks_that_sum() {
        let p = ReceiveProgress::new();
        let mut a = SpanMark::new(&p);
        let mut b = SpanMark::new(&p);
        rec(&mut a, 0, 10);
        rec(&mut b, 0, 4);
        rec(&mut a, 0, 10); // a retry of span a
        rec(&mut b, 0, 8);
        assert_eq!(p.filled(), 18);
    }

    /// Only a non-empty copy sets `first_byte`, and only the first one does.
    #[test]
    fn first_non_empty_copy_sets_first_byte_once() {
        let p = ReceiveProgress::new();
        let mut span = SpanMark::new(&p);
        assert_eq!(rec(&mut span, 0, 0), 0);
        assert!(p.first_byte().is_none(), "an empty copy is no first byte");
        rec(&mut span, 0, 5);
        let first = p.first_byte().expect("first byte set");
        rec(&mut span, 5, 5);
        rec(&mut span, 0, 10);
        assert_eq!(p.first_byte().expect("still set"), first);
    }

    /// A copy that starts beyond the mark would count a hole nobody filled: it
    /// is an error and changes nothing (count, mark, first byte).
    #[test]
    fn a_copy_beyond_the_mark_is_refused_and_changes_nothing() {
        let p = ReceiveProgress::new();
        let mut span = SpanMark::new(&p);
        // A gap on the very first copy: no first byte either.
        let err = span.record_copy(10, 5).expect_err("gap at the start");
        assert!(
            err.to_string()
                .contains("non-contiguous copy: start 10 beyond mark 0"),
            "{err}"
        );
        assert_eq!((p.filled(), span.mark()), (0, 0));
        assert!(p.first_byte().is_none(), "a refused copy is no first byte");
        // A gap after real progress.
        assert_eq!(rec(&mut span, 0, 20), 20);
        let err = span.record_copy(25, 5).expect_err("gap after the mark");
        assert!(err.to_string().contains("start 25 beyond mark 20"), "{err}");
        assert_eq!((p.filled(), span.mark()), (20, 20));
        // A copy that starts exactly at the mark is contiguous.
        assert_eq!(rec(&mut span, 20, 5), 5);
        assert_eq!(p.filled(), 25);
    }

    /// The total saturates instead of wrapping or panicking when spans together
    /// exceed `u64::MAX`.
    #[test]
    fn the_total_never_wraps_when_spans_overflow_u64() {
        let p = ReceiveProgress::new();
        let half = 1u64 << 63;
        let mut a = SpanMark::new(&p);
        let mut b = SpanMark::new(&p);
        let mut c = SpanMark::new(&p);
        assert_eq!(rec(&mut a, 0, half), half);
        assert_eq!(rec(&mut b, 0, half), half);
        // The sum is now 2^64, one past u64::MAX; it must not wrap to 0.
        assert!(p.filled() >= half, "wrapped: {}", p.filled());
        let before = p.filled();
        assert_eq!(rec(&mut c, 0, 7), 7);
        assert!(p.filled() >= before, "wrapped: {}", p.filled());
    }

    /// An end past `u64::MAX` saturates in the mark instead of panicking.
    #[test]
    fn an_end_that_would_overflow_saturates_instead_of_panicking() {
        let p = ReceiveProgress::new();
        let mut span = SpanMark::new(&p);
        assert_eq!(rec(&mut span, 0, u64::MAX - 1), u64::MAX - 1);
        assert_eq!(rec(&mut span, u64::MAX - 2, 10), 1);
        assert_eq!(span.mark(), u64::MAX);
    }
}
