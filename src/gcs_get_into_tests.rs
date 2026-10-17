// src/gcs_get_into_tests.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Tests of the GCS caller-buffer receive loop (plan §4.3, §4.5 A2/A7/A8, §9 "Providers -> GCS").
//
// The loop is driven by a deterministic fake that behaves like the service where it matters:
// ranges and generations are honoured, an overwritten object is reachable only under its new
// generation, and failures are the real typed `google_cloud_gax` errors, so the production
// classifier and the typed-only retry rule are what decide.  Local only: no network, no
// credentials.

use super::caller_buffer::{
    loggable, read_config, receive_into, ChunkStream, ObjectFacts, OpenRequest, Opened, ReadOpener,
    ReceiveConfig,
};
use super::*;
use crate::object_error::{object_error, ObjectErrorKind};
use crate::receive_progress::ReceiveProgress;
use async_trait::async_trait;
use google_cloud_gax::error::rpc::{Code, Status};
use google_cloud_gax::error::Error as Gax;
use google_cloud_storage::model::Object;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

const FILL: u8 = 0xAB;
const CANARY: usize = 16;
const GEN: i64 = 77;
const WHAT: &str = "gs://bucket/object";

/// A typed failure the fake can raise.
#[derive(Clone, Copy, Debug)]
enum Fail {
    Http(u16),
    /// A definite client error whose payload merely looks like a retryable status.
    HttpWithText(u16),
    Grpc(Code),
    /// A gRPC status whose message looks retryable to the old message-text helper.
    GrpcWithText(Code),
    Io,
    /// A deserialization failure whose text looks retryable.
    DeserWithText,
}

impl Fail {
    fn build(self) -> Gax {
        match self {
            Fail::Http(c) => Gax::http(c, http::HeaderMap::new(), Bytes::from_static(b"body")),
            Fail::HttpWithText(c) => Gax::http(
                c,
                http::HeaderMap::new(),
                Bytes::from_static(b"503 unavailable timeout"),
            ),
            Fail::Grpc(c) => Gax::service(Status::default().set_code(c).set_message("injected")),
            Fail::GrpcWithText(c) => Gax::service(
                Status::default()
                    .set_code(c)
                    .set_message("UNAVAILABLE 503 timeout cancel resource_exhausted"),
            ),
            Fail::Io => Gax::io("connection reset"),
            Fail::DeserWithText => Gax::deser("503 unavailable timeout"),
        }
    }
}

/// What the next `open` does.
enum Step {
    /// Serve the requested range completely.
    Serve,
    /// The open itself fails.
    FailOpen(Fail),
    /// Serve this many bytes of the range, then a stream error.
    BreakAfter(usize, Fail),
    /// Serve this many bytes of the range, then end cleanly (a short body).
    EndShort(usize),
    /// Serve the range and this many extra bytes.
    Overrun(usize),
    /// Serve this many bytes, then never produce another chunk.
    HangAfter(usize),
    /// The object is overwritten (new bytes, new generation) before this request is served.
    Overwrite {
        data: Vec<u8>,
        generation: i64,
        then: Box<Step>,
    },
}

enum Tail {
    End,
    Fail(Fail),
    Hang,
}

struct State {
    data: Vec<u8>,
    /// The live generation.
    generation: i64,
    /// What responses report as the generation (`0` = the response carried none).
    report_generation: Option<i64>,
    /// A request naming a generation that is not live gets 404, as when the old bytes are gone.
    honour_generation: bool,
    script: VecDeque<Step>,
    /// `(start, limit, generation)` of every open, in order.
    requests: Vec<(u64, Option<u64>, Option<i64>)>,
    chunk: usize,
    /// Chunks actually pulled from any stream.
    pulled: usize,
    describes: usize,
    /// The generation each `describe` was asked for, in order.
    describe_generations: Vec<Option<i64>>,
    describe_fail: Option<Fail>,
    /// What responses report as the object's size (`None` = its real length).
    report_size: Option<i64>,
}

struct Fake(Arc<Mutex<State>>);

impl Fake {
    fn new(data: Vec<u8>, chunk: usize) -> Self {
        Fake(Arc::new(Mutex::new(State {
            data,
            generation: GEN,
            report_generation: None,
            honour_generation: true,
            script: VecDeque::new(),
            requests: Vec::new(),
            chunk,
            pulled: 0,
            describes: 0,
            describe_generations: Vec::new(),
            describe_fail: None,
            report_size: None,
        })))
    }

    fn script(self, steps: impl IntoIterator<Item = Step>) -> Self {
        self.0.lock().expect("lock").script.extend(steps);
        self
    }

    fn with<R>(&self, f: impl FnOnce(&mut State) -> R) -> R {
        f(&mut self.0.lock().expect("lock"))
    }

    fn requests(&self) -> Vec<(u64, Option<u64>, Option<i64>)> {
        self.with(|s| s.requests.clone())
    }
}

struct FakeStream {
    chunks: VecDeque<Bytes>,
    tail: Option<Tail>,
    state: Arc<Mutex<State>>,
}

#[async_trait]
impl ChunkStream for FakeStream {
    async fn next_chunk(&mut self) -> Option<Result<Bytes, Gax>> {
        if let Some(chunk) = self.chunks.pop_front() {
            self.state.lock().expect("lock").pulled += 1;
            return Some(Ok(chunk));
        }
        match self.tail.take()? {
            Tail::End => None,
            Tail::Fail(f) => Some(Err(f.build())),
            Tail::Hang => std::future::pending().await,
        }
    }
}

#[async_trait]
impl ReadOpener for Fake {
    async fn open(&self, request: &OpenRequest) -> Result<Opened, Gax> {
        let mut st = self.0.lock().expect("lock");
        st.requests
            .push((request.start, request.limit, request.generation));
        let mut step = st.script.pop_front().unwrap_or(Step::Serve);
        while let Step::Overwrite {
            data,
            generation,
            then,
        } = step
        {
            st.data = data;
            st.generation = generation;
            step = *then;
        }
        if let Step::FailOpen(f) = step {
            return Err(f.build());
        }
        if let Some(g) = request.generation {
            if st.honour_generation && g != st.generation {
                return Err(Fail::Http(404).build());
            }
        }
        let start = usize::try_from(request.start)
            .expect("start")
            .min(st.data.len());
        let end = match request.limit {
            Some(l) => start
                .saturating_add(usize::try_from(l).expect("limit"))
                .min(st.data.len()),
            None => st.data.len(),
        };
        let body = st.data[start..end].to_vec();
        let (mut served, tail) = match step {
            Step::Serve => (body, Tail::End),
            Step::BreakAfter(n, f) => (body[..n.min(body.len())].to_vec(), Tail::Fail(f)),
            Step::EndShort(n) => (body[..n.min(body.len())].to_vec(), Tail::End),
            Step::Overrun(extra) => {
                let mut b = body;
                b.extend(std::iter::repeat_n(0xEE, extra));
                (b, Tail::End)
            }
            Step::HangAfter(n) => (body[..n.min(body.len())].to_vec(), Tail::Hang),
            Step::FailOpen(_) | Step::Overwrite { .. } => unreachable!("handled above"),
        };
        let mut chunks = VecDeque::new();
        while !served.is_empty() {
            let rest = served.split_off(st.chunk.min(served.len()));
            chunks.push_back(Bytes::from(std::mem::replace(&mut served, rest)));
        }
        let facts = ObjectFacts {
            size: st
                .report_size
                .unwrap_or_else(|| i64::try_from(st.data.len()).expect("size")),
            generation: st.report_generation.unwrap_or(st.generation),
        };
        Ok(Opened {
            facts,
            stream: Box::new(FakeStream {
                chunks,
                tail: Some(tail),
                state: Arc::clone(&self.0),
            }),
        })
    }

    async fn describe(&self, generation: Option<i64>) -> Result<Object, Gax> {
        let mut st = self.0.lock().expect("lock");
        st.describes += 1;
        st.describe_generations.push(generation);
        if let Some(f) = st.describe_fail {
            return Err(f.build());
        }
        Ok(Object::default()
            .set_size(
                st.report_size
                    .unwrap_or_else(|| i64::try_from(st.data.len()).expect("size")),
            )
            .set_generation(st.report_generation.unwrap_or(st.generation)))
    }
}

/// Bytes that never equal the fill value.
fn pattern(n: usize, seed: usize) -> Vec<u8> {
    (0..n)
        .map(|i| u8::try_from((i * 7 + seed) % 200).expect("below 200"))
        .collect()
}

fn config(rapid: bool) -> ReceiveConfig {
    ReceiveConfig {
        rapid,
        chunk_timeout: None,
        retry_delay: |_| Duration::ZERO,
        max_open_attempts: 3,
        max_repairs: 4,
    }
}

/// The generation each descriptor request asked for.
fn descriptor_generations(fake: &Fake) -> Vec<Option<i64>> {
    fake.with(|s| s.describe_generations.clone())
}

struct Outcome {
    result: Result<()>,
    /// `dst` followed by a canary tail, both pre-filled with `FILL`.
    buf: Vec<u8>,
    n: usize,
    progress: ReceiveProgress,
}

impl Outcome {
    fn dst(&self) -> &[u8] {
        &self.buf[..self.n]
    }

    fn kind(&self) -> ObjectErrorKind {
        let err = self.result.as_ref().expect_err("a failure");
        object_error(err).expect("typed").kind
    }

    /// Nothing beyond `dst` was written.
    fn assert_canary_intact(&self) {
        assert!(
            self.buf[self.n..].iter().all(|b| *b == FILL),
            "a write went beyond the destination"
        );
    }
}

async fn run(fake: &Fake, n: usize, cfg: &ReceiveConfig) -> Outcome {
    let mut buf = vec![FILL; n + CANARY];
    let progress = ReceiveProgress::new();
    let result = receive_into(fake, WHAT, &mut buf[..n], &progress, cfg).await;
    Outcome {
        result,
        buf,
        n,
        progress,
    }
}

// ---------------------------------------------------------------------------
// Lengths
// ---------------------------------------------------------------------------

/// An object of exactly the destination's size is received in full and counted once.
#[tokio::test]
async fn exact_object_is_received_and_counted_once() {
    let data = pattern(10, 1);
    let fake = Fake::new(data.clone(), 4);
    let out = run(&fake, 10, &config(false)).await;
    out.result.as_ref().expect("received");
    assert_eq!(out.dst(), &data[..]);
    assert_eq!(out.progress.filled(), 10);
    assert!(out.progress.first_byte().is_some());
    assert_eq!(fake.requests(), vec![(0, None, None)]);
    out.assert_canary_intact();
}

/// An object shorter than the destination fails on the size alone: no byte is copied.
#[tokio::test]
async fn short_object_is_a_length_mismatch_before_any_copy() {
    let fake = Fake::new(pattern(6, 1), 4);
    let out = run(&fake, 10, &config(false)).await;
    assert_eq!(
        out.kind(),
        ObjectErrorKind::LengthMismatch {
            expected: 10,
            actual: Some(6)
        }
    );
    assert_eq!(
        fake.with(|s| s.pulled),
        0,
        "the body must be dropped unread"
    );
    assert_eq!(out.progress.filled(), 0);
    assert!(out.dst().iter().all(|b| *b == FILL));
    out.assert_canary_intact();
}

/// An object longer than the destination fails on the size alone too.
#[tokio::test]
async fn long_object_is_a_length_mismatch_before_any_copy() {
    let fake = Fake::new(pattern(14, 1), 4);
    let out = run(&fake, 10, &config(false)).await;
    assert_eq!(
        out.kind(),
        ObjectErrorKind::LengthMismatch {
            expected: 10,
            actual: Some(14)
        }
    );
    assert_eq!(fake.with(|s| s.pulled), 0);
    assert_eq!(out.progress.filled(), 0);
    out.assert_canary_intact();
}

/// A body longer than the object's announced size is refused: the chunk that would pass the
/// end of the destination is not written at all, not even its part that would fit.
#[tokio::test]
async fn body_longer_than_announced_never_writes_past_dst() {
    let data = pattern(10, 1);
    let fake = Fake::new(data.clone(), 4).script([Step::Overrun(5)]);
    let out = run(&fake, 10, &config(false)).await;
    assert_eq!(
        out.kind(),
        ObjectErrorKind::LengthMismatch {
            expected: 10,
            actual: None
        }
    );
    // Chunks are 0..4, 4..8, then 8..12 which would pass the end.
    assert_eq!(&out.dst()[..8], &data[..8]);
    assert!(
        out.dst()[8..].iter().all(|b| *b == FILL),
        "the overrunning chunk was partly written"
    );
    assert_eq!(out.progress.filled(), 8);
    out.assert_canary_intact();
}

/// A clean but short body on a pinned read is repaired: the suffix of the same span is read
/// from the cursor, under the first response's generation.
#[tokio::test]
async fn short_body_is_repaired_from_the_cursor_under_the_pinned_generation() {
    let data = pattern(10, 2);
    let fake = Fake::new(data.clone(), 4).script([Step::EndShort(6), Step::Serve]);
    let out = run(&fake, 10, &config(false)).await;
    out.result.as_ref().expect("repaired");
    assert_eq!(out.dst(), &data[..]);
    assert_eq!(out.progress.filled(), 10);
    assert_eq!(
        fake.requests(),
        vec![(0, None, None), (6, Some(4), Some(GEN))]
    );
    out.assert_canary_intact();
}

/// A repair that brings nothing is not repeated: the read ends as a length mismatch that
/// says how much was received.
#[tokio::test]
async fn short_body_without_progress_is_a_length_mismatch() {
    let fake = Fake::new(pattern(10, 2), 4).script([Step::EndShort(6), Step::EndShort(0)]);
    let out = run(&fake, 10, &config(false)).await;
    assert_eq!(
        out.kind(),
        ObjectErrorKind::LengthMismatch {
            expected: 10,
            actual: Some(6)
        }
    );
    assert_eq!(fake.requests().len(), 2, "one repair, then stop");
    assert_eq!(out.progress.filled(), 6);
    out.assert_canary_intact();
}

// ---------------------------------------------------------------------------
// Repair, counting and generation pinning
// ---------------------------------------------------------------------------

/// Several tail repairs: every one resumes at the cursor with the bounded remainder, every one
/// is pinned to the first response's generation, and each position is counted once.
#[tokio::test]
async fn multiple_tail_repairs_resume_at_the_cursor_all_pinned() {
    let data = pattern(100, 3);
    let fake = Fake::new(data.clone(), 10).script([
        Step::BreakAfter(30, Fail::Io),
        Step::BreakAfter(20, Fail::Http(503)),
        Step::BreakAfter(15, Fail::Grpc(Code::Unavailable)),
        Step::Serve,
    ]);
    let out = run(&fake, 100, &config(false)).await;
    out.result.as_ref().expect("repaired");
    assert_eq!(out.dst(), &data[..]);
    assert_eq!(out.progress.filled(), 100);
    assert_eq!(
        fake.requests(),
        vec![
            (0, None, None),
            (30, Some(70), Some(GEN)),
            (50, Some(50), Some(GEN)),
            (65, Some(35), Some(GEN)),
        ],
        "the first request has nothing to pin; every repair is pinned and bounded"
    );
    out.assert_canary_intact();
}

/// Retries before any response cannot pin (nothing is known yet); once a response arrives, a
/// retry that still has no bytes is pinned too.
#[tokio::test]
async fn retries_pin_as_soon_as_a_generation_is_known() {
    let data = pattern(20, 3);
    let fake = Fake::new(data.clone(), 5).script([
        Step::FailOpen(Fail::Http(503)),
        Step::BreakAfter(0, Fail::Io),
        Step::Serve,
    ]);
    let out = run(&fake, 20, &config(false)).await;
    out.result.as_ref().expect("received");
    assert_eq!(out.dst(), &data[..]);
    assert_eq!(
        fake.requests(),
        vec![(0, None, None), (0, None, None), (0, None, Some(GEN))]
    );
}

/// The span's progress survives failure: a read that fails reports the positions it touched.
#[tokio::test]
async fn progress_is_kept_when_the_read_fails() {
    let data = pattern(100, 3);
    let fake = Fake::new(data.clone(), 10).script([Step::BreakAfter(40, Fail::Http(403))]);
    let out = run(&fake, 100, &config(false)).await;
    assert_eq!(out.kind(), ObjectErrorKind::Forbidden);
    assert_eq!(out.progress.filled(), 40);
    assert_eq!(&out.dst()[..40], &data[..40]);
    assert!(out.dst()[40..].iter().all(|b| *b == FILL));
}

/// Repairs that make no progress are bounded: after the allowed number the typed error is
/// returned, with the progress made.  The first read delivers 10 bytes; every repair after it
/// breaks again at the same cursor.
#[tokio::test]
async fn repairs_without_progress_are_bounded() {
    let fake = Fake::new(pattern(100, 3), 10).script(
        std::iter::once(Step::BreakAfter(10, Fail::Io))
            .chain((0..10).map(|_| Step::BreakAfter(0, Fail::Io))),
    );
    let out = run(&fake, 100, &config(false)).await;
    assert_eq!(out.kind(), ObjectErrorKind::Transport);
    // The first read plus four repairs, none of which advanced the cursor.
    assert_eq!(fake.requests().len(), 5);
    assert_eq!(out.progress.filled(), 10);
    let err = out.result.as_ref().expect_err("failed");
    assert!(
        err.downcast_ref::<Gax>().is_some(),
        "the typed source stays recoverable"
    );
    out.assert_canary_intact();
}

/// A repair that advances the cursor starts the count again: a large read that needs far more
/// repairs than `max_repairs` (4 here) still completes, every repair resuming at the cursor
/// under the pinned generation, and each position counted once.
#[tokio::test]
async fn advancing_repairs_do_not_exhaust_the_budget() {
    let data = pattern(100, 3);
    // Ten bytes per request: the first read and nine repairs.
    let fake = Fake::new(data.clone(), 5).script((0..9).map(|_| Step::BreakAfter(10, Fail::Io)));
    let out = run(&fake, 100, &config(false)).await;
    out.result.as_ref().expect("received through many repairs");
    assert_eq!(out.dst(), &data[..]);
    assert_eq!(out.progress.filled(), 100);
    let requests = fake.requests();
    assert_eq!(requests.len(), 10);
    for (index, (start, limit, generation)) in requests.iter().enumerate().skip(1) {
        let at = u64::try_from(index * 10).expect("offset");
        assert_eq!(*start, at);
        assert_eq!(*limit, Some(100 - at));
        assert_eq!(*generation, Some(GEN));
    }
    out.assert_canary_intact();
}

/// The bound counts repairs since the cursor last advanced: stalls that are interleaved with
/// progress never add up to the limit, and the read still fails after `max_repairs` stalls in a
/// row at one place.
#[tokio::test]
async fn the_count_restarts_after_progress_and_then_bounds_the_next_stall() {
    // 10 bytes, then three stalls (3 repairs), then 10 more bytes, then stalls without end.
    let mut steps = vec![Step::BreakAfter(10, Fail::Io)];
    steps.extend((0..3).map(|_| Step::BreakAfter(0, Fail::Io)));
    steps.push(Step::BreakAfter(10, Fail::Io));
    steps.extend((0..10).map(|_| Step::BreakAfter(0, Fail::Io)));
    let fake = Fake::new(pattern(100, 3), 10).script(steps);
    let out = run(&fake, 100, &config(false)).await;
    assert_eq!(out.kind(), ObjectErrorKind::Transport);
    // 1 first read + 3 stalled repairs + 1 repair that advanced + 4 stalled repairs.
    assert_eq!(fake.requests().len(), 9);
    assert_eq!(out.progress.filled(), 20);
    out.assert_canary_intact();
}

/// Opens that keep failing are bounded too.
#[tokio::test]
async fn failing_opens_are_bounded() {
    let fake =
        Fake::new(pattern(10, 1), 4).script((0..10).map(|_| Step::FailOpen(Fail::Http(503))));
    let out = run(&fake, 10, &config(false)).await;
    assert_eq!(out.kind(), ObjectErrorKind::Throttled);
    assert_eq!(fake.requests().len(), 3);
}

/// A stall is a typed timeout: it is a transport failure, retryable, and repaired.
#[tokio::test]
async fn chunk_timeout_is_a_retryable_transport_failure() {
    let data = pattern(10, 4);
    let fake = Fake::new(data.clone(), 4).script([Step::HangAfter(4), Step::Serve]);
    let mut cfg = config(false);
    cfg.chunk_timeout = Some(Duration::from_millis(50));
    let out = run(&fake, 10, &cfg).await;
    out.result.as_ref().expect("repaired after the stall");
    assert_eq!(out.dst(), &data[..]);
    assert_eq!(
        fake.requests(),
        vec![(0, None, None), (4, Some(6), Some(GEN))]
    );
}

// ---------------------------------------------------------------------------
// Version changes
// ---------------------------------------------------------------------------

/// A same-size overwrite between the break and the repair: the repair names the old generation,
/// the service no longer has it (404), and that is a version change - never absence, and
/// never a mix of old and new bytes.
#[tokio::test]
async fn same_size_overwrite_is_a_version_change_never_mixed_data() {
    let old = pattern(10, 1);
    let new = pattern(10, 9);
    assert_ne!(old, new);
    let fake = Fake::new(old.clone(), 4).script([
        Step::BreakAfter(4, Fail::Io),
        Step::Overwrite {
            data: new,
            generation: GEN + 1,
            then: Box::new(Step::Serve),
        },
    ]);
    let out = run(&fake, 10, &config(false)).await;
    assert_eq!(out.kind(), ObjectErrorKind::VersionChanged);
    let err = out.result.as_ref().expect_err("failed");
    assert!(!crate::delete_contract::is_not_found(err));
    assert_eq!(&out.dst()[..4], &old[..4]);
    assert!(
        out.dst()[4..].iter().all(|b| *b == FILL),
        "bytes of the new object were mixed in"
    );
    assert_eq!(out.progress.filled(), 4);
    assert_eq!(
        fake.requests(),
        vec![(0, None, None), (4, Some(6), Some(GEN))]
    );
}

/// Even a service that ignores the requested generation and answers with the new object is
/// caught: the response's generation differs from the pinned one, so none of its body is used.
#[tokio::test]
async fn a_response_with_another_generation_is_never_combined() {
    let old = pattern(10, 1);
    let fake = Fake::new(old.clone(), 4).script([
        Step::BreakAfter(4, Fail::Io),
        Step::Overwrite {
            data: pattern(10, 9),
            generation: GEN + 1,
            then: Box::new(Step::Serve),
        },
    ]);
    fake.with(|s| s.honour_generation = false);
    let out = run(&fake, 10, &config(false)).await;
    assert_eq!(out.kind(), ObjectErrorKind::VersionChanged);
    assert_eq!(&out.dst()[..4], &old[..4]);
    assert!(out.dst()[4..].iter().all(|b| *b == FILL));
    assert_eq!(
        fake.with(|s| s.pulled),
        1,
        "no chunk of the other generation may be pulled"
    );
}

/// The pinned generation matches but the size does not: the pinned version is inconsistent,
/// which is a version change, not a plain length mismatch.
#[tokio::test]
async fn a_size_change_on_a_pinned_repair_is_a_version_change() {
    let fake = Fake::new(pattern(10, 1), 4).script([
        Step::BreakAfter(4, Fail::Io),
        Step::Overwrite {
            data: pattern(12, 1),
            generation: GEN,
            then: Box::new(Step::Serve),
        },
    ]);
    fake.with(|s| s.honour_generation = false);
    let out = run(&fake, 10, &config(false)).await;
    assert_eq!(out.kind(), ObjectErrorKind::VersionChanged);
    assert!(out.dst()[4..].iter().all(|b| *b == FILL));
}

/// Absence from the very first request is absence.
#[tokio::test]
async fn first_request_404_is_not_found_and_not_retried() {
    let fake = Fake::new(pattern(10, 1), 4).script([Step::FailOpen(Fail::Http(404))]);
    let out = run(&fake, 10, &config(false)).await;
    assert_eq!(out.kind(), ObjectErrorKind::NotFound);
    assert!(crate::delete_contract::is_not_found(
        out.result.as_ref().expect_err("failed")
    ));
    assert_eq!(fake.requests().len(), 1);
}

/// Once a generation is pinned, a 404 - on a retry that has no bytes yet as well as on a
/// repair - is a version change, in HTTP and gRPC form.
#[tokio::test]
async fn pinned_404_is_a_version_change_on_retry_and_on_repair() {
    for (break_at, not_found) in [
        (0, Fail::Http(404)),
        (4, Fail::Http(404)),
        (0, Fail::Grpc(Code::NotFound)),
        (4, Fail::Grpc(Code::NotFound)),
    ] {
        let fake = Fake::new(pattern(10, 1), 4).script([
            Step::BreakAfter(break_at, Fail::Io),
            Step::FailOpen(not_found),
        ]);
        let out = run(&fake, 10, &config(false)).await;
        assert_eq!(
            out.kind(),
            ObjectErrorKind::VersionChanged,
            "{break_at} {not_found:?}"
        );
        assert!(!crate::delete_contract::is_not_found(
            out.result.as_ref().expect_err("failed")
        ));
        assert_eq!(fake.requests().len(), 2);
    }
}

// ---------------------------------------------------------------------------
// Unpinned responses
// ---------------------------------------------------------------------------

/// With no usable generation a single complete response is accepted.
#[tokio::test]
async fn unpinned_single_response_is_accepted() {
    let data = pattern(10, 5);
    let fake = Fake::new(data.clone(), 4);
    fake.with(|s| s.report_generation = Some(0));
    let out = run(&fake, 10, &config(false)).await;
    out.result.as_ref().expect("accepted");
    assert_eq!(out.dst(), &data[..]);
}

/// With no usable generation a partially received request fails: it is neither repaired nor
/// retried from pieces of unknown versions.
#[tokio::test]
async fn unpinned_partial_read_fails_without_a_second_request() {
    let data = pattern(10, 5);
    let fake = Fake::new(data.clone(), 4).script([Step::BreakAfter(4, Fail::Io), Step::Serve]);
    fake.with(|s| s.report_generation = Some(0));
    let out = run(&fake, 10, &config(false)).await;
    assert_eq!(out.kind(), ObjectErrorKind::Transport);
    assert_eq!(fake.requests().len(), 1);
    assert_eq!(out.progress.filled(), 4);
    assert!(out.dst()[4..].iter().all(|b| *b == FILL));
}

/// An unpinned clean short end is a length mismatch, not a repair.
#[tokio::test]
async fn unpinned_short_body_is_a_length_mismatch() {
    let fake = Fake::new(pattern(10, 5), 4).script([Step::EndShort(6), Step::Serve]);
    fake.with(|s| s.report_generation = Some(0));
    let out = run(&fake, 10, &config(false)).await;
    assert_eq!(
        out.kind(),
        ObjectErrorKind::LengthMismatch {
            expected: 10,
            actual: Some(6)
        }
    );
    assert_eq!(fake.requests().len(), 1);
}

/// With no bytes written yet an unpinned retry is allowed (nothing is combined).
#[tokio::test]
async fn unpinned_retry_before_any_byte_is_allowed() {
    let data = pattern(10, 5);
    let fake = Fake::new(data.clone(), 4).script([Step::FailOpen(Fail::Http(503)), Step::Serve]);
    fake.with(|s| s.report_generation = Some(0));
    let out = run(&fake, 10, &config(false)).await;
    out.result.as_ref().expect("retried");
    assert_eq!(out.dst(), &data[..]);
    assert_eq!(fake.requests(), vec![(0, None, None), (0, None, None)]);
}

// ---------------------------------------------------------------------------
// The typed-only retry rule
// ---------------------------------------------------------------------------

/// Typed transient and transport failures are retried, on the open and in the stream.
#[tokio::test]
async fn typed_transient_failures_are_retried() {
    for fail in [
        Fail::Http(503),
        Fail::Http(429),
        Fail::Http(500),
        Fail::Http(408),
        Fail::Grpc(Code::Unavailable),
        Fail::Grpc(Code::ResourceExhausted),
        Fail::Grpc(Code::DeadlineExceeded),
        Fail::Io,
    ] {
        let data = pattern(10, 6);
        let fake = Fake::new(data.clone(), 4).script([Step::FailOpen(fail), Step::Serve]);
        let out = run(&fake, 10, &config(false)).await;
        out.result
            .as_ref()
            .unwrap_or_else(|e| panic!("{fail:?}: {e:#}"));
        assert_eq!(out.dst(), &data[..], "{fail:?}");
        assert_eq!(fake.requests().len(), 2, "{fail:?} on open");

        let fake = Fake::new(data.clone(), 4).script([Step::BreakAfter(4, fail), Step::Serve]);
        let out = run(&fake, 10, &config(false)).await;
        out.result
            .as_ref()
            .unwrap_or_else(|e| panic!("{fail:?}: {e:#}"));
        assert_eq!(out.dst(), &data[..], "{fail:?}");
        assert_eq!(fake.requests().len(), 2, "{fail:?} in the stream");
    }
}

/// Nothing else is retried: not a failure whose text merely contains a retryable status or
/// word, not `Other`, not 403 or PermissionDenied, not an authentication failure.
#[tokio::test]
async fn untyped_and_definite_failures_are_not_retried() {
    let cases = [
        (Fail::Http(403), ObjectErrorKind::Forbidden),
        (
            Fail::Grpc(Code::PermissionDenied),
            ObjectErrorKind::Forbidden,
        ),
        (Fail::HttpWithText(400), ObjectErrorKind::Other),
        (Fail::DeserWithText, ObjectErrorKind::Other),
        (Fail::GrpcWithText(Code::Unknown), ObjectErrorKind::Other),
        (Fail::GrpcWithText(Code::Cancelled), ObjectErrorKind::Other),
        (Fail::Http(401), ObjectErrorKind::Other),
        (Fail::Grpc(Code::Unauthenticated), ObjectErrorKind::Other),
        (Fail::Http(404), ObjectErrorKind::NotFound),
    ];
    for (fail, kind) in cases {
        let fake = Fake::new(pattern(10, 6), 4).script([Step::FailOpen(fail), Step::Serve]);
        let out = run(&fake, 10, &config(false)).await;
        assert_eq!(out.kind(), kind, "{fail:?} on open");
        assert_eq!(fake.requests().len(), 1, "{fail:?} must not be retried");

        // In the stream, after a pinned partial read: a repair must not be attempted either.
        if matches!(kind, ObjectErrorKind::NotFound) {
            continue; // a 404 cannot arise mid-stream
        }
        let fake = Fake::new(pattern(10, 6), 4).script([Step::BreakAfter(4, fail), Step::Serve]);
        let out = run(&fake, 10, &config(false)).await;
        assert_eq!(out.kind(), kind, "{fail:?} in the stream");
        assert_eq!(fake.requests().len(), 1, "{fail:?} must not be repaired");
        assert_eq!(out.progress.filled(), 4);
    }
}

// ---------------------------------------------------------------------------
// Zero length
// ---------------------------------------------------------------------------

/// HTTP, zero-length destination, empty object: an unranged read whose size header is 0.
#[tokio::test]
async fn zero_length_http_reads_unranged_and_accepts_an_empty_object() {
    let fake = Fake::new(Vec::new(), 4);
    let out = run(&fake, 0, &config(false)).await;
    out.result.as_ref().expect("empty object received");
    assert_eq!(fake.requests(), vec![(0, None, None)]);
    assert_eq!(out.progress.filled(), 0);
    assert!(out.progress.first_byte().is_none());
    out.assert_canary_intact();
}

/// HTTP, zero-length destination, non-empty object: a length mismatch, and the body is
/// dropped unread.
#[tokio::test]
async fn zero_length_http_with_a_non_empty_object_is_a_mismatch_with_the_body_unread() {
    let fake = Fake::new(pattern(5, 1), 4);
    let out = run(&fake, 0, &config(false)).await;
    assert_eq!(
        out.kind(),
        ObjectErrorKind::LengthMismatch {
            expected: 0,
            actual: Some(5)
        }
    );
    assert_eq!(fake.with(|s| s.pulled), 0);
    out.assert_canary_intact();
}

/// A body on an object that announced size 0 is refused, not discarded silently.
#[tokio::test]
async fn zero_length_http_with_an_unannounced_body_is_a_mismatch() {
    let fake = Fake::new(Vec::new(), 4).script([Step::Overrun(3)]);
    let out = run(&fake, 0, &config(false)).await;
    assert_eq!(
        out.kind(),
        ObjectErrorKind::LengthMismatch {
            expected: 0,
            actual: None
        }
    );
    out.assert_canary_intact();
}

/// Rapid, zero-length destination: the descriptor alone answers.  No data range is requested
/// (a full read of an empty object may fail with OUT_OF_RANGE), and no size header is used.
#[tokio::test]
async fn zero_length_rapid_reads_the_descriptor_and_requests_no_data() {
    let fake = Fake::new(Vec::new(), 4);
    let out = run(&fake, 0, &config(true)).await;
    out.result.as_ref().expect("empty object");
    assert!(fake.requests().is_empty(), "no data range may be requested");
    assert_eq!(fake.with(|s| s.describes), 1);
    assert_eq!(descriptor_generations(&fake), vec![None]);
    out.assert_canary_intact();

    // A non-empty object: the descriptor's size is the authority.
    let fake = Fake::new(pattern(5, 1), 4);
    let out = run(&fake, 0, &config(true)).await;
    assert_eq!(
        out.kind(),
        ObjectErrorKind::LengthMismatch {
            expected: 0,
            actual: Some(5)
        }
    );
    assert!(fake.requests().is_empty());
    assert_eq!(fake.with(|s| s.pulled), 0);
}

/// A Rapid descriptor failure is classified: absence is absence, a transient failure is
/// retried, a refusal is not.
#[tokio::test]
async fn zero_length_rapid_descriptor_failures_are_classified() {
    let fake = Fake::new(Vec::new(), 4);
    fake.with(|s| s.describe_fail = Some(Fail::Grpc(Code::NotFound)));
    let out = run(&fake, 0, &config(true)).await;
    assert_eq!(out.kind(), ObjectErrorKind::NotFound);
    assert_eq!(fake.with(|s| s.describes), 1);
    assert_eq!(descriptor_generations(&fake), vec![None]);

    let fake = Fake::new(Vec::new(), 4);
    fake.with(|s| s.describe_fail = Some(Fail::Http(403)));
    let out = run(&fake, 0, &config(true)).await;
    assert_eq!(out.kind(), ObjectErrorKind::Forbidden);
    assert_eq!(fake.with(|s| s.describes), 1);
    assert_eq!(descriptor_generations(&fake), vec![None]);

    let fake = Fake::new(Vec::new(), 4);
    fake.with(|s| s.describe_fail = Some(Fail::Grpc(Code::Unavailable)));
    let out = run(&fake, 0, &config(true)).await;
    assert_eq!(out.kind(), ObjectErrorKind::Throttled);
    assert_eq!(
        fake.with(|s| s.describes),
        3,
        "retried up to the attempt limit"
    );
    assert_eq!(descriptor_generations(&fake), vec![None; 3]);
}

// ---------------------------------------------------------------------------
// Rapid OUT_OF_RANGE on the first full read
// ---------------------------------------------------------------------------

/// A non-zero destination and an empty Rapid object: the full read fails OUT_OF_RANGE; the
/// descriptor says the object is empty, so the result is a length mismatch (expected 10,
/// actual 0), not an opaque error and never absence.
#[tokio::test]
async fn rapid_out_of_range_on_an_empty_object_is_a_length_mismatch() {
    let fake = Fake::new(Vec::new(), 4).script([Step::FailOpen(Fail::Grpc(Code::OutOfRange))]);
    let out = run(&fake, 10, &config(true)).await;
    assert_eq!(
        out.kind(),
        ObjectErrorKind::LengthMismatch {
            expected: 10,
            actual: Some(0)
        }
    );
    assert_eq!(fake.with(|s| s.describes), 1);
    assert_eq!(descriptor_generations(&fake), vec![None]);
    out.assert_canary_intact();
}

/// OUT_OF_RANGE with a non-empty descriptor is kept as a failure: it is never turned into
/// absence or success, and it is not retried.
#[tokio::test]
async fn rapid_out_of_range_with_a_non_empty_descriptor_stays_a_failure() {
    let fake = Fake::new(pattern(5, 1), 4)
        .script([Step::FailOpen(Fail::Grpc(Code::OutOfRange)), Step::Serve]);
    let out = run(&fake, 10, &config(true)).await;
    assert_eq!(out.kind(), ObjectErrorKind::Other);
    assert!(!crate::delete_contract::is_not_found(
        out.result.as_ref().expect_err("failed")
    ));
    assert_eq!(fake.requests().len(), 1);
}

/// The same OUT_OF_RANGE on a standard (HTTP) read is not consulted against a descriptor.
#[tokio::test]
async fn out_of_range_on_the_http_route_stays_a_failure_without_a_descriptor_check() {
    let fake = Fake::new(Vec::new(), 4).script([Step::FailOpen(Fail::Grpc(Code::OutOfRange))]);
    let out = run(&fake, 10, &config(false)).await;
    assert_eq!(out.kind(), ObjectErrorKind::Other);
    assert_eq!(fake.with(|s| s.describes), 0);
    assert!(descriptor_generations(&fake).is_empty());
}

/// Rapid, non-zero destination, ordinary object: the existing full-read route is used and
/// the first response pins the generation for repair.
#[tokio::test]
async fn rapid_receive_repairs_with_the_generation_pinned() {
    let data = pattern(30, 8);
    let fake = Fake::new(data.clone(), 10).script([Step::BreakAfter(10, Fail::Io), Step::Serve]);
    let out = run(&fake, 30, &config(true)).await;
    out.result.as_ref().expect("repaired");
    assert_eq!(out.dst(), &data[..]);
    assert_eq!(
        fake.requests(),
        vec![(0, None, None), (10, Some(20), Some(GEN))]
    );
    assert_eq!(out.progress.filled(), 30);
    out.assert_canary_intact();
}

// ---------------------------------------------------------------------------
// The span is complete, the stream still fails
// ---------------------------------------------------------------------------

/// A retryable stream error that arrives after ALL bytes were received has no suffix left to
/// resume.  It is returned as the typed Transport error: no repair is asked for, because a
/// zero-length segment means "to the end" in this SDK fork, and exactly one request was made.
#[tokio::test]
async fn a_stream_error_after_every_byte_is_returned_without_a_repair() {
    let data = pattern(10, 2);
    let fake = Fake::new(data.clone(), 4).script([Step::BreakAfter(10, Fail::Io), Step::Serve]);
    let out = run(&fake, 10, &config(false)).await;
    assert_eq!(out.kind(), ObjectErrorKind::Transport);
    assert_eq!(
        fake.requests(),
        vec![(0, None, None)],
        "no repair of a complete span"
    );
    assert_eq!(out.progress.filled(), 10);
    assert_eq!(out.dst(), &data[..]);
    out.assert_canary_intact();
}

// ---------------------------------------------------------------------------
// Sizes
// ---------------------------------------------------------------------------

/// A negative size is an error, never a zero or a wrapped huge number: the read fails as
/// `Other` before any byte is copied.
#[tokio::test]
async fn a_negative_reported_size_is_an_error_and_nothing_is_written() {
    for (data, n) in [(pattern(10, 1), 10), (Vec::new(), 0)] {
        let fake = Fake::new(data, 4);
        fake.with(|s| s.report_size = Some(-1));
        let out = run(&fake, n, &config(false)).await;
        assert_eq!(out.kind(), ObjectErrorKind::Other, "destination of {n}");
        assert_eq!(
            fake.with(|s| s.pulled),
            0,
            "the body must be dropped unread"
        );
        assert_eq!(fake.requests().len(), 1, "a bad size is not retried");
        assert_eq!(out.progress.filled(), 0);
        assert!(out.dst().iter().all(|b| *b == FILL));
        out.assert_canary_intact();
    }
}

// ---------------------------------------------------------------------------
// Error context and logging
// ---------------------------------------------------------------------------

/// The error context names the phase that failed: opening a read, or its stream.
#[tokio::test]
async fn the_error_context_names_the_phase() {
    let fake = Fake::new(pattern(10, 1), 4).script([Step::FailOpen(Fail::Http(403))]);
    let out = run(&fake, 10, &config(false)).await;
    let message = format!("{:#}", out.result.as_ref().expect_err("failed"));
    assert!(message.contains("could not open"), "{message}");
    assert!(!message.contains("HEAD"), "{message}");

    let fake = Fake::new(pattern(10, 1), 4).script([Step::BreakAfter(4, Fail::Http(403))]);
    let out = run(&fake, 10, &config(false)).await;
    let message = format!("{:#}", out.result.as_ref().expect_err("failed"));
    assert!(message.contains("stream failed"), "{message}");
    assert!(!message.contains("HEAD"), "{message}");
}

/// What is logged about a failure is its classification only: server-supplied text in the
/// error body never reaches a log line.
#[test]
fn logged_failures_carry_the_classification_and_never_the_body() {
    let classified = |fail: Fail| {
        let err = attach_gcs_classification(anyhow::Error::new(fail.build()), false);
        loggable(&err)
    };
    let text = classified(Fail::HttpWithText(503));
    assert!(text.contains("status 503"), "{text}");
    assert!(!text.contains("unavailable timeout"), "{text}");
    let text = classified(Fail::GrpcWithText(Code::Unavailable));
    assert!(text.contains("Unavailable"), "{text}");
    assert!(!text.contains("resource_exhausted"), "{text}");
    assert_eq!(
        loggable(&anyhow::anyhow!("SECRET-PAYLOAD")),
        "unclassified failure"
    );
}

// ---------------------------------------------------------------------------
// The chunk timeout of the real routes
// ---------------------------------------------------------------------------

/// The standard (HTTP) route applies the same chunk timeout as Rapid: the configuration the
/// real entry point builds is the same for both, sized to the bytes expected, and honours the
/// documented environment override (0 turns it off).
#[test]
fn both_routes_apply_the_chunk_timeout() {
    let var = "S3DLIO_GCS_READ_CHUNK_TIMEOUT_SECS";
    // The configurations are gathered with the environment lock held and the variable restored;
    // the assertions run after it is released, so a failure cannot poison the lock for the
    // other environment tests.
    let timeouts = |value: Option<&str>, len: usize| -> Vec<(bool, Option<Duration>)> {
        let _guard = super::tests::ENV_MUTEX
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let before = std::env::var(var).ok();
        match value {
            Some(value) => std::env::set_var(var, value),
            None => std::env::remove_var(var),
        }
        let found = [false, true]
            .into_iter()
            .map(|rapid| {
                let cfg = read_config(rapid, len);
                (rapid, cfg.ok().and_then(|cfg| cfg.chunk_timeout))
            })
            .collect();
        match before {
            Some(value) => std::env::set_var(var, value),
            None => std::env::remove_var(var),
        }
        found
    };

    let seven = Some(Duration::from_secs(7));
    assert_eq!(
        timeouts(Some("7"), 4096),
        vec![(false, seven), (true, seven)]
    );
    assert_eq!(timeouts(Some("0"), 4096), vec![(false, None), (true, None)]);

    let size = 8 * 1024 * 1024;
    let scaled = Some(Duration::from_secs(12));
    assert_eq!(timeouts(None, size), vec![(false, scaled), (true, scaled)]);
}

// ---------------------------------------------------------------------------
// Structure
// ---------------------------------------------------------------------------

/// The loop decides retries with the typed-only verdict and never with the message-text
/// helper, and never holds `dst` in a spawned task.
#[test]
fn the_receive_loop_uses_only_the_typed_verdict_and_stays_in_its_task() {
    let src = include_str!("google_gcs_client.rs");
    let start = src
        .find("pub(super) async fn receive_into<")
        .expect("receive_into");
    let end = src[start..]
        .find("// -- typed HEAD (pure decision logic)")
        .expect("end of the loop");
    let body = &src[start..start + end];
    let budget_start = src.find("struct Budget").expect("Budget");
    let budget = &src[budget_start..start];
    for code in [body, budget] {
        assert!(!code.contains("is_retryable_gcs_read_error"));
        assert!(!code.contains("tokio::spawn"));
        assert!(!code.contains("spawn_blocking"));
        // A failure's source chain can carry server-supplied body text; logs use `loggable`.
        assert!(!code.contains("err:#"));
        assert!(!code.contains("{e:#}"));
    }
    assert!(budget.contains("gcs_typed_retry_verdict"));
}

/// The futures the ObjectStore wiring will await are `Send` (a compile-time check): the
/// receive loop, and both client entry points.
#[test]
fn the_receive_futures_are_send() {
    fn is_send<T: Send>(_: &T) {}
    fn client_futures(client: &GcsClient) {
        let mut buf = [0u8; 1];
        let progress = ReceiveProgress::new();
        is_send(&client.get_object_into("b", "o", &mut buf, &progress));
        is_send(&client.head_typed("b", "o"));
    }
    std::hint::black_box(client_futures as fn(&GcsClient));

    let fake = Fake::new(Vec::new(), 4);
    let mut buf = [0u8; 1];
    let progress = ReceiveProgress::new();
    let cfg = config(false);
    is_send(&receive_into(&fake, WHAT, &mut buf, &progress, &cfg));
}
