// src/gcs_range_into_tests.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Tests of the GCS ranged receive (`receive_span` with `ranged == true`, stage G R3): the range
// engine receives each range straight into its window of one master buffer, through the same
// receive loop as the caller-buffer GET. A small deterministic fake serves ranges and
// generations the way the service does; failures are the real typed `google_cloud_gax` errors.
// Local only: no network, no credentials.

use super::caller_buffer::{
    receive_span, ChunkStream, ObjectFacts, OpenRequest, Opened, ReadOpener, ReceiveConfig,
};
use super::*;
use crate::object_error::{object_error, ObjectErrorKind};
use crate::receive_progress::ReceiveProgress;
use async_trait::async_trait;
use google_cloud_gax::error::Error as Gax;
use google_cloud_storage::model::Object;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

const FILL: u8 = 0xAB;
const CANARY: usize = 16;
const GEN: i64 = 77;
const WHAT: &str = "gs://bucket/object";

struct State {
    data: Vec<u8>,
    generation: i64,
    chunk: usize,
    /// The first open serves this many bytes of its window and then fails with a transport error.
    break_first_after: Option<usize>,
    /// The object is replaced (new size, same generation reported) before the second open.
    replace_before_second: Option<Vec<u8>>,
    /// `(start, limit, generation)` of every open.
    requests: Vec<(u64, Option<u64>, Option<i64>)>,
}

struct Fake(Arc<Mutex<State>>);

impl Fake {
    fn new(data: Vec<u8>, chunk: usize) -> Self {
        Fake(Arc::new(Mutex::new(State {
            data,
            generation: GEN,
            chunk,
            break_first_after: None,
            replace_before_second: None,
            requests: Vec::new(),
        })))
    }

    fn requests(&self) -> Vec<(u64, Option<u64>, Option<i64>)> {
        self.0.lock().expect("lock").requests.clone()
    }
}

struct FakeStream {
    chunks: VecDeque<Bytes>,
    fail_at_end: bool,
}

#[async_trait]
impl ChunkStream for FakeStream {
    async fn next_chunk(&mut self) -> Option<Result<Bytes, Gax>> {
        if let Some(chunk) = self.chunks.pop_front() {
            return Some(Ok(chunk));
        }
        if self.fail_at_end {
            self.fail_at_end = false;
            return Some(Err(Gax::io("connection reset")));
        }
        None
    }
}

#[async_trait]
impl ReadOpener for Fake {
    async fn open(&self, request: &OpenRequest) -> Result<Opened, Gax> {
        let mut st = self.0.lock().expect("lock");
        st.requests
            .push((request.start, request.limit, request.generation));
        if st.requests.len() == 2 {
            if let Some(data) = st.replace_before_second.take() {
                st.data = data;
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
        let mut body = st.data[start..end].to_vec();
        let mut fail_at_end = false;
        if st.requests.len() == 1 {
            if let Some(n) = st.break_first_after {
                body.truncate(n);
                fail_at_end = true;
            }
        }
        let mut chunks = VecDeque::new();
        while !body.is_empty() {
            let rest = body.split_off(st.chunk.min(body.len()));
            chunks.push_back(Bytes::from(std::mem::replace(&mut body, rest)));
        }
        Ok(Opened {
            facts: ObjectFacts {
                size: i64::try_from(st.data.len()).expect("size"),
                generation: st.generation,
            },
            stream: Box::new(FakeStream {
                chunks,
                fail_at_end,
            }),
        })
    }

    async fn describe(&self, _generation: Option<i64>) -> Result<Object, Gax> {
        let st = self.0.lock().expect("lock");
        Ok(Object::default()
            .set_size(i64::try_from(st.data.len()).expect("size"))
            .set_generation(st.generation))
    }
}

fn config() -> ReceiveConfig {
    ReceiveConfig {
        rapid: false,
        chunk_timeout: None,
        retry_delay: |_| Duration::ZERO,
        max_open_attempts: 3,
        max_repairs: 4,
        max_total_repairs: 16,
    }
}

/// Bytes that never equal the fill value.
fn pattern(n: usize) -> Vec<u8> {
    (0..n)
        .map(|i| u8::try_from((i * 7 + 3) % 200).expect("below 200"))
        .collect()
}

struct Outcome {
    result: Result<usize>,
    /// The window followed by a canary tail, both pre-filled with `FILL`.
    buf: Vec<u8>,
    n: usize,
    progress: ReceiveProgress,
}

async fn run(fake: &Fake, base: u64, n: usize) -> Outcome {
    let mut buf = vec![FILL; n + CANARY];
    let progress = ReceiveProgress::new();
    let result = receive_span(fake, WHAT, base, true, &mut buf[..n], &progress, &config()).await;
    Outcome {
        result,
        buf,
        n,
        progress,
    }
}

impl Outcome {
    fn assert_canary_intact(&self) {
        assert!(
            self.buf[self.n..].iter().all(|b| *b == FILL),
            "a write went beyond the window"
        );
    }
}

/// A window in the middle of the object is received whole, with its own start and limit.
#[tokio::test]
async fn a_window_in_the_middle_is_received_at_its_offset() {
    let data = pattern(100);
    let fake = Fake::new(data.clone(), 7);
    let out = run(&fake, 30, 40).await;
    assert_eq!(out.result.as_ref().expect("received"), &40);
    assert_eq!(&out.buf[..40], &data[30..70]);
    assert_eq!(out.progress.filled(), 40);
    assert_eq!(fake.requests(), vec![(30, Some(40), None)]);
    out.assert_canary_intact();
}

/// A window that runs past the end of the object gets the bytes that exist, and the count says
/// how many: the caller decides whether that is an error (the range engine's tail probe).
#[tokio::test]
async fn a_window_past_the_end_is_clamped_and_the_count_is_returned() {
    let data = pattern(100);
    let fake = Fake::new(data.clone(), 8);
    let out = run(&fake, 95, 10).await;
    assert_eq!(out.result.as_ref().expect("received"), &5);
    assert_eq!(&out.buf[..5], &data[95..]);
    assert!(
        out.buf[5..10].iter().all(|b| *b == FILL),
        "only 5 bytes written"
    );
    assert_eq!(out.progress.filled(), 5);
    out.assert_canary_intact();
}

/// A window that starts past the end of the object holds nothing: a length mismatch.
#[tokio::test]
async fn a_window_starting_past_the_end_is_a_length_mismatch() {
    let fake = Fake::new(pattern(100), 8);
    let out = run(&fake, 101, 10).await;
    let err = out.result.as_ref().expect_err("a failure");
    let kind = object_error(err).expect("typed").kind;
    assert!(
        matches!(kind, ObjectErrorKind::LengthMismatch { .. }),
        "{kind:?}"
    );
    assert!(out.buf[..10].iter().all(|b| *b == FILL), "nothing written");
    out.assert_canary_intact();
}

/// A body that breaks part-way is repaired from the cursor, pinned to the first generation, and
/// every position is counted once.
#[tokio::test]
async fn a_broken_body_is_repaired_inside_the_window_and_pinned() {
    let data = pattern(100);
    let fake = Fake::new(data.clone(), 4);
    fake.0.lock().expect("lock").break_first_after = Some(12);
    let out = run(&fake, 20, 30).await;
    assert_eq!(out.result.as_ref().expect("repaired"), &30);
    assert_eq!(&out.buf[..30], &data[20..50]);
    assert_eq!(out.progress.filled(), 30);
    assert_eq!(
        fake.requests(),
        vec![(20, Some(30), None), (32, Some(18), Some(GEN))]
    );
    out.assert_canary_intact();
}

/// A repair that finds the object a different size is a version change, never combined.
#[tokio::test]
async fn a_size_change_between_attempts_is_a_version_change() {
    let fake = Fake::new(pattern(100), 4);
    {
        let mut st = fake.0.lock().expect("lock");
        st.break_first_after = Some(12);
        st.replace_before_second = Some(pattern(120));
    }
    let out = run(&fake, 20, 30).await;
    let err = out.result.as_ref().expect_err("a failure");
    let kind = object_error(err).expect("typed").kind;
    assert!(matches!(kind, ObjectErrorKind::VersionChanged), "{kind:?}");
    out.assert_canary_intact();
}

/// A zero-length window needs no request at all.
#[tokio::test]
async fn an_empty_window_makes_no_request() {
    let fake = Fake::new(pattern(100), 4);
    let out = run(&fake, 10, 0).await;
    assert_eq!(out.result.as_ref().expect("empty"), &0);
    assert!(fake.requests().is_empty());
}

/// A window that starts exactly AT the end of the object holds nothing either: a length
/// mismatch, as `get_object_range_into` documents (it used to return `Ok(0)`).
#[tokio::test]
async fn a_window_starting_at_the_end_is_a_length_mismatch() {
    let fake = Fake::new(pattern(100), 8);
    let out = run(&fake, 100, 10).await;
    let err = out
        .result
        .as_ref()
        .expect_err("a window at the end must fail");
    let kind = object_error(err).expect("typed").kind;
    assert!(
        matches!(kind, ObjectErrorKind::LengthMismatch { .. }),
        "{kind:?}"
    );
    out.assert_canary_intact();
}
