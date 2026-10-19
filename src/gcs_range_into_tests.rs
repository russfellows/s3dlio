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
    receive_span, ChunkStream, ObjectFacts, OpenRequest, Opened, ReadOpener, ReceiveConfig, SpanOf,
};
use super::*;
use crate::object_error::{object_error, ObjectErrorKind};
use crate::range_version::ObjectVersion;
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
    /// The object is replaced by this content and generation just BEFORE the n-th open (counting
    /// from 1) is answered, so open n and every later one see the new version.
    replace_at_open: Option<(usize, Vec<u8>, i64)>,
    /// The first open fails with a 403 (not retryable).
    fail_first_open: bool,
    /// `(start, limit, generation)` of every open.
    requests: Vec<(u64, Option<u64>, Option<i64>)>,
}

#[derive(Clone)]
struct Fake(Arc<Mutex<State>>);

impl Fake {
    fn new(data: Vec<u8>, chunk: usize) -> Self {
        Fake(Arc::new(Mutex::new(State {
            data,
            generation: GEN,
            chunk,
            break_first_after: None,
            replace_before_second: None,
            replace_at_open: None,
            fail_first_open: false,
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
        if st
            .replace_at_open
            .as_ref()
            .is_some_and(|(n, _, _)| *n == st.requests.len())
        {
            if let Some((_, data, generation)) = st.replace_at_open.take() {
                st.data = data;
                st.generation = generation;
            }
        }
        if st.fail_first_open && st.requests.len() == 1 {
            return Err(Gax::http(
                403,
                http::HeaderMap::new(),
                Bytes::from_static(b"denied"),
            ));
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
    // One range standing alone: its own version.
    let version = ObjectVersion::new();
    let span = SpanOf::Window {
        base,
        version: &version,
    };
    let result = receive_span(fake, WHAT, span, &mut buf[..n], &progress, &config()).await;
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

// ---- one object version per logical GET (Blue stage G review, item G-F2) -----------------------
//
// The range engine reads an object as several independent ranged requests. Each used to start
// unpinned, so replacing the object with another of the SAME length between two ranges gave a
// successful body that was part one version and part the other. These tests run the real range
// engine over the real ranged receive against a fake that replaces the object (new generation,
// same length) just before a chosen open, and inspect EVERY open's requested generation.

const GEN_B: i64 = 88;
const OBJECT_LEN: usize = 100;
const RANGE_LEN: usize = 25;

fn engine_config() -> crate::range_engine_generic::RangeEngineConfig {
    crate::range_engine_generic::RangeEngineConfig {
        chunk_size: RANGE_LEN,
        max_concurrent_ranges: 4,
        min_split_size: 1,
        range_timeout: Duration::from_secs(10),
    }
}

/// One logical GET of the whole fake object through the range engine, the way
/// `GcsObjectStore::get_with_range_engine` does it. `one_response` is the fallback for an object
/// with no generation: here it opens the object once, unranged, through the fake and drains it.
async fn logical_get(fake: &Fake) -> Result<Bytes> {
    let version = Arc::new(ObjectVersion::<i64>::new());
    let fill = {
        let fake = fake.clone();
        let version = Arc::clone(&version);
        move |offset: u64, mut window: bytes::BytesMut| {
            let fake = fake.clone();
            let version = Arc::clone(&version);
            async move {
                let progress = ReceiveProgress::new();
                let span = SpanOf::Window {
                    base: offset,
                    version: &version,
                };
                let written =
                    receive_span(&fake, WHAT, span, &mut window[..], &progress, &config()).await?;
                Ok((window, written))
            }
        }
    };
    let one_response = || async {
        let opened = fake
            .open(&OpenRequest {
                start: 0,
                limit: None,
                generation: None,
            })
            .await
            .map_err(|e| anyhow!("single response failed: {e}"))?;
        let mut stream = opened.stream;
        let mut body = Vec::new();
        while let Some(chunk) = stream.next_chunk().await {
            body.extend_from_slice(&chunk.map_err(|e| anyhow!("single response body: {e}"))?);
        }
        Ok(Bytes::from(body))
    };
    let (bytes, _) = crate::object_store::range_engine_download_versioned(
        engine_config(),
        OBJECT_LEN as u64,
        false,
        &version,
        fill,
        one_response,
    )
    .await?;
    Ok(bytes)
}

/// The result is an error or the complete bytes of exactly one of the two versions.
fn assert_not_mixed(result: &Result<Bytes>, a: &[u8], b: &[u8], fake: &Fake) {
    if let Ok(body) = result {
        assert!(
            body.as_ref() == a || body.as_ref() == b,
            "RED: a successful GET returned bytes of two versions; opens: {:?}",
            fake.requests()
        );
    }
}

/// The object is replaced by another of the same length and a new generation as the SECOND open
/// is answered: the first range saw version A, the others would see B. No success may mix them,
/// and every open after the first must name version A's generation.
#[tokio::test]
async fn a_same_size_replacement_between_ranges_is_never_a_mixed_body() {
    let a = pattern(OBJECT_LEN);
    let b: Vec<u8> = a.iter().map(|byte| byte.wrapping_add(1) % 200).collect();
    let fake = Fake::new(a.clone(), 7);
    fake.0.lock().expect("lock").replace_at_open = Some((2, b.clone(), GEN_B));

    let result = tokio::time::timeout(Duration::from_secs(20), logical_get(&fake))
        .await
        .expect("the logical GET finishes");
    assert_not_mixed(&result, &a, &b, &fake);

    let opens = fake.requests();
    assert!(opens.len() >= 2, "several ranges were requested: {opens:?}");
    assert_eq!(
        opens[0].2, None,
        "RED: the first range establishes the version and names none: {opens:?}"
    );
    for (i, open) in opens.iter().enumerate().skip(1) {
        assert_eq!(
            open.2,
            Some(GEN),
            "RED: open {} of a logical GET is not pinned to the first response's generation: {opens:?}",
            i + 1
        );
    }
    // The range engine reports a failed range as text, so the type is not available here (the
    // typed `VersionChanged` of a pinned range is asserted by the receive-loop tests).
    let err = result.expect_err("the pinned generation was replaced, so the read must fail");
    assert!(
        format!("{err:#}").contains("version changed"),
        "the error says the version changed: {err:#}"
    );
}

/// An unchanged object is read completely with one open per range, the first unpinned and every
/// other one pinned to the generation the first reported.
#[tokio::test]
async fn an_unchanged_object_pins_every_later_range_to_the_first_generation() {
    let a = pattern(OBJECT_LEN);
    let fake = Fake::new(a.clone(), 7);
    let got = logical_get(&fake).await.expect("read");
    assert_eq!(got.as_ref(), a.as_slice());
    let opens = fake.requests();
    assert_eq!(opens.len(), OBJECT_LEN / RANGE_LEN, "{opens:?}");
    assert_eq!(opens[0].2, None, "{opens:?}");
    assert!(
        opens[1..].iter().all(|open| open.2 == Some(GEN)),
        "{opens:?}"
    );
}

/// An object whose first response carries no generation cannot be pinned, and unpinned ranges are
/// never combined: the other ranges fail at once and the object is read again as ONE unranged
/// response. A replacement as the second open is answered gives the whole new object (the one
/// response), never a mix, and no sibling range is ever opened.
#[tokio::test]
async fn without_a_generation_the_object_is_read_as_one_response() {
    let a = pattern(OBJECT_LEN);
    let b: Vec<u8> = a.iter().map(|byte| byte.wrapping_add(1) % 200).collect();
    let fake = Fake::new(a.clone(), 7);
    {
        let mut st = fake.0.lock().expect("lock");
        st.generation = 0;
        st.replace_at_open = Some((2, b.clone(), 0));
    }

    let result = tokio::time::timeout(Duration::from_secs(20), logical_get(&fake))
        .await
        .expect("the logical GET finishes");
    assert_not_mixed(&result, &a, &b, &fake);

    let opens = fake.requests();
    assert_eq!(
        opens.first().map(|open| open.2),
        Some(None),
        "the first range names no generation: {opens:?}"
    );
    assert!(
        opens.iter().all(|open| open.2.is_none()),
        "there is no generation to name: {opens:?}"
    );
    assert_eq!(
        opens.last(),
        Some(&(0, None, None)),
        "RED: the object was not read again as one unranged response: {opens:?}"
    );
    assert_eq!(
        opens.len(),
        2,
        "the first range and the one whole-object response, no sibling range: {opens:?}"
    );
    assert_eq!(
        result.expect("the single response succeeds").as_ref(),
        b.as_slice(),
        "the single response is the object as it is now"
    );
}

/// A first range that fails (a 403, not retryable) releases the other ranges with its reason: the
/// read fails promptly, it does not wait for ever for a generation that will never arrive, and
/// no later range is opened.
#[tokio::test]
async fn a_failed_first_range_fails_the_read_without_hanging() {
    let fake = Fake::new(pattern(OBJECT_LEN), 7);
    fake.0.lock().expect("lock").fail_first_open = true;
    let result = tokio::time::timeout(Duration::from_secs(20), logical_get(&fake))
        .await
        .expect("RED: the read hung waiting for the first range");
    let err = result.expect_err("the first range failed");
    assert!(
        format!("{err:#}").contains("forbidden"),
        "the error carries the failed range's reason, whichever range reports it: {err:#}"
    );
    assert_eq!(
        fake.requests().len(),
        1,
        "no other range is opened after the first failed: {:?}",
        fake.requests()
    );
}

/// An object that has not been replaced is not affected by the pinning: the same bytes, and the
/// pinned requests are exactly the ones a standalone range would make.
#[tokio::test]
async fn a_single_range_standing_alone_is_unpinned_and_unchanged() {
    let data = pattern(OBJECT_LEN);
    let fake = Fake::new(data.clone(), 7);
    let out = run(&fake, 25, 25).await;
    assert_eq!(out.result.as_ref().expect("received"), &25);
    assert_eq!(&out.buf[..25], &data[25..50]);
    assert_eq!(fake.requests(), vec![(25, Some(25), None)]);
}
