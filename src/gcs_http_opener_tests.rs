// src/gcs_http_opener_tests.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Tests of the REAL `HttpOpener` (plan §4.3 GCS row, §4.5 A2/A8, §9 "Providers -> GCS").
//
// The receive-loop tests in `gcs_get_into_tests.rs` use a fake opener, so they cannot notice if
// the real opener forgets `set_generation`, swaps the start and the limit of a range, or reads
// the size or generation from the wrong place.  These tests drive the production opener through
// the pinned SDK fork's `Storage` client against a local HTTP server that records every request
// it receives and answers with the headers the live JSON media API sends (measured on real GCS:
// `x-goog-generation`, `x-goog-stored-content-length` carrying the FULL object size on ranged
// responses too, `content-range` on 206).  Local only: loopback, anonymous credentials.
//
// The client is built with `NeverResume`, so the SDK's own read-resume layer does not hide a
// failed body from the receive loop: what these tests observe is the opener's and the loop's own
// requests.
//
// The Rapid opener has no counterpart here.  It speaks BidiReadObject over gRPC and the pinned
// fork has no local bidi server to point it at, so its `set_generation`, range mapping and
// descriptor facts are covered by the live Rapid gate of plan §8, not by a unit test.

use super::caller_buffer::{
    receive_into, ChunkStream, HttpOpener, OpenRequest, ReadOpener, ReceiveConfig,
};
use super::*;
use crate::object_error::{object_error, ObjectErrorKind};
use crate::receive_progress::ReceiveProgress;
use google_cloud_auth::credentials::anonymous::Builder as Anonymous;
use google_cloud_storage::read_resume_policy::NeverResume;
use std::collections::VecDeque;
use std::net::SocketAddr;
use std::sync::Mutex;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const GEN: i64 = 77;
const BUCKET: &str = "projects/_/buckets/bucket";
const OBJECT: &str = "dir/object";
const TOTAL: usize = 16;

/// The 16 bytes of the test object.
fn object_bytes() -> Vec<u8> {
    (0..TOTAL)
        .map(|i| u8::try_from(i * 7 + 3).expect("below 256"))
        .collect()
}

/// One scripted response.
struct Reply {
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    /// The `content-length` announced; more than `body.len()` makes the body end early.
    announced: usize,
    /// Keep the connection open after the body, sending nothing more.
    hold_open: bool,
}

impl Reply {
    fn new(status: u16, body: &[u8]) -> Self {
        Reply {
            status,
            headers: Vec::new(),
            body: body.to_vec(),
            announced: body.len(),
            hold_open: false,
        }
    }

    fn header(mut self, name: &str, value: impl ToString) -> Self {
        self.headers.push((name.to_string(), value.to_string()));
        self
    }

    /// An unranged 200 for the whole object, as the live media API answers it.
    fn whole(generation: i64, body: &[u8]) -> Self {
        Reply::new(200, body)
            .header("x-goog-generation", generation)
            .header("x-goog-stored-content-length", TOTAL)
    }

    /// A ranged 206 starting at `start`: `x-goog-stored-content-length` is still the FULL size.
    fn ranged(generation: i64, start: usize, body: &[u8]) -> Self {
        let end = start + body.len() - 1;
        Reply::new(206, body)
            .header("x-goog-generation", generation)
            .header("x-goog-stored-content-length", TOTAL)
            .header("content-range", format!("bytes {start}-{end}/{TOTAL}"))
    }

    /// Announce `announced` bytes but send only the body, then close (or hold open).
    fn announcing(mut self, announced: usize) -> Self {
        self.announced = announced;
        self
    }

    fn held_open(mut self) -> Self {
        self.hold_open = true;
        self
    }
}

/// What the server saw of one request.
#[derive(Clone, Debug)]
struct Seen {
    target: String,
    headers: Vec<(String, String)>,
}

impl Seen {
    fn path(&self) -> &str {
        self.target.split('?').next().unwrap_or("")
    }

    fn query(&self, name: &str) -> Option<String> {
        let query = self.target.split_once('?')?.1;
        query
            .split('&')
            .filter_map(|pair| pair.split_once('='))
            .find(|(key, _)| *key == name)
            .map(|(_, value)| value.to_string())
    }

    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key == name)
            .map(|(_, value)| value.as_str())
    }
}

/// A loopback HTTP/1.1 server that answers scripted replies in order (a request with no reply
/// left gets 500) and records what it was asked.  Every response closes its connection.
struct Server {
    addr: SocketAddr,
    replies: std::sync::Arc<Mutex<VecDeque<Reply>>>,
    seen: std::sync::Arc<Mutex<Vec<Seen>>>,
    accept: tokio::task::JoinHandle<()>,
}

impl Server {
    async fn start(replies: impl IntoIterator<Item = Reply>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
        let addr = listener.local_addr().expect("address");
        let replies = std::sync::Arc::new(Mutex::new(replies.into_iter().collect::<VecDeque<_>>()));
        let seen = std::sync::Arc::new(Mutex::new(Vec::new()));
        let accept = {
            let (replies, seen) = (replies.clone(), seen.clone());
            tokio::spawn(async move {
                while let Ok((socket, _)) = listener.accept().await {
                    tokio::spawn(answer(socket, replies.clone(), seen.clone()));
                }
            })
        };
        Server {
            addr,
            replies,
            seen,
            accept,
        }
    }

    fn requests(&self) -> Vec<Seen> {
        self.seen.lock().expect("lock").clone()
    }

    fn unused_replies(&self) -> usize {
        self.replies.lock().expect("lock").len()
    }

    async fn client(&self) -> Storage {
        Storage::builder()
            .with_endpoint(format!("http://{}", self.addr))
            .with_credentials(Anonymous::new().build())
            .with_read_resume_policy(NeverResume)
            .build()
            .await
            .expect("client")
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        self.accept.abort();
    }
}

async fn answer(
    mut socket: TcpStream,
    replies: std::sync::Arc<Mutex<VecDeque<Reply>>>,
    seen: std::sync::Arc<Mutex<Vec<Seen>>>,
) {
    let mut request = Vec::new();
    let mut chunk = [0u8; 2048];
    while !request.windows(4).any(|w| w == b"\r\n\r\n") {
        match socket.read(&mut chunk).await {
            Ok(0) | Err(_) => return,
            Ok(n) => request.extend_from_slice(&chunk[..n]),
        }
    }
    let text = String::from_utf8_lossy(&request).to_string();
    let mut lines = text.split("\r\n");
    let target = lines
        .next()
        .and_then(|line| line.split(' ').nth(1))
        .unwrap_or("")
        .to_string();
    let headers = lines
        .take_while(|line| !line.is_empty())
        .filter_map(|line| line.split_once(':'))
        .map(|(name, value)| (name.trim().to_ascii_lowercase(), value.trim().to_string()))
        .collect();
    seen.lock().expect("lock").push(Seen { target, headers });

    let reply = replies
        .lock()
        .expect("lock")
        .pop_front()
        .unwrap_or_else(|| Reply::new(500, b"no reply scripted"));
    let mut head = format!("HTTP/1.1 {} Scripted\r\n", reply.status);
    for (name, value) in &reply.headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    head.push_str(&format!(
        "content-length: {}\r\nconnection: close\r\n\r\n",
        reply.announced
    ));
    if socket.write_all(head.as_bytes()).await.is_err()
        || socket.write_all(&reply.body).await.is_err()
        || socket.flush().await.is_err()
    {
        return;
    }
    if reply.hold_open {
        std::future::pending::<()>().await;
    }
}

fn opener(storage: &Storage) -> HttpOpener<'_> {
    HttpOpener {
        storage,
        bucket_name: BUCKET,
        object: OBJECT,
    }
}

async fn drain(stream: &mut dyn ChunkStream) -> Vec<u8> {
    let mut all = Vec::new();
    while let Some(chunk) = stream.next_chunk().await {
        all.extend_from_slice(&chunk.expect("chunk"));
    }
    all
}

fn config() -> ReceiveConfig {
    ReceiveConfig {
        rapid: false,
        chunk_timeout: None,
        retry_delay: |_| Duration::ZERO,
        max_open_attempts: 3,
        max_repairs: 4,
    }
}

// ---------------------------------------------------------------------------
// The request the opener builds, and the facts it reads back
// ---------------------------------------------------------------------------

/// An unpinned, unranged open names no generation and sends no `Range`, and reads the size and
/// the generation from the response headers.
#[tokio::test]
async fn an_unpinned_full_open_sends_no_generation_and_no_range() {
    let data = object_bytes();
    let server = Server::start([Reply::whole(GEN, &data)]).await;
    let storage = server.client().await;
    let mut opened = opener(&storage)
        .open(&OpenRequest {
            start: 0,
            limit: None,
            generation: None,
        })
        .await
        .map_err(|e| e.to_string())
        .expect("opened");

    assert_eq!(opened.facts.size, i64::try_from(TOTAL).expect("size"));
    assert_eq!(opened.facts.generation, GEN);
    assert_eq!(drain(opened.stream.as_mut()).await, data);

    let seen = server.requests();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].path(), "/storage/v1/b/bucket/o/dir%2Fobject");
    assert_eq!(seen[0].query("alt").as_deref(), Some("media"));
    assert_eq!(seen[0].query("generation"), None, "nothing to pin yet");
    assert_eq!(seen[0].header("range"), None, "a full read is unranged");
}

/// A pinned bounded open (a repair of the suffix) carries the generation as a query parameter
/// and the range as `bytes=start-last`.  The 206 reports the FULL size in
/// `x-goog-stored-content-length`, which is what the size check of a repair compares.
#[tokio::test]
async fn a_pinned_bounded_open_sends_the_generation_and_the_exact_range() {
    let data = object_bytes();
    let server = Server::start([Reply::ranged(GEN, 6, &data[6..])]).await;
    let storage = server.client().await;
    let mut opened = opener(&storage)
        .open(&OpenRequest {
            start: 6,
            limit: Some(10),
            generation: Some(GEN),
        })
        .await
        .map_err(|e| e.to_string())
        .expect("opened");

    assert_eq!(
        opened.facts.size,
        i64::try_from(TOTAL).expect("size"),
        "the whole object's size, not the length of the range"
    );
    assert_eq!(opened.facts.generation, GEN);
    assert_eq!(drain(opened.stream.as_mut()).await, &data[6..]);

    let seen = server.requests();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].query("generation").as_deref(), Some("77"));
    assert_eq!(
        seen[0].header("range"),
        Some("bytes=6-15"),
        "start 6, limit 10: the range is start-(start+limit-1)"
    );
}

/// A pinned open with no limit reads to the end of the object: `bytes=start-`.
#[tokio::test]
async fn a_pinned_open_without_a_limit_reads_to_the_end() {
    let data = object_bytes();
    let server = Server::start([Reply::ranged(GEN, 10, &data[10..])]).await;
    let storage = server.client().await;
    let mut opened = opener(&storage)
        .open(&OpenRequest {
            start: 10,
            limit: None,
            generation: Some(GEN),
        })
        .await
        .map_err(|e| e.to_string())
        .expect("opened");
    assert_eq!(opened.facts.size, i64::try_from(TOTAL).expect("size"));
    assert_eq!(drain(opened.stream.as_mut()).await, &data[10..]);

    let seen = server.requests();
    assert_eq!(seen[0].query("generation").as_deref(), Some("77"));
    assert_eq!(seen[0].header("range"), Some("bytes=10-"));
}

/// The HTTP route has no descriptor: asking for one is a typed, non-retryable error and sends
/// no request at all (it must not download the object to answer).
#[tokio::test]
async fn the_http_route_has_no_descriptor_and_makes_no_request() {
    let server = Server::start([Reply::whole(GEN, &object_bytes())]).await;
    let storage = server.client().await;
    let err = opener(&storage)
        .describe(None)
        .await
        .expect_err("no descriptor on the HTTP route");

    assert!(server.requests().is_empty(), "describe must not read data");
    assert_eq!(server.unused_replies(), 1);
    let classified = attach_gcs_classification(anyhow::Error::new(err), false);
    assert!(
        !gcs_typed_retry_verdict(&classified),
        "a missing descriptor is never retried"
    );
}

// ---------------------------------------------------------------------------
// The real opener inside the receive loop
// ---------------------------------------------------------------------------

/// A body that ends early is repaired from the cursor: the first request is unpinned and
/// unranged; the repair names the first response's generation and asks for exactly the missing
/// suffix, and the object arrives intact and is counted once.
#[tokio::test]
async fn a_broken_body_is_repaired_with_the_pinned_generation_and_the_suffix_range() {
    let data = object_bytes();
    let server = Server::start([
        Reply::whole(GEN, &data[..6]).announcing(TOTAL),
        Reply::ranged(GEN, 6, &data[6..]),
    ])
    .await;
    let storage = server.client().await;
    let mut dst = vec![0u8; TOTAL];
    let progress = ReceiveProgress::new();
    receive_into(
        &opener(&storage),
        "gs://bucket/dir/object",
        &mut dst,
        &progress,
        &config(),
    )
    .await
    .map_err(|e| format!("{e:#}"))
    .expect("repaired");

    assert_eq!(dst, data);
    assert_eq!(progress.filled(), 16);
    let seen = server.requests();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].query("generation"), None);
    assert_eq!(seen[0].header("range"), None);
    assert_eq!(seen[1].query("generation").as_deref(), Some("77"));
    assert_eq!(seen[1].header("range"), Some("bytes=6-15"));
}

/// A body that stalls is a typed timeout (Transport) and is repaired the same way: the HTTP
/// route has a chunk timeout.
#[tokio::test]
async fn a_stalled_body_times_out_and_is_repaired() {
    let data = object_bytes();
    let server = Server::start([
        Reply::whole(GEN, &data[..4]).announcing(TOTAL).held_open(),
        Reply::ranged(GEN, 4, &data[4..]),
    ])
    .await;
    let storage = server.client().await;
    let mut cfg = config();
    cfg.chunk_timeout = Some(Duration::from_millis(300));
    let mut dst = vec![0u8; TOTAL];
    let progress = ReceiveProgress::new();
    receive_into(
        &opener(&storage),
        "gs://bucket/dir/object",
        &mut dst,
        &progress,
        &cfg,
    )
    .await
    .map_err(|e| format!("{e:#}"))
    .expect("repaired after the stall");

    assert_eq!(dst, data);
    let seen = server.requests();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[1].query("generation").as_deref(), Some("77"));
    assert_eq!(seen[1].header("range"), Some("bytes=4-15"));
}

/// The timeout is what ends a stall, and it is a Transport failure: with no repair allowed the
/// read fails with the typed error after one request (it does not wait forever).
#[tokio::test]
async fn a_stall_with_no_repair_left_fails_as_transport() {
    let data = object_bytes();
    let server = Server::start([Reply::whole(GEN, &data[..4]).announcing(TOTAL).held_open()]).await;
    let storage = server.client().await;
    let mut cfg = config();
    cfg.chunk_timeout = Some(Duration::from_millis(300));
    cfg.max_repairs = 0;
    let mut dst = vec![0u8; TOTAL];
    let progress = ReceiveProgress::new();
    let err = receive_into(
        &opener(&storage),
        "gs://bucket/dir/object",
        &mut dst,
        &progress,
        &cfg,
    )
    .await
    .expect_err("a stall");
    assert_eq!(
        object_error(&err).expect("typed").kind,
        ObjectErrorKind::Transport
    );
    assert_eq!(server.requests().len(), 1);
    assert_eq!(progress.filled(), 4);
}

/// A pinned repair that finds the object gone (404) is a version change, never absence, over
/// the real SDK error mapping.
#[tokio::test]
async fn a_404_on_a_pinned_repair_is_a_version_change() {
    let data = object_bytes();
    let server = Server::start([
        Reply::whole(GEN, &data[..6]).announcing(TOTAL),
        Reply::new(404, b"No such object"),
    ])
    .await;
    let storage = server.client().await;
    let mut dst = vec![0u8; TOTAL];
    let progress = ReceiveProgress::new();
    let err = receive_into(
        &opener(&storage),
        "gs://bucket/dir/object",
        &mut dst,
        &progress,
        &config(),
    )
    .await
    .expect_err("the object is gone");
    assert_eq!(
        object_error(&err).expect("typed").kind,
        ObjectErrorKind::VersionChanged
    );
    assert!(!crate::delete_contract::is_not_found(&err));
    assert_eq!(progress.filled(), 6);
}

// ---------------------------------------------------------------------------
// Literal object names: the URL path segment the opener builds
// ---------------------------------------------------------------------------

/// An opener for the literal object name `object`.
fn opener_for<'a>(storage: &'a Storage, object: &'a str) -> HttpOpener<'a> {
    HttpOpener {
        storage,
        bucket_name: BUCKET,
        object,
    }
}

/// The path of the single request an unpinned full open of `object` sends.
async fn path_sent_for(object: &str) -> String {
    let data = object_bytes();
    let server = Server::start([Reply::whole(GEN, &data)]).await;
    let storage = server.client().await;
    let mut opened = opener_for(&storage, object)
        .open(&OpenRequest {
            start: 0,
            limit: None,
            generation: None,
        })
        .await
        .map_err(|e| e.to_string())
        .expect("opened");
    assert_eq!(drain(opened.stream.as_mut()).await, data);
    let seen = server.requests();
    assert_eq!(seen.len(), 1, "one request for {object:?}");
    seen[0].path().to_string()
}

/// The pinned fork's URL encoder leaves `%` alone, so a literal `%2F` in an object name would
/// reach GCS as an escape and be decoded to `/` (a different object).  The opener therefore
/// sends every literal `%` as `%25`, and the fork encodes the rest.  Each row is
/// (literal object name, the path segment after `/o/` that must reach the server).
#[tokio::test]
async fn literal_object_names_reach_the_server_as_the_exact_path_segment() {
    let rows = [
        ("a%2Fb", "a%252Fb"),
        ("100%", "100%25"),
        ("%", "%25"),
        ("%25", "%2525"),
        ("x@y", "x%40y"),
        ("k=v", "k%3Dv"),
        ("p/q/r", "p%2Fq%2Fr"),
        ("name.", "name."),
        ("a b", "a%20b"),
        ("a\\b", "a%5Cb"),
        ("a\\%b", "a%5C%25b"),
        ("dir/a%2Fb c=d@e%", "dir%2Fa%252Fb%20c%3Dd%40e%25"),
    ];
    let mut wrong = Vec::new();
    for (literal, segment) in rows {
        let want = format!("/storage/v1/b/bucket/o/{segment}");
        let got = path_sent_for(literal).await;
        if got != want {
            wrong.push(format!("{literal:?}: sent {got}, wanted {want}"));
        }
    }
    assert!(wrong.is_empty(), "wrong paths:\n{}", wrong.join("\n"));
}

/// The escape is applied to the name on every open, so a repair (pinned, ranged) of a name with
/// a `%` names the same object as the first request.
#[tokio::test]
async fn a_repair_open_of_a_percent_name_names_the_same_object() {
    let data = object_bytes();
    let server = Server::start([Reply::ranged(GEN, 6, &data[6..])]).await;
    let storage = server.client().await;
    let mut opened = opener_for(&storage, "a%2Fb")
        .open(&OpenRequest {
            start: 6,
            limit: None,
            generation: Some(GEN),
        })
        .await
        .map_err(|e| e.to_string())
        .expect("opened");
    assert_eq!(drain(opened.stream.as_mut()).await, &data[6..]);
    let seen = server.requests();
    assert_eq!(seen[0].path(), "/storage/v1/b/bucket/o/a%252Fb");
    assert_eq!(seen[0].query("generation").as_deref(), Some("77"));
}

/// The Rapid route puts the object name in a protobuf field, not in a URL, so it must receive the
/// literal name unchanged: the escape belongs to `HttpOpener::open` alone.  No local server can
/// drive the Rapid opener (see the header), so this reads the source: the escape helper is called
/// only inside the HTTP opener, and the Rapid opener and the Rapid branch of `get_object_into`
/// hand `object` to `open_object` untouched.
#[test]
fn the_rapid_route_receives_the_literal_name_unchanged() {
    let src = include_str!("google_gcs_client.rs");
    let rapid_at = src
        .find("impl ReadOpener for RapidOpener")
        .expect("Rapid opener impl");
    let rapid_end = src[rapid_at..]
        .find("impl GcsClient")
        .expect("end of Rapid impl")
        + rapid_at;
    let rapid = &src[rapid_at..rapid_end];
    assert!(rapid.contains("open_object(self.bucket_name, self.object)"));
    assert!(!rapid.contains("http_media_object_name"), "{rapid}");

    let into_at = src
        .find("pub async fn get_object_into")
        .expect("get_object_into");
    let into = &src[into_at..];
    let into = &into[..into.find("pub async fn head_typed").expect("end")];
    let rapid_branch = &into[into.find("if rapid {").expect("rapid branch")..];
    let rapid_branch = &rapid_branch[..rapid_branch.find("} else {").expect("else")];
    assert!(
        rapid_branch.contains("            object,\n"),
        "{rapid_branch}"
    );
    assert!(
        !rapid_branch.contains("http_media_object_name"),
        "{rapid_branch}"
    );

    // The helper has exactly one caller outside its definition and tests: the HTTP opener.
    let before_tests = &src[..src.find("#[cfg(test)]\nmod tests").expect("tests")];
    let calls: Vec<_> = before_tests
        .match_indices("http_media_object_name(")
        .collect();
    assert_eq!(calls.len(), 2, "one definition and one call");
    let http_at = before_tests
        .find("impl ReadOpener for HttpOpener")
        .expect("HTTP opener impl");
    assert!(calls[1].0 > http_at && calls[1].0 < rapid_at);
}
