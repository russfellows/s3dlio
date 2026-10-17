// tests/s3_get_into_support/mod.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Shared helpers of the caller-buffer GET tests (`tests/s3_get_into_*.rs`,
// `tests/s3_head_typed_*.rs`; plan docs/implementation-plans/v0.10.0-caller-buffer-get-plan.md
// §4.3 and §9).  Each test binary includes this with `#[path]` and uses a subset.
//
// Every test runs against the in-process mock S3 server and registers its OWN object under a
// unique key, so concurrent tests never see each other's traffic.  Nothing here reaches a real
// endpoint.

#[path = "../common/mock_s3_server.rs"]
#[allow(dead_code)] // the mock server serves several test binaries; each uses a subset
pub mod mock;

use mock::{key_path, MockS3State};
use s3dlio::object_error::{object_error, ObjectErrorKind};
use s3dlio::receive_progress::ReceiveProgress;
use s3dlio::s3_ops::S3Ops;
use std::sync::Once;

pub const BUCKET: &str = "gi-bucket";
/// The byte a destination is filled with before a read.
pub const FILL: u8 = 0xEE;
/// The byte of the guard region after the destination.
pub const CANARY: u8 = 0xA5;
/// How many guard bytes follow the destination.
pub const CANARY_LEN: usize = 64;

static INIT: Once = Once::new();

/// Start (once) the shared mock server and pin the knobs the assertions depend on.  Set before
/// any client or cache reads them, and unconditionally, so a value inherited from a developer
/// shell cannot change the outcome.  `extra` is applied in the same single pass.
pub fn init(extra: &[(&str, &str)]) -> (u16, MockS3State) {
    let server = mock::ensure_mock_s3_server();
    INIT.call_once(|| {
        for name in [
            "S3DLIO_SKIP_HEAD",
            "S3DLIO_CHUNK_SIZE",
            "S3DLIO_RANGE_CONCURRENCY",
            "S3DLIO_RANGE_THRESHOLD_MB",
            "S3DLIO_FOLLOW_REDIRECTS",
        ] {
            std::env::remove_var(name);
        }
        for (k, v) in [
            ("S3DLIO_MAX_RETRY_ATTEMPTS", "3"),
            ("S3DLIO_LOAD_DOTENV", "0"),
            ("S3DLIO_ENABLE_RANGE_OPTIMIZATION", "1"),
            ("AWS_EC2_METADATA_DISABLED", "true"),
        ] {
            std::env::set_var(k, v);
        }
        for (k, v) in extra {
            std::env::set_var(k, v);
        }
    });
    server
}

/// A client for the mock server on `port` (never the process-wide client).
pub async fn client_for(port: u16) -> aws_sdk_s3::Client {
    s3dlio::s3_client::create_s3_client_for_endpoint(&format!("http://127.0.0.1:{port}"), None)
        .await
        .expect("a client for the mock endpoint")
}

/// A handler on `client` that writes no op-log row.
pub fn ops(client: aws_sdk_s3::Client) -> S3Ops {
    S3Ops::new(client, None, "", "")
}

/// A unique key for one test.
pub fn key(tag: &str) -> String {
    mock::unique_test_key(&format!("gi-{tag}"))
}

/// `len` deterministic bytes that differ between positions and between seeds.
pub fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
        .collect()
}

/// The raw request path of `key` in [`BUCKET`].
pub fn path_of(key: &str) -> String {
    key_path(BUCKET, key)
}

/// A destination of `len` bytes (filled with [`FILL`]) followed by [`CANARY_LEN`] canary bytes,
/// all in ONE allocation, so a write past the destination is caught.
pub struct Dst {
    buf: Vec<u8>,
    len: usize,
}

impl Dst {
    pub fn new(len: usize) -> Self {
        let mut buf = vec![FILL; len];
        buf.extend(std::iter::repeat_n(CANARY, CANARY_LEN));
        Self { buf, len }
    }

    /// The destination slice handed to the read.
    pub fn slice(&mut self) -> &mut [u8] {
        &mut self.buf[..self.len]
    }

    /// The destination bytes.
    pub fn bytes(&self) -> &[u8] {
        &self.buf[..self.len]
    }

    /// Nothing past the destination was written.
    pub fn assert_canaries(&self) {
        assert!(
            self.buf[self.len..].iter().all(|b| *b == CANARY),
            "bytes past the destination were written"
        );
    }

    /// Nothing at all was written.
    pub fn assert_untouched(&self) {
        assert!(
            self.bytes().iter().all(|b| *b == FILL),
            "the destination was written: {:?}",
            self.bytes()
        );
        self.assert_canaries();
    }
}

/// The typed kind an error carries; panics when it carries none (every failure must be typed).
pub fn kind_of(err: &anyhow::Error) -> ObjectErrorKind {
    object_error(err)
        .unwrap_or_else(|| panic!("an untyped error: {err:#}"))
        .kind
}

/// Read `key` into `dst` through `ops`.
pub async fn read(
    ops: &S3Ops,
    key: &str,
    dst: &mut Dst,
    progress: &ReceiveProgress,
) -> anyhow::Result<()> {
    ops.get_into(BUCKET, key, dst.slice(), progress).await
}

/// The GET requests the server saw for `key`, in arrival order.
pub fn gets(state: &MockS3State, key: &str) -> Vec<mock::RecordedRequest> {
    state
        .requests_for(BUCKET, key)
        .into_iter()
        .filter(|r| r.method == "GET")
        .collect()
}

/// How many GETs the server saw for `key` that asked for exactly `range` (`bytes=a-b`).
pub fn gets_of_range(state: &MockS3State, key: &str, range: &str) -> usize {
    gets(state, key)
        .iter()
        .filter(|r| r.range.as_deref() == Some(range))
        .count()
}

/// The attempts one request (or one range part) gets: `S3DLIO_MAX_RETRY_ATTEMPTS`, pinned to 3
/// by [`init`], read back through the code under test so the tests follow its real default.
pub fn attempts() -> usize {
    let configured = s3dlio::constants::max_retry_attempts();
    usize::try_from(configured).expect("the attempt budget fits in usize")
}

/// The HEAD requests the server saw for `key`.
pub fn heads(state: &MockS3State, key: &str) -> usize {
    state
        .requests_for(BUCKET, key)
        .iter()
        .filter(|r| r.method == "HEAD")
        .count()
}

/// Drive `read` until `progress` reports at least `filled` positions, then DROP it.  The read
/// is expected to stall (a stalled body), so it finishing is a failure, and so is not reaching
/// `filled` within `bound`.  Returns with the read dropped and its borrows released.
pub async fn drive_then_drop<F: std::future::Future>(
    read: F,
    progress: &ReceiveProgress,
    filled: u64,
    bound: std::time::Duration,
) {
    let mut read = Box::pin(read);
    let watch = async {
        while progress.filled() < filled {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    };
    tokio::select! {
        _ = &mut read => panic!("the read finished; it should have stalled"),
        reached = tokio::time::timeout(bound, watch) => {
            reached.unwrap_or_else(|_| {
                panic!("fewer than {filled} positions within {bound:?} (filled {})", progress.filled())
            });
        }
    }
    drop(read);
}
