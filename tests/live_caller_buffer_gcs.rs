// tests/live_caller_buffer_gcs.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// LIVE tests of the caller-buffer GET (`ObjectStore::get_into`) and the typed HEAD
// (`ObjectStore::head`) against a real service (v0.10.0 plan, sections 8 and 9, #264).
//
// EVERY test here is `#[ignore]`d and also returns early (with a skip message on stderr) when
// its environment is missing, so `cargo test` never reaches a service.  Run them explicitly:
//
//   cargo test --test live_caller_buffer_gcs --features full-backends -- \
//       --ignored --test-threads=1 --nocapture
//
// Configuration comes ONLY from the environment (no bucket, project, account or host name is
// written in this file):
//
//   S3DLIO_LIVE_BUCKET   the bucket (the same name is used for both groups)
//   S3DLIO_LIVE_PREFIX   a prefix the runner chooses, unique to this run (for example one that
//                        carries a timestamp).  Every object lives under
//                        `<PREFIX>/<scheme>/<case>/`; nothing outside it is touched.
//
// Two groups run the same cases, one module each:
//
//   `s3_path`     `s3://$BUCKET/$PREFIX/s3/...`   The runner points the S3 client at an
//                 S3-compatible endpoint (GCS S3 interop): AWS_ENDPOINT_URL, AWS_ACCESS_KEY_ID,
//                 AWS_SECRET_ACCESS_KEY and AWS_REGION (and, to keep a stray `.env` out of the
//                 run, S3DLIO_LOAD_DOTENV=0).
//   `native_gcs`  `gs://$BUCKET/$PREFIX/gs/...`   The official GCS client with Application
//                 Default Credentials.  Compiled only with `backend-gcs` and without
//                 `gcs-community`.
//
// Range fan-out (S3 path).  The range threshold, chunk size and concurrency are read ONCE per
// process (`OnceLock`-latched), so a test cannot change them for itself.  The simplest correct
// approach is that the RUNNER sets them for the whole process, which then covers both routes in
// one run: objects below 1 MiB take the single unranged GET, objects at or above it take the
// ranged fan-out with a fresh HEAD and `If-Match` on every range:
//
//   S3DLIO_RANGE_THRESHOLD_MB=1  S3DLIO_CHUNK_SIZE=524288  S3DLIO_RANGE_CONCURRENCY=4
//
// This file never sets an environment variable.  The S3 test of the object that must take the
// fan-out (`large_object_round_trip`) reads the threshold and SKIPS, saying so, unless it is 1
// or less: without it the test would pass without ever leaving the single-GET route.  The 1 MiB
// + 7 byte object of `sizes_round_trip` takes the fan-out under the same setting (three ranges,
// the last one 7 bytes).
//
// Rules this file follows (plan section 8):
//   * deterministic, unique contents: every object's bytes are generated from its full URI, so
//     two objects never share contents and a read of the wrong object cannot compare equal;
//   * an isolated prefix, and no listing: objects are found only by their exact keys;
//   * independent readback: every body is read back through the allocating `get` as well as
//     compared with the source bytes;
//   * cleanup: each case runs under a guard that deletes every key it wrote (exact keys,
//     registered BEFORE the PUT) even when an assertion fails, deletes the first key a second
//     time (an already-absent DELETE succeeds), and finally asserts each key HEADs as absent.
//
// Destination contract asserted after a failure: the bytes past the destination slice (the
// canaries) are never written, `progress.filled()` never exceeds the slice, and the error is
// typed.  On the S3 path a size known before the body is read (single GET Content-Length, or
// the fan-out HEAD) also means NOTHING was written, so the destination is asserted untouched;
// for native GCS the prefix state is printed, not asserted.
//
// Op-log (S3 path only): `s3_path::op_log_one_row_per_call` initialises the op-log, which is
// process-global and cannot be restarted, to a temporary file, builds its own store with the
// logger, finalises it at the end and counts rows.  No other test in this file builds a logged
// store, so they are unaffected.  If another op-log session is already active in the process
// the test says so and skips.  The runner may add S3DLIO_OPLOG_LOSSLESS=1 so a full queue
// waits instead of dropping a row.
//
// Size.  Each group stores about 3 MiB and reads about 8 MiB at most (see the report that came
// with this file for the per-test request counts).  Run with `--test-threads=1`.

use anyhow::Result;
use bytes::Bytes;
use futures::FutureExt;
use s3dlio::delete_contract::delete_exact_object;
use s3dlio::object_error::{object_error, ObjectErrorKind};
use s3dlio::object_store::{
    store_for_uri, store_for_uri_with_logger, HeadMetadata, ObjectStore, ReceiveProgress,
};
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

const ENV_BUCKET: &str = "S3DLIO_LIVE_BUCKET";
const ENV_PREFIX: &str = "S3DLIO_LIVE_PREFIX";

const MIB: usize = 1024 * 1024;
/// Just above 1 MiB, so it is not a multiple of any chunk size.
const MID_LEN: usize = MIB + 7;
/// Above two chunks of 512 KiB and above the 1 MiB threshold: several ranges on the fan-out.
const LARGE_LEN: usize = 2 * MIB + 5;
/// The sizes of `sizes_round_trip`.
const SIZES: [usize; 4] = [0, 1, 16, MID_LEN];

/// The byte a destination holds before a read.
const FILL: u8 = 0xEE;
/// The byte of the guard region after a destination.
const CANARY: u8 = 0xA5;
/// How many guard bytes follow a destination.
const CANARY_LEN: usize = 64;

// -------------------------------------------------------------------------------------------
// Configuration and the shared runtime
// -------------------------------------------------------------------------------------------

struct Config {
    scheme: &'static str,
    bucket: String,
    prefix: String,
}

impl Config {
    /// The configuration, or `None` (after saying why on stderr) when the environment is
    /// incomplete.  A prefix is trimmed of slashes; an empty one is refused, so a run can
    /// never write at the bucket root.
    fn from_env(scheme: &'static str, case: &str) -> Option<Self> {
        let get = |name: &str| {
            std::env::var(name)
                .ok()
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        };
        let (Some(bucket), Some(prefix)) = (get(ENV_BUCKET), get(ENV_PREFIX)) else {
            eprintln!("SKIP {scheme}/{case}: {ENV_BUCKET} and {ENV_PREFIX} must both be set");
            return None;
        };
        let prefix = prefix.trim_matches('/').to_string();
        if prefix.is_empty() {
            eprintln!("SKIP {scheme}/{case}: {ENV_PREFIX} must name a non-empty prefix");
            return None;
        }
        Some(Self {
            scheme,
            bucket,
            prefix,
        })
    }

    /// The URI the store is built from: a URI inside the run's prefix.
    fn store_uri(&self, case: &str) -> String {
        format!(
            "{}://{}/{}/{}/{case}/",
            self.scheme, self.bucket, self.prefix, self.scheme
        )
    }
}

/// One runtime for the whole process, so a client built by one test keeps working for the
/// next (a per-test runtime would take its pooled connections with it).
fn runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<Option<tokio::runtime::Runtime>> = OnceLock::new();
    let built = RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(4)
            .enable_all()
            .build()
            .ok()
    });
    match built {
        Some(rt) => rt,
        None => panic!("cannot build the tokio runtime"),
    }
}

/// Is the ranged fan-out configured for objects of 1 MiB and more?  Mirrors the switches of
/// the library (range optimisation on, HEAD not skipped, threshold at most 1 MiB); the values
/// are the runner's, this file never sets them.
fn fan_out_configured() -> bool {
    let off = |name: &str, off_values: &[&str]| {
        std::env::var(name)
            .ok()
            .is_some_and(|v| off_values.contains(&v.to_lowercase().as_str()))
    };
    let threshold_ok = std::env::var("S3DLIO_RANGE_THRESHOLD_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .is_some_and(|mb| mb <= 1);
    threshold_ok
        && !off(
            "S3DLIO_ENABLE_RANGE_OPTIMIZATION",
            &["0", "false", "no", "off", "disable", "disabled"],
        )
        && !off("S3DLIO_SKIP_HEAD", &["1", "true", "yes", "on"])
}

// -------------------------------------------------------------------------------------------
// Deterministic contents and the guarded destination
// -------------------------------------------------------------------------------------------

/// `len` bytes derived from `seed` (the object's full URI): splitmix64 over an FNV-1a hash of
/// the seed.  Different URIs give different streams.
fn content(seed: &str, len: usize) -> Vec<u8> {
    let mut state: u64 = 0xcbf2_9ce4_8422_2325;
    for b in seed.bytes() {
        state ^= u64::from(b);
        state = state.wrapping_mul(0x0100_0000_01b3);
    }
    state ^= (len as u64).rotate_left(32);
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        let word = z.to_le_bytes();
        let take = word.len().min(len - out.len());
        out.extend_from_slice(&word[..take]);
    }
    out
}

/// A destination of `len` bytes (filled with `FILL`) followed by `CANARY_LEN` canary bytes, all
/// in ONE allocation, so a write past the destination is caught.
struct Dst {
    buf: Vec<u8>,
    len: usize,
}

impl Dst {
    fn new(len: usize) -> Self {
        let mut buf = vec![FILL; len];
        buf.extend(std::iter::repeat_n(CANARY, CANARY_LEN));
        Self { buf, len }
    }

    /// The destination slice handed to the read.
    fn slice(&mut self) -> &mut [u8] {
        &mut self.buf[..self.len]
    }

    fn bytes(&self) -> &[u8] {
        &self.buf[..self.len]
    }

    fn assert_canaries(&self, what: &str) {
        assert!(
            self.buf[self.len..].iter().all(|b| *b == CANARY),
            "{what}: bytes past the destination were written"
        );
    }

    fn assert_untouched(&self, what: &str) {
        assert!(
            self.bytes().iter().all(|b| *b == FILL),
            "{what}: the destination was written"
        );
        self.assert_canaries(what);
    }

    /// How many destination bytes differ from the fill value (an observation, for the report).
    fn touched(&self) -> usize {
        self.bytes().iter().filter(|b| **b != FILL).count()
    }
}

/// Assert two byte strings are equal, naming the first difference instead of dumping both.
fn assert_same(what: &str, got: &[u8], want: &[u8]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: length {} != {}",
        got.len(),
        want.len()
    );
    let first_diff = got.iter().zip(want).position(|(a, b)| a != b);
    assert!(
        first_diff.is_none(),
        "{what}: bytes differ, first at offset {first_diff:?}"
    );
}

/// The typed kind an error carries; fails the test when it carries none.
fn kind_of(err: &anyhow::Error) -> ObjectErrorKind {
    match object_error(err) {
        Some(typed) => typed.kind,
        None => panic!("an untyped error: {err:#}"),
    }
}

/// The error is a `LengthMismatch` for `expected`; when it names an actual size, it is `size`.
fn assert_length_mismatch(what: &str, err: &anyhow::Error, expected: u64, size: u64) {
    match kind_of(err) {
        ObjectErrorKind::LengthMismatch {
            expected: got_expected,
            actual,
        } => {
            assert_eq!(got_expected, expected, "{what}: expected length");
            if let Some(actual) = actual {
                assert_eq!(actual, size, "{what}: actual size");
            }
        }
        other => panic!("{what}: wanted LengthMismatch, got {other:?}: {err:#}"),
    }
}

// -------------------------------------------------------------------------------------------
// The scope: a store, the keys it wrote, and the counters of the summary
// -------------------------------------------------------------------------------------------

#[derive(Default)]
struct Counters {
    puts: AtomicU64,
    heads: AtomicU64,
    get_intos: AtomicU64,
    gets: AtomicU64,
    deletes: AtomicU64,
    put_bytes: AtomicU64,
    get_into_bytes: AtomicU64,
    get_bytes: AtomicU64,
    get_into_ok: AtomicU64,
    get_into_err: AtomicU64,
    get_into_ok_bytes: AtomicU64,
}

impl Counters {
    fn add(counter: &AtomicU64, n: u64) {
        counter.fetch_add(n, Ordering::Relaxed);
    }

    fn read(counter: &AtomicU64) -> u64 {
        counter.load(Ordering::Relaxed)
    }
}

struct Scope {
    cfg: Config,
    case: &'static str,
    store: Box<dyn ObjectStore>,
    written: Mutex<Vec<String>>,
    counters: Counters,
}

type CaseFuture<'a> = Pin<Box<dyn Future<Output = Result<()>> + 'a>>;
type CaseFn = for<'a> fn(&'a Scope) -> CaseFuture<'a>;

impl Scope {
    fn new(cfg: Config, case: &'static str, store: Box<dyn ObjectStore>) -> Self {
        Self {
            cfg,
            case,
            store,
            written: Mutex::new(Vec::new()),
            counters: Counters::default(),
        }
    }

    fn is_gs(&self) -> bool {
        self.cfg.scheme == "gs"
    }

    fn uri(&self, key: &str) -> String {
        format!("{}{key}", self.cfg.store_uri(self.case))
    }

    fn written_keys(&self) -> Vec<String> {
        self.written
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }

    /// PUT `data` under `key`.  The key is registered for cleanup BEFORE the request, so a
    /// PUT that fails half way is still deleted.
    async fn put(&self, key: &str, data: &[u8]) -> Result<()> {
        self.written
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(key.to_string());
        Counters::add(&self.counters.puts, 1);
        Counters::add(&self.counters.put_bytes, data.len() as u64);
        self.store
            .put(&self.uri(key), Bytes::copy_from_slice(data))
            .await
    }

    async fn head(&self, key: &str) -> Result<Option<HeadMetadata>> {
        Counters::add(&self.counters.heads, 1);
        self.store.head(&self.uri(key)).await
    }

    /// Read `key` into `dst`; the progress comes back with the result, also after a failure.
    async fn get_into(&self, key: &str, dst: &mut Dst) -> (ReceiveProgress, Result<()>) {
        Counters::add(&self.counters.get_intos, 1);
        let progress = ReceiveProgress::new();
        let result = self
            .store
            .get_into(&self.uri(key), dst.slice(), &progress)
            .await;
        Counters::add(&self.counters.get_into_bytes, progress.filled());
        match &result {
            Ok(()) => {
                Counters::add(&self.counters.get_into_ok, 1);
                Counters::add(&self.counters.get_into_ok_bytes, progress.filled());
            }
            Err(_) => Counters::add(&self.counters.get_into_err, 1),
        }
        (progress, result)
    }

    /// The independent readback: the allocating `get`.
    async fn get(&self, key: &str) -> Result<Bytes> {
        Counters::add(&self.counters.gets, 1);
        let bytes = self.store.get(&self.uri(key)).await?;
        Counters::add(&self.counters.get_bytes, bytes.len() as u64);
        Ok(bytes)
    }

    /// Delete every written key (exact keys, no listing), delete the first one again (an
    /// already-absent DELETE succeeds), then assert each HEADs as absent.  Returns the problems
    /// found; never panics.
    async fn cleanup(&self) -> Vec<String> {
        let keys = self.written_keys();
        let mut problems = Vec::new();
        for key in &keys {
            Counters::add(&self.counters.deletes, 1);
            if let Err(e) = delete_exact_object(self.store.as_ref(), &self.uri(key)).await {
                problems.push(format!("delete of {key} failed: {e:#}"));
            }
        }
        if let Some(first) = keys.first() {
            Counters::add(&self.counters.deletes, 1);
            if let Err(e) = delete_exact_object(self.store.as_ref(), &self.uri(first)).await {
                problems.push(format!("already-absent delete of {first} failed: {e:#}"));
            }
        }
        for key in &keys {
            match self.head(key).await {
                Ok(None) => {}
                Ok(Some(meta)) => problems.push(format!(
                    "{key} still exists after delete ({} bytes)",
                    meta.size
                )),
                Err(e) => problems.push(format!("HEAD of {key} after delete failed: {e:#}")),
            }
        }
        problems
    }

    fn summary(&self) {
        let c = &self.counters;
        let (puts, heads) = (Counters::read(&c.puts), Counters::read(&c.heads));
        let (get_intos, gets) = (Counters::read(&c.get_intos), Counters::read(&c.gets));
        let deletes = Counters::read(&c.deletes);
        eprintln!(
            "[live-caller-buffer] {}/{}: api calls={} (put={puts} head={heads} get_into={get_intos} \
             get={gets} delete={deletes}); bytes put={} get_into={} get={}",
            self.cfg.scheme,
            self.case,
            puts + heads + get_intos + gets + deletes,
            Counters::read(&c.put_bytes),
            Counters::read(&c.get_into_bytes),
            Counters::read(&c.get_bytes),
        );
    }
}

/// Run `case` against `scope`, then ALWAYS clean up, then report: an assertion failure inside
/// the case is caught, the cleanup runs, and the original failure is re-raised afterwards.
fn run_scope(scope: &Scope, case: CaseFn) {
    let (body, problems) = runtime().block_on(async {
        let body = AssertUnwindSafe(case(scope)).catch_unwind().await;
        let problems = scope.cleanup().await;
        (body, problems)
    });
    scope.summary();
    for problem in &problems {
        eprintln!("[live-caller-buffer] cleanup problem: {problem}");
    }
    match body {
        Err(panic) => std::panic::resume_unwind(panic),
        Ok(Err(e)) => panic!("{}/{} failed: {e:#}", scope.cfg.scheme, scope.case),
        Ok(Ok(())) => {}
    }
    assert!(problems.is_empty(), "cleanup problems: {problems:?}");
}

/// The entry of every test: the environment, the store (built outside the runtime, exactly as
/// a user would), the scope.
fn run_case(scheme: &'static str, case: &'static str, run: CaseFn) {
    let Some(cfg) = Config::from_env(scheme, case) else {
        return;
    };
    let store = match store_for_uri(&cfg.store_uri(case)) {
        Ok(store) => store,
        Err(e) => panic!("{scheme}/{case}: cannot build the store: {e:#}"),
    };
    run_scope(&Scope::new(cfg, case, store), run);
}

// -------------------------------------------------------------------------------------------
// Cases (shared by both groups)
// -------------------------------------------------------------------------------------------

/// PUT `len` bytes under `key`, HEAD them, read them with `get_into` into an exact destination
/// and with the allocating `get`, and compare everything.
async fn round_trip(scope: &Scope, key: &str, len: usize) -> Result<()> {
    let data = content(&scope.uri(key), len);
    scope.put(key, &data).await?;

    let meta = scope
        .head(key)
        .await?
        .unwrap_or_else(|| panic!("{key}: HEAD says the object just written is absent"));
    assert_eq!(meta.size, len as u64, "{key}: HEAD size");
    if scope.is_gs() {
        assert!(
            meta.version_id.is_some(),
            "{key}: native GCS reports the generation as the version id"
        );
    } else {
        assert!(meta.etag.is_some(), "{key}: HEAD reports an ETag");
    }

    let mut dst = Dst::new(len);
    let (progress, result) = scope.get_into(key, &mut dst).await;
    result.map_err(|e| e.context(format!("{key}: get_into of {len} bytes")))?;
    assert_same(&format!("{key}: get_into vs source"), dst.bytes(), &data);
    dst.assert_canaries(key);
    assert_eq!(progress.filled(), len as u64, "{key}: progress.filled()");

    let independent = scope.get(key).await?;
    assert_same(&format!("{key}: get vs source"), &independent, &data);
    assert_same(
        &format!("{key}: get vs get_into"),
        &independent,
        dst.bytes(),
    );
    eprintln!(
        "[live-caller-buffer] {}: {key} {len} bytes ok (etag={:?}, version_id={:?})",
        scope.cfg.scheme, meta.etag, meta.version_id
    );
    Ok(())
}

/// Case 1: sizes 0, 1, 16 and 1 MiB + 7.
fn case_sizes(scope: &Scope) -> CaseFuture<'_> {
    Box::pin(async move {
        for len in SIZES {
            round_trip(scope, &format!("size-{len}"), len).await?;
        }
        Ok(())
    })
}

/// Case 1, the object above the fan-out threshold.  On the S3 path it needs the runner's
/// threshold setting (see the header) and is skipped, saying so, without it.
fn case_large(scope: &Scope) -> CaseFuture<'_> {
    Box::pin(async move {
        if !scope.is_gs() && !fan_out_configured() {
            eprintln!(
                "SKIP s3/large_object_round_trip: S3DLIO_RANGE_THRESHOLD_MB must be 1 or less \
                 (set it, with S3DLIO_CHUNK_SIZE=524288 and S3DLIO_RANGE_CONCURRENCY=4, in the \
                 runner's environment) or this object would not take the ranged fan-out"
            );
            return Ok(());
        }
        round_trip(scope, "large", LARGE_LEN).await
    })
}

/// Case 2: an expected length one too long or one too short is a typed `LengthMismatch`.
fn case_wrong_length(scope: &Scope) -> CaseFuture<'_> {
    Box::pin(async move {
        for len in [0usize, 16, MID_LEN] {
            let key = format!("wrong-{len}");
            let data = content(&scope.uri(&key), len);
            scope.put(&key, &data).await?;

            let mut candidates = vec![len + 1];
            if len > 0 {
                candidates.push(len - 1);
            }
            for expected in candidates {
                let what = format!("{key}: expected_len {expected} for a {len} byte object");
                let mut dst = Dst::new(expected);
                let (progress, result) = scope.get_into(&key, &mut dst).await;
                let err = match result {
                    Ok(()) => panic!("{what}: get_into succeeded"),
                    Err(e) => e,
                };
                assert_length_mismatch(&what, &err, expected as u64, len as u64);
                dst.assert_canaries(&what);
                assert!(
                    progress.filled() <= expected as u64,
                    "{what}: progress {} exceeds the destination",
                    progress.filled()
                );
                if scope.is_gs() {
                    eprintln!(
                        "[live-caller-buffer] gs: {what}: filled={} touched={}",
                        progress.filled(),
                        dst.touched()
                    );
                } else {
                    dst.assert_untouched(&what);
                    assert_eq!(progress.filled(), 0, "{what}: nothing was received");
                }
            }
        }
        Ok(())
    })
}

/// Case 3: a missing key is `NotFound` for `get_into` and `Ok(None)` for `head`.
fn case_missing(scope: &Scope) -> CaseFuture<'_> {
    Box::pin(async move {
        let key = "never-written";
        assert!(
            scope.head(key).await?.is_none(),
            "{key}: HEAD of a missing key must be Ok(None)"
        );
        for len in [0usize, 16] {
            let mut dst = Dst::new(len);
            let (progress, result) = scope.get_into(key, &mut dst).await;
            let err = match result {
                Ok(()) => panic!("{key}: get_into of a missing key succeeded (len {len})"),
                Err(e) => e,
            };
            assert_eq!(
                kind_of(&err),
                ObjectErrorKind::NotFound,
                "{key} (len {len}): {err:#}"
            );
            dst.assert_untouched(key);
            assert_eq!(progress.filled(), 0, "{key}: nothing was received");
        }
        Ok(())
    })
}

/// Case 4: literal keys round-trip, and a near-neighbour key is absent.
fn case_literal_keys(scope: &Scope) -> CaseFuture<'_> {
    Box::pin(async move {
        let mut keys = vec!["a%2Fb", "100%", "x@y", "k=v", "p/q/r", "name."];
        // GCS allows a backslash in an object name (not among the documented restrictions).  An
        // unescaped HTTP read would turn 'a\b' into 'a/b', the absent neighbour checked below.
        // Only gs:// is asked for it: other stores treat a backslash differently.
        if scope.is_gs() {
            keys.push("a\\b");
        }
        for key in keys {
            let len = 32 + key.len();
            let data = content(&scope.uri(key), len);
            scope.put(key, &data).await?;
            let meta = scope
                .head(key)
                .await?
                .unwrap_or_else(|| panic!("{key}: HEAD says the object just written is absent"));
            assert_eq!(meta.size, len as u64, "{key}: HEAD size");

            let mut dst = Dst::new(len);
            let (progress, result) = scope.get_into(key, &mut dst).await;
            result.map_err(|e| e.context(format!("{key}: get_into")))?;
            assert_same(&format!("{key}: get_into vs source"), dst.bytes(), &data);
            dst.assert_canaries(key);
            assert_eq!(progress.filled(), len as u64, "{key}: progress.filled()");
            // The existing get is cross-checked on every key, '%' keys on gs:// included (#270).
            let independent = scope.get(key).await?;
            assert_same(&format!("{key}: get vs source"), &independent, &data);
        }
        // A key is literal: the neighbour of a written key is a different, absent object.
        for neighbour in ["a/b", "100", "name"] {
            assert!(
                scope.head(neighbour).await?.is_none(),
                "{neighbour}: HEAD of an unwritten neighbour must be Ok(None)"
            );
            let mut dst = Dst::new(16);
            let (_, result) = scope.get_into(neighbour, &mut dst).await;
            let err = match result {
                Ok(()) => panic!("{neighbour}: get_into of an unwritten neighbour succeeded"),
                Err(e) => e,
            };
            assert_eq!(
                kind_of(&err),
                ObjectErrorKind::NotFound,
                "{neighbour}: {err:#}"
            );
            dst.assert_untouched(neighbour);
        }
        Ok(())
    })
}

/// Case 5: a zero-length destination on a non-empty object is a `LengthMismatch` and nothing
/// is written.
fn case_zero_length_destination(scope: &Scope) -> CaseFuture<'_> {
    Box::pin(async move {
        let key = "non-empty";
        let len = 16usize;
        let data = content(&scope.uri(key), len);
        scope.put(key, &data).await?;

        let mut dst = Dst::new(0);
        let (progress, result) = scope.get_into(key, &mut dst).await;
        let err = match result {
            Ok(()) => panic!("{key}: a zero-length destination read a non-empty object"),
            Err(e) => e,
        };
        assert_length_mismatch(key, &err, 0, len as u64);
        dst.assert_untouched(key);
        assert_eq!(progress.filled(), 0, "{key}: nothing was received");
        Ok(())
    })
}

// -------------------------------------------------------------------------------------------
// Op-log (S3 path only)
// -------------------------------------------------------------------------------------------

/// Calls whose rows `s3_path::op_log_one_row_per_call` counts: a success, a wrong length and a
/// missing key, each through `get_into` and `head`.
fn case_op_log_calls(scope: &Scope) -> CaseFuture<'_> {
    Box::pin(async move {
        let mut lens = vec![16usize];
        if fan_out_configured() {
            lens.push(MID_LEN);
        }
        for len in lens {
            let key = format!("logged-{len}");
            let data = content(&scope.uri(&key), len);
            scope.put(&key, &data).await?;
            let meta = scope.head(&key).await?;
            assert_eq!(meta.map(|m| m.size), Some(len as u64), "{key}: HEAD size");

            let mut dst = Dst::new(len);
            let (_, result) = scope.get_into(&key, &mut dst).await;
            result?;
            assert_same(&format!("{key}: get_into vs source"), dst.bytes(), &data);

            let mut wrong = Dst::new(len + 1);
            let (_, result) = scope.get_into(&key, &mut wrong).await;
            assert!(result.is_err(), "{key}: a wrong length must fail");
        }
        assert!(scope.head("logged-missing").await?.is_none());
        let mut dst = Dst::new(16);
        let (_, result) = scope.get_into("logged-missing", &mut dst).await;
        assert!(result.is_err(), "a missing key must fail");
        Ok(())
    })
}

/// Finalises the process-global op-log when dropped, so a failed assertion cannot leave the
/// writer thread running.
struct FinalizeOnDrop;

impl Drop for FinalizeOnDrop {
    fn drop(&mut self) {
        s3dlio::finalize_op_logger();
    }
}

fn op_log_one_row_per_call() {
    let Some(cfg) = Config::from_env("s3", "op_log_one_row_per_call") else {
        return;
    };
    let dir = match tempfile::tempdir() {
        Ok(dir) => dir,
        Err(e) => panic!("cannot create a temporary directory: {e}"),
    };
    let path = dir.path().join("live-ops.tsv.zst");
    let Some(path_str) = path.to_str() else {
        eprintln!("SKIP s3/op_log_one_row_per_call: the temporary path is not UTF-8");
        return;
    };
    if let Err(e) = s3dlio::init_op_logger(path_str) {
        eprintln!("SKIP s3/op_log_one_row_per_call: the op-log cannot be started here: {e}");
        return;
    }
    let guard = FinalizeOnDrop;
    let Some(logger) = s3dlio::global_logger() else {
        eprintln!("SKIP s3/op_log_one_row_per_call: no active op-log session after init");
        return;
    };
    let store = match store_for_uri_with_logger(&cfg.store_uri("op_log"), Some(logger)) {
        Ok(store) => store,
        Err(e) => panic!("cannot build the logged store: {e:#}"),
    };
    let scope = Scope::new(cfg, "op_log", store);
    run_scope(&scope, case_op_log_calls);

    // Everything is done (cleanup included): finalise, then read the rows back.
    drop(guard);
    let raw = match std::fs::read(&path) {
        Ok(raw) => raw,
        Err(e) => panic!("the op-log file cannot be read: {e}"),
    };
    let text = match zstd::decode_all(raw.as_slice()) {
        Ok(text) => String::from_utf8_lossy(&text).into_owned(),
        Err(e) => panic!("the op-log is not a complete zstd stream: {e}"),
    };
    let mut lines = text.lines();
    let header = lines.next().unwrap_or_default();
    assert!(
        header.starts_with("idx\tthread\top\t"),
        "unexpected op-log header: {header}"
    );
    const COL_OP: usize = 2;
    const COL_BYTES: usize = 5;
    const COL_ERROR: usize = 8;
    let rows: Vec<Vec<&str>> = lines.map(|l| l.split('\t').collect()).collect();
    assert!(
        rows.iter().all(|r| r.len() == 13),
        "every row has 13 columns"
    );
    let of_op = |op: &str| rows.iter().filter(|r| r[COL_OP] == op).count();
    let c = &scope.counters;

    let get_rows: Vec<_> = rows.iter().filter(|r| r[COL_OP] == "GET").collect();
    assert_eq!(
        get_rows.len() as u64,
        Counters::read(&c.get_intos),
        "one GET row per get_into call (no helper, range or retry rows)"
    );
    let ok_rows: Vec<_> = get_rows
        .iter()
        .filter(|r| r[COL_ERROR].is_empty())
        .collect();
    assert_eq!(
        ok_rows.len() as u64,
        Counters::read(&c.get_into_ok),
        "successful GET rows"
    );
    let ok_bytes: u64 = ok_rows
        .iter()
        .filter_map(|r| r[COL_BYTES].parse::<u64>().ok())
        .sum();
    assert_eq!(
        ok_bytes,
        Counters::read(&c.get_into_ok_bytes),
        "the successful GET rows carry the logical byte count"
    );
    assert_eq!(
        (get_rows.len() - ok_rows.len()) as u64,
        Counters::read(&c.get_into_err),
        "failed GET rows"
    );
    assert_eq!(
        of_op("HEAD") as u64,
        Counters::read(&c.heads),
        "one HEAD row per head call (the fan-out's own HEAD writes none)"
    );
    eprintln!(
        "[live-caller-buffer] s3/op_log: {} rows: GET={} (ok={}) HEAD={} PUT={} DELETE={}",
        rows.len(),
        get_rows.len(),
        ok_rows.len(),
        of_op("HEAD"),
        of_op("PUT"),
        of_op("DELETE")
    );
}

// -------------------------------------------------------------------------------------------
// The tests: one module per group, the same cases in each
// -------------------------------------------------------------------------------------------

macro_rules! live_group {
    ($group:ident, $scheme:literal) => {
        mod $group {
            use super::*;

            #[test]
            #[ignore = "live: needs S3DLIO_LIVE_BUCKET and credentials; run explicitly"]
            fn sizes_round_trip() {
                run_case($scheme, "sizes", case_sizes);
            }

            #[test]
            #[ignore = "live: needs S3DLIO_LIVE_BUCKET and credentials; run explicitly"]
            fn large_object_round_trip() {
                run_case($scheme, "large", case_large);
            }

            #[test]
            #[ignore = "live: needs S3DLIO_LIVE_BUCKET and credentials; run explicitly"]
            fn wrong_expected_length_is_a_length_mismatch() {
                run_case($scheme, "wrong_length", case_wrong_length);
            }

            #[test]
            #[ignore = "live: needs S3DLIO_LIVE_BUCKET and credentials; run explicitly"]
            fn missing_key_is_not_found() {
                run_case($scheme, "missing", case_missing);
            }

            #[test]
            #[ignore = "live: needs S3DLIO_LIVE_BUCKET and credentials; run explicitly"]
            fn literal_keys_round_trip() {
                run_case($scheme, "literal_keys", case_literal_keys);
            }

            #[test]
            #[ignore = "live: needs S3DLIO_LIVE_BUCKET and credentials; run explicitly"]
            fn zero_length_destination_is_a_length_mismatch() {
                run_case($scheme, "zero_length_dst", case_zero_length_destination);
            }
        }
    };
}

live_group!(s3_path, "s3");

#[cfg(all(feature = "backend-gcs", not(feature = "gcs-community")))]
live_group!(native_gcs, "gs");

mod s3_op_log {
    #[test]
    #[ignore = "live: needs S3DLIO_LIVE_BUCKET and credentials; run explicitly"]
    fn op_log_one_row_per_call() {
        super::op_log_one_row_per_call();
    }
}
