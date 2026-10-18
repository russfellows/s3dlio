// tests/live_caller_buffer_azure.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// LIVE tests of the caller-buffer GET (`ObjectStore::get_into`) and the typed HEAD
// (`ObjectStore::head`) against a real Azure Blob account (v0.10.0 plan, sections 8 and 9,
// #264).  It mirrors `live_caller_buffer_gcs.rs`.
//
// EVERY test here is `#[ignore]`d and also returns early (with a skip message on stderr) when
// its environment is missing, so `cargo test` never reaches a service.  Run them explicitly:
//
//   cargo test --test live_caller_buffer_azure --features full-backends -- \
//       --ignored --test-threads=1 --nocapture
//
// Configuration comes ONLY from the environment (no account or container name is written in
// this file):
//
//   S3DLIO_LIVE_AZURE_ACCOUNT    the storage account
//   S3DLIO_LIVE_AZURE_CONTAINER  an existing container the runner may write to
//   S3DLIO_LIVE_PREFIX           a prefix the runner chooses, unique to this run (for example
//                                one that carries a timestamp).  Every blob lives under
//                                `<PREFIX>/<case>/`; nothing outside it is touched.
//
// Credentials are whatever the runner's environment provides to s3dlio's Azure client: for
// example `S3DLIO_AZURE_CREDENTIAL=cli` after an Azure CLI login, or `auto` (the default
// credential chain), or `sas` with `S3DLIO_AZURE_SAS_TOKEN`.  Nothing here reads or sets a
// credential.  To keep a stray `.env` out of the run the runner may set `S3DLIO_LOAD_DOTENV=0`.
//
// What the Azure path does (plan section 4.3): one bounded request of N+1 bytes for an
// N-byte destination, with the partition size N+1 and parallelism 1, so there is NO range
// fan-out and no request-count knob to set.  The whole size comes from the response headers
// and is checked before any byte is copied, so a wrong expected length leaves the destination
// untouched.  Zero length uses a bounded 0..1 request.
//
// Rules this file follows (plan section 8):
//   * deterministic, unique contents: every blob's bytes are generated from its full URI, so
//     two blobs never share contents and a read of the wrong blob cannot compare equal;
//   * an isolated prefix, and no listing: blobs are found only by their exact keys;
//   * independent readback: every body is read back through the allocating `get` as well as
//     compared with the source bytes;
//   * cleanup: each case runs under a guard that deletes every key it wrote (exact keys,
//     registered BEFORE the PUT) even when an assertion fails, deletes the first key a second
//     time (an already-absent DELETE succeeds), and finally asserts each key HEADs as absent;
//   * budget: the plan allows about 1 GB of egress for the whole Azure round; this file keeps
//     a single run under `EGRESS_BUDGET` (50 MiB) and asserts it before each read.
//
// Missing container.  A missing container is NOT object absence: `head` must return an error
// (never `Ok(None)`) and `get_into` an error that is not `NotFound`.  The container name used
// is derived from the run's prefix (a fixed text and a hash of the prefix), so it is
// effectively certain not to exist; the test never creates it.
//
// Size.  One full run stores about 6.5 MiB and reads about 13 MiB (see the report that came
// with this file for the per-test request counts).  Run with `--test-threads=1`.

#![cfg(feature = "backend-azure")]

use anyhow::Result;
use bytes::Bytes;
use futures::FutureExt;
use s3dlio::delete_contract::delete_exact_object;
use s3dlio::object_error::{object_error, ObjectErrorKind};
use s3dlio::object_store::{store_for_uri, HeadMetadata, ObjectStore, ReceiveProgress};
use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

const ENV_ACCOUNT: &str = "S3DLIO_LIVE_AZURE_ACCOUNT";
const ENV_CONTAINER: &str = "S3DLIO_LIVE_AZURE_CONTAINER";
const ENV_PREFIX: &str = "S3DLIO_LIVE_PREFIX";

const MIB: usize = 1024 * 1024;
/// Just above 1 MiB, so it is not a multiple of any chunk size.
const MID_LEN: usize = MIB + 7;
/// Several MiB and not a multiple of anything: the largest object of the run.
const LARGE_LEN: usize = 4 * MIB + 3;
/// The sizes of `sizes_round_trip`.
const SIZES: [usize; 4] = [0, 1, 16, MID_LEN];

/// The byte a destination holds before a read.
const FILL: u8 = 0xEE;
/// The byte of the guard region after a destination.
const CANARY: u8 = 0xA5;
/// How many guard bytes follow a destination.
const CANARY_LEN: usize = 64;

/// The most data one run may ask the service to send (a conservative estimate: the size of
/// every read that is started, plus one byte per request for the bounded N+1 read).
const EGRESS_BUDGET: u64 = 50 * 1024 * 1024;
static EGRESS: AtomicU64 = AtomicU64::new(0);

/// Account `bytes` of expected egress for `what`; fails the case when the run's budget would be
/// exceeded.
fn charge_egress(bytes: u64, what: &str) {
    let total = EGRESS.fetch_add(bytes, Ordering::Relaxed) + bytes;
    assert!(
        total <= EGRESS_BUDGET,
        "{what}: the run's egress budget of {EGRESS_BUDGET} bytes would be exceeded ({total})"
    );
}

// -------------------------------------------------------------------------------------------
// Configuration and the shared runtime
// -------------------------------------------------------------------------------------------

struct Config {
    account: String,
    container: String,
    prefix: String,
}

impl Config {
    /// The configuration, or `None` (after saying why on stderr) when the environment is
    /// incomplete.  A prefix is trimmed of slashes; an empty one is refused, so a run can
    /// never write at the container root.
    fn from_env(case: &str) -> Option<Self> {
        let get = |name: &str| {
            std::env::var(name)
                .ok()
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
        };
        let (Some(account), Some(container), Some(prefix)) =
            (get(ENV_ACCOUNT), get(ENV_CONTAINER), get(ENV_PREFIX))
        else {
            eprintln!(
                "SKIP azure/{case}: {ENV_ACCOUNT}, {ENV_CONTAINER} and {ENV_PREFIX} must all be set"
            );
            return None;
        };
        if account.contains('/') || container.contains('/') {
            eprintln!("SKIP azure/{case}: the account and the container must not contain '/'");
            return None;
        }
        let prefix = prefix.trim_matches('/').to_string();
        if prefix.is_empty() {
            eprintln!("SKIP azure/{case}: {ENV_PREFIX} must name a non-empty prefix");
            return None;
        }
        Some(Self {
            account,
            container,
            prefix,
        })
    }

    /// The URI the store is built from: a URI inside the run's prefix.
    fn store_uri(&self, case: &str) -> String {
        format!(
            "az://{}/{}/{}/{case}/",
            self.account, self.container, self.prefix
        )
    }

    /// A container that does not exist, derived from the run's prefix.
    fn absent_container(&self) -> String {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for b in self.prefix.bytes() {
            hash ^= u64::from(b);
            hash = hash.wrapping_mul(0x0100_0000_01b3);
        }
        format!("s3dlio-absent-{hash:016x}")
    }

    /// The URI of `key` in the container that does not exist.
    fn absent_uri(&self, key: &str) -> String {
        format!("az://{}/{}/{key}", self.account, self.absent_container())
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

// -------------------------------------------------------------------------------------------
// Deterministic contents and the guarded destination
// -------------------------------------------------------------------------------------------

/// `len` bytes derived from `seed` (the blob's full URI): splitmix64 over an FNV-1a hash of
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
    /// A store built for a container that does not exist (used by `missing_container` only).
    absent_store: Box<dyn ObjectStore>,
    written: Mutex<Vec<String>>,
    counters: Counters,
}

type CaseFuture<'a> = Pin<Box<dyn Future<Output = Result<()>> + 'a>>;
type CaseFn = for<'a> fn(&'a Scope) -> CaseFuture<'a>;

impl Scope {
    fn new(
        cfg: Config,
        case: &'static str,
        store: Box<dyn ObjectStore>,
        absent_store: Box<dyn ObjectStore>,
    ) -> Self {
        Self {
            cfg,
            case,
            store,
            absent_store,
            written: Mutex::new(Vec::new()),
            counters: Counters::default(),
        }
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
        {
            let mut written = self
                .written
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if !written.iter().any(|k| k == key) {
                written.push(key.to_string());
            }
        }
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
        charge_egress(dst.len as u64 + 1, key);
        Counters::add(&self.counters.get_intos, 1);
        let progress = ReceiveProgress::new();
        let result = self
            .store
            .get_into(&self.uri(key), dst.slice(), &progress)
            .await;
        Counters::add(&self.counters.get_into_bytes, progress.filled());
        (progress, result)
    }

    /// The independent readback: the allocating `get` of an object of `len` bytes.
    async fn get(&self, key: &str, len: usize) -> Result<Bytes> {
        charge_egress(len as u64, key);
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
            "[live-caller-buffer] azure/{}: api calls={} (put={puts} head={heads} \
             get_into={get_intos} get={gets} delete={deletes}); bytes put={} get_into={} get={}; \
             run egress estimate={} of {EGRESS_BUDGET}",
            self.case,
            puts + heads + get_intos + gets + deletes,
            Counters::read(&c.put_bytes),
            Counters::read(&c.get_into_bytes),
            Counters::read(&c.get_bytes),
            EGRESS.load(Ordering::Relaxed),
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
        Ok(Err(e)) => panic!("azure/{} failed: {e:#}", scope.case),
        Ok(Ok(())) => {}
    }
    assert!(problems.is_empty(), "cleanup problems: {problems:?}");
}

/// The entry of every test: the environment, the stores (built outside the runtime, exactly as
/// a user would), the scope.
fn run_case(case: &'static str, run: CaseFn) {
    let Some(cfg) = Config::from_env(case) else {
        return;
    };
    let store = match store_for_uri(&cfg.store_uri(case)) {
        Ok(store) => store,
        Err(e) => panic!("azure/{case}: cannot build the store: {e:#}"),
    };
    let absent_store = match store_for_uri(&cfg.absent_uri("")) {
        Ok(store) => store,
        Err(e) => panic!("azure/{case}: cannot build the store for the absent container: {e:#}"),
    };
    run_scope(&Scope::new(cfg, case, store, absent_store), run);
}

// -------------------------------------------------------------------------------------------
// Cases
// -------------------------------------------------------------------------------------------

/// PUT `len` bytes under `key`, HEAD them, read them with `get_into` into an exact destination
/// and with the allocating `get`, and compare everything.
async fn round_trip(scope: &Scope, key: &str, len: usize) -> Result<()> {
    let data = content(&scope.uri(key), len);
    scope.put(key, &data).await?;

    let meta = scope
        .head(key)
        .await?
        .unwrap_or_else(|| panic!("{key}: HEAD says the blob just written is absent"));
    assert_eq!(meta.size, len as u64, "{key}: HEAD size");
    assert!(meta.etag.is_some(), "{key}: HEAD reports an ETag");

    let mut dst = Dst::new(len);
    let (progress, result) = scope.get_into(key, &mut dst).await;
    result.map_err(|e| e.context(format!("{key}: get_into of {len} bytes")))?;
    assert_same(&format!("{key}: get_into vs source"), dst.bytes(), &data);
    dst.assert_canaries(key);
    assert_eq!(progress.filled(), len as u64, "{key}: progress.filled()");

    let independent = scope.get(key, len).await?;
    assert_same(&format!("{key}: get vs source"), &independent, &data);
    assert_same(
        &format!("{key}: get vs get_into"),
        &independent,
        dst.bytes(),
    );
    eprintln!(
        "[live-caller-buffer] azure: {key} {len} bytes ok (etag={:?}, version_id={:?})",
        meta.etag, meta.version_id
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

/// Case 1, the largest object (4 MiB + 3).  There is no fan-out on the Azure path: it is still
/// one bounded request of N+1 bytes.
fn case_large(scope: &Scope) -> CaseFuture<'_> {
    Box::pin(async move { round_trip(scope, "large", LARGE_LEN).await })
}

/// Case 2: an expected length one too long or one too short is a typed `LengthMismatch`, and
/// because the size is checked from the headers before any copy, nothing is written.
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
                let what = format!("{key}: expected_len {expected} for a {len} byte blob");
                // The body is dropped unread, but count the whole blob as possibly sent.
                charge_egress(len as u64, &what);
                let mut dst = Dst::new(expected);
                let (progress, result) = scope.get_into(&key, &mut dst).await;
                let err = match result {
                    Ok(()) => panic!("{what}: get_into succeeded"),
                    Err(e) => e,
                };
                assert_length_mismatch(&what, &err, expected as u64, len as u64);
                dst.assert_untouched(&what);
                assert_eq!(progress.filled(), 0, "{what}: nothing was received");
            }
        }
        Ok(())
    })
}

/// Case 3: a missing key is `NotFound` for `get_into` and `Ok(None)` for `head`.
fn case_missing_key(scope: &Scope) -> CaseFuture<'_> {
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

/// Case 3b: a missing CONTAINER is an error, never absence.
fn case_missing_container(scope: &Scope) -> CaseFuture<'_> {
    Box::pin(async move {
        let uri = scope.cfg.absent_uri("some-key");

        Counters::add(&scope.counters.heads, 1);
        match scope.absent_store.head(&uri).await {
            Err(e) => eprintln!("[live-caller-buffer] azure: HEAD in an absent container: {e:#}"),
            Ok(None) => panic!(
                "HEAD in a missing container returned Ok(None): a missing container must be an \
                 error, not object absence"
            ),
            Ok(Some(meta)) => panic!(
                "the container derived from the prefix exists ({} byte blob found); the test \
                 needs a container that does not exist",
                meta.size
            ),
        }

        let mut dst = Dst::new(16);
        let progress = ReceiveProgress::new();
        Counters::add(&scope.counters.get_intos, 1);
        let result = scope
            .absent_store
            .get_into(&uri, dst.slice(), &progress)
            .await;
        let err = match result {
            Ok(()) => panic!("get_into in a missing container succeeded"),
            Err(e) => e,
        };
        let kind = kind_of(&err);
        eprintln!("[live-caller-buffer] azure: get_into in an absent container: {kind:?}: {err:#}");
        assert_ne!(
            kind,
            ObjectErrorKind::NotFound,
            "a missing container must not be reported as an absent object: {err:#}"
        );
        dst.assert_untouched("missing container");
        assert_eq!(
            progress.filled(),
            0,
            "missing container: nothing was received"
        );
        Ok(())
    })
}

/// Case 4: literal keys round-trip, and a near-neighbour key is absent.
fn case_literal_keys(scope: &Scope) -> CaseFuture<'_> {
    Box::pin(async move {
        let keys = ["a%2Fb", "100%", "x@y", "k=v", "p/q/r", "name."];
        for key in keys {
            let len = 32 + key.len();
            let data = content(&scope.uri(key), len);
            scope.put(key, &data).await?;
            let meta = scope
                .head(key)
                .await?
                .unwrap_or_else(|| panic!("{key}: HEAD says the blob just written is absent"));
            assert_eq!(meta.size, len as u64, "{key}: HEAD size");

            let mut dst = Dst::new(len);
            let (progress, result) = scope.get_into(key, &mut dst).await;
            result.map_err(|e| e.context(format!("{key}: get_into")))?;
            assert_same(&format!("{key}: get_into vs source"), dst.bytes(), &data);
            dst.assert_canaries(key);
            assert_eq!(progress.filled(), len as u64, "{key}: progress.filled()");

            let independent = scope.get(key, len).await?;
            assert_same(&format!("{key}: get vs source"), &independent, &data);
        }
        // A key is literal: the neighbour of a written key is a different, absent blob.
        for neighbour in ["a/b", "100"] {
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
        // Azure strips a trailing '.' from a blob name in the URL, so "name." and "name" are the
        // same blob (Azure naming rules); recorded as a provider limitation.  "name" is NOT
        // registered as a written key: deleting "name." removes the aliased blob.
        let alias_len = 32 + "name.".len();
        let alias_data = content(&scope.uri("name."), alias_len);
        let written = scope
            .head("name.")
            .await?
            .unwrap_or_else(|| panic!("name.: HEAD says the blob just written is absent"));
        let alias = scope
            .head("name")
            .await?
            .unwrap_or_else(|| panic!("name: HEAD of the trailing-dot alias says absent"));
        assert_eq!(alias.size, written.size, "name: alias HEAD size");
        assert_eq!(alias.etag, written.etag, "name: alias HEAD ETag");
        let mut alias_dst = Dst::new(alias_len);
        let (alias_progress, alias_result) = scope.get_into("name", &mut alias_dst).await;
        alias_result.map_err(|e| e.context("name: get_into of the trailing-dot alias"))?;
        assert_same(
            "name: alias get_into vs source",
            alias_dst.bytes(),
            &alias_data,
        );
        alias_dst.assert_canaries("name");
        assert_eq!(
            alias_progress.filled(),
            alias_len as u64,
            "name: alias progress.filled()"
        );
        Ok(())
    })
}

/// Case 5: a zero-length destination on a non-empty blob is a `LengthMismatch` and nothing is
/// written (the bounded 0..1 request shows a byte exists).
fn case_zero_length_destination(scope: &Scope) -> CaseFuture<'_> {
    Box::pin(async move {
        let key = "non-empty";
        let len = 16usize;
        let data = content(&scope.uri(key), len);
        scope.put(key, &data).await?;

        let mut dst = Dst::new(0);
        let (progress, result) = scope.get_into(key, &mut dst).await;
        let err = match result {
            Ok(()) => panic!("{key}: a zero-length destination read a non-empty blob"),
            Err(e) => e,
        };
        assert_length_mismatch(key, &err, 0, len as u64);
        dst.assert_untouched(key);
        assert_eq!(progress.filled(), 0, "{key}: nothing was received");
        Ok(())
    })
}

/// Case 6: HEAD reports the size, an ETag and the version identifier when the account keeps
/// one; two HEADs of an unchanged blob agree; a same-size overwrite changes the identity; and
/// `get_into` returns the new content.
fn case_head_identity(scope: &Scope) -> CaseFuture<'_> {
    Box::pin(async move {
        let key = "identity";
        let len = 4096usize;
        let first_data = content(&scope.uri("identity-first"), len);
        let second_data = content(&scope.uri("identity-second"), len);
        assert_ne!(first_data, second_data, "the two bodies must differ");

        scope.put(key, &first_data).await?;
        let first = scope
            .head(key)
            .await?
            .unwrap_or_else(|| panic!("{key}: HEAD says the blob just written is absent"));
        let again = scope
            .head(key)
            .await?
            .unwrap_or_else(|| panic!("{key}: the second HEAD says the blob is absent"));
        assert_eq!(first.size, len as u64, "{key}: HEAD size");
        assert!(first.etag.is_some(), "{key}: HEAD reports an ETag");
        assert_eq!(first, again, "{key}: two HEADs of an unchanged blob differ");

        let mut dst = Dst::new(len);
        let (_, result) = scope.get_into(key, &mut dst).await;
        result?;
        assert_same(&format!("{key}: first body"), dst.bytes(), &first_data);

        // Same size, new content: the identity must change and the read must follow it.
        scope.put(key, &second_data).await?;
        let second = scope
            .head(key)
            .await?
            .unwrap_or_else(|| panic!("{key}: HEAD after the overwrite says the blob is absent"));
        assert_eq!(second.size, len as u64, "{key}: size after the overwrite");
        assert_ne!(
            first.etag, second.etag,
            "{key}: a same-size overwrite must change the ETag"
        );
        let mut dst = Dst::new(len);
        let (_, result) = scope.get_into(key, &mut dst).await;
        result?;
        assert_same(&format!("{key}: second body"), dst.bytes(), &second_data);
        eprintln!(
            "[live-caller-buffer] azure: {key}: etag {:?} -> {:?}, version_id {:?} -> {:?}",
            first.etag, second.etag, first.version_id, second.version_id
        );
        Ok(())
    })
}

// -------------------------------------------------------------------------------------------
// The tests
// -------------------------------------------------------------------------------------------

#[test]
#[ignore = "live: needs the S3DLIO_LIVE_AZURE_* variables and credentials; run explicitly"]
fn sizes_round_trip() {
    run_case("sizes", case_sizes);
}

#[test]
#[ignore = "live: needs the S3DLIO_LIVE_AZURE_* variables and credentials; run explicitly"]
fn large_object_round_trip() {
    run_case("large", case_large);
}

#[test]
#[ignore = "live: needs the S3DLIO_LIVE_AZURE_* variables and credentials; run explicitly"]
fn wrong_expected_length_is_a_length_mismatch() {
    run_case("wrong_length", case_wrong_length);
}

#[test]
#[ignore = "live: needs the S3DLIO_LIVE_AZURE_* variables and credentials; run explicitly"]
fn missing_key_is_not_found() {
    run_case("missing_key", case_missing_key);
}

#[test]
#[ignore = "live: needs the S3DLIO_LIVE_AZURE_* variables and credentials; run explicitly"]
fn missing_container_is_an_error_not_absence() {
    run_case("missing_container", case_missing_container);
}

#[test]
#[ignore = "live: needs the S3DLIO_LIVE_AZURE_* variables and credentials; run explicitly"]
fn literal_keys_round_trip() {
    run_case("literal_keys", case_literal_keys);
}

#[test]
#[ignore = "live: needs the S3DLIO_LIVE_AZURE_* variables and credentials; run explicitly"]
fn zero_length_destination_is_a_length_mismatch() {
    run_case("zero_length_dst", case_zero_length_destination);
}

#[test]
#[ignore = "live: needs the S3DLIO_LIVE_AZURE_* variables and credentials; run explicitly"]
fn head_reports_size_etag_and_identity() {
    run_case("head_identity", case_head_identity);
}
