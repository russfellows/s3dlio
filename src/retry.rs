// src/retry.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Shared retry helper for body-transfer-phase failures (issue #148,
// audit §3.4). Introduced as part of Phase 4 (streaming connector +
// centralized retry).
//
// Why this exists
// ───────────────
// Once the reqwest connector streams response bodies to the SDK
// (Phase 4b), the SDK's own smithy-level retry no longer covers
// body-transfer failures — smithy considers the request "done" as
// soon as `send()` returns Ok(response), which now resolves at
// response headers rather than after the full body downloads. Errors
// that surface while the caller consumes the streaming body must be
// retried at the caller level.
//
// Before this module, `s3_client.rs`, `python_api/python_core_api.rs`,
// and `google_gcs_client.rs` each hand-rolled their own retry shape
// (see audit §3.4). Phase 4 wraps four s3 body-transfer call sites in
// `retry_get_body` instead of adding a fourth independently-invented
// loop.
//
// Design
// ──────
// - Bounded by `crate::constants::max_retry_attempts()` (env-tunable
//   via `S3DLIO_MAX_RETRY_ATTEMPTS`, default 3).
// - Linear backoff: sleep `100ms * attempt` between attempts. Linear
//   (not exponential) is deliberate — the caller has bounded retries
//   and each failure has already burned network time; keeping the
//   backoff small stays responsive without dogpiling a struggling
//   backend the way `0ms * 3` does.
// - Retries only TRANSIENT failures (issue #240). The error type decides,
//   through [`RetryClassify`]: a definite server answer (404 NoSuchKey,
//   403 AccessDenied, any other non-transient 4xx) is returned after the
//   first attempt, so a missing key costs one request and no backoff.
//   The caller decides idempotency (GETs are idempotent; the body-transfer
//   sites this wraps are all GETs).
// - Each attempt re-invokes the `FnMut() -> Fut` closure to produce a
//   FRESH future. Callers relying on shared mutable state (e.g. a
//   pre-allocated `BytesMut` segment) must declare the "cursor" state
//   (like `written = 0`) inside the async block so each attempt starts
//   from a clean position — see the range-chunk retry in
//   `s3_utils.rs::concurrent_range_get_impl` for the canonical shape,
//   and `tests/test_phase4_retry.rs` for the fault-injection regression
//   test that locks this invariant in.

use std::future::Future;
use std::time::Duration;

use aws_sdk_s3::error::ProvideErrorMetadata;
use aws_smithy_runtime_api::client::orchestrator::HttpResponse;
use aws_smithy_runtime_api::client::result::SdkError;

use crate::constants::max_retry_attempts;
use crate::object_error::ObjectErrorKind;

/// The throttling error codes the AWS SDK's own retry classifier recognises:
/// `aws_runtime::retries::classifiers::THROTTLING_ERRORS`, copied from **aws-runtime
/// 1.7.5** (the version `Cargo.lock` pins; `src/retries/classifiers.rs`).
///
/// The SDK retries these inside one operation; when its attempts are spent it returns
/// the final service error, and the OUTER loop in this module must still treat the same
/// codes as transient, or it stops after the SDK's budget instead of starting its own.
/// A unit test (`aws_runtime_code_tables_are_all_retryable`) compares this constant with
/// the pinned crate's tables. It fails `cargo test` (the build itself still succeeds) when
/// an SDK upgrade changes those tables; the fix is to update the constants here.
const AWS_RUNTIME_THROTTLING_ERROR_CODES: &[&str] = &[
    "Throttling",
    "ThrottlingException",
    "ThrottledException",
    "RequestThrottledException",
    "TooManyRequestsException",
    "ProvisionedThroughputExceededException",
    "TransactionInProgressException",
    "RequestLimitExceeded",
    "BandwidthLimitExceeded",
    "LimitExceededException",
    "RequestThrottled",
    "SlowDown",
    "PriorRequestNotComplete",
    "EC2ThrottledException",
];

/// The transient error codes the AWS SDK's retry classifier recognises:
/// `aws_runtime::retries::classifiers::TRANSIENT_ERRORS`, copied from **aws-runtime
/// 1.7.5** (see [`AWS_RUNTIME_THROTTLING_ERROR_CODES`]). `RequestTimeoutException` is
/// the one the first version of this classifier missed.
const AWS_RUNTIME_TRANSIENT_ERROR_CODES: &[&str] = &["RequestTimeout", "RequestTimeoutException"];

/// Codes s3dlio treats as transient on top of the SDK's tables.
const S3DLIO_EXTRA_TRANSIENT_CODES: &[&str] = &[
    "InternalError",
    "ServiceUnavailable",
    // Credential expiry: the SDK re-resolves its identity (and may refresh
    // STS credentials) on the next attempt, so a later attempt can succeed.
    "ExpiredToken",
    "RequestExpired",
    // S3 documents this one as "retry the request".
    "OperationAborted",
];

/// Is `code` an S3 / AWS error code that means "try again later" whatever HTTP status
/// the server attached to it (S3 itself reports `RequestTimeout` as a 400 and some
/// S3-compatible servers report `SlowDown` as a 4xx)?
fn is_transient_error_code(code: &str) -> bool {
    AWS_RUNTIME_THROTTLING_ERROR_CODES.contains(&code)
        || AWS_RUNTIME_TRANSIENT_ERROR_CODES.contains(&code)
        || S3DLIO_EXTRA_TRANSIENT_CODES.contains(&code)
}

/// Is an HTTP status, with an optional S3 error code, worth retrying?
///
/// Transient: 408, 429, every 5xx, and any status carrying one of the codes
/// [`is_transient_error_code`] accepts (the AWS SDK's throttling and transient tables
/// plus a few S3-specific ones). Definite (not retried): every other 3xx and
/// 4xx (400, 401, 403, 404, 405, 409, 412, 416, ...). Statuses outside
/// 300-599 cannot be a definite client error, so they are retried.
pub(crate) fn status_is_retryable(status: u16, code: Option<&str>) -> bool {
    if let Some(code) = code {
        if is_transient_error_code(code) {
            return true;
        }
    }
    match status {
        408 | 429 => true,
        300..=499 => false,
        _ => true,
    }
}

/// Classify an AWS SDK error: `true` when the failure is transient.
///
/// - connection / dispatch failures and timeouts: transient;
/// - a service or response error is judged by its raw HTTP status and S3
///   error code via [`status_is_retryable`];
/// - a construction failure (the request could not even be built) is
///   deterministic and not retried;
/// - any variant added to the SDK later is treated as transient.
pub(crate) fn sdk_error_is_retryable<E>(e: &SdkError<E, HttpResponse>) -> bool
where
    E: ProvideErrorMetadata,
{
    match e {
        SdkError::ConstructionFailure(_) => false,
        SdkError::DispatchFailure(_) | SdkError::TimeoutError(_) => true,
        SdkError::ServiceError(s) => status_is_retryable(s.raw().status().as_u16(), s.err().code()),
        SdkError::ResponseError(r) => status_is_retryable(r.raw().status().as_u16(), None),
        _ => true,
    }
}

/// Classify an S3 failure from its HTTP `status` and service error `code`
/// (plan §4.5 A2).  Pure: no I/O, no environment.
///
/// `pinned` is true once the read has pinned an ETag; the pinned-read override
/// then takes precedence over the table below: a 404 (or `NoSuchKey` /
/// `NotFound`) or a 412 on a pinned read means the pinned version is gone, so
/// it is [`ObjectErrorKind::VersionChanged`], never absence.
///
/// Otherwise the table is code-first, preserving [`status_is_retryable`]'s
/// order for ordinary failures:
///
/// 1. a code [`is_transient_error_code`] accepts (the AWS throttling and
///    transient tables plus `S3DLIO_EXTRA_TRANSIENT_CODES`, including
///    `ExpiredToken` and `RequestExpired`) is retryable even on a 403: a
///    throttling-table code is `Throttled`, any other transient code `Server`;
/// 2. 404, `NoSuchKey` or `NotFound` is `NotFound`;
/// 3. 412 is `VersionChanged`;
/// 4. any remaining 403 (e.g. `AccessDenied`) is `Forbidden`;
/// 5. 429 or 503 is `Throttled`; 408 or any other 5xx is `Server`;
/// 6. a failure with no HTTP response (connect, timeout, body stream) is
///    `Transport`;
/// 7. anything else is `Other`.
pub fn classify_s3_failure(
    status: Option<u16>,
    code: Option<&str>,
    pinned: bool,
) -> ObjectErrorKind {
    if pinned {
        let gone = status == Some(404)
            || status == Some(412)
            || matches!(code, Some("NoSuchKey" | "NotFound" | "PreconditionFailed"));
        if gone {
            return ObjectErrorKind::VersionChanged;
        }
    }
    if let Some(code) = code {
        if is_transient_error_code(code) {
            return if AWS_RUNTIME_THROTTLING_ERROR_CODES.contains(&code) {
                ObjectErrorKind::Throttled
            } else {
                ObjectErrorKind::Server
            };
        }
        if matches!(code, "NoSuchKey" | "NotFound") {
            return ObjectErrorKind::NotFound;
        }
    }
    match status {
        Some(404) => ObjectErrorKind::NotFound,
        Some(412) => ObjectErrorKind::VersionChanged,
        Some(403) => ObjectErrorKind::Forbidden,
        Some(429 | 503) => ObjectErrorKind::Throttled,
        Some(408 | 500..=599) => ObjectErrorKind::Server,
        None => ObjectErrorKind::Transport,
        Some(_) => ObjectErrorKind::Other,
    }
}

/// Attach the S3 classification of `err` as an
/// [`ObjectError`](crate::object_error::ObjectError) context.
///
/// The status and code come from the [`SdkFailure`] `err` carries (a failure
/// from the AWS SDK).  An error with no `SdkFailure`, e.g. a body-stream
/// error, had no HTTP response and is therefore `Transport` (step 6).  An
/// error that already carries an `ObjectError` is returned unchanged: the
/// existing classification is the more specific one.  The attached message
/// names the status and code only; the SDK text stays in the source below it.
pub fn attach_s3_classification(err: anyhow::Error, pinned: bool) -> anyhow::Error {
    if crate::object_error::object_error(&err).is_some() {
        return err;
    }
    let (status, code) = match err.downcast_ref::<SdkFailure>() {
        Some(f) => (f.status(), f.code().map(str::to_owned)),
        None => (None, None),
    };
    let kind = classify_s3_failure(status, code.as_deref(), pinned);
    let message = match (status, code.as_deref()) {
        (Some(s), Some(c)) => format!("status {s}, code {c}"),
        (Some(s), None) => format!("status {s}"),
        (None, Some(c)) => format!("code {c}"),
        (None, None) => "no HTTP response".to_string(),
    };
    crate::object_error::attach_object_error(err, kind, "s3", message)
}

/// The anyhow error that [`crate::s3_utils`]'s `sdk_context` / `sdk_anyhow`
/// produce for a failed S3 request. Its `Display` is the same flattened
/// `"<ctx>: <formatted SdkError>"` text as before; the extra field keeps the
/// transient/definite verdict so that [`retry_get_body`] can still make the
/// decision after the typed `SdkError` has been flattened into a message.
/// Recover it with `anyhow::Error::downcast_ref::<SdkFailure>()`; it survives
/// any number of `.context(..)` layers.
///
/// It also keeps the HTTP `status` and the service error `code` of the failed
/// request (both `None` when the failure carried no HTTP response, or the
/// response carried no S3 error code), so a typed classification
/// ([`classify_s3_failure`]) can be made after the SDK error was flattened.
#[derive(Debug)]
pub(crate) struct SdkFailure {
    message: String,
    retryable: bool,
    status: Option<u16>,
    code: Option<String>,
}

impl SdkFailure {
    pub(crate) fn new(
        message: String,
        retryable: bool,
        status: Option<u16>,
        code: Option<String>,
    ) -> Self {
        Self {
            message,
            retryable,
            status,
            code,
        }
    }

    /// The HTTP status of the failed request, if a response was received.
    pub(crate) fn status(&self) -> Option<u16> {
        self.status
    }

    /// The service error code of the failed request, if the response had one.
    pub(crate) fn code(&self) -> Option<&str> {
        self.code.as_deref()
    }
}

impl std::fmt::Display for SdkFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for SdkFailure {}

/// Decides whether a failed attempt is worth repeating.
///
/// Implemented for `anyhow::Error` (what every S3 caller produces) and, as
/// always-retry, for plain string errors. `retry_get_body` stops at the first
/// error whose `is_retryable()` is false.
pub trait RetryClassify {
    /// `true` when the failure may be transient and the request should be
    /// repeated; `false` when the server gave a definite answer.
    fn is_retryable(&self) -> bool;
}

impl RetryClassify for anyhow::Error {
    /// An error produced from an S3 SDK failure carries its verdict (see
    /// [`SdkFailure`]). Anything else -- a body-transfer error from
    /// `ByteStream::collect`, an I/O error, any error we cannot classify --
    /// is retried: retrying an unknown failure is the conservative choice.
    ///
    /// A typed [`ObjectError`](crate::object_error::ObjectError) is read first
    /// (plan §4.5 A2): NotFound, VersionChanged, Forbidden, LengthMismatch and
    /// Unsupported are never retried, and a generic retryable source below
    /// them cannot override that.  Every other kind (Throttled, Transport,
    /// Server, Other), and an error with no `ObjectError`, keeps the verdict
    /// described above.
    fn is_retryable(&self) -> bool {
        if let Some(o) = crate::object_error::object_error(self) {
            match o.kind {
                ObjectErrorKind::NotFound
                | ObjectErrorKind::VersionChanged
                | ObjectErrorKind::Forbidden
                | ObjectErrorKind::LengthMismatch { .. }
                | ObjectErrorKind::Unsupported => return false,
                ObjectErrorKind::Throttled
                | ObjectErrorKind::Transport
                | ObjectErrorKind::Server
                | ObjectErrorKind::Other => {}
            }
        }
        match self.downcast_ref::<SdkFailure>() {
            Some(f) => f.retryable,
            None => true,
        }
    }
}

/// I/O errors carry no server verdict: retried (the pre-#240 behaviour).
impl RetryClassify for std::io::Error {
    fn is_retryable(&self) -> bool {
        true
    }
}

/// Opaque boxed errors carry no server verdict: retried (the pre-#240
/// behaviour).
impl RetryClassify for Box<dyn std::error::Error + Send + Sync> {
    fn is_retryable(&self) -> bool {
        true
    }
}

impl RetryClassify for &str {
    fn is_retryable(&self) -> bool {
        true
    }
}

impl RetryClassify for String {
    fn is_retryable(&self) -> bool {
        true
    }
}

/// Retry a body-transfer-phase GET a bounded number of times with
/// linear backoff, but only while the failure is transient.
///
/// Each attempt calls `op()` to produce a NEW future — the closure is
/// `FnMut` so callers may capture and mutate their own bookkeeping
/// (a wall-clock start time, an attempt-counter for logs, etc.).
///
/// Correctness contract for the closure body: if `op`'s future writes
/// into a shared mutable buffer (e.g. streaming into a pre-allocated
/// `BytesMut` segment), the write cursor state (`written`, etc.) must
/// be declared INSIDE the async block so each attempt begins at
/// position 0. Otherwise a partially-completed failed attempt can
/// silently corrupt the shared buffer when the retry writes at an
/// offset carried over from the previous attempt.
///
/// Which errors are retried (issue #240): decided by
/// [`RetryClassify::is_retryable`]. For `anyhow::Error` that means
/// connection/dispatch failures, timeouts, 5xx, 408, 429 and S3 throttling
/// codes (`SlowDown`, ...), and body-transfer (streaming) errors are retried;
/// definite answers such as 400, 401, 403, 404, 405, 409, 412 and 416 are
/// returned immediately, with no backoff. An error that cannot be classified
/// (anything that did not come from the S3 SDK) is retried -- the
/// conservative choice. This is the outer s3dlio loop only; the AWS SDK's
/// own retry layer is separate and unchanged.
///
/// Backoff is `100ms * attempt` (no jitter — the retry budget is small and
/// every retry has already burned round-trip time on a real failure).
pub async fn retry_get_body<F, Fut, T, E>(mut op: F) -> Result<T, E>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, E>>,
    E: RetryClassify,
{
    let attempts = max_retry_attempts().max(1);
    let mut last_err: Option<E> = None;
    for attempt in 1..=attempts {
        match op().await {
            Ok(v) => return Ok(v),
            Err(e) => {
                if !e.is_retryable() {
                    return Err(e);
                }
                let is_final = attempt >= attempts;
                last_err = Some(e);
                if !is_final {
                    tokio::time::sleep(Duration::from_millis(100 * attempt as u64)).await;
                }
            }
        }
    }
    // Loop is entered at least once because attempts >= 1, and every
    // Err branch stores into last_err, so an Err return is well-defined.
    Err(last_err.expect("attempts >= 1 guarantees at least one Err was recorded"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Instant;

    // Tests that mutate the process-global `S3DLIO_MAX_RETRY_ATTEMPTS` env var
    // hold the crate-wide guard `crate::test_env` (issue #216): a lock private
    // to this module could not exclude `constants::max_retry_tests`, which sets
    // the same variable.  The guard is async-capable so it can be held across
    // `retry_get_body(...).await` without tripping clippy's `await_holding_lock`.

    /// Phase 4a — budget is respected: after N failed attempts, the
    /// helper gives up and returns the last error rather than looping
    /// forever.
    #[tokio::test]
    async fn retry_respects_max_attempts_budget() {
        let _guard = crate::test_env::lock_async().await;
        let _env =
            crate::test_env::EnvRestore::capture(&[crate::constants::ENV_MAX_RETRY_ATTEMPTS]);
        let call_count = Arc::new(AtomicUsize::new(0));
        // Force a known small budget.
        std::env::set_var(crate::constants::ENV_MAX_RETRY_ATTEMPTS, "3");

        let op = {
            let call_count = Arc::clone(&call_count);
            move || {
                let call_count = Arc::clone(&call_count);
                async move {
                    call_count.fetch_add(1, Ordering::Relaxed);
                    Err::<(), &'static str>("boom")
                }
            }
        };

        let err = retry_get_body(op).await.expect_err("should fail");
        assert_eq!(err, "boom");
        assert_eq!(call_count.load(Ordering::Relaxed), 3);
    }

    /// Phase 4a — first-success path returns immediately without
    /// invoking op() twice.
    #[tokio::test]
    async fn retry_returns_first_success_without_retrying() {
        let _guard = crate::test_env::lock_async().await;
        let _env =
            crate::test_env::EnvRestore::capture(&[crate::constants::ENV_MAX_RETRY_ATTEMPTS]);
        let call_count = Arc::new(AtomicUsize::new(0));
        std::env::set_var(crate::constants::ENV_MAX_RETRY_ATTEMPTS, "5");

        let op = {
            let call_count = Arc::clone(&call_count);
            move || {
                let call_count = Arc::clone(&call_count);
                async move {
                    call_count.fetch_add(1, Ordering::Relaxed);
                    Ok::<u32, &'static str>(42)
                }
            }
        };

        let v = retry_get_body(op).await.expect("should succeed");
        assert_eq!(v, 42);
        assert_eq!(call_count.load(Ordering::Relaxed), 1);
    }

    /// Phase 4a — mid-loop success stops retrying at the first Ok.
    #[tokio::test]
    async fn retry_stops_on_first_success_mid_loop() {
        let _guard = crate::test_env::lock_async().await;
        let _env =
            crate::test_env::EnvRestore::capture(&[crate::constants::ENV_MAX_RETRY_ATTEMPTS]);
        let call_count = Arc::new(AtomicUsize::new(0));
        std::env::set_var(crate::constants::ENV_MAX_RETRY_ATTEMPTS, "5");

        let op = {
            let call_count = Arc::clone(&call_count);
            move || {
                let call_count = Arc::clone(&call_count);
                async move {
                    let n = call_count.fetch_add(1, Ordering::Relaxed);
                    if n < 2 {
                        Err::<&'static str, &'static str>("try again")
                    } else {
                        Ok("ok")
                    }
                }
            }
        };

        let v = retry_get_body(op).await.expect("should succeed on 3rd try");
        assert_eq!(v, "ok");
        assert_eq!(call_count.load(Ordering::Relaxed), 3);
    }

    // ── Issue #240: only transient failures are retried ──────────────────

    use aws_sdk_s3::operation::get_object::GetObjectError;
    use aws_smithy_runtime_api::http::StatusCode;
    use aws_smithy_types::body::SdkBody;
    use aws_smithy_types::error::metadata::ErrorMetadata;

    /// A service error as the SDK would surface it for `status` / `code`,
    /// run through the same flattening the real callers use (`sdk_context`).
    fn service_failure(status: u16, code: &str) -> anyhow::Error {
        use crate::s3_utils::SdkResultExt;
        let meta = ErrorMetadata::builder()
            .code(code)
            .message("injected")
            .build();
        let err = GetObjectError::generic(meta);
        let resp = HttpResponse::new(
            StatusCode::try_from(status).expect("valid status"),
            SdkBody::empty(),
        );
        let r: Result<(), _> = Err(SdkError::service_error(err, resp));
        r.sdk_context("S3 GET failed").expect_err("error")
    }

    #[test]
    fn definite_client_errors_are_not_retryable() {
        for (status, code) in [
            (400, "InvalidArgument"),
            (401, "InvalidAccessKeyId"),
            (403, "AccessDenied"),
            (404, "NoSuchKey"),
            (404, "NoSuchBucket"),
            (405, "MethodNotAllowed"),
            (409, "Conflict"),
            (412, "PreconditionFailed"),
            (416, "InvalidRange"),
        ] {
            assert!(
                !service_failure(status, code).is_retryable(),
                "{status} {code} is a definite answer and must not be retried"
            );
        }
    }

    /// The pinned SDK's own tables are all retryable at the outer classifier, on the
    /// statuses real services attach to them (a 400 and a 403 are the ones that would
    /// otherwise be "definite"). Compares against the crate itself so an SDK upgrade
    /// that adds a code cannot silently shrink the outer retry budget.
    #[test]
    fn aws_runtime_code_tables_are_all_retryable() {
        use aws_runtime::retries::classifiers::{THROTTLING_ERRORS, TRANSIENT_ERRORS};
        for code in THROTTLING_ERRORS.iter().chain(TRANSIENT_ERRORS) {
            for status in [400u16, 403, 409] {
                assert!(
                    service_failure(status, code).is_retryable(),
                    "{status} {code} is in aws-runtime's tables and must be retried"
                );
            }
            assert!(is_transient_error_code(code), "{code}");
        }
        // The copied constants cover the crate's tables exactly: nothing missing, and
        // nothing left over that the SDK no longer lists.
        let mut ours: Vec<&str> = AWS_RUNTIME_THROTTLING_ERROR_CODES
            .iter()
            .chain(AWS_RUNTIME_TRANSIENT_ERROR_CODES)
            .copied()
            .collect();
        let mut theirs: Vec<&str> = THROTTLING_ERRORS
            .iter()
            .chain(TRANSIENT_ERRORS)
            .copied()
            .collect();
        ours.sort_unstable();
        theirs.sort_unstable();
        assert_eq!(
            ours, theirs,
            "update the constants from the pinned aws-runtime"
        );
    }

    /// Blue's probe: a 400 `RequestTimeoutException` is the final error of an SDK
    /// operation whose attempts are spent; the outer loop must still retry it.
    #[test]
    fn request_timeout_exception_is_retryable() {
        assert!(service_failure(400, "RequestTimeoutException").is_retryable());
        assert!(service_failure(400, "ThrottledException").is_retryable());
        assert!(service_failure(403, "RequestThrottledException").is_retryable());
        assert!(service_failure(400, "TransactionInProgressException").is_retryable());
        assert!(service_failure(400, "LimitExceededException").is_retryable());
        assert!(service_failure(400, "PriorRequestNotComplete").is_retryable());
        assert!(service_failure(400, "EC2ThrottledException").is_retryable());
        // Codes that really are definite stay definite.
        assert!(!service_failure(400, "InvalidArgument").is_retryable());
        assert!(!service_failure(400, "RequestTimeTooSkewed").is_retryable());
    }

    #[test]
    fn transient_failures_are_retryable() {
        for (status, code) in [
            (500, "InternalError"),
            (502, "BadGateway"),
            (503, "SlowDown"),
            (503, "ServiceUnavailable"),
            (504, "GatewayTimeout"),
            (408, "RequestTimeout"),
            (429, "TooManyRequests"),
            // S3 reports these codes under a 400.
            (400, "RequestTimeout"),
            (400, "SlowDown"),
            // Credential expiry and S3's "retry me" abort: a later attempt
            // can succeed once the SDK has refreshed its identity.
            (400, "ExpiredToken"),
            (403, "ExpiredToken"),
            (400, "RequestExpired"),
            (409, "OperationAborted"),
        ] {
            assert!(
                service_failure(status, code).is_retryable(),
                "{status} {code} is transient and must be retried"
            );
        }
    }

    #[test]
    fn dispatch_timeout_and_unknown_errors_are_retryable() {
        use aws_smithy_runtime_api::client::result::ConnectorError;
        let dispatch: SdkError<GetObjectError, HttpResponse> =
            SdkError::dispatch_failure(ConnectorError::io("connection reset".into()));
        assert!(sdk_error_is_retryable(&dispatch));
        let timeout: SdkError<GetObjectError, HttpResponse> = SdkError::timeout_error("timed out");
        assert!(sdk_error_is_retryable(&timeout));
        let build: SdkError<GetObjectError, HttpResponse> =
            SdkError::construction_failure("bad request");
        assert!(!sdk_error_is_retryable(&build));
        // A body-transfer error and any plain error carry no verdict: retried.
        assert!(anyhow::anyhow!("body stream failed").is_retryable());
        assert!("boom".is_retryable());
        assert!(String::from("boom").is_retryable());
        assert!(std::io::Error::other("reset").is_retryable());
        let boxed: Box<dyn std::error::Error + Send + Sync> = "boom".into();
        assert!(boxed.is_retryable());
    }

    #[test]
    fn verdict_survives_context_layers_and_display_is_unchanged() {
        use anyhow::Context;
        let e = service_failure(404, "NoSuchKey");
        assert!(e
            .to_string()
            .starts_with("S3 GET failed: service error: NoSuchKey"));
        let wrapped: anyhow::Error = Err::<(), _>(e)
            .context("outer")
            .context("outermost")
            .unwrap_err();
        assert!(!wrapped.is_retryable());
    }

    /// A definite error ends the loop after ONE attempt, with no backoff.
    #[tokio::test]
    async fn definite_error_is_returned_after_one_attempt() {
        let _guard = crate::test_env::lock_async().await;
        let _env =
            crate::test_env::EnvRestore::capture(&[crate::constants::ENV_MAX_RETRY_ATTEMPTS]);
        std::env::set_var(crate::constants::ENV_MAX_RETRY_ATTEMPTS, "5");
        let calls = Arc::new(AtomicUsize::new(0));
        let start = Instant::now();
        let op = {
            let calls = Arc::clone(&calls);
            move || {
                let calls = Arc::clone(&calls);
                async move {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Err::<(), anyhow::Error>(service_failure(404, "NoSuchKey"))
                }
            }
        };
        let err = retry_get_body(op).await.expect_err("404 fails");
        assert!(format!("{err:#}").contains("NoSuchKey"));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        assert!(start.elapsed() < Duration::from_millis(100), "no backoff");
    }

    /// A transient error is still retried up to the budget, then returned.
    #[tokio::test]
    async fn transient_error_uses_the_whole_budget() {
        let _guard = crate::test_env::lock_async().await;
        let _env =
            crate::test_env::EnvRestore::capture(&[crate::constants::ENV_MAX_RETRY_ATTEMPTS]);
        std::env::set_var(crate::constants::ENV_MAX_RETRY_ATTEMPTS, "3");
        let calls = Arc::new(AtomicUsize::new(0));
        let op = {
            let calls = Arc::clone(&calls);
            move || {
                let calls = Arc::clone(&calls);
                async move {
                    calls.fetch_add(1, Ordering::Relaxed);
                    Err::<(), anyhow::Error>(service_failure(503, "SlowDown"))
                }
            }
        };
        retry_get_body(op).await.expect_err("503 keeps failing");
        assert_eq!(calls.load(Ordering::Relaxed), 3);
    }

    /// Phase 4a — linear backoff timing: two failures then a success
    /// should sleep 100ms after attempt 1 and 200ms after attempt 2.
    /// Total elapsed ≥ 300ms.
    #[tokio::test]
    async fn retry_applies_linear_backoff() {
        let _guard = crate::test_env::lock_async().await;
        let _env =
            crate::test_env::EnvRestore::capture(&[crate::constants::ENV_MAX_RETRY_ATTEMPTS]);
        std::env::set_var(crate::constants::ENV_MAX_RETRY_ATTEMPTS, "5");
        let call_count = Arc::new(AtomicUsize::new(0));
        let start = Instant::now();

        let op = {
            let call_count = Arc::clone(&call_count);
            move || {
                let call_count = Arc::clone(&call_count);
                async move {
                    let n = call_count.fetch_add(1, Ordering::Relaxed);
                    if n < 2 {
                        Err::<(), &'static str>("try again")
                    } else {
                        Ok(())
                    }
                }
            }
        };

        retry_get_body(op).await.expect("should succeed");
        let elapsed = start.elapsed();
        assert!(
            elapsed >= Duration::from_millis(300),
            "linear backoff should sleep 100ms + 200ms between the 3 attempts; \
             elapsed = {:?}. If this is much lower, the backoff isn't firing.",
            elapsed
        );
    }

    // ── Plan §4.5 A2: typed S3 classification and the ObjectError-first verdict ──

    use crate::object_error::{attach_object_error, object_error};

    fn kind_of(err: &anyhow::Error) -> ObjectErrorKind {
        object_error(err).expect("classified").kind
    }

    fn classified(status: u16, code: &str, pinned: bool) -> anyhow::Error {
        attach_s3_classification(service_failure(status, code), pinned)
    }

    /// `sdk_anyhow` keeps the HTTP status and the service code next to the verdict.
    #[test]
    fn sdk_failure_carries_status_and_code() {
        let e = service_failure(403, "ExpiredToken");
        let f = e.downcast_ref::<SdkFailure>().expect("SdkFailure");
        assert_eq!(f.status(), Some(403));
        assert_eq!(f.code(), Some("ExpiredToken"));
        // A failure with no HTTP response has neither.
        let dispatch: SdkError<GetObjectError, HttpResponse> = SdkError::dispatch_failure(
            aws_smithy_runtime_api::client::result::ConnectorError::io("reset".into()),
        );
        let e = crate::s3_utils::sdk_anyhow("S3 GET failed", dispatch);
        let f = e.downcast_ref::<SdkFailure>().expect("SdkFailure");
        assert_eq!((f.status(), f.code()), (None, None));
        assert!(e.is_retryable());
    }

    /// The seven-step table, one row per step, in order.
    #[test]
    fn s3_table_steps_in_order() {
        use ObjectErrorKind::*;
        let c = |s, code, pinned| classify_s3_failure(s, code, pinned);
        // 1. a transient code wins even on a 403: throttling table -> Throttled, other -> Server.
        assert_eq!(c(Some(403), Some("ExpiredToken"), false), Server);
        assert_eq!(c(Some(403), Some("RequestExpired"), false), Server);
        assert_eq!(c(Some(400), Some("SlowDown"), false), Throttled);
        assert_eq!(
            c(Some(403), Some("RequestThrottledException"), false),
            Throttled
        );
        assert_eq!(c(Some(400), Some("RequestTimeout"), false), Server);
        assert_eq!(c(Some(409), Some("OperationAborted"), false), Server);
        // 2. absence by status or by code.
        assert_eq!(c(Some(404), Some("NoSuchKey"), false), NotFound);
        assert_eq!(c(Some(404), None, false), NotFound);
        assert_eq!(c(Some(400), Some("NotFound"), false), NotFound);
        // 3. 412.
        assert_eq!(
            c(Some(412), Some("PreconditionFailed"), false),
            VersionChanged
        );
        // 4. any remaining 403.
        assert_eq!(c(Some(403), Some("AccessDenied"), false), Forbidden);
        assert_eq!(c(Some(403), None, false), Forbidden);
        // 5. 429/503 throttled; 408 and other 5xx server.
        assert_eq!(c(Some(429), None, false), Throttled);
        assert_eq!(c(Some(503), None, false), Throttled);
        assert_eq!(c(Some(408), None, false), Server);
        assert_eq!(c(Some(500), Some("InternalError"), false), Server);
        assert_eq!(c(Some(502), None, false), Server);
        // 6. no HTTP response.
        assert_eq!(c(None, None, false), Transport);
        // 7. the rest.
        assert_eq!(c(Some(400), Some("InvalidArgument"), false), Other);
        assert_eq!(c(Some(416), Some("InvalidRange"), false), Other);
        assert_eq!(c(Some(301), None, false), Other);
    }

    /// The pinned override takes precedence over the table; other statuses are unaffected.
    #[test]
    fn s3_pinned_404_and_412_are_version_changed() {
        use ObjectErrorKind::*;
        assert_eq!(
            classify_s3_failure(Some(404), Some("NoSuchKey"), true),
            VersionChanged
        );
        assert_eq!(classify_s3_failure(Some(404), None, true), VersionChanged);
        assert_eq!(classify_s3_failure(Some(412), None, true), VersionChanged);
        // Over a transient code, the override still wins.
        assert_eq!(
            classify_s3_failure(Some(404), Some("SlowDown"), true),
            VersionChanged
        );
        // Not-found by code alone is also absence of the pinned version.
        assert_eq!(
            classify_s3_failure(None, Some("NoSuchKey"), true),
            VersionChanged
        );
        // Ordinary failures are classified as without a pin.
        assert_eq!(
            classify_s3_failure(Some(403), Some("AccessDenied"), true),
            Forbidden
        );
        assert_eq!(classify_s3_failure(Some(503), None, true), Throttled);
        assert_eq!(classify_s3_failure(None, None, true), Transport);
        // End to end, with the verdict.
        let e = classified(404, "NoSuchKey", true);
        assert_eq!(kind_of(&e), VersionChanged);
        assert!(!e.is_retryable());
        let e = classified(412, "PreconditionFailed", true);
        assert_eq!(kind_of(&e), VersionChanged);
        assert!(!e.is_retryable());
        // Unpinned (the first request): absence.
        assert_eq!(kind_of(&classified(404, "NoSuchKey", false)), NotFound);
    }

    /// 403 `ExpiredToken` and `RequestExpired` stay retryable; an ordinary 403 is Forbidden
    /// and is not retried.
    #[test]
    fn s3_expired_credentials_retry_but_ordinary_403_does_not() {
        for code in ["ExpiredToken", "RequestExpired"] {
            let e = classified(403, code, false);
            assert!(e.is_retryable(), "403 {code} must stay retryable");
            assert_ne!(kind_of(&e), ObjectErrorKind::Forbidden);
        }
        let e = classified(403, "AccessDenied", false);
        assert_eq!(kind_of(&e), ObjectErrorKind::Forbidden);
        assert!(!e.is_retryable());
    }

    /// An `Other`-tagged definite 400 keeps today's `SdkFailure` verdict: not retried.
    /// Transient kinds keep today's verdict too (retried).
    #[test]
    fn other_and_transient_kinds_fall_through_to_todays_verdict() {
        let e = classified(400, "InvalidArgument", false);
        assert_eq!(kind_of(&e), ObjectErrorKind::Other);
        assert!(!e.is_retryable(), "a definite 400 is not retried");
        for (status, code) in [
            (503, "SlowDown"),
            (500, "InternalError"),
            (408, "RequestTimeout"),
        ] {
            let e = classified(status, code, false);
            assert!(e.is_retryable(), "{status} {code}");
        }
        // No SdkFailure and no ObjectError: today's conservative default.
        assert!(anyhow::anyhow!("body stream failed").is_retryable());
        // A body-stream error has no HTTP response: Transport, and still retried.
        let body = attach_s3_classification(anyhow::anyhow!("body stream failed"), false);
        assert_eq!(kind_of(&body), ObjectErrorKind::Transport);
        assert!(body.is_retryable());
        // Other over a source with no verdict also falls through to "retry".
        let untyped = attach_object_error(anyhow::anyhow!("x"), ObjectErrorKind::Other, "s3", "");
        assert!(untyped.is_retryable());
    }

    /// Length errors, absence, version changes, forbidden and unsupported are never retried,
    /// even over a source that would be retried on its own.
    #[test]
    fn non_retryable_kinds_override_a_retryable_source() {
        for kind in [
            ObjectErrorKind::NotFound,
            ObjectErrorKind::VersionChanged,
            ObjectErrorKind::Forbidden,
            ObjectErrorKind::Unsupported,
            ObjectErrorKind::LengthMismatch {
                expected: 8,
                actual: Some(4),
            },
            ObjectErrorKind::LengthMismatch {
                expected: 8,
                actual: None,
            },
        ] {
            // A retryable SdkFailure (503) below the typed context.
            let over_sdk = attach_object_error(service_failure(503, "SlowDown"), kind, "s3", "");
            assert!(!over_sdk.is_retryable(), "{kind:?} over a 503");
            // An untyped source (default "retry") below it.
            let over_plain = attach_object_error(anyhow::anyhow!("reset"), kind, "s3", "");
            assert!(!over_plain.is_retryable(), "{kind:?} over a plain error");
            // And through further string contexts.
            let nested = {
                use anyhow::Context;
                Err::<(), _>(over_plain)
                    .context("a")
                    .context("b")
                    .unwrap_err()
            };
            assert!(!nested.is_retryable(), "{kind:?} nested");
        }
        // Throttled / Transport / Server over a plain source are retried (unchanged default).
        for kind in [
            ObjectErrorKind::Throttled,
            ObjectErrorKind::Transport,
            ObjectErrorKind::Server,
        ] {
            let e = attach_object_error(anyhow::anyhow!("x"), kind, "s3", "");
            assert!(e.is_retryable(), "{kind:?}");
        }
    }

    /// A definite typed error ends `retry_get_body` after ONE attempt, with no backoff.
    #[tokio::test]
    async fn typed_forbidden_and_length_errors_use_one_attempt() {
        let _guard = crate::test_env::lock_async().await;
        let _env =
            crate::test_env::EnvRestore::capture(&[crate::constants::ENV_MAX_RETRY_ATTEMPTS]);
        std::env::set_var(crate::constants::ENV_MAX_RETRY_ATTEMPTS, "5");
        for kind in [
            ObjectErrorKind::Forbidden,
            ObjectErrorKind::LengthMismatch {
                expected: 2,
                actual: Some(1),
            },
        ] {
            let calls = Arc::new(AtomicUsize::new(0));
            let op = {
                let calls = Arc::clone(&calls);
                move || {
                    let calls = Arc::clone(&calls);
                    async move {
                        calls.fetch_add(1, Ordering::Relaxed);
                        Err::<(), anyhow::Error>(attach_object_error(
                            anyhow::anyhow!("reset"),
                            kind,
                            "s3",
                            "",
                        ))
                    }
                }
            };
            retry_get_body(op).await.expect_err("fails");
            assert_eq!(calls.load(Ordering::Relaxed), 1, "{kind:?}");
        }
    }
}
