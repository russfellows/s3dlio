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

/// S3 error codes that mean "try again later" whatever the HTTP status the
/// server attached to them (S3 itself reports `RequestTimeout` as a 400 and
/// some S3-compatible servers report `SlowDown` as a 4xx).
const TRANSIENT_S3_ERROR_CODES: &[&str] = &[
    "SlowDown",
    "Throttling",
    "ThrottlingException",
    "RequestThrottled",
    "RequestLimitExceeded",
    "TooManyRequestsException",
    "ProvisionedThroughputExceededException",
    "BandwidthLimitExceeded",
    "RequestTimeout",
    "InternalError",
    "ServiceUnavailable",
    // Credential expiry: the SDK re-resolves its identity (and may refresh
    // STS credentials) on the next attempt, so a later attempt can succeed.
    "ExpiredToken",
    "RequestExpired",
    // S3 documents this one as "retry the request".
    "OperationAborted",
];

/// Is an HTTP status, with an optional S3 error code, worth retrying?
///
/// Transient: 408, 429, every 5xx, and any status carrying one of
/// [`TRANSIENT_S3_ERROR_CODES`]. Definite (not retried): every other 3xx and
/// 4xx (400, 401, 403, 404, 405, 409, 412, 416, ...). Statuses outside
/// 300-599 cannot be a definite client error, so they are retried.
pub(crate) fn status_is_retryable(status: u16, code: Option<&str>) -> bool {
    if let Some(code) = code {
        if TRANSIENT_S3_ERROR_CODES.contains(&code) {
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

/// The anyhow error that [`crate::s3_utils`]'s `sdk_context` / `sdk_anyhow`
/// produce for a failed S3 request. Its `Display` is the same flattened
/// `"<ctx>: <formatted SdkError>"` text as before; the extra field keeps the
/// transient/definite verdict so that [`retry_get_body`] can still make the
/// decision after the typed `SdkError` has been flattened into a message.
/// Recover it with `anyhow::Error::downcast_ref::<SdkFailure>()`; it survives
/// any number of `.context(..)` layers.
#[derive(Debug)]
pub(crate) struct SdkFailure {
    message: String,
    retryable: bool,
}

impl SdkFailure {
    pub(crate) fn new(message: String, retryable: bool) -> Self {
        Self { message, retryable }
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
    fn is_retryable(&self) -> bool {
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
}
