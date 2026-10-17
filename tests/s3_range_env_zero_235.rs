// tests/s3_range_env_zero_235.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// RED-then-GREEN regression tests for s3dlio issue #235.
//
// Bug: the zero-limit checks of #211 validate caller ARGUMENTS only. A zero taken from the
// environment was not validated:
//   * S3DLIO_CHUNK_SIZE=0 made the concurrent range GET loop forever building ranges
//     (unbounded memory);
//   * S3DLIO_RANGE_CONCURRENCY=0 created a semaphore with no permits, so every range waited
//     forever.
//
// Contract under test: a zero from either variable is rejected with an error that names the
// variable, before any range is allocated or any semaphore is awaited; valid settings still
// download the object unchanged.
//
// Mechanism: each case runs in its OWN child process (this test binary re-executed on one
// `--exact` test), against the in-process mock S3 server, under a hard wall-clock timeout and
// an address-space limit. The unfixed code either spins allocating ranges (the limit turns
// that into an abort instead of exhausting the host) or hangs on the empty semaphore (the
// timeout turns that into a failure); neither can freeze or exhaust the test run.

// The address-space limit (`ulimit -v`) is a Linux mechanism, so the file is Linux-only.
#![cfg(target_os = "linux")]

// Only the mock server file is included, not all of tests/common.  This binary uses part of
// the mock's API (a few of its request-recording fields and helper methods are unused here),
// so the expectation is on this one module and fails if the warnings ever stop.
#[path = "common/mock_s3_server.rs"]
#[expect(
    dead_code,
    reason = "this binary uses only part of the shared mock server API"
)]
mod mock_s3_server;

#[path = "common/child_case.rs"]
mod child_case;

use mock_s3_server::{
    ensure_mock_s3_server, key_path, unique_test_key, SIZED_OBJECT_BODY, SIZED_OBJECT_MARKER,
};
use std::time::Duration;

const BUCKET: &str = "mock-bucket";

/// Environment variable that marks a re-executed child and selects what it does.
const CHILD_CASE: &str = "S3DLIO_TEST_235_CASE";
/// What the child must observe: `reject:<VARIABLE>` or `ok`.
const CHILD_EXPECT: &str = "S3DLIO_TEST_235_EXPECT";
/// Printed by the child when its assertions held.
const SENTINEL: &str = "S3DLIO_235_CHILD_VERIFIED";

/// Hard wall-clock limit for one child.
const HARD_TIMEOUT: Duration = Duration::from_secs(45);
/// Address-space limit (KiB) for one child: generous for the runtime and SDK threads, small
/// enough that an unbounded range vector aborts quickly rather than exhausting the host.
const CHILD_ADDRESS_SPACE_KIB: u64 = 3 * 1024 * 1024;

// ─────────────────────────────── child side ───────────────────────────────

/// Runs only when re-executed by [`run_case`]; a plain `cargo test` finds no marker and returns.
#[test]
fn child_run_case() {
    let Ok(case) = std::env::var(CHILD_CASE) else {
        return;
    };
    let expect = std::env::var(CHILD_EXPECT).expect("the parent sets the expectation");

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("runtime");
    let state = ensure_mock_s3_server().1;
    // The mock reports a 16-byte object for a key with the sized marker, and an EMPTY object
    // (HEAD says 0 bytes) for every other key.
    let key = if case.ends_with("_empty") {
        unique_test_key("empty-obj")
    } else {
        unique_test_key(SIZED_OBJECT_MARKER)
    };
    let uri = format!("s3://{BUCKET}/{key}");

    let result = rt.block_on(async {
        let ranged = |offset: u64| {
            s3dlio::s3_utils::get_object_concurrent_range_async(&uri, offset, None, None, None)
        };
        // `get_object_concurrent_range_async` is `get_object_concurrent_range_with_client` on
        // the process-wide client, so these reach the per-endpoint entry point's own checks.
        let ranged_with = |offset: u64,
                           length: Option<u64>,
                           chunk_size: Option<usize>,
                           max_concurrency: Option<usize>| {
            s3dlio::s3_utils::get_object_concurrent_range_async(
                &uri,
                offset,
                length,
                chunk_size,
                max_concurrency,
            )
        };
        match case.as_str() {
            "concurrent_range" | "concurrent_range_empty" => ranged(0).await,
            // An offset and a length at `u64::MAX`: `offset + length` must saturate (not wrap
            // or panic), the range is empty, and the limits are still validated first.
            "max_offset_max_length" => ranged_with(u64::MAX, Some(u64::MAX), None, None).await,
            // The same call with explicit, valid limits: an explicit argument wins over a zero
            // in the environment.
            "max_offset_max_length_explicit" => {
                ranged_with(u64::MAX, Some(u64::MAX), Some(4), Some(2)).await
            }
            // A length at `u64::MAX` from offset 8 saturates to "the rest": the last 8 bytes.
            "tail_max_length" => ranged_with(8, Some(u64::MAX), None, None).await,
            // Exactly at the end of the 16-byte object, and well beyond it.
            "concurrent_range_at_eof" => ranged(SIZED_OBJECT_BODY.len() as u64).await,
            "concurrent_range_past_eof" => ranged(1000).await,
            // Threshold 0 sends the (16-byte, or empty) object through the known-size range path.
            "uri_optimized" | "uri_optimized_empty" => {
                s3dlio::s3_utils::get_object_uri_optimized_async(&uri).await
            }
            // The caller-buffer receive path (plan §4.5 A8) uses the same validation: a fan-out
            // read (threshold 0) of the registered 16-byte object, with an ETag to pin to.
            "get_into" => {
                state.put_object(
                    BUCKET,
                    &key,
                    mock_s3_server::MockObject::new(SIZED_OBJECT_BODY.to_vec()),
                );
                let client = s3dlio::s3_client::aws_s3_client_async().await?;
                let ops = s3dlio::s3_ops::S3Ops::new(client, None, "", "");
                let mut dst = vec![0u8; SIZED_OBJECT_BODY.len()];
                let progress = s3dlio::receive_progress::ReceiveProgress::new();
                ops.get_into(BUCKET, &key, &mut dst, &progress)
                    .await
                    .map(|()| bytes::Bytes::from(dst))
            }
            other => panic!("unknown case {other}"),
        }
    });

    match expect.strip_prefix("reject:") {
        Some(variable) => {
            let err = result.expect_err("a zero setting must be rejected, not downloaded");
            let msg = format!("{err:#}");
            assert!(msg.contains(variable), "error must name {variable}: {msg}");
            assert!(
                msg.contains(">= 1"),
                "error must give the valid range: {msg}"
            );
            // Rejected before any range was requested.
            let counts = state.counts_for(&key_path(BUCKET, &key));
            assert_eq!(
                counts.get_calls, 0,
                "no GET may be sent for a rejected setting"
            );
            if case == "get_into" {
                // The receive path validates before its HEAD as well.
                assert_eq!(counts.head_calls, 0, "no HEAD may be sent either");
            }
        }
        None if expect == "ok-empty" => {
            // Valid settings: an empty object, or a range at or past the end, is still an empty
            // success (the validation added by #235 must not turn it into an error).
            let body = result.expect("valid settings must still succeed");
            assert!(body.is_empty(), "expected no bytes, got {}", body.len());
        }
        None if expect == "ok-tail" => {
            let body = result.expect("valid settings must download the tail");
            assert_eq!(&body[..], &SIZED_OBJECT_BODY[8..], "the last 8 bytes");
        }
        None => {
            assert_eq!(expect, "ok");
            let body = result.expect("valid settings must still download the object");
            assert_eq!(&body[..], &SIZED_OBJECT_BODY[..], "bytes must be unchanged");
        }
    }
    println!("{SENTINEL}");
}

// ─────────────────────────────── parent side ───────────────────────────────

/// How this binary's children are run (the harness is shared: `tests/common/child_case.rs`).
const SPEC: child_case::ChildSpec = child_case::ChildSpec {
    case_var: CHILD_CASE,
    expect_var: CHILD_EXPECT,
    sentinel: SENTINEL,
    child_test: "child_run_case",
    hard_timeout: HARD_TIMEOUT,
    address_space_kib: CHILD_ADDRESS_SPACE_KIB,
    failure_hint: "a crash here is the unbounded range vector hitting the address-space limit",
};

fn run_case(case: &str, expect: &str, env: &[(&str, &str)]) -> child_case::ChildOutcome {
    child_case::run_case(&SPEC, case, expect, env)
}

fn assert_verified(what: &str, outcome: child_case::ChildOutcome) {
    child_case::assert_verified(&SPEC, what, outcome)
}

#[test]
fn chunk_size_env_zero_is_rejected_by_name_on_the_concurrent_range_path() {
    assert_verified(
        "S3DLIO_CHUNK_SIZE=0, get_object_concurrent_range_async",
        run_case(
            "concurrent_range",
            "reject:S3DLIO_CHUNK_SIZE",
            &[("S3DLIO_CHUNK_SIZE", "0")],
        ),
    );
}

#[test]
fn range_concurrency_env_zero_is_rejected_by_name_on_the_concurrent_range_path() {
    // A valid chunk size of 4 bytes splits the 16-byte object into four ranges, so the
    // concurrency limit is reached.
    assert_verified(
        "S3DLIO_RANGE_CONCURRENCY=0, get_object_concurrent_range_async",
        run_case(
            "concurrent_range",
            "reject:S3DLIO_RANGE_CONCURRENCY",
            &[
                ("S3DLIO_CHUNK_SIZE", "4"),
                ("S3DLIO_RANGE_CONCURRENCY", "0"),
            ],
        ),
    );
}

#[test]
fn chunk_size_env_zero_is_rejected_by_name_on_the_known_size_path() {
    assert_verified(
        "S3DLIO_CHUNK_SIZE=0, get_object_uri_optimized_async",
        run_case(
            "uri_optimized",
            "reject:S3DLIO_CHUNK_SIZE",
            &[
                ("S3DLIO_RANGE_THRESHOLD_MB", "0"),
                ("S3DLIO_CHUNK_SIZE", "0"),
            ],
        ),
    );
}

#[test]
fn range_concurrency_env_zero_is_rejected_by_name_on_the_known_size_path() {
    assert_verified(
        "S3DLIO_RANGE_CONCURRENCY=0, get_object_uri_optimized_async",
        run_case(
            "uri_optimized",
            "reject:S3DLIO_RANGE_CONCURRENCY",
            &[
                ("S3DLIO_RANGE_THRESHOLD_MB", "0"),
                ("S3DLIO_CHUNK_SIZE", "4"),
                ("S3DLIO_RANGE_CONCURRENCY", "0"),
            ],
        ),
    );
}

#[test]
fn valid_settings_still_download_the_object_unchanged() {
    assert_verified(
        "valid S3DLIO_CHUNK_SIZE=4 / S3DLIO_RANGE_CONCURRENCY=2, get_object_concurrent_range_async",
        run_case(
            "concurrent_range",
            "ok",
            &[
                ("S3DLIO_CHUNK_SIZE", "4"),
                ("S3DLIO_RANGE_CONCURRENCY", "2"),
            ],
        ),
    );
    assert_verified(
        "valid settings, get_object_uri_optimized_async",
        run_case(
            "uri_optimized",
            "ok",
            &[
                ("S3DLIO_RANGE_THRESHOLD_MB", "0"),
                ("S3DLIO_CHUNK_SIZE", "4"),
                ("S3DLIO_RANGE_CONCURRENCY", "2"),
            ],
        ),
    );
}

/// The object is 16 bytes and the default chunk is 1 MiB, so the object fits one chunk and
/// the concurrency setting is never needed.  It is still an invalid setting: whether it is
/// rejected must not depend on the object's size.
#[test]
fn range_concurrency_env_zero_is_rejected_for_a_small_object_on_the_concurrent_range_path() {
    assert_verified(
        "S3DLIO_RANGE_CONCURRENCY=0 (small object), get_object_concurrent_range_async",
        run_case(
            "concurrent_range",
            "reject:S3DLIO_RANGE_CONCURRENCY",
            &[("S3DLIO_RANGE_CONCURRENCY", "0")],
        ),
    );
}

#[test]
fn range_concurrency_env_zero_is_rejected_for_a_small_object_on_the_known_size_path() {
    assert_verified(
        "S3DLIO_RANGE_CONCURRENCY=0 (small object), get_object_uri_optimized_async",
        run_case(
            "uri_optimized",
            "reject:S3DLIO_RANGE_CONCURRENCY",
            &[
                ("S3DLIO_RANGE_THRESHOLD_MB", "0"),
                ("S3DLIO_RANGE_CONCURRENCY", "0"),
            ],
        ),
    );
}

/// A concurrency above what a semaphore can hold would panic in `Semaphore::new`; it is
/// rejected by name instead, for a small object and for one that is split into ranges.
#[test]
fn range_concurrency_env_above_the_semaphore_limit_is_rejected_by_name() {
    let too_many = usize::MAX.to_string();
    assert_verified(
        "S3DLIO_RANGE_CONCURRENCY=usize::MAX (small object), get_object_concurrent_range_async",
        run_case(
            "concurrent_range",
            "reject:S3DLIO_RANGE_CONCURRENCY",
            &[("S3DLIO_RANGE_CONCURRENCY", &too_many)],
        ),
    );
    assert_verified(
        "S3DLIO_RANGE_CONCURRENCY=usize::MAX (split object), get_object_uri_optimized_async",
        run_case(
            "uri_optimized",
            "reject:S3DLIO_RANGE_CONCURRENCY",
            &[
                ("S3DLIO_RANGE_THRESHOLD_MB", "0"),
                ("S3DLIO_CHUNK_SIZE", "4"),
                ("S3DLIO_RANGE_CONCURRENCY", &too_many),
            ],
        ),
    );
}

// ───────────── empty objects and ranges at or past the end (stage B follow-up F1) ─────────────
//
// An empty object, or a range that starts at or after the end, returns an empty success before
// any range is built.  That shortcut used to run BEFORE the environment-derived chunk size and
// concurrency were resolved, so a zero setting escaped the validation for exactly those calls.
// The validation must not depend on the object's size or the offset.

/// One reject case per (entry path, shape of the call, variable).
fn assert_zero_rejected_for(case: &str, shape: &str) {
    let threshold: &[(&str, &str)] = if case.starts_with("uri_optimized") {
        &[("S3DLIO_RANGE_THRESHOLD_MB", "0")]
    } else {
        &[]
    };
    for variable in ["S3DLIO_CHUNK_SIZE", "S3DLIO_RANGE_CONCURRENCY"] {
        let mut env: Vec<(&str, &str)> = threshold.to_vec();
        env.push((variable, "0"));
        assert_verified(
            &format!("{variable}=0, {shape}, {case}"),
            run_case(case, &format!("reject:{variable}"), &env),
        );
    }
}

#[test]
fn zero_env_settings_are_rejected_for_an_empty_object_on_the_concurrent_range_path() {
    assert_zero_rejected_for("concurrent_range_empty", "empty object");
}

#[test]
fn zero_env_settings_are_rejected_for_an_empty_object_on_the_known_size_path() {
    assert_zero_rejected_for("uri_optimized_empty", "empty object");
}

#[test]
fn zero_env_settings_are_rejected_for_an_offset_at_the_end_on_the_concurrent_range_path() {
    assert_zero_rejected_for("concurrent_range_at_eof", "offset at the end");
}

#[test]
fn zero_env_settings_are_rejected_for_an_offset_past_the_end_on_the_concurrent_range_path() {
    assert_zero_rejected_for("concurrent_range_past_eof", "offset past the end");
}

/// Controls: valid settings (and the defaults) still give an empty success for the same calls,
/// so the validation added above is not a blanket refusal.
#[test]
fn valid_settings_still_give_an_empty_success_for_an_empty_object_and_for_offsets_at_and_past_the_end(
) {
    let valid = [
        ("S3DLIO_CHUNK_SIZE", "4"),
        ("S3DLIO_RANGE_CONCURRENCY", "2"),
    ];
    for case in [
        "concurrent_range_empty",
        "concurrent_range_at_eof",
        "concurrent_range_past_eof",
    ] {
        assert_verified(
            &format!("valid settings, {case}"),
            run_case(case, "ok-empty", &valid),
        );
        assert_verified(
            &format!("default settings, {case}"),
            run_case(case, "ok-empty", &[]),
        );
    }
    let mut known_size = vec![("S3DLIO_RANGE_THRESHOLD_MB", "0")];
    known_size.extend(valid);
    assert_verified(
        "valid settings, uri_optimized_empty",
        run_case("uri_optimized_empty", "ok-empty", &known_size),
    );
}

// ───────────── the caller-buffer receive path (plan §4.5 A8) uses the same validation ─────────────

#[test]
fn zero_env_settings_are_rejected_by_name_before_any_request_on_the_get_into_path() {
    for variable in ["S3DLIO_CHUNK_SIZE", "S3DLIO_RANGE_CONCURRENCY"] {
        assert_verified(
            &format!("{variable}=0, get_into fan-out"),
            run_case(
                "get_into",
                &format!("reject:{variable}"),
                &[("S3DLIO_RANGE_THRESHOLD_MB", "0"), (variable, "0")],
            ),
        );
    }
}

#[test]
fn valid_settings_still_fill_the_caller_buffer_on_the_get_into_path() {
    assert_verified(
        "valid settings, get_into fan-out",
        run_case(
            "get_into",
            "ok",
            &[
                ("S3DLIO_RANGE_THRESHOLD_MB", "0"),
                ("S3DLIO_CHUNK_SIZE", "4"),
                ("S3DLIO_RANGE_CONCURRENCY", "2"),
            ],
        ),
    );
}

// ───────────── an offset and a length at `u64::MAX` on the per-endpoint entry point ─────────────
//
// `get_object_concurrent_range_async` runs `get_object_concurrent_range_with_client` (the entry
// point of a per-endpoint store).  `offset + length` there saturates, the limits are resolved
// before the empty-range shortcut, and an explicit argument beats the environment.

#[test]
fn zero_env_settings_are_rejected_by_name_for_offset_and_length_at_u64_max() {
    for variable in ["S3DLIO_CHUNK_SIZE", "S3DLIO_RANGE_CONCURRENCY"] {
        assert_verified(
            &format!("{variable}=0, offset and length u64::MAX"),
            run_case(
                "max_offset_max_length",
                &format!("reject:{variable}"),
                &[(variable, "0")],
            ),
        );
    }
}

#[test]
fn valid_settings_give_an_empty_success_for_offset_and_length_at_u64_max() {
    assert_verified(
        "valid settings, offset and length u64::MAX",
        run_case(
            "max_offset_max_length",
            "ok-empty",
            &[
                ("S3DLIO_CHUNK_SIZE", "4"),
                ("S3DLIO_RANGE_CONCURRENCY", "2"),
            ],
        ),
    );
}

#[test]
fn explicit_valid_arguments_win_over_a_zero_environment_for_offset_and_length_at_u64_max() {
    assert_verified(
        "explicit arguments over a zero environment, offset and length u64::MAX",
        run_case(
            "max_offset_max_length_explicit",
            "ok-empty",
            &[
                ("S3DLIO_CHUNK_SIZE", "0"),
                ("S3DLIO_RANGE_CONCURRENCY", "0"),
            ],
        ),
    );
}

#[test]
fn a_length_at_u64_max_saturates_to_the_rest_of_the_object() {
    assert_verified(
        "offset 8, length u64::MAX",
        run_case(
            "tail_max_length",
            "ok-tail",
            &[
                ("S3DLIO_CHUNK_SIZE", "4"),
                ("S3DLIO_RANGE_CONCURRENCY", "2"),
            ],
        ),
    );
}
