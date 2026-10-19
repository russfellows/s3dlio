// tests/s3_get_into_routes.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Which ROUTE an S3 caller-buffer GET takes (plan §4.3 "S3", §9): one unranged GET with no HEAD,
// or the ranged fan-out (a fresh HEAD, then pinned ranges).  The choice is made from the
// destination length and three process-wide settings:
//
//   * `S3DLIO_SKIP_HEAD` (and the `set_skip_head` latch): no HEAD, so no fan-out;
//   * `S3DLIO_ENABLE_RANGE_OPTIMIZATION=0`: no range splitting;
//   * `S3DLIO_RANGE_THRESHOLD_MB`: fan out when the length is AT OR ABOVE the threshold.
//
// Each of these is latched once per process (`OnceLock`), so each case runs in its own child
// process (the harness in `tests/common/child_case.rs`) against the in-process mock server, and
// the child counts what the server saw: a HEAD and ranged GETs (fan-out), or one unranged GET.
// A control case per switch shows the fan-out IS taken when the switch is at its default, so
// "single" cannot be an accident of the setup.

// The address-space limit (`ulimit -v`) is a Linux mechanism, so the file is Linux-only.
#![cfg(target_os = "linux")]

// Only part of the mock server API is used here.
#[path = "common/mock_s3_server.rs"]
#[expect(
    dead_code,
    reason = "this binary uses only part of the shared mock server API"
)]
mod mock_s3_server;

#[path = "common/child_case.rs"]
mod child_case;

use mock_s3_server::{ensure_mock_s3_server, unique_test_key, MockObject};
use std::time::Duration;

const BUCKET: &str = "route-bucket";
const MIB: usize = 1024 * 1024;

const CHILD_CASE: &str = "S3DLIO_TEST_ROUTE_CASE";
const CHILD_EXPECT: &str = "S3DLIO_TEST_ROUTE_EXPECT";
const SENTINEL: &str = "S3DLIO_ROUTE_CHILD_VERIFIED";

const SPEC: child_case::ChildSpec = child_case::ChildSpec {
    case_var: CHILD_CASE,
    expect_var: CHILD_EXPECT,
    sentinel: SENTINEL,
    child_test: "child_run_case",
    hard_timeout: Duration::from_secs(45),
    address_space_kib: 3 * 1024 * 1024,
    failure_hint: "the route taken is not the expected one, or the read failed",
};

// ─────────────────────────────── child side ───────────────────────────────

/// Runs only when re-executed by [`run_case`]; a plain `cargo test` finds no marker and returns.
///
/// The case is `<length>` or `<length>:latch` (call `set_skip_head(true)` before the read); the
/// expectation is `single` or `fanout`.
#[test]
fn child_run_case() {
    let Ok(case) = std::env::var(CHILD_CASE) else {
        return;
    };
    let expect = std::env::var(CHILD_EXPECT).expect("the parent sets the expectation");
    let (length, latch) = match case.split_once(':') {
        Some((length, "latch")) => (length, true),
        Some((_, other)) => panic!("unknown case flag {other}"),
        None => (case.as_str(), false),
    };
    let length: usize = length.parse().expect("a numeric length");

    let state = ensure_mock_s3_server().1;
    let key = unique_test_key("route");
    let data: Vec<u8> = (0..length)
        .map(|i| u8::try_from(i % 251).unwrap_or(0))
        .collect();
    state.put_object(BUCKET, &key, MockObject::new(data.clone()));

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("runtime");
    let dst = rt.block_on(async {
        if latch {
            s3dlio::s3_utils::set_skip_head(true);
        }
        let client = s3dlio::s3_client::aws_s3_client_async()
            .await
            .expect("the process-wide client");
        let ops = s3dlio::s3_ops::S3Ops::new(client, None, "", "");
        let mut dst = vec![0u8; length];
        let progress = s3dlio::receive_progress::ReceiveProgress::new();
        ops.get_into(BUCKET, &key, &mut dst, &progress)
            .await
            .expect("the read");
        dst
    });
    assert!(dst == data, "the bytes must be the object's");

    let requests = state.requests_for(BUCKET, &key);
    let heads = requests.iter().filter(|r| r.method == "HEAD").count();
    let gets: Vec<_> = requests.iter().filter(|r| r.method == "GET").collect();
    match expect.as_str() {
        "single" => {
            assert_eq!(heads, 0, "the single route sends no HEAD: {requests:?}");
            assert_eq!(gets.len(), 1, "one GET: {requests:?}");
            assert_eq!(gets[0].range, None, "unranged: {requests:?}");
        }
        "fanout" => {
            assert_eq!(heads, 1, "the fan-out takes one fresh HEAD: {requests:?}");
            assert!(gets.len() >= 2, "several ranges: {requests:?}");
            assert!(
                gets.iter()
                    .all(|r| r.range.is_some() && r.if_match.is_some()),
                "every GET is a pinned range: {requests:?}"
            );
        }
        other => panic!("unknown expectation {other}"),
    }
    println!("{SENTINEL}");
}

// ─────────────────────────────── parent side ───────────────────────────────

fn run_case(case: &str, expect: &str, env: &[(&str, &str)]) -> child_case::ChildOutcome {
    child_case::run_case(&SPEC, case, expect, env)
}

fn assert_verified(what: &str, outcome: child_case::ChildOutcome) {
    child_case::assert_verified(&SPEC, what, outcome)
}

/// Settings that make a 4 KiB read a four-range fan-out when nothing else forbids it.
const SMALL_FANOUT: &[(&str, &str)] = &[
    ("S3DLIO_RANGE_THRESHOLD_MB", "0"),
    ("S3DLIO_CHUNK_SIZE", "1024"),
    ("S3DLIO_RANGE_CONCURRENCY", "2"),
];

/// Those settings plus one more.
fn small_fanout_with<'a>(extra: (&'a str, &'a str)) -> Vec<(&'a str, &'a str)> {
    let mut env = SMALL_FANOUT.to_vec();
    env.push(extra);
    env
}

/// The control for every case below: nothing forbids the fan-out and the length is at the
/// threshold, so the fan-out is taken.
#[test]
fn with_every_switch_at_its_default_a_read_at_the_threshold_fans_out() {
    assert_verified(
        "defaults, threshold 0",
        run_case("4096", "fanout", SMALL_FANOUT),
    );
}

#[test]
fn skip_head_in_the_environment_forces_the_single_get() {
    for value in ["1", "true"] {
        assert_verified(
            &format!("S3DLIO_SKIP_HEAD={value}"),
            run_case(
                "4096",
                "single",
                &small_fanout_with(("S3DLIO_SKIP_HEAD", value)),
            ),
        );
    }
}

/// The runtime latch (`set_skip_head`, used by the loader option `skip_head`) does the same.
#[test]
fn the_skip_head_latch_forces_the_single_get() {
    assert_verified(
        "set_skip_head(true)",
        run_case("4096:latch", "single", SMALL_FANOUT),
    );
}

#[test]
fn disabling_range_optimisation_forces_the_single_get() {
    for value in ["0", "false"] {
        assert_verified(
            &format!("S3DLIO_ENABLE_RANGE_OPTIMIZATION={value}"),
            run_case(
                "4096",
                "single",
                &small_fanout_with(("S3DLIO_ENABLE_RANGE_OPTIMIZATION", value)),
            ),
        );
    }
}

/// `expected >= threshold` fans out: one byte below the threshold is the single GET, exactly at
/// it and one above are the fan-out.
#[test]
fn the_threshold_boundary_is_at_or_above() {
    let env = [
        ("S3DLIO_RANGE_THRESHOLD_MB", "1"),
        ("S3DLIO_CHUNK_SIZE", "262144"),
        ("S3DLIO_RANGE_CONCURRENCY", "2"),
    ];
    for (length, expect) in [(MIB - 1, "single"), (MIB, "fanout"), (MIB + 1, "fanout")] {
        assert_verified(
            &format!("a {length}-byte read against a 1 MiB threshold"),
            run_case(&length.to_string(), expect, &env),
        );
    }
}

/// An unparseable threshold is the 32 MiB default, so a small read is the single GET.
#[test]
fn an_unparseable_threshold_is_the_default_and_a_small_read_is_single() {
    assert_verified(
        "S3DLIO_RANGE_THRESHOLD_MB=not-a-number",
        run_case(
            "4096",
            "single",
            &[("S3DLIO_RANGE_THRESHOLD_MB", "not-a-number")],
        ),
    );
}
