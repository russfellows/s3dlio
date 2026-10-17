// tests/common/child_case.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>
//
// Shared child-process harness for tests of process-latched knobs (`OnceLock` settings read
// once per process): each case runs in its OWN child process (the test binary re-executed on
// one `--exact` test), under a hard wall-clock timeout and an address-space limit, with the
// environment scrubbed of anything that would change which server it talks to.  A binary
// includes this with `#[path = "common/child_case.rs"] mod child_case;` and describes itself
// with a [`ChildSpec`].  Linux only (`ulimit -v`).

use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// What distinguishes one test binary's children from another's.
pub struct ChildSpec {
    /// Environment variable that marks a re-executed child and selects what it does.
    pub case_var: &'static str,
    /// Environment variable that carries what the child must observe.
    pub expect_var: &'static str,
    /// Printed by the child when its assertions held.
    pub sentinel: &'static str,
    /// The `#[test]` function the child re-executes (`--exact`).
    pub child_test: &'static str,
    /// Hard wall-clock limit for one child.
    pub hard_timeout: Duration,
    /// Address-space limit (KiB) for one child.
    pub address_space_kib: u64,
    /// Added to a failure that is not a timeout (what a crash or a non-zero exit usually means).
    pub failure_hint: &'static str,
}

pub struct ChildOutcome {
    timed_out: bool,
    /// Set when the child was still alive after the whole SIGTERM ladder.
    survivor_pid: Option<u32>,
    status: Option<std::process::ExitStatus>,
    output: String,
}

/// The most output kept from one child stream; the rest is read and discarded.
const KEEP_OUTPUT_BYTES: usize = 1 << 20;
/// How long to wait for a reader thread to see end-of-file after the child exited.
const READER_GRACE: Duration = Duration::from_secs(5);

/// Drain `stream` on its own thread, so a chatty child can never fill a pipe and block.
/// The kept text (at most [`KEEP_OUTPUT_BYTES`]) is sent when the stream ends.
fn drain<R: Read + Send + 'static>(mut stream: R) -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut kept = Vec::new();
        let mut buf = [0u8; 8192];
        loop {
            match stream.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(n) => {
                    let room = KEEP_OUTPUT_BYTES.saturating_sub(kept.len());
                    kept.extend_from_slice(&buf[..n.min(room)]);
                }
            }
        }
        // The receiver may be gone (the parent gave up); nothing else to do then.
        tx.send(String::from_utf8_lossy(&kept).into_owned()).ok();
    });
    rx
}

/// A two-step SIGTERM ladder with the default disposition (the child installs no handler).
/// Returns whether the child exited.  There is no SIGKILL: a child that outlives the ladder
/// is reported to the caller, which must not wait on it.
fn terminate(child: &mut std::process::Child) -> bool {
    for _ in 0..2 {
        Command::new("kill")
            .args(["-TERM", &child.id().to_string()])
            .status()
            .ok();
        let until = Instant::now() + Duration::from_secs(3);
        while Instant::now() < until {
            if matches!(child.try_wait(), Ok(Some(_))) {
                return true;
            }
            std::thread::sleep(Duration::from_millis(50));
        }
    }
    false
}

/// Variables from a developer or CI shell that would change which server the child talks
/// to, which credentials or profile it resolves, or how it retries and times out.  Every
/// endpoint, profile and transport variable `s3_client.rs` and the AWS SDK honour is here.
const SCRUBBED_ENV: &[&str] = &[
    "S3DLIO_CHUNK_SIZE",
    "S3DLIO_RANGE_CONCURRENCY",
    "S3DLIO_RANGE_THRESHOLD_MB",
    "S3DLIO_SKIP_HEAD",
    "S3DLIO_FOLLOW_REDIRECTS",
    "S3DLIO_OPERATION_TIMEOUT_SECS",
    "S3DLIO_CONNECT_TIMEOUT_SECS",
    "S3DLIO_MAX_RETRY_ATTEMPTS",
    "S3DLIO_EXTRA_TRANSIENT_CODES",
    "S3DLIO_RT_THREADS",
    "S3DLIO_RT_THREADS_UNSAFE",
    "AWS_ENDPOINT_URL_S3",
    "AWS_ENDPOINT_URL",
    "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE",
    "AWS_CONFIG_FILE",
    "AWS_SHARED_CREDENTIALS_FILE",
    "AWS_SESSION_TOKEN",
    "AWS_CA_BUNDLE",
    "AWS_DEFAULT_REGION",
    "AWS_ROLE_ARN",
    "AWS_WEB_IDENTITY_TOKEN_FILE",
    "AWS_CONTAINER_CREDENTIALS_FULL_URI",
    "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
];

/// Run one child case.  `env` is applied last, so it overrides the defaults set here
/// (`S3DLIO_ENABLE_RANGE_OPTIMIZATION=1`, no `.env` file, no instance-metadata lookups).
pub fn run_case(spec: &ChildSpec, case: &str, expect: &str, env: &[(&str, &str)]) -> ChildOutcome {
    let exe = std::env::current_exe().expect("test executable path");
    let address_space_kib = spec.address_space_kib;
    let mut cmd = Command::new("sh");
    cmd.arg("-c")
        .arg(format!(
            "ulimit -v {address_space_kib} && exec \"$0\" \"$@\""
        ))
        .arg(exe)
        .args([
            "--exact",
            spec.child_test,
            "--nocapture",
            "--test-threads=1",
        ])
        .env(spec.case_var, case)
        .env(spec.expect_var, expect)
        // Fewer malloc arenas keep the address-space limit about real memory.
        .env("MALLOC_ARENA_MAX", "2")
        .env("S3DLIO_ENABLE_RANGE_OPTIMIZATION", "1")
        .env("S3DLIO_LOAD_DOTENV", "0")
        // No instance-metadata lookups if credentials were somehow missing.
        .env("AWS_EC2_METADATA_DISABLED", "true")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    // Values inherited from a developer shell must not change the outcome.
    for name in SCRUBBED_ENV {
        cmd.env_remove(name);
    }
    for (k, v) in env {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn().expect("spawn the child test process");
    let stdout = child.stdout.take().map(drain);
    let stderr = child.stderr.take().map(drain);

    let deadline = Instant::now() + spec.hard_timeout;
    let mut timed_out = false;
    let mut survivor_pid = None;
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break Some(s),
            Ok(None) if Instant::now() >= deadline => {
                timed_out = true;
                if !terminate(&mut child) {
                    survivor_pid = Some(child.id());
                }
                break child.try_wait().ok().flatten();
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(50)),
            Err(_) => break None,
        }
    };

    // A surviving child still holds its pipes open: waiting for end-of-file would block.
    let collect = |rx: Option<mpsc::Receiver<String>>| match (rx, survivor_pid) {
        (Some(rx), None) => rx
            .recv_timeout(READER_GRACE)
            .unwrap_or_else(|_| "(output incomplete)".to_string()),
        (Some(_), Some(_)) => "(child still running; output not collected)".to_string(),
        (None, _) => "(no output stream)".to_string(),
    };
    let output = format!("{}\n{}", collect(stdout), collect(stderr));
    ChildOutcome {
        timed_out,
        survivor_pid,
        status,
        output,
    }
}

/// Panic unless the child ran to its sentinel (exit 0, not hung, not killed).
pub fn assert_verified(spec: &ChildSpec, what: &str, o: ChildOutcome) {
    if let Some(pid) = o.survivor_pid {
        panic!(
            "{what}: the child (pid {pid}) is STILL RUNNING after the SIGTERM ladder; it was not \
             killed and not waited for, and needs to be stopped by hand"
        );
    }
    let hard_timeout = spec.hard_timeout;
    assert!(
        !o.timed_out,
        "{what}: HUNG past {hard_timeout:?}\n{}",
        o.output
    );
    assert!(
        o.status.is_some_and(|s| s.success()) && o.output.contains(spec.sentinel),
        "{what}: the child did not verify the contract (exit {:?}; {})\n{}",
        o.status,
        spec.failure_hint,
        o.output
    );
}
