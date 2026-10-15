// tests/dotenv_switch_243.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Regression tests for s3dlio issue #243: implicit `.env` loading.
//
// Before the fix every S3 client build loaded `.env` from the process's working
// directory into the environment of the whole host process. Now ONE switch,
// `S3DLIO_LOAD_DOTENV`, decides:
//   * library use (Rust, Python): OFF unless asked (`1`/`true` = the working
//     directory's `.env`, a path = exactly that file);
//   * the `s3-cli` binary: ON unless told otherwise (`0`/`false`, or a path).
// A variable already set in the real environment is never overridden.
//
// Library behaviour is checked in a fresh child process (this very test binary
// re-run on `probe_child`) because the switch is read once per process; the CLI
// behaviour goes through the real `s3-cli` binary against the shared mock S3.

#[allow(dead_code)]
mod common;

use common::mock_s3_server::ensure_mock_s3_server;
use std::path::{Path, PathBuf};
use std::process::Command;

fn scratch(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("dotenv243-{name}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

// ---------------------------------------------------------------------------
// Library mode: the child process
// ---------------------------------------------------------------------------

/// Runs only as the child of `probe` (marked by `S3DLIO_243_PROBE`); otherwise a no-op.
#[test]
fn probe_child() {
    if std::env::var_os("S3DLIO_243_PROBE").is_none() {
        return;
    }
    if std::env::var_os("S3DLIO_243_LOG").is_some() {
        // Show INFO logs on stdout so the parent can see the "not loaded" hint.
        let _ = tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .with_ansi(false)
            .with_writer(std::io::stdout)
            .try_init();
    }
    // Which library entry point runs first (every one of them must apply the policy
    // before it reads any setting). The default is the S3 client build.
    let entry = std::env::var("S3DLIO_243_ENTRY").unwrap_or_else(|_| "client".to_string());
    let client = match entry.as_str() {
        "client" => {
            let _ = s3dlio::s3_client::aws_s3_client();
            match s3dlio::s3_client::aws_s3_client() {
                Ok(_) => "ok".to_string(),
                Err(e) => format!("err:{e}"),
            }
        }
        "runtime" => {
            let _ = s3dlio::s3_client::run_on_global_rt(async { Ok::<(), anyhow::Error>(()) });
            "n/a".to_string()
        }
        "store_file" => {
            let _ = s3dlio::object_store::store_for_uri("file:///");
            "n/a".to_string()
        }
        #[cfg(feature = "backend-azure")]
        "azure" => {
            let _ = s3dlio::azure_client::AzureBlob::with_default_credential("acct", "cont");
            "n/a".to_string()
        }
        #[cfg(feature = "backend-gcs")]
        "gcs" => {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let _ = rt.block_on(s3dlio::google_gcs_client::GcsClient::new());
            "n/a".to_string()
        }
        other => panic!("unknown S3DLIO_243_ENTRY {other}"),
    };
    let var = |n: &str| std::env::var(n).unwrap_or_else(|_| "<unset>".to_string());
    println!(
        "PROBE marker={} preset={} key={} client={client}",
        var("S3DLIO_243_MARKER"),
        var("S3DLIO_243_PRESET"),
        var("AWS_ACCESS_KEY_ID"),
    );
}

#[derive(Debug)]
struct Probe {
    marker: String,
    preset: String,
    key: String,
    client: String,
}

/// Re-run this test binary on `probe_child`, in `cwd`, with a cleaned environment plus
/// `extra`, and parse what it reports.
fn probe(cwd: &Path, extra: &[(&str, &str)]) -> Probe {
    parse_probe(&probe_stdout(cwd, extra))
}

fn probe_stdout(cwd: &Path, extra: &[(&str, &str)]) -> String {
    let mut cmd = Command::new(std::env::current_exe().unwrap());
    cmd.args(["--exact", "probe_child", "--nocapture", "--test-threads=1"])
        .current_dir(cwd)
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("S3DLIO_243_PROBE", "1")
        .env("S3DLIO_243_PRESET", "real-env")
        .env("AWS_EC2_METADATA_DISABLED", "true");
    for (k, v) in extra {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("run probe child");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn parse_probe(stdout: &str) -> Probe {
    // libtest prints `test probe_child ... ` before the child's own output.
    let line = stdout
        .lines()
        .find_map(|l| l.find("PROBE ").map(|i| &l[i + "PROBE ".len()..]))
        .unwrap_or_else(|| panic!("no PROBE line\nstdout={stdout}"));
    let field = |name: &str| {
        line.split(' ')
            .find_map(|f| f.strip_prefix(&format!("{name}=")))
            .unwrap_or_else(|| panic!("no {name} in {line}"))
            .to_string()
    };
    // `client=` is last and may contain spaces.
    let client = line.split("client=").nth(1).unwrap().to_string();
    Probe {
        marker: field("marker"),
        preset: field("preset"),
        key: field("key"),
        client,
    }
}

const DOTENV: &str = "AWS_ACCESS_KEY_ID=dotenv-key\n\
                      AWS_SECRET_ACCESS_KEY=dotenv-secret\n\
                      AWS_REGION=us-east-1\n\
                      S3DLIO_243_MARKER=from-dotenv\n\
                      S3DLIO_243_PRESET=from-dotenv\n";

fn dir_with_dotenv(name: &str) -> PathBuf {
    let dir = scratch(name);
    std::fs::write(dir.join(".env"), DOTENV).unwrap();
    dir
}

fn assert_not_loaded(p: &Probe, what: &str) {
    assert_eq!(
        p.marker, "<unset>",
        "{what}: .env must not be loaded: {p:?}"
    );
    assert_eq!(p.key, "<unset>", "{what}: {p:?}");
    assert!(
        p.client.starts_with("err:") && p.client.contains("Missing AWS_ACCESS_KEY_ID"),
        "{what}: without credentials the client build must fail: {p:?}"
    );
    assert_eq!(p.preset, "real-env", "{what}: {p:?}");
}

fn assert_loaded(p: &Probe, what: &str) {
    assert_eq!(
        p.marker, "from-dotenv",
        "{what}: .env must be loaded: {p:?}"
    );
    assert_eq!(p.key, "dotenv-key", "{what}: {p:?}");
    assert_eq!(p.client, "ok", "{what}: {p:?}");
    // Never overrides a variable that is already set.
    assert_eq!(p.preset, "real-env", "{what}: real environment wins: {p:?}");
}

#[test]
fn library_does_not_load_dotenv_by_default() {
    let dir = dir_with_dotenv("lib-default");
    assert_not_loaded(&probe(&dir, &[]), "switch unset");
    assert_not_loaded(&probe(&dir, &[("S3DLIO_LOAD_DOTENV", "")]), "switch empty");
}

#[test]
fn library_switch_off_never_loads() {
    let dir = dir_with_dotenv("lib-off");
    for v in ["0", "false", "FALSE", "no", "Off"] {
        assert_not_loaded(&probe(&dir, &[("S3DLIO_LOAD_DOTENV", v)]), v);
    }
}

#[test]
fn library_switch_on_loads_the_working_directory_dotenv() {
    let dir = dir_with_dotenv("lib-on");
    for v in ["1", "true", "True", "yes", "ON"] {
        assert_loaded(&probe(&dir, &[("S3DLIO_LOAD_DOTENV", v)]), v);
    }
}

#[test]
fn library_switch_path_loads_exactly_that_file() {
    let dir = dir_with_dotenv("lib-path");
    let other = scratch("lib-path-other").join("named.env");
    std::fs::write(
        &other,
        "AWS_ACCESS_KEY_ID=named-key\nAWS_SECRET_ACCESS_KEY=named-secret\n\
         AWS_REGION=us-east-1\nS3DLIO_243_MARKER=from-named\n",
    )
    .unwrap();
    let p = probe(&dir, &[("S3DLIO_LOAD_DOTENV", other.to_str().unwrap())]);
    // The named file, and not the working directory's `.env`.
    assert_eq!(p.marker, "from-named", "{p:?}");
    assert_eq!(p.key, "named-key", "{p:?}");
    assert_eq!(p.preset, "real-env", "{p:?}");
    assert_eq!(p.client, "ok", "{p:?}");
}

#[test]
fn library_switch_naming_a_missing_file_loads_nothing_and_does_not_crash() {
    let dir = dir_with_dotenv("lib-missing");
    let missing = dir.join("no-such.env");
    assert_not_loaded(
        &probe(&dir, &[("S3DLIO_LOAD_DOTENV", missing.to_str().unwrap())]),
        "missing named file (the working directory's .env is NOT a fallback)",
    );
}

const HINT: &str = "S3DLIO_LOAD_DOTENV=1";

#[test]
fn missing_credentials_error_names_the_ignored_dotenv() {
    // A Python user otherwise sees only "Missing ..." and never learns that the .env in
    // the working directory is being ignored.
    let dir = dir_with_dotenv("lib-err-hint");
    let p = probe(&dir, &[]);
    assert!(
        p.client.contains("Missing AWS_ACCESS_KEY_ID") && p.client.contains(HINT),
        "the error must say how to load the .env: {p:?}"
    );
    // An explicit decision to load nothing gets the plain message.
    let p = probe(&dir, &[("S3DLIO_LOAD_DOTENV", "0")]);
    assert!(
        p.client.contains("Missing AWS_ACCESS_KEY_ID") && !p.client.contains(HINT),
        "no hint after an explicit 0: {p:?}"
    );
}

/// Every library entry point applies the policy before it reads any setting, so a
/// program whose FIRST call is that entry still gets the file's values.
fn assert_entry_loads(entry: &str, extra: &[(&str, &str)]) {
    let dir = dir_with_dotenv(&format!("entry-{entry}"));
    let mut env = vec![("S3DLIO_LOAD_DOTENV", "1"), ("S3DLIO_243_ENTRY", entry)];
    env.extend_from_slice(extra);
    let p = probe(&dir, &env);
    assert_eq!(p.marker, "from-dotenv", "entry {entry}: {p:?}");
    assert_eq!(p.preset, "real-env", "entry {entry}: {p:?}");
}

#[test]
fn global_runtime_entry_loads_the_file() {
    assert_entry_loads("runtime", &[]);
}

#[test]
fn store_factory_entry_loads_the_file() {
    assert_entry_loads("store_file", &[]);
}

#[cfg(feature = "backend-azure")]
#[test]
fn azure_client_entry_loads_the_file() {
    assert_entry_loads("azure", &[]);
}

#[cfg(feature = "backend-gcs")]
#[test]
fn gcs_client_entry_loads_the_file() {
    assert_entry_loads("gcs", &[]);
}

#[test]
fn library_logs_a_one_time_hint_when_a_dotenv_exists_but_is_not_loaded() {
    let dir = dir_with_dotenv("lib-hint");
    let out = probe_stdout(&dir, &[("S3DLIO_243_LOG", "1")]);
    // (The failed client build's own error text also carries the hint, on the PROBE
    // line; the log line is the INFO one.)
    let info_hints = out
        .lines()
        .filter(|l| l.contains("INFO") && l.contains(HINT))
        .count();
    assert_eq!(
        info_hints, 1,
        "exactly one hint, although the client was built twice\n{out}"
    );
}

#[test]
fn library_logs_no_hint_without_a_dotenv_or_with_an_explicit_off() {
    let empty = scratch("lib-hint-none");
    // `.env` discovery also looks in parent directories; a developer's own `.env` above
    // the repository would make "no .env" untrue here, so that half is skipped then.
    if let Some(parent_env) = empty.ancestors().find(|d| d.join(".env").is_file()) {
        eprintln!(
            "SKIPPED the no-.env half: {} has a .env, which discovery finds from {}",
            parent_env.display(),
            empty.display()
        );
    } else {
        let out = probe_stdout(&empty, &[("S3DLIO_243_LOG", "1")]);
        assert!(!out.contains(HINT), "no .env, no hint\n{out}");
    }

    let dir = dir_with_dotenv("lib-hint-off");
    let out = probe_stdout(
        &dir,
        &[("S3DLIO_243_LOG", "1"), ("S3DLIO_LOAD_DOTENV", "0")],
    );
    assert!(!out.contains(HINT), "an explicit 0 is a decision\n{out}");
}

// ---------------------------------------------------------------------------
// The s3-cli binary: on by default
// ---------------------------------------------------------------------------

fn s3_cli_ls(cwd: &Path, bucket: &str, extra: &[(&str, &str)]) -> (bool, String, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_s3-cli"));
    cmd.args(["ls", &format!("s3://{bucket}/")])
        .current_dir(cwd)
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("AWS_EC2_METADATA_DISABLED", "true");
    for (k, v) in extra {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("run s3-cli");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// A `.env` that points s3-cli at the mock with credentials; nothing else does.
fn cli_dir(name: &str, port: u16) -> PathBuf {
    let dir = scratch(name);
    std::fs::write(
        dir.join(".env"),
        format!(
            "AWS_ENDPOINT_URL=http://127.0.0.1:{port}\nAWS_ACCESS_KEY_ID=test-access-key\n\
             AWS_SECRET_ACCESS_KEY=test-secret-key\nAWS_REGION=us-east-1\n"
        ),
    )
    .unwrap();
    dir
}

#[test]
fn s3_cli_loads_dotenv_by_default() {
    let (port, state) = ensure_mock_s3_server();
    state.seed_bucket("cli243-default", &["obj-a"]);
    let dir = cli_dir("cli-default", port);
    let (ok, stdout, stderr) = s3_cli_ls(&dir, "cli243-default", &[]);
    assert!(
        ok,
        "s3-cli must keep reading .env\nstdout={stdout}\nstderr={stderr}"
    );
    assert!(stdout.contains("obj-a"), "stdout={stdout}");
}

#[test]
fn s3_cli_switch_off_does_not_load_dotenv() {
    let (port, state) = ensure_mock_s3_server();
    state.seed_bucket("cli243-off", &["obj-a"]);
    let dir = cli_dir("cli-off", port);
    for v in ["0", "false"] {
        let (ok, stdout, stderr) = s3_cli_ls(&dir, "cli243-off", &[("S3DLIO_LOAD_DOTENV", v)]);
        assert!(
            !ok,
            "{v}: .env must not be read\nstdout={stdout}\nstderr={stderr}"
        );
        assert!(!stdout.contains("obj-a"), "{v}: stdout={stdout}");
        assert!(stderr.contains("Missing"), "{v}: stderr={stderr}");
    }
}

#[test]
fn s3_cli_switch_path_loads_that_file_instead_of_dotenv() {
    let (port, state) = ensure_mock_s3_server();
    state.seed_bucket("cli243-path", &["obj-a"]);
    // The working directory's `.env` is useless (no credentials); the named file works.
    let dir = scratch("cli-path");
    std::fs::write(dir.join(".env"), "AWS_REGION=us-east-1\n").unwrap();
    let named = scratch("cli-path-other").join("good.env");
    std::fs::write(
        &named,
        format!(
            "AWS_ENDPOINT_URL=http://127.0.0.1:{port}\nAWS_ACCESS_KEY_ID=test-access-key\n\
             AWS_SECRET_ACCESS_KEY=test-secret-key\nAWS_REGION=us-east-1\n"
        ),
    )
    .unwrap();
    let (ok, stdout, stderr) = s3_cli_ls(
        &dir,
        "cli243-path",
        &[("S3DLIO_LOAD_DOTENV", named.to_str().unwrap())],
    );
    assert!(ok, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("obj-a"), "stdout={stdout}");
}

#[test]
fn s3_cli_reports_a_missing_named_file_on_stderr() {
    let (port, state) = ensure_mock_s3_server();
    state.seed_bucket("cli243-missing", &["obj-a"]);
    let dir = cli_dir("cli-missing", port);
    let missing = dir.join("no-such.env");
    let missing = missing.to_str().unwrap();
    let (_ok, _stdout, stderr) =
        s3_cli_ls(&dir, "cli243-missing", &[("S3DLIO_LOAD_DOTENV", missing)]);
    // No log subscriber exists when s3-cli loads the file, so main prints the problem.
    assert!(stderr.contains("S3DLIO_LOAD_DOTENV"), "stderr={stderr}");
    assert!(stderr.contains(missing), "stderr={stderr}");
}

#[test]
fn s3_cli_never_overrides_the_real_environment() {
    let (port, state) = ensure_mock_s3_server();
    state.seed_bucket("cli243-real", &["obj-a"]);
    // `.env` names a dead endpoint; the real environment names the mock and wins.
    let dir = scratch("cli-real");
    std::fs::write(
        dir.join(".env"),
        "AWS_ENDPOINT_URL=http://127.0.0.1:1\nAWS_ACCESS_KEY_ID=test-access-key\n\
         AWS_SECRET_ACCESS_KEY=test-secret-key\nAWS_REGION=us-east-1\n",
    )
    .unwrap();
    let url = format!("http://127.0.0.1:{port}");
    let (ok, stdout, stderr) =
        s3_cli_ls(&dir, "cli243-real", &[("AWS_ENDPOINT_URL", url.as_str())]);
    assert!(ok, "stdout={stdout}\nstderr={stderr}");
    assert!(stdout.contains("obj-a"), "stdout={stdout}");
}
