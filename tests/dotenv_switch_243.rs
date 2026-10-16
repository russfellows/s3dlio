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
        // The first-use reads Blue found (#243 review): each is made the FIRST s3dlio
        // call of the process, with the setting only in the `.env`.
        "s3_dataset" => {
            // A host-form URI is parsed against AWS_ENDPOINT_URL before anything lists.
            match s3dlio::data_loader::S3BytesDataset::from_prefix("s3://127.0.0.1:1/bkt/p") {
                Ok(_) => "ok".to_string(),
                Err(e) => format!("err:{e}"),
            }
        }
        "pool" => {
            s3dlio::s3_client::configure_thread_pools(0);
            format!("rayon={}", rayon::current_num_threads())
        }
        "unsigned" => format!("unsigned={}", s3dlio::s3_client::unsigned_payload_enabled()),
        "logger" => {
            let path = std::env::current_dir().unwrap().join("oplog.tsv.zst");
            let logger = s3dlio::s3_logger::Logger::new(path.to_str().unwrap()).unwrap();
            format!("lossless={}", logger.stats().lossless)
        }
        "skip_head" => {
            // Own runtime: the global one would apply the policy by itself.
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let uri = std::env::var("S3DLIO_243_URI").unwrap();
            let r = rt.block_on(s3dlio::s3_utils::get_object_uri_optimized_async(&uri));
            format!("get={}", r.is_ok())
        }
        "store_file" => {
            let _ = s3dlio::object_store::store_for_uri("file:///");
            "n/a".to_string()
        }
        // Round 7 (Blue R8/R9): public setters and setting-reading helpers/builders that are
        // independently public, each made the FIRST s3dlio call with the value only in the file.
        "endpoint_override" => {
            s3dlio::s3_client::set_s3_endpoint_override("http://new:9000");
            format!(
                "endpoint={}",
                s3dlio::s3_client::s3_endpoint_url_from_env().unwrap_or_else(|| "<none>".into())
            )
        }
        "h2_window" => {
            let c = s3dlio::constants::H2WindowConfig::from_env();
            format!(
                "adaptive={} stream_mb={} conn_mb={}",
                c.adaptive, c.stream_window_mb, c.conn_window_mb
            )
        }
        "connect_timeout" => format!(
            "connect_timeout={}",
            s3dlio::constants::connect_timeout_secs()
        ),
        "max_retry" => format!("max_retry={}", s3dlio::constants::max_retry_attempts()),
        "size_cache" => {
            // The public constructor is the FIRST s3dlio call; the entry limit it latches
            // (OnceLock) must come from the file. Five puts show the limit in the entry count.
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            rt.block_on(async {
                let cache = s3dlio::object_size_cache::ObjectSizeCache::new(
                    std::time::Duration::from_secs(60),
                );
                for i in 0..5 {
                    cache.put(format!("s3://bkt/k{i}"), 1).await;
                }
                format!("entries={}", cache.stats().await.total_entries)
            })
        }
        "export_quiesce" => format!("quiesce={:?}", s3dlio::constants::export_quiesce_timeout()),
        "thread_budget" => format!(
            "world={} budget={}",
            s3dlio::constants::mpi_world_size(),
            s3dlio::constants::effective_thread_budget()
        ),
        "budget_first" => format!("budget={}", s3dlio::constants::effective_thread_budget()),
        "bounce_config" => {
            let c = s3dlio::direct_bounce::config();
            format!(
                "size={} pairs={} busy={:?} get={:?}",
                c.buffer_size, c.pairs_per_node, c.busy, c.get
            )
        }
        "aware_budget" => format!("aware={}", s3dlio::constants::mpi_aware_thread_budget()),
        "transport" => transport_wire_protocol(),
        "multi_endpoint" => {
            // The scheme of a server-named endpoint depends on S3DLIO_S3_ENDPOINT_TLS_PORTS.
            let store = s3dlio::multi_endpoint::MultiEndpointStore::new(
                vec!["s3://127.0.0.1:9443/bkt/".to_string()],
                s3dlio::multi_endpoint::LoadBalanceStrategy::RoundRobin,
                None,
            );
            match store {
                Ok(s) => format!("urls={:?}", s.get_store_endpoint_urls()),
                Err(e) => format!("err:{e}"),
            }
        }
        "multi_endpoint_config" => {
            use s3dlio::multi_endpoint::{
                EndpointConfig, LoadBalanceStrategy, MultiEndpointStore, MultiEndpointStoreConfig,
            };
            let config = MultiEndpointStoreConfig {
                endpoints: vec![EndpointConfig::new("s3://127.0.0.1:9443/bkt/")],
                strategy: LoadBalanceStrategy::RoundRobin,
                default_thread_count: None,
            };
            match MultiEndpointStore::from_config(config) {
                Ok(s) => format!("urls={:?}", s.get_store_endpoint_urls()),
                Err(e) => format!("err:{e}"),
            }
        }
        "concurrent_threshold" => {
            // First call is a ranged read with a known length: the threshold decides between
            // one plain ranged GET and the concurrent path (which sizes the object with a HEAD).
            use s3dlio::object_store::{ObjectStore, S3ObjectStore};
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let uri = std::env::var("S3DLIO_243_URI").unwrap();
            let r = rt.block_on(S3ObjectStore::new().get_range_optimized(
                &uri,
                0,
                Some(4 * 1024 * 1024),
                None,
                None,
            ));
            format!("range={}", r.is_ok())
        }
        #[cfg(feature = "backend-azure")]
        "azure_account_url" => s3dlio::azure_client::resolve_azure_account_url("acct")
            .unwrap_or_else(|e| format!("error: {e}")),
        #[cfg(feature = "arrow-backend")]
        "arrow_get" => {
            use s3dlio::object_store::ObjectStore;
            use s3dlio::object_store_arrow::ArrowObjectStore;
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            match rt.block_on(ArrowObjectStore::new().get("s3://dotenv243-bkt/arrow-key")) {
                Ok(_) => "arrow=ok".to_string(),
                Err(e) if e.to_string().contains("credentials not found") => {
                    "arrow=no-credentials".to_string()
                }
                // Anything else happened after the credential check.
                Err(_) => "arrow=credentials-seen".to_string(),
            }
        }
        #[cfg(feature = "gcs-community")]
        "gcs_endpoint" => format!(
            "gcs_endpoint={:?}",
            s3dlio::gcs_client::resolve_gcs_endpoint()
        ),
        #[cfg(feature = "backend-azure")]
        "azure_config" => format!(
            "download_concurrency={}",
            s3dlio::object_store::AzureConfig::default().sdk_download_concurrency
        ),
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

/// First bytes a client built by `build_smithy_http_client` puts on the wire towards a
/// plain-HTTP endpoint: the HTTP/2 prior-knowledge preface (`PRI * HTTP/2.0`) when H2C is
/// on, an HTTP/1.1 request line otherwise. The client is opaque, so the wire is the proof.
fn transport_wire_protocol() -> String {
    use aws_sdk_s3::config::{BehaviorVersion, Credentials, Region};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let reader = std::thread::spawn(move || {
        use std::io::Read;
        let (mut sock, _) = listener.accept().unwrap();
        sock.set_read_timeout(Some(std::time::Duration::from_secs(10)))
            .unwrap();
        let mut buf = [0u8; 24];
        let mut got = 0;
        while got < 4 {
            match sock.read(&mut buf[got..]) {
                Ok(0) | Err(_) => break,
                Ok(n) => got += n,
            }
        }
        String::from_utf8_lossy(&buf[..got]).into_owned()
    });
    // reqwest wants a Tokio context to be built in; this is the test's own runtime, not
    // s3dlio's (whose creation would apply the policy by itself).
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let _guard = rt.enter();
    let http = s3dlio::reqwest_client::build_smithy_http_client(None).unwrap();
    let conf = aws_sdk_s3::Config::builder()
        .behavior_version(BehaviorVersion::latest())
        .region(Region::new("us-east-1"))
        .credentials_provider(Credentials::new("k", "s", None, None, "test"))
        .endpoint_url(format!("http://127.0.0.1:{port}"))
        .force_path_style(true)
        .retry_config(aws_sdk_s3::config::retry::RetryConfig::disabled())
        .http_client(http)
        .build();
    let client = aws_sdk_s3::Client::from_conf(conf);
    // The server closes after reading the first bytes, so the call fails; only the wire matters.
    let _ = rt.block_on(tokio::time::timeout(
        std::time::Duration::from_secs(15),
        client.list_buckets().send(),
    ));
    let first = reader.join().unwrap();
    format!(
        "wire={}",
        if first.starts_with("PRI ") {
            "h2c"
        } else {
            "http1"
        }
    )
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
    dir_with_dotenv_plus(name, "")
}

/// A scratch directory whose `.env` is the standard one plus `extra` lines.
fn dir_with_dotenv_plus(name: &str, extra: &str) -> PathBuf {
    let dir = scratch(name);
    std::fs::write(dir.join(".env"), format!("{DOTENV}{extra}")).unwrap();
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

// -- first-use reads: the setting exists ONLY in the .env, the API is the first call --

#[test]
fn s3_dataset_host_form_uri_sees_the_endpoint_from_the_file() {
    let dir = dir_with_dotenv_plus("first-dataset", "AWS_ENDPOINT_URL=http://127.0.0.1:1\n");
    let p = probe(
        &dir,
        &[
            ("S3DLIO_LOAD_DOTENV", "1"),
            ("S3DLIO_243_ENTRY", "s3_dataset"),
        ],
    );
    // Parsing must not fail for want of the endpoint the file supplies; what is left is
    // the connection to the (closed) port.
    assert!(
        !p.client
            .contains("cannot be used with the process-wide S3 client"),
        "the host-form URI was parsed before the file was loaded: {p:?}"
    );
}

#[test]
fn thread_pool_sizing_sees_the_world_size_from_the_file() {
    if std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        < 2
    {
        eprintln!("SKIPPED: one CPU cannot show a smaller pool");
        return;
    }
    // A world size above any CPU count makes the per-process budget 1.
    let dir = dir_with_dotenv_plus("first-pool", "WORLD_SIZE=100000\n");
    let p = probe(
        &dir,
        &[("S3DLIO_LOAD_DOTENV", "1"), ("S3DLIO_243_ENTRY", "pool")],
    );
    assert_eq!(p.client, "rayon=1", "{p:?}");
}

#[test]
fn unsigned_payload_switch_is_read_after_the_file_loads() {
    let dir = dir_with_dotenv_plus("first-unsigned", "S3DLIO_UNSIGNED_PAYLOAD=1\n");
    let p = probe(
        &dir,
        &[
            ("S3DLIO_LOAD_DOTENV", "1"),
            ("S3DLIO_243_ENTRY", "unsigned"),
        ],
    );
    assert_eq!(p.client, "unsigned=true", "{p:?}");
}

#[test]
fn logger_settings_come_from_the_file() {
    let dir = dir_with_dotenv_plus("first-logger", "S3DLIO_OPLOG_LOSSLESS=1\n");
    let p = probe(
        &dir,
        &[("S3DLIO_LOAD_DOTENV", "1"), ("S3DLIO_243_ENTRY", "logger")],
    );
    assert_eq!(p.client, "lossless=true", "{p:?}");
}

#[test]
fn a_latched_get_setting_sees_the_file_on_the_first_get() {
    // S3DLIO_SKIP_HEAD is latched for the process on its first read. The endpoint and
    // credentials are in the real environment; only the switch is in the file.
    let (port, state) = ensure_mock_s3_server();
    let key = common::mock_s3_server::unique_test_key("dotenv243-skip-head");
    let dir = dir_with_dotenv_plus("first-skip-head", "S3DLIO_SKIP_HEAD=1\n");
    let uri = format!("s3://dotenv243-bkt/{key}");
    let endpoint = format!("http://127.0.0.1:{port}");
    let p = probe(
        &dir,
        &[
            ("S3DLIO_LOAD_DOTENV", "1"),
            ("S3DLIO_243_ENTRY", "skip_head"),
            ("S3DLIO_243_URI", &uri),
            ("AWS_ENDPOINT_URL", &endpoint),
            ("AWS_ACCESS_KEY_ID", "test-access-key"),
            ("AWS_SECRET_ACCESS_KEY", "test-secret-key"),
            ("AWS_REGION", "us-east-1"),
        ],
    );
    let reqs = state.requests_containing(&key);
    assert!(!reqs.is_empty(), "the GET never reached the mock: {p:?}");
    let heads: Vec<_> = reqs.iter().filter(|r| r.method == "HEAD").collect();
    assert!(
        heads.is_empty(),
        "S3DLIO_SKIP_HEAD from the file must be seen by the first get: {reqs:?}"
    );
}

// -- Round 7 (Blue R8, R9): independently public entry points, FIRST call, value only in the file --

#[test]
fn endpoint_override_first_call_beats_the_files_service_endpoint() {
    // R8: the file's AWS_ENDPOINT_URL_S3 (service-specific, higher precedence) must be applied
    // BEFORE the setter inspects the endpoint variables, so the setter replaces it.
    let dir = dir_with_dotenv_plus("r8-override", "AWS_ENDPOINT_URL_S3=http://old:9000\n");
    let p = probe(
        &dir,
        &[
            ("S3DLIO_LOAD_DOTENV", "1"),
            ("S3DLIO_243_ENTRY", "endpoint_override"),
        ],
    );
    assert_eq!(p.client, "endpoint=http://new:9000", "{p:?}");
}

fn first_call(name: &str, entry: &str, file_lines: &str) -> Probe {
    let dir = dir_with_dotenv_plus(name, file_lines);
    probe(
        &dir,
        &[("S3DLIO_LOAD_DOTENV", "1"), ("S3DLIO_243_ENTRY", entry)],
    )
}

#[test]
fn transport_builder_first_call_sees_h2c_from_the_file() {
    // R9: the client's protocol is fixed at construction; a later load cannot repair it.
    let p = first_call("r9-transport-on", "transport", "S3DLIO_H2C=1\n");
    assert_eq!(p.client, "wire=h2c", "{p:?}");
    // Control: without the setting the very same sequence speaks HTTP/1.1.
    let p = first_call("r9-transport-off", "transport", "");
    assert_eq!(p.client, "wire=http1", "{p:?}");
}

#[test]
fn h2_window_config_first_call_sees_the_file() {
    let p = first_call(
        "r9-h2-window",
        "h2_window",
        "S3DLIO_H2_ADAPTIVE_WINDOW=0\nS3DLIO_H2_STREAM_WINDOW_MB=7\nS3DLIO_H2_CONN_WINDOW_MB=9\n",
    );
    assert_eq!(p.client, "adaptive=false stream_mb=7 conn_mb=9", "{p:?}");
}

#[test]
fn timeout_and_retry_helpers_first_call_see_the_file() {
    let p = first_call(
        "r9-connect",
        "connect_timeout",
        "S3DLIO_CONNECT_TIMEOUT_SECS=37\n",
    );
    assert_eq!(p.client, "connect_timeout=37", "{p:?}");
    let p = first_call("r9-retry", "max_retry", "S3DLIO_MAX_RETRY_ATTEMPTS=9\n");
    assert_eq!(p.client, "max_retry=9", "{p:?}");
}

#[test]
fn size_cache_constructor_first_call_sees_the_limit_from_the_file() {
    // Blue integration P2-2: ObjectSizeCache::new latches the limit on its first call.
    let p = first_call(
        "p22-size-cache",
        "size_cache",
        "S3DLIO_SIZE_CACHE_MAX_ENTRIES=2\n",
    );
    assert_eq!(p.client, "entries=2", "{p:?}");
    // Control: without the setting the same five puts are all kept.
    let p = first_call("p22-size-cache-off", "size_cache", "");
    assert_eq!(p.client, "entries=5", "{p:?}");
}

#[test]
fn export_quiesce_timeout_first_call_sees_the_file() {
    let p = first_call(
        "p31-quiesce",
        "export_quiesce",
        "S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS=7\n",
    );
    assert_eq!(p.client, "quiesce=7s", "{p:?}");
}

#[test]
fn effective_thread_budget_first_call_sees_the_file() {
    // Called first and alone: no other helper has applied the policy before it.
    let p = first_call("r9-budget-first", "budget_first", "S3DLIO_RT_THREADS=11\n");
    assert_eq!(p.client, "budget=11", "{p:?}");
}

#[test]
fn multi_endpoint_from_config_first_call_sees_the_tls_ports_from_the_file() {
    let p = first_call(
        "r9-multi-endpoint-config",
        "multi_endpoint_config",
        "S3DLIO_S3_ENDPOINT_TLS_PORTS=9443\n",
    );
    assert_eq!(p.client, "urls=[Some(\"https://127.0.0.1:9443\")]", "{p:?}");
}

#[test]
fn ranged_read_with_known_length_sees_the_concurrent_threshold_from_the_file() {
    // The file lowers the concurrent-range threshold to 1 MiB; a 4 MiB ranged read as the
    // FIRST call must take the concurrent path, which (unlike the plain ranged GET) sizes the
    // object with a HEAD first. Endpoint and credentials are in the real environment.
    let (port, state) = ensure_mock_s3_server();
    let key = common::mock_s3_server::unique_test_key("dotenv243-sized-obj");
    let dir = dir_with_dotenv_plus("first-threshold", "S3DLIO_CONCURRENT_THRESHOLD=1048576\n");
    let uri = format!("s3://dotenv243-bkt/{key}");
    let endpoint = format!("http://127.0.0.1:{port}");
    let p = probe(
        &dir,
        &[
            ("S3DLIO_LOAD_DOTENV", "1"),
            ("S3DLIO_243_ENTRY", "concurrent_threshold"),
            ("S3DLIO_243_URI", &uri),
            ("AWS_ENDPOINT_URL", &endpoint),
            ("AWS_ACCESS_KEY_ID", "test-access-key"),
            ("AWS_SECRET_ACCESS_KEY", "test-secret-key"),
            ("AWS_REGION", "us-east-1"),
        ],
    );
    let reqs = state.requests_containing(&key);
    assert!(!reqs.is_empty(), "nothing reached the mock: {p:?}");
    assert!(
        reqs.iter().any(|r| r.method == "HEAD"),
        "the file's threshold must route the first ranged read down the concurrent path: {reqs:?}"
    );
}

#[cfg(feature = "backend-azure")]
#[test]
fn azure_account_url_resolver_first_call_sees_the_file() {
    let p = first_call(
        "r9-azure-url",
        "azure_account_url",
        "AZURE_STORAGE_ENDPOINT=http://127.0.0.1:10000\n",
    );
    assert_eq!(p.client, "http://127.0.0.1:10000/acct", "{p:?}");
}

#[cfg(feature = "arrow-backend")]
#[test]
fn arrow_store_first_call_sees_credentials_from_the_file() {
    // The file carries the credentials (DOTENV); the first call must get past the credential
    // check (what happens after it is a connection to a closed port, which is fine).
    let p = first_call(
        "r9-arrow",
        "arrow_get",
        "AWS_ENDPOINT_URL=http://127.0.0.1:1\n",
    );
    assert_eq!(p.client, "arrow=credentials-seen", "{p:?}");
}

#[test]
fn thread_budget_helpers_first_call_see_the_file() {
    let p = first_call(
        "r9-budget-rt",
        "thread_budget",
        "S3DLIO_RT_THREADS=11\nWORLD_SIZE=3\n",
    );
    assert_eq!(p.client, "world=3 budget=11", "{p:?}");
    // The MPI-aware budget on its own: a world size above any CPU count gives 1.
    let p = first_call("r9-budget-aware", "aware_budget", "WORLD_SIZE=100000\n");
    assert_eq!(p.client, "aware=1", "{p:?}");
}

#[test]
fn multi_endpoint_store_first_call_sees_the_tls_ports_from_the_file() {
    // The public constructor decides http versus https for a server-named endpoint before
    // any client exists.
    let p = first_call(
        "r9-multi-endpoint",
        "multi_endpoint",
        "S3DLIO_S3_ENDPOINT_TLS_PORTS=9443\n",
    );
    assert_eq!(p.client, "urls=[Some(\"https://127.0.0.1:9443\")]", "{p:?}");
}

#[cfg(feature = "gcs-community")]
#[test]
fn community_gcs_endpoint_resolver_first_call_sees_the_file() {
    let p = first_call(
        "r9-gcs-endpoint",
        "gcs_endpoint",
        "GCS_ENDPOINT_URL=http://gcs.test:4443\n",
    );
    assert_eq!(
        p.client, "gcs_endpoint=Some(\"http://gcs.test:4443\")",
        "{p:?}"
    );
}

#[cfg(feature = "backend-azure")]
#[test]
fn azure_config_default_first_call_sees_the_file() {
    let p = first_call(
        "r9-azure-config",
        "azure_config",
        "S3DLIO_AZURE_DOWNLOAD_CONCURRENCY=11\n",
    );
    assert_eq!(p.client, "download_concurrency=11", "{p:?}");
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

/// The CLI loads the file before its Tokio runtime exists: a Tokio setting that only the
/// `.env` carries (TOKIO_WORKER_THREADS, read when the runtime is built) must size it.
///
/// Tokio names its worker threads and its blocking-pool threads alike, so the count of
/// runtime threads is `workers + however many blocking threads were started` (the blocking
/// pool starts threads lazily, a few at most for `s3-cli ls` waiting on a silent server).
/// The test therefore uses a worker count far below the host's default (the default is one
/// worker per CPU) and bounds the stable count: it must be at least the configured workers
/// and no more than the workers plus a small allowance for blocking threads, which is
/// still well below the CPU count. On a host with too few CPUs for that gap it skips.
#[cfg(target_os = "linux")]
#[test]
fn s3_cli_loads_dotenv_before_the_runtime_is_built() {
    const WORKERS: usize = 2;
    const BLOCKING_ALLOWANCE: usize = 3;
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    if cpus < WORKERS + BLOCKING_ALLOWANCE + 3 {
        eprintln!(
            "SKIPPED: {cpus} CPUs leave no gap between TOKIO_WORKER_THREADS={WORKERS} \
             (+{BLOCKING_ALLOWANCE} possible blocking threads) and the default of one worker per CPU"
        );
        return;
    }

    // A server that accepts and never answers keeps `s3-cli ls` alive while we look.
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    let dir = scratch("cli-runtime");
    std::fs::write(
        dir.join(".env"),
        format!(
            "AWS_ENDPOINT_URL=http://127.0.0.1:{port}\nAWS_ACCESS_KEY_ID=k\n\
             AWS_SECRET_ACCESS_KEY=s\nAWS_REGION=us-east-1\nTOKIO_WORKER_THREADS={WORKERS}\n"
        ),
    )
    .unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_s3-cli"))
        .args(["ls", "s3://cli243-runtime/"])
        .current_dir(&dir)
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("AWS_EC2_METADATA_DISABLED", "true")
        .spawn()
        .expect("spawn s3-cli");
    let pid = child.id();
    let count_runtime_threads = || {
        std::fs::read_dir(format!("/proc/{pid}/task"))
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .filter_map(|e| std::fs::read_to_string(e.path().join("comm")).ok())
                    .filter(|c| {
                        c.trim().starts_with("tokio-rt-worker")
                            || c.trim().starts_with("tokio-runtime-w")
                    })
                    .count()
            })
            .unwrap_or(0)
    };
    // Wait until the count is non-zero and unchanged for a full second (or give up).
    let mut count = 0;
    let mut stable_polls = 0;
    for _ in 0..150 {
        let now = count_runtime_threads();
        if now > 0 && now == count {
            stable_polls += 1;
            if stable_polls >= 10 {
                break;
            }
        } else {
            stable_polls = 0;
            count = now;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    // Stop it with SIGTERM (never a forced kill) and reap it.
    let _ = Command::new("kill")
        .args(["-TERM", &pid.to_string()])
        .status();
    let _ = child.wait();
    drop(listener);
    assert!(
        stable_polls >= 10,
        "the thread count never settled (last {count})"
    );
    assert!(
        (WORKERS..=WORKERS + BLOCKING_ALLOWANCE).contains(&count),
        "the runtime must be built after the .env is loaded (TOKIO_WORKER_THREADS={WORKERS}): \
         {count} runtime threads on a {cpus}-CPU host"
    );
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

#[test]
fn direct_bounce_config_first_call_sees_the_file() {
    // Blue integration item 5: `direct_bounce::config()` latches the bounce settings once, so
    // a first call must already see the values that only the opted-in file defines.
    let p = first_call(
        "integration-bounce-config",
        "bounce_config",
        "S3DLIO_DIRECT_IO_BOUNCE_SIZE_MB=16\nS3DLIO_DIRECT_IO_BOUNCE_PAIRS_PER_NODE=3\nS3DLIO_DIRECT_IO_BOUNCE_WHEN_BUSY=fallback\nS3DLIO_DIRECT_IO_GET_MODE=pooled\n",
    );
    assert_eq!(
        p.client, "size=16777216 pairs=3 busy=Fallback get=Pooled",
        "{p:?}"
    );
}
