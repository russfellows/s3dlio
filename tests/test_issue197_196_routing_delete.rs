// tests/test_issue197_196_routing_delete.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Regression tests for
//   #197  an S3 operation given a URI that names a server (`s3://host:port/b/k`)
//         other than the one the store / client talks to must be REJECTED before
//         any request is sent -- never silently executed against the wrong server;
//   #196  `delete_prefix` on a per-endpoint client must report per-key errors
//         that S3 returns inside an HTTP 200 `DeleteObjects` body.
//
// Everything runs against in-process mock S3 servers that record every request
// they receive (`"<METHOD> <target>"`).  The "wrong server" assertions are on
// the mocks, so an unfixed build is observable as requests received (or data
// returned from the wrong server), not merely as a missing error.
//
//   G = the process-wide (global) client's server (`AWS_ENDPOINT_URL`)
//   A = the server a per-endpoint store talks to
//   B = the server a URI names

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Once, OnceLock};
use std::thread;

use bytes::Bytes;
use s3dlio::object_store::{
    direct_io_store_for_uri, high_performance_store_for_uri, ObjectStore, S3ObjectStore,
};
use s3dlio::s3_utils;

// -----------------------------------------------------------------------------
// Mock S3 server
// -----------------------------------------------------------------------------

/// What the mock answers with.
#[derive(Clone, Default)]
struct MockCfg {
    /// Keys a `ListObjectsV2` request returns.
    list_keys: Vec<String>,
    /// `DeleteObjects` answers HTTP 200 with every submitted key as an `<Error>`.
    delete_per_key_errors: bool,
}

struct Mock {
    port: u16,
    requests: Arc<Mutex<Vec<String>>>,
}

impl Mock {
    fn start() -> Self {
        Self::start_with(MockCfg::default())
    }

    fn start_with(cfg: MockCfg) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock");
        let port = listener.local_addr().expect("mock addr").port();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&requests);
        // The thread (and the listener) live for the rest of the test process.
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let seen = Arc::clone(&seen);
                let cfg = cfg.clone();
                thread::spawn(move || serve_one(stream, seen, cfg));
            }
        });
        Mock { port, requests }
    }

    fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// `host:port`, the authority a URI uses to name this mock.
    fn authority(&self) -> String {
        format!("127.0.0.1:{}", self.port)
    }

    fn requests(&self) -> Vec<String> {
        self.requests.lock().map(|r| r.clone()).unwrap_or_default()
    }

    /// Requests that mention `marker` (a bucket name unique to one test).
    fn requests_mentioning(&self, marker: &str) -> Vec<String> {
        self.requests()
            .into_iter()
            .filter(|r| r.contains(marker))
            .collect()
    }
}

fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
    hay.windows(needle.len()).position(|w| w == needle)
}

/// Every `<Key>..</Key>` value in a request body.
fn keys_in_body(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = body;
    while let Some(s) = rest.find("<Key>") {
        let after = &rest[s + "<Key>".len()..];
        match after.find("</Key>") {
            Some(e) => {
                out.push(after[..e].to_string());
                rest = &after[e + "</Key>".len()..];
            }
            None => break,
        }
    }
    out
}

fn serve_one(mut stream: TcpStream, seen: Arc<Mutex<Vec<String>>>, cfg: MockCfg) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    let header_end = loop {
        if let Some(pos) = find(&buf, b"\r\n\r\n") {
            break pos + 4;
        }
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => return,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    };
    let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
    let request_line = head.lines().next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("").to_string();

    let header = |name: &str| -> Option<String> {
        head.lines().find_map(|l| {
            let (n, v) = l.split_once(':')?;
            n.eq_ignore_ascii_case(name).then(|| v.trim().to_string())
        })
    };
    let content_length = header("content-length")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let chunked = header("transfer-encoding").is_some_and(|v| v.contains("chunked"));
    if chunked {
        while !buf[header_end..].ends_with(b"0\r\n\r\n") {
            match stream.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => buf.extend_from_slice(&chunk[..n]),
            }
        }
    } else {
        while buf.len() < header_end + content_length {
            match stream.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => buf.extend_from_slice(&chunk[..n]),
            }
        }
    }
    let body = String::from_utf8_lossy(&buf[header_end..]).to_string();

    if let Ok(mut r) = seen.lock() {
        r.push(format!("{method} {target}"));
    }

    let xml = "Content-Type: application/xml\r\n";
    let (status, extra, payload): (&str, String, String) = if method == "HEAD" {
        // Headers only; Content-Length describes the object, not a body.
        (
            "200 OK",
            "Content-Length: 4\r\nETag: \"abcd\"\r\n".to_string(),
            String::new(),
        )
    } else if method == "GET" && target.contains("list-type=2") {
        let contents: String = cfg
            .list_keys
            .iter()
            .map(|k| {
                format!(
                    "<Contents><Key>{k}</Key><Size>4</Size><ETag>\"abcd\"</ETag>\
                     <LastModified>2024-01-01T00:00:00.000Z</LastModified></Contents>"
                )
            })
            .collect();
        (
            "200 OK",
            xml.to_string(),
            format!(
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?><ListBucketResult \
                 xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\"><Name>b</Name>\
                 <Prefix></Prefix><KeyCount>{}</KeyCount><MaxKeys>1000</MaxKeys>\
                 <IsTruncated>false</IsTruncated>{contents}</ListBucketResult>",
                cfg.list_keys.len()
            ),
        )
    } else if method == "GET" {
        ("200 OK", String::new(), "data".to_string())
    } else if method == "POST" && target.contains("delete") {
        let errs: String = if cfg.delete_per_key_errors {
            keys_in_body(&body)
                .iter()
                .map(|k| {
                    format!(
                        "<Error><Key>{k}</Key><Code>AccessDenied</Code>\
                         <Message>mock per-key failure</Message></Error>"
                    )
                })
                .collect()
        } else {
            String::new()
        };
        (
            "200 OK",
            xml.to_string(),
            format!(
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?><DeleteResult \
                 xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">{errs}</DeleteResult>"
            ),
        )
    } else {
        ("200 OK", String::new(), String::new())
    };

    // HEAD must not carry a body; its Content-Length is in `extra`.
    let length = if method == "HEAD" {
        String::new()
    } else {
        format!("Content-Length: {}\r\n", payload.len())
    };
    let resp = format!(
        "HTTP/1.1 {status}\r\n{extra}{length}Connection: close\r\nx-amz-request-id: 1\r\n\r\n{payload}"
    );
    let _ = stream
        .write_all(resp.as_bytes())
        .and_then(|()| stream.flush());
}

// -----------------------------------------------------------------------------
// Environment: the process-wide client points at mock G
// -----------------------------------------------------------------------------

static GLOBAL_MOCK: OnceLock<Mock> = OnceLock::new();
static ENV: Once = Once::new();

/// G, the server `AWS_ENDPOINT_URL` names.  The environment must be set before
/// anything touches the process-wide client (it is built once from it).
fn global_mock() -> &'static Mock {
    ENV.call_once(|| {
        let g = Mock::start();
        #[allow(deprecated)]
        {
            std::env::set_var("AWS_ACCESS_KEY_ID", "test-key");
            std::env::set_var("AWS_SECRET_ACCESS_KEY", "test-secret");
            std::env::set_var("AWS_REGION", "us-east-1");
            std::env::set_var("AWS_ENDPOINT_URL", g.url());
        }
        let _ = GLOBAL_MOCK.set(g);
    });
    GLOBAL_MOCK.get().expect("global mock set by call_once")
}

static COUNTER: AtomicUsize = AtomicUsize::new(0);

/// A bucket name unique to one test, so the shared mock G can be asked what
/// *this* test sent it.
fn unique_bucket(tag: &str) -> String {
    format!(
        "b197-{tag}-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}

async fn endpoint_store(server: &Mock) -> S3ObjectStore {
    global_mock();
    S3ObjectStore::for_endpoint(&server.url())
        .await
        .expect("for_endpoint with fake credentials")
}

/// The error must name the URI and both servers (the one it names and the one
/// the store talks to).
fn assert_names(err: &anyhow::Error, uri: &str, servers: &[&str]) {
    let msg = format!("{err:#}");
    assert!(msg.contains(uri), "error must name the URI {uri}: {msg}");
    for s in servers {
        assert!(msg.contains(s), "error must name server {s}: {msg}");
    }
}

// -----------------------------------------------------------------------------
// #197 -- per-endpoint store, URI names another server
// -----------------------------------------------------------------------------

/// Every single-object operation of a per-endpoint store refuses a URI that
/// names another server: an error naming the URI and both servers, and neither
/// the store's server (A) nor the named one (B) sees a request.
#[tokio::test]
async fn per_endpoint_store_rejects_a_uri_naming_another_server() {
    let a = Mock::start();
    let b = Mock::start();
    let store = endpoint_store(&a).await;
    let uri = format!("s3://{}/bkt/dir/k", b.authority());
    let prefix = format!("s3://{}/bkt/dir/", b.authority());
    let servers = [b.authority(), a.authority()];
    let servers: Vec<&str> = servers.iter().map(String::as_str).collect();

    assert_names(&store.get(&uri).await.expect_err("get"), &uri, &servers);
    assert_names(
        &store
            .get_range(&uri, 0, Some(2))
            .await
            .expect_err("get_range"),
        &uri,
        &servers,
    );
    assert_names(
        &store
            .put(&uri, Bytes::from_static(b"x"))
            .await
            .expect_err("put"),
        &uri,
        &servers,
    );
    assert_names(
        &store
            .put_multipart(&uri, Bytes::from_static(b"x"), None)
            .await
            .expect_err("put_multipart"),
        &uri,
        &servers,
    );
    assert_names(&store.stat(&uri).await.expect_err("stat"), &uri, &servers);
    assert_names(
        &store.list(&prefix, true).await.expect_err("list"),
        &prefix,
        &servers,
    );
    {
        use futures::StreamExt;
        let mut s = store.list_stream(&prefix, true);
        let first = s.next().await.expect("list_stream yields the error");
        assert_names(&first.expect_err("list_stream"), &prefix, &servers);
    }
    assert_names(
        &store
            .delete_prefix(&prefix)
            .await
            .expect_err("delete_prefix"),
        &prefix,
        &servers,
    );
    assert_names(
        &store.get_optimized(&uri).await.expect_err("get_optimized"),
        &uri,
        &servers,
    );
    assert_names(
        &store
            .get_range_optimized(&uri, 0, Some(2), None, None)
            .await
            .expect_err("get_range_optimized"),
        &uri,
        &servers,
    );
    assert_names(
        &store.get_writer(&uri).await.err().expect("get_writer"),
        &uri,
        &servers,
    );

    assert!(
        a.requests().is_empty(),
        "the store's own server received requests: {:?}",
        a.requests()
    );
    assert!(
        b.requests().is_empty(),
        "the named server received requests: {:?}",
        b.requests()
    );
}

/// The same operations still work for URIs that name this store's own server or
/// no server at all.
#[tokio::test]
async fn per_endpoint_store_still_serves_its_own_server_and_bucket_form() {
    let a = Mock::start();
    let store = endpoint_store(&a).await;
    let own = format!("s3://{}/bkt/dir/k", a.authority());
    let bucket_form = "s3://bkt/dir/k";

    for uri in [own.as_str(), bucket_form] {
        assert_eq!(store.get(uri).await.expect("get").as_ref(), b"data");
        assert_eq!(store.stat(uri).await.expect("stat").size, 4);
        store.put(uri, Bytes::from_static(b"x")).await.expect("put");
        store.get_range(uri, 0, Some(2)).await.expect("get_range");
    }
    let reqs = a.requests();
    assert!(
        reqs.iter().any(|r| r.starts_with("GET /bkt/dir/k")),
        "{reqs:?}"
    );
    assert!(
        reqs.iter().any(|r| r.starts_with("PUT /bkt/dir/k")),
        "{reqs:?}"
    );
}

// -----------------------------------------------------------------------------
// #197 -- global-client store and the s3_utils helpers
// -----------------------------------------------------------------------------

/// A store on the process-wide client (G) refuses a URI naming a different
/// server; G and the named server receive nothing for it.
#[tokio::test]
async fn global_store_rejects_a_uri_naming_another_server() {
    let g = global_mock();
    let b = Mock::start();
    let bucket = unique_bucket("gstore");
    let store = S3ObjectStore::new();
    let uri = format!("s3://{}/{bucket}/k", b.authority());
    let prefix = format!("s3://{}/{bucket}/", b.authority());
    let servers = [b.authority(), g.authority()];
    let servers: Vec<&str> = servers.iter().map(String::as_str).collect();

    assert_names(&store.get(&uri).await.expect_err("get"), &uri, &servers);
    assert_names(
        &store
            .get_range(&uri, 0, Some(2))
            .await
            .expect_err("get_range"),
        &uri,
        &servers,
    );
    assert_names(
        &store
            .put(&uri, Bytes::from_static(b"x"))
            .await
            .expect_err("put"),
        &uri,
        &servers,
    );
    assert_names(
        &store
            .put_multipart(&uri, Bytes::from_static(b"x"), None)
            .await
            .expect_err("put_multipart"),
        &uri,
        &servers,
    );
    assert_names(&store.stat(&uri).await.expect_err("stat"), &uri, &servers);
    assert_names(
        &store.list(&prefix, true).await.expect_err("list"),
        &prefix,
        &servers,
    );
    assert_names(
        &store
            .delete_prefix(&prefix)
            .await
            .expect_err("delete_prefix"),
        &prefix,
        &servers,
    );
    assert_names(
        &store.get_optimized(&uri).await.expect_err("get_optimized"),
        &uri,
        &servers,
    );
    assert_names(
        &store.get_writer(&uri).await.err().expect("get_writer"),
        &uri,
        &servers,
    );

    assert!(
        g.requests_mentioning(&bucket).is_empty(),
        "the global server received requests: {:?}",
        g.requests_mentioning(&bucket)
    );
    assert!(b.requests().is_empty(), "{:?}", b.requests());
}

/// The `s3_utils` `*_uri_*` helpers (global client) refuse a URI naming a
/// different server, before any request.
#[tokio::test]
async fn s3_utils_uri_helpers_reject_a_uri_naming_another_server() {
    let g = global_mock();
    let b = Mock::start();
    let bucket = unique_bucket("helpers");
    let uri = format!("s3://{}/{bucket}/k", b.authority());
    let servers = [b.authority(), g.authority()];
    let servers: Vec<&str> = servers.iter().map(String::as_str).collect();

    assert_names(
        &s3_utils::get_object_uri_async(&uri).await.expect_err("get"),
        &uri,
        &servers,
    );
    assert_names(
        &s3_utils::get_object_uri_optimized_async(&uri)
            .await
            .expect_err("get_optimized"),
        &uri,
        &servers,
    );
    assert_names(
        &s3_utils::get_object_range_uri_async(&uri, 0, Some(2))
            .await
            .expect_err("get_range"),
        &uri,
        &servers,
    );
    assert_names(
        &s3_utils::get_object_range_uri_timed_async(&uri, 0, Some(2))
            .await
            .expect_err("get_range_timed"),
        &uri,
        &servers,
    );
    assert_names(
        &s3_utils::get_object_concurrent_range_async(&uri, 0, None, None, None)
            .await
            .expect_err("concurrent_range"),
        &uri,
        &servers,
    );
    assert_names(
        &s3_utils::stat_object_uri_async(&uri)
            .await
            .expect_err("stat"),
        &uri,
        &servers,
    );
    assert_names(
        &s3_utils::put_object_uri_async(&uri, Bytes::from_static(b"x"))
            .await
            .expect_err("put"),
        &uri,
        &servers,
    );
    assert_names(
        &s3_utils::put_object_multipart_uri_async(&uri, Bytes::from_static(b"x"), None)
            .await
            .expect_err("put_multipart"),
        &uri,
        &servers,
    );

    assert!(
        g.requests_mentioning(&bucket).is_empty(),
        "the global server received requests: {:?}",
        g.requests_mentioning(&bucket)
    );
    assert!(b.requests().is_empty(), "{:?}", b.requests());
}

/// The batch helpers behind Python `get_many` / `get_many_stats` / `stat_many`
/// validate the WHOLE batch first: one bad URI means no request at all, not even
/// for the good URIs in the batch.
#[tokio::test]
async fn batch_helpers_reject_the_whole_batch_before_any_request() {
    let g = global_mock();
    let b = Mock::start();
    let bucket = unique_bucket("batch");
    let good = format!("s3://{bucket}/good");
    let bad = format!("s3://{}/{bucket}/bad", b.authority());
    let servers = [b.authority(), g.authority()];
    let servers: Vec<&str> = servers.iter().map(String::as_str).collect();

    let uris = vec![good.clone(), bad.clone()];
    let u = uris.clone();
    let r = tokio::task::spawn_blocking(move || s3_utils::get_objects_parallel(&u, 4))
        .await
        .expect("join");
    assert_names(&r.expect_err("get_objects_parallel"), &bad, &servers);

    let u = uris.clone();
    let r = tokio::task::spawn_blocking(move || {
        s3_utils::get_objects_parallel_with_progress(&u, 4, None)
    })
    .await
    .expect("join");
    assert_names(
        &r.expect_err("get_objects_parallel_with_progress"),
        &bad,
        &servers,
    );

    assert_names(
        &s3_utils::stat_object_many_async(uris)
            .await
            .expect_err("stat_object_many_async"),
        &bad,
        &servers,
    );

    assert!(
        g.requests_mentioning(&bucket).is_empty(),
        "the good URI of a rejected batch must not be requested either: {:?}",
        g.requests_mentioning(&bucket)
    );
    assert!(b.requests().is_empty(), "{:?}", b.requests());
}

/// `upload_files` / `download_objects` / the multipart sink use the global
/// client, so they refuse a URI naming a different server up front.
#[tokio::test]
async fn copy_and_multipart_entry_points_reject_a_uri_naming_another_server() {
    let g = global_mock();
    let b = Mock::start();
    let bucket = unique_bucket("copy");
    let prefix = format!("s3://{}/{bucket}/p/", b.authority());
    let servers = [b.authority(), g.authority()];
    let servers: Vec<&str> = servers.iter().map(String::as_str).collect();

    let dir = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join(&bucket);
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let file = dir.join("f.bin");
    std::fs::write(&file, b"payload").expect("scratch file");

    let r = s3dlio::s3_copy::upload_files(&prefix, &[file.clone()], 2, true, None).await;
    assert_names(&r.expect_err("upload_files"), &prefix, &servers);

    let r = s3dlio::s3_copy::download_objects(&prefix, &dir.join("out"), 2, true, None).await;
    assert_names(&r.expect_err("download_objects"), &prefix, &servers);

    let sink_uri = format!("s3://{}/{bucket}/mpu", b.authority());
    let uri = sink_uri.clone();
    let r = tokio::task::spawn_blocking(move || {
        s3dlio::MultipartUploadSink::from_uri(&uri, s3dlio::MultipartUploadConfig::default())
            .map(|_| ())
    })
    .await
    .expect("join");
    assert_names(
        &r.expect_err("MultipartUploadSink::from_uri"),
        &sink_uri,
        &servers,
    );

    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        g.requests_mentioning(&bucket).is_empty(),
        "the global server received requests: {:?}",
        g.requests_mentioning(&bucket)
    );
    assert!(b.requests().is_empty(), "{:?}", b.requests());
}

/// The data-loader dataset hands one store to every URI: a list whose URIs name
/// different servers is rejected when the dataset is built.
#[test]
fn multi_backend_dataset_rejects_uris_naming_different_servers() {
    use s3dlio::data_loader::async_pool_dataloader::MultiBackendDataset;
    let a = Mock::start();
    let b = Mock::start();
    global_mock();
    let first = format!("s3://{}/bkt/one", a.authority());
    let other = format!("s3://{}/bkt/two", b.authority());

    let err = MultiBackendDataset::from_uris(vec![first.clone(), other.clone()])
        .err()
        .expect("URIs naming two servers cannot share one store");
    assert_names(&err, &other, &[&b.authority(), &a.authority()]);
    assert!(a.requests().is_empty() && b.requests().is_empty());

    // Same server (and bucket-form mixed in) is fine.
    MultiBackendDataset::from_uris(vec![first, format!("s3://{}/bkt/three", a.authority())])
        .expect("one server");
}

/// Same-server URIs through the global client still work: bucket form, and the
/// host form that names exactly the server `AWS_ENDPOINT_URL` points at.
/// (One test: the process-wide client is bound to the runtime that builds it.)
#[tokio::test]
async fn global_client_still_serves_bucket_form_and_its_own_server() {
    let g = global_mock();
    let bucket = unique_bucket("same");
    let own = format!("s3://{}/{bucket}/k", g.authority());
    let bucket_form = format!("s3://{bucket}/k");

    for uri in [&bucket_form, &own] {
        let data = s3_utils::get_object_uri_async(uri).await.expect("get");
        assert_eq!(data.as_ref(), b"data");
    }
    let stat = s3_utils::stat_object_uri_async(&own).await.expect("stat");
    assert_eq!(stat.size, 4);

    let store = S3ObjectStore::new();
    assert_eq!(store.get(&own).await.expect("store get").as_ref(), b"data");

    let n = g
        .requests_mentioning(&format!("/{bucket}/k"))
        .iter()
        .filter(|r| r.starts_with("GET"))
        .count();
    assert!(n >= 3, "all three GETs reached G: {:?}", g.requests());
}

// -----------------------------------------------------------------------------
// #197 -- factories that used to return the global store regardless of the URI
// -----------------------------------------------------------------------------

/// `direct_io_store_for_uri` / `high_performance_store_for_uri` given a
/// host-form URI must build a store for THAT server (route), not the global
/// one: the GET goes to B and G sees nothing.
#[tokio::test]
async fn performance_factories_route_a_host_form_uri_to_its_server() {
    let g = global_mock();
    let b = Mock::start();
    let bucket = unique_bucket("factory");
    let uri = format!("s3://{}/{bucket}/k", b.authority());

    // The factories create the per-endpoint client with `run_on_global_rt`,
    // which must not run on a runtime worker.
    let u = uri.clone();
    let stores = tokio::task::spawn_blocking(move || {
        (
            direct_io_store_for_uri(&u).expect("direct_io factory"),
            high_performance_store_for_uri(&u).expect("high-performance factory"),
        )
    })
    .await
    .expect("join");

    for store in [&stores.0, &stores.1] {
        let data = store.get(&uri).await.expect("get via routed store");
        assert_eq!(data.as_ref(), b"data");
    }
    assert!(
        g.requests_mentioning(&bucket).is_empty(),
        "the global server must not be used: {:?}",
        g.requests_mentioning(&bucket)
    );
    assert_eq!(
        b.requests_mentioning(&format!("/{bucket}/k")).len(),
        2,
        "{:?}",
        b.requests()
    );
}
