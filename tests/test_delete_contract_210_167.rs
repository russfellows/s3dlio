// tests/test_delete_contract_210_167.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Regression tests for s3dlio issues #210 and #167 and the architect-approved
// `s3-cli delete` contract:
//
//   delete s3://bkt/foo     ONE delete request for exactly the key `foo`: no HEAD,
//                           no listing
//   delete s3://bkt/foo/    ONE delete request for exactly the key `foo/` (a trailing
//                           slash alone never authorizes a recursive delete)
//   either, object absent   exit 0 (S3 deletes are idempotent); a real failure
//                           (403 on the DELETE) is a non-zero exit naming the key
//   delete -r s3://bkt/foo/ deletes the REAL objects under the literal prefix `foo/`
//   delete -r s3://bkt/foo  literal prefix `foo` (also matches `foobar`)
//   -p/--pattern            only with -r; filters inside the -r selection
//
// Each test drives the real `s3-cli` binary against a recording in-process mock
// S3 server, once through the process-wide client (`AWS_ENDPOINT_URL` = mock,
// `s3://bkt/...`) and once through a per-endpoint client (`s3://127.0.0.1:<port>/
// bkt/...`, with `AWS_ENDPOINT_URL` pointing at a decoy that must see nothing).
// The mock keeps real state (it deletes what DeleteObjects / DELETE name) and
// logs method, path, query and the keys in every DeleteObjects body.

use std::collections::BTreeSet;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;

// ----------------------------------------------------------------------------
// Recording mock S3
// ----------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct Req {
    method: String,
    path: String,
    query: String,
    body_keys: Vec<String>,
}

impl Req {
    fn is_list(&self) -> bool {
        self.method == "GET" && self.query.contains("list-type=2")
    }
    fn is_delete_objects(&self) -> bool {
        self.method == "POST" && self.query.contains("delete")
    }
    fn is_delete_one(&self) -> bool {
        self.method == "DELETE"
    }
    fn render(&self) -> String {
        if self.is_delete_objects() {
            format!(
                "{} {}?{} body-keys={:?}",
                self.method, self.path, self.query, self.body_keys
            )
        } else if self.query.is_empty() {
            format!("{} {}", self.method, self.path)
        } else {
            format!("{} {}?{}", self.method, self.path, self.query)
        }
    }
}

#[derive(Default)]
struct State {
    keys: BTreeSet<String>,
    log: Vec<Req>,
    /// When set, every delete request (DELETE, or POST ?delete) is answered
    /// with this status and an AccessDenied body, and deletes nothing.
    delete_status: Option<u16>,
    /// Keys DeleteObjects must report as per-key errors (and not delete).
    delete_error_keys: BTreeSet<String>,
    /// When set, delimiter-less listings are paged this many keys at a time.
    page_size: Option<usize>,
    /// Answer delimiter-less listings with 403 (all of them, or from page 2 on).
    list_failure: ListFailure,
    /// A truncated page that carries no continuation token.
    truncate_without_token: bool,
    /// Delay (ms) before answering any DeleteObjects.
    delete_delay_ms: u64,
    /// Delay (ms) before answering a delimiter-less listing page.
    list_delay_ms: u64,
}

#[derive(Default, Clone, Copy, PartialEq)]
enum ListFailure {
    #[default]
    Never,
    EveryList,
    AfterFirstPage,
}

struct Mock {
    port: u16,
    state: Arc<Mutex<State>>,
}

fn url_decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut o = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            if let Ok(v) = u8::from_str_radix(&s[i + 1..i + 3], 16) {
                o.push(v);
                i += 3;
                continue;
            }
        }
        if b[i] == b'+' {
            o.push(b' ');
        } else {
            o.push(b[i]);
        }
        i += 1;
    }
    String::from_utf8_lossy(&o).to_string()
}

fn serve(mut s: TcpStream, st: Arc<Mutex<State>>) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 8192];
    let he = loop {
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p + 4;
        }
        match s.read(&mut chunk) {
            Ok(0) | Err(_) => return,
            Ok(n) => buf.extend_from_slice(&chunk[..n]),
        }
    };
    let head = String::from_utf8_lossy(&buf[..he]).to_string();
    let lower = head.to_ascii_lowercase();
    let cl: usize = lower
        .lines()
        .find_map(|l| {
            l.strip_prefix("content-length:")
                .map(|v| v.trim().parse().unwrap_or(0))
        })
        .unwrap_or(0);
    let chunked = lower.contains("transfer-encoding: chunked");
    if chunked {
        while !buf.ends_with(b"0\r\n\r\n") {
            match s.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => buf.extend_from_slice(&chunk[..n]),
            }
        }
    } else {
        while buf.len() < he + cl {
            match s.read(&mut chunk) {
                Ok(0) | Err(_) => break,
                Ok(n) => buf.extend_from_slice(&chunk[..n]),
            }
        }
    }
    let body = String::from_utf8_lossy(&buf[he..]).to_string();
    let rl = head.lines().next().unwrap_or("").to_string();
    let mut p = rl.split_whitespace();
    let method = p.next().unwrap_or("").to_string();
    let target = p.next().unwrap_or("").to_string();
    let (path, query) = target.split_once('?').unwrap_or((&target, ""));
    let q: Vec<(String, String)> = query
        .split('&')
        .filter(|x| !x.is_empty())
        .map(|kv| {
            let (k, v) = kv.split_once('=').unwrap_or((kv, ""));
            (url_decode(k), url_decode(v))
        })
        .collect();
    let get = |k: &str| q.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone());
    let key = path
        .trim_start_matches('/')
        .split_once('/')
        .map(|(_, k)| url_decode(k))
        .unwrap_or_default();
    let body_keys: Vec<String> = if method == "POST" {
        body.split("<Key>")
            .skip(1)
            .filter_map(|p| p.split_once("</Key>").map(|x| x.0.to_string()))
            .collect()
    } else {
        Vec::new()
    };
    let mut st = st.lock().expect("mock state");
    st.log.push(Req {
        method: method.clone(),
        path: url_decode(path),
        query: query.to_string(),
        body_keys: body_keys.clone(),
    });
    let mut delay_ms = 0u64;
    let fails_list = method == "GET"
        && get("list-type").is_some()
        && get("delimiter").is_none()
        && match st.list_failure {
            ListFailure::Never => false,
            ListFailure::EveryList => true,
            ListFailure::AfterFirstPage => get("continuation-token").is_some(),
        };
    let (status, resp) = if fails_list {
        (
            403,
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Error><Code>AccessDenied</Code>\
             <Message>mock refuses to list</Message></Error>"
                .to_string(),
        )
    } else if method == "GET" && get("list-type").is_some() {
        let prefix = get("prefix").unwrap_or_default();
        let delim = get("delimiter");
        if delim.is_none() {
            delay_ms = st.list_delay_ms;
        }
        let mut content_keys: Vec<String> = Vec::new();
        let mut cps: BTreeSet<String> = BTreeSet::new();
        for k in st.keys.iter().filter(|k| k.starts_with(&prefix)) {
            let rest = &k[prefix.len()..];
            match (&delim, rest.find('/')) {
                (Some(_), Some(i)) => {
                    cps.insert(format!("{}{}", prefix, &rest[..=i]));
                }
                _ => content_keys.push(k.clone()),
            }
        }
        // Paging (delimiter-less listings only). Like S3, the continuation token
        // is a KEY (the last one returned), so deleting earlier keys between
        // pages does not shift the next page.
        let after = get("continuation-token");
        let rest_keys: Vec<&String> = content_keys
            .iter()
            .filter(|k| after.as_ref().is_none_or(|a| k.as_str() > a.as_str()))
            .collect();
        let page = match (st.page_size, &delim) {
            (Some(n), None) => n,
            _ => usize::MAX,
        };
        let truncated = rest_keys.len() > page;
        let page_keys: Vec<&String> = rest_keys.into_iter().take(page).collect();
        let token = match (truncated && !st.truncate_without_token, page_keys.last()) {
            (true, Some(last)) => format!("<NextContinuationToken>{last}</NextContinuationToken>"),
            _ => String::new(),
        };
        let mut contents = String::new();
        for k in &page_keys {
            // Folder markers (`other/`) are real, zero-byte objects.
            let size = if k.ends_with('/') { 0 } else { 1 };
            contents.push_str(&format!(
                "<Contents><Key>{k}</Key><LastModified>2026-01-01T00:00:00.000Z\
                 </LastModified><ETag>\"x\"</ETag><Size>{size}</Size>\
                 <StorageClass>STANDARD</StorageClass></Contents>"
            ));
        }
        let cp: String = cps
            .iter()
            .map(|c| format!("<CommonPrefixes><Prefix>{c}</Prefix></CommonPrefixes>"))
            .collect();
        (
            200,
            format!(
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?><ListBucketResult><Name>bkt</Name>\
                 <Prefix>{prefix}</Prefix><KeyCount>1</KeyCount><MaxKeys>1000</MaxKeys>\
                 <IsTruncated>{truncated}</IsTruncated>{token}{contents}{cp}</ListBucketResult>"
            ),
        )
    } else if st.delete_status.is_some()
        && (method == "DELETE" || (method == "POST" && query.contains("delete")))
    {
        (
            st.delete_status.unwrap_or(403),
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Error><Code>AccessDenied</Code>\
             <Message>mock refuses to delete</Message></Error>"
                .to_string(),
        )
    } else if method == "POST" && query.contains("delete") {
        delay_ms = st.delete_delay_ms;
        let mut del = String::new();
        for k in &body_keys {
            if st.delete_error_keys.contains(k) {
                del.push_str(&format!(
                    "<Error><Key>{k}</Key><Code>AccessDenied</Code>\
                     <Message>mock refuses this key</Message></Error>"
                ));
            } else {
                st.keys.remove(k);
                del.push_str(&format!("<Deleted><Key>{k}</Key></Deleted>"));
            }
        }
        (
            200,
            format!("<?xml version=\"1.0\" encoding=\"UTF-8\"?><DeleteResult>{del}</DeleteResult>"),
        )
    } else if method == "DELETE" {
        st.keys.remove(&key);
        (204, String::new())
    } else if method == "PUT" {
        st.keys.insert(key);
        (200, String::new())
    } else if method == "HEAD" {
        if st.keys.contains(&key) {
            (200, String::new())
        } else {
            (404, String::new())
        }
    } else {
        (200, String::new())
    };
    drop(st);
    if delay_ms > 0 {
        thread::sleep(std::time::Duration::from_millis(delay_ms));
    }
    let _ = write!(
        s,
        "HTTP/1.1 {status} X\r\ncontent-type: application/xml\r\ncontent-length: {}\r\n\
         connection: close\r\n\r\n{resp}",
        resp.len()
    );
}

impl Mock {
    fn start(seed: &[&str]) -> Mock {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock");
        let port = listener.local_addr().expect("local addr").port();
        let state = Arc::new(Mutex::new(State {
            keys: seed.iter().map(|s| s.to_string()).collect(),
            ..State::default()
        }));
        let st = Arc::clone(&state);
        thread::spawn(move || {
            for s in listener.incoming().flatten() {
                let st = Arc::clone(&st);
                thread::spawn(move || serve(s, st));
            }
        });
        Mock { port, state }
    }
    fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }
    fn log(&self) -> Vec<Req> {
        self.state.lock().expect("state").log.clone()
    }
    fn keys(&self) -> BTreeSet<String> {
        self.state.lock().expect("state").keys.clone()
    }
    fn log_text(&self) -> String {
        self.log()
            .iter()
            .map(Req::render)
            .collect::<Vec<_>>()
            .join("\n    ")
    }
}

// ----------------------------------------------------------------------------
// Fixture: global-client mode and per-endpoint mode
// ----------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Mode {
    /// `s3://bkt/...`, `AWS_ENDPOINT_URL` = the mock.
    Global,
    /// `s3://127.0.0.1:<port>/bkt/...`, `AWS_ENDPOINT_URL` = a decoy that must see nothing.
    Endpoint,
}

struct Fixture {
    mode: Mode,
    mock: Mock,
    decoy: Mock,
}

struct Run {
    code: i32,
    out: String,
}

impl Fixture {
    fn new(mode: Mode, seed: &[&str]) -> Fixture {
        Fixture {
            mode,
            mock: Mock::start(seed),
            decoy: Mock::start(&[]),
        }
    }

    fn uri(&self, key: &str) -> String {
        match self.mode {
            Mode::Global => format!("s3://bkt/{key}"),
            Mode::Endpoint => format!("s3://127.0.0.1:{}/bkt/{key}", self.mock.port),
        }
    }

    fn run(&self, flags: &[&str], key: &str) -> Run {
        self.run_uri(flags, &self.uri(key))
    }

    fn run_uri(&self, flags: &[&str], uri: &str) -> Run {
        let endpoint = match self.mode {
            Mode::Global => self.mock.url(),
            Mode::Endpoint => self.decoy.url(),
        };
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_s3-cli"));
        cmd.arg("delete")
            .args(flags)
            .arg(uri)
            .current_dir(env!("CARGO_TARGET_TMPDIR"))
            .env_clear()
            .env("PATH", std::env::var_os("PATH").unwrap_or_default())
            .env("AWS_ENDPOINT_URL", endpoint)
            .env("AWS_ACCESS_KEY_ID", "test-access-key")
            .env("AWS_SECRET_ACCESS_KEY", "test-secret-key")
            .env("AWS_REGION", "us-east-1")
            .env("AWS_EC2_METADATA_DISABLED", "true");
        let out = cmd.output().expect("run s3-cli");
        let text = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        Run {
            code: out.status.code().unwrap_or(-1),
            out: text,
        }
    }

    /// The decoy (the `AWS_ENDPOINT_URL` server in per-endpoint mode) saw nothing.
    fn assert_decoy_untouched(&self) {
        if let Mode::Endpoint = self.mode {
            assert!(
                self.decoy.log().is_empty(),
                "[{:?}] AWS_ENDPOINT_URL decoy received requests:\n    {}",
                self.mode,
                self.decoy.log_text()
            );
        }
    }

    fn ctx(&self, r: &Run) -> String {
        format!(
            "[{:?}] exit={} output={:?}\n  requests:\n    {}\n  keys now: {:?}",
            self.mode,
            r.code,
            r.out,
            self.mock.log_text(),
            self.mock.keys()
        )
    }
}

fn set(keys: &[&str]) -> BTreeSet<String> {
    keys.iter().map(|s| s.to_string()).collect()
}

/// Every key named in any DeleteObjects body plus every single DELETE path key.
fn deleted_request_keys(f: &Fixture) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for r in f.mock.log() {
        if r.is_delete_objects() {
            out.extend(r.body_keys.iter().cloned());
        } else if r.is_delete_one() {
            if let Some((_, k)) = r.path.trim_start_matches('/').split_once('/') {
                out.insert(k.to_string());
            }
        }
    }
    out
}

/// Every key named in any delete request, one entry per occurrence (a key sent
/// twice appears twice).
fn deleted_request_key_list(f: &Fixture) -> Vec<String> {
    let mut out = Vec::new();
    for r in f.mock.log() {
        if r.is_delete_objects() {
            out.extend(r.body_keys.iter().cloned());
        } else if r.is_delete_one() {
            if let Some((_, k)) = r.path.trim_start_matches('/').split_once('/') {
                out.push(k.to_string());
            }
        }
    }
    out
}

/// The contract sends no lookup: no HEAD, no listing, and exactly the named key
/// in exactly one delete request.
fn assert_one_delete_no_lookup(f: &Fixture, r: &Run, key: &str) {
    let log = f.mock.log();
    assert!(
        !log.iter().any(|q| q.method == "HEAD"),
        "a HEAD was issued:\n{}",
        f.ctx(r)
    );
    assert!(
        !log.iter().any(Req::is_list),
        "a listing was issued:\n{}",
        f.ctx(r)
    );
    assert_eq!(
        deleted_request_key_list(f),
        vec![key.to_string()],
        "{}",
        f.ctx(r)
    );
    f.assert_decoy_untouched();
}

/// Generate one `#[test]` per mode from a body taking `Mode`.
macro_rules! both_modes {
    ($name:ident, $body:expr) => {
        mod $name {
            use super::*;
            #[test]
            fn global_client() {
                let body: fn(Mode) = $body;
                body(Mode::Global);
            }
            #[test]
            fn per_endpoint_client() {
                let body: fn(Mode) = $body;
                body(Mode::Endpoint);
            }
        }
    };
}

// ----------------------------------------------------------------------------
// 1. #210: a prefix that matches nothing must not delete unrelated objects
// ----------------------------------------------------------------------------
both_modes!(
    t1_issue210_no_slash_deletes_only_the_named_key_and_survivors_stay,
    |mode| {
        let f = Fixture::new(mode, &["other/", "other/x"]);
        let r = f.run(&[], "testprefix");
        // the key does not exist: S3's DELETE succeeds, so the command does
        assert_eq!(r.code, 0, "{}", f.ctx(&r));
        assert_one_delete_no_lookup(&f, &r, "testprefix");
        assert_eq!(f.mock.keys(), set(&["other/", "other/x"]), "{}", f.ctx(&r));
        assert!(
            r.out.contains("Delete request succeeded: ")
                && r.out.contains("testprefix")
                && r.out.contains("does not report whether the object existed")
                && !r.out.contains("Deleted s3://")
                && !r.out.contains("-r"),
            "must say the request succeeded, not that an object was removed, and carry no \
             -r hint (no trailing slash):\n{}",
            f.ctx(&r)
        );
    }
);

// ----------------------------------------------------------------------------
// 2. #167: `delete <prefix>/` without -r where only nested keys exist
// ----------------------------------------------------------------------------
both_modes!(
    t2_issue167_trailing_slash_without_r_targets_only_the_slash_key,
    |mode| {
        let seed = [
            "testprefix/obj_1.txt/f.txt",
            "testprefix/obj_2.txt/f.txt",
            "testprefix/obj_3.txt/f.txt",
        ];
        let f = Fixture::new(mode, &seed);
        let r = f.run(&[], "testprefix/");
        assert_eq!(r.code, 0, "{}", f.ctx(&r));
        assert_one_delete_no_lookup(&f, &r, "testprefix/");
        assert_eq!(f.mock.keys(), set(&seed), "{}", f.ctx(&r));
        assert!(
            r.out.contains("Delete request succeeded: ")
                && r.out.contains("use -r to delete everything under a prefix")
                && !r.out.contains("Successfully deleted"),
            "{}",
            f.ctx(&r)
        );
    }
);

// ----------------------------------------------------------------------------
// 3. A legitimate folder marker is an exact object: only it is deleted
// ----------------------------------------------------------------------------
both_modes!(t3_folder_marker_exact_delete_leaves_children, |mode| {
    let f = Fixture::new(mode, &["other/", "other/x"]);
    let r = f.run(&[], "other/");
    assert_eq!(r.code, 0, "{}", f.ctx(&r));
    assert!(r.out.contains("Delete request succeeded"), "{}", f.ctx(&r));
    assert_one_delete_no_lookup(&f, &r, "other/");
    assert_eq!(deleted_request_keys(&f), set(&["other/"]), "{}", f.ctx(&r));
    assert_eq!(f.mock.keys(), set(&["other/x"]), "{}", f.ctx(&r));
    f.assert_decoy_untouched();
});

// ----------------------------------------------------------------------------
// 4. Exact key `foo` never expands to `foobar` or `foo/child`
// ----------------------------------------------------------------------------
both_modes!(t4_exact_key_does_not_expand, |mode| {
    let f = Fixture::new(mode, &["foo", "foobar", "foo/child"]);
    let r = f.run(&[], "foo");
    assert_eq!(r.code, 0, "{}", f.ctx(&r));
    assert_one_delete_no_lookup(&f, &r, "foo");
    assert_eq!(deleted_request_keys(&f), set(&["foo"]), "{}", f.ctx(&r));
    assert_eq!(
        f.mock.keys(),
        set(&["foobar", "foo/child"]),
        "{}",
        f.ctx(&r)
    );
    f.assert_decoy_untouched();
});

// ----------------------------------------------------------------------------
// 5. -r foo/ : the real objects under the literal prefix, listed recursively
// ----------------------------------------------------------------------------
both_modes!(
    t5_recursive_trailing_slash_deletes_real_objects_only,
    |mode| {
        let f = Fixture::new(
            mode,
            &["foo/", "foo/a", "foo/b/c", "foobar", "fo", "other/x"],
        );
        let r = f.run(&["-r"], "foo/");
        assert_eq!(r.code, 0, "{}", f.ctx(&r));
        assert!(
            r.out
                .contains("Sent delete requests for 3 objects found under prefix"),
            "{}",
            f.ctx(&r)
        );
        let lists: Vec<Req> = f.mock.log().into_iter().filter(Req::is_list).collect();
        assert!(!lists.is_empty(), "no listing issued:\n{}", f.ctx(&r));
        for l in &lists {
            assert!(
                !l.query.contains("delimiter"),
                "listing used a delimiter (would return CommonPrefixes): {}\n{}",
                l.render(),
                f.ctx(&r)
            );
            assert!(
                l.query.contains("prefix=foo%2F"),
                "listing prefix is not the literal `foo/`: {}\n{}",
                l.render(),
                f.ctx(&r)
            );
        }
        assert_eq!(
            deleted_request_keys(&f),
            set(&["foo/", "foo/a", "foo/b/c"]),
            "{}",
            f.ctx(&r)
        );
        assert_eq!(
            f.mock.keys(),
            set(&["foobar", "fo", "other/x"]),
            "{}",
            f.ctx(&r)
        );
        f.assert_decoy_untouched();
    }
);

// ----------------------------------------------------------------------------
// 6. -r foo (no slash): literal prefix `foo`, so `foobar` goes too -- and nothing
//    that merely CONTAINS `foo` (no regex inferred from the key).
// ----------------------------------------------------------------------------
both_modes!(t6_recursive_no_slash_is_literal_prefix, |mode| {
    let f = Fixture::new(
        mode,
        &[
            "foo",
            "foobar",
            "foo/a",
            "afoo",
            "other/foo.txt",
            "f.o.o",
            "other/x",
        ],
    );
    let r = f.run(&["-r"], "foo");
    assert_eq!(r.code, 0, "{}", f.ctx(&r));
    assert_eq!(
        deleted_request_keys(&f),
        set(&["foo", "foobar", "foo/a"]),
        "{}",
        f.ctx(&r)
    );
    assert_eq!(
        f.mock.keys(),
        set(&["afoo", "other/foo.txt", "f.o.o", "other/x"]),
        "{}",
        f.ctx(&r)
    );
    f.assert_decoy_untouched();
});

// ----------------------------------------------------------------------------
// 7. A real failure of the delete request is an error, never "already absent"
// ----------------------------------------------------------------------------
both_modes!(t7_delete_403_is_an_error_naming_the_key, |mode| {
    let f = Fixture::new(mode, &["foo", "foobar", "foo/child"]);
    f.mock.state.lock().expect("state").delete_status = Some(403);
    let r = f.run(&[], "foo");
    assert_ne!(r.code, 0, "must fail:\n{}", f.ctx(&r));
    assert!(
        r.out.contains("failed to delete") && r.out.contains("foo"),
        "the error must name the key:\n{}",
        f.ctx(&r)
    );
    assert!(
        !r.out.contains("Delete request succeeded"),
        "no success claim:\n{}",
        f.ctx(&r)
    );
    // one attempt, no lookup, no listing, nothing else touched
    assert_one_delete_no_lookup(&f, &r, "foo");
    assert_eq!(
        f.mock.keys(),
        set(&["foo", "foobar", "foo/child"]),
        "{}",
        f.ctx(&r)
    );
});

// ----------------------------------------------------------------------------
// 8. -p needs -r, and filters inside the -r selection
// ----------------------------------------------------------------------------
both_modes!(
    t8a_pattern_without_r_is_an_error_and_sends_nothing,
    |mode| {
        let f = Fixture::new(mode, &["logs/a.log"]);
        let r = f.run(&["-p", "\\.log$"], "logs/");
        assert_ne!(r.code, 0, "must fail:\n{}", f.ctx(&r));
        assert!(
            r.out.contains("-p/--pattern") && r.out.contains("add -r"),
            "message must say -p needs -r:\n{}",
            f.ctx(&r)
        );
        assert!(
            f.mock.log().is_empty(),
            "nothing may be sent:\n{}",
            f.ctx(&r)
        );
        f.assert_decoy_untouched();
        assert_eq!(f.mock.keys(), set(&["logs/a.log"]), "{}", f.ctx(&r));
    }
);

both_modes!(t8b_pattern_filters_the_recursive_selection, |mode| {
    let f = Fixture::new(
        mode,
        &["logs/a.log", "logs/b.txt", "logs/sub/c.log", "other/z.log"],
    );
    let r = f.run(&["-r", "-p", "\\.log$"], "logs/");
    assert_eq!(r.code, 0, "{}", f.ctx(&r));
    assert!(
        r.out
            .contains("Sent delete requests for 2 objects selected under prefix")
            && r.out.contains("by pattern '\\.log$'")
            && r.out.contains("(3 listed, 2 selected; 0 reported errors)"),
        "{}",
        f.ctx(&r)
    );
    assert_eq!(
        deleted_request_keys(&f),
        set(&["logs/a.log", "logs/sub/c.log"]),
        "{}",
        f.ctx(&r)
    );
    assert_eq!(
        f.mock.keys(),
        set(&["logs/b.txt", "other/z.log"]),
        "{}",
        f.ctx(&r)
    );
    f.assert_decoy_untouched();
});

// ----------------------------------------------------------------------------
// 9. A DeleteObjects per-key error: non-zero exit naming the key
// ----------------------------------------------------------------------------
both_modes!(t9_per_key_delete_error_fails_and_names_the_key, |mode| {
    let f = Fixture::new(mode, &["foo/a", "foo/b", "foo/c"]);
    f.mock
        .state
        .lock()
        .expect("state")
        .delete_error_keys
        .insert("foo/b".to_string());
    let r = f.run(&["-r"], "foo/");
    assert_ne!(r.code, 0, "must fail:\n{}", f.ctx(&r));
    assert!(r.out.contains("foo/b"), "must name the key:\n{}", f.ctx(&r));
    assert!(
        !r.out.contains("Successfully deleted"),
        "must not claim success:\n{}",
        f.ctx(&r)
    );
    assert_eq!(f.mock.keys(), set(&["foo/b"]), "{}", f.ctx(&r));
    f.assert_decoy_untouched();
});

// ----------------------------------------------------------------------------
// 10. A paged listing (continuation tokens) and more than one DeleteObjects batch
// ----------------------------------------------------------------------------
both_modes!(t10_recursive_pages_and_batches_cover_every_object, |mode| {
    let big: Vec<String> = (0..2500).map(|i| format!("big/{i:05}")).collect();
    let mut seed: Vec<&str> = big.iter().map(String::as_str).collect();
    seed.extend(["bigger", "other/x"]);
    let f = Fixture::new(mode, &seed);
    f.mock.state.lock().expect("state").page_size = Some(1000);
    let r = f.run(&["-r"], "big/");
    assert_eq!(r.code, 0, "exit {} output {:?}", r.code, r.out);
    assert!(
        r.out
            .contains("Sent delete requests for 2500 objects found under prefix"),
        "{:?}",
        r.out
    );
    let log = f.mock.log();
    assert_eq!(
        log.iter().filter(|q| q.is_list()).count(),
        3,
        "3 pages of up to 1000"
    );
    let bodies: Vec<&Req> = log.iter().filter(|q| q.is_delete_objects()).collect();
    assert_eq!(bodies.len(), 3, "2500 keys = 1000 + 1000 + 500");
    assert!(bodies.iter().all(|q| q.body_keys.len() <= 1000));
    let mut sent = deleted_request_key_list(&f);
    assert_eq!(sent.len(), 2500, "each key must be sent exactly once");
    sent.sort();
    assert_eq!(sent, big, "each key must be sent exactly once");
    assert_eq!(f.mock.keys(), set(&["bigger", "other/x"]));
    f.assert_decoy_untouched();
});

// ----------------------------------------------------------------------------
// 11. After a failure no further DeleteObjects request is SENT, not even for a
//     partial `chunks(1000)` buffer that the end of the stream flushes.
//
//     Timing (ms): listing pages of 500 keys arrive every 100; the first batch
//     (1000 keys: pages 1+2) is sent at 200 and its answer, a per-key error,
//     arrives at 350. Page 3 (the start of batch 2) is buffered at 300, page 4
//     would complete it at 400 -- but the failure at 350 ends the stream, which
//     flushes the 500 buffered keys. They must be counted as not attempted and
//     never sent.
// ----------------------------------------------------------------------------
both_modes!(t11_no_delete_request_is_sent_after_a_failure, |mode| {
    let keys: Vec<String> = (0..3000).map(|i| format!("many/{i:05}")).collect();
    let seed: Vec<&str> = keys.iter().map(String::as_str).collect();
    let f = Fixture::new(mode, &seed);
    {
        let mut st = f.mock.state.lock().expect("state");
        st.page_size = Some(500);
        st.list_delay_ms = 100;
        st.delete_delay_ms = 150;
        st.delete_error_keys.insert("many/00000".to_string());
    }
    let r = f.run(&["-r"], "many/");
    assert_ne!(r.code, 0, "must fail:\n{}", f.ctx(&r));
    let sent = deleted_request_key_list(&f);
    assert_eq!(
        sent.len(),
        1000,
        "only the first batch may be sent, got {}: {:?}",
        sent.len(),
        r.out
    );
    assert!(
        r.out
            .contains("Sent delete requests for 1000 of the 1500 listed")
            && r.out.contains("500 selected objects were NOT attempted"),
        "message must report what was and was not attempted: {:?}",
        r.out
    );
    assert!(r.out.contains("many/00000"), "{:?}", r.out);
    f.assert_decoy_untouched();
});

// ----------------------------------------------------------------------------
// 12. A truncated page with no continuation token is an error
// ----------------------------------------------------------------------------
both_modes!(t12_truncated_page_without_token_is_an_error, |mode| {
    let f = Fixture::new(
        mode,
        &["foo/a", "foo/b", "foo/c", "foo/d", "foo/e", "other/x"],
    );
    {
        let mut st = f.mock.state.lock().expect("state");
        st.page_size = Some(3);
        st.truncate_without_token = true;
    }
    let r = f.run(&["-r"], "foo/");
    assert_ne!(r.code, 0, "must fail:\n{}", f.ctx(&r));
    assert!(
        r.out.contains("truncated") && r.out.contains("listing failed"),
        "{}",
        f.ctx(&r)
    );
    // The incomplete listing cancels the batch being assembled: the three keys
    // of the first page are not sent, they are counted as not attempted.
    assert!(
        r.out.contains("Sent delete requests for 0 of the 3 listed")
            && r.out.contains("3 selected objects were NOT attempted"),
        "{}",
        f.ctx(&r)
    );
    assert!(deleted_request_key_list(&f).is_empty(), "{}", f.ctx(&r));
    assert_eq!(
        f.mock.keys(),
        set(&["foo/a", "foo/b", "foo/c", "foo/d", "foo/e", "other/x"])
    );
    f.assert_decoy_untouched();
});

// ----------------------------------------------------------------------------
// 13. A listing error part-way: non-zero, accurate message, nothing after it
// ----------------------------------------------------------------------------
both_modes!(t13_listing_error_midway_is_reported_accurately, |mode| {
    let f = Fixture::new(mode, &["foo/a", "foo/b", "foo/c", "foo/d", "foo/e"]);
    {
        let mut st = f.mock.state.lock().expect("state");
        st.page_size = Some(2);
        st.list_failure = ListFailure::AfterFirstPage;
    }
    let r = f.run(&["-r"], "foo/");
    assert_ne!(r.code, 0, "must fail:\n{}", f.ctx(&r));
    // The listing failed while the batch holding foo/a and foo/b was being
    // assembled: that batch is NOT sent, its keys are counted as not attempted.
    assert!(
        r.out.contains("Sent delete requests for 0 of the 2 listed")
            && r.out.contains("no delete request reported an error")
            && r.out.contains("2 selected objects were NOT attempted")
            && r.out.contains("listing failed")
            && r.out.contains("mock refuses to list"),
        "{}",
        f.ctx(&r)
    );
    assert!(!r.out.contains("0 reported errors"), "{}", f.ctx(&r));
    assert!(
        deleted_request_key_list(&f).is_empty(),
        "no delete request may be sent:\n{}",
        f.ctx(&r)
    );
    assert_eq!(
        f.mock.keys(),
        set(&["foo/a", "foo/b", "foo/c", "foo/d", "foo/e"]),
        "{}",
        f.ctx(&r)
    );
    f.assert_decoy_untouched();
});

// A full batch that was already complete when the listing failed IS sent; only
// the partly filled batch in which the failure occurred is held back.
both_modes!(
    t13b_listing_error_sends_full_batches_but_not_the_partial_one,
    |mode| {
        let names: Vec<String> = (0..2500).map(|i| format!("many/{i:05}")).collect();
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        let f = Fixture::new(mode, &refs);
        {
            let mut st = f.mock.state.lock().expect("state");
            st.page_size = Some(1200);
            st.list_failure = ListFailure::AfterFirstPage;
        }
        let r = f.run(&["-r"], "many/");
        assert_ne!(r.code, 0, "must fail:\n{}", f.ctx(&r));
        // page 1 listed 1200 keys: 1000 form a full batch (sent), 200 are the
        // partial batch in which page 2 failed (not sent)
        assert!(
            r.out
                .contains("Sent delete requests for 1000 of the 1200 listed")
                && r.out.contains("200 selected objects were NOT attempted")
                && r.out.contains("mock refuses to list"),
            "{}",
            f.ctx(&r)
        );
        let sent = deleted_request_key_list(&f);
        assert_eq!(sent.len(), 1000, "{}", f.ctx(&r));
        assert_eq!(f.mock.keys().len(), 1500, "{}", f.ctx(&r));
        assert!(f.mock.keys().contains("many/01100"));
        f.assert_decoy_untouched();
    }
);

// ----------------------------------------------------------------------------
// 14. A listing that fails at once reports the listing error, not a delete summary
// ----------------------------------------------------------------------------
both_modes!(
    t14_first_page_listing_failure_is_reported_directly,
    |mode| {
        let f = Fixture::new(mode, &["foo/a", "foo/b"]);
        f.mock.state.lock().expect("state").list_failure = ListFailure::EveryList;
        let r = f.run(&["-r"], "foo/");
        assert_ne!(r.code, 0, "must fail:\n{}", f.ctx(&r));
        assert!(
            r.out.contains("cannot list objects under prefix")
                && r.out.contains("nothing was deleted")
                && r.out.contains("mock refuses to list"),
            "{}",
            f.ctx(&r)
        );
        assert!(
            !r.out.contains("Sent delete requests"),
            "no delete summary for a listing failure:\n{}",
            f.ctx(&r)
        );
        assert!(deleted_request_key_list(&f).is_empty(), "{}", f.ctx(&r));
        assert_eq!(f.mock.keys(), set(&["foo/a", "foo/b"]));
        f.assert_decoy_untouched();
    }
);

// ----------------------------------------------------------------------------
// 15. -p matching nothing: said plainly, exit 0, nothing sent
// ----------------------------------------------------------------------------
both_modes!(t15_pattern_matching_nothing_is_reported, |mode| {
    let f = Fixture::new(mode, &["logs/a.log", "logs/b.txt", "logs/c.txt"]);
    let r = f.run(&["-r", "-p", "zzz"], "logs/");
    assert_eq!(r.code, 0, "{}", f.ctx(&r));
    assert!(
        r.out.contains("No objects under prefix")
            && r.out.contains("match pattern 'zzz'")
            && r.out.contains("(3 listed)"),
        "{}",
        f.ctx(&r)
    );
    assert!(deleted_request_key_list(&f).is_empty(), "{}", f.ctx(&r));
    assert_eq!(f.mock.keys().len(), 3);
    f.assert_decoy_untouched();
});

// ----------------------------------------------------------------------------
// 16. A URI the store cannot use (`-r s3://bkt`, no key separator) is reported
//     as the listing error it is
// ----------------------------------------------------------------------------
#[test]
fn t16_recursive_on_a_malformed_uri_reports_the_listing_error() {
    let f = Fixture::new(Mode::Global, &["a", "b/c"]);
    let r = f.run_uri(&["-r"], "s3://bkt");
    assert_ne!(r.code, 0, "{}", f.ctx(&r));
    assert!(
        r.out.contains("cannot list objects under prefix s3://bkt")
            && r.out.contains("nothing was deleted")
            && !r.out.contains("Sent delete requests"),
        "{}",
        f.ctx(&r)
    );
    assert!(f.mock.log().is_empty(), "{}", f.ctx(&r));
    assert_eq!(f.mock.keys(), set(&["a", "b/c"]));
}

// ----------------------------------------------------------------------------
// 17. file:// : an exact delete of a DIRECTORY is refused (it would otherwise be
//     a recursive delete without -r); -r with a trailing '/' still works, a
//     regular file is deleted, a missing path is success
// ----------------------------------------------------------------------------
#[test]
fn t17_file_backend_exact_delete_of_a_directory_is_refused() {
    let f = Fixture::new(Mode::Global, &[]);
    let base = std::path::Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("t17-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let dir = base.join("d");
    std::fs::create_dir_all(dir.join("sub")).expect("mkdir");
    std::fs::write(dir.join("f1"), b"1").expect("write");
    std::fs::write(dir.join("sub").join("g"), b"3").expect("write");
    std::fs::write(base.join("plain"), b"x").expect("write");

    for uri in [
        format!("file://{}", dir.display()),
        format!("file://{}/", dir.display()),
    ] {
        let r = f.run_uri(&[], &uri);
        assert_ne!(r.code, 0, "must fail:\n{}", f.ctx(&r));
        assert!(
            r.out.contains("is a directory") && r.out.contains("-r"),
            "{}",
            f.ctx(&r)
        );
        assert!(!r.out.contains("Delete request succeeded"), "{}", f.ctx(&r));
        assert!(dir.join("f1").exists() && dir.join("sub").join("g").exists());
    }

    // -r with a trailing '/': the existing recursive contract
    let r = f.run_uri(&["-r"], &format!("file://{}/", dir.display()));
    assert_eq!(r.code, 0, "{}", f.ctx(&r));
    assert!(!dir.join("f1").exists() && !dir.join("sub").join("g").exists());

    // a regular file is deleted; a path that is already gone is success
    let plain = format!("file://{}", base.join("plain").display());
    let r = f.run_uri(&[], &plain);
    assert_eq!(r.code, 0, "{}", f.ctx(&r));
    assert!(!base.join("plain").exists());
    let r = f.run_uri(&[], &plain);
    assert_eq!(r.code, 0, "{}", f.ctx(&r));
    let _ = std::fs::remove_dir_all(&base);
}
