// tests/common/recording_s3.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Shared helpers for the bucket-routing tests (#183, #199, #200): a recording
// in-process mock S3 server and a runner for the real `s3-cli` binary in a
// scrubbed environment. Included with `#[path = "common/recording_s3.rs"]` by
// each test file that needs it (it is deliberately NOT part of `common/mod.rs`,
// so no other test binary compiles it).

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;

/// A minimal HTTP/1.1 S3 mock. Every request is recorded as `"<METHOD> <PATH>"`
/// (`"GET /"` for the service root) and answered `200`; `GET /` returns a
/// `ListAllMyBucketsResult` naming `buckets`.
pub struct MockS3 {
    pub port: u16,
    requests: Arc<Mutex<Vec<String>>>,
}

impl MockS3 {
    pub fn start(buckets: &[&str]) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock S3 listener");
        let port = listener.local_addr().expect("mock S3 local addr").port();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&requests);
        let list_body = Arc::new(list_buckets_xml(buckets));
        // The thread (and the listener) live for the rest of the test process.
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let seen = Arc::clone(&seen);
                let list_body = Arc::clone(&list_body);
                thread::spawn(move || serve_one(stream, seen, &list_body));
            }
        });
        MockS3 { port, requests }
    }

    pub fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    pub fn requests(&self) -> Vec<String> {
        self.requests.lock().map(|r| r.clone()).unwrap_or_default()
    }
}

fn list_buckets_xml(buckets: &[&str]) -> String {
    let entries: String = buckets
        .iter()
        .map(|b| {
            format!("<Bucket><Name>{b}</Name><CreationDate>2020-01-01T00:00:00.000Z</CreationDate></Bucket>")
        })
        .collect();
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
         <ListAllMyBucketsResult xmlns=\"http://s3.amazonaws.com/doc/2006-03-01/\">\
         <Owner><ID>owner</ID><DisplayName>owner</DisplayName></Owner>\
         <Buckets>{entries}</Buckets></ListAllMyBucketsResult>"
    )
}

fn serve_one(mut stream: TcpStream, seen: Arc<Mutex<Vec<String>>>, list_body: &str) {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let header_end = loop {
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
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
    let method = parts.next().unwrap_or("");
    let target = parts.next().unwrap_or("");
    // The SDK sends a bucket-level request as `/bkt/`; compare without the
    // trailing slash so the tests pin the bucket, not the SDK's spelling.
    let path = target.split('?').next().unwrap_or("").trim_end_matches('/');
    let recorded_path = if path.is_empty() { "/" } else { path };
    if let Ok(mut r) = seen.lock() {
        r.push(format!("{method} {recorded_path}"));
    }
    let content_length = head
        .lines()
        .find_map(|l| {
            let (name, value) = l.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    let mut body_have = buf.len() - header_end;
    while body_have < content_length {
        match stream.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => body_have += n,
        }
    }
    let response = if method == "GET" && path.is_empty() {
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/xml\r\nContent-Length: {}\r\n\
             Connection: close\r\n\r\n{}",
            list_body.len(),
            list_body
        )
    } else {
        "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_string()
    };
    let _ = stream
        .write_all(response.as_bytes())
        .and_then(|()| stream.flush());
}

// ----------------------------------------------------------------------------
// CLI harness
// ----------------------------------------------------------------------------

/// Run the real `s3-cli <args>` in a scrubbed environment. `env` is added on top
/// of the credentials/region. The working directory is a cargo scratch dir so no
/// stray `.env` is read.
// Not used by the library-only test binaries that include this file for `MockS3`.
#[allow(dead_code)]
pub fn run_s3_cli(args: &[&str], env: &[(&str, &str)]) -> std::process::Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_s3-cli"));
    cmd.args(args)
        .current_dir(env!("CARGO_TARGET_TMPDIR"))
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("AWS_ACCESS_KEY_ID", "test-access-key")
        .env("AWS_SECRET_ACCESS_KEY", "test-secret-key")
        .env("AWS_REGION", "us-east-1");
    for (k, v) in env {
        cmd.env(k, v);
    }
    cmd.output().expect("run s3-cli")
}

// Not used by the library-only test binaries that include this file for `MockS3`.
#[allow(dead_code)]
pub fn text(out: &std::process::Output) -> String {
    format!(
        "stdout: {}; stderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}
