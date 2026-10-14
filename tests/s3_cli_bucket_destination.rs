// tests/s3_cli_bucket_destination.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// Regression test for s3dlio issue #194 (and #178): `s3-cli create-bucket` and
// `s3-cli delete-bucket` given `s3://host:port/bucket` must send the request to
// THAT server, not to the one `AWS_ENDPOINT_URL` names.
//
// This runs the real `s3-cli` binary (the code path an operator uses: argument
// parsing, `extract_container_name`, `store_for_uri_with_logger`, and the
// per-endpoint client's `create_container` / `delete_container`) against two
// local mock S3 servers:
//
//   A = AWS_ENDPOINT_URL  -- must receive ZERO requests
//   B = the server named in the URI  -- must receive the CreateBucket /
//       DeleteBucket request for `bkt`
//
// The earlier CLI tests only checked bucket-name extraction; they could not see
// a request going to the wrong server.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;

/// A minimal HTTP/1.1 server that answers every request with `200 OK` (empty
/// body) and records `"<METHOD> <PATH>"` for each request it receives.
struct MockS3 {
    port: u16,
    requests: Arc<Mutex<Vec<String>>>,
}

impl MockS3 {
    fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock S3 listener");
        let port = listener.local_addr().expect("mock S3 local addr").port();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&requests);
        // The thread (and the listener) live for the rest of the test process.
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let seen = Arc::clone(&seen);
                thread::spawn(move || serve_one(stream, seen));
            }
        });
        MockS3 { port, requests }
    }

    fn url(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    fn requests(&self) -> Vec<String> {
        self.requests.lock().map(|r| r.clone()).unwrap_or_default()
    }
}

/// Read one request (headers plus any `Content-Length` body), record it, reply
/// `200` with an empty body and close the connection.
fn serve_one(mut stream: TcpStream, seen: Arc<Mutex<Vec<String>>>) {
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
    // trailing slash so the test pins the bucket, not the SDK's spelling.
    let path = target.split('?').next().unwrap_or("").trim_end_matches('/');
    if let Ok(mut r) = seen.lock() {
        r.push(format!("{method} {path}"));
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
    let _ = stream
        .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
        .and_then(|()| stream.flush());
}

/// Run the real `s3-cli <subcommand> <uri>` with `AWS_ENDPOINT_URL` = `env_endpoint`.
/// The working directory is a cargo-provided scratch dir so no stray `.env` is read.
fn run_s3_cli(subcommand: &str, uri: &str, env_endpoint: &str) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_s3-cli"))
        .arg(subcommand)
        .arg(uri)
        .current_dir(env!("CARGO_TARGET_TMPDIR"))
        .env_clear()
        .env("PATH", std::env::var_os("PATH").unwrap_or_default())
        .env("AWS_ENDPOINT_URL", env_endpoint)
        .env("AWS_ACCESS_KEY_ID", "test-access-key")
        .env("AWS_SECRET_ACCESS_KEY", "test-secret-key")
        .env("AWS_REGION", "us-east-1")
        .output()
        .expect("run s3-cli")
}

fn check_destination(subcommand: &str, expected_request: &str) {
    let a = MockS3::start(); // AWS_ENDPOINT_URL: the WRONG server for this URI
    let b = MockS3::start(); // the server named in the URI
    let uri = format!("s3://127.0.0.1:{}/bkt", b.port);

    let out = run_s3_cli(subcommand, &uri, &a.url());
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);

    assert_eq!(
        a.requests(),
        Vec::<String>::new(),
        "{subcommand} {uri}: the AWS_ENDPOINT_URL server must receive no request \
         (stdout: {stdout}; stderr: {stderr})"
    );
    assert_eq!(
        b.requests(),
        vec![expected_request.to_string()],
        "{subcommand} {uri}: the server named in the URI must receive exactly \
         {expected_request:?} (stdout: {stdout}; stderr: {stderr})"
    );
    assert!(
        out.status.success(),
        "{subcommand} {uri} should succeed against the mock (stdout: {stdout}; stderr: {stderr})"
    );
}

#[test]
fn create_bucket_goes_to_the_server_named_in_the_uri() {
    check_destination("create-bucket", "PUT /bkt");
}

#[test]
fn delete_bucket_goes_to_the_server_named_in_the_uri() {
    check_destination("delete-bucket", "DELETE /bkt");
}
