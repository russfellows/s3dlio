//! Minimal h2c (HTTP/2 cleartext) test server for verifying h2c prior-knowledge negotiation.
//!
//! Listens on plain `http://127.0.0.1:9080`.  No TLS is involved — there is no certificate
//! and no ALPN.  HTTP/2 clients must use the **h2c prior-knowledge** mechanism (RFC 7540 §3.4):
//! the client sends the HTTP/2 connection preface immediately without any upgrade handshake.
//!
//! `hyper_util::server::conn::auto::Builder` detects the HTTP/2 connection preface
//! (`PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n`) and serves HTTP/2; ordinary HTTP/1.1 connections
//! are also accepted.  Per-request the HTTP version is logged so you can confirm which
//! protocol was negotiated.
//!
//! ---
//!
//! # Run the server
//!
//! ```sh
//! cargo run --example h2c_test_server
//! ```
//!
//! # Test with curl (in another terminal)
//!
//! ```sh
//! # HTTP/2 via h2c prior-knowledge — server should log "[server] HEAD /bucket/key HTTP/2"
//! curl --http2-prior-knowledge -sv http://127.0.0.1:9080/bucket/key
//!
//! # Force HTTP/1.1 — server should log "[server] HEAD /bucket/key HTTP/1.1"
//! curl --http1.1 -sv http://127.0.0.1:9080/bucket/key
//! ```
//!
//! # Test with s3-cli (in another terminal)
//!
//! ```sh
//! export AWS_ENDPOINT_URL=http://127.0.0.1:9080
//!
//! # FORCED h2c — all requests must use HTTP/2
//! S3DLIO_H2C=1 cargo run --bin s3-cli -- -v stat s3://bucket/key
//!
//! # AUTO-PROBE — h2c probe fires on the first connection; succeeds → all requests use HTTP/2
//! cargo run --bin s3-cli -- -v stat s3://bucket/key
//!
//! # FORCED HTTP/1.1 — skips h2c entirely
//! S3DLIO_H2C=0 cargo run --bin s3-cli -- -v stat s3://bucket/key
//!
//! # PUT (write path)
//! S3DLIO_H2C=1 cargo run --bin s3-cli -- -v put s3://bucket/prefix -n 1 -s 4096
//! ```
//!
//! # Expected output
//!
//! **Server side (stderr) with S3DLIO_H2C=1 or auto-probe (succeeds):**
//! ```text
//! [server] new connection from 127.0.0.1:XXXXX
//! [server] HEAD /bucket/key HTTP/2
//! ```
//!
//! **Client side (s3-cli -v) with S3DLIO_H2C=1:**
//! ```text
//! INFO HTTP version mode: FORCED HTTP/2 (S3DLIO_H2C=1)
//! INFO HTTP protocol (first response): HTTP/2.0
//! ```
//!
//! **Client side (s3-cli -v) with auto-probe (S3DLIO_H2C unset), h2c server:**
//! ```text
//! INFO HTTP version mode: auto (http:// → h2c prior-knowledge probe on first connection)
//! INFO HTTP protocol (first response): HTTP/2.0
//! ```

use std::convert::Infallible;
use std::net::SocketAddr;

use bytes::Bytes;
use http_body_util::Full;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as AutoBuilder;
use tokio::net::TcpListener;

/// Handle any HTTP request — returns a minimal S3-compatible 200 response.
/// Logs the HTTP version and method so we can confirm h2c was negotiated.
async fn handle(req: Request<hyper::body::Incoming>) -> Result<Response<Full<Bytes>>, Infallible> {
    let version = match req.version() {
        hyper::Version::HTTP_2  => "HTTP/2",
        hyper::Version::HTTP_11 => "HTTP/1.1",
        hyper::Version::HTTP_10 => "HTTP/1.0",
        _                       => "HTTP/?",
    };
    eprintln!("[server] {} {} {version}", req.method(), req.uri().path());

    // HEAD gets an empty body; everything else gets a tiny payload.
    let body_bytes: Bytes = if req.method() == Method::HEAD {
        Bytes::new()
    } else {
        Bytes::from_static(b"hello\n")
    };
    let body_len = body_bytes.len();

    let resp = Response::builder()
        .status(StatusCode::OK)
        .header("content-length", body_len)
        .header("content-type", "application/octet-stream")
        .header("etag", "\"test-etag-abcdef\"")
        .header("last-modified", "Thu, 17 Apr 2026 00:00:00 GMT")
        .header("x-amz-request-id", "TESTREQID0000001")
        .body(Full::new(body_bytes))
        .unwrap();

    Ok(resp)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let addr: SocketAddr = "127.0.0.1:9080".parse()?;
    let listener = TcpListener::bind(addr).await?;

    println!("Listening on http://{addr}  (h2c prior-knowledge + HTTP/1.1, no TLS)");
    println!();
    println!("Test with curl:");
    println!("  # HTTP/2 via h2c prior-knowledge:");
    println!("  curl --http2-prior-knowledge -sv http://127.0.0.1:9080/bucket/key");
    println!("  # Force HTTP/1.1:");
    println!("  curl --http1.1 -sv http://127.0.0.1:9080/bucket/key");
    println!();
    println!("Test with s3-cli (in another terminal):");
    println!("  export AWS_ENDPOINT_URL=http://127.0.0.1:9080");
    println!("  S3DLIO_H2C=1  cargo run --bin s3-cli -- -v stat s3://bucket/key  # forced h2c");
    println!("                cargo run --bin s3-cli -- -v stat s3://bucket/key  # auto-probe");
    println!("  S3DLIO_H2C=0  cargo run --bin s3-cli -- -v stat s3://bucket/key  # HTTP/1.1");
    println!();
    println!("Press Ctrl-C to stop");
    println!();

    loop {
        let (tcp, peer) = listener.accept().await?;

        tokio::spawn(async move {
            eprintln!("[server] new connection from {peer}");

            // AutoBuilder detects the HTTP/2 connection preface and serves h2c;
            // plain HTTP/1.1 connections are also handled transparently.
            let io = TokioIo::new(tcp);
            if let Err(e) = AutoBuilder::new(TokioExecutor::new())
                .serve_connection(io, hyper::service::service_fn(handle))
                .await
            {
                // Connection-reset / broken-pipe are normal (client closed cleanly).
                let msg = e.to_string();
                if !msg.contains("connection reset") && !msg.contains("broken pipe") {
                    eprintln!("[server] {peer}: serve error: {e}");
                }
            }
        });
    }
}
