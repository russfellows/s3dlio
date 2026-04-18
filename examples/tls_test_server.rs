//! Minimal TLS + HTTP/2 test server for verifying ALPN negotiation.
//!
//! Generates a fresh self-signed certificate for 127.0.0.1/localhost at startup,
//! writes it to /tmp/tls_test_server.crt, then listens on https://127.0.0.1:9443
//! and advertises ["h2", "http/1.1"] in ALPN.
//!
//! Per-connection it logs the negotiated ALPN protocol and the HTTP version of
//! each request — so you can confirm h2 is being selected end-to-end.
//!
//! # Run the server
//! ```
//! cargo run --example tls_test_server
//! ```
//!
//! # Test with curl (in another terminal)
//! ```sh
//! curl --http2 --cacert /tmp/tls_test_server.crt -sv https://127.0.0.1:9443/bucket/key
//! ```
//!
//! # Test with s3-cli (in another terminal)
//! ```sh
//! export AWS_CA_BUNDLE=/tmp/tls_test_server.crt
//! export AWS_ENDPOINT_URL=https://127.0.0.1:9443
//! cargo run --bin s3-cli -- stat s3://bucket/key
//! cargo run --bin s3-cli -- put s3://bucket/key /etc/hostname
//! ```

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::Full;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::{TokioExecutor, TokioIo};
use hyper_util::server::conn::auto::Builder as AutoBuilder;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use rustls::ServerConfig;
use tokio::net::TcpListener;
use tokio_rustls::TlsAcceptor;

/// Handle any HTTP request — returns a minimal S3-compatible 200 response.
/// Logs the HTTP version so we can confirm h2 was negotiated.
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
    // Install aws-lc-rs as the rustls default crypto provider (matches the rest of s3dlio).
    // Ignore error if a provider is already installed.
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

    // -------------------------------------------------------------------------
    // Generate a self-signed certificate for 127.0.0.1 and localhost.
    // rcgen uses ring for key generation here; rustls uses aws-lc-rs for TLS.
    // -------------------------------------------------------------------------
    let subject_alt_names = vec!["127.0.0.1".to_string(), "localhost".to_string()];
    let rcgen::CertifiedKey { cert, key_pair } =
        rcgen::generate_simple_self_signed(subject_alt_names)?;

    // Write the PEM cert so the client (curl / s3-cli) can trust it.
    let cert_path = "/tmp/tls_test_server.crt";
    std::fs::write(cert_path, cert.pem())?;
    println!("Certificate written: {cert_path}");
    println!();
    println!("Test with curl:");
    println!("  curl --http2 --cacert {cert_path} -sv https://127.0.0.1:9443/");
    println!();
    println!("Test with s3-cli (in another terminal):");
    println!("  export AWS_CA_BUNDLE={cert_path}");
    println!("  export AWS_ENDPOINT_URL=https://127.0.0.1:9443");
    println!("  cargo run --bin s3-cli -- stat s3://bucket/key");
    println!();

    // -------------------------------------------------------------------------
    // Build rustls ServerConfig with ALPN ["h2", "http/1.1"].
    // -------------------------------------------------------------------------
    let cert_der: CertificateDer<'static> = cert.der().clone();
    let key_der: PrivateKeyDer<'static> =
        PrivatePkcs8KeyDer::from(key_pair.serialize_der()).into();

    let mut server_config = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert_der], key_der)?;

    // Offer both protocols — the client will select h2 if it supports it.
    server_config.alpn_protocols = vec![b"h2".to_vec(), b"http/1.1".to_vec()];

    let acceptor = TlsAcceptor::from(Arc::new(server_config));

    // -------------------------------------------------------------------------
    // Bind and serve.
    // -------------------------------------------------------------------------
    let addr: SocketAddr = "127.0.0.1:9443".parse()?;
    let listener = TcpListener::bind(addr).await?;
    println!("Listening on https://{addr}  (Ctrl-C to stop)");

    loop {
        let (tcp, peer) = listener.accept().await?;
        let acceptor = acceptor.clone();

        tokio::spawn(async move {
            // TLS handshake.
            let tls = match acceptor.accept(tcp).await {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("[server] {peer}: TLS handshake error: {e}");
                    return;
                }
            };

            // Log which protocol was negotiated.
            let proto = tls
                .get_ref()
                .1
                .alpn_protocol()
                .map(|p| String::from_utf8_lossy(p).into_owned())
                .unwrap_or_else(|| "none".into());
            eprintln!("[server] {peer}: ALPN negotiated = \"{proto}\"");

            // AutoBuilder detects HTTP/2 vs HTTP/1.1 from the connection preface.
            let io = TokioIo::new(tls);
            if let Err(e) = AutoBuilder::new(TokioExecutor::new())
                .serve_connection(io, hyper::service::service_fn(handle))
                .await
            {
                eprintln!("[server] {peer}: serve error: {e}");
            }
        });
    }
}
