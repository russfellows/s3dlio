// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

//! Issue #253: the Azure credential modes, end to end against local fake services. No Azure
//! account, network access or `az` login is used.
//!
//! One process, one test function, because the credentials are process-wide and the
//! environment is process-wide:
//!
//! * a fake HTTPS Blob service and a fake Entra ID (AAD) token endpoint on 127.0.0.1, with a
//!   throwaway CA that the SDK's TLS stack trusts through `SSL_CERT_FILE`;
//! * a fake App Service managed-identity endpoint (`IDENTITY_ENDPOINT`, plain http);
//! * a fake plain-http Blob service for SAS.
//!
//! The Blob service refuses (401) any request without the bearer token the matching token
//! endpoint issued, so a pass means the whole chain worked: credential selection, token
//! request, token reuse, and the token sent on every blob request.
//!
//! Not covered here: the real IMDS address (hard-coded to 169.254.169.254 in the SDK), real
//! Entra ID, and a real AKS federated token.

#![cfg(feature = "backend-azure")]

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use s3dlio::object_store::{AzureObjectStore, ObjectStore};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use tokio::net::TcpListener;

const MI_TOKEN: &str = "token-from-managed-identity";
const SECRET_TOKEN: &str = "token-from-client-secret";
const WORKLOAD_TOKEN: &str = "token-from-workload-identity";
const CERT_TOKEN: &str = "token-from-client-certificate";
/// Tenant used only by the client-certificate scenario, so the fake token endpoint can tell
/// its requests from the workload-identity ones (both send a signed assertion).
const CERT_TENANT: &str = "99999999-8888-7777-6666-555555555555";
const SAS: &str = "sv=2024-11-04&sig=abc%2Bdef%3D&se=2099-01-01";

/// What the fake services saw.
#[derive(Default)]
struct Seen {
    /// (method, decoded path, raw query, authorization header) of every blob request.
    blob_requests: Vec<(String, String, String, Option<String>)>,
    /// Form bodies posted to the token endpoint.
    aad_requests: Vec<String>,
    /// Query strings of managed-identity token requests.
    msi_requests: Vec<String>,
    blobs: HashMap<String, Vec<u8>>,
}

type State = Arc<Mutex<Seen>>;

fn response(status: StatusCode, body: impl Into<Bytes>) -> Response<Full<Bytes>> {
    Response::builder()
        .status(status)
        .body(Full::new(body.into()))
        .unwrap()
}

/// Decode `%XX` escapes (and `+` as a space when `plus_is_space`, as in form bodies).
fn pct_decode(s: &str, plus_is_space: bool) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'%' if i + 3 <= bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
                if let Ok(b) = u8::from_str_radix(hex, 16) {
                    out.push(b);
                    i += 3;
                    continue;
                }
                out.push(b'%');
            }
            b'+' if plus_is_space => out.push(b' '),
            b => out.push(b),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The value of `name` in a query string or form body (`&`-separated `k=v`).
fn query_param(query: &str, name: &str) -> Option<String> {
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        (pct_decode(k, true) == name).then(|| pct_decode(v, false))
    })
}

/// Which bearer token a blob account demands (`None` = SAS, no bearer token).
fn expected_token(account: &str) -> Option<&'static str> {
    match account {
        "acctmi" => Some(MI_TOKEN),
        "acctcs" | "acctauto" => Some(SECRET_TOKEN),
        "acctwi" => Some(WORKLOAD_TOKEN),
        "acctcert" => Some(CERT_TOKEN),
        "acctforbidden" => Some(MI_TOKEN),
        _ => None,
    }
}

async fn handle(state: State, req: Request<Incoming>) -> Response<Full<Bytes>> {
    let method = req.method().clone();
    let path = percent_decode(req.uri().path());
    let query = req.uri().query().unwrap_or("").to_string();
    let auth = req
        .headers()
        .get("authorization")
        .map(|v| v.to_str().unwrap_or("").to_string());
    let range = req
        .headers()
        .get("range")
        .or_else(|| req.headers().get("x-ms-range"))
        .map(|v| v.to_str().unwrap_or("").to_string());
    let body = req
        .into_body()
        .collect()
        .await
        .map(|b| b.to_bytes())
        .unwrap_or_default();

    // Entra ID token endpoint: POST /<tenant>/oauth2/v2.0/token
    if method == Method::POST && path.ends_with("/oauth2/v2.0/token") {
        let form = String::from_utf8_lossy(&body).to_string();
        state.lock().unwrap().aad_requests.push(form.clone());
        let token = if path.starts_with(&format!("/{CERT_TENANT}/")) {
            // A JWT signed with the certificate's key: three dot-separated parts.
            let assertion = query_param(&form, "client_assertion").unwrap_or_default();
            if assertion.split('.').count() != 3
                || query_param(&form, "client_assertion_type").as_deref()
                    != Some("urn:ietf:params:oauth:client-assertion-type:jwt-bearer")
            {
                return response(StatusCode::UNAUTHORIZED, "{\"error\":\"invalid_client\"}");
            }
            CERT_TOKEN
        } else if query_param(&form, "client_assertion").is_some() {
            if query_param(&form, "client_assertion").as_deref() != Some("federated-jwt") {
                return response(StatusCode::UNAUTHORIZED, "{\"error\":\"invalid_client\"}");
            }
            WORKLOAD_TOKEN
        } else {
            if query_param(&form, "client_secret").as_deref() != Some("the-secret") {
                return response(StatusCode::UNAUTHORIZED, "{\"error\":\"invalid_client\"}");
            }
            SECRET_TOKEN
        };
        return response(
            StatusCode::OK,
            format!(
                "{{\"access_token\":\"{token}\",\"expires_in\":3600,\"token_type\":\"Bearer\"}}"
            ),
        );
    }

    // App Service managed-identity endpoint: GET /msi/token
    if path == "/msi/token" {
        state.lock().unwrap().msi_requests.push(query.clone());
        let expires = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs()
            + 3600;
        return response(
            StatusCode::OK,
            format!(
                "{{\"access_token\":\"{MI_TOKEN}\",\"expires_on\":\"{expires}\",\
                 \"token_type\":\"Bearer\",\"resource\":\"https://storage.azure.com\"}}"
            ),
        );
    }

    // Blob service: /<account>/<container>[/<key>]
    let mut parts = path.trim_start_matches('/').splitn(3, '/');
    let account = parts.next().unwrap_or("").to_string();
    let container = parts.next().unwrap_or("").to_string();
    let key = parts.next().unwrap_or("").to_string();
    state.lock().unwrap().blob_requests.push((
        method.to_string(),
        path.clone(),
        query.clone(),
        auth.clone(),
    ));

    match expected_token(&account) {
        Some(token) => {
            if auth.as_deref() != Some(&format!("Bearer {token}")) {
                return response(StatusCode::UNAUTHORIZED, "no or wrong bearer token");
            }
            if account == "acctforbidden" {
                return response(StatusCode::FORBIDDEN, "not allowed");
            }
        }
        None => {
            if auth.is_some() {
                return response(StatusCode::BAD_REQUEST, "bearer token sent in SAS mode");
            }
            let sig = query_param(&query, "sig");
            if sig.as_deref() != Some("abc+def=") || query_param(&query, "sv").is_none() {
                return response(StatusCode::FORBIDDEN, "missing or wrong SAS");
            }
        }
    }

    let id = format!("{account}/{container}/{key}");
    let comp = query_param(&query, "comp");
    if method == Method::GET && key.is_empty() && comp.as_deref() == Some("list") {
        let seen = state.lock().unwrap();
        let prefix = format!("{account}/{container}/");
        let mut blobs = String::new();
        for (name, data) in seen.blobs.iter().filter(|(n, _)| n.starts_with(&prefix)) {
            blobs.push_str(&format!(
                "<Blob><Name>{}</Name><Properties><Content-Length>{}</Content-Length>\
                 </Properties></Blob>",
                &name[prefix.len()..],
                data.len()
            ));
        }
        return response(
            StatusCode::OK,
            format!(
                "<?xml version=\"1.0\" encoding=\"utf-8\"?><EnumerationResults \
                 ContainerName=\"{container}\"><Blobs>{blobs}</Blobs><NextMarker /></EnumerationResults>"
            ),
        );
    }
    match method {
        Method::PUT => {
            state.lock().unwrap().blobs.insert(id, body.to_vec());
            Response::builder()
                .status(StatusCode::CREATED)
                .header("etag", "\"0x1\"")
                .header("last-modified", "Wed, 14 Oct 2026 10:00:00 GMT")
                .body(Full::new(Bytes::new()))
                .unwrap()
        }
        Method::DELETE => {
            let existed = state.lock().unwrap().blobs.remove(&id).is_some();
            response(
                if existed {
                    StatusCode::ACCEPTED
                } else {
                    StatusCode::NOT_FOUND
                },
                "",
            )
        }
        Method::GET | Method::HEAD => {
            let Some(data) = state.lock().unwrap().blobs.get(&id).cloned() else {
                return response(StatusCode::NOT_FOUND, "");
            };
            let total = data.len();
            let (status, chunk, content_range) = match range.as_deref() {
                Some(r) => {
                    let r = r.trim_start_matches("bytes=");
                    let (s, e) = r.split_once('-').unwrap();
                    let start: usize = s.parse().unwrap();
                    let end: usize = if e.is_empty() {
                        total - 1
                    } else {
                        e.parse::<usize>().unwrap().min(total - 1)
                    };
                    (
                        StatusCode::PARTIAL_CONTENT,
                        data[start..=end].to_vec(),
                        Some(format!("bytes {start}-{end}/{total}")),
                    )
                }
                None => (StatusCode::OK, data, None),
            };
            let mut builder = Response::builder()
                .status(status)
                .header("etag", "\"0x1\"")
                .header("last-modified", "Wed, 14 Oct 2026 10:00:00 GMT")
                .header("x-ms-blob-type", "BlockBlob")
                .header("accept-ranges", "bytes")
                .header("content-length", chunk.len().to_string());
            if let Some(cr) = content_range {
                builder = builder.header("content-range", cr);
            }
            let body = if method == Method::HEAD {
                Bytes::new()
            } else {
                Bytes::from(chunk)
            };
            // HEAD must announce the full length.
            if method == Method::HEAD {
                builder = builder.header("content-length", total.to_string());
            }
            let mut resp = builder.body(Full::new(body)).unwrap();
            if method == Method::HEAD {
                resp.headers_mut()
                    .insert("content-length", total.to_string().parse().unwrap());
            }
            resp
        }
        _ => response(StatusCode::METHOD_NOT_ALLOWED, ""),
    }
}

fn percent_decode(s: &str) -> String {
    pct_decode(s, false)
}

/// Serve `state` on `listener`, with TLS if `tls` is given.
fn serve(listener: TcpListener, state: State, tls: Option<tokio_rustls::TlsAcceptor>) {
    tokio::spawn(async move {
        loop {
            let Ok((tcp, _)) = listener.accept().await else {
                continue;
            };
            let state = Arc::clone(&state);
            let tls = tls.clone();
            tokio::spawn(async move {
                let service = service_fn(move |req| {
                    let state = Arc::clone(&state);
                    async move { Ok::<_, Infallible>(handle(state, req).await) }
                });
                let builder = hyper::server::conn::http1::Builder::new();
                match tls {
                    Some(acceptor) => {
                        if let Ok(stream) = acceptor.accept(tcp).await {
                            let _ = builder
                                .serve_connection(TokioIo::new(stream), service)
                                .await;
                        }
                    }
                    None => {
                        let _ = builder.serve_connection(TokioIo::new(tcp), service).await;
                    }
                }
            });
        }
    });
}

fn tls_material(dir: &std::path::Path) -> (tokio_rustls::TlsAcceptor, std::path::PathBuf) {
    use rcgen::{BasicConstraints, CertificateParams, DnType, IsCa, KeyPair};
    use rustls::pki_types::{PrivateKeyDer, PrivatePkcs8KeyDer};
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

    let ca_key = KeyPair::generate().unwrap();
    let mut ca_params = CertificateParams::new(vec![]).unwrap();
    ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    ca_params
        .distinguished_name
        .push(DnType::CommonName, "s3dlio test CA");
    let ca = ca_params.self_signed(&ca_key).unwrap();

    let leaf_key = KeyPair::generate().unwrap();
    let leaf = CertificateParams::new(vec!["127.0.0.1".to_string(), "localhost".to_string()])
        .unwrap()
        .signed_by(&leaf_key, &ca, &ca_key)
        .unwrap();

    let ca_path = dir.join("test-ca.pem");
    std::fs::write(&ca_path, ca.pem()).unwrap();

    let key: PrivateKeyDer<'static> = PrivatePkcs8KeyDer::from(leaf_key.serialize_der()).into();
    let config = rustls::ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![leaf.der().clone()], key)
        .unwrap();
    (tokio_rustls::TlsAcceptor::from(Arc::new(config)), ca_path)
}

async fn bind() -> (TcpListener, u16) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    (listener, port)
}

fn uri(account: &str, key: &str) -> String {
    format!("az://{account}/ctr/{key}")
}

fn set(name: &str, value: &str) {
    std::env::set_var(name, value);
}

fn unset(name: &str) {
    std::env::remove_var(name);
}

/// put, get (full and range), stat, list and delete one object; every call must succeed.
async fn exercise(store: &AzureObjectStore, account: &str, key: &str) {
    let data = Bytes::from(vec![0x42u8; 5000]);
    store
        .put(&uri(account, key), data.clone())
        .await
        .unwrap_or_else(|e| panic!("put: {e:#}"));
    assert_eq!(
        store
            .get(&uri(account, key))
            .await
            .unwrap_or_else(|e| panic!("get: {e:#}")),
        data
    );
    assert_eq!(
        store
            .get_range(&uri(account, key), 10, Some(20))
            .await
            .unwrap()
            .len(),
        20
    );
    assert_eq!(store.stat(&uri(account, key)).await.unwrap().size, 5000);
    let listed = store
        .list(&format!("az://{account}/ctr/"), true)
        .await
        .unwrap();
    assert!(listed.iter().any(|u| u.ends_with(key)), "{listed:?}");
    store
        .delete(&uri(account, key))
        .await
        .unwrap_or_else(|e| panic!("delete: {e:#}"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn credential_modes_end_to_end() {
    // Start from a known environment whatever the developer's shell holds: nothing that selects
    // a credential or an endpoint may leak in.
    for (name, _) in std::env::vars() {
        if name.starts_with("AZURE_")
            || name.starts_with("S3DLIO_AZURE_")
            || matches!(
                name.as_str(),
                "IDENTITY_ENDPOINT"
                    | "IDENTITY_HEADER"
                    | "IMDS_ENDPOINT"
                    | "MSI_ENDPOINT"
                    | "MSI_SECRET"
            )
        {
            std::env::remove_var(name);
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let (acceptor, ca_path) = tls_material(dir.path());
    // The SDK's TLS stack (reqwest + rustls platform verifier) trusts this CA.
    set("SSL_CERT_FILE", ca_path.to_str().unwrap());

    let https_state: State = Arc::default();
    let http_state: State = Arc::default();
    let (https_listener, https_port) = bind().await;
    let (http_listener, http_port) = bind().await;
    serve(https_listener, Arc::clone(&https_state), Some(acceptor));
    serve(http_listener, Arc::clone(&http_state), None);

    let https_base = format!("https://127.0.0.1:{https_port}");
    set("AZURE_STORAGE_ENDPOINT", &https_base);
    set("AZURE_AUTHORITY_HOST", &https_base);
    for name in ["AZURE_BLOB_ENDPOINT_URL", "S3DLIO_AZURE_SAS_TOKEN"] {
        unset(name);
    }
    let store = AzureObjectStore::new();

    // ---- managed identity (App Service endpoint style), token reused across requests --------
    set("S3DLIO_AZURE_CREDENTIAL", "Managed-Identity");
    set(
        "IDENTITY_ENDPOINT",
        &format!("http://127.0.0.1:{http_port}/msi/token"),
    );
    set("IDENTITY_HEADER", "header-secret");
    unset("AZURE_CLIENT_ID");
    for i in 0..4 {
        exercise(&store, "acctmi", &format!("mi-{i}.bin")).await;
    }
    {
        let seen = http_state.lock().unwrap();
        assert_eq!(
            seen.msi_requests.len(),
            1,
            "managed identity asked for a token {} times for 4 objects x 6 calls: {:?}",
            seen.msi_requests.len(),
            seen.msi_requests
        );
        assert!(seen.msi_requests[0].contains("resource=https%3A%2F%2Fstorage.azure.com"));
    }

    // ---- client secret -------------------------------------------------------------------
    set("S3DLIO_AZURE_CREDENTIAL", "client-secret");
    set("AZURE_TENANT_ID", "11111111-2222-3333-4444-555555555555");
    set("AZURE_CLIENT_ID", "app-client-id");
    set("AZURE_CLIENT_SECRET", "the-secret");
    for i in 0..3 {
        exercise(&store, "acctcs", &format!("cs-{i}.bin")).await;
    }
    {
        let seen = https_state.lock().unwrap();
        let aad: Vec<_> = seen
            .aad_requests
            .iter()
            .filter(|f| f.contains("client_secret"))
            .collect();
        assert_eq!(
            aad.len(),
            1,
            "client secret asked for a token {} times: {aad:?}",
            aad.len()
        );
        assert!(aad[0].contains("grant_type=client_credentials"));
        assert!(aad[0].contains("client_id=app-client-id"));
        assert!(aad[0].contains("storage.azure.com"), "{}", aad[0]);
    }

    // ---- auto picks the client secret from the environment; the credential is shared -------
    unset("S3DLIO_AZURE_CREDENTIAL");
    exercise(&store, "acctauto", "auto.bin").await;
    {
        let seen = https_state.lock().unwrap();
        let n = seen
            .aad_requests
            .iter()
            .filter(|f| f.contains("client_secret"))
            .count();
        assert_eq!(
            n, 1,
            "auto reused the client-secret credential; {n} token requests"
        );
    }

    // ---- workload identity (federated token file) ---------------------------------------
    let token_file = dir.path().join("federated-token");
    std::fs::write(&token_file, "federated-jwt").unwrap();
    set("S3DLIO_AZURE_CREDENTIAL", "workload-identity");
    set("AZURE_FEDERATED_TOKEN_FILE", token_file.to_str().unwrap());
    exercise(&store, "acctwi", "wi.bin").await;
    {
        let seen = https_state.lock().unwrap();
        let wi: Vec<_> = seen
            .aad_requests
            .iter()
            .filter(|f| f.contains("client_assertion="))
            .collect();
        assert_eq!(wi.len(), 1, "{wi:?}");
        assert!(wi[0].contains("client_assertion=federated-jwt"));
    }

    // ---- errors name the mode ------------------------------------------------------------
    // 403 from the blob service with a valid token.
    set("S3DLIO_AZURE_CREDENTIAL", "managed-identity");
    let err = store
        .stat(&uri("acctforbidden", "x.bin"))
        .await
        .unwrap_err();
    let message = format!("{err:#}");
    assert!(
        message.contains("Azure credential mode managed-identity") && message.contains("403"),
        "a 403 must name the credential mode: {message}"
    );
    // A missing variable. (A failing token request is covered by a unit test in
    // azure_credential.rs: the credentials above are process-wide and already hold tokens.)
    set("S3DLIO_AZURE_CREDENTIAL", "client-certificate");
    let err = store.stat(&uri("acctcert", "x.bin")).await.unwrap_err();
    let message = format!("{err:#}");
    assert!(
        message.contains("Azure credential mode client-certificate"),
        "{message}"
    );
    // A real certificate (needs the cargo feature `azure-client-certificate` and the openssl CLI).
    #[cfg(feature = "azure-client-certificate")]
    {
        let run = |args: &[&str]| {
            let out = std::process::Command::new("openssl")
                .args(args)
                .output()
                .unwrap();
            assert!(
                out.status.success(),
                "openssl {args:?}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
        };
        let (key, cert, pfx) = (
            dir.path().join("k.pem"),
            dir.path().join("c.pem"),
            dir.path().join("c.pfx"),
        );
        run(&[
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "2",
            "-subj",
            "/CN=s3dlio-test",
            "-keyout",
            key.to_str().unwrap(),
            "-out",
            cert.to_str().unwrap(),
        ]);
        run(&[
            "pkcs12",
            "-export",
            "-inkey",
            key.to_str().unwrap(),
            "-in",
            cert.to_str().unwrap(),
            "-out",
            pfx.to_str().unwrap(),
            "-passout",
            "pass:pfx-password",
        ]);
        set("S3DLIO_AZURE_CREDENTIAL", "client-certificate");
        set("AZURE_TENANT_ID", CERT_TENANT);
        set("AZURE_CLIENT_CERTIFICATE_PATH", pfx.to_str().unwrap());
        // Wrong password: the error names the mode and the password variable.
        set("AZURE_CLIENT_CERTIFICATE_PASSWORD", "wrong");
        let err = store.stat(&uri("acctcert", "x.bin")).await.unwrap_err();
        let message = format!("{err:#}");
        assert!(
            message.contains("Azure credential mode client-certificate")
                && message.contains("AZURE_CLIENT_CERTIFICATE_PASSWORD"),
            "{message}"
        );
        set("AZURE_CLIENT_CERTIFICATE_PASSWORD", "pfx-password");
        exercise(&store, "acctcert", "cert.bin").await;
        let seen = https_state.lock().unwrap();
        let cert_requests = seen
            .aad_requests
            .iter()
            .filter(|f| f.contains("client_assertion_type"))
            .count();
        assert!(cert_requests >= 1);
    }

    // An unknown mode.
    set("S3DLIO_AZURE_CREDENTIAL", "password");
    let err = store.stat(&uri("acctmi", "x.bin")).await.unwrap_err();
    let message = format!("{err:#}");
    for valid in [
        "auto",
        "cli",
        "managed-identity",
        "client-secret",
        "client-certificate",
        "workload-identity",
        "sas",
        "none",
    ] {
        assert!(message.contains(valid), "{valid} missing from: {message}");
    }
    // The az CLI is not installed (empty PATH): the cli mode says so, naming the mode.
    set("S3DLIO_AZURE_CREDENTIAL", "cli");
    let old_path = std::env::var("PATH").unwrap_or_default();
    set("PATH", dir.path().to_str().unwrap());
    let err = store.stat(&uri("acctcli", "x.bin")).await.unwrap_err();
    set("PATH", &old_path);
    let message = format!("{err:#}");
    assert!(message.contains("Azure credential mode cli"), "{message}");

    // ---- SAS over plain http: the query reaches GET, HEAD, PUT, DELETE and LIST ----------
    set(
        "AZURE_STORAGE_ENDPOINT",
        &format!("http://127.0.0.1:{http_port}"),
    );
    set("S3DLIO_AZURE_CREDENTIAL", "sas");
    set("S3DLIO_AZURE_SAS_TOKEN", &format!("?{SAS}"));
    exercise(&store, "acctsas", "sas.bin").await;
    {
        let seen = http_state.lock().unwrap();
        let sas_requests: Vec<_> = seen
            .blob_requests
            .iter()
            .filter(|(_, path, _, _)| path.starts_with("/acctsas/"))
            .collect();
        let methods: std::collections::BTreeSet<_> =
            sas_requests.iter().map(|(m, ..)| m.as_str()).collect();
        for m in ["GET", "HEAD", "PUT", "DELETE"] {
            assert!(
                methods.contains(m),
                "no {m} request reached the SAS account: {methods:?}"
            );
        }
        assert!(
            sas_requests
                .iter()
                .any(|(_, _, q, _)| q.contains("comp=list")),
            "no LIST request: {sas_requests:?}"
        );
        for (method, path, query, auth) in &sas_requests {
            assert!(auth.is_none(), "{method} {path} carried a bearer token");
            assert_eq!(
                query_param(query, "sig").as_deref(),
                Some("abc+def="),
                "{method} {path}?{query} lost the SAS signature"
            );
            assert_eq!(
                query_param(query, "sv").as_deref(),
                Some("2024-11-04"),
                "{query}"
            );
        }
    }
    // SAS mode without a token names the mode and the variable, and never a request.
    unset("S3DLIO_AZURE_SAS_TOKEN");
    let before = http_state.lock().unwrap().blob_requests.len();
    let err = store.stat(&uri("acctsas2", "x.bin")).await.unwrap_err();
    let message = format!("{err:#}");
    assert!(
        message.contains("mode sas") && message.contains("S3DLIO_AZURE_SAS_TOKEN"),
        "{message}"
    );
    assert_eq!(http_state.lock().unwrap().blob_requests.len(), before);

    // ---- a bearer-token mode on an http endpoint is refused, not silently unauthenticated --
    set("S3DLIO_AZURE_CREDENTIAL", "managed-identity");
    let err = store.stat(&uri("acctmi2", "x.bin")).await.unwrap_err();
    assert!(
        format!("{err:#}").contains("https"),
        "a bearer-token mode over http must say it needs https: {err:#}"
    );

    // ---- none: no credential, no SAS (anonymous / Azurite) --------------------------------
    set("S3DLIO_AZURE_CREDENTIAL", "none");
    let err = store.stat(&uri("acctnone", "x.bin")).await.unwrap_err();
    // The fake service answers 403 (no SAS); the point is that a request was made at all and
    // carried no Authorization header.
    let _ = err;
    let seen = http_state.lock().unwrap();
    let none_request = seen
        .blob_requests
        .iter()
        .find(|(_, path, _, _)| path.starts_with("/acctnone/"))
        .expect("none mode sends the request");
    assert!(none_request.3.is_none());
}
