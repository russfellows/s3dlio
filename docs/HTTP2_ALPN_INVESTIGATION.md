# HTTP/2 Support — Implementation Notes and Test Server

**Last Updated**: 2026-04-18  
**Branch**: `feature/http2-v0.9.90`  
**Status**: Complete — all modes implemented, tested, and verified end-to-end

---

## Summary

s3dlio supports HTTP/2 on both cleartext (`http://`) and TLS (`https://`) endpoints.
The behaviour is automatic by default and can be controlled with the `S3DLIO_H2C`
environment variable.

---

## How HTTP/2 Works in s3dlio

### TLS endpoints (`https://`)

HTTP/2 is negotiated automatically via **TLS ALPN**.  The reqwest client (backed by
rustls + aws-lc-rs) advertises `["h2", "http/1.1"]` in every TLS handshake.  If the
server selects `h2`, the connection is HTTP/2; if it selects `http/1.1`, it is HTTP/1.1.
No configuration is needed.  `S3DLIO_H2C` has no effect on TLS connections.

**TLS dependency chain:**
- `reqwest 0.13.2` — `http2`, `rustls`, `__rustls-aws-lc-rs` features active
- `hyper 1.8.1` — `http1`, `http2` features active
- `h2 0.4.13` — present in the dependency tree
- `rustls 0.23` — uses `aws-lc-rs` crypto provider
- ALPN is set in `reqwest`'s `TlsBackend::Rustls` branch via `HttpVersionPref::All` (the default)

### Cleartext endpoints (`http://`)

Plain HTTP does not have TLS, so ALPN is unavailable.  HTTP/2 over cleartext uses the
**h2c prior-knowledge** mechanism (RFC 7540 §3.4): the client sends the HTTP/2 connection
preface immediately without an HTTP/1.1 upgrade handshake.

s3dlio supports three modes controlled by `S3DLIO_H2C`:

| `S3DLIO_H2C` | Mode | Behaviour |
|---|---|---|
| *(not set)* | **Auto** | Probe h2c on the first `http://` connection; fall back to HTTP/1.1 if rejected. All subsequent connections use the probed result. |
| `1` (or `true`, `yes`, `on`, `enable`) | **ForceH2c** | Always use h2c prior-knowledge; never fall back. Use this for systems that require HTTP/2 cleartext. |
| `0` (or `false`, `no`, `off`, `disable`) | **ForceHttp1** | Always use HTTP/1.1; skip the auto-probe entirely. |

The h2c probe fires exactly **once per process**.  A failed probe (the server returns
anything other than a valid HTTP/2 frame — e.g. an HTTP 400) is caught, the request is
retried transparently on the HTTP/1.1 client, and all future connections skip the probe.

---

## Client Selection Routing

The routing logic lives in `src/reqwest_client.rs::select_client()` and is fully
unit-tested.  The decision matrix is:

| Scheme | Mode | auto_state | Client used | is_auto_probe |
|--------|------|-----------|-------------|---------------|
| `http://` | ForceH2c | any | **h2c** | false |
| `https://` | ForceH2c | any | http1 (ALPN) | false |
| `http://` | ForceHttp1 | any | http1 | false |
| `https://` | ForceHttp1 | any | http1 | false |
| `http://` | Auto | UNKNOWN | **h2c** | **true** |
| `http://` | Auto | OK | **h2c** | false |
| `http://` | Auto | FAILED | http1 | false |
| `https://` | Auto | any | http1 (ALPN) | false |

**Key rule**: `h2c_client` (built with `.http2_prior_knowledge()`) is **never** used for
`https://` URLs.  Sending an h2c preface before TLS causes "broken pipe" errors.
For TLS endpoints HTTP/2 is negotiated by the TLS layer, and the `http1_client` handles it.

---

## ALPN Investigation Findings

During development, HTTP/2 was not being negotiated on a local MinIO instance even though
the client stack appeared correct.  The investigation found:

- A **tcpdump packet capture** confirmed reqwest IS advertising ALPN `["h2", "http/1.1"]`
  in the TLS ClientHello (bytes confirmed on the wire).
- **curl** confirmed the root cause: `* ALPN: server accepted http/1.1` — the MinIO
  instance does not accept h2 and always selects `http/1.1`.
- The earlier `openssl s_client -alpn h2` result that appeared to show h2 was misleading:
  `openssl s_client` echoes back the `-alpn` flag value when the server does not override it.
  This does **not** mean the server will negotiate h2 for real requests.

**Conclusion**: The s3dlio client code is correct.  HTTP/2 negotiation depends entirely on
whether the server accepts h2 in its ALPN response.  On systems whose S3 API supports h2,
s3dlio will automatically use HTTP/2 with no configuration changes needed.

---

## Test Server (`examples/tls_test_server.rs`)

To verify the full ALPN → HTTP/2 path without requiring a specific production server, a
minimal TLS + HTTP/2 test server is included at `examples/tls_test_server.rs`.

### What it does

- Generates a fresh **self-signed certificate** (via `rcgen`) for `127.0.0.1` / `localhost`
  at startup; writes the PEM to `/tmp/tls_test_server.crt`
- Builds a `rustls` `ServerConfig` with `alpn_protocols = ["h2", "http/1.1"]`
- Listens on `https://127.0.0.1:9443`
- Logs the negotiated ALPN protocol and HTTP version for every connection and request
- Returns minimal S3-compatible responses (200 with `etag`, `content-length`,
  `last-modified`, `x-amz-request-id`) — enough for HeadObject and PutObject

### Running the test server

```bash
# Build and start the server
cargo run --example tls_test_server

# Server output:
# Certificate written: /tmp/tls_test_server.crt
# Listening on https://127.0.0.1:9443  (Ctrl-C to stop)
```

### Test with curl

```bash
# HTTP/2 (ALPN negotiation)
curl --http2 --cacert /tmp/tls_test_server.crt -sv https://127.0.0.1:9443/bucket/key

# Expected curl output includes:
# * ALPN: curl offers h2,http/1.1
# * ALPN: server accepted h2
# * using HTTP/2

# Force HTTP/1.1 (to confirm server handles both)
curl --http1.1 --cacert /tmp/tls_test_server.crt -sv https://127.0.0.1:9443/bucket/key
```

### Test with s3-cli (verifies the full stack)

```bash
export AWS_CA_BUNDLE=/tmp/tls_test_server.crt
export AWS_ENDPOINT_URL=https://127.0.0.1:9443
export RUST_LOG=info

# stat — should show "ALPN negotiated = h2" in server output and
#         "HTTP protocol (first response): HTTP/2.0" in client output
cargo run --bin s3-cli -- -v stat s3://bucket/key

# put — should show "protocol=HTTP/2" in the PUT summary line
cargo run --bin s3-cli -- -v put s3://bucket/prefix -n 1 -s 4096
```

### Expected output

**Server side (stderr):**
```
[server] 127.0.0.1:XXXXX: ALPN negotiated = "h2"
[server] HEAD /bucket/key HTTP/2
```

**Client side:**
```
INFO AWS_CA_BUNDLE set — loading CA bundle from: /tmp/tls_test_server.crt
INFO HTTP version mode: auto (https:// → HTTP/2 via TLS ALPN; ...)
INFO HTTP protocol (first response): HTTP/2.0
...
PUT summary: attempted=1, succeeded=1, failed=0, protocol=HTTP/2
```

### Test h2c (cleartext HTTP/2, for plain http:// endpoints)

A dedicated h2c test server is provided at `examples/h2c_test_server.rs`.  It listens on
`http://127.0.0.1:9080` with no TLS and accepts both h2c prior-knowledge and HTTP/1.1
connections.  `hyper_util::server::conn::auto::Builder` detects the HTTP/2 connection
preface (`PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n`) automatically — no ALPN involved.

```bash
# Terminal 1 — start the h2c server
cargo run --example h2c_test_server
# Output: Listening on http://127.0.0.1:9080  (h2c prior-knowledge + HTTP/1.1, no TLS)
```

```bash
# Terminal 2 — test with curl
# HTTP/2 via h2c prior-knowledge:
curl --http2-prior-knowledge -sv http://127.0.0.1:9080/bucket/key
# Server logs: [server] HEAD /bucket/key HTTP/2

# Force HTTP/1.1 to confirm both modes work:
curl --http1.1 -sv http://127.0.0.1:9080/bucket/key
# Server logs: [server] HEAD /bucket/key HTTP/1.1
```

```bash
# Terminal 2 — test the full stack with s3-cli
export AWS_ENDPOINT_URL=http://127.0.0.1:9080

# Forced h2c (S3DLIO_H2C=1):
S3DLIO_H2C=1 cargo run --bin s3-cli -- -v stat s3://bucket/key
# Expected client output: "HTTP version mode: FORCED HTTP/2 (S3DLIO_H2C=1)"
#                         "HTTP protocol (first response): HTTP/2.0"
# Expected server log:    "[server] HEAD /bucket/key HTTP/2"

# Auto-probe (S3DLIO_H2C unset) — probe fires once, succeeds, all requests use h2c:
cargo run --bin s3-cli -- -v stat s3://bucket/key
# Expected client output: "HTTP version mode: auto (http:// → h2c prior-knowledge probe ...)"
#                         "HTTP protocol (first response): HTTP/2.0"

# Forced HTTP/1.1 (S3DLIO_H2C=0) — skips probe entirely:
S3DLIO_H2C=0 cargo run --bin s3-cli -- -v stat s3://bucket/key
# Expected server log: "[server] HEAD /bucket/key HTTP/1.1"

# PUT write path:
S3DLIO_H2C=1 cargo run --bin s3-cli -- -v put s3://bucket/prefix -n 1 -s 4096
# Expected: "PUT summary: ... protocol=HTTP/2"
```

### Verified results (2026-04-18, s3dlio v0.9.90)

Two independent end-to-end test runs against `h2c_test_server` on `http://127.0.0.1:9080`
— first with `AWS_CA_BUNDLE` set to a local cert, second with `AWS_CA_BUNDLE` unset
(system default TLS trust store).  Both runs produced identical results, confirming that
CA bundle configuration has no bearing on plain `http://` connections.

| `S3DLIO_H2C` | Mode | Key log line | Protocol confirmed |
|---|---|---|---|
| `1` | Forced h2c | `HTTP version mode: FORCED HTTP/2 (S3DLIO_H2C=1)` → `HTTP protocol (first response): HTTP/2.0` | ✅ HTTP/2 |
| *(unset)* | Auto-probe | `h2c auto-probe succeeded — HTTP/2 cleartext active` → `HTTP protocol (first response): HTTP/2.0` | ✅ HTTP/2 |
| `0` | Forced HTTP/1.1 | `HTTP version mode: forced HTTP/1.1 (S3DLIO_H2C=0)` → `HTTP protocol (first response): HTTP/1.1` | ✅ HTTP/1.1 |

The auto-probe result is the most significant: with no configuration, s3dlio detected
that the server speaks h2c and promoted the connection to HTTP/2 automatically.

---

## Unit Tests

The routing logic in `select_client()` is comprehensively unit-tested in
`src/reqwest_client.rs` (see the `test_select_client_*` test functions).  These tests
cover all combinations of scheme × mode × auto_state without requiring network access:

- `test_select_client_force_h2c_plain_http` — ForceH2c + http:// → H2c
- `test_select_client_force_h2c_tls` — **ForceH2c + https:// → Http1** (critical: prevents broken pipe)
- `test_select_client_force_http1_plain_http` — ForceHttp1 + http:// → Http1
- `test_select_client_force_http1_tls` — ForceHttp1 + https:// → Http1
- `test_select_client_auto_plain_http_first_connection` — Auto + http:// + UNKNOWN → H2c + is_probe=true
- `test_select_client_auto_plain_http_probe_succeeded` — Auto + http:// + OK → H2c
- `test_select_client_auto_plain_http_probe_failed` — Auto + http:// + FAILED → Http1
- `test_select_client_auto_tls_*` — Auto + https:// (all states) → Http1
