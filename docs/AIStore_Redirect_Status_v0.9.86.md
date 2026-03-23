# AIStore Compatibility Assessment: Highly Likely to Work ✅

> **Version**: v0.9.86  
> **Date**: March 23, 2026  
> **Status**: Implemented — awaiting end-to-end validation against a live AIStore cluster  
> **Source reviewed**: `~/Documents/Code/aistore` (NVIDIA AIStore Go source)

---

## Summary

The v0.9.86 implementation of `RedirectFollowingConnector` is **highly likely to work correctly with NVIDIA AIStore**. The code was validated against the AIStore Go source at the protocol level. No end-to-end test against a live cluster has been performed yet. See the [Known Limitations](#known-limitations--caveats) section for one specific configuration scenario to watch for.

---

## AIStore Redirect Protocol (Confirmed from Source)

These findings come from reading the actual AIStore Go source (`ais/prxs3.go`, `ais/prxred.go`, `core/meta/smap.go`):

### `ais/prxs3.go` — `s3Redirect()` function

- Status is always **HTTP 307** (`http.StatusTemporaryRedirect`)
- The `Location` header is always set before the status write
- Response body is S3-compatible XML: `<Error><Code>TemporaryRedirect</Code><Message>Redirect</Message><Endpoint>host:port/s3</Endpoint><Bucket>bucket</Bucket></Error>`
- If `feat.S3ReverseProxy` is set cluster-wide, AIStore reverse-proxies the request transparently — no 307 is sent; s3dlio gets a 200 directly

### `ais/prxred.go` — `redurl()` / `_preparse()` functions

- The Location URL includes AIStore-specific query parameters appended by the proxy:
  - `ais_pid` (proxy ID) — always present
  - `ais_ts` (Unix timestamp) — always present
  - `ais_smver`, `ais_nonce`, `ais_hmac` — present only on CSK-secured clusters
- The scheme (`http://` vs `https://`) in the Location URL is determined entirely by how the cluster network was configured at startup (`NetInfo.Init(proto, ...)` in `core/meta/smap.go`)
- The target node validates the AIStore query params via `isRedirect(query)` — they must arrive intact

---

## s3dlio Compatibility Analysis

### What works correctly ✅

| Behavior | AIStore | s3dlio | Compatible? |
|----------|---------|--------|-------------|
| Redirect status code | 307 | Handles 301, 302, 303, 307, 308 | ✅ |
| Location header | Always present | Guards against missing Location | ✅ |
| AIStore query params in Location URL | `ais_pid`, `ais_ts`, etc. appended to URL | `set_uri(location)` copies full Location URL verbatim — all query params preserved | ✅ |
| 307 response body (XML) | Sent before final response | SDK never sees 307: connector intercepts below SDK response-parsing layer | ✅ |
| Authorization on target hop | Not expected by target node (uses AIStore params instead) | Stripped on cross-host hops (RFC 9110 §11.6.2) | ✅ |
| Reverse-proxy mode | Transparent proxy, no 307 | Gets 200 directly — no redirect handling required | ✅ |
| All-HTTP cluster | Location is `http://` | Scheme-downgrade check only fires if initial request was HTTPS | ✅ |
| Redirect depth | Single hop (proxy → target) | Default MAX=5, more than sufficient | ✅ |

### Why the SDK layer separation matters

`RedirectFollowingConnector` implements `HttpConnector` in the AWS Smithy runtime stack. This layer operates **below** the SDK's response parsing. When AIStore sends a 307 with an XML body, the connector catches it, extracts the `Location` header, follows the redirect, and returns only the final response to the SDK. The SDK never sees the 307 body and never attempts to parse it as an S3 error. This is the correct interception point.

---

## Known Limitations / Caveats

### HTTPS front-end with HTTP intra-cluster nodes

Some AIStore deployments terminate TLS at a load balancer or reverse proxy in front of the cluster, while intra-cluster target nodes communicate over plain HTTP. In this scenario:

- Client connects to AIStore over `https://`
- Proxy returns a 307 with `Location: http://target-node:9000/...`
- s3dlio's scheme-downgrade protection **refuses** this hop (HTTPS → HTTP)

For typical AIStore deployments (uniformly HTTP or uniformly HTTPS throughout the cluster), this is not an issue. If you encounter this scenario, the workaround is to connect to AIStore over HTTP instead of HTTPS, or ensure your AIStore deployment uses end-to-end TLS.

### Certificate pinning across redirect chain (security gap — pending)

The `CertVerifyStore` struct is in place and the redirect-chain cert-comparison policy is implemented and unit-tested, but the store is currently empty in production. There is no point at which the TLS handshake's peer certificate is recorded into it, because the AWS Smithy SDK does not expose a public injection point for a custom `rustls::client::danger::ServerCertVerifier`.

**Impact**: WebPKI validation (hostname + CA chain) continues to run on every hop. The scheme-downgrade risk (HIGH) is blocked. Cert pinning adds defense in depth for environments with non-standard CA configurations.

**Full analysis**: See [security/HTTPS_Redirect_Security_Issues.md](security/HTTPS_Redirect_Security_Issues.md) §7 for the complete API investigation, dead-end analysis (three specific SDK paths exhausted), viable implementation path via pre-flight TLS probe, and future PR checklist.

---

## Implementation Files

| File | Role |
|------|------|
| `src/redirect_client.rs` | Full implementation: `RedirectFollowingConnector`, `RedirectFollowingHttpClient`, `CertVerifyStore`, `make_redirecting_client()`, 21 unit tests |
| `docs/AIStore_307_Redirect_Proposal.md` | Design rationale, rejected alternatives, protocol analysis |
| `docs/security/HTTPS_Redirect_Security_Issues.md` | Security gap analysis: scheme downgrade (fixed), cert pinning (pending) |
| `docs/api/Environment_Variables.md` | `S3DLIO_FOLLOW_REDIRECTS` and `S3DLIO_REDIRECT_MAX` documented |

---

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `S3DLIO_FOLLOW_REDIRECTS` | `0` (disabled) | Set to `1` to enable redirect following |
| `S3DLIO_REDIRECT_MAX` | `5` | Maximum redirect hops per request |

---

## Test Coverage

21 unit tests in `src/redirect_client.rs` cover:
- Basic 302/307 redirect following
- Cross-host `Authorization` stripping (RFC 9110 §11.6.2)
- Max-redirect loop termination with `ConnectorError`
- Scheme-downgrade refusal (HTTPS → HTTP) — active production protection
- Cert pinning match / mismatch
- RFC-correct 301/302 POST → GET method rewrite
- 303 See Other handling
- Scheme-downgrade regression test that confirmed the production gap and its fix

---

## What Would Confirm Full Compatibility

1. **Live AIStore test** — Run `S3DLIO_FOLLOW_REDIRECTS=1` against an actual AIStore cluster doing S3 GET/PUT/DELETE through the proxy node
2. **CSK-secured cluster test** — Verify that the HMAC query params (`ais_nonce`, `ais_hmac`) are preserved correctly through the redirect hop
3. **HTTPS AIStore test** — Confirm behavior when the cluster is configured with end-to-end TLS (Location URL contains `https://`)
