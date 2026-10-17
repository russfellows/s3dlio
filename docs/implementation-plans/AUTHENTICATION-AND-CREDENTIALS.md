# Authentication and credential management

Date: 2026-10-17

Status: proposal for a separate enhancement round; no implementation approved by this document.

## Goal

Provide a common, per-client authentication configuration for Rust, Python and
the CLI, delegating credential acquisition and refresh to the provider SDKs.
Users should have one consistent object interface without losing the identity
methods supported by S3, private S3 services, native GCS, GCS's S3-compatible
endpoint, or Azure Blob.

This work follows the current caller-buffer release work. It does not expand
that release's acceptance criteria, change the zero-copy ownership contract,
or require an upgrade of the vendored GCS Rapid SDK. Existing GCS and Rapid
support remain in scope for compatibility; modernizing Rapid is separate.

## Current implementation and gaps

These observations describe the source reviewed on 2026-10-17, before this
proposal. They are not claims that the proposed client API already exists.

| Area | Current behavior | Consequence |
| --- | --- | --- |
| S3 | Both client-construction paths in `src/s3_client.rs` require access-key environment variables before loading AWS SDK configuration. | The gate prevents otherwise valid SDK credential sources such as profiles and workload or instance identities. Existing issue #265 covers removing this limitation. |
| Azure | `src/azure_credential.rs` supports automatic selection, workload identity, managed identity, client secrets, developer CLI credentials, SAS, explicit anonymous access, and feature-gated client certificates. | Azure already has substantial production authentication support; it should not be characterized as developer-login-only. Certificate support must be checked against the installed build. |
| Azure automatic selection | A previously successful candidate is cached; if it later fails, the automatic chain can try other candidates. | Convenient discovery can change identity after failure. Explicit production identity selection needs predictable behavior. |
| Native GCS | Storage and control clients in `src/google_gcs_client.rs` delegate authentication to the Google SDK and are process-wide. | Preserve SDK authentication; add explicit client-scoped configuration and verify which advanced methods the pinned SDK supports. |
| Client/cache scope | Azure credentials are cached by mode, service clients by endpoint and mode; Python store cache keys include routing but not an independent identity scope. Configuration commonly comes from process environment. | Two identities using the same endpoint need explicit isolation without mutating global environment variables. |

Azure Shared Key is not supported by the current adapter. It is not a
prerequisite for calling managed/workload identity production-capable. Any
additional authentication mode should have an explicit use case and SDK
support check rather than being promised implicitly.

Authentication support and packaging are separate: the 0.9.128 PyPI wheels
contain S3 and Azure; native GCS requires a GCS-enabled source build. A common
configuration API must report an unavailable compiled backend clearly, not
silently select another backend or identity.

## 1. One client interface, with named connection profiles

Keep four concerns independent:

1. Backend, endpoint, region and addressing configuration.
2. Credential source and identity-selection policy.
3. TLS and network trust configuration.
4. Operational settings such as retries and timeouts.

Rust, Python and CLI entry points should resolve the same configuration model.
Environment defaults remain convenient and compatible. Explicit client
configuration takes precedence; document the full precedence order and validate
conflicting options rather than silently guessing.

For illustration only, a possible Python interface is:

```python
primary = s3dlio.Client(profile="primary-storage")
archive = s3dlio.Client(profile="archive-storage")
await primary.get_into_async(uri, destination, expected_len=size)
```

The two clients may target different providers or the same endpoint with
different identities. Construction must not change process environment.
Connection settings should be immutable after construction, while the
underlying credential provider can refresh its credentials.

Profiles should reference credentials through supported providers, environment
variables, protected files or external providers. Do not encourage secrets in
version-controlled profiles. Exact API names and profile serialization remain
design decisions for the implementation round.

## 2. Preserve each provider's authentication methods

| Backend | Intended methods and configuration |
| --- | --- |
| AWS S3 | SDK default chain, named profiles, SSO, assumed roles, web/workload identity, container and instance identities, explicit temporary or static credentials. Verify each against the pinned SDK and feature set. |
| Private S3 | Explicit endpoint, signing region and addressing style, with scoped static credentials or supported temporary/external providers. Do not assume an AWS identity service exists. |
| Native GCS | Application Default Credentials (ADC), attached workload identity, federation and impersonation where the pinned Rust SDK supports them. |
| GCS through S3 compatibility | Explicit Google HMAC credentials and the interoperability endpoint. This is distinct from native GCS OAuth/ADC authentication. |
| Azure Blob | Managed and workload identity, service principals, developer login, SAS and explicit anonymous access; certificates where built and supported. |

Use the official SDKs for token acquisition, caching and refresh rather than
building another identity framework. Standardizing the caller interface does
not make the underlying authentication protocols interchangeable.

AWS documents its [Rust credential providers](https://docs.aws.amazon.com/sdk-for-rust/latest/dg/credproviders.html)
and [standardized providers](https://docs.aws.amazon.com/sdkref/latest/guide/standardized-credentials.html).
Provider support varies between SDKs and versions. Google documents
[ADC](https://docs.cloud.google.com/docs/authentication/application-default-credentials)
and the separate [GCS interoperability credentials](https://docs.cloud.google.com/storage/docs/interoperability).
Azure documents [data authorization](https://learn.microsoft.com/en-us/azure/storage/common/authorize-data-access?tabs=blobs)
and [Rust SDK authentication](https://learn.microsoft.com/en-us/azure/developer/rust/sdk/authentication/overview).

## 3. Make identity selection predictable

Offer automatic discovery for development and explicit identity selection for
production. Preserve compatible automatic defaults, but make the chosen policy
visible and document its fallback behavior.

An explicitly selected production identity must not silently become a developer
CLI identity when refresh fails. Failure should retain the configured identity
and report the authentication problem. Tests must cover an initially successful
provider failing later, not just startup selection.

Anonymous access must be intentional. Missing credentials are not permission
to downgrade to anonymous access. Separate unsupported credential modes from
temporarily unavailable providers and from remote access denial.

## 4. Refresh, rotation and isolation

- Retain refreshable credential providers, not snapshots of their current
  access tokens. Explicit static credentials remain static unless a documented
  rotation source is configured.
- Coordinate refresh under concurrency using SDK facilities where possible;
  many I/O requests must not trigger hundreds of simultaneous identity calls.
- Bound credential acquisition and refresh time, including behavior during an
  identity-service outage and cancellation. Define how this budget relates to
  an operation's overall deadline.
- Define rotation behavior for files, external processes/providers and SAS.
  Document whether changes are picked up automatically or require an explicit
  reload or client replacement. Do not promise refresh for an unrefreshable SAS.
- Scope caches to connection configuration and provider identity, not merely
  endpoint or authentication mode. Use a stable, non-secret provider identity
  for cache isolation; do not use raw credentials as keys or recreate the cache
  for every refreshed token.
- Keep routine refresh in Rust/the SDK, off the Python GIL and outside the
  per-object Python hot path. An optional extension must not require a Python
  callback for every request.
- Provide a bounded external-provider extension for secret brokers. Prefer
  existing SDK hooks before inventing a new protocol or adding bespoke
  integrations with every secret manager.

Client reuse remains important for performance. Isolation must not turn every
operation into new TLS connections, clients or token requests. Authentication
changes must not introduce payload copies or alter caller-buffer lifetimes.

## 5. Useful diagnostics without leaking secrets

Provide a Rust/Python diagnostic surface and a CLI command explaining the
resolved profile, credential mechanism, endpoint and available refresh status.
Output must be redacted by default. Account, tenant and endpoint identifiers
may themselves be sensitive; make their disclosure deliberate. Never expose
tokens, keys, signed URL queries, SAS values or external-provider output.

Distinguish at least:

- no applicable credential source;
- credential acquisition or refresh failure;
- authenticated access denied by the storage service;
- unsupported method or unavailable compiled backend;
- transport or TLS failure.

Expose only what SDKs reliably make observable. An unknown expiry or source
detail should remain unknown rather than being inferred misleadingly.

Authorization remains the remote service's IAM policy. A connection check must
not require listing buckets or enumerating a namespace: a client can be allowed
to read or write a prefix while lacking enumeration permission. Default
diagnostics should inspect configuration without storage I/O. Any live probe
must name its operation and require explicit opt-in; write probes need a
caller-supplied disposable target and cleanup policy.

Private endpoints need custom CA support and an explicit redirect trust policy.
Do not forward credentials to arbitrary redirect destinations. Errors and
diagnostics must sanitize SDK and transport messages, not just s3dlio's own
formatted fields.

## 6. Delivery order and acceptance

Finish the current release, then take a focused authentication round:

1. Implement the AWS SDK credential chain under existing issue #265, preserving
   environment-key compatibility and endpoint/region/TLS behavior.
2. Introduce per-client configuration and identity-scoped cache isolation.
3. Define and verify explicit identity selection, refresh, rotation and bounded
   external providers across the supported SDKs.
4. Add safe diagnostics and complete the cross-provider test matrix. Design
   error categories and redaction alongside the preceding work, not afterward.

Required checks include two different identities on the same endpoint, token
expiry under concurrency, rotation, identity-service outages, cancellation,
refresh deadlines, no production identity fallback, explicit anonymous access,
TLS/redirect trust, and secret redaction. Verify connection reuse and that the
Python GIL and payload-copy behavior do not regress.

Use deterministic provider doubles for expiry, concurrency and failure tests,
then opt-in live checks for the actual supported providers. Check AWS/private
S3, native GCS, GCS S3 compatibility and Azure separately. Record the pinned SDK
and build-feature support matrix; unsupported advanced methods must fail
clearly. Existing Rapid behavior gets compatibility coverage without upgrading
its vendored dependency in this round.

## Tracking

Existing prerequisite: [#265 — standard AWS credential chain](https://github.com/russfellows/s3dlio/issues/265).

New P3 enhancements:

- [#273 — per-client connection profiles and identity-scoped cache isolation](https://github.com/russfellows/s3dlio/issues/273): section 1, cache isolation in section 4, and connection trust in section 5.
- [#274 — predictable identity selection, refresh and credential rotation](https://github.com/russfellows/s3dlio/issues/274): sections 2–4, coordinated with #265 and #273.
- [#275 — redacted diagnostics and cross-provider validation](https://github.com/russfellows/s3dlio/issues/275): sections 5–6, with error categories and redaction developed alongside the client and lifecycle work.
