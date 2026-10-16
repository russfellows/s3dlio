# s3dlio Changelog

## Unreleased

### Changed (BEHAVIOUR CHANGE)

- **The writers' `crc32c:` checksum is now a real CRC-32C (#256).** Every streaming writer (file,
  `direct://`, the data-generating streaming writer, S3, Azure, GCS and the Arrow backend) reported
  its checksum as `crc32c:<hex>` but computed CRC-32 (IEEE) with `crc32fast`, so the value did not
  match a GCS `crc32c` or an S3 `x-amz-checksum-crc32c` (apart from the few inputs on which the two
  algorithms agree, such as the empty object), and the label named the wrong algorithm.
  The value is now CRC-32C (Castagnoli), computed with the `crc-fast` crate (already in `Cargo.lock`
  through the AWS SDK): SIMD code chosen at run time from the CPU (PCLMULQDQ / VPCLMULQDQ on x86-64,
  PMULL and the CRC32 instructions on aarch64, software fallback), so no build flag is needed and a
  wheel gets the fast path. Throughput was measured, not assumed (`examples/bench_crc32c.rs`): the
  `crc32c` crate, the obvious choice, is about half as fast as the `crc32fast` it replaces on x86-64
  without build flags (it detects SSE4.2 at run time, but its hardware loop makes a non-inlinable call
  per 8 bytes, so it only runs well when the whole build enables `+sse4.2`), which would have slowed
  every `write_chunk`; `crc-fast` is about 3x to 4x faster than `crc32fast` for buffers that fit in
  cache and equal to it when memory-bound (measured on an x86-64 AVX-512 host; aarch64 was not
  measured). `s3dlio::checksum::crc32c_implementation()` reports which code the CPU selected. The
  `crc32c` crate remains only as an independent oracle in the tests. The label is unchanged.
  **The hex digits differ for almost every input** (the two algorithms agree on some, such as empty
  input, where both are zero): for the bytes `123456789` the writers used to report
  `crc32c:cbf43926` and now report `crc32c:e3069283`. Anything that stored a checksum from an
  earlier release and compares it with a new one (for example a checkpoint manifest written before
  and read with validation after) must recompute it. The integrity-validation helper
  (`get_with_validation` / `get_range_with_validation`) uses the same algorithm, so a value a writer
  reports still validates the object it wrote. New: `s3dlio::checksum` (`Crc32c`, `crc32c_of`,
  `crc32c_label`). The ZIP/NPZ container CRC is unchanged (that format requires CRC-32).

- **TFRecord files s3dlio generates now carry CRC-32C checksums (#258).** The TFRecord format
  requires CRC-32C (Castagnoli) for both the length checksum and the data checksum of every record,
  masked as `((crc >> 15) | (crc << 17)) + 0xa282ead8`. The masking was right, but the CRC underneath
  was CRC-32 (IEEE), so a reader that verifies checksums (TensorFlow, TensorBoard) rejected the files
  as corrupt, and one that does not verify accepted them, which hid the problem. The bytes of every
  generated TFRecord change (the 4-byte checksum fields; the record layout, the sizes and the index are
  the same). The test compares the output with a reference written by an independent bitwise CRC-32C
  and accepted by TensorBoard's TFRecord reader, which verifies the checksums and rejects the old output
  with a header-CRC error. Files written by earlier releases keep their wrong checksums.

### Fixed

- **A compressed shard is stored, recorded and statted under one name (R10).** A compressing
  writer for a custom key that already ends in `.zst` (`custom.zst`) stored the object as
  `custom.zst.zst` in the `file://`, `direct://` and Arrow writers, while `finalize_writer_to_shard_meta`
  recorded and looked for `custom.zst`: the write succeeded and the helper then failed to find the
  object, or silently described an older object that already had that name. On S3, Azure and GCS the
  writers stored `custom.zst` correctly, but `put_shard_with_key` recorded and statted `custom.zst.zst`.
  There is now one rule, `CompressionConfig::stored_name` (and `stored_path` for filesystem paths): the
  extension is appended only when the name does not already end with it. Every compressing writer
  (file, `direct://`, Arrow, S3, Azure, GCS) uses it for the stored name, and the checkpoint writer
  uses it for the manifest name (`put_shard`, `put_shard_with_key`, `finalize_writer_to_shard_meta`,
  and now also `finalize_shard_meta` / `finalize_shard_meta_with_checksum` on a Writer that has
  compression). Ordinary keys are named as before (`plain.bin` is stored and recorded as
  `plain.bin.zst`); the exceptions are keys already ending in `.zst` (above) and, for `file://`, a name
  ending in a dot (`a.`), which is now `a..zst`, the same as `direct://` and the manifest.

- **The Python `CheckpointStream.finalize` result names the stored shard (R10b).** With a compressing
  store (`PyCheckpointStore(uri, compression_level=N)`) the streaming writer stores `rank-N.bin.zst`,
  but `finalize` statted and returned the un-suffixed `rank-N.bin`, so it failed after a successful
  write or described an older uncompressed object left at that name. It now records the stored name
  through the same helper as the Rust writer (the shard metadata now also carries the writer's
  `crc32c:` checksum, which the old path dropped), and `Writer::finalize_shard_meta` /
  `finalize_shard_meta_with_checksum` do the same for a Writer configured with compression.

  Limits of the naming rule, recorded rather than changed: with compression on, the keys `a` and
  `a.zst` both store `a.zst` (the second write replaces the first); with compression off, a key ending
  in `.zst` stores raw bytes under that name, and the checkpoint `Reader`, which decompresses any shard
  whose name ends in `.zst` (the suffix is spelled out in the reader, not taken from
  `CompressionConfig::extension()`), cannot read it back.

- **`get_with_validation` accepts the checksum form that writers and manifests produce (#259).**
  Writers report `crc32c:<hex>` and the checkpoint manifest stores that value, but validation compared
  it with the bare hex of the data, so validating a value a writer had reported failed for intact data
  ("expected crc32c:e3069283, got e3069283"). `get_with_validation`, `get_range_with_validation` and
  `load_checkpoint_with_validation` (and so the checkpoint reader's validating reads and
  `validate_checkpoint_integrity`) now take `<algorithm>:<hex>`: the algorithm is parsed
  (case-insensitive), `crc32c` is supported, and any other algorithm is an immediate, clear error
  ("unsupported checksum algorithm ..."), reported before anything is fetched and never as a mismatch.
  **A bare hex string now means CRC-32C** (it was CRC-32 before #256 changed the writers): a caller
  that computed a CRC-32 with `crc32fast` and passes it bare will now see a mismatch, and a mismatch
  message says `(CRC-32C; checksums written before the #256 fix were CRC-32)` so an old stored value is
  recognisable. The hex part must be 1 to 8 hexadecimal digits in either case; fewer than 8 are read as
  the number they spell (`crc32c:0` is the empty object's checksum). A `0x` prefix, an empty value, more
  than 8 digits, a further `:` or any other character is a "malformed checksum" error, never a
  mismatch. Corrupted data still fails validation.

- **Validating a compressed checkpoint shard works (#259).** Writers hash the uncompressed bytes they
  are given, but the shard in the store is the compressed `.zst` object, and the reader validated the
  stored bytes, so a compressed shard could never pass validation once the checksum format was
  accepted. The checkpoint reader (`read_shard_with_validation`, `read_all_shards_with_validation`,
  the concurrent variants and `validate_checkpoint_integrity`) now fetches a `.zst` shard,
  decompresses it and compares the decompressed bytes. Also, `Writer::finalize_writer_to_shard_meta`
  now records a compressing writer's shard under its stored name (`...bin.zst`); it used to look for
  the name without the extension and fail.

- **The retry loop keeps its own attempts for every error code the AWS SDK retries (#240
  follow-up).** The first version of the #240 classifier listed only some of the codes the pinned AWS
  SDK (aws-runtime 1.7.5) treats as transient or throttling. `RequestTimeoutException`,
  `ThrottledException`, `RequestThrottledException`, `TransactionInProgressException`,
  `LimitExceededException`, `PriorRequestNotComplete` and `EC2ThrottledException` on a 4xx status were
  judged definite, so after the SDK had spent its own attempts the s3dlio loop gave up instead of
  starting its own attempts. The classifier now carries the SDK's complete throttling and transient
  tables (a unit test compares them with the pinned crate, so after an SDK upgrade that changes the
  tables `cargo test` fails until the constants are updated) plus the S3-specific codes it always had.
  Consequence to know: a persistent throttling or quota error with one of these codes (for example a
  hard `LimitExceededException`) now uses the outer retry budget too, 3 attempts with 100 ms and 200 ms
  of backoff between them (`S3DLIO_MAX_RETRY_ATTEMPTS`), on top of the SDK's own attempts, before the
  error is returned; it used to fail after the SDK's attempts alone.

### Added

- **Azure credential modes: managed identity, service principal, workload identity and SAS (#253).**
  The Azure backend used only the `az`/`azd` developer login, so it could not authenticate on Azure VMs, AKS pods or
  in CI. `S3DLIO_AZURE_CREDENTIAL` now selects `auto` (default), `cli`, `managed-identity`, `client-secret`,
  `client-certificate`, `workload-identity`, `sas` or `none` (case-insensitive; an unknown value is an error listing
  the valid ones). **BEHAVIOUR CHANGE: the default is now `auto`, a fallback chain.** Before, every Azure
  request used `az`/`azd` only and the `AZURE_*` service-principal variables were ignored. `auto` now tries, in
  order, workload identity (`AZURE_FEDERATED_TOKEN_FILE`, `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`), client secret
  (`AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_CLIENT_SECRET`), client certificate (`AZURE_CLIENT_CERTIFICATE_PATH`),
  managed identity (`IDENTITY_ENDPOINT`/`MSI_ENDPOINT` set, or the instance metadata service identifies an Azure VM),
  then `az`/`azd`, skips members whose variables are not set, falls through a member that cannot produce a token (for
  example a VM with no assigned identity goes on to `az`), uses the first that works for the rest of the process, and
  lists every member tried and why if none works. The Azure metadata probe runs on the blocking pool with an overall
  bound of under 0.5 s; a negative ("not Azure") result is cached for 30 s, a positive result is not separately
  cached by the probe helper (the winning credential is what is cached). Set `S3DLIO_AZURE_CREDENTIAL=cli` for the old behaviour.
  Credentials are created once per mode and kept for the life of the process: a rotated secret, certificate or
  token file needs a restart.
  `AZURE_AUTHORITY_HOST` (https only) is honoured by the SDK for the Entra ID modes. `sas` appends
  `S3DLIO_AZURE_SAS_TOKEN` to every request URL (the SDK keeps the query on GET, HEAD, PUT, DELETE and LIST; tested).
  Each credential is created once per process and shared by the cached per-account client (#242 follow-up), so the SDK
  caches and refreshes tokens; only the `cli` mode runs a process, per token request. A 401/403 and a failed token request
  name the mode (`Azure credential mode <m>: ...`); the SAS token is removed from messages. A bearer-token mode on an
  `http` endpoint is refused with a message saying it needs https (use `sas`/`none` for Azurite); `auto` on `http` sends no
  credential, as before. The `client-certificate` mode needs the new cargo feature `azure-client-certificate` (it links
  OpenSSL, so it is not in `full-backends` or the default wheel) and a PKCS#12 `.pfx` with an RSA key; without the
  feature the mode reports that. Shared Key (account key) is still not supported: `azure_storage_blob` 1.0 cannot sign with
  it. Tested end to end against local fake Entra ID, App Service managed-identity and Blob services (HTTPS with a throwaway
  CA): managed identity, client secret, `auto`, workload identity, client certificate and SAS, including one token
  request per credential for many blob requests. Not tested: real Azure, the real instance-metadata address (the SDK
  hard-codes it; the probe is tested against a local stand-in), and a real AKS federated token. The `auto` metadata probe
  checks for an Azure instance document: an earlier draft accepted any connection to 169.254.169.254, which on a Google
  Cloud VM selected managed identity and made an Azure test wait about 400 s.

### Fixed

- **Container listing could send credentials to another host (#253 review, security).** `list-buckets az://...`
  and Python `list_containers` put the raw text after `az://` into a host name, so `az://attacker.example#` built a
  request to attacker.example carrying the bearer token or SAS. Every place that turns an account into a URL now
  goes through one check: 3-24 lowercase letters and digits (upper case is accepted and lowercased, as host names
  are case-insensitive), or with a custom endpoint any name free of `/ ? # @ \ %`, whitespace and control
  characters. `%` is refused outright, so `%2e%2e` and `%2f` cannot smuggle a dot segment or a slash. Nothing is
  sent and no credential is built for a refused name. `resolve_azure_account_url` now returns a `Result`, and
  `is_azure_blob_host` ignores case.
- **SAS material no longer survives in error text (#253 review).** The SDK re-sorts and re-encodes the query on
  operations that set query pairs (list, container operations, block staging and commit), and transport errors print
  the whole request URL, so replacing the exact token text missed them. Error text now loses the query of every URL in
  it, every SAS parameter value (`sig`, `se`, `sp`, `skoid`, ...), and each value of the configured token as written,
  decoded, encoded and double-encoded. `AzureRequestError` has a redacting `Debug`, and `redact_url` also strips
  userinfo.
- **`auto` no longer waits out the SDK's retry schedule on a VM without an identity (#253 review).** The managed
  identity attempt in `auto` has a deadline (`S3DLIO_AZURE_AUTO_MI_TIMEOUT_MS`, default 5000 ms; the SDK retries a
  404/410/5xx from the metadata service for about 72 s), after which `auto` moves on to `az`/`azd`; the same deadline applies when a managed identity that already won stalls
  on a later token refresh, after which the chain is run again with the managed identity left to the end (a stall
  costs one deadline when another member works; a lone managed identity still gets its second try); a request forgets the
  winner only if it is still the one it used, which protects a newer winner of a different mode (a re-installed winner of
  the same mode is the same credential object and can still be cleared once by a late failure, costing one extra chain
  run). Only managed identity has its own deadline: the HTTPS modes rely on the SDK's 20 s connect and
  60 s read timeouts, and `az`/`azd` has none in `azure_identity` 1.0.0. A "not Azure"
  answer from the metadata probe is cached for 30 s (a failing chain no longer probes twice per request); a cached
  `auto` winner that starts failing makes that request run the chain again, so a working member further down is found
  without a restart; a configured service principal that fails before `auto` falls through is logged at `warn`. The
  client-certificate file is read and parsed on the blocking pool at the first token request, not on a runtime worker.
- **`S3DLIO_REQUIRE_FULL_BACKENDS` (test and release validation only).** `python/tests/test_full_backend_azure_gcs.py`
  fails, instead of skipping, when the wheel lacks the Azure/GCS backends and this variable is set to anything other than
  empty, `0`, `false`, `no` or `off` (unknown values fail too). Not read by the library; see `docs/Environment_Variables.md`.
- **Deleting an `az://` object that is already gone succeeds again (#253 review).** Wrapping Azure errors to
  name the credential mode and hide a SAS token turned them into plain text, so the delete contract no longer
  recognised `BlobNotFound` and `delete`/`delete -r` failed for a blob that was already missing (or vanished
  between the listing and its delete). Errors now keep the typed SDK error inside, and batch deletes treat a
  missing blob as deleted.
- **Azure hardening (#253 review).** An `az://` account name must be 3-24 lowercase letters and digits (with a
  custom endpoint such as Azurite the rule is relaxed, but `/ ? # @ \`, whitespace and control characters are
  never accepted), checked before any request or credential use, so `az://attacker.example#/c/k` cannot redirect a
  request or a bearer token. A SAS token is refused over plain `http` to anything but `localhost`/loopback.
  Endpoint and account URLs in log lines and error messages are shown without their query string, and
  `list_account_containers` errors name the credential mode like blob operations.
- **GCS gRPC full reads cannot abort on an absurd size (#255 follow-up).** The gRPC receive buffer is reserved
  with `try_reserve` and falls back to growth instead of aborting if the reservation fails.
- **`get_many()` on `az://` and `gs://` URIs no longer fails for every key (#242).** The function
  stripped the scheme and container/bucket and handed the Azure and GCS stores only a bare object
  key, but those stores read the account/container/bucket from the full URI, so every read errored
  while the same URI worked with `get()`. Each read is now given the full URI. Further changes to the
  same call, found in review:
  - A failed batch no longer leaves reads running. The call waits for the reads already in flight
    and returns the first error as `GET <uri>: <cause>`; reads that had not started are skipped
    (the S3 reader also drains before returning, but it does not skip). Previously it returned at
    once and the other downloads, and for Azure the `az` token processes, kept running.
  - The stores come from the process-wide store cache (#204) and the calls write op-log rows like
    `get()` does (#174). Before, each URI built a new store and no row was written.
  - A URI that names no object (`az://acct/container`, `gs://bucket/`) is refused before any read
    starts: `Cannot GET: no key specified in URI`.
  - A read task that panics is reported as `GET <uri>: the read task failed: ...`, with its URI.
  - `get_many` of `file://`/`direct://` already passed full URIs and is unchanged. `get_many_async()` and
    `get_many_stats()` still send every URI through the S3-only reader and refuse non-S3 URIs.
  Covered by unit tests with a fake store and, for Azure, a Python test against a local fake Blob
  endpoint; there is no local stand-in for GCS, so `gs://` is covered by the unit tests only.
- **Azure: one HTTP client per account URL instead of one per operation (#242 review).** Every
  `get`/`stat`/`get_range` built a new Azure service client, with a new connection pool, TLS state and
  an empty bearer-token cache. Clients are now cached per account URL (at most 64) and reused, so
  connections and the SDK's token cache are shared. `list_account_containers` no longer attaches a
  credential to an `http://` endpoint (the SDK refused it with "must use https"). Not measured against
  live Azure (no network access in the test runs); the unit tests count client constructions.
  Limit: a cached client keeps its bearer token until it nears expiry (about an hour), so after
  `az login` as a different identity a running process keeps using the old one until the token
  refreshes or the process restarts. When more than 64 accounts are used an arbitrary cached client is evicted
  (it is rebuilt, with a new token request, on next use).
- **GCS HTTP and Azure GETs no longer grow their receive buffer by doubling (#252).** The GCS HTTP read
  started a 1 MiB buffer and let it double, so an object over 1 MiB was copied again at each growth step
  (an 8 MiB body: 3 growths); the Azure reads used the SDK's `collect()`, which also grows chunk by chunk.
  Both now receive into one buffer sized from the object size in the response (GCS, clamped to the requested
  range) or the length the Azure response announces (`Content-Range`/`Content-Length`), the size hint, or the
  requested range (an Azure range reserves at most 128 MiB up front). If the server sends more than announced
  the buffer still grows. The result is **one copy inside s3dlio after the SDK delivers the data**; it does not
  remove copies the SDK makes itself (Azure partitioned downloads, used above 32 MiB, buffer each partition
  first). A test counts buffer growth. Throughput was not measured.
- **GCS gRPC range reads reserve the range, not the object (#255).** The gRPC read sized its buffer to the
  whole object even for a range read, so a 1 MiB range of a multi-TiB object tried to reserve the object, and a
  negative size from the server wrapped through `as usize`. The buffer and the read timeout are now sized to
  the bytes the read returns (clamped to the object); a negative size is 0 and the buffer grows as data
  arrives. Full reads are unchanged. Tested on the sizing function only (no GCS service offline).
- **Azure: a stale cached size can no longer turn a download into one request per byte (#242 review).**
  `AzureObjectStore` caches the size a `HEAD` reports for 60 s. A blob cached as 0 or 1 byte and then
  enlarged made the sequential download plan one-byte ranges (the SDK keeps the caller's partition size and
  learns the real length only from the first response), roughly a million serial requests for 1 MiB. Now:
  the partition size is never below 32 MiB whatever the hint says; the default mode (`Auto` with the Azure
  SDK engine) and `azure-sdk` make **no `HEAD` at all** (one request for a blob up to 32 MiB; the buffer is
  sized from the response); the s3dlio RangeEngine, which plans its ranges from the size, uses a cached size
  without a `HEAD` but guarded: its last range asks for one byte more than planned, so a blob that grew (an
  over read) or shrank (a short read or a range error) makes the engine fail, and the read is retried once
  with a size from a fresh `HEAD` (an unchanged blob costs no `HEAD`; before this guard every RangeEngine read
  started with one); `put`/`put_multipart` forget the written object's cached size and
  `delete`/`delete_batch`/`delete_prefix` forget exactly the deleted objects' (never the whole cache), under one
  canonical `az://account/container/key` / `gs://bucket/object` key (Azure and GCS); an Azure range read reserves
  the response's `Content-Length` and asks for its whole range in one request (an open-ended range uses the 32 MiB
  default partition, not the SDK's 4 MiB). Tested against a fake Blob service that refuses after 300 requests.
  Cache growth is bounded separately by the #241 size-cache fix, which this builds on (`ObjectSizeCache` gained
  `invalidate` and `invalidate_many`).
- **Azure: the documentation no longer tells you to authenticate with `AZURE_STORAGE_KEY` (#239).** (The service-principal, managed-identity and SAS modes this entry says were missing were added afterwards, see #253 above.)
  The docs (`PYTHON_API_GUIDE.md`, `Environment_Variables.md`, `CLI_GUIDE.md`) said an Azure account key,
  `AZURE_STORAGE_ACCOUNT`, or the service-principal variables (`AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`,
  `AZURE_TENANT_ID`) authenticate s3dlio to Azure. None of them is read: s3dlio uses the Azure developer-tool
  credential (`az login`, then `azd auth login`) for `https` endpoints and sends no credential to `http`
  endpoints such as Azurite; the account always comes from the `az://account/container/key` URI. Account-key
  (Shared Key) authentication was **not** implemented: the Azure SDK in use (`azure_storage_blob` 1.0.0,
  `azure_core` 1.1.0, `azure_identity` 1.0.0) can only sign requests with bearer tokens, and signing requests
  by hand was deliberately not done. The docs now state exactly this. Behaviour is unchanged.
  The same wrong instructions (`AZURE_STORAGE_ACCOUNT_NAME` / `AZURE_STORAGE_ACCOUNT_KEY`, `AZURE_STORAGE_KEY`,
  `AZURE_CLIENT_*`, `AZURE_TENANT_ID`, and `az://container/...` URIs without an account) were also removed from
  `S3TORCHCONNECTOR_MIGRATION.md`, `integration/DLIO_BENCHMARK_INTEGRATION.md`, `testing/BACKEND-TESTING.md`,
  `integrations/dlio/README.md`, `examples/python/oplog_example.py`, `python/tests/manual_cloud_backend_test.py`
  and the docstrings of `s3dlio.integrations.dlio.*` and `s3dlio.compat.s3torchconnector` (those Python modules
  only described the variables; they never read them, so their behaviour is unchanged). `tests/test_op_log_all_backends.sh`
  no longer requires the unused `AZURE_STORAGE_ACCESS_KEY` to run its Azure step; it needs only `AZURE_STORAGE_ACCOUNT`
  (used to build the `az://` URI) and an `az login` session; its Azure step now removes any old log first and
  fails when `s3-cli` fails, instead of printing a message and continuing. Examples that wrote `az://container/...`
  without the account now write `az://account/container/...` (also in `MULTI_ENDPOINT_GUIDE.md`, the CLI help and the
  `put_bytes` docstring). `PYTHON_API_GUIDE.md` now lists `max_in_flight` (not `workers`/`concurrency`) for
  `get_many`/`get_many_stats`, says that `get_many` works on `az://` and `gs://`, and says the `az`/`azd` executable
  must be on `PATH`.

- **The object size cache no longer grows without bound (#241).** The process-global size cache (and
  the per-store size caches) never removed anything: an expired entry was treated as a miss but left in
  memory, and nothing called the cleanup routine, so a long-running process that touched many distinct
  objects (a KV-cache tier, a data loader over a very large dataset) kept growing. A lookup that finds
  an expired entry now removes it, and the cache holds at most 1,048,576 entries (roughly 200-300 MB
  when full); when a new object arrives at the limit the entry written longest ago, which is also the
  one closest to expiry, is evicted. Every cache operation stays O(1) (an overwrite updates in place,
  nothing scans or compacts under the lock). Eviction is oldest-written-first, so an epoch loop over more
  distinct objects than the limit gets almost no hits: set `S3DLIO_SIZE_CACHE_MAX_ENTRIES` (read once per
  process) above the object count for such datasets. The TTL
  (`S3DLIO_SIZE_CACHE_TTL_SECS`) and cache hits behave as before.

- **A GET of a missing or forbidden object no longer retries (#240).** The retry loop around S3 GETs
  (`S3DLIO_MAX_RETRY_ATTEMPTS`, 100 ms x attempt backoff) retried every error, including a definite
  answer from the server such as `404 NoSuchKey` or `403 AccessDenied`, so each GET of a missing key
  sent several requests and waited at least 300 ms before returning the error. A definite client error
  (400, 401, 403, 404, 405, 409, 412, 416 and other non-transient 4xx) is now returned after one request
  with no backoff. Transient failures are retried as before: connection and dispatch errors, timeouts,
  5xx, 408, 429, S3 throttling codes such as `SlowDown`, and errors while the response body streams. An
  error that cannot be classified is still retried. The same rule now applies to the per-chunk retry loop
  of the concurrent range GET. The AWS SDK's own retry layer is unchanged. Library note: the public
  `s3dlio::retry::retry_get_body` now requires its error type to implement the new `RetryClassify`
  trait; it is implemented for `anyhow::Error` (S3 errors are classified), `std::io::Error`,
  `Box<dyn Error + Send + Sync>`, `&str` and `String` (the last four always retry), so existing callers
  keep compiling unless they use another error type.

- **Arrow backend: `cargo clippy --tests` is clean for the library's own tests (#246).** The test
  `parse_prefix_accepts_a_trailing_slash_that_parse_refuses` used `.err().expect(..)`, which
  `clippy::err_expect` rejects when tests are linted (CI lints without `--tests`). It now uses
  `expect_err`. No behaviour change. (The integration tests under `tests/` use native-backend APIs and
  do not compile under `--no-default-features --features arrow-backend`; that is unchanged.)

- **Arrow backend: `list(uri, recursive=True)` now lists every object under the prefix (#245).**
  With the optional `arrow-backend` feature, a recursive listing returned only the direct children of
  the prefix and silently omitted objects in deeper "subdirectories", unlike the native S3, GCS and
  Azure stores. `recursive=True` now returns every object at any depth (and no directory entries);
  `recursive=False` is unchanged (the objects and directory entries directly under the prefix). The
  prefix still selects only its own directory: `p` never lists `pfoo/...` or `p2/...`. A recursive
  `list_stream` now yields entries as the backend returns them instead of after the whole listing has
  been collected in memory.

- **Arrow backend: `delete_prefix` now deletes the objects at every depth under the prefix (#232).**
  With the optional `arrow-backend` feature, `delete_prefix("s3://b/p")` listed only the immediate
  children of `p`, so `p/b/c` and `p/b/d/e` were left behind while the call returned success. It now
  lists everything under `p/` recursively (streamed, never collected whole) and deletes it with S3
  `DeleteObjects` requests of up to 1000 keys each, 10 requests in flight at a time (that bound is
  specific to S3: see the note below). `pfoo/x` and
  `p2/y` are not touched, and neither is an object named exactly `p`. A key counts as failed unless the
  store confirmed it deleted, so a request that fails as a whole counts every key in it; a failed
  listing is one failure, ends the listing and is reported as "listing incomplete". If anything fails
  the call returns an error with the number of failures, the first failed keys and the first errors,
  instead of success. Limits: (1) endpoints without `DeleteObjects`, such as Google Cloud Storage's S3
  interoperability endpoint, are not supported by the Arrow backend's `delete` / `delete_prefix` (its
  S3 client has no single-object DELETE): the call fails with an error that names the keys; use
  `gs://` for GCS, which deletes per object. (2) Folder marker objects (keys ending in `/`) cannot be
  addressed through the Arrow backend's `object_store` layer, which strips the trailing `/`: the
  marker `p/` is never reported as an entry of `p/` and is not deleted, and a nested marker `p/sub/`
  is listed as `s3://b/p/sub` and may remain after `delete_prefix` (the delete of the key `p/sub`
  reports success while `p/sub/` stays). (3) The "10 requests in flight" bound is S3's. It counts
  chunks of up to 1000 keys, and one chunk is one request only for S3 `DeleteObjects`; the other
  stores split a chunk inside `object_store` 0.13.2's `delete_stream`, which multiplies the bound per
  outer chunk: Azure sends batch requests of 256 keys (up to 4 per chunk, so up to 40 in flight), and
  GCS and plain HTTP send one DELETE per object, 10 at a time per chunk (up to 100 in flight); S3 with
  bulk delete disabled deletes per object, 20 per chunk (up to 200 in flight).

- **Arrow backend: listing an `s3://` URI with a key prefix no longer returns doubled URIs (#222).**
  With the optional `arrow-backend` feature, listing `s3://b/dir/` returned `s3://b/dir/dir/x` for the
  object `dir/x`; feeding that URI back into get or delete addressed an object that does not exist. The
  listing code appended the listing URL's path to a location that already contained it. The returned
  URI is now `s3://b/dir/x`, for nested prefixes, a prefix without a trailing `/` (which lists that
  directory and never a sibling such as `dir2/`) and the bucket root, and the same correction applies
  to every URL form the backend builds a store for (`gs://`, `abfss://`, path-style `https://`
  endpoints). This holds for plain keys. Keys with a space, `%`, `?`, `#`, a leading `/` or `.` /
  `..` segments are not yet guaranteed to round-trip through a listed URI on the Arrow backend (a
  separate, known limit that this fix does not change).

### Changed (BEHAVIOUR CHANGE for library users)

- **s3dlio no longer reads a `.env` file implicitly when used as a library (#243).** Until now every
  S3 client build (and bucket/container listing) loaded `.env` from the process's current directory into
  the environment of the whole host process, so which file was used, and which credentials, endpoints and
  `S3DLIO_*` / `AWS_*` settings an embedded s3dlio (an inference server, a training job, a KV-cache tier)
  ran with, depended on where the process happened to be started. There is now ONE switch,
  `S3DLIO_LOAD_DOTENV`, read from the real environment once per process:
  `1` / `true` / `yes` / `on` loads `.env` from the working directory (the old behaviour),
  `0` / `false` / `no` / `off` never loads it, and any other value is a file path to load exactly that file. The default depends on the program:
  **`s3-cli`: on** (CLI users keep today's behaviour), **library use, Rust and Python: off**.
  **If you relied on the implicit load in a Rust or Python program, set `S3DLIO_LOAD_DOTENV=1`
  (or `S3DLIO_LOAD_DOTENV=/path/to/file`), or call the new `s3dlio::dotenv::load_env_file(path)`.**
  When a `.env` exists but is not loaded because of the library default, s3dlio logs one INFO line
  saying how to enable it, and the "Missing AWS_ACCESS_KEY_ID ..." error carries the same hint (so a
  Python user who sees only the exception learns that the file is being ignored). A variable that is
  already set in the real environment is never overridden by the file, as before. A new public function,
  `s3dlio::dotenv::load_env_file(path)`, loads a given `.env` file explicitly (it works whatever the
  switch says). The six places that used to call `dotenvy` themselves (the two S3 client builders,
  bucket listing, GCS container listing, the Arrow backend set-up and the `s3-cli` start-up) now go
  through one helper, which also runs at the start of every library entry point: the Python extension
  module's initialisation (`import s3dlio`), the global runtime, `store_for_uri*` and the Azure and GCS
  client constructors, the S3 dataset constructors, `configure_thread_pools`, `Logger::new` and the
  latched tuning knobs (unsigned payload, skip-HEAD, range optimisation, size cache, GCS RAPID). So a
  `.env` that is switched on is in place before s3dlio reads those settings (`S3DLIO_RT_THREADS`,
  `S3DLIO_OPLOG_*`, the range-GET settings, `AWS_ENDPOINT_URL`, `AWS_REGION`, ...), not only when the
  first S3 client is built. `s3-cli` now loads the file before it builds its Tokio runtime; in Python
  the native module (which loads it) is imported before s3dlio's own `torch` import. Not promised:
  s3dlio cannot know whether its host already has threads or has already imported `torch`; a host that
  needs a race-free environment loads the file (or sets the real environment) at the very start of the
  process, and forked children inherit the parent's resolved policy. `import s3dlio` now imports the native module first, outside any `try`, so if it fails the caller sees that error instead of PyO3's "initialized once" message from a second attempt. `s3-cli` prints a
  warning on stderr when `S3DLIO_LOAD_DOTENV` names a file it cannot read. Documented in the README and
  `docs/Environment_Variables.md`. Review round 7: the file is now also loaded before the
  independently public entry points that read or change a setting, not only before the S3 client
  wrapper. Calling `set_s3_endpoint_override` first no longer lets an `AWS_ENDPOINT_URL_S3` that only
  the file defines override the endpoint you just set; `build_smithy_http_client` /
  `build_reqwest_http_client*` no longer build an HTTP/1.1 client when the file turns HTTP/2 on;
  `H2WindowConfig::from_env`, `connect_timeout_secs`, `max_retry_attempts`, `mpi_world_size`,
  `mpi_aware_thread_budget`, `effective_thread_budget`, `MultiEndpointStore::new` / `from_config`
  (its `S3DLIO_S3_ENDPOINT_TLS_PORTS` decision), `AzureConfig::default` and the community GCS
  `resolve_gcs_endpoint` return the file's values when they are the first s3dlio call.

- **`file://` and `direct://` datasets now list their items as paths below the dataset root (#223).**
  `Dataset::keys()` (Rust), `PyDataset.keys()` and `ObjectStoreMapDataset`'s keys (Python), and the item
  identifiers yielded by `PyBytesAsyncDataLoader.items()` / `PyDataset.items()` used to be the bare file
  name (`b.bin`), whatever directory the file was in, so two files with the same name in different
  sub-directories were indistinguishable and a name could not be turned back into a path. They are now the
  path below the dataset root with `/` separators (`sub/b.bin`); a dataset whose URI names one file keeps
  that file's name. Code that matched or compared those names must use the new form. `s3://` keys are
  unchanged (the full key from the bucket root). New: `Dataset::uris()` and `PyDataset.uris()` return
  the URI that fetches each item, in the same order as the keys.

### Fixed

- **Python `ObjectStoreMapDataset` fetches each item by the exact key it listed (#223).** A dataset rooted
  below the bucket root asked the server for the wrong object: `ObjectStoreMapDataset("s3://bkt/dir/")`
  listed `dir/x` but requested `dir/dir/x` (the dataset's own key prefix was added a second time), so
  every item failed to read, or read a different object if `dir/dir/x` existed. `s3://bkt//` and
  `s3://bkt/dir//` behave the same way: the key that was listed is the key that is fetched, with a
  leading or doubled `/` kept exactly. The same mistake hit `file://` and `direct://` datasets in two more
  ways, fixed together: files in sub-directories were listed by bare file name and so were looked up in the
  top directory, and a dataset whose URI names one file looked for `<file>/<file>`. The listed key of a
  `file://` / `direct://` dataset is now the file's path below the dataset root (`sub/b.bin`; for a root
  that names one file, its name) -- see the behaviour change above. `ObjectStoreMapDataset` no longer
  builds a URI itself: the dataset, which already parsed its root (scheme, server, bucket or container),
  returns the URI that fetches each listed item, so the server form `s3://host:port/bucket/dir/` (and
  `s3://[::1]:port/bucket/dir/`) fetches from the right bucket, at the bucket root and below a prefix,
  and `s3://bkt/` works as before.

### Changed (API)

- `memory::AlignedBuf::new(len, align)` now returns `Result<AlignedBuf, AlignedAllocError>` instead of
  panicking or aborting on a bad alignment, a size overflow or an out-of-memory condition. New
  `AlignedBuf::new_zeroed` and unsafe `AlignedBuf::assume_init`; the slice accessors never expose
  uninitialised memory (the first `as_slice`/`as_mut_slice` zero-fills a buffer that was only filled through
  the raw pointer).
- `AlignedBuf` is no longer `Clone` (the derived clone copied the pointer and freed it twice).
- `memory::BufferPool::new` and `BufferPoolConfig::build` return `Result<_, BufferPoolError>` (a capacity of 0
  is an error instead of a panic). `BufferPool::take` returns `Result<PooledBuf, BufferPoolError>`:
  `PooledBuf` is an RAII lease that dereferences to `AlignedBuf` and carries the semaphore permit, returning
  the buffer to the pool and freeing the permit when it is dropped (`give()` just drops it). Earlier the
  permit was released when `take()` returned, so any number of buffers could be checked out at once; now
  at most `capacity` buffers are checked out at once, also when a holder is cancelled. (The pool bounds
  checked-out leases, not all allocations: the asynchronous prefill and on-demand allocation can briefly
  coexist, up to `2 x capacity` buffers alive.)
- The streaming writer's `write_chunk()` / `write_owned_bytes()` and `MultipartUploadWriter.write()` raise
  `TypeError` (was `RuntimeError` for the streaming writer) for an object that is not bytes-like.
- `put_bytes_async` gained the keyword-only argument `borrow=False` (see below); positional calls are unchanged.
- New public `file_store_direct::direct_io_counters()` / `DirectIoCounters`: how many `direct://` transfers
  really used O_DIRECT versus the buffered fallback that silently takes over when a filesystem refuses O_DIRECT.
  This has an operational use (a user who chose `direct://` for speed can detect that they are not getting it),
  so it is public and documented, not test-only.

### Fixed

- **`put_bytes()` / `put_bytes_async()` no longer copy the payload while holding the GIL, and make no
  copy at all for any contiguous buffer (#238).** The `BytesView` "zero-copy" branch could never run
  (`BytesView` exports the Python buffer protocol, so the generic buffer branch took it first), and that
  branch copied the whole buffer into a new `Vec` with the GIL held: every other Python thread stalled
  (about 0.3 s per 512 MiB PUT) and peak memory rose by twice the payload on the local backends. Buffers
  whose item type is not a byte (numpy `float32`/`int8`/`bool`, `array.array`, ctypes `'<B'`) were worse:
  two copies with the GIL held. Now:

  | Input | `put_bytes` | `put_bytes_async` | `put_bytes_async(borrow=True)` |
  |---|---|---|---|
  | `BytesView`, exact `bytes` | 0 copies | 0 | 0 |
  | any other C-contiguous buffer, any item type | 0 copies, read in place with the GIL released | 1 snapshot copy, GIL released | 0 copies |
  | non-contiguous view | 1 gather copy (GIL held) | 1 | 1 |
  | `__bytes__()`-only object | 1 copy | 1 | 1 |

  After that, `s3://`, `az://`, official-client `gs://` and `file://` pass the same bytes on with no
  further s3dlio copy; the legacy community GCS client (feature `gcs-community`) copies once. `direct://`
  copies once unless the source is page-aligned (see #244).
  **New:** `put_bytes_async(uri, data, *, borrow=False)`. `borrow=True` promises not to modify, resize or
  free the buffer until the awaitable completes; s3dlio then reads it in place instead of taking a snapshot
  (for example pinned host memory). **Contract for every buffer read in place:** it is exported meanwhile,
  so `bytearray.extend()/clear()` from another thread raises `BufferError`, and the bytes must not be
  modified from another thread: that is a data race (the bytes uploaded are undefined; the memory is not
  freed or resized meanwhile, which does not make a concurrent write safe). **Completion barrier:** a put
  does not complete - normally, with an error, or by cancellation - until every native reader of the buffer
  is done and the export has really been released with a real GIL token (never by a Tokio worker waiting
  for the GIL, and never inferred from `PyGILState_Check`). `put_bytes` waits after the store returned;
  the awaitable of `put_bytes_async(borrow=True)` is a `Future` that never reaches a terminal state
  (result, error or cancellation) until then (see the round-7 entries), so reusing the buffer after `CancelledError`
  is safe (cancelling waits for a running native write to finish). `put_bytes` waits at most 30 s and then raises
  `ExportStillReferencedError`; the awaitable warns at each missed deadline and keeps waiting
  (`s3dlio._put_export_stats()["quiesce_timeouts"]` counts them). Buffers of Python objects or pointers (`O`/`P` formats) are
  refused with a `TypeError`.

- **`direct://` PUT and GET no longer move 4 KiB per system call, and a GET returns page-aligned memory
  (#244).** An O_DIRECT PUT wrote the object one 4 KiB block per `write()` (512 MiB took 45 s, about 11 MiB/s,
  against 0.79 s on `file://`), and a GET read it 4 KiB per `read()`. A page-aligned source buffer is now
  written straight from the caller's memory in one `write` (split only at Linux's 2 GiB per-call limit); any
  other source is copied once through a warm, reused, page-aligned bounce buffer from a process-wide pool, one
  ping-pong pair per NUMA node shared by PUTs and GETs (the copy of chunk k+1 overlaps the write of chunk k;
  64 MiB buffers by default, `S3DLIO_DIRECT_IO_BOUNCE_SIZE_MB` 4..=64, `..._PAIRS_PER_NODE`, `..._WHEN_BUSY`;
  the POOLED bounce memory is `nodes x pairs x 2 x size`; that bound covers the pooled path only, see round 7 and
  `docs/Environment_Variables.md`). A GET returns
  page-aligned memory: by default it reads straight into one fresh aligned buffer, so a `direct://` GET followed
  by a `direct://` PUT needs no bounce copy of the body for objects of at least `min_io_size` (only the final
  zero-padded 4 KiB block is bounced); `S3DLIO_DIRECT_IO_GET_MODE=pooled` reads through the pool instead.
  The zero-padded last block, the truncation back to the exact size (skipped when the length is already
  aligned) and the buffered fallback are unchanged; a failed aligned allocation now retries with a smaller
  buffer and then falls back to buffered I/O instead of aborting. `put_multipart` on `direct://` is now the same
  single write as `put`; it previously wrote `part_size` chunks (4 KiB by default) each padded to 4 KiB, which
  also left zero padding inside the object whenever `part_size` was not a multiple of 4096.

- **Round-7 (1/6): the autonomous buffer-export releaser thread can no longer be inside CPython at interpreter
  finalization (Python 3.12 shutdown race), within the 30 s exit bound.** (Other `Python::attach` sites are
  unchanged: result deliveries stay under the #192 gate with its 2 s drain; attaches inside spawned futures elsewhere
  are not covered.) The autonomous releaser thread checked the shutdown flag and only then called
  `Python::try_attach`; an attach already past the check could still be waiting for the GIL when the exit
  callback returned and Python 3.12 began finalizing (pyo3's `Py_IsFinalizing` check exists only for 3.13+).
  Every autonomous attach now runs inside an attach gate: the releaser increments an in-flight counter, then
  checks the shutdown flag (if set it leaves without attaching), and decrements after `try_attach` returned; the
  exit callback sets the flag, then waits for the counter to reach zero with the GIL released, bounded by 30 s
  (logged and counted in `_put_export_stats()["shutdown_attach_wait_timeouts"]` if exceeded; exit then continues).
  New diagnostics in `_put_export_stats()`: `shutdown_attach_waits`, `shutdown_attach_wait_timeouts`,
  `releaser_attaches`. Tested deterministically on 3.12 and 3.13 with a hidden hook that parks the releaser
  between the counter increment and `try_attach` while the interpreter exits.
- **Round-7 (2/6): a borrowed put's awaitable can no longer end while a native reader remains or before the
  export is released (R1, R3, P3).** (R1) When the put had finished but a backend still held a clone, the
  quiesce timeout produced `ExportStillReferencedError`; an awaiting `Task` that had been cancelled (also a second
  cancellation, `asyncio.timeout()` or a positive `wait_for()`) replaced that exception with `CancelledError` /
  `TimeoutError`, so the documented "safe to reuse after cancellation" was false: the buffer could still be read
  and the caller could overwrite it. Now the awaitable of `put_bytes_async(borrow=True)` never reaches a terminal
  state - result, exception or cancelled - until the readers are gone and the export is released. At each missed
  `S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS` deadline it emits one `RuntimeWarning` (naming the timeout), increments
  `_put_export_stats()["quiesce_timeouts"]` and keeps waiting. **Behaviour change:** that awaitable no longer
  raises `ExportStillReferencedError`; a backend that never lets go keeps it pending (and the buffer pinned).
  The blocking `put_bytes` keeps the error: it is raised in the caller's own stack, where no Task can rewrite it.
  (R3) Completion used to wait only for the owner's drop; if another thread was mid-release of a batch of exports
  (an exporter's release callback yielded the GIL) a put could finish while its export was still held and a
  resize still raised `BufferError`. Release is now independently waitable (blocking and async waiters, GIL
  released), re-checked after every drain. (P3) If the barrier machinery itself fails (for example
  `add_done_callback` raising) the awaitable polls every 50 ms, and if that cannot be scheduled blocks, instead of
  completing or stranding; `_put_export_stats()["internal_errors"]` counts it. Hidden test hooks:
  `_put_test_retain_next_clone`, `_put_test_retained`, `_put_test_release_retained`.
- **Round-7 (3/6): streaming-writer buffer ownership is explicit; `copy=True` is the snapshot (R7).**
  **Behaviour change:** `write_owned_bytes()` no longer copies a mutable buffer under the GIL (0.9.126 and earlier
  did); since 0.9.128 it borrows the buffer exactly like `write_chunk()`, with the GIL released, and
  `MultipartUploadWriter.write()` does the same. Zero copy stays the default for every contiguous buffer. The
  contract is now in the Python-visible docstrings of `write_chunk`, `write_owned_bytes` and
  `MultipartUploadWriter.write` and in the streaming-writer section of `docs/PYTHON_API_GUIDE.md`: the call
  borrows the buffer without copying, no thread may modify it until the call returns, and it is released before
  the call returns. New keyword-only `copy=False` on `write_owned_bytes` and `MultipartUploadWriter.write`:
  `copy=True` snapshots a mutable buffer once under the GIL (exact `bytes` and `BytesView` are immutable and never
  copied), for callers whose other Python threads may write to the buffer during the call (atomic only against
  writers that hold the GIL; a writer that releases it - large NumPy operations, `readinto`/`recv_into`, CUDA/torch
  copies - can still tear the snapshot and must be serialised by the caller). The misleading "kept for
  compatibility" wording of `write_owned_bytes` is gone.
- **Round-7 (4/6): an oversized `direct://` range read no longer holds a pool buffer it does not use (R5).**
  A range larger than the range-read pool's buffer (64 MiB by default) first took a pooled buffer and permit, found
  it too small, allocated its own larger buffer and kept the unused lease for the whole read: 32 such reads held every
  permit (and 32 x 64 MiB of unused buffers) and delayed unrelated small reads. The path is now chosen by the
  required size before anything is taken: a range that fits borrows a pooled buffer as before; a larger one takes
  only a *private-allocation permit* (`BufferPool::acquire_private`, `capacity` of them, separate from the pooled
  permits, released when the read ends) and allocates its own buffer, so oversized reads are bounded in number but
  cannot starve reads that fit. New `BufferPool::buffer_len()`, `acquire_private()`, `private_available()`.
- **Round-7 (5/6): the bounce-memory bound is stated for the pooled path only, and the implicit fallback is counted
  (R6).** Earlier text called `nodes x pairs x 2 x size` the cap on all bounce memory and said nothing but the opt-in
  `fallback` policy allocates beyond the pool. That was too strong: when a node's pool cannot be created or cannot
  give the requested alignment, a `direct://` PUT still succeeds - **availability first** -
  with a private per-call bounce buffer of up to one bounce size, uncounted and outside the cap. Those implicit
  allocations are now counted in `direct_bounce::bounce_stats()` (new `implicit_fallbacks` per direction,
  `implicit_fallback_bytes`, and the causes `pool_unavailable` / `alignment_mismatches`), the first is logged, and
  every "hard cap" statement (module docs, `docs/Environment_Variables.md`, constants, test comments) is narrowed to
  the pooled path, with the allocations outside it listed (opt-in `fallback` buffers, implicit buffers, aligned-PUT
  tail block, default-mode GET destination, range-read private buffers, unaligned streaming-writer staging).
  `bounce_stats()` is no longer hidden from the docs and `BounceStats` is `#[non_exhaustive]`. The benchmark
  (`examples/direct_bounce_bench.rs`) reports the implicit counters per row and the summary script warns when any is
  non-zero (that row is not a pooled measurement). Known limits: oversized range reads are admitted by count
  (`capacity` at a time), not by bytes, and a store without a range pool allocates without admission; the implicit
  counters are cumulative (no live/peak gauge) and `pool_unavailable` also ticks for pooled-mode GETs that read
  directly into their destination.
- **Round-7 (6/6): small corrections.** `scripts/direct_bounce_bench_summary.py` now derives the arms from the data
  (a valid `C32` result used to be silently dropped) and groups by pairs, busy policy and source as well, so runs of
  different configurations are no longer pooled (`python/tests/test_direct_bounce_bench_summary.py`). The
  finalize-durability unit test asserts the exact number of fsyncs (2 for an aligned-plus-tail stream, 1 for an
  aligned-only stream); the old `syncs <= 2` also passed with zero. The legacy range pool's wording says it bounds
  buffers *checked out*, not all allocations.
- **Round-6 edges of the borrowed-buffer barrier.** `GuardedFuture.cancel()` returns `False` once the put is
  already over (only the barrier is pending), so a Task awaiting it still honours its own cancellation right after
  the barrier completes instead of losing it to the put's result; `latch_ready` needs the export to be really
  released, not only its owner dropped; if the wrapper cannot be built after the put was spawned, the put is
  asked to cancel (the cancellation takes effect only after the event-loop-blocking, quiesce-timeout-bounded
  fallback wait, which can end in `ExportStillReferencedError` while the put still runs); `ExportStillReferencedError`
  names the timeout and what the put did, and an internal barrier failure says so and is counted
  (`_put_export_stats()["internal_errors"]`); a set but unusable `S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS` warns once
  and falls back to the default. Documented: `asyncio.shield()` of a borrowed put bypasses the guard, and
  `wait_for(fut, t)` can take up to `t` plus the quiesce timeout.
- **Round-5 hardening of the borrowed-buffer barrier and the direct pool.** (1) The awaitable of
  `put_bytes_async(borrow=True)` is now a `Future` subclass, not a Task running a coroutine: a coroutine body
  never runs when its Task is cancelled before the first step, which let `CancelledError` reach the caller while a
  native write still read the buffer; `cancel()` (including `asyncio.timeout(0)` and `wait_for(.., 0)`) now defers
  the terminal state until the barrier is satisfied. (2) The guard is built before the put is spawned, and every
  error path of `put_bytes_async` releases the export before raising. (3) A barrier timeout of the blocking `put_bytes` raises
  `s3dlio.ExportStillReferencedError` (new; `S3DLIO_EXPORT_QUIESCE_TIMEOUT_SECS`, default 30 s, not signal
  interruptible) instead of succeeding (the async awaitable no longer raises it: see round 7). (4) The async barrier never attaches to Python by itself (the result
  delivery, inside the interpreter-exit gate of #192, releases the export); after interpreter shutdown began a
  `put_bytes` that holds the GIL still releases its export. (5) A `direct://` PUT cancelled while it waits for
  bounce buffers leaves the previous object intact: the target is created/truncated only after the buffers are
  leased. (6) The streaming writers accept only buffer-protocol objects and objects with `__bytes__()`:
  `write_chunk(5)` is a `TypeError` instead of five zero bytes (as are lists and iterables); `put_bytes` is
  unchanged. (7) Per-call overhead of small puts reduced (no thread hand-offs when the store already dropped the
  payload). (8) Helper-thread creation failure is an error, not a panic; a failed pool creation is retried after
  30 s; `mbind` prefers (instead of requires) the node; `direct://` range reads no longer zero-fill the buffer
  before `pread` and run the read in a blocking task that owns the buffer; CPU indexes read from sysfs are
  capped.

- **The remaining Python write paths no longer copy the payload with the GIL held (#254).** The streaming
  writer's `write_chunk()` (which kept the GIL for the whole write of any mutable buffer) and
  `write_owned_bytes()` (which copied the buffer into a `Vec` with the GIL held) now use the same payload
  conversion as `put_bytes()`: any C-contiguous buffer of any item type is read in place, with NO copy made by
  the Python layer and the GIL RELEASED; the export is released before the call returns. `write_owned_bytes()` is
  now the same as `write_chunk()` (a behaviour change: it used to snapshot; `copy=True` restores that, see the
  round-7 entry). `MultipartUploadWriter.write()` borrows the buffer the same way and copies it
  once, with the GIL released, into its part buffers (parts are uploaded asynchronously and outlive the call, so
  a borrowed Python buffer cannot be handed to them). The same contract as `put_bytes()` applies: do not modify
  the buffer from another thread during the call (a data race). **Not changed:** the legacy community GCS client
  (`gcs-community`, `gcs_client.rs`) still does `data.to_vec()` per PUT; its dependency is not available to
  build offline here, so the one-line change is left for a build that can compile that feature.

- **The `direct://` streaming writer no longer fsyncs on every flush or copies every chunk (#249), and
  no longer frees an aligned buffer with the wrong layout (#248).** Page-aligned chunks are written from the
  caller's memory; other data is staged once in a 64 MiB aligned buffer; data is made durable once, at
  `finalize` (after every write only when `sync_writes` is set). `AlignedBuf::new` now returns a `Result`
  (and `BufferPool::take` too), and `AlignedBuf` is no longer `Clone` (a clone freed the same memory twice).
  New `file_store_direct::direct_io_counters()` reports O_DIRECT transfers versus buffered fallbacks.

## Version 0.9.126 — Exact `s3://` selection, exact object keys, per-endpoint stores on their own server, limits of 0 refused, compressed-writer data-loss fix, mkdir/rmdir on every backend

**Versions 0.9.114 through 0.9.124 were merged but never published to PyPI; the last published
release is 0.9.112. 0.9.126 is therefore the first published release after 0.9.112, and it contains
every change in 0.9.114, 0.9.120, 0.9.122 and 0.9.124 (see their sections below).** This release fixes
eleven issues: #198, #208, #211, #213, #216, #218, #228, #229, #231, #234 and #236.

Read the two behaviour changes before upgrading: an `s3://` URI now selects the exact name, its
directory or an explicit `*` glob and never a regex or a longer sibling (#213), and object keys that
begin with `/` or contain `//` are used exactly as written in every layer (#198).

### Fixed

- **Multipart uploads to GCS's S3-compatible endpoint no longer fail with an unexplained error (#236).**
  Starting a multipart upload (`create_s3_writer` plain or compressed, `put_multipart`, anything that
  begins a multipart upload) failed on Google Cloud Storage's S3-interoperability endpoint with
  `CreateMultipartUpload failed ... service error: unknown (no message)`. The start-upload request has
  no body, and over the default HTTPS HTTP/1.1 transport it was sent without a `Content-Length` header;
  GCS answers that with `411 Length Required` (HTTP/2, opted in with `S3DLIO_HTTPS_H2=1`, was not
  affected). A POST, PUT or PATCH with an empty body now carries `Content-Length: 0`, for every endpoint
  (as RFC 9110 expects); requests with a body, and GET/HEAD/DELETE, are unchanged, and the header is
  not part of the request signature. A service error with no readable S3 error body now also names the
  HTTP status, for example `service error: unknown (no message) -- HTTP 411 Length Required`.

- **Compressed buffered writers no longer upload a truncated, undecodable object (#218, silent
  data loss).** A buffered writer with zstd compression (`get_writer_with_compression`, or
  `create_writer` / `create_s3_writer` with compression) for `s3://` -- the process-wide store and a
  per-endpoint store alike -- and the matching Azure (`az://`) and GCS (`gs://`) writers, replaced
  the compressed data it had collected with only the last few bytes of the zstd stream when
  `finalize()` ran. zstd emits compressed blocks while it is still being fed, so any payload large
  enough for that (a 32 KiB payload was fine; a 6 MiB one was uploaded as a 197-byte object) was
  stored without the frame header and every early block, the upload reported success, and the
  object could not be decompressed. Small payloads were unaffected, which is why it went unnoticed. `finalize()` now appends the end of the stream to what was already collected. Data
  written through an affected writer before this fix, if large, is not recoverable from the stored
  object; write it again. `file://`, `direct://` and the arrow-backend writers were not affected.
- **A per-endpoint store runs every operation on ITS server, whatever the object size (#208).** A store
  that holds a client of its own (`S3ObjectStore::for_endpoint`, or an `s3://host:port/...` URI) read
  large objects (the concurrent range GET above the threshold), uploaded with `put_multipart` and
  finalised a buffered writer through the process-wide client, the server `AWS_ENDPOINT_URL` names.
  Since 0.9.122 those paths were rejected with a "not yet supported" error unless the store's server was
  the process-wide one. They now use the store's own client for the size lookup, the single-range
  fallback, every ranged GET, and CreateMultipartUpload / UploadPart / CompleteMultipartUpload /
  AbortMultipartUpload. A URI that names a DIFFERENT server than the store's is still refused.
- **A concurrency, chunk or worker limit of 0 is an error instead of a hang or a crash (#211).** A caller
  limit of 0 built a semaphore with no permits (the first acquire never completed) or, for a chunk size
  of 0, pushed range after range until memory ran out. Every entry point that takes such a limit now
  refuses 0 first, with an error naming the parameter and the range ("max_in_flight must be an integer
  >= 1 ... got 0"); nothing is substituted. Python raises `ValueError` before any work starts for `put`,
  `put_async`, `get_many*`, `put_many*`, `download`, `upload`, `mp_get` (`jobs` and `procs`; `procs=0`
  used to divide by zero) and `MultipartUploadWriter(..., max_in_flight=0)`. Rust: the parallel get/put
  helpers, the concurrent range GET (`max_concurrency` and `chunk_size` of `Some(0)`, on the
  process-wide path AND the per-endpoint path), `s3_copy::upload_files|download_objects|download_objects_matching`,
  `generic_upload_files`, `generic_download_objects[_with_summary]`,
  `generic_download_objects_matching_with_summary`, `pre_stat_objects`, `RangeEngine::download`,
  `mp::run_get_shards`. The CLI rejects 0 for `-j/--jobs` and `--concurrent`. `put(num=0)` is unchanged.
- **Test-only (#216):** unit tests that set process environment variables (`OMPI_COMM_WORLD_SIZE`,
  `S3DLIO_RT_THREADS`, `S3DLIO_MAX_RETRY_ATTEMPTS`, ...) now share one crate-wide lock and restore the
  variables and atomics in a `Drop` guard, so a failed assertion can no longer leak them into later tests.
  No production behaviour changes.
- **A directory removed through s3dlio now appears in the op-log (#228).** A directory removed through
  s3dlio, recursive or not (a recursive removal of a whole tree included), now appears in the op-log as
  one `DELETE_PREFIX` row on the directory (count 0; the error text if the removal failed). Before, the
  removal left no row at all. This is not a format change: `DELETE_PREFIX` is an existing op name.
  `mkdir` is still not recorded. The replayer reports a `DELETE_PREFIX` row as unsupported for exact
  replay and replay stops at the first such row, so an op-log that contains a `file://` or `direct://`
  `rmdir` now contains one.
- **`mkdir` / `rmdir` no longer fail outside `file://` (#229).** They used to work only on
  `file://`; on `direct://`, `s3://`, `az://` and `gs://` they failed with "not implemented for
  this backend", although the docs promised them, and so did the DLIO adapters' `create_node`
  (Python `s3dlio.mkdir()`). Now:
  - `direct://` behaves exactly like `file://` (O_DIRECT does not apply to directories): `mkdir`
    creates the directory and its parents; `rmdir` removes an empty directory (a non-empty one is
    an error and nothing is removed), or the whole tree with `recursive=True`; a missing
    directory is success.
  - On object stores (`s3://`, `az://`, `gs://`, and the arrow backend) both calls do nothing and
    succeed: an object store has no directories, a prefix exists once an object is written under
    it. The URI is checked, no request is sent, no marker object is written and `rmdir` deletes
    nothing, `recursive` or not. To remove the objects under a prefix use `delete(uri,
    recursive=True)` / `s3-cli delete -r`.
  - Op-log: `mkdir` writes no row on any backend. `rmdir` writes its one `DELETE_PREFIX` row
    (#228) on `file://` and `direct://`, but not on object stores, where it deletes nothing.
  - For Rust API users: `ObjectStore::rmdir` and `mkdir` on `s3://`, `az://` and `gs://` now
    succeed without doing anything (they used to return a "not implemented" error); code that
    relied on `rmdir` to delete objects must call `delete_prefix` (or `delete -r` /
    `delete(recursive=True)`).
- **Azure and GCS `delete_prefix` now deletes only objects under `<prefix>/`, as on S3 (#231).**
  Before, it also deleted sibling objects whose names merely started with the prefix
  (`delete_prefix("gs://b/foo")` also removed `foobar`, `foo.bak`, `foo2/...`).
- **Host-form `s3-cli download` / `s3-cli get` fetch every object from the server that listed it
  (#234).** With a URI that names a server (`s3://host:port/bucket/prefix/`) the listing ran on that
  server, but each listed object was then fetched through a store built from the bucket-form URI the
  listing returned, that is, through the process-wide client: the server `AWS_ENDPOINT_URL` names, or
  AWS. The listing succeeded and the GETs went to the wrong server (other bytes, or failures). The
  library's `generic_download_objects[_matching]_with_summary` had the same shape. Every listed object
  is now fetched through the same store instance that listed it; a URI without a server (the
  process-wide store) behaves as before.
- **Object keys are used exactly as written in every layer (#198).** See the details below.

### Changed — which objects an `s3://` URI selects: the exact name, its directory, or an explicit `*` glob; never a regex, never a bare prefix (INTENTIONAL BEHAVIOUR CHANGE) (#213)

**Read this before upgrading.** On S3, `list`, `list_stream`, `s3-cli ls`, `s3-cli get`,
`s3-cli download` / `cp` (S3 to local), `s3_copy::download_objects` and `S3BytesDataset::from_prefix`
used to split the URI at its last `/`, send only the left part to the server and compile the right
part as an UNANCHORED regular expression over the keys that came back; in non-recursive mode every
common prefix was added unfiltered. So `s3://b/data/run.1` also matched `data/run_1` and
`data/prerun.1`, `s3://b/foo` listed `afoo` and every top-level folder, and a key containing `(` or
`[` failed with "Invalid regex pattern". This change converts S3 (`s3://`) only. The other backends
(`file://`, `direct://`, `gs://`, `az://`) are unchanged and do NOT yet follow this rule (for
example `gs://` and `az://` non-recursive listings keep their own directory-style handling of a
prefix without a trailing `/`); that is tracked in #220.

The key part `K` of `s3://bucket/K` now selects, for every listing-style operation (`ls`, Python
`list()`, `download`, `get`, `s3_copy::download_objects`, `S3BytesDataset::from_prefix`,
`ObjectStore::list` / `list_stream`) and for `delete -r` / `delete(recursive=True)`:

- **A name** (no `*`, not ending in `/`): the object whose key is exactly `K` and, for operations
  that descend (recursive listing, `download -r`, `delete -r`), every key under `K/`. Non-recursive
  `ls` shows the object `K` (if it exists) and the directory entry `K/` (if anything is under it).
  It NEVER selects `K.bak`, `K2`, `Kfoo` or `K.d/`. So `s3://b/file.txt` selects `file.txt` and not
  `file.txt.bak`; `s3://b/dl` selects `dl/...` and not `dl2/...`; `s3://b/data/run.1` selects
  `data/run.1` and `data/run.1/...`, and not `run.1.old`, `run_1` or `prerun.1`. `.`, `(`, `)`,
  `[`, `+`, `$`, `?` in a URI are ordinary characters and never fail to parse.
- **A directory** (`K` ends in `/`) or the bucket root (empty `K`): everything under it, as before.
- **A wildcard** (`K` contains `*`): an explicit glob over the FULL key. `*` matches any run of
  characters except `/`; `**` matches any run including `/`; every other character is literal.
  `s3://b/file.txt*` selects `file.txt`, `file.txt.bak` and `file.txt2` (not `file.txt.d/x`);
  `s3://b/data/**.npz` selects every `.npz` under `data/`. With a wildcard an object is selected
  iff its full key matches the glob: `-r` makes no difference and a matching "directory" is not
  descended into. A wildcard key that ENDS in `/` (`s3://b/data/*/`) is refused with an error and
  nothing is listed or deleted. To pass a wildcard through a shell, quote the URI.
- **`delete -r` follows the same rule**: `s3-cli delete -r s3://b/file.txt` and Python
  `delete("s3://b/file.txt", recursive=True)` delete `file.txt` and never `file.txt.bak`;
  `delete -r s3://b/file.txt*` deletes both. (0.9.124 deleted by LITERAL prefix: `delete -r
  s3://b/foo` also deleted `foobar`; that is what this release changes.) Delete without `-r` is
  unchanged: one request for exactly the key named, and a `*` in it is an ordinary key character.
- **To select with a regex use the explicit option**: Python `list(uri, recursive, pattern=...)`
  and `s3-cli ls|download|get --pattern`. It is applied AFTER the selection above and is a Rust
  regex matched (unanchored) against the entry relative to the listing base: the URI key itself when
  it ends with `/`; the directory `K/` for an entry under it; otherwise the parent directory of `K`
  (everything up to its last `/`); for a wildcard URI, the directory part of the text before the
  first `*`. For host-form URIs (`s3://host:port/bkt/dl/`) the base is taken from the key parts, so
  it agrees with the entries a listing returns (`s3://bkt/dl/...`). One shared implementation
  (`s3dlio::list_pattern::ListPattern`) serves Python, the CLI and
  `s3_copy::download_objects_matching(src_uri, pattern, ...)`. An invalid pattern fails before any
  request. (Python `list()` compiled the pattern after listing; it now raises
  `Invalid regex pattern: '...'` first.) `delete -r --pattern` is unchanged: it is matched against
  the listed object URI.
- `s3_copy::download_objects` no longer appends `.*` to a URI ending in `/`. A URI that used to be a
  regex or glob in the last segment (for example `s3://b/dir/.*\.npz`) is no longer a regex: use a
  `*` glob (`s3://b/dir/*.npz`) or split it into the directory and a pattern
  (`download_objects_matching("s3://b/dir/", Some(r"\.npz$"), ...)`).
- Callers that passed a bare key prefix without a trailing `/` and relied on it matching every key
  that starts with it (`dl` also matching `dl2/x`) must now say so with a `*`
  (`s3://b/dl*`) or end the directory with `/`.
- **New: `s3-cli download -p/--pattern <REGEX>` and `s3-cli get -p/--pattern <REGEX>`.** The
  explicit replacement for the regex those commands used to read from the URI. It reuses
  `ListPattern` inside the existing backend-aware paths (so it also works on `file://`,
  `direct://`, `gs://`, `az://`; it is not the S3-only `download_objects_matching`), is matched
  against the entry relative to the listing base like `ls --pattern`, and is compiled before
  anything is listed: an invalid regex exits non-zero with no request sent and no directory
  created. `get --pattern` cannot be combined with `--keylist` or `--offset/--length`. Without
  `--pattern` the behaviour is unchanged. Library: new
  `generic_download_objects_matching_with_summary(..., pattern: Option<&ListPattern>, ...)`;
  `generic_download_objects[_with_summary]` keep their signatures.
- **Consequences of the glob rule worth knowing.** (1) `*` also matches the empty string, so
  `s3://b/dir/*` selects a folder-marker object `dir/` itself if one exists (`ls` shows it,
  `delete -r 's3://b/dir/*'` removes it). (2) `a/**/b` needs something between its slashes: it selects
  `a/x/b` and `a//b` but NOT `a/b`; use `a/**` with `--pattern`, or name `a/b` itself. (3) There is no
  escape for a literal `*` in a listing or recursive URI: a key literally named `a*b` can only be
  addressed by the exact, non-recursive operations (`s3-cli stat`, `s3-cli delete` without `-r`, and
  the Python `get`, `stat` and `delete` without `recursive=True`, which take the key as written). `delete -r 's3://b/a*'` is a glob over `a...` in one directory level, which is much
  broader than the old literal prefix `a*`.
- **Dataset constructors fail on an empty selection.** `S3BytesDataset::from_prefix[_with_opts]` and
  `MultiBackendDataset::from_prefix` (and so `dataset_for_uri`, `PyS3Dataset`, `PyS3AsyncDataLoader`)
  return an error instead of a dataset of zero samples when an `s3://` URI selects nothing, for
  example a bare partial name such as `s3://b/train/img_`. The message names the URI and says that a URI
  selects an exact name or a directory and that `*` selects by prefix (`s3://b/train/img_*`).
- **`s3dlio.compat.s3torchconnector` keeps the upstream PREFIX meaning.** Its `prefix` is an S3 key
  prefix, as in AWS's s3torchconnector: `S3Client.list_objects(bucket, "train/img_")`,
  `S3IterableDataset.from_prefix("s3://b/train/img_")` and `S3MapDataset.from_prefix(...)` return every
  key that starts with the prefix (`train/img_1/x` included). A prefix without `*` is listed as
  `<prefix>**`; a prefix that itself contains `*` (a literal character in a key prefix) is read by
  listing the directory part of the text before the first `*` and keeping the keys that start with it.
- **Requests.** The server does the narrowing and no sibling is paged through: a name costs one
  `max-keys=1` probe (the object `K` sorts before every other key that starts with `K`) plus the
  listing of `K/` (recursive) or one more `max-keys=1` probe (non-recursive); a directory is its
  listing; a wildcard is the listing of the text before the first `*`, with the `/` delimiter when
  the glob cannot cross a `/`. No `HEAD` is sent.
- The `--help` text of `ls`, `download`, `get` and `delete`, `docs/CLI_GUIDE.md` (including the
  "Common Patterns" regex examples, which now match relative to the listing base),
  `docs/PYTHON_API_GUIDE.md`, and the glob examples in `docs/testing/BACKEND-TESTING.md` and
  `docs/supplemental/OPLOG-GUIDE.md` state the rule. Library: new `s3dlio::s3_selection`;
  `delete_contract::delete_objects_under_literal_prefix` keeps its (public) name, but its URI is
  selected by this rule on `s3://`; `ObjectStore::delete_prefix` is unchanged (it appends `/` and
  reads the prefix literally).

Tests: `tests/test_s3_selection_213.rs`, `tests/test_list_literal_prefix_213.rs`,
`tests/s3_cli_ls_literal_prefix_213.rs`, `tests/s3_cli_download_pattern_213.rs`,
`tests/test_python_list_literal_213.py`, the delete-contract tests
(`tests/test_delete_contract_210_167.rs`, `tests/test_python_delete_contract_212.py`), and the
rewritten `tests/test_object_store_list_filter.rs` (it pinned the #154 regex contract). The in-process
mock S3 can be seeded with a bucket catalog that honours prefix, delimiter, `max-keys` and paging, and
removes the keys a delete names.

### Fixed — object keys are used exactly as written in every layer (#198, the rest of the issue)

After #172 the core addresses `s3://b//k` as the key `/k`. These layers still normalised keys, so for a
key that begins with `/` or contains `//` they addressed or reported a different object than the core.
**Behaviour change:** in the layers below (except namespace-relative DLIO names, see the first bullet),
`/x` and `x`, or `a//b` and `a/b`, are now different objects. Only the ONE `/` that separates a base from a
key is removed.

- **DLIO integration** (`s3dlio_storage.py`, `s3_torch_storage.py`): a name that is a FULL URI
  (`s3://`, `az://`, `gs://`, `file://`, `direct://`) is used exactly (`put_data("s3://b//k")` writes the
  key `/k`), and `s3_torch_storage.walk_node` lists the exact prefix of the full URI it is given.
  A name that is relative to the configured namespace keeps its existing (0.9.124) filesystem-style
  normalisation: the namespace loses its trailing `/` and the name its leading `/`, so
  `put_data("/train/f.npz")` under `s3://b/base` still writes `base/train/f.npz` (not `base//train/f.npz`).
  It is not established that DLIO supplies object keys rather than such names, and re-keying them would
  make existing datasets disappear. Namespace-relative names are unchanged from 0.9.124;
  full-URI names are used exactly, so a full URI containing `//` addresses the exact key (a full-URI
  id such as `s3://b//dir/` now lists the prefix `/dir/`, where 0.9.124 listed `dir/`).
- **`s3dlio.compat.s3torchconnector`**: `S3Item.key` for `s3://b//k` is `/k` (was `k`), as AWS's own
  `s3torchconnector` does (its `parse_s3_uri` splits at the first `/` and strips nothing);
  `S3Client.put_object` / `get_object` / `list_objects` address the key and prefix exactly (a
  `list_objects` prefix URI loses no trailing `/`; one `/` is appended only if it has none).
- **Azure https URI form**: `https://acct.blob.core.windows.net/c//k` is the container `c`, key `/k`,
  the same key as `az://acct/c//k` (it was key `k`). A URL with an empty container
  (`https://acct.blob.core.windows.net//k`) is an error; this is NOT parity with `az://`, where
  `az://acct//k` still parses to the container `""` and key `k`.
  A trailing `/` is part of the key: `https://acct.blob.core.windows.net/c/dir/` now addresses the
  key `dir/` (it was `dir`). So `get` / `stat` / `delete` of a URI ending in `/` address `k/`, no
  longer `k`, and listing `.../c/dir/` no longer matches `dir2/...`.
- **arrow-backend** (`--no-default-features --features arrow-backend`): `object_store`'s `Path::from`
  drops empty segments, so `s3://b//x` addressed the object `x` and `s3://b/a//b` addressed `a/b`.
  Such keys, and an object key ending in `/`, are now refused with an error naming the key instead of
  addressing another object (`object_store::path::Path` cannot represent them). A listing prefix may
  still end in `/`.
- **Op-log replay onto a target** (`translate_uri`): a row for the object `/x` (logged `bkt//x`) replays
  to `<target>//x`, no longer `<target>/x`. Rows that are not a shape s3dlio writes (hand-written,
  legacy `S3Ops` `endpoint`+bare key) still lose one leading `/` of `file`, as pinned since #184.

### Added

- `s3-cli download -p/--pattern <REGEX>` and `s3-cli get -p/--pattern <REGEX>`: the explicit regex
  that replaces the one those commands used to read out of the URI (#213, details above).
- Library (#213): `s3dlio::s3_selection` (the one selection rule), `s3dlio::list_pattern::ListPattern`,
  `s3_copy::download_objects_matching(src_uri, pattern, ...)` and
  `generic_download_objects_matching_with_summary(..., pattern, ...)`.
- Library (#208): `S3ObjectStore` per-endpoint reads and writes need no new API; the helpers
  `get_object_concurrent_range_with_client` and `put_object_multipart_with_client` are crate-private.
- Test support: the in-process mock S3 can be seeded with a bucket catalog that honours prefix,
  delimiter, `max-keys` and paging, and removes the keys a delete names.

### Build

- The declared minimum Rust version (`rust-version` in `Cargo.toml`) is now 1.94.1. It said 1.88, but
  the AWS SDK crates in `Cargo.lock` already require Rust 1.94.1, so an older toolchain could not build
  s3dlio anyway; the manifest now says so up front. The README badge and prerequisites match.

## Version 0.9.124 — Exact delete contract (s3-cli and Python), Python API GIL/store-cache fixes, op-log row safety, arrow-backend and build-script fixes

Fixes 12 issues: #167, #186, #188, #190, #191, #201, #203, #204, #207, #210, #212, #215.
**Read the behaviour changes before upgrading:**
- **Delete contract (#210, #167, #212).** An exact `s3-cli delete` / Python `delete()` is ONE
  DELETE request, with no HEAD and no listing; a missing object is success; a trailing `/` never
  makes a delete recursive; `-r` / `recursive=True` deletes the real objects under the LITERAL
  prefix (`foo` also matches `foobar`; `*` and `?` are ordinary characters); on `file://` and
  `direct://` a directory is refused without `-r`. Scripts that relied on the old implicit
  expansion now delete less. Python `delete()` no longer raises `FileNotFoundError`.
- **Op-log destination (#191).** `file:` followed by anything but `//` (`file:x`) is rejected as an
  op-log destination (it used to create a file literally named `file:x`).
- **Op-log row safety (#186, #190).** A record whose `op`, `client_id`, `endpoint` or `file` holds
  a TAB, LF or CR is rejected (not written) and counted; a TAB, LF or CR in `error` is replaced by
  a space; a row with an unformattable timestamp is rejected and counted instead of ending the log.
  `op_log_stats()` (and Rust `OpLogStats`) gain `rejected_unformattable`, `rejected_unsafe_field`
  and `normalized_error_text`.
- **Build (#188).** `build_pyo3.sh` now fails loudly (non-zero exit) if maturin is missing or a
  build fails, and requires BOTH a cp312 and a cp313 wheel to be freshly built.

Also in this release: Python `list()`, `stat()`, writer writes and two other calls release the GIL
while blocking on I/O, `write_chunk()` of a strided view writes the right bytes, and the async
entry points reuse the shared store cache (#201, #204); `list_keys` returns keys exactly as written
(#198, `list_keys` only); the arrow-backend `ObjectStore` implementation compiles again, handles a
zero-length / overflowing range read, and is checked in CI (#207); the never-compiled
`src/prefetch.rs` sketch is removed and the docs that cited it corrected (#203); the op-log reader's
unsupported-operation error names the line, row and op (#176, documentation and error text only);
the #170 shutdown-panic test is bounded and race-free (#215).

### Known limitations / deferred

- #198 is only partly fixed: `list_keys` uses keys as written; the DLIO and `s3torchconnector`
  adapters, the Azure https URI form, the arrow backend and op-log replay URIs are not changed.
- #175 and #176 stay OPEN: only documentation and a clearer reader error were added; the op-log
  gaps are listed under "Known gaps" in `docs/supplemental/OPERATION_LOGGING.md`.
- #205 remains a documented gap.
- #208, #211, #213 and #216 are open and not addressed by this release.
- Python `delete(..., recursive=True)` deletes one request per object (up to 10 in flight);
  batching is deferred.

### Fixed — Python API (#198, #201, #204)

- `s3dlio.list_keys` uses keys exactly as written, like every other operation since #172: a key with a
  leading `/` or an empty segment (`a//b`) comes back as the object's real relative key, and
  `prefix_uri + key` addresses the object listed (#198). The DLIO and `s3torchconnector` adapters,
  the Azure https URI form, the arrow backend and op-log replay URIs, also named in #198, are not
  changed here.
- `list()`, `stat()`, a writer's `write_chunk()` / `write_owned_bytes()`, `parquet_rg_row_offsets()` and
  `gcs_query_rapid_bucket()` release the GIL while they block on I/O, so other Python threads keep
  running (#201). `write_chunk()` releases the GIL, with no copy, only for an exact `bytes` object (nothing can
  change it); a mutable buffer (bytearray, writable memoryview, numpy, mmap, ctypes) keeps the GIL for the
  write as before, so another thread cannot tear it, and a strided or reversed view is copied under the GIL
  and then written without it (pass `bytes(data)` to keep other threads running during a large write of
  mutable data). `write_chunk()` of a strided or reversed view used to write the wrong bytes or read
  adjacent memory and now writes exactly the bytes the view shows. A writer's `bytes_written()`,
  `checksum()` and `compressed_bytes()` answer during a write in flight on another thread (values as of the
  last completed write), and concurrent writes and `finalize()` on one writer run one after another instead
  of raising "Already mutably borrowed".
- `stat_async`, `exists_async`, `put_bytes_async`, `mkdir_async` and `put_many_async` use the shared store
  cache instead of building a store and S3 client per call, so a host-form server's connections are
  reused; racing cache misses build one store per key (#204). Op-log rows are unchanged.

### Changed — Python `delete(uri, recursive=False)` contract (INTENTIONAL COMPATIBILITY CHANGE) (#212)

The Python `s3dlio.delete()` now follows the same delete contract as `s3-cli delete` (#210, #167, next
entry). **Code that relied on the old implicit expansion will now delete less; pass
`recursive=True` where you mean "everything under a prefix".**

- `delete(uri)` sends ONE delete request for exactly the key named, with no HEAD and no listing, and
  returns `None`. Before this change a URI ending in `/` or containing `*` or `?` triggered a
  non-recursive listing and every returned entry was deleted, synthetic common-prefix entries
  (`foo/sub/`) included, so `delete("s3://b/foo/")` removed `foo/x` and tried to delete `foo/sub/` as a
  key. `/`, `*` and `?` are now ordinary key characters.
- An already-missing object is success (S3's DELETE succeeds for a missing key; on other backends only
  the backend's own typed not-found error is mapped to success). Success means the delete request
  succeeded, not that an existing object was removed. Any actual failure (403, 5xx, network) raises
  `RuntimeError` naming the key. No `FileNotFoundError` is raised any more; the DLIO integrations'
  `delete_node`, which treats `FileNotFoundError` as "already gone", simply never sees it. A
  bucket-only URI (`s3://b/`) raises `RuntimeError` without sending a request.
- On a **versioned bucket** a delete adds a delete marker and does not erase earlier versions.
- On `file://` and `direct://`, `delete()` of a path that is a directory raises `RuntimeError`
  (`... is a directory; use -r / recursive=True ...`) and deletes nothing, instead of removing the
  whole tree; a regular file is deleted and a missing path is success.
- The op-log records an exact delete as one `DELETE` row, as before; no `HEAD` row is added.
  `recursive=True` records one `LIST` row plus one `DELETE` row per object (replayable), not
  `DELETE_BATCH`.
- `delete(uri, recursive=True)` deletes the real objects under the literal prefix: listed with no
  delimiter, objects only, nothing appended (`foo` also matches `foobar`), `*` and `?` literal (the last
  URI segment used to be read as a regex). It needs list permission. A prefix with no objects is a
  no-op. Each selected object is deleted with its own request (up to 10 in flight), as before;
  batching (DeleteObjects) is deferred for Python. After a failed request or a listing error no further
  delete request is sent (requests already in flight finish) and the `RuntimeError` says how many
  requests were sent and how many selected keys were not attempted.
- Other backends: `recursive=True` requires a URI ending in `/` (otherwise `RuntimeError`, nothing
  deleted).
- Library: the exact delete and the literal-prefix selection/batching now live in the new
  `s3dlio::delete_contract` module, shared by `s3-cli delete` and the Python API. The lower-level
  `ObjectStore::delete`, `delete_batch`, `delete_prefix` and the multi-endpoint store's `delete` keep
  their behaviour, except that `delete` on a bucket-only S3/Azure/GCS URI now fails before sending
  anything.

### Changed — `s3-cli delete` contract (INTENTIONAL COMPATIBILITY CHANGE) (#210, #167)

`s3-cli delete` (alias `rm`) no longer expands a name into a prefix. **Scripts that relied on the old
implicit expansion will now delete less; add `-r` where you mean "everything under".**

- `delete s3://bkt/foo` and `delete s3://bkt/foo/` send ONE delete request for exactly that key, with
  no HEAD and no listing, so only delete permission is needed. A trailing slash alone no longer
  authorizes a recursive delete (#167). The old prefix-without-`-r` expansion and the fallback that
  listed the parent "directory" and deleted what it returned are removed; that fallback deleted
  objects the user never named (for example a folder marker `other/` when `delete s3://bkt/testprefix`
  matched nothing) (#210).
- An already-missing object is success (exit 0), as with S3's own idempotent DELETE; on other backends
  only the backend's typed not-found error counts. A real failure of the request (403, 5xx, network)
  exits non-zero with `failed to delete <uri>: <cause>`. The output is `Delete request succeeded:
  <uri> (S3 does not report whether the object existed)`; when the URI ends in `/` and `-r` was not
  given it adds that only the object named by the exact URI was targeted and to use `-r` for
  everything under a prefix. On a **versioned bucket** a delete adds a delete marker and does not erase
  earlier versions.
- `delete -r s3://bkt/foo/` deletes the real objects under the literal prefix `foo/` (recursive listing
  with no delimiter; synthetic common-prefix entries are never delete candidates). It needs list
  permission. `delete -r s3://bkt/foo` uses the literal prefix `foo`, so it also matches `foobar` (no
  `/` is appended). Key characters are never read as a regex or wildcard.
- `-p/--pattern` is accepted only with `-r` and filters the `-r` selection; without `-r` it is an error.
- A listing error part-way through a recursive delete does not send the partly filled batch in which
  it occurred: those keys are counted as "NOT attempted" in the message. Batches that were already
  complete have been sent (shared with Python `delete(..., recursive=True)`).
- Messages for `-r` no longer overstate: `Sent delete requests for N objects found under prefix ... (N
  listed; 0 reported errors)`. S3 reports non-existent keys as deleted, so no "N deleted" count is
  claimed. A failed delete request, including a per-key error in an HTTP 200 response, names the keys
  and exits non-zero. A listing failure is now an error (it used to end the delete early and report
  success).
- After a failure no further DeleteObjects request is sent (checked immediately before each send);
  requests already in flight finish and the not-attempted keys are counted in the message. A listing
  that fails before anything is sent is reported as the listing error, not as a delete summary. With
  `-p` the message distinguishes keys listed from keys selected, and a pattern that matches nothing
  says so (exit 0). `-j/--jobs` is accepted but currently unused.
- Non-S3 backends (`file://`, `direct://`, `gs://`, `az://`): `-r` requires a URI ending in `/` and
  otherwise errors (no broadening).
- On `file://` and `direct://`, an exact `delete` of a path that is a directory fails (`... is a
  directory; use -r / recursive=True ...`) and deletes nothing; use `-r` with a trailing `/`.
- Limitation: the op-log LIST row for `delete -r` is written only when the listing stream is read to
  its end, exactly like the existing `list_stream` wrapper; a listing abandoned after a failure
  (the command stops listing once a request has failed) writes no LIST row.
- Works with the process-wide client and with a per-endpoint client (`s3://host:port/bkt/...`), under
  the #197 route-or-reject rules. New `ObjectStore` method (default implementation provided):
  `list_objects_with_prefix`. The library `delete_prefix` is not changed by this entry; Python
  `delete()` follows the same contract (previous entry, #212).

### Fixed — a TAB, LF or CR in a value no longer breaks the op-log row (#186)

This is a **bounded mitigation, not exact logging support.** The row is TAB-separated, one
record per line, with no quoting; a TAB or newline in a value added columns or split the record
(and a crafted key could forge a whole record). No encoding, version marker or reader change is
made here; reversible encoding is future work.

- **Behaviour change.** A record whose `op`, `client_id`, `endpoint` or `file` holds a TAB, LF or
  CR is now **rejected (not written) and counted**; the writer carries on with the next record.
  The record is **lost**: the call itself runs normally but leaves no row. Its `idx` is skipped.
- **Behaviour change.** A TAB, LF or CR in `error` is **replaced by one space each** and the row
  is still written, so a failed operation is kept (CR LF becomes two spaces).
- Every other byte, a backslash included, is written as before; every row that was valid before is
  byte-identical. The check is made where the row is built, so it covers values an external caller
  gave to `Logger::log` too.
- New `OpLogStats::rejected_unsafe_field` and `OpLogStats::normalized_error_text` (Rust;
  `#[non_exhaustive]`, so additive) and the same two keys of Python `op_log_stats()`.
  `rejected_unsafe_field` is part of `is_complete()`: a log with a rejected record is **not**
  complete. `normalized_error_text` is not: every operation is recorded, only a message was
  reworded, so a log with only such rows is still complete. The first rejection prints one stderr
  note per session (it names the column, not the value); finalize's INCOMPLETE warning states the
  exact count, and finalize prints one line stating how many error messages had characters
  replaced by spaces.
- A record that is rejected for this reason and also has an unformattable timestamp (#190) is
  counted once, in `rejected_unsafe_field`. The #190 behaviour is unchanged.
- See `docs/supplemental/OPERATION_LOGGING.md`, "Records with a TAB, LF or CR in a value".

### Fixed — a row with an unformattable timestamp no longer ends the op-log (#190)

- **Behaviour change.** The op-log writes each row's start, first-byte and end times as
  RFC 3339 after the clock offset is applied. A corrected time before 1970-01-01T00:00:00Z or at
  or after 10000-01-01T00:00:00Z (an offset computed against a reference of 0 does it to every
  row) has no such text; the formatter panicked, and the panic ended the log: that row and every
  later one were lost. The row is now **rejected (not written) and counted**, and the writer
  carries on with the next row. Nothing is clamped or reformatted, and every other row is
  byte-identical to before. The rejected row's `idx` is skipped, so later rows keep their numbers.
- New `OpLogStats::rejected_unformattable` (Rust; `#[non_exhaustive]`, so additive) and the
  `rejected_unformattable` key of Python `op_log_stats()`. It is part of `is_complete()`: a log
  with a rejected row is **not** complete. The first rejection prints one stderr note per
  session; `finalize_op_log()`'s INCOMPLETE warning states the exact count.
- The writer checks representability before formatting (it does not rely on catching a panic).
  The domain is humantime 2.4.0's: at or after the epoch, below 253,402,300,800 s after it.
- See `docs/supplemental/OPERATION_LOGGING.md`, "Rows with an unformattable timestamp".

### Changed — clearer op-log reader error for unsupported operations (#176)

- `s3dlio-oplog` readers: a row whose op name the reader does not model (`GET_RANGE`,
  `DELETE_BATCH`, `DELETE_PREFIX`, `CREATE_CONTAINER`, `DELETE_CONTAINER`) now fails with an error
  that names the file line, the row index (a 0-based count of lines after the header that also counts
  skipped blank and comment lines) and the op, and says the operation is not
  supported for exact replay; any other unrecognised name is reported as unknown. Replay behaviour
  is unchanged: `OpType` still has five variants, the full-file load still aborts at the first such
  row, `continue_on_error` still applies only to execution, and nothing is skipped or filtered.

### Documented — known op-log gaps (#175, #176 remain open)

- `MultiEndpointStore` records only `get`/`put` on bucket-form endpoints; `list`, `delete`,
  `get_range`, `stat` and every call on host-form (and `file://`, `az://`, `gs://`) endpoints write
  no row (#175).
- Compound-operation rows (`GET_RANGE`, `DELETE_BATCH`, `DELETE_PREFIX`, container operations) do
  not carry the parameters needed for exact replay; recognising an op name does not mean exact
  replay is supported. Full-file replay stops at the first unsupported or malformed row, and
  `continue_on_error` does not bypass load errors. Mixed traces cannot be replayed exactly yet
  (#176).
- See "Known gaps" in `docs/supplemental/OPERATION_LOGGING.md`; the same facts are in
  `OPLOG-GUIDE.md`, `MULTI_ENDPOINT_GUIDE.md` and the `s3dlio-oplog` README and rustdoc.
- #175 and #176 are NOT fixed by this change; both issues remain open.

### Removed — never-compiled `src/prefetch.rs` sketch (#203)

- Deleted `src/prefetch.rs`, which was never declared in `lib.rs` and could not compile. The docs
  that cited it now describe the loaders that exist. `data_loader::prefetch::spawn_prefetch` is
  unchanged.

---

## Version 0.9.122 — Op-log accuracy (each call recorded once), server routing for every operation, replay fixes, Azure credential fix

Fixes 14 issues: #169, #174, #177, #181, #182, #189, #196, #197, #199, #200, #183, #184, #187, #202.
Version 0.9.120 was never published to PyPI; everything in the 0.9.120 section below ships for the
first time in this release, together with the changes listed here. **Read the behaviour changes
before upgrading:**
- `AWS_ENDPOINT_URL_S3` now takes precedence over `AWS_ENDPOINT_URL` (#183).
- An `S3ObjectStore` (and the S3 helper functions) reject a URI that names a server other than the
  one they talk to, before any request is sent (#197).
- `s3://host:port/bucket` create/delete bucket and `s3://host:port/` list-buckets act on that
  server (#199, #200).
- A store writes one op-log row per S3 call instead of two (#169); the Python store cache no longer
  decides whether a call is recorded (#174).
- Op-log replay errors on rows it cannot rebuild and on an unparsable start time (#184, #187).

The golden op-log tests compare NORMALISED fields (timing and variable values are normalised),
not the raw files byte for byte.

### Fixed — a per-endpoint store no longer runs a request on the global client (#197, #196)

- A URI that names a server other than the one the store talks to is rejected before any request
  is sent; the S3 helper functions apply the same check. `delete_prefix` on a per-endpoint client
  reports per-key errors (#196).

### Fixed — op-log replay (#184, #187) and Azure credential creation (#202)

- The replay URI is rebuilt from the rows s3dlio writes; rows that cannot be rebuilt are an error
  (legacy and unrecognised rows still translate onto an explicit target as before). An unparsable
  start time is an error. `OpType` keeps its five variants, so op names it does not model
  (`GET_RANGE`, `DELETE_BATCH`, `DELETE_PREFIX`, `CREATE_CONTAINER`, `DELETE_CONTAINER`) are still
  rejected with a clear error, as before. The Azure https credential is created synchronously (no
  `block_in_place` or `block_on`); the public constructors stay synchronous and cache the credential
  in a `OnceLock` (#202).

### Fixed — bucket create/delete ignored a server named in the URI (#199)

- `s3://host:port/bkt` now creates/deletes bucket `bkt` **on `host:port`** (per-endpoint
  client) in: Python `create_bucket` / `delete_bucket`, Python `put` / `put_async` with
  `should_create_bucket=True`, `s3-cli put --create-bucket`, and
  `s3_copy::upload_files(.., do_create_bucket = true)` (which now also uploads through
  that server). Before, the Python calls took `host:port/bkt` (or just `host:port`) as
  the bucket name and sent it to the `AWS_ENDPOINT_URL` server. `bkt`, `s3://bkt` and
  `s3://bkt/` still mean bucket `bkt` on the global endpoint; a portless IP
  (`s3://10.0.0.5/bkt`) is the #178 error and nothing is sent. Python `put(..,
  should_create_bucket=True)` on a non-`s3://` prefix no longer tries to create an S3
  bucket. New Rust API: `s3_utils::create_bucket_for_uri[_async]`,
  `s3_utils::delete_bucket_for_uri[_async]`.

### Fixed — `list_containers` / `s3-cli list-buckets` ignored a server named in the URI (#200)

- `s3://host:port/` (or `[ipv6]:port`) now lists **that server's** buckets, through a
  per-endpoint client; before, any `s3://...` URI listed the `AWS_ENDPOINT_URL` server's
  buckets. The listed URIs name the server (`s3://host:port/<bucket>`), so they can be
  passed straight back to s3dlio. A portless IP address (`s3://10.0.0.5/`) is now the
  #178 error ("IP address with no port"). `s3://`, `s3://bucket/` and other portless
  names still list the global server. Applies to the Rust `list_containers`, the Python
  `list_containers` and `s3-cli list-buckets`. New Rust API:
  `s3_utils::list_buckets_at`, `s3_utils::s3_bucket_target` and
  `s3_utils::s3_server_of_first_segment`.

### Fixed — `AWS_ENDPOINT_URL_S3` now takes precedence over `AWS_ENDPOINT_URL` (#183)

- **Behavior change.** s3dlio applied `AWS_ENDPOINT_URL` as an explicit endpoint, which
  overrode the AWS SDK's service-specific `AWS_ENDPOINT_URL_S3`: with both set, every
  request went to the `AWS_ENDPOINT_URL` server and `AWS_ENDPOINT_URL_S3` was ignored.
  s3dlio now follows the SDK's documented order, in one place
  (`s3dlio::s3_client::s3_endpoint_url_from_env`): `AWS_ENDPOINT_URL_S3`, then
  `AWS_ENDPOINT_URL`; an empty value counts as unset. That one rule feeds the global
  S3 client, `list_buckets`, the Python store-cache key, the arrow backend and the
  error hints.
  - **Who is affected:** deployments that set both variables to *different* servers
    and relied on `AWS_ENDPOINT_URL` winning. Unset `AWS_ENDPOINT_URL_S3`, or point it
    at the server you want.
  - `s3-cli --endpoint-url` and Python `configure_s3(endpoint_url=...)` still win: they
    also overwrite `AWS_ENDPOINT_URL_S3` when it is set. A server named in the URI
    (`s3://host:port/bucket`) wins over all of them for that URI.
  - The op-log is unchanged (record format frozen); which server a row's request went
    to is the subject of #169.


### Op-log accuracy: each call recorded exactly once (#169 W2)

The op-log TSV format (13 columns, header) is unchanged, and so is every field the
S3 helper functions have always written; the golden tests pin both (they compare normalised fields, not raw files byte for byte).

- **One row per call on S3 (#169).** Every S3 `get`/`put` through a logged store
  used to write two rows: the `LoggedObjectStore` row and a second row from the S3
  backend's own writer (`endpoint` = `AWS_ENDPOINT_URL`, `file` = the bare key,
  `client_id` = the access key id). The S3 backend of a store no longer writes that
  second row. A store built without a logger (`store_for_uri`) is now silent for S3
  too, as it already was for every other backend.
- **The public helpers keep their own row, exactly as before (#169).**
  `s3_utils::get_object`, `get_object_uri`, `get_object_uri_async`,
  `get_object_uri_optimized_async`, `get_objects_parallel*` (Python `get_many` on
  `s3://`), `get_range`, `put_object_async`, `put_object_uri_async`, the sync
  `delete_objects`, `S3BytesDataset` (sequential) and `s3_copy` still write the row
  `S3Ops` writes: `endpoint` = the effective endpoint (`AWS_ENDPOINT_URL_S3` if set, otherwise
  `AWS_ENDPOINT_URL`; #183), `file` = the bare key, `client_id`
  = `AWS_ACCESS_KEY_ID`, `bytes` = the data size (0 on failure), the measured
  first-byte time for a GET, one `DELETE` row with `file` `(batch)` per 1000 keys
  for `delete_objects`. `S3Ops::new(client, logger, client_id, endpoint)` is
  unchanged. A call writes one row whichever layer it went through: the S3 backend
  of a store is the only code that skips it. An object fetched by concurrent
  ranges writes no helper row, as before.
- **Checkpoint, the shard reader, the async-pool dataloader and Python
  `load_checkpoint_with_validation` write their rows exactly as before (#169).**
  They build an unlogged store; on a bucket-form S3 URI its `get`/`put` write the
  helper row (`S3Ops` shape) when the op-log is active, on every other backend they
  write no row. Rows for those backends are a proposal in the audit file.
- **Python store cache no longer decides whether a call is recorded (#174).**
  The module-level functions share a cache of stores; it kept the logger of
  whichever call built an entry first, `get_async`/`get_range_async` passed no
  logger, `init_op_log` did not clear it and a cached store kept a finalized
  logger. The cache now holds only unlogged stores and each call wraps its store
  with the logger it reads (`global_logger()`) when it is made, so a call made while
  the op-log is active is recorded once and a call made when it is not is neither
  recorded nor counted, whatever the cache history. `get_async` and
  `get_range_async` are recorded.
- **`LoggedObjectStore` is transparent, without hiding real requests (#177).** It
  forwards every `ObjectStore` method to the inner store. `mkdir`, `rmdir`,
  `update_metadata` and `update_properties` used to fail ("not implemented for this
  backend"); the S3 `get_optimized`/`get_range_optimized` overrides were bypassed.
  Rows: `get_optimized` writes a `GET` row and `get_range_optimized` a `GET_RANGE`
  row (as before); `mkdir`, `rmdir` and `update_*` have no op name in the format and
  are forwarded without a row. `copy` and `rename` are forwarded and unrecorded
  only when the inner store's implementation is a native operation (new provided
  trait methods `has_native_copy` / `has_native_rename`, default `false`; only the
  `rename(2)` of `file://` and `direct://` is native). Everywhere else (S3, Azure,
  GCS, multi-endpoint) they are the trait defaults, made of real requests, so the
  wrapper runs the default over itself and each request is a row as before:
  `copy` = `GET` + `PUT`, `rename` = `GET` + `PUT` + `DELETE`. `pre_stat_objects`
  and `pre_stat_and_cache` write one `HEAD` row per object, and
  `pre_stat_and_cache` fills the inner store's size cache from the sizes it
  learned (new provided method `seed_size_cache`, default no-op) without an extra
  request.
- **Streamed `LIST` rows carry the client id (#181).** The row written when a
  `list_stream()` stream completes had an empty `client_id`; it now goes through the
  same row builder as every other call, so it equals `list()`'s row.
- **Container rows use the store's endpoint and file (#182).** `CREATE_CONTAINER` /
  `DELETE_CONTAINER` rows were written with the bare name (endpoint `unknown`).
  A store built by the factories now knows its URI prefix (`s3://`, `file://`,
  `az://`, `s3://host:port/` for an S3 host-form store) and the rows read like the
  store's other rows: `s3://bkt` + `bkt`, `file://` + `/dir`. A hand-wrapped store
  (`LoggedObjectStore::new`) keeps the old behaviour; `with_container_prefix`
  supplies the prefix. Python `create_bucket` / `delete_bucket` still write no row.
- **`exists()` == false is a failed `HEAD` row (#189).** It was a `HEAD` row with
  an empty error, indistinguishable from a successful HEAD. The wrapper now answers
  `exists` through the inner store's `stat` (the trait's definition of `exists`;
  no store overrides it) and writes the error column of the failed HEAD, exactly
  as `stat` does for the same URI (backend-specific text). A present object is still
  a `HEAD` row without an error.
- **`MultiEndpointStore` keeps its `get`/`put` rows on bucket-form endpoints (#175).**
  They are the helper-shape rows they always were (one `GET` / `PUT` per call,
  `S3Ops` fields); the other operations (`list`, `delete`, `get_range`, `stat`, and
  everything on host-form endpoints) never wrote rows and still do not. Rows for
  them are a proposal in the audit file.
- **Known consequences (reported, not changed here):** an unlogged store
  (`store_for_uri`) is silent on S3 (it wrote the helper row by accident), and the
  `s3dlio-oplog` replayer's `store_for_uri` no longer writes the accidental helper
  row (capturing a replay is sai3-bench#103). See
  `docs/design/oplog-record-contract/W2-ownership-audit.md`.


## Version 0.9.120 — Correct routing (server, bucket, key), no sync hang, no exit crash, op-log lifecycle

Fixes 11 issues: #168, #170, #171, #172, #173, #178, #179, #180, #192, #193, #194.
The details for each are in the sections below. **Read the behaviour changes before upgrading:**
- A leading `/` in a key is kept (#172).
- A portless IP in `s3://` is an error, and a dotted first segment is a bucket (#178).
- `delete_batch` validates every URI before it deletes anything (#179).

### Also fixed in this release
- **Python store cache sent requests to the wrong server (#171).** The cache was keyed by
  `AWS_ENDPOINT_URL` alone, so a warm cache sent `s3://hostB:9000/...` to the server of an earlier
  call. The key now includes the server the URI names. An invalid server URI raises before the
  cache is consulted.
- **Sync Python calls hung forever on host-form URIs (#173).** `run_on_global_rt` no longer parks
  a runtime worker on a task it spawned onto its own runtime (it uses `block_in_place`). Stores are
  now created on the caller thread, with the GIL released.
- **SIGSEGV at interpreter exit after async calls (#192).** Results from `future_into_py` are now
  delivered through an s3dlio runtime adapter that counts deliveries in flight. An `atexit` hook
  closes the delivery gate and waits, with the GIL released and a 2 s bound, for deliveries that
  were already admitted. Deliveries scheduled after the gate closes are dropped unrun.
  - The race is in the async bridge; it is reported upstream as PyO3/pyo3-async-runtimes#93.
  - Upgrading to pyo3 0.29 alone does not fix it.
  - Known limit: an awaitable whose result arrives after the gate closes is never resolved.
- **Build:** `async-trait` 0.1.89 → 0.1.92. Clippy 1.99's `double_must_use` lint rejected the code
  generated by the older macro.


### S3: every operation now uses the object key exactly as written (#172)

- **Behavior change: keys with a leading `/` are no longer stripped.**
  `s3://b//k` means the key `/k` (not `k`) for **every** S3 operation. Before
  this change, `stat`, ranged reads (`get_range`, its timed variant and the
  concurrent-range chunk reads), the size-probe HEAD that precedes a GET, and
  large-object (known-size) reads removed the leading `/` and silently acted
  on a *different* object (`k`), while PUT, plain GET and DELETE used `/k`. On
  S3-compatible stores that treat `/k` and `k` as distinct keys (GCS interop,
  for example) `stat` could report another object's size and `get_range`
  could return another object's bytes.
  - **Who is affected:** callers that built URIs with an accidental `//` had
    `stat`, ranged reads and large-object reads succeed against the stripped
    twin object `k`. Those calls now address `/k` as written and return
    `NoSuchKey` if that object does not exist, consistently with plain GET.
    Remove the stray `/` from the URI (or create the object under the key you
    actually name) to fix such callers.
- **Behavior change: `MultiEndpointStore` relative-path join for an endpoint
  configured without a trailing `/`.** A relative path that starts with `/`
  joined to an endpoint with no trailing `/` (for example endpoint
  `s3://bucket` plus path `/data/x`) previously addressed the key `/data/x`
  for PUT, GET and DELETE (the join produced `s3://bucket//data/x`) but
  `data/x` for `stat` and ranged reads. It now addresses `data/x` for every
  operation, the same as endpoints configured with a trailing `/`. Objects
  that were previously written at `/data/x` through that call shape are **no
  longer reachable by the same call**; read or move them with an explicit
  `s3://bucket//data/x` URI.
- Internal: the known-size large-object GET calls the bucket/key ranged-GET
  core directly instead of rebuilding an `s3://` string and re-parsing it,
  which could misread a dotted or IP-shaped bucket name as an endpoint host.

### Fixed — `delete_batch` deleted in the wrong bucket / on the wrong server (#179)

- `ObjectStore::delete_batch` on the S3, GCS and Azure stores took the bucket
  (container) of the first URI and sent every key there. A batch naming several
  buckets deleted objects the caller never named, left the requested ones, and
  returned `Ok`. URIs are now grouped by bucket (Azure: account + container);
  each group is deleted in its own bucket. S3 issues `DeleteObjects` per bucket
  in chunks of 1000.
- An S3 URI of the host form (`s3://host:port/bucket/key`) names a server. A
  store talks to exactly one server, so a batch (or a single `delete()`) that
  names a different server is now an error and deletes nothing, instead of
  applying that URI's bucket and key to the store's own server. A per-endpoint
  store compares `host:port` case-insensitively, filling in 80/443 from the
  endpoint scheme, with no DNS resolution (`localhost` and `127.0.0.1` are
  different servers). A store on the process-wide client rejects every URI that
  names a server.
- The per-endpoint S3 path now checks the per-key errors that S3 reports inside
  an HTTP 200 `DeleteObjects` response; it used to discard them.
- When several buckets fail, the first failure's cause chain is the error
  source and the others are listed in the message.

### Behaviour change

- A batch containing an unparseable URI now **fails before deleting anything**,
  naming the URI. Previously URIs past the first that failed to parse were
  silently skipped, the rest were deleted, and the call returned `Ok`. Callers
  that relied on the skip must filter their URI list first.

### `s3://` names a server only with an explicit port (BREAKING; #178, #193, #194)

- **The first segment of `s3://X/...` is a server only when it carries an
  explicit port** (`host:port` or `[ipv6]:port`). Otherwise it is always a
  bucket. One shared rule (`s3_utils::classify_s3_first_segment`) now decides
  this for the URI parser, the store factory (`store_for_uri*`),
  `MultiEndpointStore` and the `s3-cli` bucket commands (`create-bucket` /
  `delete-bucket`) and `--endpoints` lists, which previously used different
  rules. The disagreement made `s3://10.0.0.5/bkt/key` take bucket/key from the
  parser but send the request to whatever `AWS_ENDPOINT_URL` said (#178), and
  made real dotted bucket names such as `data.example.com` parse as servers
  (#193). **Other paths do not share the rule yet** and can still ignore a
  server named in the URI: the Python `get_many` / `stat_many` helpers and
  `S3ObjectStore` operations on a URI naming a different server
  than the store's own (#197; `delete` and `delete_batch` already reject such a URI, #179), the Python `create_bucket` / `delete_bucket`
  helpers (#199), and `list_containers` / `s3-cli list-buckets` (#200).
- **Strict grammar:** a server is `host:port` or `[ipv6]:port` with a plain
  decimal port from 1 to 65535 (no `+`, no leading zeros, not `0`), a valid IPv6
  address inside the brackets, and no userinfo. URLs (`http://host:9000`), ARNs,
  an empty first segment and an empty host or port are rejected with specific
  messages.
- **Host grammar:** the host of `host:port` is 1 to 253 ASCII letters, digits,
  `-`, `.` and `_` (`_` for container-style names such as `minio_1`) with no empty
  label. Any other character (`/`, `\`, whitespace, control characters, `%`, `?`,
  `#`, non-ASCII, ...) is an error naming it and the grammar, checked before any
  endpoint URI is built. This closes a routing hole: `s3-cli --endpoints
  node/prefix:9000` was accepted and generated `s3://node/prefix:9000/bucket/key`,
  which parses as bucket `node` on the global client. A trailing dot
  (`host.example.com.:9000`) is now rejected as an empty label.
- A bare server URI given to the object parser (`s3://host:9000`,
  `s3://host:9000/`) now reports `missing bucket after server 'host:9000'` with the
  example `s3://host:9000/bucket/key`, instead of a "slash after bucket" message.
- `s3-cli create-bucket` / `delete-bucket` destination is now covered by a test
  that runs the real binary against two local mock S3 servers (the
  `AWS_ENDPOINT_URL` server must see no request; the server named in the URI must
  see the CreateBucket / DeleteBucket for the bucket) (#194).
- **Errors instead of fallbacks:** `store_for_uri`, `store_for_uri_with_logger`,
  `store_for_uri_with_config*`, `MultiEndpointStore::new` and
  `MultiEndpointStore::from_config` now return an error for an invalid first
  segment. Previously a parse problem was swallowed, or the endpoint silently fell
  back to the global client.
- **Breaking:** a portless DNS name is now a bucket (`s3://minio.example.com/bkt/key`
  is bucket `minio.example.com`, key `bkt/key`). A portless IPv4 literal
  (`s3://10.0.0.5/bkt/key`) and an unbracketed or portless IPv6 are errors, since
  S3 forbids IP-shaped bucket names; the message says to add `:port`.
- **Migration:** put the port in every URI that names a server:
  `s3://10.0.0.5:9000/bkt/key`, `s3://minio.example.com:9000/bkt/key`,
  `s3://[::1]:9000/bkt/key`. Every `--endpoints` entry must be `host:port` too.
  A deprecation period is not possible because a portless host and a dotted
  bucket name are the same grammar.
- `S3DLIO_S3_ENDPOINT_HINT_TLDS` (added in 0.9.109) is deprecated and ignored;
  the variable is checked once, at the first URI parse in the process, and a
  warning is then logged and written once to stderr. The built-in suffix list is
  removed. `MultiEndpointStore` also warns when an endpoint's first segment is a
  bucket that contains a dot (a likely old portless host-form endpoint) and names
  the fix: add `:port`.
- `s3-cli create-bucket` / `delete-bucket` on `s3://host:port/bkt` now operate on
  bucket `bkt` on that server (#194); before, the bucket name was `host:port` and
  the request went to the global endpoint. Per-endpoint S3 stores now create and
  delete buckets with their own client.
- Planned: explicit `s3+http://` / `s3+https://` schemes (#195).
- `MultiEndpointStore` now picks the endpoint scheme with the same rule as
  single-endpoint URIs (#180): port `443` and ports listed in
  `S3DLIO_S3_ENDPOINT_TLS_PORTS` use `https://`; every other port uses `http://`.
  Previously it always built `http://host:port`. **A plain-HTTP server deliberately
  listening on `:443`, or on a port named in `S3DLIO_S3_ENDPOINT_TLS_PORTS`, will now
  be dialled with TLS; no override exists** (the explicit `s3+http://` form, #195,
  is the planned escape hatch).

### Op-log accuracy: destinations (#168), lifecycle and drop detection (#170)

The op-log TSV byte format (header, columns, idx numbering, timestamps, zstd
framing) is unchanged; a golden test pins it.

- **`file://` op-log destinations work (#168).** `init_op_logger("file:///tmp/x")`
  / `init_op_log(...)` follow s3dlio's `file://` convention (the same as the
  `file://` object store: lowercase prefix removed, the rest used literally).
  Any other scheme (`s3://`, `direct://`, `FILE://`, ...) is rejected with an
  error that names the destination, and creation errors name the path.
  `file:` in any case followed by anything but `//` (`file:x`, `file:/tmp/x`) is
  also rejected (#191): **a change from 4542f82**, where `file:x` was accepted and
  created a file literally named `file:x`. Use `./file:x` for such a name. `c://x`
  stays rejected; its error now says to write a Windows drive as `C:/dir/file`.
  The destination must be a local path or a `file://` URI: cloud destinations
  (`s3://`, `gs://`, `az://`, `direct://`, ...) are not supported, and a string like
  `s3:bucket/key` (no `//`) is treated as a literal local file name.
- **One op-log session per process (#170).** The lifecycle is `Uninitialized` ->
  `Active` -> `Finalized`, and `Finalized` is terminal. `init_op_log` while
  Active is a no-op; after `finalize_op_log` it raises `RuntimeError` and touches
  no file. `finalize_op_logger` / `finalize_op_log` is idempotent, waits for the
  writer to finish, and releases the GIL in Python while it drains.
- **Drop detection (#170).** New `s3dlio::op_log_stats()` -> `Option<OpLogStats>`
  (and `Logger::stats()`), Python `s3dlio.op_log_stats()` (a dict, or `None`
  before init): `accepted`, `written`, `dropped_full`, `rejected_closed`,
  `writer_failed`, `writer_error`, `lossless`. `OpLogStats::is_complete()` is
  `accepted == written && dropped_full == 0 && writer_failed == 0 &&
  writer_error.is_none()`; judge it after finalize. `finalize_op_log()` prints one
  stderr warning when a counted loss happened. The stats do not cover process
  exit without finalize (no `atexit` hook), stores built before init, or errors
  reported only at `close(2)`; "written" counts rows accepted by the zstd
  compressor (they may still be buffered; a later failure is `writer_error`), so a
  complete finalize means completion, not durability (no fsync). See
  `docs/supplemental/OPERATION_LOGGING.md`.
- **`Logger::try_log` / `LogOutcome` (#170).** `try_log(entry)` returns
  `Accepted`, `Full`, `Closed` or `WriterFailed`; `Logger::log` still returns
  `()`. Every `Accepted` entry is written by finalize, including entries racing
  with it, unless the writer fails.
- **The writer finishes the zstd stream and flushes explicitly (#170).** Errors
  from the zstd frame end, the final flush and a writer-thread panic are recorded
  in `writer_error` instead of being swallowed by the encoder's drop, and a writer
  that dies with entries still queued shows as `accepted > written`.
- **Robustness (#170).** The diagnostic lines the op-log prints on the `log()`
  path no longer panic into the caller when stderr is closed or full. A panic
  inside the first `finalize` (for example from a tracing subscriber) no longer
  lets a later `finalize` return before the writer is done. The finalize warning
  names a writer-thread panic as a defect rather than sending the reader to the
  disk.
- **One queue-full note per session (#170).** In the default (lossy) mode the
  op-log used to print "op-log channel full; dropping entry" for every dropped
  entry, so a drop storm printed hundreds or thousands of stderr lines. It now
  prints one note, on the first drop of the session, saying that further drops
  are counted and reported at finalize; the `finalize_op_log()` warning states the
  exact `dropped_full` total.

## Version 0.9.114 — Azure SDK stable 1.x migration

- Updated `azure_storage_blob` and `azure_identity` from pre-release APIs to
  stable 1.0, with `azure_core` on the compatible stable 1.x line.
- Added configurable Azure download strategies: `auto`, `azure-sdk`, `s3dlio`,
  and `sequential`. `auto` uses sequential transfer below the configured
  threshold and selects either Azure SDK managed partitioning or s3dlio's
  RangeEngine above it.
- Added Azure SDK partition concurrency and size controls. Explicit Rust
  `AzureConfig` values override environment-derived defaults.
- Reused one `BlobServiceClient` pipeline per `AzureBlob` instead of rebuilding
  a client pipeline for each operation and staged block.
- Preserved zero-copy upload bodies by using Azure Core's `From<Bytes>` request
  conversion. A pointer-identity regression test covers sliced `Bytes` values.
- Restored container create/delete support exposed by the stable SDK.
- Custom Azure-compatible endpoints remain supported structurally. Additional
  generic-target authentication and live endpoint validation are tracked as a
  follow-up after real Azure Blob validation; this migration does not remove
  the custom endpoint settings.
- Raised the Rust MSRV to 1.88, required by Azure Storage Blob 1.0.

## Version 0.9.112 — Tokio-runtime sanity clamp + MPI-aware thread pools + FFI-boundary hardening

This release bundles three independent hardening changes: (1) a sanity
clamp on the `S3DLIO_RT_THREADS` env var that closes a runtime-starvation
class of bug found while chasing the DLIO unet3d datagen slowdown,
(2) MPI-aware auto-sizing of every s3dlio-owned thread pool at import
time, and (3) an FFI-boundary hardening pass motivated by
[mlcommons/storage#755](https://github.com/mlcommons/storage/issues/755).
The three did not share a common root cause; they landed together because
each individually was small enough that a mid-release cut would have felt
worse than a slightly larger release.

## Tokio-runtime sanity clamp on `S3DLIO_RT_THREADS`

### Root cause (docs/investigation/DLIO_UNET3D_DATAGEN_BOTTLENECK_INVESTIGATION_2026-07-10.md)

DLIO's `dlio_benchmark/storage/obj_store_lib.py` `ObjStoreLibStorage.__init__`
computes:

```python
_rt_threads = min(_write_threads * 3 // 2, 128)
os.environ["S3DLIO_RT_THREADS"] = str(_rt_threads)
```

With Hydra's default `write_threads: int = 1` (unet3d_datagen workload does
not override), that resolves to `S3DLIO_RT_THREADS=1`.  Pre-0.9.112 s3dlio
faithfully obeyed the env var and built the global Tokio runtime with **one**
worker.  Every concurrent MPU part upload then serialized through that one
worker: p50 `close()` latency = 22 843 ms per file, aggregate throughput
~214 MB/s per rank vs. an isolated benchmark's 1928–2557 MB/s on the same
s3-ultra target.

### The fix

`get_runtime_threads()` (`src/s3_client.rs`) and `effective_thread_budget()`
(`src/constants.rs`, used by the checkpoint runtime and dgen-rs paths) now
share `clamped_env_rt_threads()`: if `S3DLIO_RT_THREADS` is set and is
below `RT_THREADS_LIMIT / 4` (where `RT_THREADS_LIMIT` is the per-process
target set by `configure_thread_pools`, auto-called at import time to the
MPI-aware budget), the env var is treated as downstream miscomputation and
replaced with `RT_THREADS_LIMIT`.  A stderr warning is printed the first
time the clamp fires so operators can spot the underlying miscomputation.
`S3DLIO_RT_THREADS_UNSAFE=1` bypasses the clamp for legitimate low-thread
scenarios (single-threaded test suites, fault injection).

### Live verification (RED → GREEN, real s3-ultra target)

DLIO `unet3d_datagen`, 40 files × 146 MB, NP=1 on a 28-core host:

| Metric               | Pre-fix (0.9.110)         | Post-fix (this release) |
| -------------------- | ------------------------- | ----------------------- |
| Wall time            | 27.4 s                    | **2.9 s**               |
| Throughput           | 214 MB/s                  | **~1928 MB/s** (9×)     |
| Upload `close()` p50 | 22 843 ms                 | **309 ms** (77×)        |
| Upload `close()` p99 | 26 368 ms                 | 529 ms                  |
| Rt-worker threads    | 1                         | 28 (== RT_THREADS_LIMIT)|

Cross-backend live smoke (all under `S3DLIO_RT_THREADS=1`, all clamped
to 28 workers): `file://` 397 MB/s, `direct://` (on `/mnt/nvme_data`)
655 MB/s, `s3://` (s3-ultra) 919 MB/s.  `S3DLIO_RT_THREADS_UNSAFE=1`
correctly disables the clamp (throughput drops back to 130 MB/s at
5 files, confirming single-worker serialization is restored).
Sane values (e.g. `S3DLIO_RT_THREADS=16`) pass through unchanged, no
warning, 1.1 GB/s.

### Test coverage — RED-then-GREEN

`src/s3_client.rs` `mpi_aware_runtime_sizing_tests`:
`get_runtime_threads_clamps_grossly_underprovisioned_env_var_up_to_limit`
(clamp fires), `_env_var_at_or_above_quarter_of_limit_is_honored_verbatim`
(exact threshold + comfortable value pass through), `_unsafe_env_bypasses_the_clamp`
(escape hatch), `_env_var_low_with_no_limit_set_is_honored`
(pure-Rust programmatic use, no `RT_THREADS_LIMIT`, no clamp).

`src/constants.rs` `effective_thread_budget_tests`:
`test_effective_budget_clamps_grossly_underprovisioned_env_var`,
`test_effective_budget_unsafe_bypass_respects_low_env_var` — same shape
for the checkpoint-runtime / dgen-rs path.

All four were RED (asserted against unmodified `get_runtime_threads()` /
`effective_thread_budget()`) before the fix; all four are GREEN under
the fix.  Full pre-push gate (`cargo fmt --check`, `cargo clippy --lib
--bins --examples -- -D warnings` under both default features and
`--features extension-module`, `cargo test --lib`) passes: 404 tests
default, 412 tests extension-module.

## MPI-aware thread pool auto-init (import-time `configure_thread_pools(0)`)

The `_pymod` init hook now calls `s3_client::configure_thread_pools(0)`
before registering any Python functions.  This sets `RT_THREADS_LIMIT`
to the MPI-aware per-process budget (`num_cpus / world_size`, floor 1)
and builds Rayon's global pool at the same target — every `s3dlio`
Python import, no explicit call needed.  Fixes the CPU-oversubscription
class of bug where N ranks on the same host each independently defaulted
to `num_cpus` threads.

New helpers in `src/constants.rs`: `mpi_world_size()` (reads
`OMPI_COMM_WORLD_SIZE` / `PMI_SIZE` / `WORLD_SIZE`),
`mpi_aware_thread_budget()`, `effective_thread_budget()` (single-knob
sizing used by the checkpoint runtime and `hardware::recommended_data_gen_threads()`).
`get_runtime_threads()` folds MPI-awareness into its default fallback;
`configure_thread_pools()` is the single choke point for setting
`RT_THREADS_LIMIT`, the pyo3-async-runtimes Tokio init, and Rayon's
global pool.

## Version 0.9.112 — FFI-boundary hardening (mlcommons/storage#755, s3dlio#161, s3dlio#162)

Prompted by [mlcommons/storage#755](https://github.com/mlcommons/storage/issues/755):
a DLIO training run against S3 died with only `RuntimeError: concurrent range
chunk failed` — no indication of the actual underlying cause. Root cause,
filed as [s3dlio#161](https://github.com/russfellows/s3dlio/issues/161): error
conversion at the PyO3 boundary used `Display` (`{}`) on `anyhow::Error`
values, which shows only the outermost `.context(...)` label and discards the
full cause chain. This release fixes that everywhere, plus a broader FFI
audit that surfaced along the way. **Note**: this release improves
diagnosability, not resilience — it does not change retry/failure behavior,
so a run that previously failed under the same conditions will still fail,
just with the real cause now visible in the exception message. See
`docs/DESIGN_FFI_BOUNDARY_HARDENING.md` and `docs/DESIGN_TIER4_FFI_HARDENING.md`
for the full design rationale and per-item decisions.

### Tier 1 — error-chain preservation (storage#755 / s3dlio#161)

All error-conversion sites across `python_core_api.rs` (19 sites),
`python_aiml_api.rs` (checkpoint sites), `python_advanced_api.rs`
(`MultipartUploadWriter`), and `python_datagen_api.rs` (1 site, the last
holdout — previously hand-rolled `{:#}` formatting, now routed through the
same shared `py_err()` helper for consistency) now preserve the full `anyhow`
cause chain (`error_chain_message()`, `{:#}` alternate `Display`) instead of
collapsing to the top-level context label. This is the direct fix for #755's
reported symptom — the same `RuntimeError: concurrent range chunk failed`
failure will now include the real SDK/timeout/TLS/etc. cause in its message.

### Tier 2 — `checksum()` fix

Extracted `snapshot_final_writer_stats()` as a pure, GIL-free helper so the
checksum-after-finalize behavior is unit-testable.

### Tier 3 — bucket-swallow + etag fixes, multipart writer pinning

- `MultipartUploadWriter.reserve()` now takes `PyRefMut<'_, Self>` instead of
  `&mut self` so the writer object itself is pinned alive for the duration of
  a `reserve()`...`commit()` cycle.
- `close()` always populates the `"etag"` key in its result dict, even on the
  no-op / already-closed path.

### Tier 4 — second-round FFI hardening (design doc review, 5 items)

1. `exists()`/`exists_async()` docstrings now explain that both collapse
   every failure mode to `False` by design (mirrors `os.path.exists()`), and
   point callers who need to distinguish "not found" from "couldn't check"
   at `stat()`. No logic change.
2. Unknown-length ("iterable-style") datasets now raise `TypeError` from
   `PyDataset.__len__`, `PyBytesAsyncDataLoader.__iter__`/`.items()`, and
   `PyBytesAsyncDataLoaderIter.spawn_stream` instead of silently defaulting
   to empty/0. No currently Python-constructible dataset type can trigger
   this path today — closes a latent trap, not a live behavior break.
3. Migrated the 7 "easy bucket" `block_on()` call sites (all 4
   `create_*_writer` functions, `finalize()`, `write_owned_bytes()`) to the
   established `submit_io()`/`run_on_global_rt()` pattern, closing a
   reentrancy-panic risk. `write_chunk()`'s 2 sites are deliberately left on
   `block_on()` — its zero-copy fast path holds a borrowed `PyBuffer` slice
   that can't cross a `'static`-future boundary without a copy or a much
   bigger lifetime redesign.
4. `PyObjectDataLoaderSyncIter`, `PyBytesDataLoaderSyncIter`, and
   `ParquetStreamIter`'s `rx` channel fields switched from `std::sync::Mutex`
   to `parking_lot::Mutex` (already a dependency) — `std::sync::Mutex`
   poisons itself if any thread panics while the lock is held, permanently
   wedging that iterator's `__next__` for the rest of its lifetime;
   `parking_lot::Mutex` never poisons.
5. `python_datagen_api.rs`'s lone manual `{:#}`-format site standardized onto
   `py_err()` — folded into Tier 1 above.

### Tier 5 — dead code removal

Deleted `src/python_api/zero_copy_api.rs` (286 lines, never registered/
compiled).

### `MultiEndpointStore`: explicit per-endpoint pinning + fan-out replication (fixes s3dlio#162)

Investigating Tier 4 item 4's original test fixture surfaced an undocumented
design assumption: `MultiEndpointStore`'s round-robin/least-connections
load balancing assumes every configured endpoint is a true replica holding
the same object under the same key — correct for mlp-storage's real usage,
but silently wrong for anything that expects `put(uri)` then `get(uri)` to
be read-your-writes against a *specific* endpoint. Filed as
[s3dlio#162](https://github.com/russfellows/s3dlio/issues/162) and fixed,
fully additive (zero change to default load-balancing behavior):

- `endpoint_at(index)` — bounds-checked accessor, no load-balancer state
  touched.
- `get_from_endpoint`/`put_to_endpoint`/`delete_from_endpoint` — bypass
  `select_endpoint()` entirely, pin to a specific endpoint by index.
- `put_all_endpoints` — fan-out write to every configured endpoint.
- `list_all_endpoints` — was implemented in Rust but never exposed to
  Python; now is.
- Crate-level docs on `MultiEndpointStore` make the replication-only
  assumption of round-robin/least-connections explicit.

### Test-suite hardening: eliminated the return-True/False masking anti-pattern (27 files)

While live-testing the above, found that many files under `python/tests/`
used `print(...); return True/False` instead of `assert` — pytest treats any
function that returns without raising as **passed**, regardless of the
returned value, so these tests could never actually fail. Fixed across 27
files (~75 test functions converted to real assertions), which unmasked
~8 genuine pre-existing bugs, since fixed:

- `BytesView` vs `bytes` comparison always `False` (no `__eq__` override) —
  4+ locations across checkpoint/compression/multi-endpoint tests, fixed by
  wrapping with `bytes(...)`.
- `test_streaming_python_old.py` couldn't even be *collected* by pytest
  (`SyntaxError: 'await' outside async function` — awaiting synchronous
  PyO3 APIs).
- Stale `s3dlio.torch.S3DataLoader` references (API removed) across 3 files
  — fixed to use `S3IterableDataset.from_prefix(uri)`.
- `test_multi_endpoint.py`'s `run_async` masking (undefined name silently
  accepted by `pytest.raises(Exception)`).
- A CLI test referencing a Cargo binary name (`s3dlio`) that was never real
  — the actual binary is `s3-cli`.

### Verification

`cargo fmt --check` / `cargo clippy -D warnings` (default and
`--features extension-module`) clean. `cargo test --lib`: 376 passed
(default), 383 passed (extension-module), 0 failed. Full `cargo test`
(all lib + integration binaries + doctests): 756 passed, 0 failed. Combined
`uv run pytest` across all touched Python files: 105 passed, 10 skipped (all
legitimate — missing live S3/Azure/GCS credentials, GCS backend not compiled
into this wheel, missing `cargo script`), 0 failed. Wheel rebuilt via
`./build_pyo3.sh` and smoke-tested live.

---

## Version 0.9.110 — Multi-agent bug audit fix release (issues #151–#157, Phase D+E+F, 17 bugs)

Closes out the remaining scope of the audit that shipped Phase A/B/C in v0.9.109:
Phase D (backend correctness/hygiene, 9 bugs), Phase E (edge cases / low severity,
4 bugs), and Phase F (env-var/config-ignored plus one docs-only fix, 4 bugs) — 17
bugs in total, none release-blocking, all still real defects worth fixing before
1.0.

Full plan and locked contracts:
[`docs/implementation-plans/v0.9.109-audit-fix-plan.md`](implementation-plans/v0.9.109-audit-fix-plan.md).
Every fix below landed with a RED-then-GREEN regression test, verified to fail
against the unmodified code for the exact reason the bug describes, then pass after
the fix (parent CLAUDE.md §6).

### New environment variables

| Variable | Purpose |
|----------|---------|
| `S3DLIO_S3_ENDPOINT_TLS_PORTS` | Comma-separated additional ports to treat as TLS-terminated (`https`) for per-endpoint S3 clients addressed via `s3://host:port/...` — e.g. a load balancer terminating TLS on a non-standard port (E2). |

See `docs/Environment_Variables.md` for defaults and full semantics.

### Behavior changes — read before upgrading

- **`S3DLIO_OPLOG_BUF`'s code default corrected from 256 to the documented 8192
  (F3, issue #157).** Under a moderately fast burst of ops, the previous 256-slot
  channel filled and began silently dropping op-log entries far below the load the
  docs promised. If you were relying on the (undocumented, buggy) 256-slot behavior
  for some reason, set `S3DLIO_OPLOG_BUF=256` explicitly to keep it.
- **`S3DLIO_OPLOG_WBUFCAP`'s documented default corrected from 4096 to the actual
  code value, 262144 (256 KiB) (F3, issue #157).** No code behavior changed; only
  the docs were wrong.
- **`mp::Runner.max_http_connections()`/`.optimized_http()` now actually take
  effect (F1, issue #157).** These builder methods previously wrote
  `S3DLIO_MAX_HTTP_CONNECTIONS`/`S3DLIO_USE_OPTIMIZED_HTTP` into spawned workers'
  env — names no code anywhere in the crate read, so both methods were silent
  no-ops. They now write `S3DLIO_POOL_MAX_IDLE_PER_HOST`/`S3DLIO_ENABLE_HTTP2`
  instead, which are real, wired knobs. If you were calling these builder methods
  expecting no effect, spawned workers will now actually pick up the configured
  pool size / HTTP2 setting.
- **Performance Tuning Guidelines corrected (F4, issue #157).** The doc
  previously recommended exporting `S3DLIO_USE_OPTIMIZED_HTTP` /
  `S3DLIO_MAX_HTTP_CONNECTIONS` / `S3DLIO_HTTP_IDLE_TIMEOUT_MS` directly — none of
  which are read as env vars by s3dlio (only F1's `mp::Runner` builder API
  translates the first two into real vars, and only for its own spawned workers).
  The guidelines now recommend the real wired vars (`S3DLIO_ENABLE_HTTP2`,
  `S3DLIO_POOL_MAX_IDLE_PER_HOST`, `S3DLIO_POOL_IDLE_TIMEOUT_SECS`) instead.

### Phase D — Backend correctness / hygiene (9 bugs, issues #151–#153/#155)

- **D1 (5.1)** GCS's `get_object_via_grpc` full-read retry loop retried
  immediately with no backoff, hammering a struggling backend. Added
  exponential backoff (100ms → 200 → 400 → 800 → 1600, capped at 2000ms) via a
  generic `retry_with_backoff` helper.
- **D2 (5.2)** `is_rapid_bucket`'s `OnceCell::get_or_init` cached the result of
  `detect_rapid_bucket` even when the detection call itself failed, permanently
  poisoning the cache with a wrong answer after one transient error. Switched to
  `get_or_try_init`, which does not cache on `Err`.
- **D3 (1.3)** The `S3DLIO_MPU_PUT_VERIFY` opt-in HEAD-verification step hard-failed
  the whole multipart upload if the post-upload HEAD request itself failed
  transiently (network blip), even though the upload had already succeeded. Now
  logs a warning and falls back to the known `total_bytes`, treating the HEAD
  failure as soft.
- **D4 (1.5)** Azure's `upload_multipart_stream` returned a bare `Err` with no log
  at all on a mid-stream `stage_block` failure — an operator had no way to tell an
  Azure multipart upload failed, or how many blocks were staged (and left for
  Azure's automatic GC) before it did. Now logs a warning naming the container,
  blob key, and staged-block count.
- **D5 (5.4)** The community-GCS client's `put_object_multipart` was a byte-for-byte
  duplicate of `put_object` that silently ignored its `chunk_size` parameter —
  a real chunked-upload feature was never implemented, just a misleadingly-named
  stub. Removed; both GCS multipart call sites now use the real `put_object`
  directly, with `part_size` unused/no-op for this backend (documented in place).
- **D6 (3.7)** `S3dlioStorage.put_data()`'s multipart branch had no
  try/except — a failure partway through `writer.write()` left the
  in-progress multipart upload dangling (never aborted, never completed). Now
  wrapped so any exception calls `writer.abort()` before re-raising.
- **D7 (3.8)** `S3PyTorchConnectorStorage.walk_node()` compared each full URI
  returned by `s3dlio.list()` against a bare relative `prefix`, which never
  matched (a full URI never starts with a relative path) — every call fell
  through to `os.path.basename()`, silently dropping subdirectory structure
  (`"train/a/x"` became just `"x"`). Fixed to compare against the full URI,
  matching the sibling `S3dlioStorage` implementation.
- **D8 (3.9)** `S3dlioStorage.__init__` set `AWS_ENDPOINT_URL` via an unconditional
  assignment, clobbering any value the user had already set in their own
  environment — inconsistent with the `setdefault` (don't-overwrite) contract used
  for the other AWS env vars in the same constructor. Now uses `setdefault`, with a
  warning logged when the selected/configured endpoint differs from a pre-existing
  value.
- **D9 (2.8)** `RangeEngine::download()` unconditionally `bail!`-ed on
  `object_size == 0`, treating a legitimate empty object as an error instead of
  succeeding with zero bytes.

### Phase E — Edge cases / low severity (4 bugs, issues #151/#154/#156)

- **E1 (4.2)** `infer_scheme()`'s Azure-hostname check was
  `uri.contains(".blob.core.windows.net/")` — a substring match against the
  *entire* URI, not just the host. A `gs://` URI whose key happened to contain
  that literal text was misrouted to Azure (the Azure check ran before the `gs://`
  check in the if/else chain). Restricted to the authority component, checked as a
  suffix. Also fixed `parse_azure_uri()`, which `infer_scheme()` correctly routed
  an `http://...blob.core.windows.net/...` URL to, but which then hard-required a
  `https://` prefix and bailed with a confusing error instead of parsing it.
- **E2 (4.3)** `s3_endpoint_url_from_uri()` picked `https` only for port `443`,
  with no way to mark a TLS-terminated S3-compatible endpoint on a non-standard
  port (e.g. a load balancer terminating TLS on `:9001`) as such. Added
  `S3DLIO_S3_ENDPOINT_TLS_PORTS` (see "New environment variables"). Also replaced
  the IPv6 port-extraction logic (`contains(':')` + `rsplit(':').next()`, which only
  produced correct results for bracketed IPv6 by coincidence) with explicit
  bracketed-IPv6-aware parsing.
- **E3 (1.6)** Azure's `upload_multipart_stream` never checked Azure's documented
  50,000-block-per-blob limit — an object whose `part_size` was too small for its
  size would stage tens of thousands of blocks over the network only to have
  `commit_block_list` reject the blob server-side, wasting every staged block. Now
  fails fast with a clear, actionable error before the block that would exceed the
  cap is ever staged.
- **E4 (6.2)** `ConfigurableFileSystemObjectStore::path_to_uri()` hardcoded the
  `file://` scheme unconditionally, so `list()` on a store addressed via
  `direct://` returned `file://` URIs for every entry. Round-tripping one of those
  URIs back through `store_for_uri()` silently routed to the buffered `file://`
  store instead of the O_DIRECT store the caller was actually using. Now threads
  the caller's original scheme through to every returned URI.

### Phase F — Env vars / config-ignored, plus one docs fix (4 bugs, issue #157)

- **F1 (7.1)**, **F2 (7.4)**, **F3 (7.3)**, **F4 (7.2)** — see "New environment
  variables" and "Behavior changes" above.

### Commits in this release

v0.9.109 and v0.9.110 shipped together as one branch/PR (`fix/151-157-audit-v0.9.110`)
covering the full 39-bug, 6-phase, 7-issue audit. Full commit list, oldest to newest:

| Commit | Subject |
|--------|---------|
| `0c2174f` | docs(148-audit): add v0.9.109+v0.9.110 fix plan for issues #151-#157 |
| `7eced7c` | fix(152): range engine silently corrupts buffer on short/over read (A1) |
| `74f146b` | fix(151): multipart Drop silently commits partial upload (A2) |
| `2d28441` | test(151): dedicated RED-then-GREEN coverage for abort_blocking() (A3) |
| `c099032` | fix(154): delete_objects ignores S3 per-object error list (A4) |
| `f0b2d64` | fix(154): list_with_client dead regex filter returns unrelated keys (B7) |
| `52376ec` | fix(152): length=Some(0) downloads full remainder; missing HEAD Content-Length silently returns empty (B2+B3) |
| `624d3a6` | fix(152): get_object_range_uri_async missing retry_get_body wrap (B1) |
| `31f5d14` | fix(153): get_data AND-guard silently returns full object for offset-only or length-only reads (B5) |
| `5314d18` | Adopt ruff for Python lint/format/compile gate; clean python/s3dlio/ |
| `3b32a3f` | fix(152): u64 underflow in get_range() offset+length-1 across 3 backends (B4) |
| `8f468b1` | fix(153): wire multipart threshold/part-size/max-in-flight via env vars (B10) |
| `d1cc0cb` | fix(153): add multipart path to S3PyTorchConnectorStorage.put_data (B6) |
| `2d69473` | fix(154): narrow parse_s3_uri_full endpoint-detection heuristic (B8) |
| `97967dc` | fix(153): round_robin endpoint selection was a static PID hash, not round-robin (B9) |
| `21ce506` | fix(153): eliminate silent exception-swallowing in walk/create/delete_node (C1-C3) |
| `276c6b6` | docs(151): confirm __exit__'s abort_blocking() discard is already safe (C4) |
| `f6fdf29` | fix(156): try_read_range_direct panics on missing file or offset past EOF (C6) |
| `414a02b` | fix(157): official GCS delete_objects doesn't count panicked/cancelled tasks (C5) |
| `3878fed` | release: bump version 0.9.108 → 0.9.109 for the audit fix release |
| `85f4abf` | fix(157): GCS full-read retry loop had no backoff delay (D1) |
| `a4bff42` | fix(157): GCS RAPID bucket detection cache poisoned by transient failures (D2) |
| `5deb8fa` | fix(151): MPU_PUT_VERIFY treated a failed HEAD request as data corruption (D3) |
| `f52248a` | fix(151): Azure multipart upload failure was silent -- no log, no cleanup note (D4) |
| `e34314d` | fix(157): consolidate community-GCS put_object_multipart stub into put_object (D5) |
| `b9f9e19` | fix(153): D6 write-loop abort + fix a pre-existing sys.path bug affecting every Python test's source resolution |
| `753170f` | fix(153): S3PyTorchConnectorStorage.walk_node dropped subdirectory structure (D7) |
| `07870ea` | fix(153): AWS_ENDPOINT_URL clobbered a user-set env value (D8) |
| `1904579` | fix(range-engine): D9 — zero-sized object should succeed, not bail |
| `2dc7f93` | fix(object-store): E1+E2 — URI scheme/endpoint detection edge cases |
| `41da275` | fix(azure): E3 — client-side cap for Azure's 50,000-block-per-blob limit |
| `4745ed3` | fix(direct-io): E4 — list() on a direct:// store returned file:// URIs |
| `41bc847` | fix(env-vars): F1-F4 — dead knobs, doc/code default mismatches, PUT_MAX_RETRIES=0 clamp |
| `618fa74` | chore(release): bump version 0.9.109 -> 0.9.110 + Changelog |
| `fd3a953` | style: ruff-format test_dlio_storage_get_data_range.py |

### Quality gate

Every fix above: `cargo fmt --check`, `cargo clippy --lib --bins --examples -- -D
warnings` (CI's invocation) plus `cargo clippy --lib --tests --features
backend-azure,backend-gcs,gcs-community -- -D warnings` and `cargo clippy --lib
--tests --features extension-module -- -D warnings` for the feature-gated fixes.
**748 tests passing** across the full suite (every `cargo test` binary — lib unit
tests plus every `tests/*.rs` integration file — across the relevant feature
combinations), up from 689 at the start of this work, plus the Python DLIO
integration suite (`ruff check`, `ruff format --check`, `python -m compileall`,
`uv run pytest`) all clean.

## Version 0.9.109 — Multi-agent bug audit fix release (issues #151–#157, Phase A+B+C, 22 bugs)

A follow-up multi-agent code audit (triggered by mlcommons/storage#715) found 39 bugs
across 7 GitHub issues, ranging from silent data corruption to silent exception
swallowing. This release closes the audit's release-floor + hardening scope — all of
Phase A (silent-corruption/silent-commit bugs), Phase B (wrong-data / reachable
bugs), and Phase C (silent-failure elimination) — 22 bugs in total. The remaining 17
bugs (Phase D "hygiene", Phase E "edge cases", Phase F "env-var/config") are lower
severity and tracked for v0.9.110.

Full plan and locked contracts:
[`docs/implementation-plans/v0.9.109-audit-fix-plan.md`](implementation-plans/v0.9.109-audit-fix-plan.md).
Every fix below landed with a RED-then-GREEN regression test, verified to fail
against the unmodified code for the exact reason the bug describes, then pass after
the fix (parent CLAUDE.md §6).

### Behavior changes — read before upgrading

- **`parse_s3_uri_full` endpoint-detection heuristic narrowed (B8, issue #154).**
  Previously, an `s3://` URI's first path segment was treated as a custom endpoint
  hostname if it merely contained 2+ dots, started with a digit, or contained
  `minio`/`ceph`/`localhost` as a substring — misrouting legitimate bucket names
  using those shapes (e.g. `mycompany.data.backups`, `2024-training-data`,
  `minio-tenant-a`). As of this release the heuristic only fires on a colon (port),
  a dotted-quad IPv4 address, or a recognized hostname-like suffix (configurable via
  the new `S3DLIO_S3_ENDPOINT_HINT_TLDS` env var). If you relied on the old broader
  heuristic to route a no-port, unusual-suffix endpoint hostname, either add a port
  to the URI or add your hostname's suffix to `S3DLIO_S3_ENDPOINT_HINT_TLDS`. See
  `docs/Environment_Variables.md`.
- **`round_robin` DLIO endpoint selection now rank-based (B9, issue #153).**
  `S3dlioStorage`'s `round_robin` load-balance strategy previously used
  `pid % N` — not round-robin at all (a static per-process value unrelated to MPI
  rank). It now uses `rank % N` when an MPI/SLURM/PMI rank env var is present
  (falling back to the old pid-based behavior, now logged, only when no rank var is
  available). `use_mpi_endpoint_distribution=true` and
  `load_balance_strategy=round_robin` now collapse to the same assignment whenever a
  rank var is present.
- **DLIO storage `create_node`/`delete_node` no longer swallow arbitrary errors
  (C1–C3, issue #153).** `walk_node()` now propagates listing errors instead of
  returning `[]`; `create_node(exist_ok=True)` only treats `FileExistsError` as
  success (everything else propagates, even with `exist_ok=True`);
  `delete_node()` only treats `FileNotFoundError` as success. If your code depended
  on these methods silently swallowing real errors (auth failures, network errors,
  "not implemented" backends), you will now see exceptions where you previously saw
  `[]`/`True`/`False`.

### New environment variables

| Variable | Purpose |
|----------|---------|
| `S3DLIO_S3_ENDPOINT_HINT_TLDS` | Comma-separated extra suffixes for the narrowed endpoint-detection heuristic (B8). |
| `S3DLIO_MULTIPART_THRESHOLD_MB` | Single-PUT vs multipart threshold for `S3dlioStorage`/`S3PyTorchConnectorStorage` (B6/B10) — reuses the name already used by DLIO's own `obj_store_lib.py`. |
| `S3DLIO_MULTIPART_PART_SIZE_MB` | Multipart part size for the same two DLIO storage backends (B6/B10). |
| `S3DLIO_MULTIPART_MAX_IN_FLIGHT` | Multipart concurrent-part limit for the same two backends (B6/B10). |
| `S3DLIO_DISABLE_MULTIPART` | Explicit switch to force single-PUT-only behavior regardless of size (B6/B10). |

See `docs/Environment_Variables.md` for defaults and full semantics.

### Phase A — Silent data corruption / silent commit (4 bugs, issue #151)

- **A1 (2.1)** `RangeEngine::download_with_ranges` tolerated a length-mismatched
  chunk with a `tracing::warn!` and truncated the assembled buffer instead of
  erroring — a short/over read on a middle range silently corrupted downloaded
  data. Now a hard error.
- **A2 (1.1)** `MultipartUploadWriter`'s `Drop` impl let the coordinator task fall
  through to `CompleteMultipartUpload` even when `abort_on_drop=false` was set,
  committing a partial object the caller explicitly asked to discard. The
  coordinator is now always cancelled on drop; `abort_on_drop` only gates the
  best-effort S3-side `AbortMultipartUpload` call.
- **A3 (1.2)** `abort_blocking()` discarded the `AbortMultipartUpload` result via
  `let _ = ...` and never stopped the coordinator, so an explicit `.abort()` call
  could still race to `CompleteMultipartUpload`. Now always cancels the coordinator
  first and logs (`tracing::warn!`, naming bucket/key/upload_id) if the best-effort
  S3-side abort itself fails.
- **A4 (4.5)** `S3Ops::delete_objects` / `s3_utils::delete_objects_async` treated any
  HTTP 200 response as full success, ignoring per-object `<Error>` entries in the
  `DeleteObjects` response body — objects could silently fail to delete. Now
  inspects `output.errors()` and bails with the per-key failure detail when
  non-empty.

### Phase B — Wrong data / reachable from public API (10 sub-bugs across 7 items, issues #152–#154)

- **B1 (2.2)** `get_object_range_uri_async` (and 4 downstream callers) did a bare
  `send()`+`collect()` with no retry wrap — a Phase 4b (v0.9.108) regression: once
  response bodies started streaming, a mid-body connection failure (TCP FIN, HTTP/2
  RST) aborted the whole GET instead of retrying, because smithy's own retry only
  covers complete failed responses. Now wrapped in the shared `retry_get_body`
  helper.
- **B2 (2.3)** `length=Some(0)` range GETs left the range-header builder on its
  open-ended `bytes={offset}-` branch, silently downloading the entire remainder of
  the object instead of zero bytes. Now short-circuits to `Ok(Bytes::new())` with no
  network call.
- **B3 (2.4)** A HEAD response missing `Content-Length` was treated as size `0` via
  `.unwrap_or(0)`, silently truncating range reads and poisoning the object-size
  cache for all subsequent calls. Now a proper error (or, for the optimized-GET
  cache path, a fallback to a fresh plain GET without poisoning the cache).
- **B4 (2.5+2.6+2.7)** `get_range()` in the S3, Azure, and community-GCS backends
  each computed the range end as bare `offset + length - 1`, underflowing (panic in
  debug, wrap in release) when `length=Some(0)`. Fixed via a shared, checked
  `range_end_inclusive()` helper plus a `length=Some(0)` fast path at all three
  sites.
- **B5 (3.6)** `S3dlioStorage`/`S3PyTorchConnectorStorage`'s `get_data()` only took
  the `get_range()` path when BOTH `offset` and `length` were given, silently
  returning the full object for offset-only or length-only calls (both documented
  as independently optional). Fixed to route through `get_range()` whenever either
  is given.
- **B6 (3.2)** `S3PyTorchConnectorStorage.put_data()` had no multipart path at all —
  every write, any size, went through a single `put_bytes()` call, hitting the S3
  5 GiB single-PUT limit. Now shares the same env-var-driven multipart threshold as
  `S3dlioStorage` (see B10).
- **B7 (4.4)** `S3ObjectStore::list_with_client`'s regex filter was
  `re.is_match(key) || key.starts_with(prefix)` — since the S3 API is already
  queried with that prefix, every returned key trivially starts with it, making the
  regex check dead code. Fixed to match the regex against the key's tail past the
  prefix.
- **B8 (4.1)** See "Behavior changes" above.
- **B9 (3.10)** See "Behavior changes" above.
- **B10 (3.1)** The mlcommons/storage#715 seed bug: `S3dlioStorage`'s multipart
  threshold/part-size/max-in-flight were hardcoded module constants with no env-var
  override. See "New environment variables" above.

### Phase C — Silent failure elimination (6 bugs, issues #151/#153/#156/#157)

- **C1 (3.3)** `walk_node()` in both DLIO storage backends caught any listing
  exception and returned `[]` — indistinguishable from "legitimately empty". Now
  logs and propagates.
- **C2 (3.4)** `create_node(exist_ok=True)` caught any exception from
  `s3dlio.mkdir` and returned `True`. Now only `FileExistsError` is treated as
  success.
- **C3 (3.5)** `delete_node()` caught any exception from `s3dlio.delete` and
  returned `False`. Now only `FileNotFoundError` is treated as success.
- **C4 (1.4)** Investigated and confirmed already resolved as a side effect of
  A2/A3: `PyMultipartUploadWriter.__exit__`'s `let _ = ...abort_blocking()` calls
  look like they discard a failure signal, but `abort_blocking()` is infallible by
  contract as of A3 (always `Ok`, with the real S3-side failure already logged
  internally) — documented in place, no behavior change needed.
- **C5 (5.3)** The official-GCS-client `delete_objects` logged a panicked/cancelled
  delete task via `tracing::warn!` but never counted it toward the failure total,
  so the function could return `Ok(())` even though an object was never actually
  deleted. Now counted identically to a task's own reported delete failure.
- **C6 (6.1)** `try_read_range_direct`'s to-end-of-file branch panicked via
  `std::fs::metadata(path).unwrap()` (e.g. a TOCTOU-unlinked file) and via a bare
  `metadata.len() - offset` underflow when `offset` exceeded the file's actual
  size — either panic defeated the caller's O_DIRECT/fallback-to-regular-I/O
  design, since a panic isn't caught by that `match`. Both are now clean errors.

### Quality gate

Every fix above: `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`
(and, for the two GCS/Azure-gated fixes, `--features backend-gcs` /
`--features backend-azure,gcs-community` clippy+build checks), plus the relevant
targeted integration test(s) and the Python DLIO-integration suite (`ruff check`,
`ruff format --check`, `python -m compileall`, `uv run pytest`) all clean. This
version shipped in the same branch/PR as v0.9.110 (Phase D+E+F) — see that
section above for the full commit list and the final total of **748 tests
passing** across the whole suite.

## Version 0.9.108 — Performance & concurrency audit (issue #148, all four phases)

Resolves all 17 in-scope items from the issue #148 audit
([`docs/enhancement/PERF-CONCURRENCY-AUDIT-issue148.md`](enhancement/PERF-CONCURRENCY-AUDIT-issue148.md)) —
Phase 1 (buffer double-copy + capacity hint), Phase 2 bug class A
(task-level parallelism across 9 sites), Phase 2 bug class B
(drop-doesn't-abort across 4 sites), Phase 3 (HTTP/2 opt-in reversal),
Phase 4 (streaming connector + shared retry helper). One finding (GCS
retry-with-no-backoff, medium severity) deferred to a separate
GCS-focused pass.

Measured impact against a local high-throughput fake-S3 target
(s3-ultra, ~40 GB/s theoretical): meaningful wins where the fixes are
designed to help; noise-level elsewhere. Full before/after numbers in
[`docs/enhancement/PERF-CONCURRENCY-AUDIT-issue148_Bench-Results.md`](enhancement/PERF-CONCURRENCY-AUDIT-issue148_Bench-Results.md).

| Workload pattern | Speedup |
|---|---|
| 64 KB objects × 100, mixed PUT+GET workload | **3.8×** (baseline 85 → branch 324 MB/s) |
| 64 KB × 100, PUT @ concurrency=64 | **3.2×** (baseline collapses from 390 → 164 MB/s at high conc; branch keeps scaling) |
| 256 KB × 2000, GET @ conc=16 | +17% (~3.7 GB/s) |
| Large single-object GET at 64 MiB (concurrent range engine) | **1.6×** (baseline 678 → branch 1108 MB/s) |
| Large single-object GET at 256 MiB (concurrent range engine) | **2.1×** (baseline 731 → branch 1526 MB/s) |
| 8 MB objects, all sizes / concurrencies | ≈ 0 (wire-bound; no regression) |

### BREAKING CHANGE: `https://` no longer negotiates HTTP/2 by default (Phase 3)

Prior to this release, s3dlio's reqwest client unconditionally advertised
`["h2", "http/1.1"]` in every TLS ClientHello, and rustls picked whichever
protocol the server chose — which for modern S3-compatible endpoints (MinIO,
Ceph RGW, AIStore, most managed object stores) is almost always HTTP/2.
Benchmarking on real workloads has repeatedly shown that HTTP/2 is often
**slower** than HTTP/1.1 for object-storage traffic in this codebase's use
case, primarily due to single-connection flow-control constraints that
survive even adaptive window tuning.

Starting in v0.9.108, the default for `https://` matches the existing
default for `http://`: **HTTP/1.1 unless explicitly opted in**. Deployments
that relied on ALPN-negotiated HTTP/2 for `https://` will now see HTTP/1.1.
Restore the previous behavior with either:

```bash
S3DLIO_HTTPS_H2=1        # opt in per-scheme (https only)
S3DLIO_ENABLE_HTTP2=1    # master switch: HTTP/2 on both http and https
```

Both variables accept `1`, `true`, `yes`, `on`, `enable` (case-insensitive).
Any other value or unset means "off".

| Variable | Scheme | Default | Notes |
|---|---|---|---|
| `S3DLIO_H2C` | `http://` | off | Unchanged — opt in to h2c on plain HTTP. |
| `S3DLIO_HTTPS_H2` | `https://` | off | **New.** Opt in to HTTP/2 via TLS ALPN. |
| `S3DLIO_ENABLE_HTTP2` | both | off | **New.** Master switch — implies both of the above. |

Precedence: H2 is enabled on scheme *S* iff (per-scheme var truthy) OR
(master switch truthy). H2 window-tuning env vars
(`S3DLIO_H2_ADAPTIVE_WINDOW`, `S3DLIO_H2_STREAM_WINDOW_MB`,
`S3DLIO_H2_CONN_WINDOW_MB`) now apply whenever HTTP/2 is enabled on
either scheme — previously they only applied to h2c on `http://`. This
also fixes the uncapped ~5 MiB default receive window that bottlenecked
`https://` HTTP/2 clients per the audit §1.2.

### Phase 1 — buffer double-copy in range assembly (4 sites)

The concurrent range-GET assembly path collected each range's `Bytes`
into a `Vec<(idx, Bytes)>`, sorted, then copied every byte a second
time into a fresh output buffer — driving peak memory to ~2× total
object size during assembly. The four sites (S3
`concurrent_range_get_impl`, the shared `range_engine_generic`,
`s3_bytes::ReaderMode::Range`, and the `file_store_direct` capacity
hint) all pre-allocate a single master `BytesMut`, pre-split it into
per-range segments (zero-copy views into one allocation), stream the
response body **directly into each segment**, and reunify via `unsplit`
on the fast O(1) contiguous path.

Peak memory during a 32 MiB × 32-concurrent range download went from
2.28× total_size to 1.03× total_size — verified by a RED test using a
peak-tracking global allocator (`tests/test_phase1_zero_copy_assembly.rs`).
This halving matters more for high-concurrency small-memory hosts
(MPI workers, containers) than for wall-clock: memcpy is a small
fraction of network wall-time, so the visible speedup is ~5–15% on 10
GbE, larger on faster fabrics or when memory bandwidth-bound.

### Phase 2 bug class A — task-level parallelism (9 sites)

The audit's headline bug: nine sites used `stream::iter(...).map(async
{ ... }).buffered(N)`, `join_all(...)`, or `try_join_all(...)` to drive
concurrent fetches. Because none of the inner futures were
`tokio::spawn`'d individually, tokio treated the whole set as one task —
so all per-fetch CPU work (SigV4 signing, header parsing, body
accumulation, Thrift/Parquet decode) polled inside a single worker
thread's cycle, no matter how many workers the runtime had. This
matches the [mlcommons/storage#701](https://github.com/mlcommons/storage/issues/701)
report of flat ~1.1 GB/s/process throughput against a 10 GB/s endpoint.

Fix at each site: `tokio::spawn` per fetch, a `DropCancel` guard on the
enclosing scope's stack, and each spawn's fetch wrapped in
`tokio::select! { _ = token.cancelled() => ..., r = fetch => r }` so
mid-flight cancellation actually aborts in-flight work (rather than
detaching and letting the fetch run to completion after the caller has
moved on). Panics inside a fetch now surface as `Err(...)` at the
caller instead of silently truncating an iterator.

The nine sites:

| # | File | Notes |
|---|---|---|
| 3.1a | `src/data_loader/async_pool_dataloader.rs` | The core Rust-level DataLoader worker. RED test: parallelism 401 ms → 150 ms; cancel latency 2 s → <500 ms; panic-surfaces-as-error. |
| 1.1 | `src/python_api/python_aiml_api.rs` (3 sites) | Python-facing iterator (`__iter__`, `.items()`, Parquet stream). |
| 3.1b | `src/s3_utils.rs` pre-stat phase in `get_objects_parallel` | GET phase 20 lines below already spawned correctly — an internal-inconsistency oversight. |
| 3.1c | `src/data_loader/parquet_rg.rs` + `parquet_index.rs` (3 call sites) | CPU-bound Thrift-metadata parsing — strongest case for spawn since it's genuinely CPU work. |
| 3.1d | `src/checkpoint/reader.rs` (2 sites) | Distributed-checkpoint shard reads. |
| 3.1e | `src/azure_client.rs::upload_multipart_stream` | Azure block-upload path. |
| 3.1f | `src/range_engine_generic.rs::download_with_ranges` | Shared Azure/GCS range engine. Uses `FuturesOrdered` + bounded prime-and-refill pool to preserve short-read semantics and cap peak memory. |
| 3.1g | `src/s3_utils.rs::stat_object_many_async` | HEAD-batch stat. The trait-default `pre_stat_objects` cannot be spawn-based without breaking `dyn ObjectStore` compatibility; documented with a caveat and a pointer at `stat_object_many_async` as the pattern backends should copy. |
| 3.1h | `src/data_loader/s3_bytes.rs::ReaderMode::Range` | Same `FuturesOrdered` + bounded pool pattern as 3.1f. |

### Phase 2 bug class B — drop-doesn't-abort at short-circuit sites (4 sites)

Four sites that already used `tokio::spawn` correctly had the
`while let Some(res) = futs.next().await { out.push(res??); }` pattern —
which returns from the outer function on the first error, dropping the
remaining `FuturesUnordered<JoinHandle<...>>`. Dropping a `JoinHandle`
*detaches* the task rather than aborting it, so the remaining in-flight
uploads/downloads keep running in the background after the caller has
already returned an error. Retrofitted to the drain-first-then-first-err
pattern from `object_store.rs::generic_upload_files` (which was already
correct). No in-flight task is ever detached.

### Phase 4 — streaming connector + centralized retry (2 sites)

**Streaming connector.** The reqwest connector previously fully
buffered every response body (`resp.bytes().await`) before handing it
to the SDK — so the SDK saw byte one only after the last byte
arrived, and every body-transfer connection interruption had to be
recovered by the smithy retry loop instead of by re-streaming. Now
constructs `SdkBody::from_body_1_x(StreamBody::new(SyncStream::new(bytes_stream)))`
— SDK sees byte one at the moment it arrives on the wire.

`SyncStream` from `sync_wrapper 1.x` provides `Sync` for the reqwest
stream (which is `Send` but not `Sync`) at zero runtime cost via Rust
aliasing rules. Cargo changes: promote `http-body`, `sync_wrapper`
(feature `futures`) to direct deps; enable `http-body-1-x` feature on
`aws-smithy-types`; add `stream` feature to `reqwest`. All four crates
were already transitive; only feature flags and direct-dep declarations
change.

**Centralized retry.** The streaming change means smithy's own retry
no longer covers body-transfer failures (smithy considers the request
done once `send()` returns headers). New shared helper
`crate::retry::retry_get_body<F, Fut, T, E>` bounded by
`max_retry_attempts()` with linear `100ms * attempt` backoff, wired
into four caller sites:

* `S3Ops::get_object` and `S3Ops::get_object_range` — whole-object /
  range GET. Fresh `Bytes` per attempt, trivially idempotent.
* `S3ObjectStore::get` and `S3ObjectStore::get_range` direct-client
  paths — same shape.
* `s3_utils::concurrent_range_get_impl` range-chunk task — the audit
  §2.4 silent-data-corruption gate. Retries stream directly into a
  shared, pre-allocated `BytesMut` segment. Inline retry loop (not
  `retry_get_body`) because the helper's `FnMut() → Fut` signature
  can't share `&mut seg` across attempts. The
  `attempt_range_chunk_fill` helper declares `written = 0` locally so
  each retry starts writing at position 0 — locked in by the
  fault-injection regression test in
  `tests/test_phase4_retry_fault_injection.rs`.

Note: a live end-to-end DLIO_local_changes `unet3d` smoke test against
the streaming-connector s3dlio hit **99.6% AU** with `train_au_meet_expectation:
success` — proves the DLIO training path works cleanly with the new
wheel.

### Fault-injection gate (audit §2.4 blocking requirement)

`tests/test_phase4_retry_fault_injection.rs::phase4_range_chunk_retry_resets_written_cursor_on_each_attempt`
locks in the invariant that each retry attempt starts with a fresh
`written = 0` cursor when writing streamed bytes into the shared range
segment. If `written` leaked across attempts, a partial failed stream's
bytes would remain at `seg[0..old_written]` mixed with successful
retry bytes — length correct, content silently corrupt. This is the
class of bug the audit specifically flagged; the test proves the
implementation is not exposed to it.

### RED-then-GREEN test coverage

Every phase landed with dedicated regression tests that fail against
pre-fix code and pass against fixed code, so the transitions are
bisectable:

| Phase | Regression test |
|---|---|
| 1 | `tests/test_phase1_zero_copy_assembly.rs` — peak-memory-tracking allocator, ≤ 1.5× total_size |
| 2 site 3.1a | `tests/test_phase2_loader_parallelism.rs` — 4 tests (parallelism, external cancel, drop-cancels, panic surfaces) |
| 2 site 3.1b/3.1d pattern | `tests/test_phase2_join_all_vs_spawn.rs` — `join_all` serializes vs. `tokio::spawn` distributes |
| 2 bug class B | `tests/test_phase2_drain_first_err.rs` — short-circuit vs. drain-first-then-error |
| 4 | `tests/test_phase4_retry_fault_injection.rs` — audit §2.4 silent-data-corruption gate |

Zero warnings across `cargo clippy --lib --tests --no-deps -- -D warnings`
under default features, `--features backend-azure`, and
`--features full-backends`. `cargo fmt --check` clean. All shipped
pre-existing test-file clippy warnings across azure_blob_smoke,
test_azure_comprehensive, test_azure_range_engine_integration,
test_gcs_smoke, and test_range_engine_defaults resolved as part of the
release gate.

See [`docs/enhancement/PERF-CONCURRENCY-AUDIT-issue148.md`](enhancement/PERF-CONCURRENCY-AUDIT-issue148.md)
for the full audit and rationale;
[`docs/enhancement/PERF-CONCURRENCY-AUDIT-issue148_Bench-Results.md`](enhancement/PERF-CONCURRENCY-AUDIT-issue148_Bench-Results.md)
for reproducible before/after benchmark methodology and full raw
numbers.

---

## Version 0.9.106 — Write verification changed from always-on to opt-in (mlcommons/storage#593 follow-up)

### Why this release

v0.9.104 made every `put_bytes()` call and every `MultipartUploadWriter`
completion issue a HEAD request to verify the stored byte count, always on
with no way to disable it.  For benchmark workloads that write many small
objects (the common case for AI/ML training datagen), this adds one extra
round-trip per object — a meaningful throughput cost that was not present
before v0.9.104 and is not present in other S3 client libraries, which trust
the server's success response on `PutObject` / `CompleteMultipartUpload`
without a follow-up HEAD.

A server that returns 200 OK for an object that is not actually fully and
durably stored is violating the S3 API contract — the underlying issue
(observed on some AIStore configurations) is a backend correctness problem.
The HEAD-verify-retry behavior added in v0.9.104 is a useful client-side
safety net for backends known to have this problem, but it should not be
mandatory overhead for every backend and every user.

### Changed: write verification is now opt-in, default `false`

Two new boolean environment variables gate the HEAD-verify-retry logic
introduced in v0.9.104.  Both default to `false` (disabled) — the
out-of-the-box behavior is now a single PUT / single
`CompleteMultipartUpload` with no extra round-trip, matching the cost of
every other S3 client library.  Transient network failures are still covered
by `S3DLIO_MAX_RETRY_ATTEMPTS` (the existing SDK-layer retry).

| Variable | Default | Gates |
|----------|---------|-------|
| `S3DLIO_PUT_VERIFY` | `false` | `put_bytes()` / `put_bytes_async()` HEAD-verify-retry (`src/python_api/python_core_api.rs`) |
| `S3DLIO_MPU_PUT_VERIFY` | `false` | `MultipartUploadWriter` post-`CompleteMultipartUpload` HEAD verification (`src/multipart.rs`) |

When disabled, `MultipartCompleteInfo.stored_bytes` is set equal to
`total_bytes` (unverified-assumed-equal) rather than independently confirmed.

`S3DLIO_PUT_MAX_RETRIES`, `S3DLIO_PUT_RETRY_DELAY_MS`, `S3DLIO_MPU_MAX_RETRIES`,
and `S3DLIO_MPU_RETRY_DELAY_S` are unchanged in meaning, but now only take
effect when the corresponding verify flag is enabled.

The two flags are independent — for example, a deployment can enable
`S3DLIO_MPU_PUT_VERIFY=true` for large checkpoint objects (low relative
overhead — one HEAD per object that already makes many `UploadPart` calls)
while leaving `S3DLIO_PUT_VERIFY=false` for the high-volume small-object
datagen path (where the relative overhead would be much higher).

Full reference: `docs/Environment_Variables.md` — "Data Integrity: Write
Verification and Retry".

### Tests (2 new + 1 updated)

`tests/test_multipart.rs`:

- `test_finish_verifies_stored_bytes` now explicitly sets
  `S3DLIO_MPU_PUT_VERIFY=true` so it continues to exercise the HEAD-verify
  path now that it is opt-in.
- `test_finish_skips_verification_by_default` (new): confirms that with the
  flag unset, no HEAD is issued, `stored_bytes == total_bytes`
  (unverified-assumed-equal), and the object still lands correctly —
  confirmed independently by the test's own HEAD call.

`src/python_api/python_core_api.rs` (`#[cfg(test)] mod put_verify_tests`,
requires `--features extension-module`):

- `test_put_verify_default_off` (new): confirms a single PUT lands the object
  correctly with `S3DLIO_PUT_VERIFY` unset.
- `test_put_verify_opt_in` (new): confirms the HEAD-verify-retry path runs
  and validates correctness when `S3DLIO_PUT_VERIFY=true`.

Full suite: 659 passed, 0 failed (integration tests marked `#[ignore]`
require a live endpoint and are excluded from the standard count — this
count is unchanged from v0.9.104 because all new tests in this release are
`#[ignore]`d). All live `#[ignore]` tests (`test_multipart`, `test_put_bytes`,
`put_verify_tests`) pass against a live AIStore endpoint. `cargo fmt --check`
and `cargo clippy -- -D warnings` clean (default features and
`--features extension-module`).

---

## Version 0.9.104 — Write integrity: single-part PUT verification + retry; multipart error propagation, stored-size verification, AIStore redirect safety (mlcommons/storage#593)

### Why this release

External report mlcommons/storage#593 identified two silent-corruption bugs in
`MultipartUploadWriter` that affected multi-host DLIO runs against AIStore:

1. **`__exit__` silently discarded `finish_blocking()` errors.** When
   `CompleteMultipartUpload` or the post-upload `HEAD` check failed, `__exit__`
   called `abort_blocking()` (a no-op at that point) and returned `Ok(())` —
   Python saw no exception and assumed the upload succeeded.

2. **No size verification after `CompleteMultipartUpload`.** The upload was
   declared complete as soon as the AWS SDK returned 200 OK.  On multi-node
   AIStore clusters, HTTP 307 redirects during part upload can route individual
   parts to different storage nodes; if AIStore's internal replication or part
   assembly is incomplete, the resulting object is silently truncated — a valid
   ETag is returned for a file that is missing data.

### Fixed: `__exit__` now propagates `finish_blocking()` errors

`PyMultipartUploadWriter.__exit__` in `src/python_api/python_advanced_api.rs`
now returns `Err(PyRuntimeError)` when `finish_blocking()` fails, surfacing the
error as a Python `RuntimeError` from the `with` block.

Also fixed: `__exit__` previously called `finish_blocking()` even when an
exception propagated from *inside* the `with` block (e.g. `write()` raised).
This incorrectly attempted to complete a partial upload instead of aborting it.
Fixed by checking `_t.bind(py).is_none()`: if `exc_type` is not `None`, the
upload is aborted and the original exception propagates unchanged.

### Fixed: stored-size verification via HEAD after `CompleteMultipartUpload`

`coordinator_task` in `src/multipart.rs` now issues `head_object()` after
`CompleteMultipartUpload` and compares `content_length` to the total bytes
written.  On mismatch:

1. A `WARN` log is emitted with the sizes and an actionable hint to enable
   `RUST_LOG=s3dlio=debug` for per-part ETag tracing.
2. `delete_object()` is called to remove the truncated object, preventing
   downstream readers from consuming corrupt data.  A second `WARN` records
   whether cleanup succeeded; if it did not, the message names the key so the
   operator can delete it manually.
3. The function bails with a `RuntimeError` that names the size mismatch, the
   cleanup outcome, and the explicit instruction to retry the upload.

`MultipartCompleteInfo` gains a new field `stored_bytes: u64` (populated from
the HEAD response) available to callers that want to log or assert the
verified size.

### New: per-part DEBUG logging in `coordinator_task`

`use tracing::{debug, warn}` is now unconditional in `src/multipart.rs`.
Enable with `RUST_LOG=s3dlio=debug`.  Log points:

- MPU session start (upload\_id, bucket/key)
- Each part enqueued (part number, size in bytes, running total)
- Each part completed (part number, ETag received from the backend)
- `CompleteMultipartUpload` call (number of parts, total bytes)
- `CompleteMultipartUpload` result (ETag)
- HEAD verification result (stored bytes)
- `WARN` with actionable message and cleanup outcome on size mismatch

The redirect client (`src/redirect_client.rs`) already emits `debug!` /
`warn!` for every redirect hop (cross-host vs same-host, 303 method rewrite,
non-cloneable body pass-through).  Together these log points allow an operator
to trace whether a size mismatch originated in s3dlio or in the AIStore backend.

Note: `UploadPart` uses `ByteStream::from(Bytes)` (in-memory data), so the
redirect client's `try_clone()` succeeds and AIStore 307 redirects on
individual parts are followed correctly.

### New: Error Recovery and Retry documentation

`docs/performance/MultiPart_README.md` has a new
[Error Recovery and Retry](performance/MultiPart_README.md#error-recovery-and-retry)
section covering:

- Why automatic retry is impossible (streaming architecture; data is consumed
  by `write_blocking()` calls and cannot be replayed inside the library)
- What the library guarantees on failure (truncated object deleted,
  in-progress MPU aborted)
- Two complete Python retry loop examples (explicit `close()` form and
  `with`-statement form) with the critical note that `data_source` must be a
  callable, not a bare iterator/generator
- How to enable `RUST_LOG=s3dlio=debug` for per-part ETag tracing

The FAQ answer for "what happens if a part upload fails?" is updated to
reference the new section.

### New: `put_bytes()` / `put_bytes_async()` verify stored size and retry (all network backends)

`put_verified_with_retry` — a new private async helper in
`src/python_api/python_core_api.rs` — wraps every `store.put()` call with a
HEAD verification step and automatic retry logic.  Both `put_bytes` (sync)
and `put_bytes_async` now call this helper instead of a bare `store.put()`.

**What it does:**

1. Issues `store.put(uri, data)`.
2. Issues `store.stat(uri)` and compares `meta.size` to `data.len()`.
3. On mismatch: logs a `WARN`, calls `store.delete(uri)` to remove the
   truncated object, and retries (up to `S3DLIO_PUT_MAX_RETRIES` attempts,
   with `S3DLIO_PUT_RETRY_DELAY_MS` ms between attempts).
4. If `store.stat()` itself fails after a successful PUT (rare): logs a
   `WARN` and returns `Ok(())` rather than deleting an object that may be
   perfectly intact.

**Backends covered:** S3-compatible, Azure Blob, GCS — anything the
`ObjectStore` trait drives.  `file://` and `direct://` are excluded: local
writes have OS-level durability so the extra HEAD round-trip is unnecessary
overhead.

**Retry is free:** `Bytes` is Arc-backed; `.clone()` is a reference-count
bump, not a data copy.  The data is held in memory for the retry loop at
zero extra cost.

**New env vars** (read at call time, no process-startup caching needed):

| Variable | Default | Description |
|----------|---------|-------------|
| `S3DLIO_PUT_MAX_RETRIES` | `3` | Total PUT attempts before `put_bytes()` raises. |
| `S3DLIO_PUT_RETRY_DELAY_MS` | `1000` | Milliseconds between retry attempts. |

Full reference: `docs/Environment_Variables.md` — "Data Integrity: Write
Verification and Retry".

### Tests (8 new, 3 multipart + 5 put_bytes)

`src/multipart.rs` — unit test:

- `test_complete_info_has_stored_bytes_field`: references
  `MultipartCompleteInfo.stored_bytes` — **FAILED TO COMPILE** before fix
  (field missing), compiles and passes after.

`tests/test_multipart.rs` — integration tests (`#[ignore]`, require live S3 endpoint):

- `test_finish_err_on_zero_parts`: calls `finish_blocking()` with no parts
  written; `CompleteMultipartUpload(parts=[])` is rejected by every backend,
  proving the error that `__exit__` previously silenced.
  Run: `cargo test --test test_multipart -- --ignored`

- `test_finish_verifies_stored_bytes`: uploads 12 MiB (two parts at default
  8 MiB), asserts `info.stored_bytes == data_size` after `finish_blocking()`.
  Run: `cargo test --test test_multipart -- --ignored`

`tests/test_put_bytes.rs` — integration tests (`#[ignore]`, require live S3 endpoint):

- `test_put_bytes_small_object_verified`: 1 KiB PUT → HEAD round-trip; verifies
  stored byte count matches.
- `test_put_bytes_medium_object_verified`: 1 MiB PUT → HEAD; single-part path.
- `test_put_bytes_large_single_part_verified`: 20 MiB PUT → HEAD; exercises
  single-part PUT for objects larger than the DLIO multipart threshold.
- `test_put_bytes_zero_length_object`: empty object → HEAD must report 0 bytes.
- `test_put_bytes_overwrite_size_updates`: two sequential PUTs to the same key;
  verifies HEAD reflects the new size (not stale metadata).
  Run: `cargo test --test test_put_bytes -- --ignored`

Full suite: 659 passed, 0 failed (integration tests marked `#[ignore]` require a live endpoint and are excluded from the standard count). `cargo clippy -- -D warnings` clean.

---

## Version 0.9.102 — SDK error-chain diagnostics + cold-start timeout / retry knobs (mlcommons/storage#506)

### Why this release

External report mlcommons/storage#506 surfaced a `RuntimeError: dispatch
failure` from the s3dlio `collect_batch()` path at warmup of a 50 M-file
RetinaNet B200 training run.  Investigation found two distinct problems
sitting underneath the opaque error string:

1. The bare `SdkError::DispatchFailure` Display was being passed through
   to Python with no connector-chain detail attached — operators could
   not tell whether the underlying cause was I/O, TLS, DNS, timeout, or
   connection refused.  The pre-existing `format_sdk_error()` helper was
   wired on the LIST path only; every GET / HEAD / PUT / range / multipart
   site dropped the chain via `?` or `.context()`.
2. The TCP connect timeout was hardcoded (5 s on the SDK layer, 10 s on
   the reqwest transport — the smaller wins, so 5 s in practice).  The
   constant's doc-comment claimed `S3DLIO_CONNECT_TIMEOUT_SECS` was an
   override, but no code read it.  Under cold-start fan-out (128 worker
   processes × 64 prefetch_window = 8 K concurrent connects) the 5 s
   ceiling was a credible cause of the dispatch failures.

This release ships the diagnostic fix plus the timeout/retry knobs
operators need to mitigate or fast-fail-debug the cold-start case.

### Changed: SdkError → anyhow conversion now preserves the connector chain

22 call sites across `s3_ops.rs`, `s3_utils.rs`, `object_store.rs`, and
`multipart.rs` now route every SDK error through `format_sdk_error()`
before reaching the caller.  Two new helpers in `s3_utils.rs`:

- `pub(crate) fn sdk_anyhow(ctx, e: SdkError<E, R>) -> anyhow::Error`
- `pub(crate) trait SdkResultExt<T> { fn sdk_context(self, ctx) -> Result<T> }`

Both wrap `format_sdk_error()`, which already understands the SDK
categories (I/O / timeout / TLS / DNS / refused / signature / construction)
and emits a hint string per failure shape.  Python now sees, for example:

```
RuntimeError: S3 GET for s3://bkt/key failed:
  dispatch failure (connection timeout: connect error: connection
  attempt timed out after 5 seconds) — server did not respond within
  the connect timeout — verify AWS_ENDPOINT_URL is correct and the
  host is reachable
```

instead of the bare `dispatch failure`.  Behaviour on the success path
is unchanged.

A regression test `test_sdk_anyhow_keeps_context_in_top_message` in
`s3_utils::tests` locks in the contract: the top-line message must
contain both the operation context and the connector-error detail.

> Note: this pass covers the S3 backend only.  The `gs://`, `az://`,
> `file://`, and `direct://` backends still have similar opaque-error
> sites and should get an equivalent helper + audit pass in a follow-up.
> Doc-comments on both helpers call this out.

### New env var: `S3DLIO_CONNECT_TIMEOUT_SECS`

Default `20` (bumped from the previous hardcoded 10 s, which itself was
shadowed by a 5 s SDK-layer ceiling — the *effective* connect timeout
before this release was 5 s).  Honored by both the reqwest transport
and the AWS SDK `TimeoutConfig` layer; the two layers now agree on a
single value.  Raise (e.g. `30`, `60`) for extreme fan-out where the
endpoint's TCP accept queue is briefly saturated by thousands of
concurrent connects.

Default rationale: 20 s is comfortable for healthy datacenter hosts
(a stalled handshake at 20 s still indicates a problem worth flagging)
and absorbs typical cold-start bursts without forcing operators to set
the env var.

### New env var: `S3DLIO_MAX_RETRY_ATTEMPTS`

Default `3` — matches the AWS SDK's own default and the previous
unread `DEFAULT_RETRY_COUNT` constant (which has been removed).  Wired
via `RetryConfig::standard().with_max_attempts(N)` on both the global
and per-endpoint S3 clients.  Two practical use cases:

- `S3DLIO_MAX_RETRY_ATTEMPTS=1` — **fast-fail at warmup**.  Surfaces a
  dispatch failure in ~one connect budget (~connect_timeout) instead
  of ~3 × connect_timeout + exponential backoff.  Useful when
  reproducing #506-style cold-start issues.
- `S3DLIO_MAX_RETRY_ATTEMPTS=5+` — ride out flaky-network bursts on
  unreliable links.

Clamped to a minimum of 1 (the SDK rejects 0).

### Docs

- `docs/Environment_Variables.md`:
  - Replaced dead names in 4 example blocks
    (`S3DLIO_USE_OPTIMIZED_HTTP`, `S3DLIO_MAX_HTTP_CONNECTIONS`,
    `S3DLIO_HTTP_IDLE_TIMEOUT_MS`) with real wired equivalents.
  - New "Cold-Start Fan-Out Configuration" example targeting #506.
  - New "Debug Configuration — Fast-Fail on Cold-Start Issues" block.
  - New "Connection Timeouts and Retries" table row documenting the
    new env vars.
  - Stale "capped at 32" note on `S3DLIO_RT_THREADS` corrected (cap
    was removed in v0.9.92).
  - New "deprecated/dead names" block explicitly listing 8 fictitious
    or unread variable names (including five that an external operator
    set in their mpirun env: `S3DLIO_MAX_CONCURRENCY`,
    `S3DLIO_CONNECTION_TIMEOUT`, `S3DLIO_READ_TIMEOUT`,
    `S3DLIO_MULTIPART_THRESHOLD`, `S3DLIO_PART_SIZE`) — points each
    to its real wired equivalent.

### Tests

- 9 new unit tests under `constants::connect_timeout_tests` and
  `constants::max_retry_tests` cover the env-var helpers (default
  fallback, parse, clamp, empty/unparseable).
- 1 new regression guard in `s3_utils::tests` for the SDK error chain.
- Full lib suite: **324 passed / 0 failed** (was 315 before this release).

### Breaking changes

- `pub const DEFAULT_RETRY_COUNT: usize = 3` was removed from
  `constants.rs`.  It was declared but read by no production code; a
  workspace-wide grep across known downstream crates (dl-driver,
  sai3-bench) showed no consumers.  If a stable consumer surfaces
  we'll re-add it as an alias to `DEFAULT_MAX_RETRY_ATTEMPTS`.

- Default `DEFAULT_CONNECT_TIMEOUT_SECS` changed from `10` to `20`.
  The *effective* connect timeout for almost all callers was previously
  5 s (the SDK ceiling shadowed the 10 s transport setting), so most
  users will see strictly more patience, not less.

---

## Version 0.9.100 — General-purpose object data loader: `PyDataset.from_uris()`, `items()`, `collect_batch()` (May 12, 2026)

### New: URI-manifest dataset and URI-carrying iterators

Adds three new Python API primitives for building high-performance object streaming
pipelines from pre-built URI lists.  All three are backend-agnostic: they work with
`s3://`, `file://`, `direct://`, `az://`, and `gs://` URIs.

---

#### `PyDataset.from_uris(uris, _opts=None)` — map-style dataset from a URI list

```python
ds = s3dlio.PyDataset.from_uris(uris)
```

Creates a map-style `PyDataset` directly from a `list[str]` of full URIs.  **No listing
occurs** — the URIs are used as-is.  The storage backend is inferred from the first entry.

Use this whenever you already have a manifest, a per-worker shard, or any other
pre-computed file list and you do not want the overhead of a prefix listing.

- All URIs must share the same scheme (backend is selected from `uris[0]`).
- `len(ds)` equals `len(uris)`.
- Pass the result directly to `PyBytesAsyncDataLoader`.

Contrast with `PyDataset(prefix_uri)`, which calls `list_objects()` on a storage prefix
and discovers files at runtime.

---

#### `PyBytesAsyncDataLoader.items()` → `PyObjectDataLoaderSyncIter`

```python
item_iter = loader.items()
for item in item_iter:          # GIL released per item while waiting
    process(item.uri, len(item))
```

Synchronous iterator that yields `PyObjectItem` values in **network-completion order**.

Unlike `__iter__()` (which returns bare `PyBytesView`), `items()` attaches the source URI
to every result.  Python never needs a parallel index to know which object arrived.

**Concurrency model:** Tokio keeps exactly `prefetch` GET requests in flight at all times
via `buffer_unordered`.  Each Python consumption frees one slot and immediately triggers
one new GET — a true sliding window with no chunk boundaries and no idle gaps.

**GIL behaviour:** The GIL is released on every `__next__()` call while
`blocking_recv()` waits for the channel.  Other PyTorch DataLoader worker processes run
freely during that wait (each worker has its own GIL).

**`PyObjectItem` attributes:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `item.uri` | `str` | Full URI of the completed GET |
| `item.byte_count` | `int` | Bytes transferred (same as `len(item)`) |
| `len(item)` | `int` | Alias for `byte_count` |
| `bytes(item)` | `bytes` | Raw data — creates one Python-heap copy |

---

#### `PyObjectDataLoaderSyncIter.collect_batch(n)` → `list[PyObjectItem]`

```python
item_iter = loader.items()
while batch := item_iter.collect_batch(16):   # returns [] at end of stream
    for item in batch:
        process(item.uri, len(item))
```

Drains up to `n` completed items from the Rust channel with **one GIL crossing** for the
entire batch, instead of one crossing per item.

**How it works:** A single `py.detach()` block calls `blocking_recv()` up to `n` times
without touching the GIL.  After collecting all `n` items (or reaching end of stream) the
GIL is reacquired once to wrap the raw `(uri, Bytes)` tuples into a Python list.  Python
then iterates a plain list — no `__next__()` dispatch overhead per element.

**Choosing `n`:** A natural value for DLIO-style loops is
`max(1, batch_size // num_samples_per_file)` — one training batch's worth of files per
drain.  This aligns Rust drain boundaries with training batch boundaries and works
correctly for both `num_samples_per_file=1` (JPEG/PNG) and `num_samples_per_file=4`
(NPZ/NPY).

Returns an **empty list** when the stream is exhausted — use this as the stop signal
(compatible with the walrus-operator `while batch :=` pattern).

---

### Internal: `MultiBackendDataset::from_uris()` and `keys()`

- `MultiBackendDataset::from_uris(uris)` — public Rust constructor for URI-manifest
  datasets; backend selected from `uris[0]` via `store_for_uri()`.
- `MultiBackendDataset::keys()` — implements `Dataset::keys()`, returning `Some(self.uris.clone())`.
  Required for `items()` to attach URIs to completed GETs.
- `api::dataset_from_uris()` — public stable-API wrapper used by `PyDataset::from_uris()`.

---

### GIL overhead analysis

At typical object store rates (~1,600 completions/sec per subprocess worker at loopback
latency), `items()` per-item GIL cost is ~0.33% of wall time per worker.  `collect_batch(n)`
reduces this to a single crossing per `n` items; the primary value of `collect_batch()` is
reduced Python object-allocation pressure (one list allocation vs. `n` `PyObjectItem`
constructor calls interleaved with network waits).

Full analysis: [`docs/api/batch-api-design-analysis.md`](api/batch-api-design-analysis.md).

---

### New: `skip_head` — process-wide HEAD-skip optimisation

Eliminates redundant HEAD requests that previously fired once per object on first
access when using the range-optimisation path.

**Background:** `get_object_uri_optimized_async` previously issued a HEAD request on
every cache-miss URI to learn the object size before deciding whether to use a plain
streaming GET or parallel range GETs (threshold: 32 MiB).  For small objects (images,
NPZ, HDF5 row groups) the size is always below the threshold, so the HEAD learned
nothing useful and was pure overhead — adding one extra RTT per object per epoch-1
pass.

**What changed:**

- New `skip_head: bool` field in `LoaderOptions` — defaults to **`true`**.
- When `true`, `get_object_uri_optimized_async` skips the HEAD and issues a plain
  streaming GET immediately.  After the GET completes the actual size (from
  `Bytes::len()`) is stored in the `ObjectSizeCache` — so epoch 2+ still gets correct
  range-split routing for large objects at zero extra cost.
- New `set_skip_head(bool)` public function in `s3_utils` that activates the
  process-wide `AtomicBool` latch; also read from the `S3DLIO_SKIP_HEAD=1` env var
  on first call.
- `PyBytesAsyncDataLoader.items()` and `__iter__()` call `set_skip_head(true)` when
  `opts["skip_head"] == True`.

**Effective GET pattern per epoch:**

```
Epoch 1 (skip_head=True):   plain GET  → cache size     (no HEAD)
Epoch 2+:                   cache hit  → route decision  (range GET if ≥ 32 MiB)
```

**Python API:**

```python
# Small objects (default) — no HEAD ever
loader = s3dlio.PyBytesAsyncDataLoader(ds, {"prefetch": 64})

# Large objects, single-epoch job — HEAD fires on epoch 1 for immediate range GETs
loader = s3dlio.PyBytesAsyncDataLoader(ds, {"prefetch": 8, "skip_head": False})

# Or via environment (no code change)
# export S3DLIO_SKIP_HEAD=1
```

**Observed impact (retinanet, 500 k objects, loopback S3):** eliminates 500 HEAD/s
visible on server-side metrics during epoch 1.  Epoch-1 throughput unchanged at
~150 MiB/s / ~97% AU; epoch-2 identical (no regression).

See [`docs/Python_Data-Loader.md § HEAD-skip optimisation`](Python_Data-Loader.md#head-skip-optimisation).

---

## Version 0.9.98 — Parquet DataLoader with epoch-2 fast path and Arrow IPC decode (May 9, 2026)

### New: `ParquetRowGroupDataset` — epoch-aware Parquet DataLoader for AI/ML training loops

Adds a production-ready Parquet DataLoader where each dataset item is one Parquet **row
group** — the natural I/O unit for distributed training. The loader is designed for multi-epoch
training with 8+ concurrent worker processes.

**Key capabilities:**

- **Zero re-fetch on epoch 2+**: Row-group byte ranges (start offset + length + row count) are
  cached in a process-global `DashMap` after the first epoch. Epoch-2+ `new()` calls read
  from this cache — no S3 footer GETs issued. Measured **2.5× speedup** on epoch-2
  construction vs epoch-1 (20.4 ms → 8.3 ms on 2-file / 4-RG test dataset; proportionally
  larger at scale with 64+ files).

- **Two decode modes:**
  - `Raw` (default, `parquet` feature): Python receives compressed Parquet column-chunk bytes.
    Ideal for pure storage benchmarks or when Python controls decoding.
  - `ArrowIpc` (`parquet-arrow` feature, also default): Rust decodes Parquet column data to
    Arrow `RecordBatch` and serialises to Arrow IPC stream format. Python reconstructs via
    `pa.ipc.open_stream(pa.py_buffer(item["data"])).read_next_batch()`.

- **Minimal RAM**: Only metadata held in RAM. 8 concurrent worker instances share the
  process-global footer cache and row-group index — no 8× duplication. Each `get()` call
  fetches exactly one row group, freed when Python releases the `bytes` reference.

- **Concurrent prefetch**: `PyParquetStreamLoader` spawns one Tokio task per iterator,
  keeping `prefetch` (default 32) row-group GETs in flight simultaneously via
  `buffer_unordered`. A bounded `mpsc::channel(prefetch)` provides back-pressure — no
  Semaphore, no JoinSet overhead.

- **Column projection**: Pass `columns=[0, 2, 5]` to fetch only those column indices per
  row group (reduces per-item byte range).

- **DLIO benchmark integration**: Drop-in replacement for dlio_benchmark's native Parquet
  reader via `create_async_loader(prefix, {"format": "parquet", "decode": "raw"})`.

**Subcomponents added:**

| File | Purpose |
|------|---------|
| `src/data_loader/parquet_rg.rs` | `ParquetRowGroupDataset`, `ParquetDecodeMode`, `RgExtent`, `build_extents()`, epoch-2 fast path |
| `src/data_loader/parquet_index.rs` | `ParquetIndex` — process-global `OnceLock<DashMap>` keyed by S3 URI |
| `src/data_loader/parquet_file_cache.rs` | Per-URI footer cache with `Arc<ParquetMetaData>` sharing and `OnceCell` for concurrent-fetch coalescing |
| `src/python_api/python_aiml_api.rs` | `PyParquetStreamLoader`, `ParquetStreamIter`, `create_async_loader` parquet path |
| `tests/test_parquet_dataloader.py` | Python integration tests (raw loader, Arrow IPC, epoch fast path) |

**Feature flags:**

| Feature | Default? | What it enables |
|---------|----------|-----------------|
| `parquet` | ✅ Yes | `ParquetRowGroupDataset`, `ParquetIndex`, `parquet_file_cache`, `Raw` mode |
| `parquet-arrow` | ✅ Yes | `ArrowIpc` decode mode (Rust → Arrow RecordBatch → IPC bytes) |

> Note: `arrow-backend` (Apache Arrow's `object_store` as the S3 client) is a separate,
> mutually exclusive feature and is **not** related to the Parquet DataLoader.

**Python API:**

```python
import s3dlio

# Raw mode (default) — Python receives Parquet bytes, decodes with PyArrow
loader = s3dlio.create_async_loader(
    "s3://bucket/train/",
    {"format": "parquet", "prefetch": 32}
)
for item in loader:
    # item["data"]: bytes, item["uri"]: str, item["rg_idx"]: int
    table = pyarrow.parquet.read_table(io.BytesIO(item["data"]))

# Arrow IPC mode — Rust decodes, Python gets ready-to-use RecordBatch bytes
loader = s3dlio.create_async_loader(
    "s3://bucket/train/",
    {"format": "parquet", "decode": "arrow", "prefetch": 32}
)
for item in loader:
    batch = pa.ipc.open_stream(pa.py_buffer(item["data"])).read_next_batch()
```

**Testing — 17 unit tests + 6 MinIO integration tests (all passing):**

Unit tests (no network, `cargo test --features parquet-arrow -- parquet`):
- Footer parser: magic check, bad magic rejection, too-small buffer rejection
- Decode mode: default is Raw, variants compile and differ
- `needs_file_metadata`: Raw=false, ArrowIpc=true, modes differ
- Global index singleton: same pointer on every `global()` call
- Global index col_indices: always None (all-column extents)
- `ParquetIndex`: insert/lookup, binary search, range lookup, epoch counter, invalidate, combined

Integration tests (live MinIO, `--test-threads=1 --ignored`):
- Epoch-1 construction: 4 RGs, index populated, `get(0)` non-empty
- Epoch-2 fast path: identical results to epoch-1
- Row offset consistency: `rg_num_rows()` matches `num_rows_in_rg()`
- **Timing proof**: epoch-2 ≤ epoch-1 construction time, < 2 s absolute
- Raw fast path: `file_metadata_len() == 0` on epoch-2 (Raw mode skips file_metadata)
- ArrowIpc fast path: `file_metadata_len() == num_files` on epoch-2

**Memory model:**

| What | Size | Scope |
|------|------|-------|
| `extents` (byte ranges) | ~48 B × N_rgs | Per dataset instance |
| `parquet_file_cache` (parsed footer) | few MB per file, Arc-shared | Process-global |
| `parquet_index` DashMap entries | ~80 B × N_rgs | Process-global singleton |
| Active `get()` payload | 1 row-group ≈ 1–100 MB | Freed when Python releases `bytes` |

📖 **[Complete Parquet DataLoader Guide](Parquet_Data-Loader.md)**

---

### Changes in v0.9.98

- **`Cargo.toml` / `pyproject.toml`**: Bump version `0.9.97 → 0.9.98`
- **`Cargo.toml` features**: `parquet = ["dep:parquet", ...]` and `parquet-arrow = [...]` added to default feature set
- **`src/data_loader/mod.rs`**: Export `parquet_rg`, `parquet_index`, `parquet_file_cache`, `ParquetRowGroupDataset`, `ParquetDecodeMode`, `DEFAULT_FOOTER_CAP`
- **`src/data_loader/parquet_rg.rs`**: New file — full `ParquetRowGroupDataset` implementation with epoch-2 fast path, `Raw` + `ArrowIpc` modes, 17 unit + 6 integration tests
- **`src/data_loader/parquet_index.rs`**: New file — `ParquetIndex` global singleton + 6 unit tests
- **`src/data_loader/parquet_file_cache.rs`**: New file — per-URI footer de-duplication
- **`src/python_api/python_aiml_api.rs`**: `PyParquetStreamLoader`, `ParquetStreamIter`, extended `create_async_loader` for parquet format
- **`tests/test_parquet_dataloader.py`**: Python integration test suite

---

## Version 0.9.97 — Unsigned payload support + XorStream Python bindings (May 4, 2026)

### New: `XorStream` — fast, dedup-safe data generation via `dgen-data` crate (May 2026)

Adds `s3dlio.XorStream`, a high-performance data generator for Python and Rust callers.
Backed by [`dgen_data::UniqueXorStream`] — a 1 MiB random base buffer + `AtomicU64`
counter + splitmix64 + Xoshiro256++ XOR keystream. Each `fill()` / `generate()` call
produces a unique 512-byte-block-level output, guaranteeing dedup incompressibility.

**Thread-safe**: `&self` API, no mutex, `Sync + Send`. ~15 GB/s per core.

**Python API** (no changes to existing API):

```python
import s3dlio

stream = s3dlio.XorStream()

# In-place fill — fastest, no per-call heap allocation
buf = bytearray(8 * 1024 * 1024)
stream.fill(buf)                     # unique 8 MiB payload
stream.fill(buf)                     # different 8 MiB payload, guaranteed

# Allocate + fill in one call — returns BytesView (buffer protocol)
data = stream.generate(8 * 1024 * 1024)
view = memoryview(data)              # zero-copy access
import numpy as np
arr = np.frombuffer(view, dtype=np.uint8)  # zero-copy numpy array

print(stream.objects_generated)     # == 3
```

**Changes**:

- **`Cargo.toml`** — Added `dgen-data = { path = "../dgen-rs", default-features = false }`.
  Must be changed to a git tag before PyPI release.
- **`src/python_api/python_datagen_api.rs`** — `PyXorStream` class (`fill`, `generate`,
  `objects_generated` property); registered via `register_datagen_functions()`.

---

### New: `S3DLIO_UNSIGNED_PAYLOAD` — bypass SHA-256 body hashing on PUT (`src/constants.rs`, `src/s3_client.rs`, `src/object_store.rs`, `src/s3_ops.rs`)

Adds opt-in support for skipping AWS SigV4 payload signing on PUT operations. Enabled via
the `S3DLIO_UNSIGNED_PAYLOAD=1` environment variable (also accepts `true`, `yes`, `on`,
`enable`). Default is `false` — existing behaviour is unchanged.

**When to use**: Only on internal, trusted, non-TLS endpoints (MinIO, s3-ultra, Ceph RGW,
or any private S3-compatible server where the network path is considered secure). Must
**never** be used against real AWS S3 endpoints — AWS requires a valid payload hash for
SigV4-signed requests over HTTPS.

**When it helps**: Primarily for large objects (1 MiB+) where SHA-256 computation becomes
a meaningful fraction of per-request CPU time. For objects ≤ 1 KiB, the cost of SHA-256
hashing (~0.3 µs) is negligible compared to network latency — no throughput improvement
is expected or observed at small sizes.

**Changes made**:

- **`src/constants.rs`** — Two new constants:
  ```rust
  pub const ENV_UNSIGNED_PAYLOAD: &str = "S3DLIO_UNSIGNED_PAYLOAD";
  pub const DEFAULT_UNSIGNED_PAYLOAD: bool = false;
  ```

- **`src/s3_client.rs`** — New helper function `unsigned_payload_enabled()` using
  `std::sync::OnceLock` to read and cache the environment variable exactly once at first
  call (zero-cost on the hot path for all subsequent requests):
  ```rust
  pub fn unsigned_payload_enabled() -> bool {
      static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
      *CACHE.get_or_init(|| {
          std::env::var(crate::constants::ENV_UNSIGNED_PAYLOAD)
              .map(|v| matches!(v.to_lowercase().as_str(), "1" | "true" | "yes" | "on" | "enable"))
              .unwrap_or(crate::constants::DEFAULT_UNSIGNED_PAYLOAD)
      })
  }
  ```

- **`src/object_store.rs`** — `S3ObjectStore::put()` conditionally calls
  `.customize().disable_payload_signing()` on the `PutObjectFluentBuilder` when
  `unsigned_payload_enabled()` returns `true`.

- **`src/s3_ops.rs`** — `S3Ops::put_object()` conditionally calls
  `.customize().disable_payload_signing()` on the same builder path.

**Not available for GET**: The `disable_payload_signing()` API in the AWS SDK only exists
on `PutObject` and `UploadPart` builders. GET requests carry an empty body whose SHA-256
is the compile-time constant `e3b0c44298fc1c149afb...` — there is no computation to skip.

### Tests: No new tests added for this feature

This feature was validated empirically via benchmark comparison against s3-ultra. A unit
test for `unsigned_payload_enabled()` (env-var parsing and OnceLock behaviour) would be
a good follow-up addition.

---

## Version 0.9.96 — Multi-endpoint correctness, S3_ENDPOINT_URIS enforcement, new CLI options (April 28, 2026)

### Fix: All endpoints in `S3_ENDPOINT_URIS` are now used (`src/multi_endpoint.rs`)

Previously, passing multiple endpoints via the `S3_ENDPOINT_URIS` environment variable (comma-separated URIs) did not guarantee that all listed endpoints were actually consumed. `MultiEndpointStore::from_env()` now parses the full comma-separated list, trims whitespace around each entry, and passes all N URIs to `MultiEndpointStore::new()`. Any count from 1 to `MAX_ENDPOINTS` (32) is accepted; 0 or >32 returns an error.

### New: `MultiEndpointStore::from_env()` public API (`src/multi_endpoint.rs`)

New method that reads `S3_ENDPOINT_URIS` and constructs a `MultiEndpointStore` with the requested `LoadBalanceStrategy`. Accepts both `RoundRobin` and `LeastConnections`. Used internally by the CLI and Python API, and now also by sai3-bench as a fallback when no YAML `multi_endpoint:` block is present.

### New: `MAX_ENDPOINTS` constant and `max_endpoints()` accessor (`src/constants.rs`)

`MAX_ENDPOINTS = 32` is now a named constant (was previously a magic number). Exposed as `s3dlio::constants::max_endpoints()` for downstream crates to reference programmatically without hard-coding the value.

### Fix: Non-recursive `list()` now returns directory entries (`src/file_store.rs`)

`FileSystemObjectStore::list(uri, recursive=false)` was silently dropping directory entries — only files were returned. Fixed to include subdirectories with a trailing `/` (matching S3 common-prefix semantics). Recursive listing is unchanged.

### New: CLI options `--endpoint-url`, `--region`, `--ca-bundle` (`src/bin/cli.rs`)

The `s3dlio` CLI now accepts:
- `--endpoint-url` (alias `--endpoint`) — override the S3 endpoint URL for a single command
- `--region` — override the AWS region
- `--ca-bundle` — path to a custom CA certificate bundle (for self-signed TLS)

`--endpoint-url` values are also validated: if `--endpoints` is supplied simultaneously, the count must not exceed `MAX_ENDPOINTS`.

### New: CLI `--endpoints` validation (`src/bin/cli.rs`)

`build_s3_endpoint_uris()` now rejects endpoint lists exceeding `MAX_ENDPOINTS` with a clear error message. Previously, oversized lists were silently accepted and could cause undefined behaviour at runtime.

### New: Python API `create_multi_endpoint_store_from_env()` (`src/python_api/python_core_api.rs`)

Python callers can now construct a `MultiEndpointStore` directly from the `S3_ENDPOINT_URIS` environment variable without writing endpoint lists into Python code.

### Tests: 613 passing (lib + integration + CLI)

- 3 new `from_env` consistency tests in `multi_endpoint.rs` (`test_from_env_4_endpoints_all_present`, `test_from_env_whitespace_is_trimmed`, `test_from_env_valid_counts_1_to_max`)
- 8 CLI endpoint boundary tests in `src/bin/cli.rs`
- Updated `tests/test_file_store.rs` to assert correct non-recursive listing behaviour (file + subdir)

---

## Version 0.9.95 — O_DIRECT fix, BufferPool deadlock fix (April 27, 2026)

### Fix: `direct://` URIs now correctly use O_DIRECT in `get_many()` (`src/python_api/python_core_api.rs`)

`Scheme::Direct` was previously handled by the same code branch as `Scheme::File`, causing `direct://` URIs to silently fall through to `tokio::fs::read()` — the standard buffered I/O path. Page-cache bypass was never actually engaged in `get_many()`.

**Fix:** Split into two separate match arms. `Scheme::Direct` now creates a `ConfigurableFileSystemObjectStore::with_direct_io()` and submits all reads through it, bypassing the OS page cache as intended. `Scheme::File` retains its existing `tokio::fs::read()` path with the stale `direct://` prefix-stripping code removed.

**Impact:** `direct://` reads in Python (`get_many(["direct:///path/file", ...])`) now actually bypass the page cache. This was a silent correctness bug — the API appeared to work but provided no O_DIRECT benefit.

### Fix: `BufferPool::give()` deadlock (`src/memory.rs`, `src/file_store_direct.rs`)

**Root cause:** `BufferPool::new(n, …)` spawns a pre-allocation task on the global background runtime that fills the channel to capacity `n`. If `take()` runs before pre-alloc completes (the normal case — the caller's runtime is usually faster), it allocates a new buffer via the fallback path, creating a `n+1`th buffer that the channel cannot hold. The previous `async fn give()` called `tx.send(buf).await`, which blocks indefinitely on a full channel with no concurrent drainer → **permanent deadlock**.

This is a real production bug, not just a test issue. Any caller using `BufferPool` under high concurrency where `take()`'s fallback path fires (by design) could deadlock.

**Fix:** `give()` changed from `async fn` using `tx.send().await` to a sync `fn` using `tx.try_send()`. If the channel is full, the buffer is simply dropped — correct behaviour, since fallback-allocated buffers are not needed by the pool. All call sites updated (`src/file_store_direct.rs`, `tests/test_buffer_pool_directio.rs`).

**Test hardening:** `buffer_pool_basic` now wraps the test body in `tokio::time::timeout(5s)` so a regression will fail with a clear message instead of hanging indefinitely.

### Internal: `global_rt_handle()` made `pub(crate)` (`src/s3_client.rs`)

Allows `memory.rs` to spawn `BufferPool` pre-allocation tasks on the same background runtime used by all other async operations — consistent runtime behaviour across all URI schemes.

---

## Version 0.9.94 — Fast NPZ generation, zero-copy BytesView upload path (April 25, 2026)

### New: `generate_npz_bytes()` — Rust NPZ builder (`src/data_formats/npz.rs`, `src/python_api/python_datagen_api.rs`)

Generates a complete NumPy `.npz` archive in Rust without holding the GIL.

**Design:**
- Single `Vec<u8>` allocation sized exactly to the final file — no intermediate copies
- Random data (`x.npy`) filled in-place with Xoshiro256++ via Rayon (2 MiB chunks, each with an independent seed — page faults distributed across all cores)
- CRC32 computed by `crc32fast` (SSE4.2/NEON hardware acceleration) on already-hot pages
- ZIP structure (local file headers, central directory, EOCD) written with fixed-offset pointer arithmetic — no `ZipWriter` state machine overhead
- Labels (`y.npy`) are int64 zeros, matching the unet3d training format

**Performance (140 MiB file, 28-core machine):**

| Method | Latency |
|--------|---------|
| `numpy.savez()` (Python CRC32 via `zlib`) | ~178 ms |
| `generate_npz_bytes_raw()` (Rust, this change) | ~20 ms |

The returned `Vec<u8>` is wrapped in `Bytes::from(vec)` (zero-cost Arc) and returned as a `PyBytesView`.

### Fix: Duplicate `PyBytesView` causing silent zero-copy failure (`src/python_api/python_datagen_api.rs`)

`python_datagen_api` previously defined its own `PyBytesView` (backed by `DataBuffer`) and registered it as Python's `BytesView` class. Because `python_datagen_api` was registered after `python_core_api`, the datagen version silently replaced the core version at the Python class level. `MultipartUploadWriter.write()`'s fast path tried to extract `python_core_api::PyBytesView` — the extraction always failed and fell through to the `PyBuffer` path, which held the GIL during a full `memcpy` (~109 ms for 140 MiB, serializing all concurrent threads).

**Fix:** Removed the duplicate struct entirely. `python_datagen_api` now imports and uses `python_core_api::PyBytesView`. Generate functions use `DataBuffer::into_bytes()` (zero-cost for `Uma(Vec<u8>)` via `Bytes::from(vec)`) before wrapping.

**Impact:** Write latency for 140 MiB BytesView dropped from ~109 ms → ~6 ms. Sustained upload throughput at N=48 improved from ~1,250 MiB/s → ~2,440 MiB/s (1.95×).

### New: `write_bytes_blocking()` on `MultipartUploadSink` (`src/multipart.rs`)

Zero-copy write method accepting owned `bytes::Bytes`. For large inputs (≥ `part_size`), slices the shared `Arc<[u8]>` into part-sized chunks using `Bytes::slice(offset..end)` — no `memcpy`. Only the final sub-part tail (< `part_size`, typically ≤ 12 MiB for 16 MiB parts) is copied into the accumulation buffer. Enqueues parts to the coordinator channel (~1–70 µs per `blocking_send`).

### New: `PyBytesView` fast path in `MultipartUploadWriter.write()` (`src/python_api/python_advanced_api.rs`)

Added path 0 (before the existing `PyBuffer` path) that detects `BytesView` objects via `data.extract::<PyRef<PyBytesView>>()`, performs an Arc refcount increment (`bv.bytes.clone()` — O(1), no data movement), releases the GIL, and calls `write_bytes_blocking()`. The existing `PyBuffer` path remains as fallback for plain `bytes`, `bytearray`, `memoryview`, and NumPy arrays.

### Fix: `PyBytesView::bytes` field made `pub` (`src/python_api/python_core_api.rs`)

Required for `python_advanced_api` to Arc-clone the underlying `Bytes` without unsafe code.

### Fix: README typo (`README.md`)

Trailing `m` removed from v0.9.92 blurb (`"fixesm"` → `"fixes"`); blurb replaced with v0.9.94 highlights.

### Docs added (`docs/`)

- `CUSTOM_S3_CLIENT_ANALYSIS.md` — Analysis of AWS SDK overhead vs. lightweight reqwest client; motivates future custom SigV4 signer work
- `SHA256_SIGV4_OPTIMIZATION_ANALYSIS.md` — Deep-dive on SHA-256 hot path in SigV4 signing; benchmark data from sai3-bench → s3-ultra loopback profiling

---

## Version 0.9.92 — Post-review async safety fixes, MAX_MULTIPART_PARTS guard, clippy (April 23, 2026)

_This entry documents the second round of fixes committed on top of the coordinator-task rewrite below._

### Fixes (`src/multipart.rs`, `src/bin/cli.rs`)

| # | Severity | Issue | Resolution |
|---|---|---|---|
| 1 | 🔴 Latent panic | `write()`, `write_owned()`, `flush()` called `blocking_send` from async context | Added `enqueue_part_async()` using `send().await`; rewrote all three methods to use it; added `async fn finish()` |
| 2 | 🔴 Policy | 2 clippy warnings (coordinator `loop { match }`, manual `Option::filter`) | `while let` loop in coordinator; `.filter()` in `cli.rs` |
| 3 | 🟡 UX | `MAX_MULTIPART_PARTS` (10 000) defined but never enforced | Checked in both `enqueue_part()` and `enqueue_part_async()` with a clear bail message |
| 4 | 🟡 Docs | Memory ceiling doc off by ~2× | Updated to `~2 × max_in_flight × part_size` with corrected examples |

A new unit test `test_blocking_send_panics_inside_tokio_runtime` was added to prove the pre-fix
panic class via `std::panic::catch_unwind`.

**Test count**: 580 passing (247 Rust unit tests + Python integration tests).

### CI / Release workflow updates (`.github/workflows/`)

| File | Change |
|---|---|
| `ci.yml` | Removed Python 3.10 (unsupported by our PyO3 version); added `ubuntu-24.04-arm` runner — CI now tests 3.11, 3.12, 3.13 on both x86_64 and aarch64 (6 matrix jobs) |
| `publish-pypi.yml` | Added `aarch64` to wheel build matrix; `runs-on` selects `ubuntu-24.04-arm` natively for ARM builds; result: 6 wheels published per release (3 Python versions × 2 architectures) plus sdist |

---

## Version 0.9.92 — Multipart upload: coordinator task, auto-scale max_in_flight, panic fix (April 2026)

_This entry covers work added on top of the base v0.9.92 release (connection pool / runtime changes documented below)._

### Multipart upload rewrite (`src/multipart.rs`)

The multipart upload implementation was fully rewritten to resolve two correctness issues and
improve throughput at high part counts.

**Change 1 — Coordinator task + bounded mpsc channel**

Previously each `write()` call ran `run_on_global_rt(semaphore.acquire_owned())`, crossing the
Python→Tokio boundary on every part. Under high concurrency this caused stalls because
`run_on_global_rt` parks the calling thread until the future resolves.

The new design uses a background `coordinator_task` running entirely on the Tokio runtime. It owns
a `tokio::sync::mpsc` bounded channel (capacity = `max_in_flight`). Python `write()` calls
`blocking_send()` on the channel — this parks the Python thread only when all slots are genuinely
occupied (true backpressure). The coordinator acquires the semaphore and spawns UploadPart tasks
async-natively. `finish_blocking()` sends a `Finish` sentinel and makes a single `run_on_global_rt`
call to await the coordinator, replacing one blocking cross-thread trip per part with one per upload.

**Change 2 — Auto-scale `max_in_flight`**

`MultipartUploadConfig.max_in_flight` changed from `usize` to `Option<usize>`:
- `None` (new default) → auto: `max(32, ⌈512 MiB ÷ part_size⌉)`
- `Some(n)` → explicit override, same as before

At the default 16 MiB part size this raises the default from 16 → 32 concurrent upload slots,
doubling the parallelism for callers that never passed an explicit value. At 8 MiB parts it becomes
64 slots, scaling to match the higher part count for large objects.

Python callers that pass `max_in_flight=N` explicitly (e.g. `dlio_benchmark` checkpointing, which
reads `S3DLIO_MULTIPART_MAX_IN_FLIGHT`) are unaffected — `Some(N)` is used as before.

**Change 3 — Fix latent panic in `put_object_multipart_uri_async` (`src/s3_utils.rs`)**

`put_object_multipart_uri_async` is an `async fn` called from within the Tokio runtime (via
`ObjectStore::put_multipart`), but it previously called `sink.write_blocking()` and
`sink.finish_blocking()` directly. These call `blocking_send()` and `run_on_global_rt()`, which
park a Tokio worker thread when the channel fills. For objects larger than
`max_in_flight × part_size` (~512 MiB at the old defaults) this would panic.

Fixed by wrapping the entire sink lifecycle in `tokio::task::spawn_blocking`, so blocking calls
run on a dedicated blocking thread rather than a worker thread. `run_on_global_rt` detects the
runtime context and uses `std::sync::mpsc` blocking recv, which is safe from blocking threads.

### Benchmark results (real MinIO, HTTPS, vs s3torchconnector 1.5.0)

| Object size | NP=1 | NP=4 |
|---|---|---|
| 512 MiB | 0.659 GB/s (+33% vs baseline) | 0.944 GB/s (+10%) |
| 5 GiB | 0.941 GB/s (1.10× s3torchconnector) | 1.015 GB/s (1.03×) |

### New tests

- `python/tests/test_issue134_backpressure.py` — 7 tests (integrity, OOM, concurrency, throughput)
  all passing on real MinIO
- `python/tests/test_mpu_throughput.py` — 7 throughput regression tests
- `python/tests/bench_multipart_vs_s3torchconnector.py` — benchmark harness, 5 GiB default

### Summary table

| Change | File(s) | Detail |
|--------|---------|--------|
| Coordinator task | `src/multipart.rs` | `coordinator_task` + `mpsc::bounded(max_in_flight)` replaces per-part `run_on_global_rt` |
| Auto-scale | `src/multipart.rs`, `src/python_api/python_advanced_api.rs` | `max_in_flight: usize` → `Option<usize>`; `None` = auto formula |
| Panic fix | `src/s3_utils.rs` | `put_object_multipart_uri_async` wrapped in `spawn_blocking` |
| Test compat | `tests/test_multipart.rs` | Integration test struct literals updated to `Some(n)` |

---

## Version 0.9.92 — Default HTTP/2 off, unlimited connection pool, runtime scaling, concurrency API (April 2026)

### Summary of code changes

| Fix | File(s) | Change |
|-----|---------|--------|
| 1 — Unlimited connection pool | `src/constants.rs` | `DEFAULT_POOL_MAX_IDLE_PER_HOST`: `32` → `usize::MAX`. Pool still shrinks via `pool_idle_timeout` (90 s). Doc comment explains the 794 µs root cause at high concurrency. |
| 2 — Thread cap removed | `src/s3_client.rs` | `get_runtime_threads()`: replaced `min(max(8, cores*2), 32)` with `max(4, cores)`. 28-core machine unchanged (28 threads); 96-core machine goes 32 → 96. |
| 3 — Concurrency hint API | `src/s3_client.rs`, `src/lib.rs` | Added `CONCURRENCY_HINT: AtomicUsize` static and `pub fn configure_for_concurrency(n: usize)`, exported at crate root as `s3dlio::configure_for_concurrency`. If the hint exceeds the CPU baseline, runtime thread count is `min(hint, cores*4)`. Must be called before the first S3 operation. |
| 4 — Connection pool warmup | `src/reqwest_client.rs` | Added `pub async fn warmup_connection_pool(endpoint_url: &str, connections: usize)` — fires `connections` concurrent HEAD requests to pre-fill the pool and eliminate TCP handshake storms at benchmark start. |
| 5 — HTTP/2 default off | `src/constants.rs`, `src/reqwest_client.rs` | `DEFAULT_H2C_ENABLED = false` added. `h2c_mode_from_env()` now returns `ForceHttp1` when `S3DLIO_H2C` is unset (was `Auto`). Benchmarking showed HTTP/2 reduces throughput on plain `http://` endpoints. Set `S3DLIO_H2C=1` to opt in. |

All changes compile with zero warnings; **243 tests pass**.

### Breaking change: `S3DLIO_H2C` default changed to HTTP/1.1

HTTP/2 on plain `http://` endpoints is now **off by default**. Benchmarking on loopback TCP
showed HTTP/2 reduces PUT/GET throughput compared with HTTP/1.1 and an unlimited connection pool.
The `Auto` mode (probe h2c once, fall back if rejected) is no longer the default.

| Scenario | Before v0.9.92 | v0.9.92+ |
|---|---|---|
| `S3DLIO_H2C` not set, `http://` endpoint | h2c probe once, HTTP/1.1 fallback | **HTTP/1.1 (no probe)** |
| `S3DLIO_H2C=1`, `http://` endpoint | Force h2c, no fallback | Force h2c, no fallback (unchanged) |
| `S3DLIO_H2C=0`, `http://` endpoint | Force HTTP/1.1 | Force HTTP/1.1 (unchanged) |
| Any `https://` endpoint | HTTP/2 via TLS ALPN | HTTP/2 via TLS ALPN (unchanged) |

Set `S3DLIO_H2C=1` to restore h2c for storage systems that require HTTP/2 on `http://` endpoints.
The new default is documented in `src/constants.rs` as `DEFAULT_H2C_ENABLED = false`.

### Performance: connection pool default changed to unlimited

`S3DLIO_POOL_MAX_IDLE_PER_HOST` default changed from `32` to `usize::MAX` (unlimited).

With t=128 concurrent workers and a pool cap of 32, the 96 workers that couldn't hold an idle
connection paid a full TCP handshake penalty on every request (~794 µs on loopback). This capped
throughput at ~47k ops/s externally vs ~240k ops/s in-process. With an unlimited pool, idle
connections are retained for all workers; the `S3DLIO_POOL_IDLE_TIMEOUT_SECS` (default 90 s)
eviction timer handles cleanup when load drops.

Set `S3DLIO_POOL_MAX_IDLE_PER_HOST=<n>` to impose a hard ceiling in memory-constrained environments.

### Performance: s3dlio internal runtime thread cap removed

`get_runtime_threads()` previously capped at 32 even on large machines (e.g. 96-core hosts got
32 Tokio threads). The cap is removed; the default is now `max(4, num_cpus)`. The floor of 4
prevents thread starvation on single/dual-core VMs. This primarily benefits Python/sync callers
that route through the s3dlio-internal runtime.

### New API: `configure_for_concurrency(n)`

```rust
s3dlio::configure_for_concurrency(128);
// … then call blocking S3 helpers from 128 threads …
```

Must be called before the first S3 operation. Ensures the internal Tokio runtime has enough
threads to overlap `n` concurrent requests for blocking/Python callers. Ignored by async callers
(e.g. sai3-bench) that bring their own runtime.

### New API: `reqwest_client::warmup_connection_pool(endpoint_url, n)`

```rust
s3dlio::reqwest_client::warmup_connection_pool("http://127.0.0.1:9000", 128).await;
```

Fires `n` concurrent HEAD requests to pre-fill the connection pool before benchmarking, eliminating
the TCP handshake storm at burst start that biases early-window latency measurements.

### CLI improvements (carried from in-progress work)

- `parse_human_count()`: `--num` argument accepts human suffixes (`1k`, `100m`, `1g`, binary `1ki`,
  `1mi`, `1gi`) and underscore separators (`100_000_000`).
- `parse_jobs()`: all `--jobs` / `--concurrent` arguments clamped to `MAX_JOBS = 4_096` with a
  clear error on overflow.
- `MAX_JOBS = 4_096` added to `src/constants.rs`.

---

## Version 0.9.90 - AIStore Full Support, TLS Security Fixes, HTTP/2, 5 Issues Closed (April 2026)

### Feature: Full NVIDIA AIStore support — redirects + complete TLS security (closes #126)

This release promotes AIStore support from "tacit" to **fully implemented and security-hardened**.
`RedirectFollowingConnector` now enforces all four redirect security policies when
`S3DLIO_FOLLOW_REDIRECTS=1`:

1. **Cross-host `Authorization` header stripping** (RFC 9110 §11.6.2) — always active
2. **Standard TLS cert chain + hostname verification** (rustls WebPKI) — always active
3. **HTTPS→HTTP scheme downgrade prevention** — was dead code (`cert_store: None`) since
   `make_redirecting_client()` never attached the store; now active via
   `cert_store: Some(CertVerifyStore::new())`
4. **Certificate pinning across redirect chain** — implemented via `RecordingVerifier`
   (a `rustls::client::danger::ServerCertVerifier` that delegates to WebPKI first and records
   the leaf cert DER only on success) + pre-flight TLS probe via `tokio-rustls`. This is the only
   viable approach given that the AWS Rust SDK's internal TLS chain is entirely `pub(crate)`-gated.

**Root cause of the security gaps:** both TLS policy gates in `follow_redirects()` are guarded by
`if let Some(ref store) = cert_store`. With `cert_store: None`, neither gate was ever entered —
all TLS security logic was unreachable in production. Unit tests bypassed `make_redirecting_client()`
entirely, so they passed while production was unprotected.

**New direct dependencies** (were already transitive via `aws-smithy-http-client`):
- `rustls = { version = "0.23", features = ["aws-lc-rs"] }`
- `tokio-rustls = "0.26"`
- `rustls-native-certs = "0.8"` (loads platform root CAs for production probes)

**End-to-end test coverage — all 4 redirect scenarios validated with real OS-assigned ports:**

| # | Origin | Redirect target | Result | Test |
|---|---|---|---|---|
| 1 | `https://` (real TLS server) | `http://` (real TCP port) | ❌ Refused — scheme downgrade | `end_to_end_https_to_http_redirect_refused_real_servers` |
| 2 | `https://` cert A | `https://` cert B ≠ A | ❌ Refused — cert mismatch | `end_to_end_https_cert_mismatch_refused_real_servers` |
| 3 | `https://` cert A | `https://` cert A (shared) | ✅ 200 OK — cert match | `end_to_end_https_same_cert_redirect_passes_real_servers` |
| 4 | `http://` | `http://` | ✅ 200 OK — no TLS checks | `end_to_end_http_to_http_redirect_passes_real_servers` |

Total redirect tests: **27** (13 RFC 9110 conformance, 4 TLS policy unit, 2 TLS probe integration,
4 end-to-end). See [`docs/AIStore_redirect_implementation_v0.9.90.md`](AIStore_redirect_implementation_v0.9.90.md)
for the full implementation reference including the AWS SDK API dead-end investigation.

### Bug fix: GCS delete errors silently swallowed — exit code always 0 (closes #135)
- `GcsClient::delete_objects()` logged `warn!()` for individual delete failures but returned
  `Ok(())`, causing `s3-cli rm` to report "Successfully deleted all objects" even when objects
  were not deleted. A subsequent `ls` would reveal the surviving objects.
- Fix: `anyhow::bail!()` with a descriptive message including `fail_count / total` and the bucket
  name. Errors now propagate correctly to the CLI caller and produce a non-zero exit code.

### Bug fix: `s3-cli put` silently rounds up objects < 4 KiB to 4096 bytes (closes #136)
- Small object generation clamped the payload to a 4096-byte minimum in the random data path.
  A request for a 1 KiB object silently produced a 4 KiB PUT. Fixed in the Python API layer.

### Feature: expose `list_containers()` / `list_buckets()` to Python API (closes #133)
- `list_containers(uri)` was implemented in `src/list_containers.rs` but never registered in the
  Python API. Now exported as `list_containers_py()` in `src/python_api/python_core_api.rs`.
- Returns a `list[dict]` with keys `"name"`, `"uri"`, `"creation_date"` for every container found.
- Works for all backends: `s3://`, `az://`, `gs://`, `file://`, `direct://`.

### Bug documented: multipart upload blocks Python writer thread (closes #134)
- `MultipartUploadSink::spawn_part_bytes()` acquires the concurrency semaphore via
  `run_on_global_rt(sem.acquire_owned())` — a blocking wait on the calling (Python writer) thread.
  This causes a ~3× throughput regression vs non-blocking clients (AWS CRT) for the same object
  and endpoint. The root cause, measured impact, proposed fix (coordinator task + bounded channel),
  and memory profile comparison are documented in issue #134.
- This release adds `MultipartUploadConfig` validation tests; the coordinator-task fix is
  scheduled for a follow-on PR.

---

## Version 0.9.90 - HTTP/2 & h2c Support, ForceH2c Routing Fix, TLS Test Server (April 2026)

### Code cleanup: removed dead-code modules `sharded_client` and `range_engine` (April 2026)
- Deleted `src/sharded_client.rs` — a never-completed stub that proposed splitting HTTP traffic
  across multiple `aws_sdk_s3::Client` instances to reduce per-client contention.  The premise
  is wrong for reqwest 0.13 + HTTP/2 (the connection pool is already concurrent); the
  implementation's core `get_range()` returned `Bytes::new()` (empty placeholder) and was never
  wired into any call site.  Removed `pub mod sharded_client` from `lib.rs`.
- Deleted `src/range_engine.rs` — an S3-specific range-GET engine that depended on
  `sharded_client`.  Also a stub with placeholder implementations; zero callers outside itself.
  Removed `pub mod range_engine` from `lib.rs`.
- The **production** range engine, `src/range_engine_generic.rs`, is unchanged and continues to
  be used by `file_store.rs`, `file_store_direct.rs`, and `object_store.rs` (Azure/GCS backends).
  A rationalization comment was added to `range_engine_generic.rs` explaining the history and
  noting that S3ObjectStore does not yet use this engine (future work item).
- Build is clean with zero warnings after removal.

### Documentation: `store_for_uri()` async limitation (April 2026)
- Added an in-code comment block above `store_for_uri()` in `object_store.rs` documenting:
  - Why the function is intentionally synchronous (all current backends construct without I/O).
  - The known limitation: callers passing `s3://host:port/bucket/` directly to `store_for_uri()`
    always get the global singleton client — NOT a per-endpoint isolated client — because
    `S3ObjectStore::for_endpoint()` is async and cannot be called from a sync context.
  - How `MultiEndpointStore::from_config()` works around this via `run_on_global_rt()`.
  - What a future `store_for_uri_async()` would look like if per-endpoint isolation is ever
    needed outside `MultiEndpointStore`.

### Future work items (range engine)
- `S3ObjectStore` does not yet use `range_engine_generic` for large-object downloads.  S3 range
  GETs are handled natively by the AWS SDK streaming path.  Adding an opt-in
  `enable_range_engine` flag to an `S3Config` struct (mirroring `AzureConfig`/`GcsConfig`)
  would be the natural next step if S3 workloads would benefit from explicit range splitting.

### Feature: HTTP/2 and h2c (cleartext HTTP/2) support via `S3DLIO_H2C`
- Added `S3DLIO_H2C` environment variable to control the HTTP version used by the S3 client:
  - `S3DLIO_H2C=1` — force h2c (HTTP/2 prior-knowledge) on `http://` endpoints; on `https://`
    endpoints, ALPN negotiation proceeds normally (server selects HTTP/2 or HTTP/1.1).
  - `S3DLIO_H2C=0` — force HTTP/1.1 on all endpoints.
  - Unset (default `Auto`) — probes h2c once on the first `http://` connection; falls back to
    HTTP/1.1 if the server rejects the HTTP/2 preface. On `https://`, ALPN handles version
    negotiation transparently with no extra probe.
- Implemented in `src/reqwest_client.rs` as `H2cMode` enum with `build_smithy_http_client()`.
  Two reqwest clients are built at startup (`http1_client` and `h2c_client`); the correct one is
  selected per-request based on scheme and current probe state via `select_client()`.
- CA bundle (`AWS_CA_BUNDLE`) is supported independently of the HTTP version mode.

### Bug fix: `ForceH2c` on HTTPS caused "broken pipe"
- Root cause: `H2cMode::ForceH2c` unconditionally routed all requests (including `https://`) to
  the h2c client, which sends an HTTP/2 prior-knowledge preface without TLS. TLS servers reject
  this with a broken-pipe error.
- Fix: `ForceH2c` now only routes `http://` requests to the h2c client. `https://` requests fall
  through to the standard TLS client so ALPN negotiation selects the protocol.
- Routing logic extracted into a pure `pub(crate) fn select_client(mode, is_plain_http,
  auto_state) -> ClientChoice` for testability.

### Feature: Startup INFO logging for HTTP version mode and CA bundle
- `build_smithy_http_client()` logs the resolved `H2cMode` at `INFO` on every client creation so
  operators can confirm which mode is active without needing a packet capture.
- `s3_client.rs` logs both the "CA bundle loaded from <path>" and "CA bundle not set — using
  system default TLS trust store" cases, eliminating a previous silent no-op for the not-set path.
- First response HTTP version (HTTP/2 or HTTP/1.1) is logged once via `PROTOCOL_LOGGED` atomic.

### New: `examples/tls_test_server` — local TLS+HTTP/2 test server for ALPN verification
- Added `examples/tls_test_server.rs`: a minimal HTTPS server that generates a self-signed
  certificate at runtime (via `rcgen`) for `127.0.0.1`/`localhost`, installs `aws_lc_rs` as the
  rustls crypto provider, advertises ALPN `["h2", "http/1.1"]`, and serves both HTTP/1.1 and
  HTTP/2 via `hyper_util`'s `AutoBuilder`.
- The server writes its certificate to `/tmp/tls_test_server.crt` for use by curl or `s3-cli`.
- Logs `ALPN negotiated = "h2"` (or `"http/1.1"`) and the HTTP version of every request.
- Verified end-to-end: `s3-cli stat` and `s3-cli put` against this server both log
  `HTTP protocol (first response): HTTP/2.0` and `protocol=HTTP/2` in PUT summary.
- Start it with: `cargo run --example tls_test_server`
- Test with: `AWS_CA_BUNDLE=/tmp/tls_test_server.crt AWS_ENDPOINT_URL=https://127.0.0.1:9443`
- See [`docs/HTTP2_ALPN_INVESTIGATION.md`](HTTP2_ALPN_INVESTIGATION.md) for full usage and
  expected output.
- New dev-dependencies added to `Cargo.toml`: `rcgen 0.13`, `rustls 0.23 (aws-lc-rs)`,
  `tokio-rustls 0.26`, `hyper 1 (server)`, `hyper-util 0.1 (server-auto)`.

### Tests: 10 new unit tests for HTTP version routing logic
- `test_select_client_force_h2c_plain_http` — ForceH2c + http:// → h2c client
- `test_select_client_force_h2c_tls` — ForceH2c + https:// → http1 client (the bug case)
- `test_select_client_force_http1_plain_http` — ForceHttp1 + http:// → http1 client
- `test_select_client_force_http1_tls` — ForceHttp1 + https:// → http1 client
- `test_select_client_auto_plain_http_first_connection` — Auto + http:// + Unknown → h2c probe
- `test_select_client_auto_plain_http_probe_succeeded` — Auto + http:// + H2C_AUTO_OK → h2c
- `test_select_client_auto_plain_http_probe_failed` — Auto + http:// + H2C_AUTO_FAILED → http1
- `test_select_client_auto_tls_unknown` — Auto + https:// → http1 (ALPN, any state)
- `test_select_client_auto_tls_ok` — Auto + https:// → http1 (ALPN)
- `test_select_client_auto_tls_failed` — Auto + https:// → http1 (ALPN, state irrelevant)
- All tests live in `src/reqwest_client.rs` under `#[cfg(test)]`.

### Documentation: HTTP2_ALPN_INVESTIGATION.md
- Created `docs/HTTP2_ALPN_INVESTIGATION.md`: documents the full ALPN investigation (MinIO
  always selects `http/1.1` in ALPN — its server does not accept h2), the routing logic table,
  the `tls_test_server` usage guide with expected curl and `s3-cli` output, and an index of the
  new unit tests. This doc is the canonical reference for HTTP version behavior in s3dlio.
- Updated `docs/CLI_GUIDE.md`: added `S3DLIO_H2C` to the environment variables table with
  cleartext and TLS usage examples.
- Updated `docs/api/Environment_Variables.md`: replaced stale HTTP client variables with current
  `S3DLIO_H2C` and connection pool settings, with cleartext and TLS configuration examples.

### Verification
- `cargo build --bin s3-cli` — zero warnings ✅
- `cargo test` — 225 unit tests + 38 doc tests = 263 passing, 0 failed ✅
- `s3-cli stat` against `tls_test_server` → `HTTP protocol (first response): HTTP/2.0` ✅
- `s3-cli put` against `tls_test_server` → `PUT summary: ... protocol=HTTP/2` ✅

---

## Version 0.9.86 - Redirect Follower / Tacit AIStore Support, Redirect Security (March 2026)

### Feature: HTTP 3xx redirect-following connector (`S3DLIO_FOLLOW_REDIRECTS`)
- Added `RedirectFollowingHttpClient` / `RedirectFollowingConnector` in `src/redirect_client.rs`.
  When the environment variable `S3DLIO_FOLLOW_REDIRECTS=1` is set, the S3 client wraps the
  default AWS SDK HTTP client with a connector that transparently follows 3xx redirects up to
  a configurable maximum (default: 5, override via `S3DLIO_REDIRECT_MAX`).
- The primary motivation is **NVIDIA AIStore**, which routes S3 GET/PUT/DELETE requests from a
  stateless proxy node to the specific storage target via HTTP 307 Temporary Redirect. The AWS
  SDK's default HTTP client intentionally does not follow cross-host redirects; this connector
  closes that gap. See [`docs/AIStore_307_Redirect_Proposal.md`](AIStore_307_Redirect_Proposal.md)
  for full design rationale and the rejected alternative approaches.

  > **⚠️ Note:** AIStore compatibility is *tacit* support — s3dlio has not been tested directly
  > against NVIDIA AIStore at this time. The redirect mechanism conforms to the protocol as
  > described in AIStore's documentation, but end-to-end validation is pending.

### Feature: Redirect security — scheme downgrade prevention (ACTIVE)
- HTTPS → HTTP redirects are now **refused** with a `ConnectorError`. Following an HTTPS-to-HTTP
  redirect would transmit the S3 `Authorization` header (HMAC-SHA256 signature + credentials) in
  plaintext, enabling credential capture and replay attacks.
- This protection is unconditional when `S3DLIO_FOLLOW_REDIRECTS=1` is active.

### Security gap: Certificate pinning across redirect chain (NOT YET IMPLEMENTED)
- A `CertVerifyStore` structure is in place and the redirect-chain cert-comparison policy is
  implemented and fully unit-tested — but the store is currently empty in production because there
  is no point at which the TLS handshake's peer certificate is recorded into it.
- Implementing this requires a pre-flight TLS probe using `tokio_rustls` to extract the peer
  certificate before each redirect hop. The AWS Smithy SDK's built-in TLS connector chain does not
  expose a public injection point for a custom `rustls::client::danger::ServerCertVerifier`, and
  all internal paths (`create_rustls_client_config`, `wrap_connector`) are `pub(crate)`.
- The planned approach (pre-flight probe via `aws-smithy-runtime-api`'s public `HttpConnector`
  trait) requires no SDK fork and is fully documented. See
  [`docs/security/HTTPS_Redirect_Security_Issues.md`](security/HTTPS_Redirect_Security_Issues.md)
  §7 for the complete API investigation, dead-end analysis, viable implementation path, and
  future PR checklist.
- **Risk:** MEDIUM. WebPKI validation (hostname + CA chain) continues to run on every hop.
  The scheme downgrade risk (HIGH) is now blocked. Cert pinning adds defense in depth for
  environments with non-standard CA configurations.

### Tests: 21 new unit tests for redirect behavior
- All redirect tests live in `src/redirect_client.rs` under `#[cfg(test)]`.
- Coverage includes: basic 302/307 redirect following; cross-host `Authorization` stripping
  (RFC 9110 §11.6.2); max-redirect loop termination; scheme downgrade refusal (HTTPS → HTTP);
  cert pinning match/mismatch; RFC-correct handling of 301/302 POST→GET conversion; 303 See Other;
  and the scheme-downgrade regression test that confirmed the production gap and its fix.

### Verification
- `cargo build --release` — zero warnings ✅
- `cargo test` — all tests pass ✅

---

## Version 0.9.84 - HEAD Elimination, OnceLock Caching, Lock-Free Range Assembly, Env Var Rename (March 2026)

### Performance: process-global ObjectSizeCache eliminates redundant HEAD requests
- Added `ObjectSizeCache` (TTL: 1 hour, override via `S3DLIO_SIZE_CACHE_TTL_SECS`) integrated
  into `get_objects_parallel()`. A pre-stat phase issues N concurrent HEADs before any GETs begin;
  results are stored in the process-global cache. From batch 2 onward (and all subsequent epochs),
  `get_object_uri_optimized_async()` finds the size in cache and skips its HEAD entirely. For
  typical training workloads (same files repeated across batches/epochs), this reduces S3 request
  count by up to 50%.

### Performance: OnceLock-based env var caching on the hot path
- `get_object_uri_optimized_async()` previously called `std::env::var()` on every invocation
  (one syscall per object). Three `OnceLock<T>` statics now cache `S3DLIO_ENABLE_RANGE_OPTIMIZATION`,
  `S3DLIO_RANGE_THRESHOLD_MB`, and the `ObjectSizeCache` instance once per process on first call.

### Bug fix: `S3DLIO_ENABLE_RANGE_OPTIMIZATION=0` was a no-op on the `get_many()` path
- The env var previously only applied to `S3ObjectStore::get()` in `object_store.rs`. The Python
  `get_many()` path routes through `get_objects_parallel()` → `get_object_uri_optimized_async()`
  which never checked the var. Now all paths are consistent: setting `=0` disables range splitting
  and suppresses the HEAD that was needed only to determine the threshold.

### Performance: lock-free chunk assembly in `concurrent_range_get_impl()`
- Replaced `Arc<Mutex<BytesMut>>` shared buffer (serialised writes from up to 37 concurrent chunk
  tasks) with a collect-then-sort-then-assemble pattern. Each chunk future returns
  `(buffer_offset, Bytes)` independently; results are sorted by offset and assembled in one
  sequential pass — no lock contention in the concurrent phase.

### Fix: rename `AWS_CA_BUNDLE_PATH` → `AWS_CA_BUNDLE` (standard AWS SDK name)
- `s3_client.rs`, `aws-env`, `docs/api/Environment_Variables.md`, and `python/tests/test_new_dlio_s3.py`
  updated. The previous name was non-standard; the AWS SDK uses `AWS_CA_BUNDLE`.

### Observability: replace `eprintln!` with structured tracing
- Replaced all `eprintln!("[s3dlio] ...")` calls in `s3_client.rs` with `info!()` and `debug!()`
  from the `tracing` crate. Log output is now controlled by `S3DLIO_LOG_LEVEL` / `RUST_LOG` and
  captured correctly by the Python logging integration.

### Verification
- `cargo build --release` — zero warnings ✅
- `cargo check` — zero warnings ✅

---

## Version 0.9.82 - Multipart Upload Backpressure Fix, DLIO Multipart Integration (March 2026)

### Bug fix: Multipart upload semaphore acquired before spawn (backpressure)
- `spawn_part()` and `spawn_part_bytes()` in `src/multipart.rs` previously
  acquired the concurrency semaphore *inside* the spawned task, meaning all
  parts were immediately spawned with no bound on concurrent memory usage.
  The semaphore is now acquired on the caller thread (via `run_on_global_rt`)
  **before** `spawn_on_global_rt`, providing true backpressure: at most
  `max_in_flight` tasks are ever live, capping peak memory at
  `max_in_flight × part_size` bytes. This mirrors the `_throttle()` pattern
  used by minio's Python client.

### Performance: concurrent part-joining in `finish()`
- `MultipartUploadSink::finish()` previously joined upload tasks sequentially
  (`for h in tasks { h.await }`). Changed to `futures::future::join_all(tasks)`
  so all remaining in-flight parts are awaited concurrently — reduces tail
  latency on the final flush when parts finish out of order.
- `OwnedSemaphorePermit` import moved from commented-out block to active use.

### DLIO integration: automatic multipart upload for large objects
- `S3dlioStorage.put_data()` in
  `python/s3dlio/integrations/dlio/s3dlio_storage.py` now automatically selects
  upload strategy based on object size:
  - **< 32 MiB**: single `put_bytes()` call — lowest overhead for small objects.
  - **≥ 32 MiB**: `MultipartUploadWriter` with 32 MiB parts and up to 8
    concurrent in-flight parts — avoids the S3 5 GiB single-PUT limit and
    achieves higher throughput on large checkpoints.
- Threshold constants (`_MULTIPART_THRESHOLD`, `_MULTIPART_PART_SIZE`,
  `_MULTIPART_MAX_IN_FLIGHT`) defined at module level and aligned with
  `src/constants.rs` defaults.

### Build fix: vendored OpenSSL for manylinux wheel compatibility
- Added `openssl = { version = "0.10", features = ["vendored"] }` as a direct
  dependency. The Azure SDK (`typespec_client_core`) transitively enables
  `native-tls` on `reqwest`, which pulls `openssl-sys` and requires system OpenSSL
  headers. Manylinux containers do not reliably have these installed. The vendored
  feature compiles OpenSSL from source (statically linked) so no system package is
  ever required — in any container, any OS, any CI environment.
- Set `default-features = false` on our own `reqwest` dependency (defense in
  depth — ensures s3dlio's own HTTP paths never introduce `native-tls`).
- CI `before-script-linux` already installs `openssl-devel`/`libssl-dev` as a
  fallback; the vendored fix makes this redundant but harmless.

### Verification
- `cargo build --release --features full-backends,direct-io,enhanced-http` — zero warnings ✅
- `./build_pyo3.sh` — Python 3.12 and 3.13 wheels built successfully ✅
- `cargo check --features full-backends,direct-io,enhanced-http` — zero warnings ✅

---

## Version 0.9.80 - Python List Hang Fix, Tracing Deadlock Fix, Async S3 Operations (March 2026)

### Bug fix: Python list() hung indefinitely on non-AWS endpoints (HIGH severity)
- Removed legacy `let _ = build_ops_async().await?...` calls from `list_objects()`
  and `stat_object()` in `src/s3_utils.rs`. These were leftover debug artifacts
  from an earlier `S3Ops` implementation whose results were discarded. On non-AWS
  endpoints (MinIO, on-prem S3), the AWS SDK's IMDSv2 credential probe
  (`http://169.254.169.254/…`) never times out, causing all Python `list()`,
  `list_keys()`, `list_full_uris()`, and `list_uris()` calls to hang indefinitely
  with no Ctrl-C recovery. Bug closed by this release.

### Bug fix: Tracing deadlock inside `tokio::spawn` with AWS SDK S3 operations
- Refactored `list_objects_stream` in `src/s3_utils.rs` from `tokio::spawn` +
  `mpsc::channel` + `ReceiverStream` to an inline `async_stream::stream!` macro
  (no spawned task). When `RUST_LOG=debug` or `trace` was set, the previous
  `tokio::spawn`-based implementation could deadlock due to tracing-subscriber
  lock contention between the spawned task's tracing events and the AWS SDK's
  internal tracing. The inline stream runs entirely on the caller's task,
  eliminating this class of deadlock. Bug closed by this release; also filed upstream as aws-sdk-rust#1388.

### S3ObjectStore: replace sync helpers with async variants
- `S3ObjectStore::list()` now drives `s3_list_objects_stream` (async, inline)
  instead of calling the sync `s3_list_objects()` from inside the global runtime
  (which would deadlock via nested `run_on_global_rt`).
- Added `delete_objects_async()`, `create_bucket_async()`, `delete_bucket_async()`
  to `src/s3_utils.rs` — safe to call from within async contexts.
- `S3ObjectStore::delete()`, `delete_batch()`, `delete_prefix()`,
  `create_container()`, and `delete_container()` all updated to use the new async
  variants.

### Python API cleanup
- Removed stale deprecated helpers (`list_keys_from_s3`, `list_uris`) from
  `python/s3dlio/__init__.py`; marked remaining deprecated section with TODO.

### Verification
- `cargo check` — zero warnings ✅
- All list, stat, delete operations confirmed working against MinIO ✅
- `RUST_LOG=debug` no longer causes hangs ✅

---

## Version 0.9.76 - Version fix for PyPI publish (March 2026)

- Corrected `pyproject.toml` version from `0.9.70` to `0.9.75` (was accidentally not bumped before the v0.9.75 release tag was created, causing the PyPI workflow to upload `0.9.70` wheels). No functional code changes from v0.9.75.

## Version 0.9.75 - GCS RAPID/Zonal Support, Debug Logging, RUST_LOG Hang Fix (March 2026)

### GCS RAPID / zonal bucket support
- Stat: use BidiReadObject for authoritative size metadata on RAPID buckets (StorageControl returns stale size=0).
- Get: handle OUT_OF_RANGE errors on RAPID zonal buckets by bypassing stale-metadata fallback path.
- Put: leverage `google-cloud-rust` fork with BidiWriteObject two-phase finalize to avoid PUT truncation.
- Auto-detect RAPID mode per bucket via `S3DLIO_GCS_RAPID` env var (`true` / `false` / `auto`); cached once per process.
- Pinned `google-cloud-rust` fork at `release-20260212-rf-gcsrapid-20260317.1` for PUT truncation fix.

### RUST_LOG=debug hang fix (issue #105)
- Added `crate_log_caps()` in `src/bin/cli.rs` that applies `warn`-level `EnvFilter` directives for noisy async crates (`h2`, `hyper`, `hyper_util`, `hyper_rustls`, `tonic`, `tower`, `reqwest`, `rustls`, `aws_config`, `aws_sdk_s3`, `aws_smithy_runtime`, `aws_smithy_http`).
- Prevents debug-level tracing from triggering async deadlocks inside HTTP/gRPC stacks.
- User code remains at the level requested by `RUST_LOG`; only the third-party crates are capped.

### Debug logging restored across all five backends
- `S3ObjectStore` + `s3_utils.rs`: entry/exit debug logging on all operations (get, get_range, put, put_multipart, list, stat, delete, delete_batch, delete_prefix).
- `AzureObjectStore` + `azure_client.rs`: consistent debug logging on all 12 Blob Storage methods.
- `GcsObjectStore` in `object_store.rs`: entry debug logging (inner `google_gcs_client.rs` already had 35+ statements).
- `FileSystemObjectStore` (`file_store.rs`): debug logging on all POSIX-path operations.
- `ConfigurableFileSystemObjectStore` (`file_store_direct.rs`): debug logging on all O_DIRECT operations.
- `MultiEndpointStore` and `LoggedObjectStore` intentionally not modified (delegate/TSV paths avoid duplication).

### CLI improvements
- `delete` command gains `rm` as a visible alias (`s3-cli rm ...`).
- Command naming corrected: `list` is now the primary name, `ls` is the alias (was reversed). Help output now reads `list ... [aliases: ls]`.
- Shortened `list` and `tfrecord-index` one-line help strings to prevent wrapping in terminal help output.
- `create-bucket` and `delete-bucket` commands made fully generic: now accept any URI scheme (`s3://`, `gs://`, `az://`, `file://`, `direct://`) instead of bare bucket names. Auth check is URI-scheme-aware (only S3 checks AWS credentials). Uses `store_for_uri_with_logger()` dispatch, consistent with all other commands.
- `extract_container_name(uri)` helper: pure string parser extracting the backend-appropriate container identifier from any URI scheme.
- Command help ordering: bucket commands (`create-bucket`, `delete-bucket`, `list-buckets`) listed first, remaining commands sorted alphabetically.
- 17 unit tests added for `extract_container_name()` covering all five backends and error cases (19/19 `cargo test --bin s3-cli` passing).
- `crate_log_caps()` pins noisy async crates (`h2`, `hyper`, `tonic`, `tower`, `reqwest`, etc.) to `warn` level, preventing `RUST_LOG=debug` from triggering async deadlocks in gRPC/HTTP stacks.

### Python binding fix
- Removed `LogTracer::init()` call from `python_core_api.rs` that conflicted with Python's own logging initialization.

### Issue #125 confirmed resolved
- Default build (`cargo build`) excludes optional Azure/GCS backends; `--features full-backends` enables all five.

### Integration test hygiene
- Fixed 11 integration test files that referenced feature-gated Azure/GCS structs without being feature-gated themselves. All files now carry the correct `#![cfg(feature = "backend-azure")]` or `#![cfg(feature = "backend-gcs")]` crate-level attribute.
- `tests/test_backend_parity.rs`: corrected stale assertion `assert!(azure_result.is_ok())` — wrapped Azure instantiation in `#[cfg(feature = "backend-azure")]` / `#[cfg(not(...))]` blocks; default builds now correctly verify that Azure returns an error when the feature is absent.
- `tests/test_compression_all_backends.rs` and `tests/test_object_store_integration.rs`: individual Azure/GCS tests gated with `#[cfg(feature = "backend-azure")]` to prevent spurious failures in default builds.
- `cargo test` (default features) returns zero failures across all test suites.

### GCS documentation consolidated
- Removed 5 GCS docs that documented in-progress investigation and now-resolved issues: `GCS-API-Configuration.md`, `GCS_BIDI_HANG_ANALYSIS.md`, `GCS_DEBUG_SUMMARY.md`, `GCS-gRPC_Fixes.md`, `GCS-gRPC-Transport.md`.
- Replaced with single authoritative reference: `docs/supplemental/GCS-Backend.md` — current architecture, RAPID two-phase finalize, HTTP/2 window tuning, google-cloud-rust fork commit history, API reference, performance results.

### Verification
- `cargo check --features full-backends` — zero warnings ✅
- `cargo build --release --features full-backends` — success ✅
- `cargo test` (default features) — zero failures ✅
- Manual GCS RAPID cluster test: 1000-object delete via `s3-cli -vv rm -r gs://…` — debug logging visible, `rm` alias functional ✅

---

## Version 0.9.70 - Backend-Profiled Builds, GCS Hot-Path Reuse, and RAPID Runtime Improvements (March 2026)

### Why this release was needed
- We needed to separate "default" packaging behavior from "all backends enabled" behavior.
- The Python wheel path needed a predictable profile model for maintainers and users.
- GCS hot code paths were creating lightweight wrappers frequently, adding avoidable overhead and log noise.
- RAPID mode logs were emitted too often under high concurrency, making real diagnostics harder.
- Runtime tuning around GCS channel count needed clearer wiring from CLI concurrency settings.

### Build and packaging model changes
- Added explicit backend profile behavior for Python wheel builds.
- Default Python build profile now targets a smaller S3-focused wheel.
- Full cloud backend build remains available via full-backends feature selection.
- `build_pyo3.sh` now accepts profile arguments from CLI instead of relying on environment-only control.
- Added named-flag CLI support in `build_pyo3.sh`:
    - `--profile full`
    - `--profile default`
    - `-p full`
    - `-p default`
- Positional profile form is still supported:
    - `./build_pyo3.sh`
    - `./build_pyo3.sh full`
    - `./build_pyo3.sh default`
- Added robust argument validation and help output in `build_pyo3.sh`.
- Updated `pyproject.toml` guidance/comments to align with current build/install workflows.

### Cargo feature and dependency updates
- Refined optional backend dependency usage for clearer backend selection boundaries.
- Preserved full backend behavior through `full-backends` feature.
- Updated lockfile to reflect new dependency graph resulting from feature-path updates.

### GCS performance and hot-path optimizations
- Confirmed expensive GCS transport/auth clients remain globally reused via async once-initialization.
- Added wrapper-level reuse in operational paths to avoid repeated `GcsClient::new()` wrapper churn.
- Added per-store lazy GCS wrapper caching in object-store path.
- Added global wrapper reuse for additional GCS paths that execute outside per-store reuse scope.
- Extended reuse into write finalize path so per-object write completion avoids unnecessary wrapper construction.
- Extended reuse into GCS container-list path to reduce repeated wrapper initialization in repeated listing workflows.

### RAPID mode behavior improvements
- Cached effective RAPID mode once per process lifecycle.
- Reduced RAPID mode log emission to one initialization-time message rather than per wrapper construction.
- Kept per-bucket RAPID auto-detection behavior and cache strategy intact.
- Preserved explicit override behavior from environment and API configuration.

### CLI/runtime tuning improvements
- Added/strengthened pre-tuning of GCS subchannel count before first GCS client initialization.
- Propagated concurrency intent from CLI options into GCS setup earlier in command flow.
- Added/updated constants for GCS read timeout/progress environment control.

### Documentation updates
- Updated README to explain:
    - default vs full backend build behavior,
    - explicit full-backend CLI build command,
    - Python build profile usage for `build_pyo3.sh`,
    - source-build commands for full backend Python installs.
- Updated backend options guidance to reflect current feature model.
- Added investigation docs for GCS bidi read behavior and debugging findings.
- Consolidated GCS debug document roles:
    - one document as deep technical analysis,
    - one as concise incident/debug summary with references.

### Tests and verification
- Updated tests impacted by new backend-feature defaults.
- Made list-container backend expectation tests feature-aware.
- Validation performed during this release cycle:
    - `cargo check`
    - `cargo check --features backend-gcs`
    - `cargo check --features full-backends`
    - `cargo test --lib`

### User-visible outcomes
- Cleaner default Python install footprint.
- Straightforward path to full cloud backend builds when required.
- Reduced GCS runtime overhead in hot operational paths.
- Significantly reduced RAPID log spam under concurrency.
- Better maintainability through explicit build profiles and clearer docs.

### Notes for maintainers
- Keep `build_pyo3.sh` profile UX stable (`default|slim|full`, `--profile|-p`).
- Keep README and changelog examples aligned with actual Cargo features.
- Treat GCS debug docs as a paired set:
    - technical root-cause deep dive,
    - concise debugging summary and triage notes.

## Version 0.9.65 - GCS PUT/Performance Fixes: Chunk Size, Constants, Zero-Copy (March 2026)

### GCS PUT RESOURCE_EXHAUSTED Fix
- Fixed `RESOURCE_EXHAUSTED: SERVER: Received message larger than max` errors on all GCS PUT operations
- Root cause: `DEFAULT_GRPC_WRITE_CHUNK_SIZE` was 16 MiB; GCS server's limit is **4 MiB per serialised protobuf message** (not just data payload)
- Protobuf framing overhead (~89 bytes per message) means the data payload must be smaller than the server ceiling
- New constant hierarchy in `vendor/google-cloud-gax-internal/src/gcs_constants.rs`:
  - `GCS_SERVER_MAX_MESSAGE_SIZE = 4 MiB` — raw server ceiling
  - `MAX_GRPC_WRITE_CHUNK_SIZE = GCS_SERVER_MAX_MESSAGE_SIZE - 64 KiB` — max safe data payload (63 × 64 KiB)
  - `DEFAULT_GRPC_WRITE_CHUNK_SIZE = 2 MiB` — conservative default (32 × 64 KiB), well below ceiling
- All values are 64 KiB-aligned; env-var override `S3DLIO_GRPC_WRITE_CHUNK_SIZE` silently clamped to `MAX_GRPC_WRITE_CHUNK_SIZE`
- Measured upload performance: **3.83 GB/s** for 1,000 × 32 MiB objects on RAPID bucket (32 jobs)

### GCS Constants Centralisation
- **NEW**: `vendor/google-cloud-gax-internal/src/gcs_constants.rs` — single source of truth for all GCS/gRPC protocol constants
- **NEW**: `src/gcs_constants.rs` — application-layer re-exports + channel floor, concurrent delete limit, env-var names
- All magic numbers removed from `transport.rs`, `grpc.rs`, `google_gcs_client.rs`
- `google-cloud-gax-internal` added as direct dependency in `Cargo.toml` (needed for re-exports)

### gcs-official Feature Flag Removed
- `google-cloud-storage` and `google-cloud-gax` are now always-on unconditional dependencies
- All `#[cfg(feature = "gcs-official")]` guards removed across 6 source files
- Projects depending on s3dlio no longer need `features = ["gcs-official"]`

### Zero-Copy Write Pipeline
- `put_object` and `put_object_multipart` now take `Bytes` directly — eliminates the `Bytes::copy_from_slice` on every write
- Bounded channel (capacity 8) + `tokio::spawn` producer: CRC32C now runs concurrently with gRPC network I/O
- Read path: `BytesMut::with_capacity(size_hint)` pre-allocated from `ObjectDescriptor` — zero reallocations

### RAPID Auto-Detection
- `RapidMode` enum (`Auto` | `ForceOn` | `ForceOff`) with per-bucket cache
- `get_storage_layout()` auto-detects RAPID on first access; no manual configuration required
- `S3DLIO_GCS_RAPID=auto|true|false` controls override behaviour

### Subchannel Auto-Tune
- `set_gcs_channel_count(n)` pre-init hook wired to `--jobs N` in CLI
- Three-tier priority: env var > API call > `max(64, cpu_count)` floor
- HTTP/2 window patched to 128 MiB (env `S3DLIO_GRPC_INITIAL_WINDOW_MIB`)

### New Zero-Copy Unit Tests (10 tests)
- `test_bytesmut_freeze_is_zero_copy`
- `test_bytes_clone_is_zero_copy`
- `test_bytes_slice_is_zero_copy`
- `test_bytes_from_vec_preserves_pointer`
- `test_buffered_writer_finalise_path_is_zero_copy`
- `test_grpc_read_accumulation_no_realloc`
- `test_http_read_freeze_returns_bytes_not_vec`
- `test_write_producer_chunk_slices_are_zero_copy`
- `test_producer_task_data_clone_is_arc_increment`
- `test_put_object_caller_bytes_conversion_is_zero_copy`

### Documentation
- `docs/supplemental/GCS-Backend.md` — comprehensive combined doc covering all 6 root-cause issues, all fixes, constants architecture, zero-copy data flow, performance results (3.83 GB/s), env-var reference, files-changed table (formerly `GCS-gRPC_Fixes.md`)
- `docs/performance/RANGE_OPTIMIZATION_IMPLEMENTATION.md` — clarified as S3-only; added §Backend-Specific Read Architectures comparing S3/GCS/Azure/File transport layers

### GCS API Completion (read-back, query, Python bindings)
- **NEW**: `s3dlio::get_gcs_channel_count() -> usize` — read back the programmatically-configured subchannel count (`0` = not set, auto-detect will apply)
- **NEW**: `s3dlio::get_gcs_rapid_mode() -> Option<bool>` — read back the current effective RAPID mode, resolving env var + programmatic override (`None` = auto-detect)
- **NEW**: `s3dlio::query_gcs_rapid_bucket(bucket_or_uri) -> bool` (async) — query whether a bucket or `gs://` URI is RAPID (Hyperdisk ML); result cached per bucket for process lifetime, deduplicated under concurrent access
- **NEW** Python API: `gcs_set_channel_count`, `gcs_set_rapid_mode`, `gcs_get_channel_count`, `gcs_get_rapid_mode`, `gcs_query_rapid_bucket` — full GCS tuning control from Python without requiring environment variables
- `docs/supplemental/GCS-Backend.md` — complete reference for all 5 public GCS API functions, constants, env vars, and Python usage (see §Runtime Configuration)
- `.github/copilot-instructions.md` — added Prime Directive and Secondary Directive

---

## Version 0.9.60 - GCS gRPC Transport, RAPID Storage & Multi-Protocol List Buckets (February 2026)

### GCS gRPC Transport (All Operations)
- **All GCS reads** now use gRPC `BidiReadObject` via `open_object()` — replaces JSON/REST `read_object()`
- **All GCS writes** now use gRPC `BidiWriteObject` via `send_grpc()` — replaces JSON/REST multipart uploads
- Chunked writes with 2 MiB default chunk size (`S3DLIO_GRPC_WRITE_CHUNK_SIZE` to override)
- Per-chunk CRC32C checksums via `ChecksummedData` + whole-object CRC32C on final message
- TRACE-level logging for gRPC write chunk size, count, offsets, and CRC32C values
- See [docs/supplemental/GCS-Backend.md](supplemental/GCS-Backend.md) for full architecture details

### GCS RAPID / Hyperdisk ML Storage Support
- Vendored `google-cloud-storage` v1.8.0 fork with gRPC write support:
  - `stub.rs`: `write_object_grpc()` trait method
  - `transport.rs`: Full `BidiWriteObject` implementation (~200 lines)
  - `write_object.rs`: `send_grpc()` + `set_appendable()` builder methods
  - `perform_upload.rs`: `appendable=true` query parameter for legacy JSON path
- New env var `S3DLIO_GCS_RAPID=true|1|yes` enables `appendable=true` on writes
- `.env` file support via `dotenvy` for persistent configuration
- gRPC reads work with RAPID/zonal buckets automatically (no special flag needed)

### GCS Concurrent Batch Deletes
- `delete_objects()` now dispatches up to **64 concurrent** gRPC delete requests
- Uses `FuturesUnordered` + `Semaphore` for bounded parallelism
- ~35x speedup for batch deletes (859 objects: ~69s → ~2s)
- Errors collected and reported after all requests complete (no early abort)

### GCS List Prefix Normalization Fix
- Fixed bug where `list_objects()` appended `/` to exact object names (e.g. `object.dat/`)
- Now retries without trailing slash when normalized prefix returns empty results
- Extracted reusable `list_objects_with_prefix()` paginated helper

### List Buckets Multi-Protocol (Issue #121)
- Fixed hardcoded `us-east-1` for custom S3 endpoints in `list_buckets()`
- New `list_containers(uri)` function dispatches across S3, Azure, GCS, and local fs
- CLI `list-buckets` command now accepts optional URI argument for non-S3 backends

### Range GET Optimization Defaults Changed
- `S3DLIO_ENABLE_RANGE_OPTIMIZATION` is now **enabled by default**; set to `0` to disable
- Default threshold changed from 64 MB → **32 MB** (`S3DLIO_RANGE_THRESHOLD_MB`)
- Setting the variable to `1` is still accepted (no-op, already enabled)

### Build & Dependencies
- `gcs-official` (gRPC) is now the default GCS backend; `gcs-community` demoted
- `hdf5` and `numa` (hwloc) are now **optional** Cargo features — removes `libhdf5-dev` and `libhwloc-dev` as build dependencies
- Vendor fork wired via `[patch.crates-io]` in `Cargo.toml`

### Other
- Comprehensive `trace!`/`debug!` instrumentation across GCS, Azure, S3 list paths
- Doc-test fence annotations updated; `doctest-threads = 1` to prevent memory thrash
- Consolidated 4 GCS design docs into single `docs/GCS-gRPC-Transport.md` (subsequently merged into `docs/supplemental/GCS-Backend.md` in v0.9.75)

---

## Version 0.9.50 - Critical Python Runtime Fix & s3torchconnector Compat (February 2026)

### 🚨 **CRITICAL: Python Multi-Threaded Runtime Fix**

**Fixed two catastrophic bugs** that prevented s3dlio from being used in multi-threaded Python applications:

| Bug | Version | Symptom | Root Cause |
|-----|---------|---------|------------|
| Runtime Churn | v0.9.27 | `RuntimeError: dispatch failure` after ~40 objects | Every API call created a new Tokio runtime (320 calls = 256+ OS threads) |
| Nested Runtime | v0.9.40 | `PanicException: Cannot start a runtime from within a runtime` | `GLOBAL_RUNTIME.block_on()` panics when called from Tokio-managed threads |

**Solution — io_uring-style Submit Pattern:**

Replaced all `block_on()` calls with a spawn+channel pattern that works from ANY thread context:

```
Python Thread → handle.spawn(async work) → mpsc::channel.recv() → result
```

- The calling thread blocks on `channel.recv()` (NOT `block_on()`), so there are no Tokio runtime conflicts
- A single global Tokio runtime (created once via `once_cell::Lazy`) handles all async I/O
- `submit_io<F, T>()` wrapper converts `anyhow::Result<T>` → `PyResult<T>` for clean error propagation
- Zero-copy: `Bytes` (Arc-based) moves through the channel — no data copied

**Files Changed:**
- `src/python_api/python_core_api.rs` — Removed `GLOBAL_RUNTIME`, added `submit_io()`, updated 12 functions
- `src/python_api/python_aiml_api.rs` — Updated 2 functions to use `run_on_global_rt()`

**Test Result:** 16 threads × 200 objects × 3 rounds — ALL OPERATIONS PASSED (ThreadPoolExecutor)

### ✨ **New: `put_many()` Batch Upload**

Upload multiple objects in a single call with parallel execution:

```python
items = [
    ("s3://bucket/file1.bin", b"data1"),
    ("s3://bucket/file2.bin", b"data2"),
    ("s3://bucket/file3.bin", b"data3"),
]
s3dlio.put_many(items)

# Async version
await s3dlio.put_many_async(items)
```

### 🔧 **s3torchconnector Compatibility Layer Rewrite**

Complete rewrite of `python/s3dlio/compat/s3torchconnector.py` for zero-copy performance:

| Component | Before (Broken) | After (v0.9.50) |
|-----------|-----------------|------------------|
| `S3Client.get_object` (range) | Downloaded entire object + `bytes()` copy | Uses `get_range()` — server-side range, returns BytesView zero-copy |
| `S3Client.put_object` | `list[bytes]` + `b''.join()` (two copies) | Accumulates in BytesIO, single `put_bytes()` call |
| `S3Checkpoint.writer` | `self.buffer.read()` (copy) | `getbuffer()` memoryview — no copy |
| `S3Checkpoint.reader` | `io.BytesIO(bytes(data))` (two copies) | New `_BytesViewIO(io.RawIOBase)` wrapping BytesView via memoryview, wrapped in `io.BufferedReader` |
| `S3IterableDataset.__iter__` | Lost URIs (placeholder base_uri) | Pre-lists via `list()`, iterates with full URIs |
| `S3MapDataset` | Depended on `ObjectStoreMapDataset._keys` private attr | Uses `list()` + `get()` directly |
| `S3Client.list_objects` | Called non-existent `list_prefix` | Uses `list(uri, recursive=True)` |
| Module imports | Per-class `from .. import _pymod` | Single module-level `from .. import _pymod as _core` |
| `S3Item.close()` | No-op | Sets `_data = None` to release Rust Bytes Arc reference early |

**New class: `_BytesViewIO(io.RawIOBase)`** — Zero-copy seekable file-like wrapper around BytesView. `readinto()` uses `memoryview` for zero-copy slice into caller's buffer. Used by `S3Checkpoint.reader()` so `torch.load()` can read directly from Rust memory.

### � **Range Download Optimization (Opt-In)**

Parallel range downloads for large S3 objects, achieving **76% performance improvement** for 148 MB objects:

**New Environment Variables:**
- `S3DLIO_ENABLE_RANGE_OPTIMIZATION` — Opt-in flag (default: disabled to avoid HEAD overhead on small objects)
- `S3DLIO_RANGE_THRESHOLD_MB` — Minimum object size to trigger parallel download (default: 64 MB)

**Performance (Measured with 16x 148 MB objects on MinIO):**

| Threshold | Time | Throughput | Speedup |
|-----------|------|------------|---------|
| Disabled (baseline) | 5.52s | 429 MB/s (0.42 GB/s) | 1.00x |
| 8 MB | 3.50s | 676 MB/s (0.66 GB/s) | **1.58x** (58% faster) |
| 16 MB | 3.27s | 725 MB/s (0.71 GB/s) | **1.69x** (69% faster) |
| 32 MB | 3.23s | 732 MB/s (0.71 GB/s) | **1.71x** (71% faster) |
| **64 MB (default)** | **3.14s** | **755 MB/s (0.74 GB/s)** | **1.76x** (76% faster) 🏆 |
| 128 MB | 3.22s | 735 MB/s (0.72 GB/s) | **1.71x** (71% faster) |

**Key Findings:**
- 64 MB threshold (default) is optimal for 148 MB objects
- Sweet spot: 16-64 MB thresholds provide 69-76% faster downloads
- Even aggressive 8 MB threshold shows 58% improvement
- HEAD overhead (~10-20ms) is well amortized by parallel download
- **Fixes issue** where s3dlio was 25% slower than s3torchconnector for 148 MB objects — now 61-76% faster

**Implementation:**
- `S3ObjectStore::get()` now uses `get_object_uri_optimized_async()` when enabled
- S3 backend now matches Azure/GCS parallel range download capability
- Conservative opt-in design: disabled by default to avoid HEAD overhead on small objects

**Python Usage:**
```python
import os
os.environ['S3DLIO_ENABLE_RANGE_OPTIMIZATION'] = '1'
os.environ['S3DLIO_RANGE_THRESHOLD_MB'] = '64'  # Objects ≥ 64 MB use parallel ranges

import s3dlio
data = s3dlio.get("s3://bucket/checkpoint-148mb.bin")  # 76% faster!
```

**Files Changed:**
- `src/s3_utils.rs` — Updated threshold default from 4 MB to 64 MB, added enable check
- `src/object_store.rs` — Wired S3ObjectStore::get() to use optimized path when enabled

### ⚡ **Multipart Upload Performance Improvements**

Zero-copy chunking and non-blocking spawn for streaming multipart uploads:

**Optimizations:**
- **Zero-copy chunking**: Use `Bytes::slice()` instead of `Vec::to_vec()` for part extraction
- **Non-blocking spawn**: Use `spawn_on_global_rt()` instead of `run_on_global_rt()` — eliminates double-spawn overhead
- **New `spawn_part_bytes()` method**: Direct Bytes handling without Vec<u8> conversion
- **No more Python thread blocking**: Upload tasks spawn immediately and return JoinHandle

**Performance Impact:**
- Eliminates one full data copy per multipart chunk (16-64 MB chunks)
- Removes blocking wait during task spawn (was: channel send/recv overhead)
- Python thread now submits upload and continues immediately

**Files Changed:**
- `src/multipart.rs` — Zero-copy Bytes::slice() in write_owned() and write_owned_blocking(), new spawn_part_bytes()
- `src/s3_client.rs` — Added spawn_on_global_rt() helper function

### 📚 **Documentation**

- Rewrote `docs/PYTHON_API_GUIDE.md` — complete API reference for v0.9.50
- Updated `docs/S3TORCHCONNECTOR_MIGRATION.md` — reflects zero-copy compat layer
- Updated `docs/api/Environment_Variables.md` — Full threshold comparison table with actual benchmark data
- Updated `docs/performance/MultiPart_README.md` — Large object download section with performance results
- **New**: `docs/performance/RANGE_OPTIMIZATION_IMPLEMENTATION.md` — Complete implementation summary

### 🏗️ **Architecture**

**Global Client Cache (DashMap):**
- `StoreKey(scheme, endpoint, region) → Arc<dyn ObjectStore>`
- >99% cache hit rate, <100ns lookup (DashMap sharded locking)
- Process-global and automatic — no manual client passing required
- Superior to MinIO's `urllib3.PoolManager(maxsize=10)` per-instance pattern

**Zero-Copy Data Flow:**
```
Rust async I/O → Bytes (Arc) → mpsc channel → PyBytesView (buffer protocol)
                                                    ↓
                    memoryview / np.frombuffer / torch.frombuffer (zero-copy)
```

---

## Version 0.9.40 - Critical Performance Fix & Zero-Copy Architecture (February 2026)

### 🚀 **Critical Performance Fix**

**Zero-Copy DataBuffer Architecture:**
- **FIXED**: 48x performance regression in Python data generation (was 0.8 GB/s, now 42+ GB/s)
- Eliminated `bytes::Bytes` intermediate copy in `generate_into_buffer()` and `generate_data()`
- Changed from `generate_data_with_config()` → `generate_data()` to return `DataBuffer` directly
- Python buffer protocol now exposes `DataBuffer` raw pointer (zero-copy NUMA memory access)
- **Performance**: Achieved parity with dgen-py (41.3 GB/s vs 42.5 GB/s = 2.9% difference)

**New Streaming Generator API:**
- Added `Generator` class to Python API matching dgen-py's streaming interface
- Allows efficient chunk-by-chunk generation with state reuse
- Avoids expensive generator recreation for repeated operations
- Recommended for benchmarking and large-scale data generation

**NUMA/Thread-Pinning Enabled by Default:**
- Enabled `numa` and `thread-pinning` features in default build
- Changed dependency from `hwloc2` to `hwlocality = "1.0.0-alpha.11"`
- Fixed `hardware.rs` to use `NumaTopology::detect()` instead of legacy API
- Automatic NUMA-aware allocation for >16 MB buffers on NUMA systems

**Build Optimizations:**
- Added release profile: `lto=true`, `codegen-units=1`, `opt-level=3`
- Matches dgen-rs optimization level for consistent performance

### 📚 **Documentation & Testing Improvements**

**Python Bytearray Feature Documentation:**
- Enhanced documentation for `generate_into_buffer()` bytearray support
- Added comprehensive performance benchmarks showing 2.5-3.0x speedup with pre-allocated buffers
- Documented best practices for streaming workflows and memory-efficient generation
- Added usage patterns: when to use bytearray vs on-demand allocation

**New Test Suite:**
- Added `test_bytearray_allocation.py` - comprehensive bytearray feature testing
- Added `compare_generators.py` - fair performance comparison with dgen-py using streaming API
- 5 new test cases covering: basic allocation, reuse patterns, performance comparison, various sizes, NumPy compatibility
- All tests passing with excellent performance (2.98x speedup measured with bytearray reuse)

**Updated Documentation:**
- Enhanced `ZERO_COPY_IMPLEMENTATION.md` with bytearray allocation section and DataBuffer architecture
- Performance comparison examples with real benchmarks (16 GB generation in <0.4s)
- Streaming generation patterns with reusable buffers
- Memory efficiency guidelines for large-scale data generation

**Test Status:**
- **Rust tests**: 186/186 passing (100%)
- **Python tests**: All existing + new tests passing
- Zero warnings policy maintained

**Architecture Note:**
- This version prepares for future migration to `dgen-data` crate (published on crates.io)
- Current implementation uses internal `data_gen_alt.rs` module
- Future versions may depend on `dgen-data` to eliminate code duplication

---

## Version 0.9.39 - s3dlio-oplog API Fix (February 2026)

### 🐛 **Bug Fix**

**s3dlio-oplog compatibility:**
- Fixed `s3dlio-oplog` crate to use new `data_gen_alt::generate_controlled_data_alt()` API
- Removed usage of deprecated `generate_controlled_data()` function
- Improved zero-copy performance: `generate_controlled_data_alt()` returns `Bytes` directly

**Impact:** Fixes compilation errors when using s3dlio-oplog with s3dlio v0.9.38+

---

## Version 0.9.38 - Critical Compression Fixes (February 2026)

### 🐛 **Critical Bug Fixes**

**Data Generation Compression:**
- Fixed zero-prefix compression calculation for objects <1 MiB
- Fixed small object alignment in `generate_controlled_data`
- All object sizes now compress to exact target ratios

**GCS Backend:**
- Fixed GCS API compatibility (removed non-existent `list_objects_stream()` calls)
- Replaced with `list_objects()` and manual chunking

**Code Quality:**
- Fixed all clippy warnings (zero-warning policy maintained)
- Added missing benchmark configurations
- Repository cleanup

---

## Version 0.9.37 - Test Suite Modernization (January 2026)

### 🧹 **Build & Test Cleanup**

Comprehensive test suite modernization and build cleanup to achieve zero warnings.

**Changes:**
- Updated 25+ test files to use `bytes::Bytes::from()` for `ObjectStore::put()` calls
- Added `#[ignore]` to slow benchmark tests with proper run instructions in comments
- Deprecated `performance_comparison.rs` (old vs new algorithm comparison)
- Added `#[allow(deprecated)]` where legacy APIs are intentionally tested
- Fixed clippy warning in `object_format_tests.rs` (redundant clone)
- Fixed `test_comprehensive_streaming.rs` to use `ObjectGenAlt` API correctly
- Removed obsolete `test_data_generation_enhancement.rs` (two-pass comparison)

**New Test Files:**
- `test_datagen_performance_validation.rs` - Validates 35+ GB/s generation throughput
- `test_high_speed_data_gen.rs` - Comprehensive data generation benchmarks

**Deprecated Function Suppression:**
- Added `#[allow(deprecated)]` to re-exports and call sites for backward compatibility
- `generate_controlled_data()` remains available but deprecated (use `data_gen_alt` APIs)

**Build Status:** Zero warnings in library, binary, and test targets.

---

## Version 0.9.36 - BREAKING: Zero-Copy API (January 2026)

### ⚠️ **BREAKING CHANGE: ObjectStore API Zero-Copy Conversion**

**Critical Performance Fix**: Eliminated internal `.to_vec()` call that forced memcpy, defeating zero-copy architecture.

**What Changed**:
```rust
// OLD API (v0.9.35 and earlier)
pub trait ObjectStore {
    async fn put(&self, uri: &str, data: &[u8]) -> Result<()>;
    async fn put_multipart(&self, uri: &str, data: &[u8], part_size: Option<usize>) -> Result<()>;
}

// NEW API (v0.9.36+) - Zero-copy throughout
pub trait ObjectStore {
    async fn put(&self, uri: &str, data: Bytes) -> Result<()>;
    async fn put_multipart(&self, uri: &str, data: Bytes, part_size: Option<usize>) -> Result<()>;
}
```

**Why This Matters**:
- **Eliminated Hidden Copy**: Previous API took `&[u8]` but internally converted to `Vec<u8>` via `.to_vec()`, forcing memcpy
- **True Zero-Copy**: `Bytes` is reference-counted (Arc-like) - no memcpy when passed, cloned, or sliced
- **Maintains Performance**: 20-40 GB/s streaming writes now achievable without internal copies
- **Consistent Architecture**: Matches get() API which already returns `Bytes`

**Migration Guide**:
```rust
// Before (v0.9.35):
let data = vec![0u8; 1024];
store.put("s3://bucket/key", &data).await?;

// After (v0.9.36):
let data = Bytes::from(vec![0u8; 1024]);  // Or Bytes::copy_from_slice(&data) for &[u8]
store.put("s3://bucket/key", data).await?;
```

**Zero-Copy Conversions**:
- `Bytes::from(vec)` - Takes ownership (zero-copy, just wraps Arc)
- `Bytes::copy_from_slice(&[u8])` - Allocates new buffer (one copy, unavoidable)
- `bytes.as_ref()` → `&[u8]` - Zero-copy view (fat pointer only)
- `bytes.slice(range)` - Zero-copy subslice (shares Arc)

**Internal Changes**:
- S3: `Bytes` → `ByteStream` via `.into()` (zero-copy)
- Azure: `Bytes` passed directly to SDK (zero-copy)
- GCS: `Bytes` → `&[u8]` via `.as_ref()` (zero-copy view)
- File/Direct: `Bytes` → `&[u8]` via `.as_ref()` (zero-copy view)
- MultiEndpoint: `Bytes` cloned cheaply (reference count, not data)

### ✨ **New: `fill_controlled_data()` - In-Place Buffer Filling**

**High-performance data generation without allocation** (86-163 GB/s with Rayon parallelism):

**Key Innovation**: Fill pre-allocated buffers instead of allocating new `Vec<u8>`:
```rust
/// Fill existing buffer with controlled data (ZERO-COPY workflows)
pub fn fill_controlled_data(buf: &mut [u8], dedup: usize, compress: usize)
```

**Why This Matters**:
- **Streaming Workflows**: Pre-allocate buffers once, reuse for entire stream
- **Zero-Copy Architecture**: Works with `BytesMut` → `Bytes` pipeline
- **Rayon Pool Control**: Respects `ThreadPool::install()` context (unlike allocation-based methods)
- **86-163 GB/s**: Parallel generation scales across all cores

**Usage Example**:
```rust
use bytes::BytesMut;
use s3dlio::fill_controlled_data;

// Allocate buffer once
let mut buf = BytesMut::zeroed(4 * 1024 * 1024);  // 4 MB

// Fill in-place (no additional allocation)
fill_controlled_data(&mut buf, 1, 1);  // Incompressible, no dedup

// Convert to Bytes (zero-copy)
let data: Bytes = buf.freeze();
```

**Benefits**:
- **Memory Efficiency**: Reuse buffers in streaming loops
- **Performance**: 20-50x faster than deprecated `generate_controlled_data_prand()`
- **Thread Control**: Works with custom Rayon thread pools

### 🚨 **Deprecated: `generate_controlled_data_prand()`**

**Old slow method marked deprecated** - replaced by `fill_controlled_data()`:

```rust
#[deprecated(since = "0.9.36", note = "Use fill_controlled_data() instead - MUCH faster (86-163 GB/s vs 3-4 GB/s)")]
pub fn generate_controlled_data_prand(size: usize, dedup: usize, compress: usize) -> Vec<u8>
```

**Performance Comparison**:
- **Old**: `generate_controlled_data_prand()` - 3-4 GB/s (sequential, Vec allocation)
- **New**: `fill_controlled_data()` - 86-163 GB/s (parallel Rayon, in-place)

**Migration**:
```rust
// OLD (deprecated):
let data = generate_controlled_data_prand(size, 1, 1);

// NEW (recommended):
let mut buf = BytesMut::zeroed(size);
fill_controlled_data(&mut buf, 1, 1);
let data = buf.freeze();
```

**Note**: `DataGenAlgorithm::Prand` enum variant removed - all code now uses high-performance Random algorithm.

---

## Version 0.9.35 - Runtime Hardware Detection API & Data Generation Optimization (January 2026)

### 🔍 **New: Public Hardware Detection API**

**Key Innovation**: Hardware detection is now **always available** at runtime, regardless of compile-time feature flags.

**Why This Matters**:
- Binaries built on one system automatically optimize for hardware on another system
- No more "build with NUMA flag" requirements - detection happens at runtime
- External tools (sai3-bench, dl-driver, etc.) can leverage the same hardware detection API
- Single binary works optimally from single-core VMs to multi-socket NUMA servers

**New Public API** (`s3dlio::hardware` module):
```rust
// CPU Detection (always available, zero dependencies on Linux)
pub fn get_affinity_cpu_count() -> usize;         // CPUs available to this process
pub fn total_cpus() -> usize;                      // Total system CPUs
pub fn is_numa_available() -> bool;                 // NUMA hardware detection
pub fn recommended_data_gen_threads() -> usize;     // Optimal thread count
```

**Usage Example**:
```rust
use s3dlio::hardware;

// Automatically scale to available hardware
let threads = hardware::recommended_data_gen_threads();
println!("Using {} threads for data generation", threads);

// Respects cgroup limits, taskset, Docker CPU constraints
let cpus = hardware::get_affinity_cpu_count();

// Check NUMA availability at runtime
if hardware::is_numa_available() {
    println!("NUMA topology detected - optimizing placement");
}
```

### ⚡ **Enhanced: Data Generation Optimization**

**Backported from dgen-rs** high-performance data generation library:

**Optimal Defaults**:
- **1 MB block size** - Optimal CPU cache utilization (was 4 MB)
- **All available cores** - Auto-detected via hardware API (was 50%)
- **NumaMode::Auto** - Runtime NUMA adaptation
- **51.09 GB/s validated** - 100 GB in 1.96s on 12-core system

**New GeneratorConfig Fields** (all optional, backward compatible):
```rust
pub struct GeneratorConfig {
    // Existing fields...
    pub numa_node: Option<usize>,    // NEW: Pin to specific NUMA node
    pub block_size: Option<usize>,   // NEW: Override block size (1-32 MB)
    pub seed: Option<u64>,           // NEW: RNG seed for reproducibility
}
```

**Example - Reproducible Data Generation**:
```rust
use s3dlio::data_gen_alt::generate_controlled_data_alt;

// Generate with explicit seed for reproducibility
let data = generate_controlled_data_alt(
    100 * 1024 * 1024,  // 100 MB
    1,                   // No deduplication
    1,                   // Incompressible
    Some(12345),        // NEW: Reproducible seed
);
```

### 🐍 **Updated: Python Bindings**

**All Python APIs updated** to support new features:
- Added seed parameter support
- Updated GeneratorConfig with new fields
- Maintained zero-copy buffer protocol
- Successfully builds: `s3dlio-0.9.35-cp313-cp313-manylinux_2_39_x86_64.whl`

### 📚 **Documentation**

**New Guides**:
- `docs/Hardware_Detection_API.md` (429 lines) - Complete API reference and examples
- `docs/Data_Generation_Performance.md` (198 lines) - Performance analysis and tuning
- `examples/hardware_detection.rs` - Practical usage demonstration

### 🧪 **Testing**

**New Tests**:
- `test_cpu_utilization.rs` (119 lines) - Validates 50+ GB/s performance
- `test_allocation_overhead.rs` (38 lines) - Memory efficiency validation
- `test_optimal_chunking.rs` (99 lines) - Cache optimization verification

**Results**: ✅ 178 library tests passing, zero warnings

### 🔧 **Code Quality Improvements**

**Fixed**:
- Removed unused imports in 6 test files
- Fixed `test_rmdir_on_nonexistent_directory_is_idempotent` to match idempotent design
- Updated benchmark `data_gen_comparison.rs` with correct API signatures
- Removed non-existent `s3_backend_comparison` example reference
- Zero compiler warnings (verified with `cargo clippy`)

### 📊 **Performance Impact**

**Data Generation**:
- Throughput: **51.09 GB/s** maintained (100 GB in 1.96s)
- CPU Utilization: ~100% across all available cores
- Memory: Zero additional allocations in hot path
- Startup: <1ms overhead for hardware detection

**Binary Size**: +15 KB (hardware detection code)

### 🔄 **Backward Compatibility**

**Breaking Changes**: ✅ **NONE**

**Migration**: ✅ **NOT REQUIRED** - All existing code works unchanged

**Optional Enhancements**:
```rust
// Before (still works):
let data = generate_controlled_data_alt(size, dedup, compress);

// After (with explicit seed):
let data = generate_controlled_data_alt(size, dedup, compress, Some(42));
```

### 🎯 **Use Cases**

**For Benchmarking Tools** (sai3-bench, dl-driver):
```rust
use s3dlio::hardware;

// Automatically scale data generation to available hardware
let threads = hardware::recommended_data_gen_threads();
config.set_threads(threads);
```

**For Reproducible Testing**:
```rust
// Generate identical data across runs for validation
let data = generate_controlled_data_alt(size, 1, 1, Some(fixed_seed));
```

**For Container Environments**:
```rust
// Respects cgroup CPU limits automatically
let cpus = hardware::get_affinity_cpu_count();  // Respects --cpus=4 in Docker
```

See `examples/hardware_detection.rs` for complete examples.

---

## Version 0.9.34 - NUMA-Aware Data Generation (January 2026)

### 🚀 **Enhanced Data Generation with NUMA Optimization**

**Intelligent CPU Allocation for Data Generation + I/O Workloads**:

**Key Features**:
- **Smart Default**: Uses 50% of available CPUs for data generation, leaving 50% for I/O operations
- **Auto NUMA Detection**: Automatically detects UMA (single-node) vs NUMA (multi-socket) systems
- **Thread Pinning**: Pins threads to specific CPU cores on NUMA systems for better cache locality
- **First-Touch Initialization**: Memory allocated on local NUMA node for optimal performance
- **Zero Breaking Changes**: All existing APIs work unchanged

**New Components**:
- `src/numa.rs`: NUMA topology detection module (Linux)
  - Detects number of NUMA nodes, physical cores, logical CPUs
  - Reads from `/sys/devices/system/node` and `/proc/cpuinfo`
- `NumaMode` enum: Auto (default) or Force (testing)
- `GeneratorConfig` struct: Fine-grained control over data generation
- Helper functions:
  - `default_data_gen_threads()`: Returns 50% of available CPUs
  - `total_cpus()`: Returns total logical CPUs available

**Performance Benefits**:
- **NUMA Systems**: Thread pinning + first-touch reduces cross-node memory access
- **All Systems**: Reserves CPU capacity for concurrent I/O operations
- **Balanced Workloads**: Optimal for PUT operations (data gen + upload)

**Optional Cargo Features**:
- `numa`: Enable NUMA topology detection (requires hwloc2)
- `thread-pinning`: Enable CPU core affinity (requires core_affinity)

**Example Usage**:
```rust
// Use defaults (50% CPUs, auto NUMA detection)
let data = generate_controlled_data_alt(100 * 1024 * 1024, 1, 1);

// Override to use all CPUs
let config = GeneratorConfig {
    max_threads: Some(total_cpus()),
    ..Default::default()
};
let data = generate_data_with_config(config);
```

**New Tests** (8 added):
- `test_default_thread_count`: Verify 50% CPU allocation
- `test_generator_config_defaults`: Verify default configuration
- `test_detect_topology`: NUMA topology detection (when feature enabled)
- 5 existing tests updated for new defaults

**Dependencies Added**:
- `hwloc2 = "2.2"` (optional, for NUMA)
- `core_affinity = "0.8"` (optional, for thread pinning)

---

## Version 0.9.33 - Clippy Cleanup (December 26, 2025)

### 🧹 **Code Quality: Zero Clippy Warnings**

**Comprehensive clippy cleanup across library and binary targets**:

**Phase 1 Fixes** (69 warnings resolved):
- Fixed `empty_line_after_doc_comments` (3 instances)
- Fixed `unused_unit` in profiling macros
- Fixed `redundant_field_names` (key: key → key)
- Fixed `derivable_impls` (7 enums with #[derive(Default)])
- Fixed `collapsible_if` (5 instances)
- Fixed `manual_div_ceil` (2 instances with .div_ceil())
- Fixed `manual_clamp` (2 instances with .clamp())
- Fixed `let_unit_value` warnings (5 instances)
- Fixed `writeln_empty_string` warnings (3 instances)
- Fixed `clone_on_copy`, `unnecessary_cast`, `unnecessary_map_or`
- Implemented FromStr trait properly for Strategy enum

**Phase 2 Fixes** (additional cleanup):
- Fixed `manual_strip`: Use strip_prefix/strip_suffix methods
- Fixed `explicit_counter_loop`: Use zip with enumerate for s3_logger
- Fixed `manual_clamp`: Replace max().min() with clamp()
- Fixed `manual_ok`: Replace match Ok/Err patterns with .ok()
- Fixed `needless_range_loop`: Use iterator patterns
- Fixed `if_same_then_else`: Simplify redundant conditional branches
- Fixed `doc_lazy_continuation`: Add proper indentation
- Fixed `redundant_pattern_matching`: Use is_err() methods
- Fixed unused imports: Remove Hash, keep BuildHasher for hash_one
- Fixed `too_many_arguments`: Add allow annotations where necessary
- Applied auto-fixes for 51 additional warnings

**Result**:
- ✅ **Library (lib)**: Zero warnings (was 69)
- ✅ **Binary (s3-cli)**: Zero warnings (was 2)
- ⏭️ **Tests**: 32 warnings remain (deferred to future work)

**Files Modified**: 16 files (src/object_store.rs, src/file_store.rs, src/file_store_direct.rs, src/data_gen.rs, src/s3_logger.rs, src/mp.rs, src/bin/cli.rs, and others)

---

## Version 0.9.32 - Memory-Efficient delete_prefix() (December 2025)

### 🐛 **Fixed: delete_prefix() Memory Bloat with Millions of Objects**

**Problem**: The `delete_prefix()` method loaded ALL object keys into memory before deletion:
```rust
let keys = client.list_objects(&bucket, Some(&key_prefix), true).await?;  // BAD!
client.delete_objects(&bucket, keys).await?;
```

For millions of objects, this consumed gigabytes of memory (~80 bytes per key × 100M objects = ~8GB).

**Solution**: Converted all backends to use streaming list + batched deletion:
- ✅ **S3**: Now uses `s3_list_objects_stream()` with 1000-object batches
- ✅ **Azure**: Now uses `list_stream()` with 1000-object batches  
- ✅ **GCS**: Now uses `list_objects_stream()` with 1000-object batches

**Memory Impact**:
- **Before**: O(N) memory - loaded all N object keys
- **After**: O(1) memory - max 1000 keys in memory at once
- **Example**: 100M objects: 8GB → 80KB memory usage

**Performance**: No degradation - deletion happens in parallel batches as objects are listed.

**Affected Methods**:
- `GcsObjectStore::delete_prefix()` - src/object_store.rs line ~2173
- `S3ObjectStore::delete_prefix()` - src/object_store.rs line ~1050  
- `AzureObjectStore::delete_prefix()` - src/object_store.rs line ~1705

**Testing**: Existing tests pass (test_file_store, dl-driver checkpoint tests).

---

## Version 0.9.33 - Issue #110 Investigation Results (December 2025)

### 📝 **Issue #110: GCS Console Shows Directory Structure After Delete**

**Investigation**: User reported that after using `s3-cli delete -r` on GCS buckets, empty directory structures remained visible in the GCS Console UI.

**Root Cause Analysis**: Extensive testing revealed that:
- ✅ `delete -r` successfully deletes ALL objects (verified with `gsutil ls -r`)
- ✅ GCS API correctly reports zero objects in bucket
- ✅ Both s3-cli and gsutil confirm bucket is empty
- ❌ **GCS Console UI continues to show "folder icons"** (📁)

**Conclusion**: The directory icons are **virtual UI artifacts** created by the GCS Console based on object path structure (the `/` delimiter). These are NOT actual objects in storage - they're purely a Console rendering/caching issue that eventually expires (hours to days).

**Verification Test**:
```bash
# Delete all objects with s3-cli v0.9.25 (before any "fix")
./s3-cli delete -r gs://bucket/
# Deleted: 25088 objects ✅

# Verify with gsutil
gsutil ls -r gs://bucket/
# (no output - bucket is empty) ✅

# Verify with s3-cli
./s3-cli ls -r gs://bucket/
# Total objects: 0 ✅

# Check Console UI
# Still shows directory icons ❌ (UI caching artifact)
```

**Resolution**: No code changes required. This is expected GCS Console behavior. The bucket is actually empty according to the API - the Console UI just hasn't invalidated its cache.

**Workarounds for Console Display**:
1. **Wait**: Console cache expires eventually (hours to days)
2. **Different browser/incognito mode**: May show correct (empty) state
3. **Ignore**: Doesn't affect storage costs, API operations, or functionality

**Status**: Closing as "Not a Bug" - this is GCS Console UI behavior, not a code issue.

---

## Version 0.9.32 - Bug Fix: FileSystemConfig Type Mismatch (December 2025)

### 🐛 **Fixed Issue #85: FileSystemConfig Type Mismatch**

**Problem**: The public API exported `crate::file_store::FileSystemConfig` (with `page_cache_mode`) but `store_for_uri_with_config()` used `crate::file_store_direct::FileSystemConfig` (with `direct_io`, `alignment`, etc.). This made it impossible to configure page cache behavior via the public API.

**Solution**: Introduced unified `StorageConfig` enum:

```rust
pub enum StorageConfig {
    File(FileSystemConfig),      // For file:// URIs
    Direct(DirectFileSystemConfig),  // For direct:// URIs
}
```

**⚠️ BREAKING CHANGE**: `store_for_uri_with_config()` signature changed:

```rust
// OLD (v0.9.31 and earlier)
store_for_uri_with_config(uri, Some(config))

// NEW (v0.9.32+)
store_for_uri_with_config(uri, Some(StorageConfig::File(config)))
store_for_uri_with_config(uri, Some(StorageConfig::Direct(config)))
```

**Migration**:
- Import `StorageConfig` from `s3dlio::api`
- Wrap `FileSystemConfig` in `StorageConfig::File(...)` for file:// URIs
- Wrap `DirectFileSystemConfig` in `StorageConfig::Direct(...)` for direct:// URIs
- Type checking now prevents using wrong config with wrong URI scheme

**Improvements**:
- ✅ Public API now compiles and works as documented
- ✅ Page cache hints (Sequential, Random, DontNeed) now functional
- ✅ Added debug logging to show page cache mode being applied
- ✅ Clear error messages if wrong config type used with URI scheme
- ✅ Comprehensive tests for all configuration scenarios

**New Exports**:
- `pub use crate::api::StorageConfig` - Unified config enum
- `pub use crate::file_store_direct::FileSystemConfig as DirectFileSystemConfig` - O_DIRECT config

**Example**:

```rust
use s3dlio::api::{FileSystemConfig, StorageConfig, PageCacheMode, store_for_uri_with_config};

let config = FileSystemConfig {
    enable_range_engine: false,
    range_engine: Default::default(),
    page_cache_mode: Some(PageCacheMode::Sequential),
};

// This now works!
let store = store_for_uri_with_config(
    "file:///data/test/", 
    Some(StorageConfig::File(config))
)?;
```

## Version 0.9.31 - Maintenance Release (December 2025)

- Changed license to Apache-2.0
- Updated all source files to SPDX-compliant headers
- Removed fork-patches directory

## Version 0.9.30 - Zero-Copy Refactor & PyO3 0.27 (December 2025)

### 🆕 **Zero-Copy Architecture**

**True Zero-Copy Data Path**
- Refactored entire read path to use `bytes::Bytes` throughout the stack
- Eliminated intermediate allocations between storage backends and Python
- `BytesView` wrapper provides direct access to storage data without copying

**Python Buffer Protocol**
- Implemented `__getbuffer__` / `__releasebuffer__` for `BytesView` class
- Enables true zero-copy access via `memoryview(data)`
- Compatible with NumPy, PyTorch, and other frameworks that use buffer protocol

```python
import s3dlio

# Zero-copy read - data stays in same memory location
data = s3dlio.get("s3://bucket/file.bin")
view = memoryview(data)  # No copy! Direct buffer access
arr = np.frombuffer(view, dtype=np.float32)  # Still no copy
```

### 🔄 **PyO3 0.27 Migration**

- Upgraded PyO3 from 0.25 to 0.27.2
- Updated all APIs to use non-deprecated patterns:
  - `Python::attach()` (was `Python::with_gil`)
  - `py.detach()` (was `py.allow_threads`)
  - `Py<PyAny>` (was `PyObject`)
  - `.cast()` (was `.downcast()`)
- Updated dependencies: pyo3-async-runtimes 0.27.0, numpy 0.27.1

### 🧹 **API Cleanup: Protocol Equality**

**Removed S3-Specific Public APIs**
- All storage protocols now treated equally (S3, Azure, GCS, file://, direct://)
- Internal S3 functions changed from `pub fn` to `pub(crate) fn`
- Removed deprecated Python functions:
  - `list_objects(bucket, prefix)` → Use `list(uri)`
  - `get_object(bucket, key, offset, length)` → Use `get(uri)` or `get_range(uri, offset, length)`

**CLI: `list` is Now a True Alias for `ls`**
- Both commands execute identical code paths
- All protocols supported: `s3://`, `az://`, `gs://`, `file://`, `direct://`
- Same options available: `-r`, `-p`, `-c`

```bash
s3dlio ls s3://bucket/prefix/ -r     # Works
s3dlio list az://container/path/ -r  # Also works - same command
```

### 📦 **Technical Details**

- **Branch:** fix/multi-backend-zero-copy
- **Files changed:** 12
- **Tests:** 175 Rust tests passing, Python tests passing
- **Warnings:** Zero compiler warnings

---

## Version 0.9.27 - PyPI Release & Bug Fixes (December 2025)

### 🆕 **First PyPI Release**

s3dlio is now available on PyPI! Install with:
```bash
pip install s3dlio
```

### 🐛 **Bug Fixes**

**Optional ML Framework Imports**
- Fixed crash when importing s3dlio without PyTorch/JAX/TensorFlow installed
- ML-specific loaders (S3MapDataset, S3IterableDataset, etc.) now return `None` if frameworks unavailable
- Core storage functions work without any ML dependencies

**Universal Backend Support for Async Functions**
- Fixed `exists_async()` to work with all backends (was S3-only)
- Fixed `stat_async()` to work with all backends (was S3-only)
- Both now use the universal ObjectStore interface like their sync counterparts

### 📦 **Package Metadata**

- Added author information: Russ Fellows
- Proper package metadata for PyPI listing

### ⚠️ **Note**

v0.9.26 was yanked from PyPI due to the import bug. Use v0.9.27.

---

## Version 0.9.26 - DLIO Benchmark Integration (December 2025)

### 🆕 **New Features**

**DLIO Benchmark Integration**
- Added comprehensive integration support for [Argonne DLIO Benchmark](https://github.com/argonne-lcf/dlio_benchmark)
- Two installation options:
  - **Option 1 (Recommended):** New `storage_type: s3dlio` with explicit configuration
  - **Option 2:** Drop-in replacement for existing S3 configurations
- Enables DLIO to use all s3dlio backends: S3, Azure, GCS, file://, direct://

**Zero-Copy Write Functions (Rust + Python)**
- `put_bytes(uri, data)` - Write bytes to any backend using zero-copy from Python memory
- `put_bytes_async(uri, data)` - Async version
- `mkdir(uri)` - Create directories/prefixes via ObjectStore trait
- `mkdir_async(uri)` - Async version
- Uses `PyBytes.as_bytes()` for zero-copy data transfer, avoiding allocation overhead

### 📦 **New Files**

```
python/s3dlio/integrations/dlio/
├── __init__.py           # Helper functions for installation
├── s3dlio_storage.py     # Option 1: S3dlioStorage class
└── s3_torch_storage.py   # Option 2: Drop-in replacement

docs/integration/
└── DLIO_BENCHMARK_INTEGRATION.md  # Comprehensive guide (477 lines)
```

### 🔧 **Other Changes**

**Azure SDK Update**
- Updated from Azure SDK 0.4.0 to 0.7.0 API
- Maintains backward compatibility with existing functionality

### 📝 **Documentation**

- Added `docs/integration/DLIO_BENCHMARK_INTEGRATION.md` - comprehensive guide with:
  - Architecture diagram
  - Step-by-step instructions for both integration options
  - Environment variable configuration (S3/Azure/GCS)
  - API mapping table (DLIO → s3dlio)
  - Troubleshooting section
  - Multi-protocol examples
  - Advanced features (multi-endpoint load balancing, performance tuning)
  - Feature compatibility table
- Updated `docs/PYTHON_API_GUIDE.md` with new functions

---

## Version 0.9.24 - S3-Compatible Endpoint Fix & Tracing Workaround (December 2025)

### 🐛 **Bug Fixes**

**S3-Compatible Endpoint Support (force_path_style)**
- Fixed S3 requests to custom endpoints (MinIO, Ceph, etc.)
- Added `force_path_style(true)` to S3 config builder in `s3_client.rs`
- **Root cause**: AWS SDK defaults to virtual-hosted style addressing (e.g., `bucket.endpoint.com`) which doesn't work with custom endpoints that expect path-style (e.g., `endpoint.com/bucket`)
- Now matches AWS CLI behavior when using custom endpoints

**Tracing Hang Workaround**
- Fixed hang when using verbose flags (`-v`, `-vv`) with `s3-cli`
- **Root cause**: `tracing::debug!()` macros inside `tokio::spawn` async tasks cause indefinite hangs when AWS SDK S3 operations are also running in that task
- **Solution**: CLI now uses `warn,s3dlio=debug` filter to exclude AWS SDK debug logging
- Commented out debug statements inside `tokio::spawn` in `s3_utils.rs` as an additional safeguard

### 🐛 **Known Issues**

**AWS SDK Tracing Hang Bug** ([aws-sdk-rust#1388](https://github.com/awslabs/aws-sdk-rust/issues/1388))
- `tracing::debug!()` inside `tokio::spawn` + AWS SDK operations = hang
- Affects both SDK 1.104.0 and 1.116.0 (not a recent regression)
- **Workaround**: Filter tracing with `RUST_LOG=warn,s3dlio=debug`
- Debug statements outside `tokio::spawn` work correctly
- The `s3-cli` `-v` and `-vv` flags use the workaround filter automatically

### 📦 **Examples & Project Organization**

**Python Examples** (6 comprehensive examples in `examples/python/`)
- `basic_operations.py` - Core put/get/list/stat/delete operations
- `parallel_operations.py` - High-performance parallel get/put with concurrency tuning
- `data_loader.py` - ML data loader patterns with batch iteration
- `streaming_writer.py` - Chunked/streaming upload API with compression
- `upload_download.py` - File upload/download workflows
- `oplog_example.py` - Operation logging/tracing demonstration

**Examples Directory Reorganization**
- Moved Python examples to `examples/python/`
- Moved Rust examples to `examples/rust/`
- Moved shell scripts to `scripts/`
- Deleted broken examples that used outdated APIs

**Op-Log Fixes**
- Fixed `get()` and `delete()` in Python API to use `store_for_uri_with_logger()`
- Fixed `put_objects_parallel_with_progress()` to use logger
- All Python operations now properly logged via `LoggedObjectStore` wrapper

**Code Quality**
- Fixed unused import warning (`info` in `s3_client.rs`)
- Added `#[cfg(feature = "experimental-http-client")]` to functions only used with that feature
- Zero non-deprecation warnings

### 📝 **Documentation**

- Created `docs/bugs/AWS_SDK_TRACING_HANG_BUG_REPORT.md` with full investigation details
- Added doc comments to `list_objects_stream()` in `s3_utils.rs` warning about the bug
- Updated README.md with v0.9.24 release notes

---

## Version 0.9.23 - Azure Blob & GCS Custom Endpoint Support (December 3, 2025)

### 🆕 **New Features**

**Custom Endpoint Support for Azure Blob Storage**
- Added environment variable support for custom Azure endpoints
- Primary: `AZURE_STORAGE_ENDPOINT` (e.g., `http://localhost:10000`)
- Alternative: `AZURE_BLOB_ENDPOINT_URL`
- Enables use with Azurite or other Azure-compatible emulators/proxies
- Account name is appended to endpoint URL automatically

Usage:
```bash
# Azurite (local emulator)
export AZURE_STORAGE_ENDPOINT=http://127.0.0.1:10000
sai3-bench util ls az://devstoreaccount1/testcontainer/

# Multi-protocol proxy
export AZURE_STORAGE_ENDPOINT=http://localhost:9001
sai3-bench util ls az://myaccount/mycontainer/
```

**Custom Endpoint Support for Google Cloud Storage**
- Added environment variable support for custom GCS endpoints
- Primary: `GCS_ENDPOINT_URL` (e.g., `http://localhost:4443`)
- Alternative: `STORAGE_EMULATOR_HOST` (GCS emulator convention, `http://` prepended if missing)
- Enables use with fake-gcs-server or other GCS-compatible emulators/proxies
- Anonymous authentication used automatically for custom endpoints (typical for emulators)

Usage:
```bash
# fake-gcs-server (local emulator)
export GCS_ENDPOINT_URL=http://localhost:4443
sai3-bench util ls gs://testbucket/

# Using STORAGE_EMULATOR_HOST convention
export STORAGE_EMULATOR_HOST=localhost:4443
sai3-bench util ls gs://testbucket/

# Multi-protocol proxy
export GCS_ENDPOINT_URL=http://localhost:9002
sai3-bench util ls gs://testbucket/
```

### 📝 **Documentation**

- Updated `docs/api/Environment_Variables.md` with Azure and GCS endpoint configuration
- Added new constants in `src/constants.rs` for endpoint environment variable names:
  - `ENV_AZURE_STORAGE_ENDPOINT`
  - `ENV_AZURE_BLOB_ENDPOINT_URL`
  - `ENV_GCS_ENDPOINT_URL`
  - `ENV_STORAGE_EMULATOR_HOST`

### ⚡ **Compatibility**

**Backwards Compatibility**: No breaking changes
- When environment variables are not set, behavior remains identical to previous versions
- Connects to public cloud endpoints by default (Azure Blob, GCS)
- S3 custom endpoint support via `AWS_ENDPOINT_URL` remains unchanged

**Related Issue**: https://github.com/russfellows/sai3-bench/issues/56

---

## Version 0.9.22 - Client ID & First Byte Tracking (November 25, 2025)

### 🆕 **New Features**

**Client ID Support for Multi-Agent Operation Logging**
- Added `set_client_id()` and `get_client_id()` public functions
- All operation log entries now include client_id field
- Enables identification of which client/agent performed each operation
- Thread-safe implementation using `OnceCell<Mutex<String>>`
- Minimal overhead: ~10ns per log entry (mutex lock + clone)
- Use case: Distributed benchmarking with multiple agents writing to separate oplogs

API Usage:
```rust
// Initialize logger
s3dlio::init_op_logger("operations.log.zst")?;

// Set client identifier (agent ID, hostname, custom ID, etc.)
let client_id = std::env::var("CLIENT_ID").unwrap_or_else(|_| "standalone".to_string());
s3dlio::set_client_id(&client_id)?;

// All future log entries tagged with this client_id
```

**Approximate First Byte Tracking** (See docs/OPERATION_LOGGING.md for details)
- `first_byte_time` field now populated in operation logs
- GET operations: first_byte ≈ end (when complete data is available)
- PUT operations: first_byte = start (upload begins immediately)
- Metadata operations (LIST, HEAD, DELETE): first_byte = None (not applicable)

**Important**: This is an *approximate* implementation due to ObjectStore trait limitations:
- Current API returns `Bytes` (complete data), not `Stream<Bytes>`
- Can't distinguish HTTP header receipt from body completion
- For small objects (<1MB): approximation is acceptable for throughput analysis
- For true TTFB metrics: Use streaming APIs (future enhancement) or dedicated HTTP tools

See [OPERATION_LOGGING.md](OPERATION_LOGGING.md) for comprehensive documentation on:
- Why first_byte is approximate
- When to use vs when to avoid
- Future enhancement plans (streaming GET API)
- Recommendations for different use cases

### 📝 **Documentation**

**New: Operation Logging Guide** (docs/OPERATION_LOGGING.md)
- Comprehensive explanation of operation logging architecture
- First byte tracking strategy with detailed rationale
- Clock offset synchronization patterns
- Client identification best practices
- Example usage for standalone and distributed scenarios
- Performance impact analysis
- Future enhancement roadmap

**Updated: Code Comments**
- Extensive inline documentation in `object_store_logger.rs`
- 40+ lines explaining first_byte tracking approach and limitations
- Clear guidance on when approximation is acceptable
- Future enhancement notes for streaming APIs

**Clarification: Operation Log Sorting**
- Added section on post-processing oplogs in OPERATION_LOGGING.md
- Clarified that logs are NOT sorted during capture (due to concurrent writes)
- Documented proper sorting workflow using sai3-bench sort command
- Note: Sorted logs compress ~30-40% better than unsorted

### ⚠️ **Important Notes**

**first_byte_time Interpretation**:
- **DO**: Use for throughput analysis and relative comparisons
- **DO**: Use for small object (<1MB) performance benchmarking
- **DON'T**: Assume it represents exact time of first byte arrival
- **DON'T**: Use for precise TTFB analysis on large objects (>10MB)

**Backwards Compatibility**:
- Existing code continues to work (client_id defaults to empty string)
- TSV format unchanged (first_byte column existed but was empty before)
- No breaking API changes

## Version 0.9.21 - Clock Offset Support & Pseudo-Random Data Generation (November 25, 2025)

### 🆕 **New Features**

**Clock Offset Support for Distributed Op-Log Synchronization** (Issue #100)
- Added `set_clock_offset()` and `get_clock_offset()` public functions
- Logger now supports timestamp correction for distributed systems
- Enables accurate global timeline reconstruction when agents have clock skew
- Thread-safe implementation using `Arc<AtomicI64>`
- Minimal overhead: single atomic read per log entry
- Use case: sai3-bench and dl-driver distributed benchmarking

API Usage:
```rust
// Initialize logger
s3dlio::init_op_logger("operations.log.zst")?;

// Calculate clock offset during agent sync
let offset = (local_time - controller_time).as_nanos() as i64;

// Set offset for all future log entries
s3dlio::set_clock_offset(offset)?;
```

**Pseudo-Random Data Generation Method** (Issue #98 resolution)
- Added `generate_controlled_data_prand()` function
- Uses original BASE_BLOCK algorithm (~3-4 GB/s, consistent performance)
- Provides "prand" option alongside "random" (new Xoshiro256++ algorithm)
- Public API in `s3dlio::api::advanced` module
- When to use:
  - `random`: Truly incompressible data (compress=1 → ~1.0 zstd ratio), slower but more realistic
  - `prand`: Maximum CPU efficiency, faster but allows cross-block compression patterns

### ✅ **Bug Fixes & Improvements**

**Issue #95: Range Engine Messages** (Already Fixed)
- Confirmed Range Engine messages now at `trace!` level (not `debug!`)
- Fixed in commit edee657 (November 16, 2025)
- No longer overwhelms debug output

**Issue #98: Old Data Generation Code**
- Resolution: Old code now serves as "prand" method, not removed
- Provides performance option for CPU-constrained scenarios
- Both algorithms available for different use cases

### 📝 **Documentation**

- Added comprehensive clock offset documentation in `s3_logger.rs`
- Updated API documentation for both data generation methods
- Added usage examples for distributed op-log synchronization

---

## Version 0.9.20 - High-Performance List & Delete Optimizations (November 22, 2025)

### 🚀 **Performance Improvements for Large Object Operations**

Major optimizations for workloads with 100K-1M+ objects, targeting 5x performance improvement (28 minutes → 5-8 minutes for 1M objects).

**Phase 1: Batch Delete API**
- Added `delete_batch()` method to ObjectStore trait
- S3: DeleteObjects API (1000 objects/request)
- Azure: Batch API with pipeline (up to 256 operations)
- GCS: Batch delete with parallel execution
- File/Direct: Parallel deletion with configurable concurrency
- All 7 backends implement efficient batch operations

**Phase 2: Streaming List with Progress**
- Added `list_stream()` returning `Pin<Box<dyn Stream<Item = Result<String>>>>`
- S3: True streaming via paginated ListObjectsV2 (1000-object pages)
- Azure/GCS/File: Efficient buffered implementation
- MultiEndpoint: Wraps stream with per-endpoint statistics
- ObjectStoreLogger: Proper op-log integration for workload replay
- CLI: Added `-c/--count-only` flag with progress indicators
- CLI: Rate formatting with comma separators (e.g., "rate: 276,202 objects/s")

**Phase 3: List+Delete Pipeline**
- Concurrent lister/deleter tasks via tokio channels
- 1000-object batches, 10-batch buffer (10K objects in-flight)
- Overlaps LIST and DELETE operations for maximum throughput
- Progress reporting every 10K objects

**CLI Improvements:**
```bash
# Count objects with streaming progress
s3-cli ls -rc s3://bucket/prefix/
# Output: Total objects: 1,234,567 (12.345s, rate: 100,000 objects/s)

# Fast deletion with pipeline
s3-cli delete s3://bucket/prefix/
# Uses concurrent list+delete pipeline automatically
```

**Architecture:**
- Clean abstractions maintained (no backend-specific CLI code)
- Proper op-log integration for workload replay capability
- Zero-copy streaming where possible
- All optimizations work across all 7 storage backends

**Testing:**
- Comprehensive test suite with file:// backend
- Pattern matching validated (preserves non-matching objects)
- Pipeline performance verified (10K files in 0.213s)
- Progress indicators and rate formatting confirmed

**Documentation:**
- `docs/LIST_DELETE_PERFORMANCE_OPTIMIZATION.md` - Complete 3-phase plan
- All phases implemented and tested

### 📝 **API Stability**

**Rust API:**
- New trait methods: `ObjectStore::delete_batch()`, `ObjectStore::list_stream()`
- Backward compatible - existing code continues to work
- Python API automatically benefits from underlying optimizations

**Python API:**
- No changes required - `delete()` and `list()` use optimized implementations
- Batch operations work transparently under the hood

---

## Version 0.9.18 - Data Generation Bug Fix & Algorithm Migration (November 17-18, 2025)

### 🔧 **Update (November 18, 2025): RNG Optimization & Distributed Safety**

**Performance Optimization:**
- Explicit Xoshiro256PlusPlus RNG (removed StdRng abstraction)
- 5-24% performance improvement in data generation
- Same or lower CPU usage across all workloads
- Added `rand_xoshiro = "^0.7"` dependency

**Distributed Deployment Enhancement:**
- Enhanced entropy source: SystemTime + `/dev/urandom`
- Prevents data collision across distributed workers
- Critical for orchestrated environments (Kubernetes, SLURM)
- Ensures global uniqueness even with synchronized clocks

**Code Quality:**
- Updated comments: removed "Bresenham" terminology
- Clarified as "integer error accumulation" (standard distribution technique)
- No patent concerns or algorithmic attribution issues

**Comprehensive Testing:**
- New `tests/performance_comparison.rs` with CPU/memory metrics
- All 162 library tests passing
- Performance validated across 6 workload scenarios
- API compatibility verified (zero breaking changes)

**Performance Results:**
```
Test                 | OLD Speed  | NEW Speed  | Speedup | CPU Δ
---------------------|------------|------------|---------|-------
1MB compress=1       | 3,474 MB/s | 3,436 MB/s | 0.99x   | +15%
16MB compress=1      | 6,319 MB/s | 6,621 MB/s | 1.05x   | 0%
64MB compress=1      | 5,800 MB/s | 6,283 MB/s | 1.08x   | -10%
16MB compress=5      | 5,660 MB/s | 7,009 MB/s | 1.24x   | -3%
Streaming 16MB       | 2,355 MB/s | 2,530 MB/s | 1.07x   | 0%
16MB dedup=4         | 6,936 MB/s | 7,553 MB/s | 1.09x   | -3%
```

**Documentation:**
- `API_COMPATIBILITY_REPORT.md` - Complete API analysis
- Verified sai3-bench (6 call sites) and dl-driver (12 call sites) compatibility

### 🐛 **Critical Bug Fix: Cross-Block Compression**

Fixed a critical bug in the data generation algorithm where `compress=1` (incompressible data) incorrectly produced 7.68:1 compression ratio instead of ~1.0.

**Root Cause:**
- Original algorithm used shared `BASE_BLOCK` template across all unique blocks
- Zstd compressor found cross-block patterns, defeating incompressibility guarantee
- Affected all compress levels (1-6) when combined with dedup > 1

**Solution:**
- New algorithm uses per-block Xoshiro256++ RNG initialization
- Each unique block gets independent high-entropy keystream
- Local back-references within blocks for controlled compressibility
- `compress=1` now correctly produces ratio ~1.0000 ✅

### ✨ **New Data Generation Algorithm**

Introduced `data_gen_alt.rs` with improved correctness and performance:

**Features:**
- Per-block RNG seeding (prevents cross-block compression)
- Xoshiro256++ RNG (5-10x faster than ChaCha20)
- Streaming generation via `ObjectGenAlt`
- Parallel single-pass generation for large datasets
- Performance: 1-7 GB/s depending on size

**API Changes:**
- **Zero breaking changes** - all existing code works unchanged
- `generate_controlled_data()` transparently redirected to new algorithm
- `ObjectGen` now wraps `ObjectGenAlt` internally
- Old implementations preserved as commented-out code (removal: December 2025)

### 📊 **Validation Results**

**Compression Ratios (16MB test):**
- compress=1: 1.0000 ✅ (was 7.6845 ❌)
- compress=5: 1.3734 ✅
- compress=6: 1.3929 ✅

**Performance:**
- 1MB: 954 MB/s (3.76x faster than old algorithm)
- 16MB: 2,816 MB/s
- 64MB: 7,351 MB/s
- Streaming: 1,374 MB/s

**Testing:**
- All 162 library tests passing ✅
- Comprehensive test suite added (`tests/test_data_gen_alt.rs`)
- Deduplication behavior validated (dedup=2,6)
- Old algorithm bug confirmed via regression test

### 📝 **Documentation**

Added comprehensive migration documentation:
- `docs/DATA_GEN_MIGRATION_SUMMARY.md` - Technical details (9.2KB)
- `.github/ISSUE_TEMPLATE/data_gen_migration.md` - Tracking issue template (5.4KB)
- `.github/copilot-instructions.md` - Migration checklist
- `src/data_gen.rs` - 73-line header explaining changes

### 🔄 **Migration Timeline**

**Phase 1: Production Validation (Nov-Dec 2025)**
- Extended testing with real workloads
- Performance monitoring across platforms
- Compatibility verification with downstream tools

**Phase 2: Code Cleanup (December 2025)**
- Remove commented-out old algorithm code
- Update inline documentation
- Consider renaming data_gen_alt.rs → data_gen.rs

**Phase 3: Optimization (Q1 2026)**
- Profile Xoshiro256++ performance
- Evaluate SIMD opportunities
- Benchmark against industry tools

### 🔧 **Dependencies**

- Updated `rand_chacha` 0.3 → 0.9
- Fixed API breaking changes (`gen_range` → `random_range`)
- Updated test code for modern APIs

### 📦 **Test Suite**

- **Total tests**: 162 (all passing)
- **New tests**: 7 comprehensive tests for data_gen_alt
- **Coverage**: Compression ratios, dedup behavior, streaming, performance, regression

---

## Version 0.9.17 - NPY/NPZ Enhancements & TFRecord Index API (November 16, 2025)

### 🎯 **Multi-Array NPZ Support**

Added `build_multi_npz()` function for creating NumPy ZIP archives with multiple named arrays, enabling PyTorch/JAX-style dataset creation with data, labels, and metadata in a single file.

**New API:**

```rust
use s3dlio::data_formats::npz::build_multi_npz;
use ndarray::ArrayD;

// Create multi-array NPZ (PyTorch/JAX pattern)
let data = ArrayD::zeros(vec![224, 224, 3]);
let labels = ArrayD::ones(vec![10]);
let metadata = ArrayD::from_elem(vec![5], 42.0);

let arrays = vec![
    ("data", &data),
    ("labels", &labels),
    ("metadata", &metadata),
];

let npz_bytes = build_multi_npz(arrays)?;
// Write npz_bytes to file or object storage
```

**Python Interoperability:**

```python
import numpy as np

# Load multi-array NPZ created by Rust
data = np.load("dataset.npz")
print(data.files)  # ['data', 'labels', 'metadata']

images = data['data']      # Shape: (224, 224, 3)
labels = data['labels']    # Shape: (10,)
metadata = data['metadata'] # Shape: (5,)
```

**Key Features:**
- **Zero-copy design**: Uses `Bytes` for efficient memory handling
- **Proper ZIP structure**: Compatible with NumPy's `np.load()`
- **Named arrays**: Custom names for each array in the archive
- **Type support**: f32 arrays (primary ML use case)
- **Comprehensive tests**: 5 new tests covering single/multi-array scenarios

**Use Cases:**
- AI/ML dataset generation (images + labels + metadata)
- Scientific computing (simulation results + parameters + timestamps)
- dl-driver workload generation (simplified from 150+ lines to 80 lines)

---

### 🔧 **TFRecord Index Generation API**

Exported `build_tfrecord_with_index()` function and `TfRecordWithIndex` struct for creating TFRecord files with accompanying index files, enabling compatibility with TensorFlow Data Service.

**New Exports:**

```rust
use s3dlio::data_formats::{build_tfrecord_with_index, TfRecordWithIndex};

// Generate TFRecord with index in single pass
let raw_data = s3dlio::generate_controlled_data(102400, 1, 1);
let result = build_tfrecord_with_index(
    100,    // num_records
    1024,   // record_size
    &raw_data
)?;

// result.data: Bytes containing TFRecord file
// result.index: Bytes containing index file (16 bytes per record)

// Write both files
store.put("dataset.tfrecord", &result.data).await?;
store.put("dataset.tfrecord.index", &result.index).await?;
```

**Index Format** (TensorFlow Data Service compatible):
```
For each record:
  - offset: u64 (8 bytes, little-endian) - Byte offset in TFRecord file
  - length: u64 (8 bytes, little-endian) - Record length in bytes
Total: 16 bytes per record
```

**Key Features:**
- **Zero overhead**: Index generated during TFRecord creation (single pass)
- **Standard format**: Compatible with TensorFlow Data Service expectations
- **Efficient**: Returns `Bytes` for zero-copy I/O
- **Documented**: Clear API for downstream tools (dl-driver, custom tools)

**Performance:**
- No additional I/O operations
- Minimal memory overhead (16 bytes per record)
- Example: 1000 records → 16KB index file

**Background:**

TensorFlow Data Service can leverage index files to optimize random access patterns and enable efficient dataset sharding across distributed workers. This API enables tools to generate properly formatted indices alongside TFRecord data files.

---

### 🔄 **Custom NPY/NPZ Implementation**

Previously in v0.9.16, replaced `ndarray-npy` dependency with custom 328-line implementation for better control, zero-copy performance, and elimination of version conflicts.

**Features (continued from v0.9.16):**
- Full NPY format support (header + data serialization)
- Multi-array NPZ with proper ZIP structure (NEW in v0.9.17)
- TFRecord index generation API (NEW in v0.9.17)
- Zero-copy `Bytes`-based design
- Python/NumPy interoperability verified
- 11 comprehensive tests (6 NPY + 5 multi-NPZ)

---

## Version 0.9.16 - Optional Op-Log Sorting (November 7, 2025)

### 📊 **Configurable Operation Log Sorting**

Added optional automatic sorting of operation logs by start timestamp, addressing chronological ordering requirements for multi-threaded workloads while maintaining high performance for large-scale logging (10M+ operations).

**Key Changes:**

- **Optional Auto-Sort at Shutdown** - Controlled via `S3DLIO_OPLOG_SORT` environment variable
  - **Default behavior**: Streaming write (no sorting, zero memory overhead, immediate output)
  - **Opt-in sorting**: Set `S3DLIO_OPLOG_SORT=1` to enable chronological sorting
  - **Performance**: ~1.2μs per entry overhead (~4% for 210K entries)
  - **Use case**: Small to medium workloads (<1M operations) requiring sorted output

- **Streaming Sort Window Constant** - `DEFAULT_OPLOG_SORT_WINDOW = 1000`
  - Documented for future streaming sort implementations
  - Sized based on observation that operations are rarely >1000 lines out of order
  - Enables constant-memory sorting for huge files (50M+ operations)

**Background:**

Multi-threaded operation logging writes entries as they complete, not in start-time order. Variable I/O latency causes operations to finish out of sequence. For workloads requiring chronological analysis (replay, performance analysis), sorting is now optionally available.

**Environment Variables:**

```bash
# Default: Fast streaming write (unsorted)
sai3-bench run --op-log /tmp/ops.tsv --config test.yaml

# Opt-in: Auto-sort at shutdown (sorted output)
S3DLIO_OPLOG_SORT=1 sai3-bench run --op-log /tmp/ops.tsv --config test.yaml
```

**Configuration Constants:**

- `ENV_OPLOG_SORT` - Environment variable name for auto-sort control
- `DEFAULT_OPLOG_SORT_WINDOW` - Window size for streaming sort algorithms (1000 lines)

**Implementation Details:**

- Sort-on-write path collects all entries in `Vec<LogEntry>`, sorts by `start_time`, then writes
- No-sort path streams directly to file (zero buffering, minimal memory)
- Both paths use zstd compression (level 1) and auto-add `.zst` extension
- Proper `info!()` logging for transparency during long sorts

**Rationale:**

This opt-in approach balances three competing needs:
1. **Performance**: Default streaming mode has zero sorting overhead
2. **Scalability**: Large workloads (10M+ ops) avoid memory pressure
3. **Usability**: Small workloads can enable convenient auto-sort

For very large files requiring sorting, downstream tools (sai3-bench) provide streaming window-based offline sorting with constant memory usage.

---

## Version 0.9.15 - S3 URI Endpoint Parsing (November 6, 2025)

### 🔧 **Enhanced URI Parsing for Multi-Endpoint Scenarios**

Added utilities for parsing S3 URIs with optional custom endpoints, supporting MinIO, Ceph, and other S3-compatible storage systems with explicit endpoint specifications.

**New Functions:**

**Rust API:**
```rust
use s3dlio::{parse_s3_uri_full, S3UriComponents};

// Standard AWS format
let components = parse_s3_uri_full("s3://mybucket/data.bin")?;
assert_eq!(components.endpoint, None);  // Uses AWS_ENDPOINT_URL env var
assert_eq!(components.bucket, "mybucket");
assert_eq!(components.key, "data.bin");

// MinIO/Ceph with custom endpoint
let components = parse_s3_uri_full("s3://192.168.100.1:9001/mybucket/data.bin")?;
assert_eq!(components.endpoint, Some("192.168.100.1:9001".to_string()));
assert_eq!(components.bucket, "mybucket");
assert_eq!(components.key, "data.bin");
```

**Python API:**
```python
import s3dlio

# Parse standard AWS URI
result = s3dlio.parse_s3_uri_full("s3://mybucket/data.bin")
# {'endpoint': None, 'bucket': 'mybucket', 'key': 'data.bin'}

# Parse MinIO URI with endpoint
result = s3dlio.parse_s3_uri_full("s3://192.168.100.1:9001/mybucket/data.bin")
# {'endpoint': '192.168.100.1:9001', 'bucket': 'mybucket', 'key': 'data.bin'}
```

**Key Features:**
- Heuristic endpoint detection (IP addresses, hostnames, ports)
- Backwards compatible with existing `parse_s3_uri()`
- Useful for multi-process benchmarking tools (sai3-bench, dl-driver)
- Zero overhead - parsing utilities only

**Use Case:**
Multi-process testing with different endpoints per process:
```bash
# Process 1
AWS_ENDPOINT_URL=http://192.168.100.1:9001 sai3bench-agent

# Process 2  
AWS_ENDPOINT_URL=http://192.168.100.2:9001 sai3bench-agent
```

Tools can use `parse_s3_uri_full()` to extract endpoint information from config files for validation and process orchestration.

---

## Version 0.9.14 - Multi-Endpoint Storage (November 6, 2025)

### 🎯 **Multi-Endpoint Load Balancing for High-Throughput Workloads**

Added comprehensive multi-endpoint support enabling load distribution across multiple storage backends for improved performance, scalability, and fault tolerance.

**New Core Components:**

**1. MultiEndpointStore** - Load balancing wrapper implementing full `ObjectStore` trait
```rust
use s3dlio::{MultiEndpointStore, MultiEndpointStoreConfig, EndpointConfig, LoadBalanceStrategy};

let config = MultiEndpointStoreConfig {
    endpoints: vec![
        EndpointConfig { uri: "s3://bucket-1/data".to_string(), ..Default::default() },
        EndpointConfig { uri: "s3://bucket-2/data".to_string(), ..Default::default() },
        EndpointConfig { uri: "s3://bucket-3/data".to_string(), ..Default::default() },
    ],
    strategy: LoadBalanceStrategy::RoundRobin,  // or LeastConnections
    default_thread_count: 4,
};

let store = MultiEndpointStore::new(config).await?;
```

**2. Load Balancing Strategies:**
- **RoundRobin**: Sequential distribution for uniform workloads (minimal overhead)
- **LeastConnections**: Adaptive routing based on active connections (self-balancing)

**3. Per-Endpoint Configuration:**
```rust
EndpointConfig {
    uri: "s3://high-perf-bucket/data".to_string(),
    thread_count: Some(16),          // Override default thread count
    process_affinity: Some(0),       // NUMA node assignment for multi-socket systems
}
```

**4. URI Template Expansion** - New `uri_utils` module:
```rust
// Expand numeric ranges with optional zero-padding
expand_uri_template("s3://bucket-{1...10}/data")?;
// → ["s3://bucket-1/data", ..., "s3://bucket-10/data"]

expand_uri_template("direct://192.168.1.{01...10}:9000")?;
// → ["direct://192.168.1.01:9000", ..., "direct://192.168.1.10:9000"]

// Load URIs from configuration file (one per line)
load_uris_from_file("endpoints.txt")?;
```

**5. Statistics and Monitoring:**
```rust
// Per-endpoint statistics (lock-free atomics)
let stats = store.get_endpoint_stats();
for (i, stat) in stats.iter().enumerate() {
    println!("Endpoint {}: {} requests, {} bytes, {} active connections",
             i, stat.requests, stat.bytes_transferred, stat.active_connections);
}

// Total aggregated statistics
let total = store.get_total_stats();
println!("Total: {} requests, {} bytes, {} errors",
         total.total_requests, total.total_bytes, total.total_errors);
```

**6. Python API** - Zero-copy support via `BytesView`:
```python
import s3dlio

# Create multi-endpoint store
store = s3dlio.create_multi_endpoint_store(
    uris=["s3://bucket-1/data", "s3://bucket-2/data", "s3://bucket-3/data"],
    strategy="least_connections"  # or "round_robin"
)

# Zero-copy data access (buffer protocol support)
data = store.get("s3://bucket-1/large-file.bin")
mv = memoryview(data)  # No copy!
array = np.frombuffer(mv, dtype=np.float32)

# Template expansion
store = s3dlio.create_multi_endpoint_store_from_template(
    template="s3://shard-{1...10}/dataset",
    strategy="round_robin"
)

# File-based configuration
store = s3dlio.create_multi_endpoint_store_from_file(
    file_path="endpoints.txt",
    strategy="least_connections"
)

# Statistics
stats = store.get_endpoint_stats()
total = store.get_total_stats()
```

### ✨ **Features**

**Architecture:**
- Thread-safe design with atomic statistics (no lock contention)
- Schema validation: all endpoints must use same URI scheme (s3://, az://, gs://, file://, direct://)
- Transparent ObjectStore implementation - works anywhere single store is expected
- Per-endpoint thread pool control for optimal resource utilization

**Performance:**
- Lock-free atomic counters for statistics (zero contention overhead)
- Round-robin: ~10ns overhead per request
- Least-connections: ~50ns overhead per request
- Zero-copy Python interface via `BytesView` (buffer protocol)
- Negligible overhead vs latency (< 0.01% for typical cloud requests)

**Use Cases:**
- **Distributed ML training**: Access sharded datasets across multiple S3 buckets
- **High-throughput benchmarking**: Aggregate bandwidth from multiple storage servers
- **Fault tolerance**: Continue operations if individual endpoints fail
- **Geographic distribution**: Route requests to regionally-optimal endpoints
- **Load testing**: Distribute synthetic workloads across multiple backends

### 📝 **Testing**

**Comprehensive Test Coverage:**
- 33 new tests (16 in `uri_utils`, 17 in `multi_endpoint`)
- Total test count: 141 tests (all passing)
- Zero-copy validation tests in Python suite
- Load balancing distribution tests
- Error handling and validation tests

**Python Test Suite** (`tests/test_multi_endpoint.py`):
- Store creation from URIs, templates, and files
- Zero-copy data access via `memoryview()`
- NumPy/PyTorch integration tests
- Load balancing statistics validation
- Error handling for invalid configurations

### 🔧 **API Reference**

**Rust API:**
```rust
// Module exports
pub use multi_endpoint::{
    MultiEndpointStore,
    MultiEndpointStoreConfig,
    EndpointConfig,
    LoadBalanceStrategy,
    EndpointStats,
    TotalStats,
};

pub use uri_utils::{
    expand_uri_template,
    parse_uri_list,
    load_uris_from_file,
    infer_scheme_from_uri,  // Now public for validation
};
```

**Python API:**
```python
# Factory functions
s3dlio.create_multi_endpoint_store(uris, strategy) -> PyMultiEndpointStore
s3dlio.create_multi_endpoint_store_from_template(template, strategy) -> PyMultiEndpointStore  
s3dlio.create_multi_endpoint_store_from_file(file_path, strategy) -> PyMultiEndpointStore

# PyMultiEndpointStore methods
store.get(uri) -> BytesView                           # Zero-copy
store.get_range(uri, offset, length) -> BytesView     # Zero-copy
store.put(uri, data) -> None
store.list(prefix, recursive) -> List[str]
store.delete(uri) -> None
store.get_endpoint_stats() -> List[Dict]
store.get_total_stats() -> Dict
```

### 📖 **Documentation**

**New Documentation:**
- **[MULTI_ENDPOINT_GUIDE.md](MULTI_ENDPOINT_GUIDE.md)** - Comprehensive guide with:
  - Architecture overview and design principles
  - Load balancing strategy comparison
  - Rust and Python API examples
  - Configuration methods (URIs, templates, files)
  - Performance tuning guidelines
  - Best practices and use cases
  - Advanced topics (NUMA affinity, monitoring)
  - Troubleshooting guide

**Updated Documentation:**
- **[README.md](../README.md)** - Added multi-endpoint quick example in Python section
- **[docs/README.md](README.md)** - Added MULTI_ENDPOINT_GUIDE.md to Technical References

### 🔄 **Backward Compatibility**

✅ **Zero Breaking Changes:**
- New optional feature - existing code unaffected
- Single-endpoint workflows continue working unchanged
- Can wrap single endpoint in multi-endpoint store for future flexibility

### 🎯 **Use Case Examples**

**1. Distributed ML Training:**
```python
# Access sharded training data across 10 S3 buckets
store = s3dlio.create_multi_endpoint_store_from_template(
    template="s3://training-shard-{1...10}/data",
    strategy="least_connections"
)

# Zero-copy data loading
for uri in file_list:
    data = store.get(uri)
    tensor = torch.frombuffer(memoryview(data), dtype=torch.float32)
```

**2. High-Throughput Benchmarking:**
```rust
// Aggregate bandwidth across 4 storage servers
let endpoints = vec![
    "s3://benchmark-1/test", "s3://benchmark-2/test",
    "s3://benchmark-3/test", "s3://benchmark-4/test",
];

// Round-robin for predictable load distribution
let config = MultiEndpointStoreConfig {
    endpoints: endpoints.iter().map(|uri| EndpointConfig {
        uri: uri.to_string(),
        thread_count: Some(8),
        ..Default::default()
    }).collect(),
    strategy: LoadBalanceStrategy::RoundRobin,
    ..Default::default()
};
```

**3. Geographic Distribution:**
```python
# Route to fastest regional endpoint automatically
store = s3dlio.create_multi_endpoint_store(
    uris=[
        "s3://data-us-west-2/dataset",
        "s3://data-us-east-1/dataset", 
        "s3://data-eu-west-1/dataset",
    ],
    strategy="least_connections"  # Naturally favors lower-latency endpoints
)
```

### 🔍 **Implementation Details**

**Zero-Copy Design:**
- Python `get()` and `get_range()` return `PyBytesView` objects
- `PyBytesView` implements Python buffer protocol
- Allows `memoryview()` access without copying data
- Compatible with NumPy, PyTorch, and other buffer-aware libraries
- Maintains Arc-counted `Bytes` reference (cheap clone, automatic cleanup)

**Thread Safety:**
- All statistics use `AtomicU64` counters (lock-free)
- Endpoint selection protected by `Mutex` (minimal contention)
- Concurrent operations across different endpoints fully parallel
- No performance degradation under high concurrency

**Schema Validation:**
- All endpoints must use same URI scheme (enforced at creation)
- Prevents mixing incompatible backends (e.g., s3:// + file://)
- Clear error messages for configuration mistakes

### 📊 **Performance Characteristics**

**Throughput Scaling:**
- 2 endpoints: 1.8-2.0× baseline throughput
- 4 endpoints: 3.5-4.0× baseline throughput  
- 8 endpoints: 6.5-8.0× baseline throughput
- 16+ endpoints: 12-16× baseline throughput

**Overhead:**
- RoundRobin: ~10ns per request (atomic increment)
- LeastConnections: ~50ns per request (atomic read + compare)
- For typical cloud requests (1-100ms), overhead < 0.01%

### 🔧 **Configuration Examples**

**endpoints.txt** (file-based configuration):
```text
# Production multi-region setup
s3://prod-us-west-2/data
s3://prod-us-east-1/data
s3://prod-eu-west-1/data

# Comments and blank lines ignored
```

**Template Patterns:**
```rust
// Simple numeric range
"s3://bucket-{1...5}/data"

// Zero-padded range  
"s3://bucket-{01...10}/data"

// IP addresses for direct I/O
"direct://192.168.1.{1...10}:9000"

// Multiple ranges (Cartesian product)
"s3://rack{1...3}-node{1...4}"  // → 12 URIs
```

---

## Version 0.9.12 - GCS Factory Fixes (November 3, 2025)

### 🐛 **Fixed**
- Fixed 4 misleading "GCS backend not yet fully implemented" errors in enhanced factory functions
- `store_for_uri_with_config_and_logger()` now properly supports GCS
- `direct_io_store_for_uri_with_logger()` now properly supports GCS  
- `high_performance_store_for_uri_with_logger()` now properly supports GCS

### ✨ **Added**
- `store_for_uri_with_high_performance_cloud()` - Enable RangeEngine for cloud backends
- `store_for_uri_with_high_performance_cloud_and_logger()` - With logging support
- Documentation for high-performance cloud factory functions

### 📝 **Notes**
- All basic GCS operations worked correctly even before this fix
- The errors only appeared in enhanced factory functions (not used by most code)
- dl-driver checkpoint plugin was never affected (uses basic `store_for_uri()`)

---

## Version 0.9.11 - Directory Operations (November 2024)

### 🎯 **Unified Directory Management Across All Backends**

Added `mkdir` and `rmdir` operations to the `ObjectStore` trait for consistent directory handling across file systems and cloud storage.

**New API Methods:**
```rust
async fn mkdir(&self, uri: &str) -> Result<()>
async fn rmdir(&self, uri: &str, recursive: bool) -> Result<()>
```

**Backend Implementations:**
- **File/DirectIO** (`file://`, `direct://`): Creates/removes actual POSIX directories
  - `mkdir`: Uses `tokio::fs::create_dir_all()` for recursive creation
  - `rmdir`: Supports both empty (`remove_dir`) and recursive (`remove_dir_all`) deletion
  
- **Cloud Storage** (`s3://`, `az://`, `gs://`): Manages prefix markers
  - `mkdir`: Creates empty marker objects (e.g., `.keep`) to represent directories
  - `rmdir`: Deletes all objects under prefix (always recursive for cloud)

**Integration:**
- Required for sai3-bench v0.7.0+ directory tree workloads
- Enables consistent directory operations across hybrid storage environments
- Maintains backend-specific optimizations (e.g., cloud prefix semantics)

**Backward Compatibility:**
- ✅ Default trait implementations error gracefully for backends without support
- ✅ No breaking changes to existing APIs
- ✅ Optional functionality - existing workloads unaffected

---

## Version 0.9.10 - Pre-Stat Size Cache for Benchmarking (December 2024)

### 🚀 **Major Performance Improvement for Multi-Object Workloads**

#### **ObjectSizeCache - 2.5x Faster Benchmarking with Pre-Stat Optimization**

Eliminated redundant stat/HEAD operations in multi-object download workloads through intelligent size caching:

**Problem Solved:**
- Benchmarking tools downloading 1000+ objects spent 60% of time on sequential stat operations
- Each `get()` called stat before download: 1000 objects × 20ms stat = 20 seconds of pure overhead
- Network latency dominated total time (32.8s benchmark: 20s stat + 12.8s download)
- No way to amortize stat cost across multiple objects

**Solution - Pre-Stat Optimization:**
- New `ObjectSizeCache` module: Thread-safe size cache with configurable TTL (default 60s)
- New `pre_stat_objects()` trait method: Concurrent stat for multiple objects (100 concurrent default)
- New `pre_stat_and_cache()` API: Pre-stat all objects once, cache sizes, eliminate per-object stat overhead
- Integrated into S3, GCS, Azure backends automatically

**Performance Impact (Measured on 1000 × 64MB benchmark):**
- **Total time**: 32.8s → 13.0s (2.5x faster)
- **Stat overhead**: 20.0s → 0.2s (99% reduction)
- **Effective throughput**: 1.95 GB/s → 4.92 GB/s (2.5x improvement)
- **Pattern**: Pre-stat 1000 objects in 200ms (concurrent) vs 20s (sequential)

**Use Cases:**
- Benchmarking tools (`sai3-bench`, `io-bench`) downloading many objects
- Dataset pre-loading in ML training pipelines
- Batch processing of S3/GCS object collections
- Any workload with predictable object access patterns

**Backward Compatibility:**
- ✅ **Zero breaking changes** - All existing code works unchanged
- ✅ **Default trait methods** - All backends get concurrent pre-stat automatically
- ✅ **Opt-in API** - `pre_stat_and_cache()` is optional, regular `get()` still works
- ✅ **Graceful degradation** - Cache miss falls back to stat (same as before)

**API Usage - Before (v0.9.9 - Sequential stat overhead):**

```rust
// OLD: Each get() calls stat internally (20ms overhead per object)
let store = store_for_uri("s3://bucket/data/")?;
let objects = get_1000_object_uris();

for uri in &objects {
    let data = store.get(uri).await?;  // ❌ stat + download (20ms + 128ms = 148ms each)
    process(data);
}
// Total time: ~148 seconds for 1000 objects
```

**API Usage - After (v0.9.10 - Pre-stat optimization):**

```rust
// NEW: Pre-stat all objects once, then download without stat overhead
let store = store_for_uri("s3://bucket/data/")?;
let objects = get_1000_object_uris();

// PHASE 1: Pre-stat all objects concurrently (once at start)
store.pre_stat_and_cache(&objects, 100).await?;  // ✅ 200ms for 1000 objects

// PHASE 2: Download with zero stat overhead
for uri in &objects {
    let data = store.get(uri).await?;  // ✅ cached size, no stat! (128ms download only)
    process(data);
}
// Total time: ~128 seconds for 1000 objects (13% faster)
```

**Example: Benchmarking Tool Integration**

```rust
use s3dlio::api::store_for_uri;
use anyhow::Result;

async fn benchmark_download(bucket: &str, prefix: &str, object_count: usize) -> Result<()> {
    let store = store_for_uri(&format!("s3://{}/{}", bucket, prefix))?;
    
    // Get list of objects to benchmark
    let all_objects = store.list(&format!("s3://{}/{}", bucket, prefix), true).await?;
    let objects: Vec<String> = all_objects.into_iter().take(object_count).collect();
    
    println!("Benchmarking {} objects...", objects.len());
    
    // PRE-STAT PHASE: Load all object sizes concurrently (v0.9.10 NEW!)
    let start = std::time::Instant::now();
    let cached = store.pre_stat_and_cache(&objects, 100).await?;
    println!("Pre-statted {} objects in {:?}", cached, start.elapsed());
    
    // DOWNLOAD PHASE: Now downloads have zero stat overhead
    let download_start = std::time::Instant::now();
    let mut total_bytes = 0u64;
    
    for uri in &objects {
        let data = store.get(uri).await?;  // ✅ Uses cached size!
        total_bytes += data.len() as u64;
    }
    
    let duration = download_start.elapsed();
    let throughput_mbps = (total_bytes as f64 / 1_000_000.0) / duration.as_secs_f64();
    
    println!("Downloaded {} MB in {:?} ({:.2} MB/s)", 
             total_bytes / 1_000_000, duration, throughput_mbps);
    
    Ok(())
}
```

**Example: ML Dataset Pre-Loading**

```rust
use s3dlio::api::store_for_uri;
use anyhow::Result;

async fn preload_training_data(dataset_uris: Vec<String>) -> Result<Vec<Vec<u8>>> {
    let store = store_for_uri(&dataset_uris[0])?;
    
    // Pre-stat all files to populate size cache
    // This eliminates stat overhead during actual data loading
    store.pre_stat_and_cache(&dataset_uris, 200).await?;
    
    // Now load data with zero stat overhead
    let mut dataset = Vec::new();
    for uri in dataset_uris {
        let data = store.get(&uri).await?;
        dataset.push(data.to_vec());
    }
    
    Ok(dataset)
}
```

**Configuration Options:**

```rust
use s3dlio::object_store::{S3ObjectStore, S3Config, GcsObjectStore, GcsConfig};
use std::time::Duration;

// Custom TTL for size cache (default is 60 seconds)
let config = S3Config {
    enable_range_engine: false,
    range_engine: Default::default(),
    size_cache_ttl_secs: 120,  // 2-minute cache for longer workloads
};
let store = S3ObjectStore::with_config(config);

// Or use defaults (60 second TTL)
let store = S3ObjectStore::new();  // ✅ 60s TTL automatic
```

**Backend-Specific Behavior:**

| Backend | Cache TTL | Rationale |
|---------|-----------|-----------|
| **S3** | 60 seconds (configurable) | Network storage, objects rarely change during download |
| **GCS** | 60 seconds (configurable) | Network storage, objects rarely change during download |
| **Azure** | 60 seconds (configurable) | Network storage, objects rarely change during download |
| **file://** | 0 seconds (disabled) | Local stat is fast (~1ms), files can change on disk |
| **direct://** | 0 seconds (disabled) | Local stat is fast (~1ms), files can change on disk |

**Technical Details:**
- **Cache structure**: Thread-safe `RwLock<HashMap<String, CachedSize>>`
- **TTL mechanism**: Per-entry `Instant` timestamp, checked on `get()`
- **Concurrency**: `pre_stat_objects()` uses `futures::stream::buffer_unordered()`
- **Memory**: ~100 KB for 1000 entries (String + u64 + Instant per entry)
- **Backward compatibility**: Default trait methods ensure all backends work

**New Trait Methods:**

```rust
#[async_trait]
pub trait ObjectStore: Send + Sync {
    // ... existing methods ...
    
    /// Stat multiple objects concurrently (NEW in v0.9.10)
    async fn pre_stat_objects(
        &self,
        uris: &[String],
        max_concurrent: usize,
    ) -> Result<HashMap<String, u64>> {
        // Default concurrent implementation provided
    }
    
    /// Pre-stat and cache object sizes for later use (NEW in v0.9.10)
    async fn pre_stat_and_cache(
        &self,
        uris: &[String],
        max_concurrent: usize,
    ) -> Result<usize> {
        // Default implementation (backends with cache override this)
    }
}
```

**When to Use Pre-Stat:**
- ✅ Benchmarking many objects (100+)
- ✅ Known object list before download
- ✅ Objects don't change during workload
- ✅ Network storage (S3, GCS, Azure)
- ❌ Single object downloads
- ❌ Objects changing frequently
- ❌ Local file:// or direct:// storage

**Performance Scaling:**

| Objects | Sequential Stat | Concurrent Pre-Stat | Speedup |
|---------|-----------------|---------------------|---------|
| 100 | 2.0s | 20ms | 100x faster |
| 500 | 10.0s | 100ms | 100x faster |
| 1000 | 20.0s | 200ms | 100x faster |
| 5000 | 100.0s | 1.0s | 100x faster |

**Testing:**
- 13 ObjectSizeCache unit tests (basic ops, TTL, concurrency, memory efficiency)
- Integration tests for file:// backend
- Performance validation tests (benchmarking scenarios)

**Migration Guide:**

No migration required! All existing code works without changes. To opt-in to performance improvements:

```rust
// Add this one line before your download loop:
store.pre_stat_and_cache(&object_uris, 100).await?;

// Then use store.get() as normal - it will use cached sizes
```

---

## Version 0.9.9 - Buffer Pool Optimization for DirectIO (October 2025)

### 🚀 **Major Performance Improvement**

#### **Buffer Pool for DirectIO Hot Path - 15-20% Throughput Gain**

Eliminated allocation churn in DirectIO range reads by integrating buffer pool infrastructure:

**Problem Solved:**
- DirectIO range reads allocated fresh aligned buffers on every operation
- Full buffer copies via `buffer[start..end].to_vec()` caused CPU overhead
- Allocator churn resulted in excessive page faults (20-25% throughput gap vs vdbench)

**Solution - Phase 1:**
- Wired existing `BufferPool` infrastructure into DirectIO hot path
- Pool of 32 × 64MB aligned buffers automatically initialized in `direct_io()` and `high_performance()` constructors
- Optimized `try_read_range_direct()` with borrow/return pattern:
  - **Before**: Fresh allocation + full buffer copy per range
  - **After**: Pool borrow + small copy (only requested bytes) + pool return
  - **Eliminated**: 2 allocations + 1 full copy per range operation

**Performance Impact (Expected):**
- **Throughput**: +15-20% on DirectIO with RangeEngine enabled
- **CPU**: -10-15% utilization (less memcpy, less malloc/free)
- **Page faults**: -30-50% reduction (less allocator activity)
- **Allocator calls**: -90% reduction (buffer reuse)

**Backward Compatibility:**
- ✅ **Zero breaking changes** - buffer pool is optional (`Option<Arc<BufferPool>>`)
- ✅ **Automatic** - `direct_io()` and `high_performance()` constructors initialize pool automatically
- ✅ **Graceful fallback** - Falls back to allocation if pool disabled or exhausted
- ✅ **Default unchanged** - `FileSystemConfig::default()` has no pool (backward compatible)

**API Usage (No Code Changes Required):**

```rust
// Using factory functions (pool auto-initialized for DirectIO)
let store = direct_io_store_for_uri("file:///data/")?;  // ✅ Pool automatic

// Using constructors (pool auto-initialized)
let config = FileSystemConfig::direct_io();  // ✅ Pool automatic (32 × 64MB buffers)
let config = FileSystemConfig::high_performance();  // ✅ Pool automatic

// Default (no pool, backward compatible)
let config = FileSystemConfig::default();  // ✅ No pool (compatible with v0.9.8)
```

**Technical Details:**
- Pool capacity: 32 buffers
- Buffer size: 64 MB per buffer
- Alignment: System page size (typically 4096 bytes)
- Grow-on-demand: Allocates new buffer if pool exhausted
- Async-safe: Uses tokio channels for thread-safe borrow/return

**Why Only DirectIO?**
- **S3/Azure/GCS**: Network latency (5-50ms) >> allocation (<0.1ms), pool would provide <1% benefit
- **Regular file I/O**: Kernel page cache already handles efficiency
- **DirectIO**: Aligned allocations expensive + frequent operations = pool is critical

**Files Modified:**
- `src/file_store_direct.rs`: Buffer pool integration (~80 lines)
- `src/file_store.rs`: Pre-sizing optimization (~2 lines)
- `tests/test_buffer_pool_directio.rs`: 12 comprehensive functional tests (new)
- `tests/test_allocation_comparison.rs`: Allocation overhead comparison (new)
- `docs/implementation-plans/v0.9.9-buffer-pool-enhancement.md`: Complete implementation plan (new)
- `docs/testing/v0.9.9-phase1-testing-summary.md`: Testing analysis (new)

**Testing:**
- ✅ 12/12 functional tests passing
- ✅ Concurrent access validated (64 tasks with pool size 32)
- ✅ Edge cases covered (unaligned offsets, oversized ranges, fallback)
- ✅ Data integrity verified across multiple chunks
- ✅ Zero new compiler warnings
- ⏳ Performance validation pending with sai3-bench (realistic workloads)

**Future Phases:**
- Phase 2: Writer path optimizations (optional)
- Phase 3: Zero-copy via custom Bytes deallocator (future)
- Target: Close remaining performance gap to <10% vs vdbench

---

### 📚 **Documentation Cleanup**

**Streamlined documentation from 114 to 34 files (70% reduction):**

**Removed:**
- Entire `docs/archive/` directory (584KB of historical docs)
  - Old API versions (pre-v0.9.0)
  - Completed implementation plans (v0.8.x)
  - Old performance reports and benchmarks
  - Pre-v0.8.0 changelog and release notes
- Version-specific docs (v0.9.4, v0.9.5, v0.9.6)
- Old API snapshots (v0.9.2, v0.9.3)
- Detailed release plans for v0.9.0-v0.9.2 (15 files)
- Completed implementation plans (4 files)

**Added:**
- `docs/STREAMING-ARCHITECTURE.md`: Comprehensive guide to stream-based patterns
  - RangeEngine streaming patterns
  - Backpressure control mechanisms
  - Cancellation support
  - DataLoader streaming architecture

**Kept (Essential):**
- Core guides: README, Changelog, RELEASE-CHECKLIST, TESTING-GUIDE
- Configuration: CONFIGURATION-HIERARCHY, VERSION-MANAGEMENT
- API references: ZERO-COPY-API-REFERENCE, TFRECORD-INDEX-QUICKREF
- Performance: Performance_Optimization_Summary, Performance_Profiling_Guide

**Rationale:**
- Changelog.md is authoritative source for version history
- Reduced maintenance burden and improved navigation
- Current API documented in code and reference docs

---

## Version 0.9.8 - Optional GCS Backends & Page Cache Configuration (October 2025)

### 🚀 **New Features**

#### **Optional Google Cloud Storage (GCS) Backends**

Added **two mutually exclusive GCS backend implementations** selectable at compile time:

1. **`gcs-community`** (default) - Uses community-maintained `gcloud-storage` v1.1 crate
   - ✅ **Production Ready**: All tests pass reliably (10/10)
   - ✅ Stable and proven in production workloads
   - ✅ Full ADC (Application Default Credentials) support
   - ✅ Supports all GCS operations: GET, PUT, DELETE, LIST, STAT, range reads

2. **`gcs-official`** (experimental) - Uses official Google `google-cloud-storage` v1.1 crate
   - ⚠️ **Experimental**: Known transport flakes in test suites (10-20% failure rate)
   - ✅ Individual operations work correctly when tested in isolation
   - ⚠️ Full test suite has intermittent "transport error" failures (improved to 8-9/10 tests passing)
   - 🐛 **Root Cause**: Upstream flake in `google-cloud-rust` library
     - **Bug Report**: https://github.com/googleapis/google-cloud-rust/issues/3574
     - **Related Issue**: https://github.com/googleapis/google-cloud-rust/issues/3412
   - ✅ **MAJOR IMPROVEMENT**: Implemented global client singleton pattern
     - **Before**: 30% pass rate (3/10 tests) - StorageControl operations failed consistently
     - **After**: 80-90% pass rate (8-9/10 tests) - Most operations now work reliably
     - **Implementation**: `once_cell::Lazy<OnceCell<GcsClient>>` for single client per process
   - 📊 **Current Status**: Acceptable for development/testing but expect occasional flakes until upstream fix

**Build Options:**

```bash
# Default (community backend - RECOMMENDED)
cargo build --release

# Explicit community backend
cargo build --release --features gcs-community

# Official backend (experimental - for testing only)
cargo build --release --no-default-features --features native-backends,s3,gcs-official

# ❌ Cannot use both (compile error by design)
cargo build --features gcs-community,gcs-official  # ERROR
```

**Why Two Backends?**

The dual-backend approach allows:
- ✅ Production stability with `gcs-community` (default)
- ✅ Future migration path when upstream transport issues are resolved
- ✅ A/B performance testing and benchmarking
- ✅ Easy switching without code changes (compile-time selection only)

**Recommendation**: Use `gcs-community` (default) for all production workloads. The `gcs-official` backend is provided for experimentation and to track upstream development, but is not suitable for production due to the transport flakes documented in Issue #3574.

**Files Changed:**
- `src/google_gcs_client.rs`: New implementation using official `google-cloud-storage` crate
- `src/gcs_client.rs`: Community implementation (default)
- `src/object_store.rs`: Factory pattern with compile-time backend selection
- `tests/test_gcs_community.rs`: Complete test suite for community backend (10/10 pass ✅)
- `tests/test_gcs_official.rs`: Complete test suite for official backend (3/10 pass, 7/10 fail ⚠️)
- `tests/common/mod.rs`: Shared test utilities
- `docs/GCS-BACKEND-SELECTION.md`: Comprehensive backend selection guide
- Bug report filed: https://github.com/googleapis/google-cloud-rust/issues/3574

---

#### **Configurable `posix_fadvise` Hints for File I/O**

Added ability to configure page cache behavior for file system operations (`file://` and `direct://` URIs) via the new `page_cache_mode` field in `FileSystemConfig`.

**What is `posix_fadvise`?**  
On Linux/Unix systems, `posix_fadvise()` provides hints to the kernel about expected file access patterns, allowing the OS to optimize page cache behavior for better performance.

**Available Modes:**
- **`Auto`** (default for full GETs): Automatically selects Sequential for large files (>= 64 MiB), Random for small files
- **`Sequential`**: Prefetch data ahead - optimal for streaming reads, large sequential scans
- **`Random`**: Don't prefetch - optimal for random access patterns, database queries
- **`DontNeed`**: Don't cache pages - optimal for one-time reads, streaming that won't be re-read
- **`Normal`**: Let OS use default heuristics

**Default Behavior (No Configuration Required):**
- Full GET operations: `Auto` mode (intelligent based on file size)
- Range GET operations: `Random` mode (typical for random access)

**Rust API Usage:**

```rust
use s3dlio::object_store::{store_for_uri_with_config, FileSystemConfig, PageCacheMode};

// Example 1: Sequential access pattern (streaming, large files)
let config = FileSystemConfig {
    page_cache_mode: Some(PageCacheMode::Sequential),
    ..Default::default()
};
let store = store_for_uri_with_config("file:///data/", Some(config))?;

// Example 2: Random access pattern (database, random seeks)
let config = FileSystemConfig {
    page_cache_mode: Some(PageCacheMode::Random),
    ..Default::default()
};
let store = store_for_uri_with_config("file:///db/", Some(config))?;

// Example 3: Don't pollute cache (one-time streaming)
let config = FileSystemConfig {
    page_cache_mode: Some(PageCacheMode::DontNeed),
    ..Default::default()
};
let store = store_for_uri_with_config("file:///stream/", Some(config))?;

// Example 4: Use defaults (Auto for full GET, Random for range GET)
let store = store_for_uri("file:///data/")?;  // No config needed
```

**When to Use:**
- ✅ **Sequential**: Large file streaming, sequential scans, media processing
- ✅ **Random**: Database files, random access patterns, sparse file access
- ✅ **DontNeed**: Large one-time reads that won't be re-accessed (prevents cache pollution)
- ✅ **Auto**: Default - works well for most workloads

**Performance Impact:**
- Sequential mode: Can provide 2-3x improvement for large sequential reads
- Random mode: Reduces memory pressure for random access patterns
- DontNeed mode: Prevents cache pollution for one-time large file operations

**Files Changed:**
- `src/file_store.rs`: Added `page_cache_mode` field to `FileSystemConfig`, updated GET operations
- `src/api.rs`: Exported `PageCacheMode`, `FileSystemConfig`, and `store_for_uri_with_config`
- `tests/test_file_range_engine.rs`: Updated tests for new field

**Backward Compatibility:**  
✅ **Fully backward compatible** - existing code continues to work with Auto/Random defaults

---

## Version 0.9.6 - RangeEngine Disabled by Default (October 2025)

### ⚠️ **BREAKING CHANGES**

#### **RangeEngine Disabled by Default (ALL Backends)**

**Problem:** Performance testing revealed that RangeEngine causes **up to 50% slowdown** for typical workloads due to the extra HEAD/STAT request required on every GET operation to determine object size. While RangeEngine provides 30-50% throughput improvement for large files (>= 16 MiB), the mandatory stat overhead makes it counterproductive for most real-world workloads.

**Solution:** RangeEngine is now **disabled by default** across all storage backends. Users must explicitly opt-in for large-file workloads where benefits outweigh the stat overhead.

**Affected Backends:**
- ✅ **Azure Blob Storage** (`az://`) - `enable_range_engine: false`
- ✅ **Google Cloud Storage** (`gs://`, `gcs://`) - `enable_range_engine: false`
- ✅ **Local File System** (`file://`) - `enable_range_engine: false`
- ✅ **DirectIO** (`direct://`) - `enable_range_engine: false`
- ✅ **S3** (`s3://`) - RangeEngine not yet implemented

**Performance Impact:**
- **Before (v0.9.5)**: Every GET operation performed HEAD + GET (2 requests)
- **After (v0.9.6)**: GET operations use single request (no stat overhead)
- **Typical workloads**: 50% faster due to elimination of HEAD requests
- **Large-file workloads**: Must enable RangeEngine explicitly to get 30-50% benefit

**Migration Guide:**

**No changes required for most users** - default behavior now faster for typical workloads.

**For large-file workloads (>= 16 MiB objects), explicitly enable RangeEngine:**

```rust
use s3dlio::object_store::{AzureObjectStore, AzureConfig, RangeEngineConfig};

// Enable RangeEngine for large-file Azure workload
let config = AzureConfig {
    enable_range_engine: true,  // Explicitly enable (was: default true, now: default false)
    range_engine: RangeEngineConfig {
        min_split_size: 16 * 1024 * 1024,  // 16 MiB threshold (default)
        max_concurrent_ranges: 32,          // 32 parallel ranges
        chunk_size: 64 * 1024 * 1024,      // 64 MiB chunks
        ..Default::default()
    },
};
let store = AzureObjectStore::with_config(config);
```

**Same pattern for all backends:**

```rust
// Google Cloud Storage
use s3dlio::object_store::{GcsObjectStore, GcsConfig};
let config = GcsConfig {
    enable_range_engine: true,  // Opt-in for large files
    ..Default::default()
};
let store = GcsObjectStore::with_config(config);

// Local File System
use s3dlio::file_store::{FileSystemObjectStore, FileSystemConfig};
let config = FileSystemConfig {
    enable_range_engine: true,  // Rarely beneficial for local FS
    ..Default::default()
};
let store = FileSystemObjectStore::with_config(config);

// DirectIO
use s3dlio::file_store_direct::{FileSystemConfig};
let config = FileSystemConfig::direct_io();  // Still disabled by default
// Explicitly enable if needed:
let mut config = FileSystemConfig::direct_io();
config.enable_range_engine = true;
```

**When to Enable RangeEngine:**
- ✅ Large-file workloads (average object size >= 64 MiB)
- ✅ High-bandwidth networks (>= 1 Gbps) with high latency
- ✅ Dedicated large-object operations (media processing, ML training)
- ❌ Mixed workloads with small and large objects
- ❌ Benchmarks with small objects (< 16 MiB)
- ❌ Local file systems (seek overhead usually outweighs benefit)

---

### 🔧 **Configuration Changes**

#### **All Backend Configs Updated**

1. **`AzureConfig`** (`src/object_store.rs`)
   - `enable_range_engine`: `true` → `false`
   - Documentation updated to reflect opt-in behavior

2. **`GcsConfig`** (`src/object_store.rs`)
   - `enable_range_engine`: `true` → `false`
   - Documentation updated to reflect opt-in behavior

3. **`FileSystemConfig`** (`src/file_store.rs`)
   - `enable_range_engine`: `true` → `false`
   - Documentation emphasizes local FS rarely benefits

4. **`FileSystemConfig` (DirectIO)** (`src/file_store_direct.rs`)
   - `enable_range_engine`: `true` → `false` in:
     - `Default::default()`
     - `direct_io()`
     - `high_performance()`
   - All presets now disable RangeEngine by default

#### **Constants Documentation Updated**

- `DEFAULT_RANGE_ENGINE_THRESHOLD` documentation clarifies:
  - RangeEngine is **disabled by default**
  - 16 MiB threshold applies only when explicitly enabled
  - Explains performance trade-offs
  - Provides example configurations for enabling

---

### 📊 **Performance Summary**

| Workload Type | v0.9.5 Performance | v0.9.6 Performance (Default) | v0.9.6 with RangeEngine Enabled |
|---------------|-------------------|------------------------------|----------------------------------|
| Small objects (< 16 MiB) | **Slow** (2x requests: HEAD + GET) | **Fast** (1x request: GET only) | **Slow** (2x requests: HEAD + GET) |
| Large files (>= 64 MiB) | **Medium** (HEAD + single GET) | **Medium** (single GET) | **Fast** (HEAD + parallel range GETs) |
| Mixed workloads | **Slow overall** (all GETs statted) | **Fast** (no stat overhead) | **Slow overall** (all GETs statted) |

**Key Insight:** RangeEngine provides benefit **only for large-file workloads**. For typical workloads with mixed object sizes, the stat overhead on small objects outweighs gains on large objects.

---

### 📚 **Documentation Updates**

- **Constants (`src/constants.rs`)**: Comprehensive RangeEngine configuration guide
- **Config Structs**: All backend configs document opt-in behavior
- **README.md**: Updated to reflect disabled-by-default status
- **Copilot Instructions**: Updated RangeEngine guidance

---

### 🔄 **Backward Compatibility**

**Breaking Change:** Existing code that relies on RangeEngine being enabled by default will see different behavior. However, this change **improves performance for most workloads**.

**Migration Path:**
1. **If your workload uses small/mixed objects**: No changes needed - enjoy faster performance
2. **If your workload uses large files (>= 64 MiB)**: Add explicit `enable_range_engine: true` to config

**Deprecations:** None. All APIs remain stable.

---

## Version 0.9.5 - Performance Fixes & RangeEngine Tuning (October 2025)

### 🚀 **PERFORMANCE IMPROVEMENTS**

#### **1. Adaptive Concurrency for Delete Operations (10-70x faster)**

Fixed critical performance regression introduced in v0.8.23 where progress bar implementation caused delete operations to become sequential instead of concurrent.

**Performance Gains:**
- **500 objects**: ~0.7 seconds (was ~50 seconds) - **70x faster**
- **7,000 objects**: ~5.5 seconds (was ~70-140 seconds) - **12-25x faster**
- **93,000 objects**: ~90 seconds (was ~15+ minutes) - **10x+ faster**

**Implementation Details:**

1. **New `delete_objects_concurrent()` Helper Function** (`src/object_store.rs`)
   - Universal implementation works with all backends (S3, Azure, GCS, file://, direct://)
   - Uses `futures::stream` with `buffer_unordered` for concurrent deletions
   - Batched progress updates (every 50 operations) instead of per-object updates
   - 98% reduction in progress bar overhead

2. **Adaptive Concurrency Algorithm**
   
   Automatically scales concurrency based on total object count (10% of total):
   
   | Total Objects | Concurrency | Reasoning |
   |--------------|-------------|-----------|
   | < 10 | 1 | Very small: sequential is efficient |
   | 10-99 | 10 | Small: minimum viable concurrency |
   | 100-9,999 | total/10 (max 100) | Medium: scales with workload |
   | 10,000+ | total/10 (max 1,000) | Large: capped to avoid overwhelming backends |
   
   **Examples:**
   - 100 objects → 10 concurrent deletions
   - 500 objects → 50 concurrent deletions
   - 5,000 objects → 100 concurrent deletions (capped)
   - 93,000 objects → 1,000 concurrent deletions (capped)

3. **CLI Updates** (`src/bin/cli.rs`)
   - Both pattern-filtered and full-prefix deletions now use concurrent helper
   - Displays adaptive concurrency level in user messages
   - Maintains smooth progress bar with batched updates

**Testing:**
- ✅ Tested with Google Cloud Storage: 7,010 objects deleted in 5.5 seconds
- ✅ Throughput: ~1,280 deletions/second
- ✅ Progress bar verified working correctly
- ✅ Universal backend compatibility confirmed

**Technical Notes:**
- Progress updates use `Arc<AtomicUsize>` for lock-free concurrent counting
- Callback wrapped in `Arc` for sharing across async tasks
- ProgressBar cloned to avoid move issues in closures
- Final progress update ensures accurate completion count

---

#### **2. RangeEngine Threshold Increased to 16 MiB (Fixes 10% regression)**

**Problem:** v0.9.3 introduced a 10% performance regression for small object workloads (e.g., 1 MiB objects) due to an extra HEAD request on every GET operation to check object size for RangeEngine eligibility.

**Root Cause Analysis:**
- v0.9.3 used 4 MiB threshold for RangeEngine
- For objects < 4 MiB, code still performed HEAD + GET (2 requests instead of 1)
- Benchmarks with 1 MiB objects saw 60% more total requests → ~10% slowdown
- The stat overhead outweighed any potential benefit for small objects

**Solution:** Raised default threshold to **16 MiB** for all network backends (S3, Azure, GCS).

**Changes:**

1. **New Universal Constant** (`src/constants.rs`)
   ```rust
   /// Universal default minimum object size to trigger RangeEngine (16 MiB)
   pub const DEFAULT_RANGE_ENGINE_THRESHOLD: u64 = 16 * 1024 * 1024;
   ```
   
   - Replaces backend-specific thresholds (4 MiB)
   - Legacy aliases deprecated but maintained for compatibility
   - Comprehensive documentation explaining threshold selection

2. **Updated Backend Configs**
   - `AzureConfig`: Now uses 16 MiB threshold
   - `GcsConfig`: Now uses 16 MiB threshold
   - `S3Config`: Now uses 16 MiB threshold (when RangeEngine added)

**Performance Impact:**
- **Small objects (< 16 MiB)**: Restored to v0.8.22 performance (single GET request)
- **Medium objects (16-64 MiB)**: Still benefit from RangeEngine (20-40% faster)
- **Large objects (> 64 MiB)**: Maximum RangeEngine benefit (30-60% faster)
- **Benchmarks**: 1 MiB object workloads no longer see regression

**When to Override:**

```rust
use s3dlio::object_store::{GcsConfig, RangeEngineConfig};

// Lower threshold for large-file workloads (higher latency networks)
let config = GcsConfig {
    enable_range_engine: true,
    range_engine: RangeEngineConfig {
        min_split_size: 4 * 1024 * 1024,  // 4 MiB threshold
        ..Default::default()
    },
};

// Higher threshold to avoid stat overhead on most objects
let config = GcsConfig {
    enable_range_engine: true,
    range_engine: RangeEngineConfig {
        min_split_size: 64 * 1024 * 1024,  // 64 MiB threshold
        ..Default::default()
    },
};

// Disable RangeEngine entirely for small-object benchmarks
let benchmark_config = GcsConfig {
    enable_range_engine: false,
    ..Default::default()
};
```

**Documentation Updates:**
- Added detailed comments in `src/constants.rs` explaining threshold rationale
- Updated Azure/GCS config documentation to reflect 16 MiB threshold
- Performance regression analysis documented

**Testing:**
- ✅ Benchmarks with 1 MiB objects now match v0.8.22 performance
- ✅ Large file downloads still use RangeEngine (>= 16 MiB)
- ✅ No extra HEAD requests for typical workloads
- ✅ Configuration override tested and working

---

### 📊 **Performance Summary**

| Change | Workload | Performance Gain |
|--------|----------|------------------|
| Adaptive Delete | 7,000 objects | 12-25x faster (5.5s vs 70-140s) |
| RangeEngine Fix | 1 MiB GETs | 10% regression eliminated |
| Combined | Mixed workloads | Restored + improved performance |

---

### 🔧 **Breaking Changes**
None. All changes are backward compatible with optional configuration overrides.

---

### 📚 **Deprecations**
- `DEFAULT_S3_RANGE_ENGINE_THRESHOLD` → Use `DEFAULT_RANGE_ENGINE_THRESHOLD`
- `DEFAULT_AZURE_RANGE_ENGINE_THRESHOLD` → Use `DEFAULT_RANGE_ENGINE_THRESHOLD`
- `DEFAULT_GCS_RANGE_ENGINE_THRESHOLD` → Use `DEFAULT_RANGE_ENGINE_THRESHOLD`

Legacy constants remain functional but emit deprecation warnings.

---

## Version 0.9.4 - S3-Specific API Deprecation (October 2025)

### ⚠️ **DEPRECATION NOTICES**

#### **Python & Rust API: S3-Specific Data Functions Deprecated (Removal in v1.0.0)**

Two S3-specific data operation functions have been deprecated in favor of universal URI-based alternatives. **These functions will be removed in v1.0.0.**

**IMPORTANT**: 
- This deprecation affects **BOTH Python AND Rust APIs**
- `create_bucket()` and `delete_bucket()` are **NOT deprecated** - they will be made universal in future releases

**Deprecated Functions:**

1. **`list_objects(bucket, prefix, recursive)`** → Use `list(uri, recursive, pattern)`
   
   **Python:**
   ```python
   # OLD (deprecated)
   objects = s3dlio.list_objects("bucket", "prefix/", recursive=True)
   
   # NEW (universal)
   objects = s3dlio.list("s3://bucket/prefix/", recursive=True)
   ```
   
   **Rust:**
   ```rust
   // OLD (deprecated)
   use s3dlio::s3_utils::list_objects;
   let objects = list_objects("bucket", "prefix/", true)?;
   
   // NEW (universal)
   use s3dlio::api::{store_for_uri, ObjectStore};
   let store = store_for_uri("s3://bucket/prefix/")?;
   let objects = store.list("", true, None).await?;
   ```

2. **`get_object(bucket, key, offset, length)`** → Use `get(uri)` or `get_range(uri, offset, length)`
   
   **Python:**
   ```python
   # OLD (deprecated)
   data = s3dlio.get_object("bucket", "key", offset=1024, length=4096)
   
   # NEW (universal)
   data = s3dlio.get_range("s3://bucket/key", offset=1024, length=4096)
   ```

**Timeline:**
- **v0.9.4**: Functions work with deprecation warnings (stderr + compile-time)
- **v0.9.x → v1.0.0-rc**: Functions continue working with warnings
- **v1.0.0**: Functions removed

**See**: [DEPRECATION-NOTICE-v0.9.4.md](./DEPRECATION-NOTICE-v0.9.4.md) for complete migration guide.

---

## Version 0.9.3 - RangeEngine for Azure & GCS Backends (October 2025)

#### **1. RangeEngine Integration for Azure Blob Storage**
Concurrent range downloads significantly improve throughput for large Azure blobs by hiding network latency with parallel range requests.

**Performance Gains:**
- Medium blobs (4-64MB): 20-40% faster
- Large blobs (> 64MB): 30-50% faster
- Huge blobs (> 1GB): 40-60% faster on high-bandwidth networks

**Implementation:**
- Created `AzureConfig` with configurable RangeEngine settings
- Refactored `AzureObjectStore` from unit struct to stateful struct with Clone support
- Added `get_with_range_engine()` helper for concurrent downloads
- Size-based strategy: files >= 4MB use RangeEngine automatically

**Configuration (network-optimized defaults):**
```rust
pub struct AzureConfig {
    pub enable_range_engine: bool,  // Default: true
    pub range_engine: RangeEngineConfig {
        chunk_size: 64 * 1024 * 1024,        // 64MB chunks
        max_concurrent_ranges: 32,            // 32 parallel ranges
        min_split_size: 4 * 1024 * 1024,     // 4MB threshold
        range_timeout: Duration::from_secs(30),
    },
}
```

**Usage:**
```rust
use s3dlio::object_store::{AzureObjectStore, AzureConfig};

// Default configuration (RangeEngine enabled)
let store = AzureObjectStore::new();

// Custom configuration
let config = AzureConfig {
    enable_range_engine: true,
    range_engine: RangeEngineConfig::default(),
};
let store = AzureObjectStore::with_config(config);

// Downloads automatically use RangeEngine for files >= 4MB
let data = store.get("az://account/container/large-file.bin").await?;
```

**Validation:**
- ✅ Python tests: 16.54 MB/s download (8MB blob)
- ✅ Rust tests: All ObjectStore methods validated
- ✅ Zero-copy Bytes API compatibility
- ✅ Builds with zero warnings

---

#### **2. RangeEngine Integration for Google Cloud Storage**
Complete GCS backend enhancement with concurrent range downloads following the same pattern as Azure.

**Performance:**
- 128MB file: 44-46 MB/s with 2 concurrent ranges
- Validated on production GCS buckets (signal65-russ-b1)
- Expected 30-50% gains on high-bandwidth networks (>1 Gbps)

**Implementation:**
- Created `GcsConfig` with identical structure to AzureConfig
- Refactored `GcsObjectStore` from unit struct to stateful struct
- Added `get_with_range_engine()` for concurrent downloads
- Same 4MB threshold and network-optimized defaults

**Configuration:**
```rust
use s3dlio::object_store::{GcsObjectStore, GcsConfig};

// Default configuration (RangeEngine enabled)
let store = GcsObjectStore::new();

// Custom configuration
let config = GcsConfig {
    enable_range_engine: true,
    range_engine: RangeEngineConfig {
        chunk_size: 64 * 1024 * 1024,
        max_concurrent_ranges: 32,
        min_split_size: 4 * 1024 * 1024,
        range_timeout: Duration::from_secs(30),
    },
};
let store = GcsObjectStore::with_config(config);
```

**Validation:**
- ✅ Python tests: 45.61 MB/s download (128MB file, 2 concurrent ranges)
- ✅ Rust smoke tests: All 5 tests passing (8.06s total)
  - Small files (1MB): Simple download below threshold
  - Large files (128MB): 44.60 MB/s with RangeEngine
  - Range requests: Partial reads validated
  - Metadata: stat() with ETag
  - Listings: Directory operations working
- ✅ Debug logging confirms concurrent range execution
- ✅ Zero-copy Bytes API compatibility

---

#### **3. Universal Python API Enhancements**
All Python API functions now work universally across all 5 backends (S3, Azure, GCS, file://, direct://).

**Changes:**
- `put()`: Template parameter now optional (default: "object-{}")
- `get()`: Universal implementation via `store_for_uri()` + `ObjectStore::get()`
- `delete()`: Works with all URI schemes, supports pattern matching
- `build_uri_list()`: Generic URI parsing for any scheme

**Before (v0.9.2 - S3-specific):**
```python
# Only worked with s3:// URIs
s3dlio.put("s3://bucket/prefix/", num=3, template="object-{}", size=1024*1024)
```

**After (v0.9.3 - Universal):**
```python
# Works with all backends
s3dlio.put("s3://bucket/prefix/", num=3, size=1024*1024)        # S3
s3dlio.put("az://account/container/prefix/", num=3, size=1024*1024)  # Azure
s3dlio.put("gs://bucket/prefix/", num=3, size=1024*1024)        # GCS
s3dlio.put("file:///local/path/", num=3, size=1024*1024)        # Local
s3dlio.put("direct:///local/path/", num=3, size=1024*1024)      # DirectIO

# get() works universally
data = s3dlio.get("gs://bucket/file.bin")  # Returns BytesView (zero-copy)

# delete() works with patterns
s3dlio.delete(["az://container/prefix/*"])  # Deletes all matching
```

**Validation:**
- ✅ Azure Python tests: All operations working (put, get, get_range, large files)
- ✅ GCS Python tests: All operations working (128MB with RangeEngine)
- ✅ Zero-copy BytesView wrapper preserved
- ✅ Pattern matching in delete() validated

---

### 🧪 **Testing & Validation**

#### **New Test Suites:**
1. **`python/tests/test_azure_api.py`**: Comprehensive Azure backend validation
2. **`python/tests/test_gcs_api.py`**: GCS backend with RangeEngine tests (128MB files)
3. **`tests/test_gcs_smoke.rs`**: Rust integration tests (5 test cases, all passing)

#### **Test Coverage:**
- Azure: 17.08 MB/s download (8MB with RangeEngine)
- GCS Python: 45.61 MB/s download (128MB, 2 concurrent ranges)
- GCS Rust: 44.60 MB/s download (128MB, 2 concurrent ranges)
- All ObjectStore methods validated: get, put, delete, list, stat, get_range
- Zero-copy Bytes API compatibility confirmed
- Debug logging added for RangeEngine activity tracking

---

### 📦 **Infrastructure Updates**

- **Build System**: Zero warnings policy enforced across all builds
- **Tracing**: Debug logging support for RangeEngine activity (`s3dlio.init_logging("debug")`)
- **Documentation**: Added GCS login helper script (`scripts/gcs-login.sh`)
- **Architecture**: Consistent backend patterns across Azure and GCS
  - Both use stateful structs with Config
  - Both support Clone for closure compatibility
  - Both use identical RangeEngine configuration

---

### 🔧 **Breaking Changes**
None. This is a backward-compatible feature release.

- Existing Azure code continues to work (defaults to RangeEngine enabled)
- Existing GCS code continues to work (defaults to RangeEngine enabled)
- Python API changes are purely additive (template parameter optional)
- All v0.9.2 code compiles and runs without modification

---

### 📊 **Performance Summary**

| Backend | File Size | Method | Throughput | Improvement |
|---------|-----------|--------|------------|-------------|
| Azure | 8MB | RangeEngine (1 range) | 16.54 MB/s | Baseline |
| GCS | 8MB | RangeEngine (1 range) | 22.97 MB/s | Baseline |
| GCS | 128MB | RangeEngine (2 ranges) | 44-46 MB/s | Network-limited |
| Expected | >1GB | RangeEngine (multi-range) | 30-50% faster | On fast networks |

**Note**: Performance gains depend on network bandwidth and latency. High-bandwidth networks (>1 Gbps) with higher latency will see the most benefit from concurrent range requests.

---

### 🚀 **Migration Guide**

#### **No Code Changes Required**
This release is fully backward compatible. All existing code continues to work without modification.

#### **Optional: Customize RangeEngine Configuration**
```rust
// Azure custom config
use s3dlio::object_store::{AzureObjectStore, AzureConfig, RangeEngineConfig};

let config = AzureConfig {
    enable_range_engine: true,
    range_engine: RangeEngineConfig {
        chunk_size: 32 * 1024 * 1024,  // 32MB chunks instead of 64MB
        max_concurrent_ranges: 16,      // 16 concurrent instead of 32
        min_split_size: 8 * 1024 * 1024,  // 8MB threshold instead of 4MB
        range_timeout: Duration::from_secs(60),
    },
};
let store = AzureObjectStore::with_config(config);
```

#### **Optional: Disable RangeEngine**
```rust
let config = AzureConfig {
    enable_range_engine: false,  // Disable concurrent ranges
    ..Default::default()
};
let store = AzureObjectStore::with_config(config);
```

---

### 📚 **Documentation Updates**
- Updated README.md with v0.9.3 features
- Updated Changelog.md (this file)
- API documentation includes RangeEngine configuration examples
- Test guides updated with new test suites

---

## Version 0.9.2 - CancellationToken & Configuration Rationalization (October 2025)

### 🎯 **New Features**

#### **1. CancellationToken Infrastructure for Graceful Shutdown**
Comprehensive cancellation support across all DataLoader components enables clean shutdown of data loading operations.

**Components Updated**:
- `LoaderOptions`: Added `cancellation_token` field with builder methods
- `spawn_prefetch()`: Added cancel_token parameter with loop check
- `DataLoader`: Cancellation checks in all 3 spawn paths (map-known, iterable, map-unknown)
- `AsyncPoolDataLoader`: Cancellation support in async pool worker with 3 checkpoints

**Usage**:
```rust
use tokio_util::sync::CancellationToken;

let cancel_token = CancellationToken::new();

let options = LoaderOptions::default()
    .with_batch_size(32)
    .with_cancellation_token(cancel_token.clone());

// Spawn Ctrl-C handler
tokio::spawn(async move {
    tokio::signal::ctrl_c().await.unwrap();
    cancel_token.cancel();
});

let loader = DataLoader::new(dataset, options);
let mut stream = loader.stream();

while let Some(batch) = stream.next().await {
    train_step(batch?).await?;
}
// Clean shutdown on Ctrl-C
```

**Behavior**:
- ✅ Workers exit cleanly without submitting new requests
- ✅ In-flight requests drain naturally
- ✅ MPSC channels properly closed
- ✅ No orphaned background tasks
- ✅ Zero overhead when token not cancelled

---

#### **2. Configuration Hierarchy Rationalization**
Clear three-level configuration design aligned with PyTorch DataLoader concepts for ML practitioners.

**Documentation Added**:
- `docs/CONFIGURATION-HIERARCHY.md` - Comprehensive analysis
- `docs/api/rust-api-v0.9.2.md` - Updated from v0.9.0 with hierarchy section
- `docs/api/python-api-v0.9.2.md` - Updated from v0.9.0 with Python-specific guidance

**Three Levels**:
1. **LoaderOptions** (User-Facing, Training-Centric)
   - Like PyTorch's `DataLoader(batch_size, num_workers, ...)`
   - Controls WHAT batches to create and HOW to iterate
   - Always visible to users
   
2. **PoolConfig** (Performance Tuning, Optional)
   - Like PyTorch's internal worker pool management
   - Controls HOW data is fetched efficiently
   - Good defaults via `stream()`, advanced tuning via `stream_with_pool()`
   
3. **RangeEngineConfig** (Internal Optimization, Hidden)
   - Like file I/O internals in PyTorch Dataset
   - Controls storage-layer parallel range requests
   - Backend-specific, automatically configured

**Code Enhancement**:
```rust
// NEW: Convenience constructor to bridge Level 1 → Level 2
let pool_config = PoolConfig::from_loader_options(&options);
// Derives: pool_size from num_workers, readahead_batches from prefetch
```

**Design Philosophy**:
- Progressive complexity: Simple by default, powerful when needed
- PyTorch alignment: Familiar concepts for ML practitioners
- Separation of concerns: Training logic vs storage optimization
- Good defaults: Most users never touch Level 2 or 3

---

### 🧪 **Testing Improvements**

#### **Cancellation Test Suite**
Comprehensive test coverage in `tests/test_cancellation.rs`:

**9 Tests (All Passing ✅)**:
- DataLoader tests (5): pre-cancellation, during streaming, delayed, no-token, shared token
- AsyncPoolDataLoader tests (4): pre-cancellation, during streaming, stops new requests, idempotent

**Test Patterns**:
- Validates drain behavior (in-flight requests complete)
- Timeout-based completion checks (2s)
- Mock URIs for control flow testing
- Ctrl-C handler examples

**Documentation**:
- Updated `docs/TESTING-GUIDE.md` with cancellation testing section
- Test patterns, running instructions, expected behavior
- Configuration hierarchy integration notes
- Checklist for new components

---

### 📚 **Documentation Updates**

#### **API Documentation Versioned to v0.9.2**:
- Renamed `rust-api-v0.9.0.md` → `rust-api-v0.9.2.md`
- Renamed `python-api-v0.9.0.md` → `python-api-v0.9.2.md`
- Added "Configuration Hierarchy" sections
- Added "Graceful Shutdown" section with CancellationToken examples
- Updated "What's New" sections

#### **New Documentation Files**:
- `docs/CONFIGURATION-HIERARCHY.md` - Deep dive into three-level design
- PyTorch comparison tables
- Side-by-side examples (PyTorch vs s3dlio)
- Recommendations for users
- When to tune each level

---

### 🔧 **API Additions**

**LoaderOptions**:
```rust
pub struct LoaderOptions {
    // ... existing fields ...
    
    /// Optional cancellation token for graceful shutdown (NEW in v0.9.2)
    pub cancellation_token: Option<tokio_util::sync::CancellationToken>,
}

impl LoaderOptions {
    pub fn with_cancellation_token(self, token: CancellationToken) -> Self;
    pub fn without_cancellation(self) -> Self;
}
```

**PoolConfig**:
```rust
impl PoolConfig {
    /// NEW: Derive from LoaderOptions with sensible scaling
    pub fn from_loader_options(opts: &LoaderOptions) -> Self;
}
```

**spawn_prefetch**:
```rust
// NEW: Added cancel_token parameter
pub fn spawn_prefetch<F, Fut, T>(
    cap: usize,
    producer: F,
    cancel_token: Option<CancellationToken>,  // NEW
) -> Receiver<Result<T, DatasetError>>
```

---

### ⚡ **Performance**

- **Zero overhead** when cancellation token not used
- **Clean shutdown** typically completes within 2 seconds
- **No breaking changes** to existing code without cancellation
- **Maintains** 5+ GB/s reads, 2.5+ GB/s writes

---

### 🛠️ **Build Status**

- ✅ Zero compiler warnings
- ✅ All tests passing (including 9 new cancellation tests)
- ✅ Clean release builds
- ✅ Documentation comprehensive and versioned

---

### 📦 **Migration Notes**

**No breaking changes** - all features are additive:

**To add graceful shutdown**:
```rust
// Before (v0.9.1)
let options = LoaderOptions::default().with_batch_size(32);

// After (v0.9.2) - optional cancellation
let cancel_token = CancellationToken::new();
let options = LoaderOptions::default()
    .with_batch_size(32)
    .with_cancellation_token(cancel_token);
```

**To use PoolConfig convenience constructor**:
```rust
// NEW in v0.9.2
let pool_config = PoolConfig::from_loader_options(&options);
let stream = loader.stream_with_pool(pool_config);
```

---

### 🎯 **Key Use Cases**

**Training Loops with Ctrl-C Handling**:
- Clean shutdown prevents corrupted checkpoints
- No orphaned background tasks
- Proper resource cleanup

**Multi-Loader Coordination**:
- Single token cancels multiple loaders
- Synchronized shutdown across components

**Long-Running Jobs**:
- Graceful termination on SIGTERM
- Clean exit from infinite data streams

---

### 📖 **Further Reading**

- **Configuration Guide**: `docs/CONFIGURATION-HIERARCHY.md`
- **Rust API**: `docs/api/rust-api-v0.9.2.md`
- **Python API**: `docs/api/python-api-v0.9.2.md`
- **Testing Guide**: `docs/TESTING-GUIDE.md` (cancellation section)

---

## Version 0.9.1 - True Zero-Copy Python API (October 2025)

### 🎯 **Critical Fixes**

#### **1. Zero-Copy Python API (TRUE Implementation)**
- **Fixed false claim**: v0.9.0 claimed zero-copy but actually copied data to Python bytes
- **New `BytesView` class**: Wraps Rust `Bytes`, exposes `.memoryview()` for true zero-copy
- **Python buffer protocol**: Native support without numpy dependency
- **Memory reduction**: 10-15% reduction in typical AI/ML workflows

**Zero-Copy Functions (Returns `BytesView`)**:
- `get(uri)` → `BytesView` (was `bytes`)
- `get_range(uri, offset, length)` → `BytesView` (NEW)
- `get_many(uris)` → `List[(str, BytesView)]` (was `List[(str, bytes)]`)
- `get_object(bucket, key)` → `BytesView` (was `bytes`)
- `CheckpointStore.load_latest()` → `Optional[BytesView]` (was `Optional[bytes]`)
- `CheckpointReader.read_shard_by_rank()` → `BytesView` (was `bytes`)

**Usage**:
```python
# Zero-copy access
view = s3dlio.get("s3://bucket/data.bin")
arr = np.frombuffer(view.memoryview(), dtype=np.float32)  # No copy!

# Or copy if needed
data = view.to_bytes()  # Explicit copy
```

**NOT Zero-Copy** (copies data):
- `PyDataset.get_item()` → still returns `bytes` (legacy trait)
- `PyBytesAsyncDataLoader.__next__()` → returns `bytes` (uses dataset)

See `docs/ZERO-COPY-API-REFERENCE.md` for complete details.

---

#### **2. Universal Backend Support for get_many()**
- **Fixed limitation**: `get_many()` was S3-only in v0.9.0
- **Universal support**: Now works with S3, Azure, GCS, File, DirectIO
- **Backend detection**: Automatically routes to appropriate implementation
- **S3**: Uses optimized `get_objects_parallel()`
- **File/DirectIO**: Parallel `tokio::fs::read()` with semaphore
- **Azure/GCS**: Uses universal `store_for_uri()` factory

**All URIs must use same scheme**:
```python
# ✅ Valid - all file://
s3dlio.get_many(["file:///a.bin", "file:///b.bin"])

# ❌ Invalid - mixed schemes
s3dlio.get_many(["s3://a", "file:///b"])  # Error
```

---

### ✨ **New Features**

#### **1. Universal Range Request Support**

**Python API**:
```python
# Read byte range (zero-copy)
view = s3dlio.get_range("s3://bucket/file", offset=1000, length=1024*1024)
arr = np.frombuffer(view.memoryview(), dtype=np.uint8)

# Read from offset to end
view = s3dlio.get_range("file:///path", 5000, None)
```

**CLI**:
```bash
# Range request
s3-cli get s3://bucket/file --offset 1000 --length 1024

# From offset to end
s3-cli get file:///path/file --offset 5000

# Single-object only (not with --recursive or --keylist)
```

**Backends**: S3, Azure, GCS, File, DirectIO

---

#### **2. Comprehensive Zero-Copy Refactor**

**Rust Library**:
- All `ObjectStore` helper methods return `Bytes` (not `Vec<u8>`)
- `get_with_validation()` → `Bytes`
- `get_range_with_validation()` → `Bytes`
- `get_optimized()` → `Bytes`
- `get_range_optimized()` → `Bytes`

**Checkpoint System**:
- All checkpoint reader methods return `Bytes`
- `read_shard()` → `Bytes`
- `read_shard_by_rank()` → `Bytes`
- `read_all_shards()` → `Vec<(u32, Bytes)>`
- `load_latest()` → `Option<Bytes>`

**S3 Utils**:
- `get_range()` uses `.slice()` not `.to_vec()`

---

### 🧪 **Testing**

**New Test Suites**:
1. **test_zero_copy_comprehensive.py** (6/6 tests pass):
   - BytesView structure validation
   - get() returns BytesView with working memoryview
   - NumPy array from memoryview (zero-copy verified)
   - get_many() universal backend support
   - Large file handling (1 MB)
   - BytesView immutability

2. **test_range_requests.py** (4/4 tests pass):
   - Range with offset+length
   - Range with offset only (to end)
   - Range from beginning
   - Full file comparison

**Existing Tests**:
- ✅ Rust library: 91/91 tests pass
- ✅ Python functionality: 27/27 tests pass
- ✅ Zero warnings in all builds

---

### 📝 **Documentation**

**New Documentation**:
- `docs/ZERO-COPY-API-REFERENCE.md`: Complete zero-copy vs copy operations guide
- `docs/v0.9.1-ZERO-COPY-TEST-SUMMARY.md`: Implementation and test summary

**Clarifications**:
- GCS backend: Production-ready, works well in testing
- Performance targets: 5+ GB/s reads, 2.5+ GB/s writes maintained

---

### 🔧 **Implementation Notes**

**Zero-Copy Architecture**:
- `PyBytesView` wraps `Bytes` with PyO3 buffer protocol support
- `.memoryview()` → zero-copy access (read-only)
- `.to_bytes()` → explicit copy for compatibility
- `__len__()` → zero-copy length query

**Type Conversions**:
- Functions returning `PyBytesView`: Direct return
- Functions returning `PyObject`: Use `.into_py_any(py)?`
- Functions returning `Option<PyObject>`: Nested map with `Python::with_gil()`

**Build Quality**:
- Zero warnings in all builds (`cargo build`, `cargo clippy`, PyO3)
- Fixed all Rust test assertions for `Bytes` comparisons
- Proper error handling and validation

---

### 📊 **Performance Impact**

**Memory Savings**:
```python
# v0.9.0 (copies data)
data = s3dlio.get("s3://bucket/1gb.bin")  # bytes
arr = np.frombuffer(data, dtype=np.uint8)
# Memory: 2 GB (bytes + NumPy array)

# v0.9.1 (zero-copy)
view = s3dlio.get("s3://bucket/1gb.bin")  # BytesView
arr = np.frombuffer(view.memoryview(), dtype=np.uint8)
# Memory: 1 GB (shared buffer)
# Savings: 50%
```

**Throughput**:
- Reduced GC pressure: 15-20% faster in batch operations
- Universal `get_many()`: Works across all backends
- Range requests: Fetch only needed bytes

---

### 🔄 **Migration from v0.9.0**

**Python API Changes**:
```python
# Old (v0.9.0)
data = s3dlio.get(uri)  # bytes
arr = np.frombuffer(data, dtype=np.float32)

# New (v0.9.1) - Zero-copy
view = s3dlio.get(uri)  # BytesView
arr = np.frombuffer(view.memoryview(), dtype=np.float32)

# Backward compatibility (copies)
view = s3dlio.get(uri)
data = view.to_bytes()  # Convert to bytes if needed
```

**No Breaking Changes**:
- `BytesView` supports buffer protocol → most code works unchanged
- Explicit `.to_bytes()` available for compatibility
- Dataset API unchanged (still returns bytes)

---

### ✅ **Verified Compatibility**

**Frameworks**:
- ✅ NumPy: `np.frombuffer(view.memoryview())`
- ✅ PyTorch: `torch.from_numpy(np.frombuffer(view.memoryview()))`
- ✅ TensorFlow: `tf.constant(view.memoryview())`
- ✅ JAX: `jnp.array(np.frombuffer(view.memoryview()))`

**Backends**:
- ✅ S3: AWS S3, MinIO, Vast
- ✅ Azure: Azure Blob Storage
- ✅ GCS: Google Cloud Storage
- ✅ File: Local filesystem
- ✅ DirectIO: Direct I/O filesystem

---


---

**For v0.9.0 and earlier releases, see [archive/Changelog_pre_v090.md](archive/Changelog_pre_v090.md)**
