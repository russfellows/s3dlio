# s3dlio op-log trace-accuracy harness

Question answered: **is every call the program made in the trace exactly once, with fields that match what
actually happened?**  Every call is judged twice, and the two verdicts are never merged:

| | meaning | verdicts |
|---|---|---|
| **(1) ACTION** | what s3dlio actually DID versus what the caller asked (server, scheme, bucket, key, object) | `OK`, `WRONG-ACTION:<what>`, `HANG`, `UNVERIFIED` |
| **(2) RECORD** | what the op-log row(s) SAY versus what actually happened, and how many rows the call produced | `OK`, `MISSING` (0 rows), `DUPLICATE` (>1 rows), `WRONG-FIELD:<field>`, `DUPLICATE+WRONG-FIELD:..`, `HANG`, `N/A` |

`<what>` / `<field>` is one or more of: `server`, `scheme`, `port`, `bucket`, `key`, `object`, `listing`, `range`,
`outcome` (call succeeded where it could not / failed where it should not), and for records also `op`, `bytes`,
`n_objects`, `error`.

A row is compared against what ACTUALLY happened, not against what was asked.  So a leading-slash `stat` that read the
wrong object is `WRONG-ACTION:key` (1) and, because its row still names the asked key, also `WRONG-FIELD:key` (2).
Fields a row does not carry are not errors but are listed as notes (`server not recorded` on `s3://bucket` rows,
`bucket not recorded` on legacy rows, `client_id carries a credential`).

## Target and prerequisites

The harness is designed and validated against **Google Cloud Storage's S3-compatible (interoperability) endpoint,
`https://storage.googleapis.com`**, and uses `gcloud storage` for the native ground truth.  It exercises the real
cloud service end to end, so it is a manual, credentialed, opt-in test; it is never run by `cargo test` or CI.

Needs: `uv`, `gcloud` (authenticated for the project that owns the two scratch buckets), `zstd`, a built s3dlio
Python wheel, and the environment below.

### Required environment

| variable | meaning |
|---|---|
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | HMAC (S3-interoperability) credentials for the buckets.  Read from the environment only; never written to disk, never printed. |
| `AWS_ENDPOINT_URL` | S3 endpoint, e.g. `https://storage.googleapis.com` |
| `OPLOG_HARNESS_BUCKET1`, `OPLOG_HARNESS_BUCKET2` | two DIFFERENT, pre-existing scratch buckets (called B1 and B2 below).  Nothing else is ever touched. |

Optional: `OPLOG_HARNESS_HOSTFORM_HOST` (hostname used by the host-form `s3://<host>:443/<bucket>/<key>` cases; it
must be a real TLS host that serves both buckets; default: the host of `AWS_ENDPOINT_URL`), `AWS_REGION`
(default `us-east-1`).  `run.sh` and `orchestrate.py` exit with status 2 and a clear message, before creating
anything, if a required variable is missing.

## Run

```
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...          # HMAC credentials
export AWS_ENDPOINT_URL=https://storage.googleapis.com
export OPLOG_HARNESS_BUCKET1=<scratch-bucket-1> OPLOG_HARNESS_BUCKET2=<scratch-bucket-2>

./run.sh                                   # newest target/wheels/*cp313*.whl, full suite, ~2 min, ~215 requests
./run.sh --wheel /path/to/s3dlio-X-cp313-cp313-manylinux_2_39_x86_64.whl   # any build (copied to ./wheels/)
./run.sh --python 3.12 --wheel ...cp312...whl                              # other interpreter / wheel tag
./run.sh --cases core,leading_slash        # subset (names from cases/manifest.json)
./run.sh --all-forms                       # run EVERY case with both op-log destination forms (more requests)
./run.sh --delbatch-bin /path/to/probe     # also run the delete_batch case (see below)
./run.sh --check-only runs/<id>            # re-evaluate saved evidence (e.g. after editing known.json); no network
```

`--wheel` defaults to the newest `target/wheels/*cp313*.whl` relative to the repository root and is an error if there
is none.  `--delbatch-bin` is optional: it is a small Rust program that takes `s3://` URIs as arguments, calls
`delete_batch` on them and prints the result (`Ok(..)` / `Err(..)`), with no logger.  Without it the `delete_batch`
case is not run and the report says `SKIPPED CASE delete_batch ... verdict SKIP` (not a failure).
Python is only ever started through `uv run --no-project --isolated --python 3.13 --with <wheel> python -I`.
Exit status: 0 = nothing unexpected, 1 = a NEW/CHANGED outcome or a case-level finding, 2 = usage/environment error.

Results land in `runs/<timestamp>/`: `report.txt` (the table, details, summary), `results.json`, `run.json`
(prefix, wheel sha256, request counts, cleanup proof), `requests.jsonl` (every gcloud request: method + URL),
`fixtures.json`, `snapshots.json` (native listings at start and end), and `cases/<case>__<form>/` with
`ground.json` (the program's call list + outcomes), `client.log` (s3dlio debug log), `ops.tsv.zst` (the op-log),
`native.json`.

## Evidence rules (what is trusted)

1. **Ground truth** = what each case asked for, written by the case itself (`ground.json`).
2. **Google-native tooling** (`gcloud storage cp/ls/objects list/objects describe`): objects are created natively with
   UNIQUE content and UNIQUE size, so returned bytes / `stat` size identify the object (and its bucket).  After every
   PUT/DELETE the case asks `gcloud storage objects describe` about the asked key and about named alternatives and
   decoys (same key in the other bucket, key without the leading slash).  `snapshots.json` holds the native listing
   before the cases (S0) and after them (FINAL); the checker audits stray objects and unasked deletions from them.
3. **s3dlio's own client debug log** (`init_logging("debug")`): `S3 client ready (… endpoint: …)`,
   `stat_object: bucket=…, key=…`, `S3 GET/PUT s3://bucket/key`, and the SDK's `will apply endpoint Endpoint { url: … }`
   (which server, scheme and bucket each HTTP request addressed).  Each call is bracketed by `@@H BEGIN/END` markers.
4. **Never a local or fast test server as ground truth.**  A local S3 emulator or a fast in-memory server has its own
   semantics (keys, buckets, redirects, host handling, error shapes) and would make the comparison meaningless; the
   native tooling must talk to the same real service that s3dlio talks to.  No benchmarking or replay tools either.

## Safety rules

* Only the two configured buckets B1 and B2, only under a fresh, unique `s3dlio-harness-<YYYYMMDD>-<random>/`
  prefix per run (plus the leading-slash namespace `/<prefix>/` in B1).  Use scratch buckets that hold nothing you
  care about.
* Credentials come only from the environment and are never written to any file.
* Cleanup is verified: everything created is removed in a `finally`, then the three namespaces are listed again.

Details:  Listings are always prefix-restricted.  No bucket is created or
deleted.  The orchestrator removes everything it created in a `finally`, then lists all three namespaces and records the
result in `run.json` (`cleanup.remaining`; the report prints it).  Each case process runs in its own session with a timeout and receives
SIGTERM only (never SIGKILL); a case whose process ignores two SIGTERMs is reported `stuck` and left alone.  The
op-log is finalized by the case itself (`finalize_op_log`) even when a call hangs: calls run in a daemon thread with a
call timeout, a hang is recorded as `HANG`, and the process exits by itself.

## Cases (`cases/manifest.json`, one script per scenario, one fresh process per case/form)

| case | what it does |
|---|---|
| `core` (path **and** `file://` op-log) | put_bytes, get, get_range(10,40), stat, list (non-recursive, recursive), delete, get/stat of a missing key (expected error), exists true/false |
| `repeat` | same object: get x3, stat x3 |
| `two_buckets` | identical key names in B1 and B2: get, stat, put, delete (a same-key decoy must survive) |
| `leading_slash` | `s3://b//<prefix>/slash/x` next to `s3://b/<prefix>/slash/x`: get, stat, get_range, put, delete (decoy: plain key) |
| `async` | put_bytes_async, get_async x2, stat_async, then a sync delete |
| `get_many` | one get_many of three objects (three logical calls in one window) |
| `local_files` (both forms) | `file://` put/get/get_range/stat/list/delete/missing in the SAME directory as the op-log; the op-log must not record itself |
| `s3_dest_reject` | `init_op_log("s3://...")` must raise, leave logging inactive, create nothing locally or natively; a corrected init then works |
| `multi_endpoint` | Python `create_multi_endpoint_store(["s3://<B1>/"])`: get, get_range, put, list, delete |
| `me_host443` | MultiEndpointStore over `s3://<hostform host>:443/...` |
| `delete_batch` | **Rust probe** (path via `--delbatch-bin`; skipped with a SKIP verdict when not given): batch across both buckets, same-key decoys in the other bucket; action checked natively; the probe has no logger, so the record verdict is `N/A` |
| `host443_{get,stat,get_range,get_async}` | host-form `s3://<hostform host>:443/...`; sync calls may hang (reported as `HANG`) |
| `host443_decoydefault_{stat,get_async,get_many}` | same URIs but `AWS_ENDPOINT_URL` points at an unroutable decoy, so a call routed to the configured endpoint instead of the URI's host fails visibly |
| `portless_{get,stat,put}` | portless `s3://10.255.255.1/bucket/key` (nothing answers there): a correct client fails; success = it went elsewhere |

Op-log destination forms: `core` and `local_files` run with both a plain path and a `file://` URI; the other cases
alternate between them (see `forms` in the manifest) so every run exercises both; `--all-forms` runs everything twice.

**Not covered:** `create_bucket` / `delete_bucket` (the harness never creates or deletes buckets); op-log rows of
`delete_batch` (needs a Rust probe with a logger); the offset/length of a range GET (no column for it, so it cannot be
verified from the trace).

## Verdict glossary

Every call gets two independent verdicts (see the table at the top):

* **ACTION** is about the *program's behaviour*: did s3dlio contact the right server, scheme, port, bucket and key, and
  return or change the right object?  `WRONG-ACTION:*` is a functional defect even if the trace looks perfect.
* **RECORD** is about the *trace*: does the op-log hold exactly one row for the call and do its fields say what
  actually happened?  `MISSING` / `DUPLICATE` / `WRONG-FIELD:*` are trace-accuracy defects even if the call itself worked.
* `HANG`: no return within the call timeout (the process still finishes by itself and finalizes the op-log).
* `N/A`: no verdict possible (e.g. the delete_batch probe has no logger).
* `SKIP`: the case was not run (e.g. no `--delbatch-bin`); not a failure.

Per-call status against `known.json`: `PASS`, `KNOWN`, `FIXED`, `CHANGED`, `NEW` (see below).

## known.json

Expected outcomes today.  A call with no entry is expected `OK`/`OK`.  Entries match on `case`, `call`, `api`, `form`
(globs or lists of globs, first match wins) and state the expected `action` and `record` signature plus the issue
numbers.  Per call the report then says:

* `PASS`    action OK and record OK;
* `KNOWN`   a listed defect reproduced exactly (expected failure);
* `FIXED`   a listed defect is gone;
* `CHANGED` a listed call fails differently than listed;
* `NEW`     a failure nobody listed (this makes the exit status 1).

In `known.json`, buckets are written as the roles `B1` / `B2` (they map to `OPLOG_HARNESS_BUCKET1/2`).
`known.json` also lists `process` exit anomalies (e.g. #192) and `audit_unasked_deletions` (decoys that `delete_batch`
removes), plus a `reported_but_not_reproduced_on_0.9.114` section for issue claims that did not show up on this wheel.
After a fix lands, the matching entries turn into `FIXED`; delete them to make `OK`/`OK` the expectation.

## Request budget

About 90 s3dlio requests (SDK attempts counted from the debug logs) + about 125 gcloud requests (setup ~48, 2 listings
S0/FINAL, ~24 `describe`, cleanup ~44) per full run; every gcloud request is logged in `requests.jsonl`.
