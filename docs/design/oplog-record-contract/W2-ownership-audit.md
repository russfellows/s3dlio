# W2: who records which call (ownership audit, round 2)

Wave W2 of the op-log work plan: each call is recorded exactly once (s3dlio #169). This appendix lists
every entry point that was recorded by the S3 backend's own ("inner") row writer, what happens to it now,
and the entry points that were never recorded. Labels: **[READ]** read in code on this branch, **[TEST]**
pinned by an automated test named in the last column.

Round 2 supersedes round 1. Round 1 also changed the SERIALIZATION of the helper rows; that was not
approved and is reverted (section 6 lists what changed back). The rule of this round: **no field of any
row changes; the only change is that a call no longer writes two rows.**

## 1. The ownership model

Before W2 the inner writer (`S3Ops::log_op`) fired on exactly three S3 helpers: `get_object`,
`put_object_async` and the synchronous `delete_objects`. Everything that reached one of them wrote an inner
row: a logged store wrote it in addition to its own row (the #169 duplicate), an unlogged store wrote it
instead of nothing (logged by accident, S3 only).

Each call now has one owner:

- **store**: a store built with a logger (`store_for_uri_with_logger(uri, Some(..))`) records through
  `LoggedObjectStore`, on every backend. Row shape: endpoint `s3://<bucket>`, file `<bucket>/<key>`,
  `client_id` from `set_client_id()`.
- **helper**: a public S3 helper function that is a user entry point records through `S3Ops`, with the
  global logger, in the shape `S3Ops` always wrote (section 1.1). Unchanged from before W2.
- **explicit**: a library entry point that built an unlogged store (`store_for_uri`) and relied on the S3
  backend's own row (checkpoint, `Reader::read_all_shards_concurrent*`, `MultiBackendDataset`, Python
  `load_checkpoint_with_validation`) uses `store_for_uri_with_helper_rows`: a bucket-form S3 URI gets
  `with_helper_rows()` (the helper row, as before), every other URI the plain unlogged store (no row, as before).
- **silent**: the S3 backend of a store builds its `S3Ops` with no logger (`s3_utils::Rows::Skip`), so a
  store's own recording is the only one. A store built without a logger records nothing.
- **MultiEndpointStore, bucket-form endpoint**: its endpoint store is built with `with_helper_rows()`, so its
  `get`/`put` write the helper row exactly as before W2 (it is never wrapped in a `LoggedObjectStore`).

### 1.1 The helper row (unchanged)

| column | value (success) | value (failure) |
|---|---|---|
| `op` | `GET`, `PUT`, or `DELETE` (batch) | same |
| `client_id` | `AWS_ACCESS_KEY_ID` (NOT `set_client_id()`) | same |
| `n_objects` | 1 (batch: keys in that chunk) | same |
| `bytes` | GET: body length. PUT: payload length. | **0** |
| `endpoint` | `AWS_ENDPOINT_URL` (the server contacted; empty if unset) | same |
| `file` | the bare key (batch: `(batch)`) | same |
| `error` | empty | GET: `{:#}` of the error chain. PUT: `format_sdk_error` detail. Batch DELETE: the per-key failure list or the SDK detail |
| `first_byte` | GET: time the response headers arrived. PUT, DELETE: empty | empty |

**[TEST]** `oplog_helper_rows_golden` compares 25 helper rows (success, 403 failures, a retried 500 PUT,
`delete_objects` of 0/1/1000/1001 keys and a per-key failure, `get_objects_parallel`, `get_range`,
`S3BytesDataset`, `s3_copy` upload/download) with a fixture CAPTURED FROM THE PARENT (9473d2c) by running the
same test there: byte-identical. The mock server's request log (method, path, host:port, retries) is checked
alongside.

## 2. Entry points that were recorded by the inner writer

| Entry point | Rows before | Decision | Rows now | Test |
|---|---|---|---|---|
| Logged store, S3 `get`, `put` (`s3://bucket/..`) | 2 (outer + inner) | store | 1 (store shape) | `oplog_exactly_once_logged_store_s3` |
| Logged store, S3 `put_multipart`, `get_range`, `stat`, `list`, `delete`, `delete_batch`, `delete_prefix` | 1 (outer; never reached the inner writer) | store | 1 (unchanged) | same |
| Logged store, S3 host-form `s3://host:port/..` | 1 (per-endpoint client bypassed the inner writer) | store | 1 (unchanged) | [READ] `object_store.rs` per-endpoint `client` branches |
| Unlogged store (`store_for_uri`, `store_for_uri_with_logger(.., None)`), S3 `get`, `put` | 1 inner row | silent | 0 | `oplog_exactly_once_unlogged_store_s3` |
| `s3_utils::get_object`, `get_object_uri_async`, `get_object_uri` (sync) | 1 helper row | helper | 1, **unchanged shape** | `oplog_helper_rows_golden` |
| `s3_utils::get_object_uri_optimized_async` | 1 helper row when served by a single GET; 0 for a range-split large object | helper | same: 1 / 0, **unchanged** | `oplog_helper_rows_golden`, `..._large_object` |
| `s3_utils::get_objects_parallel*` (Python `get_many` for `s3://`, `get_many_stats`, `get_many_async`) | 1 per small object, 0 per large object | helper | same, **unchanged** | same |
| `s3_utils::get_range` (downloads the whole object, then slices) | 1 helper `GET` | helper | 1, **unchanged** | `oplog_helper_rows_golden` |
| `s3_utils::put_object_async`, `put_object_uri_async` | 1 helper row | helper | 1, **unchanged** | `oplog_helper_rows_golden` |
| `s3_utils::delete_objects` (sync) | 1 `DELETE` row per 1000 keys, `file` = `(batch)`; none for an empty list | helper | same, **unchanged** | `oplog_helper_rows_golden` (0, 1, 1000, 1001 keys) |
| `s3_copy::upload_files`, `s3_copy::download_objects` | 1 per object | helper | same, **unchanged** | `oplog_helper_rows_golden` |
| `S3BytesDataset` sequential mode | 1 per object | helper | same, **unchanged** | `oplog_helper_rows_golden` |
| Checkpoint `CheckpointStore::open*`, `Reader::read_all_shards_concurrent*` | bucket-form S3: helper rows via the store's S3 backend; other backends: none | explicit (unlogged store; bucket-form S3 gets `with_helper_rows`) | **unchanged**: bucket-form S3 helper rows, no row elsewhere | `oplog_explicit_callers_golden` |
| `MultiBackendDataset::from_prefix`, `from_uris` | same | same | **unchanged** | same |
| Python `load_checkpoint_with_validation` | S3: 1 helper row | same | **unchanged** | [READ] (same `store_for_uri_with_helper_rows`) |
| `MultiEndpointStore` `get`/`put`, bucket-form endpoints | 1 helper row | helper (via `with_helper_rows`) | 1, **unchanged** | `oplog_multi_endpoint_rows` (also passes on the parent) |

`Reader::read_all_shards_concurrent*` looks up its own store from the base URI, as it did before W2 (the borrowed
`&dyn ObjectStore` is not `'static`, so it cannot be moved into the spawned tasks); it does not substitute a
logged store.

## 3. Rows that are gone (reported, not fixed here)

| Entry point | Rows before | Rows now | Why |
|---|---|---|---|
| Unlogged store, S3 `get`/`put` | 1 helper row | 0 | ruling D4/O1: a store built without a logger is silent |
| `s3dlio-oplog` replayer (`crates/s3dlio-oplog/src/replayer.rs`, `store_for_uri`) | `GET`/`PUT`: 1 helper row if a global logger is active in the replaying process | 0 | consumer tool; capturing a replay is sai3-bench#103 |

## 4. Entry points that were never recorded by the inner writer (unchanged)

| Entry point | Today | Issue / note |
|---|---|---|
| `get_object_range_uri_async`, `get_object_range_uri_timed_async`, `get_object_concurrent_range_async` | no row from the helper (a logged store records `GET_RANGE`) | not an inner-writer path |
| Large-object (range-split) reads through `get_object_uri_optimized_async` / `get_objects_parallel*` | no row (round 1 added one in a new shape; reverted) | proposal P-large in section 7 |
| `stat_object`, `stat_object_uri_async`, `stat_object_many_async`, `list_objects*`, `list_objects_stream`, `delete_objects_async`, `put_object_multipart_uri_async`, `create_bucket*`, `delete_bucket*` | no row from the helper | not an inner-writer path |
| Python `create_bucket`, `delete_bucket` | no row | #182 (Python bucket calls) |
| Python `get_many` for `file://`, `direct://`, `az://`, `gs://` | no row | built without a logger |
| Python `create_s3_writer` and the other writers | no row | writer logging is a separate gap |
| Parquet readers (`parquet_rg`, `parquet_file_cache`), `parquet_gen` | no row | built with `store_for_uri` (silent) |
| `S3BytesDataset` range mode | no row | uses `get_object_range_uri_async` |
| `MultiEndpointStore` `list`, `delete`, `get_range`, `stat`, host-form endpoints | no row | proposal P-MEP in section 7 |

## 5. Rows and requests per logical operation

R = op-log rows, Q = requests the S3 server (the in-process mock) received. SDK retries add requests, never
rows (`oplog_helper_rows_golden`: a 500 PUT is retried by the SDK, one row; `oplog_rows_copy_rename_prestat`:
same through a store). A read of an object whose size is not known first sends a HEAD size probe before its
GET; that probe has never had a row of its own and is not counted below.

| logical operation | backend | R | Q | test |
|---|---|---|---|---|
| logged store `copy` | S3 | `GET`, `PUT` | GET src, PUT dst | `oplog_rows_copy_rename_prestat` |
| logged store `copy`, PUT fails | S3 | `GET`, `PUT`(error, bytes = payload) | GET, PUT x attempts (>= 2) | same |
| logged store `rename` | S3 (and Azure, GCS, multi-endpoint) | `GET`, `PUT`, `DELETE` | GET src, PUT dst, DELETE src | same |
| logged store `pre_stat_objects` | S3 | one `HEAD` per URI (error row for a missing one) | one HEAD per URI | same |
| logged store `pre_stat_and_cache` | S3 | one `HEAD` per URI | one HEAD per URI; the size cache is filled with NO further request | same |
| logged store `copy` | file:// | `GET`, `PUT` | read, write | same |
| logged store `rename` | file://, direct:// | none (`rename(2)` is native; no op name describes it) | one `rename(2)` | same, `oplog_logged_store_file_transparent` |
| logged store `pre_stat_objects`, `pre_stat_and_cache` | file:// | one `HEAD` per URI | one `stat` per URI | `oplog_rows_copy_rename_prestat` |
| logged store `mkdir`, `rmdir`, `update_metadata`, `update_properties` | all | none (no op name; forwarded) | the backend's own | `oplog_logged_store_forwards_every_method` |
| helper `put_object_async`, `put_object_uri_async` | S3 | 1 `PUT` | 1 PUT | `oplog_helper_rows_golden` |
| helper `get_object*`, `get_range`, `get_objects_parallel*` (small object) | S3 | 1 `GET` per object | 1 GET (+ size probe) per object | same |
| helper `get_object_uri_optimized_async`, `get_objects_parallel*` (large object) | S3 | none | ranged GETs | `..._large_object` |
| helper `delete_objects`, 0 keys | S3 | none | none | `oplog_helper_rows_golden` |
| helper `delete_objects`, 1 / 1000 / 1001 keys | S3 | 1 / 1 / 2 `DELETE (batch)` rows (n = 1 / 1000 / 1000 + 1) | 1 / 1 / 2 DeleteObjects POSTs | same |
| helper host-form `get_many` | S3 | 1 `GET`, endpoint = `AWS_ENDPOINT_URL` (the server contacted) | 1 GET to `AWS_ENDPOINT_URL` | `oplog_helper_rows_host_form` |
| `MultiEndpointStore.get`/`put`, bucket-form | S3 | 1 helper row each | 1 GET (+ size probe) / 1 PUT | `oplog_multi_endpoint_rows` |
| `MultiEndpointStore.rename`, bucket-form | S3 | `GET`, `PUT` (the DELETE is silent) | GET, PUT, DELETE | same |
| `MultiEndpointStore` `list`, `delete`, `get_range`, `stat`; any host-form operation | S3 | none | 1 request each | same |
| Python store cache (#174) | all | a call is recorded exactly when the op-log is active at the call | n/a | `tests/test_oplog_store_cache.py`, `store_cache_logger_tests` |

The op-log never records its own I/O: no row names the log file (`oplog_rows_copy_rename_prestat`,
`oplog_lifecycle_no_self_log_*`).

### 5.1 What `bytes` and `error` mean on a failed PUT (kept as they were, per path)

Read from the code history (`S3Ops` since 0e07a59 and #506, `LoggedObjectStore` since 263e32c) and
`docs/supplemental/OPERATION_LOGGING.md` ("`bytes`: data transferred"; no document says what a failed PUT
transferred). The two paths differ and each is kept as it was; they are not aligned.

- **Helper path (`S3Ops`)**: `bytes` is what was transferred and stored: the payload length only on success
  and **0 on failure** (the first `S3Ops` already passed `0` for a failed GET; #506 added the failure rows of
  the other operations, also with `0`). `error` is the same detail the caller's error carries (#506 made every
  failure arm log `format_sdk_error(&e)` "so the s3dlio op-log file matches the caller's view"):
  `service error: AccessDenied (...) — check AWS_ACCESS_KEY_ID ...`. Pinned by `oplog_helper_rows_golden`
  (three failing PUTs, one of them retried).
- **Store path (`LoggedObjectStore`)**: `bytes` is the size of the payload handed to `put()`, captured before
  the call and written whether the call succeeds or not (**the payload size on failure**); the PUT sample in
  OPERATION_LOGGING.md is exactly this code. `error` is `e.to_string()` of the error the store returned (the
  S3 store's error already carries the SDK chain and the operation context). Pinned by
  `oplog_rows_copy_rename_prestat` (failed PUT, bytes 4).

## 6. What changed back from round 1

| item | round 1 | round 2 |
|---|---|---|
| `S3Ops` | writer removed, `S3Ops::new(client)` | restored byte-identical to 9473d2c |
| helper `client_id` | `set_client_id()` | `AWS_ACCESS_KEY_ID` |
| helper `endpoint` / `file` | `s3://<bucket>` / `<bucket>/<key>` | `AWS_ENDPOINT_URL` / bare key |
| helper `first_byte` | GET = end, PUT = start | GET = measured header time, PUT = empty |
| helper failure `error` | store-style text | S3Ops text (GET `{:#}`, PUT/DELETE `format_sdk_error`) |
| `delete_objects` | one `DELETE_BATCH` row for the call, `file` = first URI | one `DELETE (batch)` row per 1000 keys |
| large-object helper read | 1 new row | 0 rows (as before) |
| host-form `get_many` endpoint column | `s3://<bucket>` (named no server) | `AWS_ENDPOINT_URL` (the server contacted) |
| `MultiEndpointStore` get/put | 0 rows | 1 row each (as before) |
| `LoggedObjectStore` copy / rename / pre_stat | forwarded, unrecorded (0 rows for real requests) | defaults recorded, native forwarded (section 5) |
| checkpoint / shard reader / `MultiBackendDataset` / Python `load_checkpoint_with_validation` | store shape on every backend (S3 helper shape lost, rows added on file/direct/az/gs) | pre-W2 exactly: helper rows on bucket-form S3, no row elsewhere |

Unavoidable serialization changes: **none were made.** One behaviour changes that is not serialization but
should be seen: a store built without a logger writes nothing on S3 (D4/O1, round 1, kept). The explicit
callers (checkpoint, shard reader, `MultiBackendDataset`, Python `load_checkpoint_with_validation`) write
exactly their pre-W2 rows again (section 2).

## 7. Proposals (NOT implemented; the architect approves or not)

### P-MEP: rows for the other `MultiEndpointStore` operations (#175)

Today `list`, `delete`, `get_range`, `stat`, and every operation on a host-form endpoint write no row.
Proposal: record them in the STORE shape (the shape `LoggedObjectStore` writes; implemented by wrapping each
endpoint store in a `LoggedObjectStore` when a logger is active, which would also move the bucket-form
`get`/`put` rows to the store shape: a serialization change, hence a proposal). Example rows (13 columns;
`c1` = `set_client_id()`; endpoint store `s3://bkt/pfx/`; `T0..T1` timestamps; `d` = duration; empty
columns are written empty, shown as `-`):

```
op          client_id n_objects bytes endpoint             file                       error start first_byte end duration_ns
Before (every row below):  no row
LIST        c1        2         0     s3://bkt             bkt/pfx/                   -     T0    -          T1  d      list("s3://bkt/pfx/")
GET_RANGE   c1        1         2     s3://bkt             bkt/pfx/k                  -     T0    T1         T1  d      get_range(.., 0, Some(2))
HEAD        c1        1         3     s3://bkt             bkt/pfx/k                  -     T0    -          T1  d      stat(..)
DELETE      c1        1         0     s3://bkt             bkt/pfx/k                  -     T0    -          T1  d      delete(..)
GET         c1        1         4     s3://127.0.0.1:9000  127.0.0.1:9000/bkt/pfx/k   -     T0    T1         T1  d      host-form get(..)
PUT         c1        1         4     s3://127.0.0.1:9000  127.0.0.1:9000/bkt/pfx/k   -     T0    T0         T1  d      host-form put(..)
```

(`idx` and `thread` columns omitted; the real rows are tab-separated.) Consumers: the `s3dlio-oplog` replayer
(`crates/s3dlio-oplog`) reads `op` in {GET, PUT, DELETE, LIST, STAT/HEAD} and rejects any other name as an
unknown operation, so a `GET_RANGE` row would make a replay fail; it rebuilds the URI as `endpoint + file`;
`tests/oplog_accuracy` judges every row against what happened; sai3-bench and warp-replay read the TSV.
Decision needed: store shape or helper shape for these rows, and whether `GET_RANGE` rows are acceptable to the
replayer.

### P-explicit: rows for the checkpoint / dataset callers on file, direct, Azure and GCS

Before W2 `CheckpointStore`, `Reader::read_all_shards_concurrent*`, `MultiBackendDataset` and Python
`load_checkpoint_with_validation` wrote NO row on any backend except bucket-form S3 (where they write the helper
row, restored). Proposal: record them on every backend through a logged store (round 1 did this). That is new
serialization for previously unrecorded calls, so it is not done. Example rows (store shape; `c1` =
`set_client_id()`; `idx`/`thread` and timestamps omitted; `-` = empty column), for a `file://` checkpoint
save + load of `/ckpt`:

```
op   client_id n_objects bytes endpoint file                                    error first_byte
Before (every row below):  no row
PUT  c1        1         16    file://  /ckpt/shards/ckpt-1-<TS>/rank-0.bin     -     start
PUT  c1        1         365   file://  /ckpt/manifests/ckpt-1-<TS>.json        -     start
PUT  c1        1         108   file://  /ckpt/latest.json                       -     start
GET  c1        1         108   file://  /ckpt/latest.json                       -     end
GET  c1        1         365   file://  /ckpt/manifests/ckpt-1-<TS>.json        -     end
GET  c1        1         16    file://  /ckpt/shards/ckpt-1-<TS>/rank-0.bin     -     end
```

(plus one `HEAD` after the shard PUT on file://). On S3 the same calls would change from the helper shape
(restored) to the store shape (`endpoint s3://bkt`, `file bkt/key`, `client_id c1`) if the proposal and the
P-MEP store-shape question are decided together. Consumers: the same as P-MEP (the replayer rejects no op here:
only GET, PUT and HEAD appear).

### P-large: a row for range-split large-object reads

A helper row in the existing shape needs a `first_byte` (the measured header time of the first response) and
a `bytes` (the assembled length). The ranged path has per-chunk header times only. Proposed row: `GET`,
`client_id` = `AWS_ACCESS_KEY_ID`, `n_objects` 1, `bytes` = total length, endpoint = `AWS_ENDPOINT_URL`,
file = bare key, `first_byte` = the earliest chunk's header time. Before: no row. Not implemented because the
choice of `first_byte` is a semantic decision.

## 8. Validation

Round-1 history: `evidence/w2/` (round-1 builds, GCS harness, kept as history, NOT evidence for round 2).
Round 2: see section 9.

## 9. Round-2 validation (ground-truth harness on Google Cloud Storage)

`tests/oplog_accuracy/` (see its README), one run on the final wheel (sha256 `d4802c6f...`, equal to the
`wheel_sha256` in the run record), GCS S3-interop endpoint, native `gcloud storage` as ground truth, 83 calls
(`delete_batch` skipped: no probe given). Evidence: `evidence/w2r2/` (bucket names replaced by the roles B1/B2).
The mock-server tests above prove the row FIELDS byte-for-byte against the parent; the harness proves the
rows against the real service.

| | integration baseline, round 1 not included (run 20261014-034453, main's known.json) | round 2 (run 20261014-045732) |
|---|---|---|
| calls | 83 | 83 |
| RECORD `DUPLICATE` | 18 | **0** |
| RECORD `MISSING` | 10 | 7 |
| RECORD `WRONG-FIELD` | 0 | **0** (round 1 had 7) |
| NEW / CHANGED | 0 / 0 | **0 / 0** |
| FIXED against main's `known.json` | 4 | 25 |

The 25 FIXED against main's `known.json`: 18 x #169 (every S3 `get`/`put_bytes`/`put_bytes_async`, one correct
row now), 3 x #174 (`async` `delete`, `host443_get_async`, `host443_decoydefault_get_async` have their row),
and 4 that this branch did NOT fix: 2 x #172 (`stat`/`get_range` of a leading-slash key) and 2 x #173
(`host443_get`, `host443_get_range`), which main already fixed (the integration run above shows the same 4
FIXED on main's wheel before this branch; main's `known.json` still lists them). `known.json` was updated
only for #169 and #174 (the DUPLICATE entry and the two #174 entries are removed); the #172/#173 entries are
left for main's owner. Re-evaluated against the updated file: PASS 70, KNOWN 9, NEW 0, CHANGED 0, FIXED 4
(exactly those four).

Compared with round 1: `multi_endpoint` `get`/`put` are OK again (were NEW MISSING), the two
`host443_decoydefault_get_many` rows are OK again (were `WRONG-FIELD:server`: the row names the server the
call really went to; the wrong routing itself is #197/#171, KNOWN in the harness), and `portless_get` /
`portless_put` are as on main (#178). The remaining 7 MISSING are the known #175 (`multi_endpoint`
`get_range`, `list`, `delete`, `me_host443` get), #178 (`portless_*`, the failed call writes no row).
Cleanup: 23 objects deleted, the run prefix is empty in both buckets, no process left running.
