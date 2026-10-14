# W2: who records which call (ownership audit)

Wave W2 of the op-log work plan: each call is recorded exactly once (s3dlio #169, design doc
`docs/DESIGN_OPLOG_RECORD_CONTRACT.md` section 8). This appendix lists every entry point that was
recorded by the S3 backend's own ("inner") writer, what happens to it now, and the entry points that
were never recorded. Labels: **[READ]** read in code on this branch, **[TEST]** pinned by an
automated test named in the last column.

Before W2 the inner writer (`S3Ops::log_op`, reached only through `s3_utils::build_ops_async()`)
fired on exactly three S3 helpers: `get_object`, `put_object_async` and the synchronous
`delete_objects`. Everything that reached one of them wrote an inner row: a logged store wrote it in
addition to its own row (the #169 duplicate), an unlogged store wrote it instead of nothing (logged
by accident, S3 only). W2 removes the inner writer (`S3Ops` no longer takes a logger, client id or
endpoint) and gives each call one owner:

- **store**: a store built with a logger (`store_for_uri_with_logger(uri, Some(..))`) records through
  `LoggedObjectStore`, on every backend;
- **helper**: a public helper function that is a user entry point records through
  `s3_utils::HelperCall`, which calls the same row builder as the store
  (`object_store_logger::record_call`), so the row has the same shape: endpoint `s3://<bucket>`,
  file `<bucket>/<key>`, `client_id` from `set_client_id()`, `first_byte` GET = end, PUT = start;
- **explicit**: a library entry point that builds its own store asks for a logged one
  (`store_for_uri_with_logger(uri, global_logger())`);
- **silent**: an internal transfer, or a store built without a logger (ruling D4/O1), writes nothing.
  The S3 backend of every store calls the `*_silent` helpers, so a store's own recording is the only one.

## 1. Entry points that were recorded by the inner writer

| Entry point | Rows before | Decision | Rows now | Test |
|---|---|---|---|---|
| Logged store, S3 `get`, `put` (`s3://bucket/..`) | 2 (outer + inner) | store | 1 | `oplog_exactly_once_logged_store_s3` |
| Logged store, S3 `put_multipart`, `get_range`, `stat`, `list`, `delete`, `delete_batch`, `delete_prefix` | 1 (outer; these never reached the inner writer) | store | 1 (unchanged) | same |
| Logged store, S3 host-form `s3://host:port/..` | 1 (per-endpoint client bypassed the inner writer) | store | 1 (unchanged) | [READ] `object_store.rs` per-endpoint `client` branches |
| Unlogged store (`store_for_uri`, `store_for_uri_with_logger(.., None)`), S3 `get`, `put` | 1 inner row | silent (D4/O1) | 0 | `oplog_exactly_once_unlogged_store_s3` |
| `s3_utils::get_object`, `get_object_uri_async`, `get_object_uri` (sync) | 1 inner row, wrong shape | helper | 1 `GET`, store shape | `oplog_exactly_once_helpers_s3` |
| `s3_utils::get_object_uri_optimized_async` | 1 inner row, only when the object was served by a single GET; 0 for a range-split large object | helper | 1 `GET` whichever path served it | same |
| `s3_utils::get_objects_parallel`, `get_objects_parallel_with_progress` (Python `get_many` for `s3://`, `get_many_stats`, `get_many_async`) | 1 inner row per small object, 0 per large object | helper | 1 `GET` per object | same |
| `s3_utils::get_range` (downloads the whole object, then slices) | 1 inner `GET` | helper | 1 `GET` (the whole-object read it performs) | [READ] |
| `s3_utils::put_object_async`, `put_object_uri_async` | 1 inner row, wrong shape | helper | 1 `PUT`, store shape | same |
| `s3_utils::delete_objects` (sync, `lib.rs` re-export) | 1 inner `DELETE` row per 1000 keys, `file` = `(batch)` | helper | 1 `DELETE_BATCH` row per call, `n_objects` = keys, `file` = first URI; no row for an empty list | same |
| `s3_copy::upload_files`, `s3_copy::download_objects` | 1 inner row per object | helper (they call the helpers above) | 1 row per object | [READ] |
| `S3BytesDataset` (Python/Rust dataloader), sequential mode | 1 inner row per object | helper | 1 `GET` per object | [READ] |
| Checkpoint `CheckpointStore::open*`, `Reader::read_all_shards_concurrent*` (save and load of shards, manifests, `latest.json`) | S3: inner rows for `get`/`put`; other backends: none | explicit | 1 row per call, every backend | `oplog_exactly_once_callers` (file://) |
| `MultiBackendDataset::from_prefix`, `from_uris` (async pool dataloader) | S3: inner rows for `get`; other backends: none | explicit | 1 row per call, every backend | same |
| Python `load_checkpoint_with_validation` | S3: 1 inner row | explicit | 1 `GET` | [READ] (needs the wheel) |

## 2. Entry points that lose an accidental inner row (reported, not fixed here)

| Entry point | Rows before | Rows now | Why not fixed in W2 |
|---|---|---|---|
| `MultiEndpointStore` `get` / `put` on bucket-form `s3://bucket/..` endpoints | 1 inner row (wrong shape) | 0 | #175 (out of scope for W2): `MultiEndpointStore` is never logged by design today; `get_range`, `list`, `delete` and host-form endpoints were already 0 |
| `s3dlio-oplog` replayer (`crates/s3dlio-oplog/src/replayer.rs`, `store_for_uri`) | S3 `GET`/`PUT`: 1 inner row, if a global logger is active in the replaying process | 0 | consumer tool; capturing a replay is sai3-bench#103 |

## 3. Entry points that were never recorded by the inner writer (unchanged)

The inner writer did not cover these, so W2 neither adds nor removes a row. They are the remaining
"missing rows" and belong to later waves:

| Entry point | Today | Issue / note |
|---|---|---|
| `get_object_range_uri_async`, `get_object_range_uri_timed_async`, `get_object_concurrent_range_async` | no row from the helper (a logged store records `GET_RANGE`) | not an inner-writer path |
| `stat_object`, `stat_object_uri_async`, `stat_object_many_async`, `list_objects*`, `list_objects_stream`, `delete_objects_async`, `put_object_multipart_uri_async`, `create_bucket*`, `delete_bucket*` | no row from the helper | not an inner-writer path |
| Python `create_bucket`, `delete_bucket` | no row | #182 (Python bucket calls) |
| Python `get_many` for `file://`, `direct://`, `az://`, `gs://` | no row | built without a logger |
| Python `create_s3_writer` and the other writers | no row (writes are not recorded) | writer logging is a separate gap |
| Parquet readers (`parquet_rg`, `parquet_file_cache`), `parquet_gen` | no row | built with `store_for_uri` (silent); only `get_range`, `stat`, `list`, `put_multipart` are used, which the inner writer never covered |
| `S3BytesDataset` range mode | no row | uses `get_object_range_uri_async` |

## 4. Validation (ground-truth harness on Google Cloud Storage)

`tests/oplog_accuracy/` (see its README), one run per wheel, same cases, GCS S3-interop endpoint,
native `gcloud storage` as ground truth, 83 calls, `delete_batch` skipped (no probe given). Evidence:
`evidence/w2/` (bucket names replaced by the roles B1/B2).

| | parent wheel (9473d2c build, sha256 d3a9ef0c...) | final wheel (W2, sha256 10238517...) |
|---|---|---|
| calls | 83 | 83 |
| RECORD `DUPLICATE` | 20 | **0** |
| RECORD `MISSING` | 7 | 6 |
| RECORD `WRONG-FIELD` | 5 | 7 |
| PASS (OK/OK) | 49 | 47 (against the parent's known.json), 68 (against the updated one) |
| KNOWN | 34 | 9 / 15 |
| NEW / CHANGED / FIXED | 0 / 0 / 0 | 2 / 4 / 21 (against the parent's known.json) |

Against the parent's `known.json` the final run reports 21 FIXED, 2 NEW and 4 CHANGED:

- FIXED, #169: 18 calls (every S3 `get`/`put_bytes`/`put_bytes_async`, including the missing-key `get`, in
  `core`, `repeat`, `two_buckets`, `leading_slash`, `async` and `s3_dest_reject`) are one correct row.
- FIXED, #174: 3 calls (`async` `delete`, `host443_get_async`, `host443_decoydefault_get_async`) now have their row.
- NEW, 2: `multi_endpoint` `get` and `put` are now MISSING. Their accidental legacy row is gone (section 2; #175).
- CHANGED, 2: `host443_decoydefault_get_many` (`m1`, `m2`) record `OK` -> `WRONG-FIELD:server`. The call
  still goes to the `AWS_ENDPOINT_URL` server (#171); the legacy row used to carry that server by accident, the
  one row now written has the endpoint the URI asked for.
- CHANGED, 2: `portless_get`, `portless_put` record `DUPLICATE+WRONG-FIELD:server` -> `WRONG-FIELD:server`.

`known.json` was updated to the new expectations (entries for #169 and #174 removed; #175 now lists
`get` and `put`; the two CHANGED entries restated). Re-evaluating the final run against it gives
PASS 68, KNOWN 15, NEW 0, CHANGED 0, FIXED 0. Both runs cleaned up: the three namespaces were empty
afterwards.
