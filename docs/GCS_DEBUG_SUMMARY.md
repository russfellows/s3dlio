# GCS gRPC Debugging Summary

> Consolidation note (v0.9.70): this file is the short operational summary.
> The deep-dive technical analysis now lives in [GCS_BIDI_HANG_ANALYSIS.md](GCS_BIDI_HANG_ANALYSIS.md).
> Keep detailed call-stack and worker-lifecycle reasoning in the analysis doc to avoid duplication.

## 1. The Original Issue & Resolution
* **The Problem:** We originally observed a persistent hang/deadlock after a few seconds of concurrent gRPC reads against GCS using `s3dlio`. Trace logs indicated HTTP/2 flow control window starvation.
* **The Cause:** A custom patch in `google-cloud-rust`'s `grpc.rs` was allocating a massive 128 MiB `initial_stream_window_size`. This exhausted connection-level window limits instantly under high concurrency.
* **The Fix:** Setting the environment variable `S3DLIO_GRPC_INITIAL_WINDOW_MIB=0` successfully disabled the patch, completely eliminating the hang and allowing traffic to flow.

## 2. The Current Issue (`OUT_OF_RANGE`)
* Now that the pipeline isn't hanging, the benchmark immediately crashes with a new error from the Google Cloud Storage backend:
  `Error: GCS GET (gRPC) failed for gs://sig65-rapid1/t1-8m/object_126_of_1000.dat: the service reports an error with code OUT_OF_RANGE described as: The provided read ranges are outside the valid object range.`
* **CRITICAL FINDING:** `gcloud storage ls -L` output confirms that the failing object (`object_126_of_1000.dat`) has a **Content-Length of exactly 0 bytes**. 

## 3. What We Have Looked At (Code Archaeology)
We conducted rigorous `diff` comparisons between the "known-good" vendored `1.8.0` SDK (extracted from older `s3dlio` history) and the currently broken `1.9.0` fork:

1. **`s3dlio` API usage:** The caller logic remains unchanged. `get_object_via_grpc` correctly requests the entire file using `ReadRange::all()`. 
2. **`ReadRange` mappings (Data Structures):**
   * We checked the translation pipeline: `ReadRange::all()` -> `RequestedRange::Offset(0)` -> Protobuf `ReadRange { read_offset: 0, read_length: 0 }`.
   * **Result:** The protobuf definitions (`google.storage.v2.rs`), `RequestedRange`, and the core `OpenObjectRequest` mappings are **structurally identical** between the working 1.8.0 code and the broken 1.9.0 code. The way the offset requested range is packaged has *not* changed.
3. **The `bidi` Module (gRPC Streaming Lifecycle):**
   * While the data structures didn't change, the internal execution engine handling the stream completely did.
   * `open_object.rs`, `connector.rs`, and particularly `worker.rs` (now updated as `worker_new.rs`) have been heavily rewritten by upstream in 1.9.0.

## 4. Where We Think The Issues Are
Since the explicit `ReadRange` protobuf boundaries being generated are identical to the working version, the `OUT_OF_RANGE` error is almost certainly tied to how the newly rewritten `bidi` stream lifecycle handles **0-byte files**.

* **Hypothesis 1:** The old `worker.rs` may have gracefully handled a 0-byte file stream returning zero data chunks by checking the metadata `Content-Length` and terminating immediately. 
* **Hypothesis 2:** The new `worker_new.rs` or `connector.rs` might be miscalculating remaining bytes or mismanaging stream resumption. If the file is 0 bytes, and the worker inadvertently attempts to read past offset 0 (or assumes a non-zero default length in its connection retry/chunk logic), GCS explicitly rejects it with `OUT_OF_RANGE`. 

## 5. What We Still Need To Know (Next Steps)
*(Note: We are not solving this yet, just documenting the plan to solve)*
1. We need to verify if other files failing with `OUT_OF_RANGE` are also 0 bytes, confirming the 0-byte file edge case.
2. Run a trace or debug print directly inside the new `connector.rs` (where `BidiReadObjectRequest` is built) to see the exact `read_offset` and `read_length` the new worker actually passes over the wire.
3. Look at `worker_new.rs` to see what it does when it encounters a file with 0 explicit bytes returned in `BidiReadObjectResponse` (does it try to request more?).

---

## 6. Related Documentation

- Deep technical analysis: [GCS_BIDI_HANG_ANALYSIS.md](GCS_BIDI_HANG_ANALYSIS.md)
- Release notes context: [Changelog.md](Changelog.md)
