# FFI boundary hardening — error-chain-loss extension, UAF fix, and related bugs

> **Status: DRAFT — awaiting independent review before further implementation.**
> This document covers work triggered by storage#755 / s3dlio#161 (the
> `RuntimeError: concurrent range chunk failed` bug fixed earlier this
> session) and a follow-up maintainer request: *"the turnaround time for
> pushing out a full new release of s3dlio is 1+ hours — let's make sure
> we're not missing more bugs before we cut this one."*
>
> Three parallel read-only audits covered all five files in
> `src/python_api/` (`python_core_api.rs`, `python_aiml_api.rs`,
> `python_datagen_api.rs`, `python_advanced_api.rs`, `zero_copy_api.rs`) for
> the same class of bug as #755/#161, plus adjacent FFI-boundary
> correctness issues (silent error swallowing, panics, wrong-shape returns).
> This document proposes fixes for the confirmed findings and asks for
> review before they're implemented (except where noted as already done).

## Hard constraints (repeated here because they govern every fix below)

1. **No Python API changes**, with two narrow, explicitly-approved
   exceptions. Every function/method signature, argument name, return
   type, and return *shape* is unchanged, **except**: §4.1 (`checksum()`
   returns its real value instead of always `None` after `finalize()`)
   and §4.3 (`close()`'s dict always has an `'etag'` key, set to `None`
   when there is none, instead of omitting the key). Both are repairs to
   a shape/value the docstring already promised, not new capability or a
   deliberate API change — see each section for why the exception is
   narrow and justified. No other item in this document changes shape or
   value; Tier 1 (§3) changes only exception-message *text*.
2. **Stay zero-copy.** No fix introduces a bulk-buffer copy that wasn't
   already there. Where a fix touches a zero-copy path, the diff is
   reviewed line-by-line below to confirm it changes only reference
   counts / control flow, not data movement.
3. **RED-then-GREEN for every fix**, per the parent and s3dlio `CLAUDE.md`.
   Test plan is in §6, including which tests could not be run live in this
   session (no S3 endpoint reachable in this sandbox) and need to run
   before the wheel ships, per the newly-added "required before PR"
   section of `CLAUDE.md`.

**Resolved (was flagged as a contradiction in review):** constraint #1 now
explicitly carves out §4.1 and §4.3 as approved exceptions rather than
claiming strict shape-preservation while proposing two shape/value
changes. Both were judged worth keeping in scope — they fix real, already-
documented-but-broken contracts, and the review confirmed the checksum fix
is sound and the etag fix reasonable — rather than deferring either to a
separate release. If that trade-off is wrong, §4.3 is the smaller, more
isolated of the two to pull back out (see §7.2).

---

## 1. Already completed (for the reviewer to check, not re-approve from scratch)

These landed earlier in this session, with their own RED/GREEN tests
already run to GREEN. Listed here for completeness/traceability, not
seeking re-approval — flag if you disagree with any of them.

### 1.1 Original error-chain-loss fix (storage#755 / s3dlio#161)

- `src/python_api/python_core_api.rs`: split `py_err()` into a new
  `error_chain_message<E: Display>(&E) -> String` using `format!("{:#}", e)`
  (alternate Display — preserves the full `anyhow` cause chain) instead of
  `format!("{}", e)` (prints only the outermost `.context()` label).
  `py_err()` now calls it.
- `src/python_api/python_aiml_api.rs`: deleted a duplicate, still-buggy
  local `py_err()`; imported the fixed shared one; migrated ~23 inline
  `PyRuntimeError::new_err(e.to_string())` sites (every data-loader
  iterator: object/bytes/batch/parquet-row-group) onto it.
- 2 new Rust unit tests in `python_core_api.rs`
  (`error_chain_preservation_tests`) proving the chain survives through
  both a raw `anyhow::Error` and a `DatasetError::Backend`-wrapped one.
  Confirmed RED (compile failure — `error_chain_message` didn't exist yet)
  against unmodified code, GREEN after.
- Gate: `cargo fmt --check`, `cargo clippy --lib --features
  extension-module -- -D warnings`, `cargo test --lib --features
  extension-module` → 374 passed / 0 failed / 2 ignored.

**This is what the 3 audits below discovered was incomplete** — it fixed
the *shared helper* and the *aiml_api.rs data-loader iterators*, but not
the checkpoint bindings (same file), nor `python_core_api.rs`'s writer/
multi-endpoint code, nor `python_advanced_api.rs`'s multipart writer at
all. That gap is what §3 proposes to close.

### 1.2 Documentation fixes

- `Cargo.toml`: added a prominent comment above `[features]` explaining
  `native-backends`/`arrow-backend` are mutually exclusive by design
  (`compile_error!` guard in `src/lib.rs:6-10`) and that `--all-features`
  can never work — do not attempt it, do not touch the guard.
- `CLAUDE.md`, `.github/copilot-instructions.md`,
  `scripts/install-system-deps.sh`: removed/replaced every remaining
  `--all-features` suggestion (3 sites) with the commands CI actually
  runs, verified each corrected command actually builds.
- `CLAUDE.md`: added a new "Required before opening any PR" section —
  build the actual wheel (`./build_pyo3.sh`) and run `uv run pytest`
  against it, and (strongly recommended for anything touching
  data-loading/error-handling/FFI) drive a real `mlp-storage` benchmark
  against a live S3-compatible target using `storage_library=s3dlio`
  before opening a PR — this is what would have caught #755/#161 before
  it reached a user.

### 1.3 `reserve()` use-after-free fix (`python_advanced_api.rs`) — DONE, needs live-test sign-off

This one is implemented and passes the full local gate (see below), but
**could not be exercised against a live S3 endpoint in this session** — s3-ultra
is not currently running in this sandbox. Flagging clearly per
constraint #3 above; this needs a live multipart run
(`python/tests/test_multipart_writer.py` already covers the happy path;
see §6.3 for the new refcount-based regression test to add) before the
wheel ships.

**The bug** (audit finding, high confidence): `reserve()` returns a
writable `memoryview` via `PyMemoryView_FromMemory(ptr, len, flags)`. This
CPython API does **not** attach an owning object to the view — `view.obj`
is left null, so the memoryview carries no reference back to the writer.
If the last Python reference to the `MultipartUploadWriter` is dropped
while its `reserve()`-returned memoryview is still held (writer never
explicitly `commit()`/`abort()`/`close()`d, or simply goes out of scope),
the writer's `Drop` frees `pending_buf`'s `Vec<u8>` — but the outstanding
memoryview still points at that freed heap allocation. Any further
read/write into it is a genuine use-after-free.

This is not hypothetical: **the exact same defect was already found and
fixed once before**, in `PyBytesView` (`python_core_api.rs:433-435`,
doc comment: *"The previous implementation used `PyMemoryView_FromMemory`,
which does not link the memoryview's lifetime to this object — a
potential use-after-free."*). `PyBytesView` was fixed by switching to
`PyMemoryView_FromObject` routed through the buffer protocol
(`__getbuffer__`/`__releasebuffer__`, with `Py_INCREF(view.obj)`).
`MultipartUploadWriter::reserve()` still has the old, already-known-unsafe
pattern.

**Why not just copy `PyBytesView`'s exact fix (implement the buffer
protocol on the writer)?** Considered and rejected for this pass:
`PyBytesView`'s buffer is read-only and immutable-length; the writer's
`pending_buf` is writable and only valid for the reserve()..commit()
window, with a `shape` pointer that would need its own stable-lifetime
storage exactly like `PyBytesView::shape_len`, plus careful handling of the
`size == 0` special case. That's a materially larger, riskier change to
land in the same pass as everything else here. Open to revisiting if the
reviewer prefers it — see §7 open question.

**Fix actually implemented — self-pin, not a buffer-protocol rewrite:**
add a `self_pin_while_pending: Option<Py<PyMultipartUploadWriter>>` field.
`reserve()` builds it via `unsafe { Py::from_borrowed_ptr(py, slf.as_ptr()) }`
(a `Py_INCREF`, **zero data copy** — constraint #2) and stores it alongside
`pending_buf`. This keeps the writer object itself alive (and therefore
`pending_buf`'s allocation, and therefore the memoryview's pointer valid)
for the entire reservation window, regardless of what the caller does with
its own reference to the writer. Cleared by a new private
`clear_pending_reservation()` helper, called wherever `pending_buf` is
consumed or the writer is closed: `commit()` (right after `.take()`),
`abort()`, `close()`, `__exit__()` (start of each, since none of those
previously cleared an abandoned `pending_buf` either — see trade-off
below).

```rust
// Before (python_advanced_api.rs):
fn reserve(&mut self, py: Python<'_>, size: usize) -> PyResult<Py<PyAny>> {
    ...
    let mv_ptr = unsafe { ffi::PyMemoryView_FromMemory(ptr, len, flags) };
    ...
    self.pending_buf = Some(buf);
    Ok(mv)
}

// After:
fn reserve(mut slf: PyRefMut<'_, Self>, py: Python<'_>, size: usize) -> PyResult<Py<PyAny>> {
    ...
    let self_pin: Py<Self> = unsafe { Py::from_borrowed_ptr(py, slf.as_ptr()) };
    ...
    let mv_ptr = unsafe { ffi::PyMemoryView_FromMemory(ptr, len, flags) }; // unchanged
    ...
    slf.pending_buf = Some(buf);
    slf.self_pin_while_pending = Some(self_pin);   // NEW
    Ok(mv)
}
```

Python-visible signature is unchanged: `reserve(self, size: int) -> memoryview`
— PyO3 exposes a `PyRefMut<'_, Self>` receiver identically to `&mut self`
from Python's point of view; this is a Rust-internal detail needed to
obtain `slf.as_ptr()` for the `Py_INCREF`.

**Known trade-off, disclosed rather than hidden:** if a caller calls
`reserve()` then abandons the writer without *ever* calling
`commit()`/`abort()`/`close()`/using it as a context manager, this
self-reference is never cleared, and — because `PyMultipartUploadWriter`
does not implement PyO3's GC protocol (`__traverse__`/`__clear__`) —
Python's cyclic garbage collector cannot break the cycle either. The
writer leaks for the life of the process. **A leak is a strict safety
improvement over the use-after-free it replaces** (worst case: wasted
memory; not memory corruption), and the existing docstring already
documents `reserve()` as requiring a paired `commit()` call, so this is a
misuse case, not a normal-use regression. Flagged for reviewer awareness,
not proposed as something to further engineer around in this pass.

**Resolved: self-pin is the release fix.** Review confirmed it's the
right small-scope choice for this cut — it matches the "no bulk-copy /
small-scope" constraints better than a full writable buffer-protocol
implementation, and the leak-on-total-misuse trade-off is accepted (see
above: a leak is a strict safety improvement over the UAF it replaces, and
only triggers on a documented misuse pattern). The buffer-protocol
rewrite is **follow-up work, not a near-term alternative** — tracked as
§7.1, to revisit only if the leak trade-off turns out to matter in
practice.

**Verification so far:** `cargo build --lib --features extension-module`,
`cargo clippy --lib --features extension-module -- -D warnings`, `cargo fmt
--all -- --check` all clean; `cargo test --lib --features extension-module`
→ 374 passed / 0 failed / 2 ignored (no regressions). **Not yet run**
against a live wheel + real Python refcount check (§6.3) — no S3 endpoint
was reachable in this sandbox session. Per the maintainer's note, this is
fine — s3-ultra credentials are created at CLI-invocation time by design,
they don't need to be anything specific; this just needs to actually run
once a target is up, before the wheel is published.

---

## 2. Audit findings summary (condensed — full detail was reported inline earlier this session)

Three parallel agents each read one or more of the 5 `python_api/` files
end-to-end (not just grepped) and cross-checked findings against the
actual dependency/trait source where relevant. Condensed by tier:

- **Tier 1 — same bug class as #755/#161, ~44 more sites**, all following
  the identical pattern: an `anyhow::Result`-returning Rust function whose
  error is converted to `PyErr` via bespoke `format!("...: {}", e)` /
  `e.to_string()` instead of the fixed `py_err()`/`error_chain_message()`.
  See §3 for the full site list.
- **Tier 2 — distinct, high-confidence real bugs.** `reserve()` UAF (§1.3,
  done). `PyObjectWriter::checksum()` silently returns `None` after
  `finalize()` instead of the real, already-computed checksum (§4.1).
- **Tier 3 — real bugs, should fix.** Silent `let _ =
  create_bucket_rs(...)` swallowing in `put()`/`put_async_py` (§4.2).
  `close()` in `python_advanced_api.rs` omits the `'etag'` key entirely
  instead of setting it to `None`, contradicting its own docstring (§4.3).
- **Tier 4 — real, but a design decision, not a mechanical fix. Explicitly
  out of scope for this pass** (maintainer chose Tier 1+2+3 only):
  `exists()`/`exists_async()` collapsing all errors to `Ok(False)`;
  `__len__`/iteration on unknown-length datasets silently yielding zero
  items; `block_on()`-inside-runtime reentrancy risk in
  `PyObjectWriter`/`create_*_writer`; mutex-poisoning `.expect()` on 4
  hot-path iterators.
- **Tier 5 — `zero_copy_api.rs` disposition.** See §5.

---

## 3. Tier 1 — extend the error-chain fix to all remaining sites

**Mechanical, uniform fix, same pattern already proven in §1.1**: replace
`PyRuntimeError::new_err(format!("...: {}", e))` /
`PyRuntimeError::new_err(e.to_string())` /
`PyRuntimeError::new_err(format!("{:?}", e))` with `.map_err(py_err)` (or
`Err(py_err(e))`), importing `py_err` from `python_core_api` where the
file doesn't already have it.

**API-contract note (constraint #1):** this changes only the *text content*
of exception messages raised on already-documented error paths — the
exception type (`RuntimeError`), the fact that an exception is raised, and
every success-path return type/shape are unchanged. The one nuance: error
messages get *longer* (they gain the cause chain). Any caller/test doing
an exact-string match on a truncated error message would need updating —
worth a grep across `python/tests/` for such matches before merging (see
§6.1).

**Zero-copy note (constraint #2):** pure string formatting of already-
captured error values; no buffer/data path is touched by any Tier 1 site.

### 3.1 `python_aiml_api.rs` — checkpoint bindings (~13 sites) + 2 Debug-format sites

The entire checkpoint API never got the original fix (it's a separate
class hierarchy from the data-loader iterators §1.1 touched):
`CheckpointStore::open_with_config`, `.save()`, `.load_latest()`,
`.list_checkpoints()`, `.delete_checkpoint()`, shard-writer creation,
`.write_chunk()`, stream finalize/cancel, `save_distributed_shard()`,
`finalize_distributed_checkpoint()`, `load_latest_manifest()`,
`read_shard_by_rank()`, plus strategy-parse/runtime-creation error paths.

Two more sites (`PyS3AsyncDataLoader::__anext__`,
`PyAsyncDataLoaderIter::__anext__`) use `format!("{:?}", e)` (Debug) on a
`DatasetError` — lower severity, since `anyhow::Error`'s Debug impl
already includes the cause chain, but it surfaces to Python as ugly
`Backend(Error { ... })` enum syntax instead of the clean chain text
`py_err` produces everywhere else. Fixing for consistency.

### 3.2 `python_core_api.rs` — 19 sites

`PyObjectWriter::write_chunk`/`write_owned_bytes` (2), all 4
`create_*_writer` functions (S3/Azure/filesystem/direct-filesystem), all 5
`PyMultiEndpointStore` methods (`get`/`get_range`/`put`/`list`/`delete`),
all 4 multi-endpoint store constructors (including
`create_multi_endpoint_store_from_env`, which wraps a `.context("Failed to
parse S3_ENDPOINT_URIS")` — a concrete two-level chain that's flattened
today).

Note: `PyObjectWriter::finalize()` in this same `impl` block **already**
correctly uses `.map_err(py_err)` — the inconsistency (finalize fixed,
write_chunk/write_owned_bytes two lines above it not) is strong evidence
these were simply missed, not intentionally different.

### 3.3 `python_advanced_api.rs` — ~10 sites (entire `MultipartUploadWriter`)

`new()`, `from_uri()`, `write()` (3 sub-paths), `commit()`, `flush()`,
`close()`, `abort()`, `__exit__()`. This file does not import `py_err` at
all today.

### 3.4 `python_datagen_api.rs` — no fix needed

One site (`generate_npz_bytes`) already uses `format!("{e:#}")` — alternate
Display, chain-preserving. Functionally correct; not routed through the
shared helper, but low-value/low-risk to touch — **proposing to leave
as-is** rather than churn a working call site. Reviewer call if you'd
rather standardize it for consistency.

---

## 4. Tier 2/3 remaining fixes

### 4.1 `PyObjectWriter::checksum()` returns `None` after `finalize()` (Tier 2)

**Bug.** `finalize()` (`python_core_api.rs`) captures `(bytes_written,
compressed_bytes)` into `self.finalized_stats` *before* `writer.finalize()`
consumes `self.inner` — but does not capture `writer.checksum()` at the
same time. `checksum()` reads `self.inner.as_ref().and_then(|w|
w.checksum())`, so once `self.inner` is `None` (always true after
`finalize()`), it unconditionally returns `None` — even though the real,
fully-computed checksum was available and briefly reachable right before
`finalize()` consumed the writer.

```rust
// Before:
pub struct PyObjectWriter {
    finalized_stats: Option<(u64, u64)>, // (bytes_written, compressed_bytes)
    inner: Option<Box<dyn ObjectWriter>>,
}
fn finalize(&mut self, py: Python<'_>) -> PyResult<(u64, u64)> {
    if let Some(writer) = self.inner.take() {
        py.detach(|| { ... async move {
            let stats = (writer.bytes_written(), writer.compressed_bytes());
            writer.finalize().await.map_err(py_err)?;
            Ok::<(u64, u64), PyErr>(stats)
        }})
        .inspect(|&stats| { self.finalized_stats = Some(stats); })
    } else { Err(...) }
}
fn checksum(&self) -> Option<String> {
    self.inner.as_ref().and_then(|w| w.checksum())   // always None post-finalize
}

// Proposed:
pub struct PyObjectWriter {
    finalized_stats: Option<(u64, u64, Option<String>)>, // + checksum
    inner: Option<Box<dyn ObjectWriter>>,
}
fn finalize(&mut self, py: Python<'_>) -> PyResult<(u64, u64)> {
    if let Some(writer) = self.inner.take() {
        py.detach(|| { ... async move {
            let stats = (writer.bytes_written(), writer.compressed_bytes(), writer.checksum());
            writer.finalize().await.map_err(py_err)?;
            Ok::<(u64, u64, Option<String>), PyErr>(stats)
        }})
        .inspect(|(bw, cb, cs)| { self.finalized_stats = Some((*bw, *cb, cs.clone())); })
        .map(|(bw, cb, _)| (bw, cb))   // finalize()'s own return shape is UNCHANGED
    } else { Err(...) }
}
fn checksum(&self) -> Option<String> {
    if let Some((_, _, checksum)) = &self.finalized_stats {
        checksum.clone()
    } else if let Some(writer) = &self.inner {
        writer.checksum()
    } else {
        None
    }
}
```

**API-contract note (constraint #1, flagged per instruction):**
`finalize()`'s return type/shape (`(int, int)`) is unchanged.
`checksum()`'s return type (`Optional[str]`) is unchanged. What changes is
the **value** `checksum()` returns after `finalize()` — from always `None`
to the real checksum. This is fixing the function to deliver on its own
docstring ("Get the checksum of the data written, if available") rather
than a new capability, but it is an observable behavior change in the
post-finalize state, called out explicitly for sign-off as requested.

**Zero-copy note:** `checksum.clone()` clones a `String` — a small
metadata value (hex/base64 digest, tens of bytes), not a bulk data buffer.
No object-data copy introduced.

**This fix depends on `checksum()` being incremental, not
finalize-only — verified, not assumed.** `ObjectWriter::checksum()`'s own
trait doc (`src/object_store.rs:883-884`) states: *"Get the computed
checksum for the uncompressed data written so far. Returns `None` if no
checksum has been computed yet."* — "so far," not "only after
`finalize()`." Every checked implementation matches: `file_store.rs:212`
and `streaming_writer.rs:183` both compute it as
`Some(format!("crc32c:{:08x}", self.hasher.clone().finalize()))` from a
rolling hasher that's updated incrementally on every write, cloneable and
readable at any point without disturbing the writer. This is exactly why
snapshotting it alongside `bytes_written()`/`compressed_bytes()` right
before `writer.finalize()` consumes the writer is safe and correct today.
**This is a contract this fix now depends on**: if a future `ObjectWriter`
implementation ever computed its checksum only inside `finalize()`
(returning `None` beforehand), this snapshot would silently capture `None`
and reintroduce the same bug in a new form. Worth a one-line comment at
the snapshot site pointing back to this constraint, and/or a
`debug_assert!`-style regression test that fails loudly if a future writer
violates it — see §6.2.

### 4.2 Silent bucket-creation-failure swallowing (Tier 3)

**Bug.** `put()` and `put_async_py` (`python_core_api.rs`) both do `let _ =
create_bucket_rs(&bucket);` when `should_create_bucket=true` — any real
failure (permission denied, invalid name, network partition) is discarded,
and the subsequent `put_objects_with_random_data_and_type` call then fails
with a confusing "no such bucket"-shaped error instead of the actual
cause. `upload()`'s S3/Azure branches already handle the analogous
`create_container` failure correctly via `tracing::warn!` (not a hard
error — bucket-already-exists is a common, benign "failure" here) but two
*earlier* steps in that same `upload()` block (`store_for_uri_with_logger`
and `parse_s3_uri` failing) are still silently swallowed via bare `if let
Ok(...)` with no `else` branch.

**Proposed fix — match the existing, already-correct `warn!()` convention,
not a new pattern:**

```rust
// put() / put_async_py, before:
if should_create_bucket {
    let _ = create_bucket_rs(&bucket);
}

// after:
if should_create_bucket {
    if let Err(e) = create_bucket_rs(&bucket) {
        warn!("Failed to create bucket {}: {}", bucket, e);
    }
}
```

```rust
// upload(), before:
if let Ok(store) = store_for_uri_with_logger(&dest_prefix_owned, logger.clone()) {
    if dest_prefix_owned.starts_with("s3://") {
        if let Ok((bucket, _)) = parse_s3_uri(&dest_prefix_owned) {
            if let Err(e) = store.create_container(&bucket).await {
                warn!("Failed to create bucket {}: {}", bucket, e);
            }
        }
    } else if ... { ... }
}

// after: add the two missing else branches
match store_for_uri_with_logger(&dest_prefix_owned, logger.clone()) {
    Ok(store) => {
        if dest_prefix_owned.starts_with("s3://") {
            match parse_s3_uri(&dest_prefix_owned) {
                Ok((bucket, _)) => {
                    if let Err(e) = store.create_container(&bucket).await {
                        warn!("Failed to create bucket {}: {}", bucket, e);
                    }
                }
                Err(e) => warn!("Failed to parse S3 URI {}: {}", dest_prefix_owned, e),
            }
        } else if ... { ... }
    }
    Err(e) => warn!("Failed to open store for bucket creation at {}: {}", dest_prefix_owned, e),
}
```

**API-contract note:** purely additive logging via the existing `warn!`
macro already used elsewhere in this same function. No return
type/behavior/exception change — `should_create_bucket=true`/`create_bucket=true`
still best-effort-creates and does not hard-fail the overall call on
creation failure (unchanged design intent — only *visibility* into the
failure changes). No zero-copy concern (no data path touched at all).

### 4.3 `close()` omits the `'etag'` key instead of `None` (Tier 3)

**Bug.** `python_advanced_api.rs::close()`'s own docstring says the
returned dict always has `'etag': str or None`. The code:

```rust
if let Some(etag) = info.e_tag {
    dict.set_item("etag", etag).ok();
}
```

When `info.e_tag` is `None`, the `"etag"` key is never inserted — not
present as `None`, simply absent. `result['etag']` raises `KeyError`
instead of returning `None` as documented.

**Proposed fix:**

```rust
dict.set_item("etag", info.e_tag).ok();
```

PyO3 converts `Option<String>` to `None`/`str` automatically, so this one
line replaces the `if let` guard entirely.

**API-contract note (flagged per instruction, same as §4.1):** this
changes dict *shape* in the `e_tag.is_none()` case — from "key absent" to
"key present with value `None`" — which is what the docstring already
promised. A caller doing `result.get('etag')` sees no difference either
way; a caller doing `result['etag']` currently gets `KeyError` and would,
post-fix, get `None`. Flagged for explicit sign-off since it's a shape
change in that one branch, even though it's aligning code with an
existing documented contract rather than introducing a new one.

**Resolved:** constraint #1 (top of document) now names this as one of two
explicitly-approved exceptions rather than the document claiming strict
shape-preservation while proposing it anyway. Kept in scope as proposed —
see §7.2 if you'd rather pull it back out as the smaller, more isolated
of the two exceptions.

---

## 5. Tier 5 — `zero_copy_api.rs` disposition

**Finding (audit, high confidence, grep-verified not just inferred):**
`src/python_api.rs:14-16` has `mod zero_copy_api;` **commented out**:

```rust
// NOTE: zero_copy_api.rs contains valuable zero-copy implementations but is disabled
// due to numpy dependency. See zero_copy_api.rs header for enabling instructions.
// mod zero_copy_api;
```

This file is not part of the compiled wheel today, and — checked directly
against current source, not assumed — would not compile if re-enabled as
written:

- Calls `crate::s3_ops::put_object_data`/`get_object`/
  `get_object_into_buffer`/`head_object`/`create_multipart_writer` as free
  functions; `src/s3_ops.rs` has no such free functions, only async
  *methods* on a client struct with different signatures.
- References `crate::object_store::StreamWriter` — no such trait exists
  anywhere in `src/object_store.rs` or elsewhere in `src/`.
- Uses the pre-`Bound<'_, T>` PyO3 API (`&PyModule`, `&PyBytes`,
  `PyObject`) throughout — inconsistent with every other file in this
  directory, which migrated to the `Bound`-based API.

**Is there design value worth preserving?** Checked
`docs/ZERO_COPY_IMPLEMENTATION.md` (234 lines) — it documents the actual,
live zero-copy design (the `PyBytesView`/buffer-protocol pattern this
session's UAF fix in §1.3 also relies on) and does not reference
`zero_copy_api.rs` at all. The dead file's core idea — dispatch by Python
type (bytes / numpy array / generic buffer-protocol object) to a
zero-copy path — is **already implemented, correctly, and more completely**
in the live code: `PyObjectWriter::write_chunk` (buffer protocol +
`PyBytes` fallback), `PyMultipartUploadWriter::write()` (adds a
`PyBytesView` Arc-clone fast path on top of the same two), and
`PyBytesView::memoryview()`/`__getbuffer__` for the get-side. There is no
unique, working idea in `zero_copy_api.rs` that isn't already better
represented elsewhere in this codebase and in `ZERO_COPY_IMPLEMENTATION.md`.

**Resolved: delete `src/python_api/zero_copy_api.rs` and its module
declaration comment in `src/python_api.rs`, no archiving.** Confirmed via
a full in-repo reference check: nothing outside this draft and the
commented-out `src/python_api.rs:16` module line references it — no other
source file, test, doc, or build script names `zero_copy_api`,
`register_zero_copy_functions`, `PyBuffer` (the local struct — name
collides with `pyo3::buffer::PyBuffer`, itself a minor latent footgun were
this ever re-enabled), `PyStreamWriter`, or any of its free functions. No
markdown extraction either — there's nothing here `ZERO_COPY_IMPLEMENTATION.md`
doesn't already cover, and preserving broken, unreferenced sample code
(wrong API generation, calls into functions that don't exist) as a
"historical design doc" would be actively misleading to a future reader
rather than useful.

**API-contract note:** zero risk — nothing in this file is reachable from
Python today (module not registered), so removing it changes no observable
behavior for any caller.

---

## 6. Test plan

### 6.1 Tier 1 (44+ mechanical sites)

Same standard as §1.1's already-GREEN fix: the correctness of
`py_err()`/`error_chain_message()` is already proven by the 2 existing
unit tests (a raw chained `anyhow::Error` and a `DatasetError`-wrapped
one) — every Tier 1 site is "route an existing call site through an
already-tested helper it was previously bypassing," not new logic. Per-
site behavior is verified by: (a) compilation success, (b) the full
existing test suite staying green (regression net), (c) `clippy`/`fmt`
clean, (d) manual code review confirming each site now calls `py_err`.
**Additionally proposed for this pass** (going further than §1.1 did):
one representative new Rust unit test per newly-touched *class* of error
source (checkpoint `anyhow::Result`, `MultipartUploadSink`
`anyhow::Result`, `MultiEndpointStore` `anyhow::Result`) constructing a
synthetic 2-level chained error and asserting `error_chain_message`
preserves it — cheap, and closes the gap between "the helper is correct"
and "this specific call site's error type actually flows through it
correctly" for the highest-traffic new sites.
Before merging: `grep -rn` across `python/tests/` and `tests/` for exact-
string matches against any of the shortened error messages, to confirm no
test asserts on the old truncated text.

### 6.2 `checksum()` fix (§4.1)

**RED:** new test constructing a `PyObjectWriter` (via existing
file/direct backend test fixtures, no live S3 needed since this only
needs a `Box<dyn ObjectWriter>` that implements `checksum()`), calling
`finalize()` then `checksum()`, asserting the result is `Some(...)`
matching the pre-finalize value. Run against unmodified code — must fail
(`checksum()` returns `None`).
**GREEN:** apply the fix, same test now passes.

### 6.3 `reserve()` UAF fix (§1.3) — needs a live/mock S3 endpoint

Cannot be proven via a pure-Rust test (needs a real Python interpreter +
GIL + refcounting, which `cargo test` under `extension-module` doesn't
have — same constraint that shaped §1.1's test design). Proposed as a new
Python test, following `python/tests/test_multipart_writer.py`'s existing
conventions (needs `s3dlio.create_bucket`, a real `s3://` URI):

```python
def test_reserve_pins_writer_alive_until_commit(self):
    uri = f"s3://{self.bucket}/uaf-regression.bin"
    w = s3dlio.MultipartUploadWriter.from_uri(uri, part_size=32<<20, max_in_flight=4)
    baseline = sys.getrefcount(w)
    mv = w.reserve(1024)
    # RED (pre-fix): refcount unchanged, no self-pin exists.
    # GREEN (post-fix): refcount increased by 1 -- the self-pin is live.
    self.assertEqual(sys.getrefcount(w), baseline + 1)
    mv[:] = b"\xAB" * 1024
    w.commit(1024)
    # Self-pin released once the reservation window ends.
    self.assertEqual(sys.getrefcount(w), baseline)
    w.close()
```

This proves the *mechanism* (self-pin lifecycle) deterministically via
refcounting, without needing to actually trigger a UAF (which would be
flaky/UB-dependent to assert on directly). **Not run in this session** —
no S3 endpoint reachable in this sandbox (s3-ultra unreachable when
checked). Needs to run for real, per constraint #3, before this ships —
the maintainer noted credentials for a local target can be anything /
are created at CLI-invocation time by design, so this just needs a target
actually up.

**CPython-specific, called out explicitly rather than presented as
interpreter-agnostic:** `sys.getrefcount()` is a CPython implementation
detail (exact reference-counting semantics), not part of the Python
language spec — it would not behave the same way on PyPy or another
interpreter with a different memory-management strategy (e.g. tracing GC
without eager refcounting). This is a non-issue in practice for this
specific test: s3dlio ships as a CPython extension module built via PyO3's
`extension-module` feature (`Cargo.toml`), and `pyproject.toml` carries no
PyPy classifiers or support claims — CPython is the only target today. The
test file should still carry a one-line comment noting the
`sys.getrefcount()` dependency explicitly, so a future contributor
extending PyPy support (if that ever happens) knows to revisit this test
rather than being surprised by it.

### 6.4 Bucket-creation-swallow fix (§4.2)

RED/GREEN via a fault-injectable backend (existing repo convention per
`tests/test_multipart_abort_blocking.rs`'s "mock server's induced-failure
mode") or a `tracing` test-subscriber capturing whether `warn!` fired —
whichever matches how existing `warn!`-path tests in this repo are
structured; will follow that precedent exactly rather than invent a new
harness.

### 6.5 `'etag'` key fix (§4.3)

RED: unit-style test constructing a `close()`-equivalent result path with
`e_tag: None` and asserting the dict has an `'etag'` key with value
`None` (not `KeyError`/absent) — fails pre-fix, passes post-fix.

---

## 7. Open questions for the reviewer

Resolved by the first review pass (kept here for traceability, not
re-asking): §1.3's self-pin-vs-buffer-protocol choice (self-pin confirmed
for this release, buffer-protocol now tracked as §7.1 follow-up, not a
live alternative), and §5's `zero_copy_api.rs` disposition (delete
confirmed, no archiving, reference check confirmed clean).

Still open:

1. **Follow-up (not blocking this release):** revisit `reserve()` as a
   full buffer-protocol implementation (matching `PyBytesView` exactly —
   `__getbuffer__`/`__releasebuffer__`, `Py_INCREF(view.obj)`, a stable
   `shape` field) if the self-pin's leak-on-total-misuse trade-off (§1.3)
   ever turns out to matter in practice. Worth a tracking issue rather
   than doing it now.
2. **§4.1/§4.3 scope** — both are now explicitly carved out as approved
   exceptions to constraint #1 (see the top of this document). Confirming
   that's the right call rather than deferring either — §4.3 (`etag`) is
   the smaller, more isolated change if you'd rather pull just one back
   out.
3. **§3.4** — leave `python_datagen_api.rs`'s one already-correct-but-
   inconsistent site alone, or standardize it onto `py_err` too while
   we're in the area?
4. Anything in Tier 4 (explicitly deferred) you'd actually like pulled
   into this pass after seeing the Tier 1-3 detail above?
