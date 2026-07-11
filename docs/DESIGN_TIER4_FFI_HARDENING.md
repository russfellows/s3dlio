# Tier 4 — design questions for independent review

> **Status: IMPLEMENTED.** Items 1-4 were decided and implemented in
> commit `d7d0a06` ("fix(python_api): FFI-boundary hardening Tiers 1-5").
> Item 5 was deliberately left as Option A (no change) — see its section
> below. Each item's "Decision & implementation" subsection records what
> was chosen and why, with file/line and test evidence. This is a
> companion to `docs/DESIGN_FFI_BOUNDARY_HARDENING.md` (Tier 1-3, already
> implemented and merged into the working tree; Tier 5 — dead code
> removal — also implemented, same commit).

Each item below is presented with the trade-offs on both sides, not a
recommendation. Several of these could reasonably be answered "leave it,
it's fine" — that's a legitimate outcome, not a failure to find a fix.

---

## 1. `exists()` / `exists_async()` collapse every error to `Ok(false)`

**Current code** (`src/python_api/python_core_api.rs:1079-1112`):

```rust
pub fn exists(py: Python<'_>, uri: &str) -> PyResult<bool> {
    let uri_owned = uri.to_owned();
    py.detach(|| {
        let logger = global_logger();
        submit_io(async move {
            let store = match get_or_create_store(&uri_owned, logger) {
                Ok(s) => s,
                Err(_) => return Ok(false), // URI parsing error → doesn't exist
            };
            match store.stat(&uri_owned).await {
                Ok(_) => Ok(true),
                Err(_) => Ok(false),
            }
        })
    })
}

fn exists_async<'py>(py: Python<'py>, uri: &str) -> PyResult<pyo3::Bound<'py, PyAny>> {
    let uri = uri.to_owned();
    let logger = global_logger();
    let store = store_for_uri_with_logger(&uri, logger).map_err(py_err)?; // this part DOES propagate
    future_into_py(py, async move {
        match store.stat(&uri).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    })
}
```

**The behavior**: any failure — a malformed URI, wrong credentials, a
network partition, a TLS handshake failure, an endpoint that's down — is
indistinguishable from "the object genuinely does not exist." A caller
doing `if s3dlio.exists(uri): skip_download()` cannot tell "already have
it" apart from "can't currently reach storage to check."

Note the two functions aren't even consistent with each other today:
`exists()` swallows *both* the store-creation error and the `stat()`
error; `exists_async()` already propagates the store-creation error via
`py_err` (real chain, post-Tier-1) but still swallows the `stat()` error.

**Option A — leave as-is.** `exists()` returning `bool` is a very common,
very convenient API shape (mirrors `os.path.exists()`, `pathlib.Path.exists()`
— both of which have exactly this same "swallows PermissionError, returns
False" behavior in Python's own stdlib, by explicit design, for the same
ergonomic reason). Changing the *return type* would be a real API-shape
break, which is off the table without a much bigger conversation.

**Option B — split the API.** Keep `exists()`'s current bool-only
contract (for ergonomic call sites that only care about the common case);
point callers who need to distinguish "not found" from "couldn't tell" at
`stat()`, which **already exists and already propagates real errors**
(`python_core_api.rs:1003-1013` — `get_or_create_store(uri,
logger).map_err(py_err)?` and `submit_io(...)?`, both real `?`
propagation, verified by reading the code, not assumed). This means
Option B is **not a new-surface design at all** — the escape hatch a
caller needs already ships today. The only remaining work is a docstring
fix: `exists()`'s doc should explicitly say "returns `False` for both
'not found' and 'could not check' — use `stat()` if you need to tell
those apart," so callers who need the distinction know where to look. No
Rust logic changes required for Option B.

**Option C — narrow what gets swallowed.** Only collapse the specific
"not found" error variant to `false`; propagate everything else (auth,
network, TLS) as a real exception. This changes `exists()`'s *behavior*
(it can now raise) without changing its *return type* when it doesn't
raise — a middle ground between A and B. Requires each backend's error
type to reliably distinguish "not found" from other failures, which may
or may not already be true across S3/Azure/GCS/file/direct backends —
worth checking before committing to this option.

**Zero-copy / API-shape note**: none of A/B/C touch any data path.
Option A is a pure no-op. Option B is a docstring/API-guidance decision
only — it does not introduce or repoint any function at the Rust layer,
since `stat()` already exists and `exists()` itself is unchanged. Option C
changes `exists()`'s exception behavior (new: can raise) without changing
its return type in the non-raising case — flag for constraint-#1-style
sign-off if C is chosen, same spirit as Tier 1-3's §4.1/§4.3 exceptions.

**Decision & implementation: Option B.** `exists()`/`exists_async()`'s
logic in `src/python_api/python_core_api.rs` is unchanged — both still
collapse every failure to `False`. What changed is the docstrings: each
now states explicitly that `False` means either "not found" or "could
not check" (malformed URI, bad credentials, network partition, TLS
failure, unreachable endpoint), names the `os.path.exists()` /
`pathlib.Path.exists()` precedent for why that's the deliberate shape,
and points callers who need to distinguish the two cases at `stat()`
(which already propagates real errors — no new Rust surface needed). The
docstring also now calls out the one asymmetry between the sync and
async versions: `exists_async()` raises on a store-creation failure
(bad URI) before it ever reaches the swallowed `stat()` call, while sync
`exists()` swallows that case too — flagged as a "use `stat()`/
`stat_async()` for fully consistent behavior" note rather than silently
left undocumented. No test changes — no logic changed to test.

---

## 2. Unknown-length dataset iteration silently yields zero items

**Current code** (`src/python_api/python_aiml_api.rs`):

- `PyDataset::__len__` (line 94-96):

  ```rust
  fn __len__(&self) -> PyResult<usize> {
      Ok(self.inner.len().unwrap_or(0))
  }
  ```

  `Dataset::len() -> Option<usize>` returns `None` for iterable-style/
  streaming datasets by design (they don't know their length up front).
  `python(dataset)`'s `len(dataset)` reports **0** for these, not an error.

- Three producer entry points all gate their entire body on
  `if let Some(len) = dataset.inner.len() { ... }` with **no `else`
  branch**, so when `len()` is `None` the producer task does nothing and
  the channel closes immediately: `__iter__` (line 217),
  `.items()` (line 380), `spawn_stream` (line 696).

**The behavior**: iterating a legitimate unknown-length (streaming)
dataset through `PyBytesAsyncDataLoader` (`__iter__`, `.items()`, or
`__aiter__`) silently yields **zero items and no error** — not a
`TypeError` explaining that this dataset type needs a different iteration
method, not a warning, nothing. A caller would see an empty loop and have
to guess why.

**Corrected fact, verified by grepping `src/python_api/*.rs`**: `as_stream()`
is a Rust-only trait method (`src/data_loader/dataset.rs:72`,
`src/data_loader/transform.rs:53`, consumed internally by
`src/data_loader/dataloader.rs:89`). It is **never exposed as a
`#[pyfunction]`/`#[pymethods]` anywhere** — the only mention inside
`python_api/` is a code *comment* (`python_aiml_api.rs:476`) referencing
the concept, not a callable. There is no Python-visible alternative for
an unknown-length dataset today. This changes the shape of both options
below — the original draft's Option A incorrectly implied a usable
Python-side escape hatch existed.

**Option A — leave as-is, document it (weaker than it first appears).**
Since there's no real Python-callable alternative to point callers at,
"document it" can only honestly say "this iteration path doesn't support
unknown-length datasets; there is currently no other way to consume one
from Python" — which documents a **product gap**, not just a discoverability
one. Still the cheapest option and changes no behavior, but the reviewer
should treat this as "acknowledge the gap exists" rather than "point
users to the real solution."

**Option B — raise a clear error instead of silently yielding nothing.**
When `dataset.inner.len()` is `None` in `__iter__`/`.items()`/
`spawn_stream`, raise (e.g.) `TypeError("this dataset has unknown length —
no streaming iteration path is currently exposed to Python")` rather than
producing an empty iterator. Given there is no existing Python-side
streaming alternative, this option is really a choice between **"keep the
current silent wrong behavior"** and **"fail loudly until a real streaming
API is built and exposed to Python"** — not a choice between two equally
usable caller paths. That framing should drive the decision more than the
"is this a real behavior change" question the original draft posed.

**Option C — make `__len__` itself raise instead of returning 0.**
`self.inner.len().unwrap_or(0)` → `self.inner.len().ok_or_else(|| PyRuntimeError::new_err(...))`.
This fixes `len(dataset)` specifically (matches Python convention: a
`__len__` that can't answer should raise, not lie), but doesn't by itself
fix the three producer call sites above — those don't call `__len__`,
they call `.len()` on the underlying `Dataset` trait object directly, so
this is a narrower, independent fix from B. Could be done alone or
combined with B.

**Zero-copy / API-shape note**: A is a no-op. B and C both add new raise
paths where today there's silent success — flag for sign-off, same
pattern as Tier 1-3's checksum/etag exceptions, if either is chosen.

**Decision & implementation: Option B + Option C, combined.** Both fixes
landed together rather than choosing one over the other, since they're
independent (B fixes the 3 producer call sites, C fixes `__len__` itself)
and the doc noted they "could be done alone or combined."

A new pure, GIL-free helper was extracted rather than inlining the check
at each of the 4 call sites: `require_known_length(len: Option<usize>)
-> Result<usize, &'static str>` in `src/data_loader/dataset.rs:100-102`,
returning `Err(UNKNOWN_LENGTH_MSG)` — *"this dataset has unknown length —
no streaming iteration path is currently exposed to Python"* — when
`len` is `None`. Unit-tested directly (`tier4_length_guard_tests`,
`dataset.rs:104-118`): `known_length_passes_through` and
`unknown_length_is_rejected_not_defaulted_to_zero`, both pure `cargo
test --lib` (no `extension-module` feature needed, since the helper
never touches PyO3 types).

Wired into all 4 sites in `src/python_api/python_aiml_api.rs`, each
converting the `&'static str` into `PyTypeError`:
- `PyDataset::__len__` (`:99-100`) — Option C: raises instead of
  returning `0`.
- `PyBytesAsyncDataLoader::__iter__` (`:215`), `.items()` (`:384`), and
  `PyBytesAsyncDataLoaderIter::spawn_stream` (`:686`) — Option B: raise
  `TypeError` instead of silently producing an empty iterator.

Verified (per the commit message) that no currently Python-constructible
dataset type (`S3BytesDataset`, `FileSystemBytesDataset`,
`DirectIOBytesDataset`, `PyVecDataset`, `TransformDataset`) can trigger
this path today — this closes a latent trap for a future streaming
dataset type, not a live caller-facing behavior break for any dataset
that works today.

---

## 3. `block_on()`-inside-runtime reentrancy risk (9 sites, `PyObjectWriter` / `create_*_writer`)

**The file's own documented architecture** (`python_core_api.rs:44-63`)
explicitly avoids `block_on()` everywhere else, with a comment explaining
why:

> "Instead of calling `block_on()` (which panics if called from within a
> Tokio runtime), we use the io_uring pattern: Python thread →
> `handle.spawn(async work)` → `channel.recv()` ... The calling thread
> blocks on channel recv (NOT on `block_on()`), so it works from ANY
> context — plain OS threads, Python `ThreadPoolExecutor`, or even inside
> another runtime."

**9 sites never got migrated to that pattern** and call
`pyo3_async_runtimes::tokio::get_runtime().block_on(...)` directly:
`PyObjectWriter::write_chunk` (lines 1973, 1984), `write_owned_bytes`
(lines 2015, 2026), `finalize` (line 2043), and all four
`create_s3_writer`/`create_azure_writer`/`create_filesystem_writer`/
`create_direct_filesystem_writer` (lines 2100, 2126, 2164, 2189).

**The risk**: if any of these methods is ever invoked from a call stack
already running inside the shared Tokio runtime (e.g. a reentrant call
triggered from async Python code, or a callback dispatched from within an
in-flight `future_into_py` future elsewhere in the same process), it
panics with "Cannot start a runtime from within a runtime" — an opaque
`PanicException` rather than a clean `PyResult` error. For `finalize()`
specifically, a panic mid-multipart-finalize would drop the writer
without a clean abort path, potentially leaking an incomplete multipart
upload on the backend.

**How likely is this to actually trigger?** Genuinely unclear without
deeper investigation — these are all synchronous PyO3 methods
(`fn write_chunk(&mut self, ...) -> PyResult<()>`, not `async fn`), so a
typical single-threaded Python caller (`writer.write_chunk(data)` called
directly from a normal, non-async Python function) would never be
"already inside the runtime" when it calls in. The risk is specifically
around *any* caller that reaches these methods from async Python code
(e.g. some future refactor that routes them through `future_into_py`, or a
Rust-side callback/reentrant path that actually executes on the shared Tokio
runtime thread). A plain `asyncio` caller using `loop.run_in_executor`
is not a convincing example by itself, because that hops to a different OS
thread rather than re-entering the Tokio runtime thread directly. Worth
checking whether any current caller
(DLIO, mlp-storage, or `python/examples/`) does that before deciding this
is worth the migration effort. This section is fundamentally about
*reachability*, not just severity of the failure mode if triggered — an
imprecise motivating example makes the risk read more immediate than the
current architecture actually demonstrates, which is why the example
above is deliberately narrow rather than reaching for the first
async-sounding scenario.

**Option A — leave as-is.** These are synchronous methods; if nothing
today calls them from inside the runtime, this is a latent risk with no
current trigger, and the migration (rewriting 9 call sites to the
`run_on_global_rt`/`submit_io` pattern, plus writing fault-injection
tests proving the reentrancy case is actually handled) is real,
non-trivial work for a risk that may never materialize.

**Option B — migrate all 9 to the established `run_on_global_rt`/
`submit_io` pattern**, for consistency with the rest of the file and to
close the risk regardless of whether it's currently reachable. This is
the more thorough fix, described as "worth at least a design discussion"
in the original Tier 1-3 audit.

**Migration cost is not uniform across the 9 sites — verified by
re-reading each one, not assumed.** They split into two structurally
different buckets:

- **Easy bucket (7 sites) — already own their inputs or can snapshot
  state into owned values**: all four `create_*_writer` functions (lines
  2100, 2126, 2164, 2189 — construct and return a fresh, owned
  `PyObjectWriter`, no borrowed Python state involved), `finalize()`
  (line 2043 — already consumes `self.inner` by `.take()` before the
  `block_on` call), and `write_owned_bytes()` (lines 2015, 2026 — the
  clue is in the name: it already copies buffer-protocol input into an
  owned `Vec<u8>`/`PyBytes::to_vec()` *before* the `block_on` call, so
  the async boundary only ever touches owned data).
- **Hard bucket (1 site, 2 call sites within it) — `write_chunk()`**
  (lines 1973, 1984). Its whole reason to exist is the *zero-copy* fast
  path: `PyBuffer::<u8>::get(data)` yields a raw pointer, and the code's
  own safety comment (`python_core_api.rs:1935-1936`) states the
  invariant explicitly: *"We hold the buffer for the entire duration of
  block_on, so the memory remains valid. The GIL is held during
  block_on."* That invariant is exactly what a migration to
  `submit_io`/`run_on_global_rt` (which spawns onto a different task /
  releases synchronous holding) would break — the borrowed slice cannot
  be proven to outlive the spawned future without either (a) copying the
  data into an owned buffer first, which defeats the entire point of this
  method existing as a separate zero-copy path from `write_owned_bytes()`,
  or (b) a materially more complex lifetime-pinning design. This is not a
  mechanical rewrite like the other 8.

**Option C — migrate only the easy bucket (7 sites: all 4
`create_*_writer` + `finalize()` + `write_owned_bytes()`'s 2 call
sites)**, leaving `write_chunk()`'s 2 call sites as an explicitly
separate, harder follow-up (not bundled with "the other 8" as originally
framed, since 7 of the 9 are mechanical and 2 are not). This closes most
of the reentrancy risk immediately, including the worst failure mode
(`finalize()`'s leaked-multipart-upload-on-panic case) without taking on
`write_chunk()`'s harder lifetime redesign in the same pass.

**Zero-copy / API-shape note**: this is purely an internal
implementation-pattern change (which mechanism blocks on the async work),
not a signature or behavior change in the success case. No data-copy
implications either way — `run_on_global_rt`/`submit_io` already carries
zero-copy `Bytes` through channels per the same architecture comment
quoted above.

**Decision & implementation: Option C — easy-bucket-only (7 of 9 sites
migrated).** All 4 `create_*_writer` functions, `finalize()`, and both
`write_owned_bytes()` call sites in
`src/python_api/python_core_api.rs` now go through `submit_io`
(the `run_on_global_rt`/channel-recv pattern), matching the rest of the
file. `write_owned_bytes()` needed a take-then-restore rewrite —
`self.inner` is only borrowed (`&mut self`), not owned, so the writer is
`.take()`n out, moved into the `submit_io` future, and unconditionally
restored afterward, preserving the existing "writer stays usable after a
failed write" contract.

`write_chunk()`'s 2 call sites (now at `:1994` and `:2005`) are
**deliberately left on raw `block_on()`**, exactly as the doc's "hard
bucket" analysis anticipated: its zero-copy fast path holds a
`PyBuffer`-derived raw slice for the exact duration of a synchronous,
GIL-held `block_on()`, and `submit_io`'s spawn-based model requires
`'static` futures, which can't hold a borrowed slice without either
copying the data (defeating the point of `write_chunk` existing as a
separate path from `write_owned_bytes()`) or a materially more complex
lifetime-pinning redesign. That redesign was explicitly not attempted in
this pass.

Regression test: `src/s3_client.rs:452-473`,
`tier4_reentrancy_tests::run_on_global_rt_survives_nested_runtime_context`
— nests a `run_on_global_rt(...)` call inside an already-running Tokio
runtime (the exact "Cannot start a runtime from within a runtime"
condition) and asserts it returns cleanly instead of panicking. RED was
confirmed by temporarily reintroducing a raw `block_on()` inside
`run_on_global_rt()` itself and observing the real panic; GREEN after
restoring the real implementation. Live-verified: all 4 tests in
`python/tests/test_multipart_writer.py` (including
`test_reserve_pins_writer_alive_until_commit` and
`test_close_always_has_etag_key`) pass against a real MinIO endpoint
with the rebuilt wheel, exercising the migrated writer paths.

---

## 4. Mutex-poisoning panics wedge 4 hot-path iterators permanently

**Current code** (`src/python_api/python_aiml_api.rs`), all identical
shape:

```rust
// PyObjectDataLoaderSyncIter::__next__ (line 558-559)
.lock()
.expect("PyObjectDataLoaderSyncIter rx mutex poisoned")

// PyObjectDataLoaderSyncIter::collect_batch (line 610-611)
.lock()
.expect("PyObjectDataLoaderSyncIter rx mutex poisoned")

// PyBytesDataLoaderSyncIter::__next__ (line 660-661)
.lock()
.expect("sync iter rx mutex poisoned")

// ParquetStreamIter::__next__ (line 2202-2203)
.lock()
.expect("ParquetStreamIter rx mutex poisoned")
```

**The risk**: `std::sync::Mutex` poisons itself if any thread panics
while holding the lock. If that ever happens (a bug anywhere else in the
same lock scope, an unrelated panic during a `blocking_recv()` call,
etc.), every *subsequent* call to `__next__()`/`collect_batch()` on that
same iterator object panics via `.expect(...)` instead of returning a
clean error — the iterator is permanently wedged for the rest of its
lifetime, not just the one call that hit the original panic.

**How likely is this to actually trigger?** Requires a prior panic while
the lock is specifically held (a narrow window), so likely rare in
practice — but per-object permanent wedging (rather than "this one call
failed, try again") is a notably worse failure mode than a normal error
return, which is why the original audit flagged it despite the low
likelihood.

**Option A — leave as-is.** `std::sync::Mutex` + `.expect()` is a common,
simple pattern; PyO3 already catches the resulting panic and turns it
into a Python `PanicException`, so this isn't unsound — just a worse
diagnostic/recovery experience than a clean error would be. Fixing all 4
requires either switching to `parking_lot::Mutex` (doesn't poison, but
that's a real dependency/behavior change worth its own evaluation) or
`.lock().unwrap_or_else(|poisoned| poisoned.into_inner())` (recovers from
poison but silently continues with whatever state existed at panic time —
its own correctness question).

**Option B — convert each `.expect()` to a clean `PyResult` error** (e.g.
`.map_err(|_| PyRuntimeError::new_err("iterator's internal channel lock
was poisoned by a prior panic — this iterator is no longer usable"))`.

**Important framing correction: Option B is a diagnostics improvement,
not a recoverability fix, and should not be evaluated as if it were the
latter.** Before Option B, a poisoned lock crashes the process-visible
call with an opaque `PanicException`; after Option B, the same call
returns a clean, catchable `RuntimeError` with an actionable message —
strictly better for debugging and for a caller that wants to catch and
log the failure. But **the iterator is exactly as dead in both cases** —
poisoning is permanent for the lifetime of that `std::sync::Mutex`
instance regardless of whether the poisoned-state code path panics or
returns `Err`. Nothing about Option B lets a caller retry, reset, or
otherwise recover the iterator; it only changes how the terminal failure
is reported. If the reviewer wants actual recoverability (the ability for
the iterator to keep working after some earlier, unrelated panic), that's
Option C below, not B.

**Option C — evaluate `parking_lot::Mutex`** (already a dependency —
`Cargo.toml` line 128 — used elsewhere in the crate) for these 4 fields
specifically, since it doesn't poison on panic at all, sidestepping the
whole class of problem rather than just improving its error message.
Bigger change, touches the struct field types, worth its own scoped
investigation rather than deciding here.

**Zero-copy / API-shape note**: Option B changes the exception type
raised in the (rare, panic-recovery) poisoned case from `PanicException`
to `RuntimeError` — a real behavior change in an edge case, flagged for
sign-off if chosen, though arguably strictly an improvement (catchable
`RuntimeError` vs. opaque `PanicException`). Option C is a bigger,
separate investigation. Neither touches the actual data-transfer path.

**Decision & implementation: Option C — `parking_lot::Mutex`.** Went
straight to eliminating the hazard class rather than just improving its
diagnostics (which is all Option B would have bought). The `rx` channel
field on `PyObjectDataLoaderSyncIter`, `PyBytesDataLoaderSyncIter`, and
`ParquetStreamIter` (`src/python_api/python_aiml_api.rs`) switched from
`std::sync::Mutex` to `parking_lot::Mutex` — already a workspace
dependency (`Cargo.toml:128`), so no new dependency was introduced.
`parking_lot::Mutex` never poisons, so a single unrelated panic while
the lock happens to be held no longer wedges that iterator's `__next__`
for the rest of its lifetime.

The lock-then-blocking-receive pattern was centralized into a new pure,
GIL-free helper, `blocking_recv_locked<T>(mtx: &parking_lot::Mutex<...>)
-> Option<T>` (`src/data_loader/dataset.rs:129-133`) — safe to call this
way because the mutex is only ever locked inside a fully synchronous
`py.detach(|| ...)` closure, never across an `.await` point, so there's
no `Send`/async hazard from the type swap. Unit-tested
(`tier4_mutex_poisoning_tests`, `dataset.rs:135-182`):
`std_mutex_poisons_and_wedges_after_panic` characterizes the original
hazard (panic while held → all subsequent `.lock()` calls fail forever);
`blocking_recv_locked_survives_panic_while_held` proves the replacement
survives the identical scenario — a panic on another thread while the
`parking_lot::Mutex` is held does not wedge a later
`blocking_recv_locked` call. RED was confirmed by temporarily reverting
`blocking_recv_locked` itself back to `std::sync::Mutex` +
`.expect(...)` and observing the real "rx mutex poisoned" panic; GREEN
after restoring `parking_lot::Mutex`.

---

## 5. (Carried over from Tier 1-3's §3.4, per instruction) `python_datagen_api.rs`'s one inconsistent-but-correct site

**Current code** (`src/python_api/python_datagen_api.rs:410`):

```rust
.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e:#}")))?;
```

This is the **only** error-conversion site in the whole `python_api/`
tree that manually uses `{:#}` (alternate Display — the chain-preserving
format) instead of routing through the shared `py_err()`/
`error_chain_message()` helper that now exists specifically to do this
(post-Tier-1). Functionally, this site is **already correct** — it
already preserves the full `anyhow` cause chain, which was the entire
point of the Tier 1-3 work. The only issue is that it's inconsistent
*style*, not a bug.

**Option A — leave it alone.** It works. Touching a correct call site
purely for stylistic consistency carries a small amount of risk (however
small) for zero behavioral benefit — the standard argument against
churn-only changes.

**Option B — standardize it onto `py_err` anyway**, for the mechanical
reason that every other error-conversion site in the codebase now goes
through the shared helper, and a lone manual `{:#}` site is the kind of
thing a future contributor might "fix" back to plain `{}` by accident,
not realizing it was already deliberately correct — routing it through
`py_err` makes the correct behavior the *only* pattern present, removing
that footgun.

The trade-off is the same shape as items 1-4 above: A costs nothing and
risks nothing but leaves a stylistic inconsistency in place; B costs a
one-line change and removes a footgun (a future contributor "fixing" the
lone `{:#}` site back to `{}` by accident, silently reintroducing the
exact chain-loss bug Tier 1 fixed everywhere else) at the standard small
risk of touching an already-working call site. Kept neutral, consistent
with every other item in this document — the user has an opinion on this
one already and is deliberately withholding it until the independent
review is complete, so this section should stay options-only rather than
staking out a lean.

**Decision & implementation: Option A — left alone.** `python_datagen_api.rs:410`
is untouched — still the lone manual `format!("{e:#}")` site, no
`py_err` call anywhere in the file. This was the one item of the five
explicitly *not* addressed in the Tier 4 implementation pass (the commit
message scopes itself to "all 4 addressed items," naming items 1-4
only). The footgun risk described above (a future contributor "fixing"
this site back to `{}`, silently reintroducing the chain-loss bug) is
still live and undecided — it remains open for a future pass if the
reviewer wants Option B after all.

---

## Summary table

| # | Item | Files/lines | Effort if fixed | Sign-off needed if fixed? | Decision |
| --- | ------ | ------------- | ------------------ | --------------------------- | --- |
| 1 | `exists()`/`exists_async()` swallow all errors | `python_core_api.rs:1079-1112` | Low (B) / Medium (C) | Only if Option C | **B** — docstring only, no logic change |
| 2 | Unknown-length iteration silently empty | `python_aiml_api.rs:94-96,217,380,696` | Low (A/C) / Medium (B) | If B or C | **B+C** — `require_known_length()` raises `TypeError` at all 4 sites |
| 3 | `block_on()` reentrancy risk, 9 sites | `python_core_api.rs:1973-2189` | Low (A) / High (B) / Medium (C) | No (internal pattern only) | **C** — 7 easy-bucket sites migrated to `submit_io`; `write_chunk()`'s 2 sites deliberately left on `block_on()` |
| 4 | Mutex-poisoning panics, 4 sites | `python_aiml_api.rs:558-2203` | Low (A/B) / Medium (C) | If B | **C** — switched to `parking_lot::Mutex` (no poisoning) |
| 5 | `datagen_api.rs` style consistency | `python_datagen_api.rs:410` | Trivial | No | **A** — left alone, still open |

Implemented in commit `d7d0a06` (items 1-4); item 5 remains undecided —
still Option A by default, per the commit's explicit "4 addressed
items" scope. See that commit's message for the full RED/GREEN
verification narrative, and `src/data_loader/dataset.rs`'s
`tier4_length_guard_tests` / `tier4_mutex_poisoning_tests`, and
`src/s3_client.rs`'s `tier4_reentrancy_tests`, for the unit-test
evidence backing items 2, 3, and 4.
