# 7 test failures found during FFI-boundary hardening live-testing (2026-07-10)

> **Status: RESOLVED.** Both groups were confirmed pre-existing, NOT caused
> by the Tier 1-3 FFI-boundary hardening work
> (docs/DESIGN_FFI_BOUNDARY_HARDENING.md) — verified by re-running the
> identical test files against the pristine pre-session `main` (commit
> `383f0d7`, before any change in this session), same 7 failures, same
> error text, same line numbers. Both are now fixed, same session:
>
> - **Group A**: `test_checkpoint_basic.py` deleted as a superseded
>   duplicate (commit `d76fec2`) — `test_checkpoint_basic_python.py`
>   already covers the same scenarios against the real, current API.
> - **Group B**: the test fixture was fixed to model true endpoint
>   replication (matching `multi_endpoint.rs`'s own Rust test suite), and
>   the underlying design gap it exposed was investigated, filed as
>   [GitHub issue #162](https://github.com/russfellows/s3dlio/issues/162),
>   and fixed — `MultiEndpointStore` gained explicit per-endpoint pinning
>   (`get_from_endpoint`/`put_to_endpoint`/`delete_from_endpoint`) and
>   fan-out replication (`put_all_endpoints`), plus crate-level docs making
>   the replication-only assumption of round-robin/least-connections
>   explicit. See commit `f1c4518`.
>
> Kept below as the original incident record.

## How these were found

While live-testing the Tier 1-3 fixes against a local s3-ultra target
(`python/tests/test_multipart_writer.py` — all green, see
`DESIGN_FFI_BOUNDARY_HARDENING.md` §1.3/§6), a broader confirmation pass
was run across other test files touching the same general surface
(checkpoints, multi-endpoint store) to check for incidental regressions.
That pass surfaced these 7 pre-existing failures, unrelated to the FFI
work. Full raw pytest output captured at the time:
`/tmp/claude-1000/-home-eval-Documents-Code/264eebff-b762-4534-aa79-c34c5ca0e64d/scratchpad/s3dlio-7-failures-full.log`
(session-scoped scratchpad — not guaranteed to persist; the essential
content is reproduced below).

Repro command:
```bash
uv run python -m pytest python/tests/test_checkpoint_basic.py python/tests/test_multi_endpoint.py -v --tb=long
```
Result: `7 failed, 14 passed`.

---

## Bug Group A — `test_checkpoint_basic.py` calls a nonexistent API (3 failures)

**Root cause, confirmed**: the test file calls `s3dlio.PyCheckpointStore.open(uri)` and
`s3dlio.PyCheckpointStore.open_auto_config(uri)` — static factory methods
that **do not exist** on `PyCheckpointStore` today.

Checked the actual Rust binding (`src/python_api/python_aiml_api.rs`,
`impl PyCheckpointStore`): the only constructor is the standard `#[new]`
(i.e. `s3dlio.PyCheckpointStore(uri, strategy, multipart_threshold)`,
positional/keyword `__init__`), no `open`/`open_auto_config` staticmethods
anywhere in the file. Confirmed by cross-checking the **passing** sibling
test file `python/tests/test_checkpoint_basic_python.py`, which
successfully calls the real API three times: `s3dlio.PyCheckpointStore(base_path, None, None)`.

**Conclusion**: `test_checkpoint_basic.py` is a stale test file — it was
never updated (or never deleted) after `PyCheckpointStore`'s constructor
API changed to the current `#[new]`-only shape. This is very likely a
**test-suite bug** (dead/duplicate test file), not a production code bug —
`test_checkpoint_basic_python.py` already covers the same scenarios
correctly against the real API. Low urgency, but it's dead weight in the
suite and will confuse anyone who runs the full suite and sees red.

**Failures:**

| Test | Line | Error |
|---|---|---|
| `test_basic_checkpoint_operations` | `test_checkpoint_basic.py:25` | `AttributeError: type object 'builtins.PyCheckpointStore' has no attribute 'open_auto_config'` |
| `test_checkpoint_versioning` | `test_checkpoint_basic.py:64` | `AttributeError: type object 'builtins.PyCheckpointStore' has no attribute 'open'` |
| `test_checkpoint_validation` | `test_checkpoint_basic.py:97` | `AttributeError: type object 'builtins.PyCheckpointStore' has no attribute 'open'` |

**Suggested fix (not yet done, needs a decision, not just a patch)**: either (a)
delete `test_checkpoint_basic.py` as superseded by `test_checkpoint_basic_python.py`,
or (b) update its 3 call sites to the real constructor if it covers
scenarios the Python-suffixed file doesn't. Whoever picks this up should
diff the two files' actual test bodies first — they may not be pure
duplicates despite the near-identical names.

---

## Bug Group B — `MultiEndpointStore` round-robin routes `get()`/`list()`/`delete()` to a *different, non-replicated* endpoint than the one just `put()` to (4 failures)

**Root cause, confirmed by reading `src/multi_endpoint.rs`**: this is
**working exactly as designed and documented** — `rewrite_uri_for_endpoint`
(`src/multi_endpoint.rs:433-479`) explicitly rewrites *any* fully-qualified
URI matching a configured endpoint's prefix to point at the endpoint
selected by the load-balancing strategy instead, even when the caller
passed a URI that already fully specifies one particular endpoint. Its own
doc comment states the intent plainly:

```
/// This enables transparent load balancing where callers can use URIs constructed
/// with any endpoint prefix, and we correctly route to the selected endpoint.
///
/// # Examples
/// // Endpoints: ["file:///tmp/ep1/", "file:///tmp/ep2/"]
/// // URI "file:///tmp/ep1/data/obj.dat" targeting ep2 -> "file:///tmp/ep2/data/obj.dat"
```

Combined with `select_endpoint()`'s round-robin (`self.next_index.fetch_add(1) % N`,
`src/multi_endpoint.rs:486-498`) advancing on **every** call, this design
assumes every configured endpoint is a **true replica** holding the same
object under the same relative key — the real-world scenario this is built
for (e.g. several real S3-compatible hosts serving the same bucket behind
a client-side load balancer, or DNS round-robin to the same storage
cluster).

The failing tests' fixture (`TestMultiEndpointOperations.multi_store`,
`test_multi_endpoint.py:89-101`) does not meet that assumption — it points
`MultiEndpointStore` at **3 independent, non-replicated local directories**
(`tmp_path/endpoint0`, `endpoint1`, `endpoint2`). A `put()` to
`file://.../endpoint0/test.txt` writes only into that one directory; the
very next call (`get()`, round-robin index advances) deterministically
rewrites the URI to `file://.../endpoint1/test.txt` — a file that was
never written. This is **not flaky/intermittent** — round-robin's
`% 3` cycling guarantees the 2nd call always lands on a different endpoint
than the 1st, every run.

**This is genuinely ambiguous whether it's a test bug or a design gap** —
flagging both readings rather than picking one, for whoever investigates:

- **Reading 1 (test bug)**: the fixture is wrong for this test's intent —
  it should use 3 directories that are kept in sync (e.g. write the same
  file to all 3 before testing reads), or the test should assert
  differently (e.g. only that *some* endpoint has the data, via
  `list_all_endpoints`, which this file's own `TestMultiEndpointOperations`
  suite doesn't use even though `src/multi_endpoint.rs:607` documents it
  as the right tool "when objects are distributed across endpoints").
- **Reading 2 (design gap worth a real look)**: is it surprising/dangerous
  that `get(specific_fully_qualified_uri)` can silently return
  `FileNotFoundError` — or worse, in a *replicated-but-inconsistent*
  scenario, silently return **different/stale data** — because it silently
  routed to a different endpoint than the URI named? A caller who does
  `put(uri); data = get(uri)` expecting read-your-writes semantics gets
  neither an error explaining the rewrite happened nor the data they just
  wrote, unless they happen to hit the same round-robin slot by luck.
  Worth asking whether `get()`/`delete()` of an *explicit* endpoint-qualified
  URI (as opposed to a relative/logical key) should honor that endpoint
  rather than rewriting it — `put()`-then-round-robin makes sense for
  spreading write load, but silently rewriting a caller's own read-back
  of a specific URI is a sharper edge than "transparent load balancing"
  suggests.

**Failures:**

| Test | Line | Error |
|---|---|---|
| `test_put_and_get` | `test_multi_endpoint.py:113` | `RuntimeError: Get failed: File not found: .../endpoint1/test.txt` (written to endpoint0) |
| `test_get_range` | `test_multi_endpoint.py:125` | `RuntimeError: Get range failed: File not found: .../endpoint1/range_test.txt` (written to endpoint0) |
| `test_list_objects` | `test_multi_endpoint.py:143` | `AssertionError: assert 1 >= 5` — only 1 of 5 written files visible (`list()` also round-robins to a single endpoint per call, per its own doc distinguishing it from `list_all_endpoints()`) |
| `test_delete_object` | `test_multi_endpoint.py:153` | `RuntimeError: Get failed: File not found: .../endpoint1/delete_me.txt` (written to endpoint0) |

**Suggested next step (not yet done)**: read `src/multi_endpoint.rs`'s
full test suite (`test_round_robin_load_balancing`,
`test_round_robin_all_4_endpoints_utilized_with_n_requests`, etc. —
`src/multi_endpoint.rs:1298+`) to see whether the *Rust-level* tests
already correctly model replicated endpoints (and thus never hit this),
which would confirm the gap is specific to this Python test's fixture
design rather than the Rust implementation itself. Then decide between
Reading 1 and Reading 2 above before touching any code.

---

## Disposition

Both groups are now resolved, same session. Group A: `test_checkpoint_basic.py`
deleted (superseded duplicate). Group B: Reading 1 (test fixture bug) fixed
directly; Reading 2 (the design gap) investigated against real usage in
DLIO_local_changes and mlp-storage, filed as
[GitHub issue #162](https://github.com/russfellows/s3dlio/issues/162)
following this repo's convention (e.g. issues #151-157 for the prior audit),
and fixed via explicit per-endpoint pinning + fan-out replication APIs. See
the status note at the top of this document for commit references.
