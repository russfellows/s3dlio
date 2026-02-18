# Handoff: v0.9.50 Python Runtime Fix & Documentation

**Date:** February 13, 2026  
**Branch:** `feature/enhance-python-runtime-v0-9-50`

## What Was Done

### 1. Critical Runtime Bug Fix (python_core_api.rs, python_aiml_api.rs)

Replaced all `GLOBAL_RUNTIME.block_on()` calls with io_uring-style `submit_io()` → `run_on_global_rt()` (spawn+channel pattern). This fixes two bugs:
- v0.9.27: "dispatch failure" from per-call `Runtime::new()`
- v0.9.40: "Cannot start a runtime from within a runtime" from `block_on()`

12 functions updated in python_core_api.rs, 2 in python_aiml_api.rs. All use owned Strings for `'static` lifetime bounds on `submit_io`.

### 2. New `put_many()` / `put_many_async()` Batch Upload

Accepts `Vec<(String, Vec<u8>)>` — list of `(uri, data)` tuples. Registered in `register_core_functions()`.

### 3. s3torchconnector Compat Layer Rewrite (python/s3dlio/compat/s3torchconnector.py)

Complete rewrite with zero-copy performance. New `_BytesViewIO(io.RawIOBase)` class for seekable file-like access to BytesView. All 9 functional tests passed.

### 4. Documentation & Version Bump (This Session)

- **docs/PYTHON_API_GUIDE.md** — Complete rewrite for v0.9.50 (architecture, zero-copy, threading, compat layer, full API reference)
- **docs/Changelog.md** — Added v0.9.50 entry with "Fixes Applied" table and all changes
- **Cargo.toml** — `workspace.package.version` bumped to `"0.9.50"`
- **pyproject.toml** — `version` bumped to `"0.9.50"`

## Test Results

- `cargo build --release` — Passes (1 pre-existing warning: unused `prefix` in azure_client.rs)
- `./build_pyo3.sh` — Passes (Python wheel built)
- `tests/test_concurrent_runtime.py` — 16 threads × 200 objects × 3 rounds: ALL PASSED
- s3torchconnector compat: 9/9 functional tests passed (all returning BytesView)

## Files Modified (vs main)

```
modified:   Cargo.toml                              # version bump + dashmap dep
modified:   pyproject.toml                          # version bump
modified:   src/python_api/python_core_api.rs       # submit_io, put_many, removed GLOBAL_RUNTIME
modified:   src/python_api/python_aiml_api.rs       # run_on_global_rt for 2 functions
modified:   src/azure_client.rs                     # pre-existing: disabled list/list_stream (Azure SDK change)
modified:   python/s3dlio/compat/s3torchconnector.py # full rewrite
modified:   docs/PYTHON_API_GUIDE.md                # full rewrite for v0.9.50
modified:   docs/Changelog.md                       # v0.9.50 entry added
new:        docs/HANDOFF_bugs_0-9-40.md             # bug analysis doc
new:        docs/PYTHON_PERFORMANCE_FIX_PLAN.md     # original fix plan
new:        docs/PYTHON_RUNTIME_PERFORMANCE_BUG.md  # original bug report
new:        tests/test_concurrent_runtime.py        # 16-thread stress test
```

## Remaining Items

1. **Build running** — `cargo build --release` was in progress when session ended (version bump triggers recompile). Needs to finish, then `./build_pyo3.sh` to rebuild wheel.
2. **Git commit** — All changes are uncommitted. Suggested commit message:
   ```
   v0.9.50: Fix critical Python multi-threaded runtime bugs, rewrite s3torchconnector compat

   - Replace GLOBAL_RUNTIME.block_on() with io_uring-style submit_io() pattern
   - Fix "dispatch failure" (v0.9.27) and "Cannot start a runtime" (v0.9.40) panics
   - Add put_many()/put_many_async() batch upload functions
   - Rewrite s3torchconnector compat layer for zero-copy performance
   - New _BytesViewIO(io.RawIOBase) for seekable BytesView access
   - Rewrite PYTHON_API_GUIDE.md for v0.9.50
   - Bump version to 0.9.50
   ```
3. **azure_client.rs warning** — `unused variable: prefix` on line 209. Pre-existing, unrelated to this work. Proper Azure SDK fix needed separately.
4. **Real endpoint testing** — All tests used `file://` backend. Should validate against actual S3/MinIO before release.
