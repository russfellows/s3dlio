# Data Generation Migration: s3dlio → dgen-data

**Date**: April 17, 2026  
**Status**: Analysis only — not yet implemented  
**Goal**: Replace `src/data_gen.rs` and `src/data_gen_alt.rs` with the `dgen-data` crate

---

## Background

`dgen-data` v0.2.3 (`/home/eval/Documents/Code/dgen-rs/`) is a standalone Rust crate that was
originally **ported from** s3dlio's `data_gen_alt.rs`. The same author owns both. sai3-bench
already uses `dgen-data` for all data generation via its `data_gen_pool.rs` module.

The goal is to remove the duplicated RNG core from s3dlio and make it an s3dlio consumer of
dgen-data, exactly as sai3-bench already does.

---

## What dgen-data Currently Provides

The public API in `dgen-data`'s `generator.rs` and `rolling_pool.rs`:

| Item | Module | Present in dgen-data |
|---|---|---|
| `DataBuffer` | `generator` | ✅ |
| `NumaMode` | `generator` | ✅ |
| `GeneratorConfig` | `generator` | ✅ |
| `generate_data_simple()` | `generator` | ✅ |
| `generate_data()` | `generator` | ✅ |
| `DataGenerator` | `generator` | ✅ |
| `RollingPool` | `rolling_pool` | ✅ (new — not in s3dlio at all) |
| `total_cpus()` | — | ❌ missing |
| `default_data_gen_threads()` | — | ❌ missing |
| `optimal_chunk_size()` | — | ❌ missing |
| `generate_data_with_config()` | — | ❌ missing |
| `generate_controlled_data_alt()` | — | ❌ missing |
| `ObjectGenAlt` | — | ❌ missing |
| `generate_controlled_data_streaming_alt()` | — | ❌ missing |

---

## Who Calls What in s3dlio

### `data_gen_alt.rs` callers (outside the data_gen files themselves)

| Caller | What it uses |
|---|---|
| `src/python_api/python_datagen_api.rs` | `DataBuffer`, `DataGenerator`, `GeneratorConfig`, `NumaMode`, `default_data_gen_threads`, `total_cpus` |
| `src/api/advanced.rs` | re-exports `generate_controlled_data_alt` as public API |
| `src/data_gen.rs` | delegates `generate_controlled_data` (deprecated) and single-pass path to `generate_controlled_data_alt` |

### `data_gen.rs` callers

| Caller | What it uses |
|---|---|
| `src/s3_utils.rs` | `generate_object()` |
| `src/bin/cli.rs` | `generate_object()` |
| `src/streaming_writer.rs` | `DataGenerator` + `ObjectGen` (s3dlio's streaming types) |
| `src/lib.rs` | re-exports `fill_controlled_data()` as stable public API |

---

## What Would Stay in s3dlio Regardless

`data_gen.rs` contains **format-specific logic** that belongs in s3dlio permanently:

- `generate_object(&Config) -> Bytes` — config-driven entry point, ties dedup/compress/format together
- `build_npz()`, `build_tfrecord()`, `build_hdf5()`, `build_raw()` — format builders
- `fill_controlled_data()` — stable public API (thin wrapper, can delegate to dgen internals)
- `DataGenerator` / `ObjectGen` streaming types — used by `streaming_writer.rs`

Net: `data_gen.rs` shrinks from ~926 lines to ~200 lines of format glue.  
Net: `data_gen_alt.rs` (~1514 lines) is deleted entirely.  
**Total removal: ~2,200 of the 2,440 lines** across the two files.

---

## The 6 Missing Items: Porting Effort Assessment

All 6 are pure thin wrappers over `DataGenerator` and `generate_data()` — they use no
s3dlio-specific types and contain no novel logic.

| Item | Lines in data_gen_alt.rs | Complexity | Notes |
|---|---|---|---|
| `total_cpus()` | 2 | **Trivial** | `num_cpus::get()` — already called internally in dgen-rs |
| `default_data_gen_threads()` | 2 | **Trivial** | Same as above |
| `optimal_chunk_size()` | 8 | **Trivial** | Pure math, no deps |
| `generate_data_with_config()` | ~20 | **Easy** | Calls `generate_data()` + `DataGenerator`, both already in dgen-rs |
| `generate_controlled_data_alt()` | ~10 | **Easy** | Wraps `generate_data_with_config()` |
| `ObjectGenAlt` + `generate_controlled_data_streaming_alt()` | ~80 | **Easy** | Thin constructor wrapper around `DataGenerator`; dgen-rs comment even says "like ObjectGenAlt from s3dlio" |

**Total work: ~125 lines**, all direct transplants from `data_gen_alt.rs`.  
Target file: append to bottom of `dgen-rs/src/generator.rs`.

### Note on `ObjectGenAlt`

`DataGenerator` in dgen-rs is already API-identical to `ObjectGenAlt`
(`new(config)`, `fill_chunk()`, `is_complete()`, `total_size()`, `position()`, `reset()`,
`set_seed()`). `ObjectGenAlt` is just a convenience constructor that accepts
`(size, dedup, compress)` directly instead of a `GeneratorConfig`. Two options:

- **Add it as a newtype/wrapper** — keeps s3dlio migration a pure search-and-replace of `use`
  statements, no call site changes needed.
- **Skip it** — update the handful of s3dlio call sites to construct `GeneratorConfig` directly.
  Slightly more churn but cleaner long-term.

---

## Type Identity Note

The Python bindings (`python_datagen_api.rs`) use `DataBuffer` and `DataGenerator` as Rust types
directly inside `#[pyclass]` structs. Once those come from `dgen_data::DataBuffer` instead of
`crate::data_gen_alt::DataBuffer`, the types change crate origin. This is **not a problem** — the
`#[pyclass]` wrappers are already owned by s3dlio; updating the inner types is a search-and-replace
of the `use` statements in `python_datagen_api.rs` only.

---

## Recommended Migration Steps

1. **Add the 6 missing items to `dgen-rs/src/generator.rs`** — ~125 lines, all transplants.
2. **Bump dgen-data version** from `0.2.3` → `0.2.4` in `dgen-rs/Cargo.toml`.
3. **Add `dgen-data` dependency to `s3dlio/Cargo.toml`**:
   ```toml
   dgen-data = { version = "0.2.4", default-features = false, features = ["thread-pinning"] }
   ```
   (Same flags as sai3-bench.)
4. **Delete `src/data_gen_alt.rs`** — replace all imports with `dgen_data::` equivalents in
   `python_datagen_api.rs`, `data_gen.rs`, `api/advanced.rs`, and `lib.rs`.
5. **Gut `src/data_gen.rs`** — keep format builders, `generate_object()`,
   `fill_controlled_data()`, and `DataGenerator`/`ObjectGen` wrappers; remove all the duplicated
   RNG generation logic (it now lives in dgen-data).
6. **Run `cargo build` + `cargo test`** — should pass with zero warnings.

---

## Issue #136 Status (4 KB Minimum Bug)

**Confirmed fixed** as of commit `0b73492` ("fix(data-gen): Critical compression bug fixes for
small objects (<1 MiB)", 2026-01-31). The old `BLK_SIZE` minimum enforcement was only in
`generate_controlled_data_original()`, which is now commented out / deprecated. The `_alt`
algorithm (and its `data_gen_alt.rs` equivalent) never had the minimum.

Verified April 18, 2026 by running `examples/test_size_136.rs` across all code paths
(use_controlled=false, use_controlled=true dedup, use_controlled=true compress):

```
requested=1    actual=1    MATCH=true
requested=10   actual=10   MATCH=true
requested=100  actual=100  MATCH=true
requested=512  actual=512  MATCH=true
requested=1024 actual=1024 MATCH=true
requested=2048 actual=2048 MATCH=true
requested=3000 actual=3000 MATCH=true
requested=4096 actual=4096 MATCH=true
```

The issue predates v0.9.37 and does not need to be fixed in the current branch.
Note: NPZ and TFRecord formats intentionally produce slightly larger output than the raw payload
due to format framing headers — that is expected behavior, not a bug.

---

## How sai3-bench Already Does This (Reference)

`sai3-bench/src/data_gen_pool.rs` is the template:

```rust
use dgen_data::generate_data_simple;
use dgen_data::RollingPool;
pub use dgen_data::constants::BLOCK_SIZE as POOL_BLOCK_SIZE;
```

`sai3-bench/Cargo.toml`:
```toml
dgen-data = { version = "0.2.3", default-features = false, features = ["thread-pinning"] }
```

The migration for s3dlio follows the same pattern, just with more call sites.
