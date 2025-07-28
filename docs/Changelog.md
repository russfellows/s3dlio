## What’s new in v0.3.3 branch: feature/data-loader (Stage 1 – Core DataLoader)

In this first incremental release of our high-level data-loading API, we’ve added:

### 1. Core abstractions (`src/data_loader/`)

- **`Dataset` trait** (`dataset.rs`)  
  - `len() -> Option<usize>`, `get(index)`, and `as_stream()`  
  - Supports both map-style (random access) and iterable datasets  
- **`LoaderOptions`** (`options.rs`)  
  - Two tuning knobs: `batch_size` and `drop_last`  
- **`DataLoader`** (`dataloader.rs`)  
  - `.stream()` returns an `async_stream` of `Vec<Item>` batches  
  - Automatically handles:
    - Map-style datasets with known/unknown length  
    - Iterable-only streams  
    - Final incomplete batch drop if requested  

### 2. Crate-root exports (`src/lib.rs`)

- Publicly re-exported:
  - `Dataset`, `DatasetError`, `DynStream`
  - `LoaderOptions`
  - `DataLoader`

### 3. Baseline tests (`tests/test_dataloader.rs`)

- Map-style dataset batching  
- Drop-last semantics  
- Iterable-only datasets  
- Unknown-length datasets  
- Empty datasets  

### 4. Divergences / next steps

| Planned in Stage 1                                                | Status                                                |
|------------------------------------------------------------------|-------------------------------------------------------|
| First-class `Dataset` impls for RAW, NPZ, HDF5, TFRecord         | **Pending** (will add in Stage 2)                     |
| Python `pyo3` wrappers (`dataset.py` / `dataloader.py`)          | **Pending** (next release)                            |
| Smoke tests against a live MinIO bucket                          | Replaced by in-memory unit tests for fast iteration   |

Everything else in our Stage 1 blueprint is here and passing CI—so you now have a minimal, idiomatic Rust API for batching S3-backed data, ready to build on in Stage 2 (shuffling, parallel fetch, auto-tuning).  
