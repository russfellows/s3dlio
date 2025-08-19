## What's new in v0.5.2 - Multi-Backend CLI Support

Added comprehensive multi-backend CLI support with a new unified `ls` command that works across all storage backends. This release provides a seamless CLI experience while maintaining full backward compatibility with existing S3-specific commands.

### 🎯 Key Features

- **Unified CLI Interface**: New generic `ls` command supporting all storage backends via URI schemes
- **Enhanced URI Scheme Support**: Full support for `file://`, `direct://`, `s3://`, and `az://` schemes
- **Automatic Backend Routing**: CLI automatically selects the appropriate backend based on URI scheme
- **Backward Compatibility**: All existing S3-specific commands continue to work unchanged
- **Code Quality Improvements**: Added centralized constants module for better maintainability

### 🔧 Technical Enhancements

- **src/bin/cli.rs**: Added `GenericList` command and `generic_list_cmd()` async function
- **src/file_store_direct.rs**: Updated URI validation to accept both `file://` and `direct://` schemes
- **src/constants.rs**: New centralized constants module replacing hardcoded values
- **Enhanced Error Handling**: Improved error messages for unsupported URI schemes

### 📚 Documentation

- **docs/BACKEND_IMPROVEMENT_PLAN.md**: Added comprehensive backend improvement planning documentation
- **Migration Path**: Clear upgrade path from S3-specific to generic commands

### 🧪 Testing & Validation

- All existing tests continue to pass
- New URI scheme validation ensures robust scheme handling
- Release build validation confirms production readiness

This release represents a significant step toward true multi-backend transparency, where users can seamlessly work with different storage systems using the same CLI interface.

---

## What's new in v0.3.3 branch: feature/data-loader (Stage 1 – Core DataLoader)# What’s new in v0.3.3 branch: feature/data-loader (Stage 1 – Core DataLoader)

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
