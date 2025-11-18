# s3dlio AI Coding Agent Instructions

## Project Overview
s3dlio is a high-performance, multi-protocol storage library built in Rust with Python bindings, designed for AI/ML workloads. It provides universal copy operations across S3, Azure, local file systems, and DirectIO with near line-speed performance.

**Current Version**: v0.9.5 (October 2025)

### Performance Targets
- **Read (GET)**: Minimum 5 GB/s (50 Gb/s) sustained, target higher
- **Write (PUT)**: Minimum 2.5 GB/s (25 Gb/s) sustained, target higher
- **Infrastructure**: Tested against Vast storage systems with bonded 100 Gb ports
- **S3 Compatibility**: MinIO, Vast, AWS S3, and other S3-compatible storage systems

## Core Architecture

### Dual-Backend System
The project uses **mutually exclusive backend selection** at compile-time:
- `native-backends` feature: AWS SDK + Azure SDK (**DEFAULT** - recommended for production)
- `arrow-backend` feature: Apache Arrow object_store implementation (experimental, optional)

```bash
# Default build (uses native-backends)
cargo build --release

# Explicit native-backends (RECOMMENDED)
cargo build --no-default-features --features native-backends

# Experimental arrow backend (NOT RECOMMENDED for production)
cargo build --no-default-features --features arrow-backend
```

**Critical**: These features are mutually exclusive by design (`compile_error!` in `src/lib.rs`).

**Backend Status**:
- **native-backends**: Default, proven performance (5+ GB/s reads, 2.5+ GB/s writes), production-ready
- **arrow-backend**: Experimental only, no proven performance benefit, kept for comparison testing

### Build Quality Standards

**CRITICAL: Zero Warnings Policy**
- ALL builds MUST be warning-free before commits
- Never use quick fixes like `_` prefix to silence unused variable warnings
- Unused variables often indicate logic errors that must be investigated
- Unused imports must be removed, not ignored

**Pre-Commit Checklist**:
1. Run `cargo build --release` and verify ZERO warnings
2. Run `cargo clippy` and fix all issues
3. Investigate root cause of any warning - do not suppress without understanding
4. If unsure about a warning, ask for clarification before committing

**Shell Command Best Practices**:
- Never use exclamation marks (`!`) in Python print statements or shell commands
- Exclamation marks cause shell escaping issues in bash
- Use simple declarative messages instead: "Import successful" not "Import successful!"

**Warning Investigation Process**:
```bash
# Check for warnings
cargo build --release 2>&1 | grep -i warning

# Get full details
cargo build --release 2>&1 | grep -A 10 warning

# For clippy suggestions
cargo clippy --all-targets --all-features
```

**Common Warning Anti-Patterns** (DO NOT DO):
- ❌ Adding `_` prefix to silence unused variable warnings
- ❌ Using `#[allow(unused)]` without understanding why
- ❌ Importing modules "just in case" they might be needed
- ❌ Leaving debug code that uses variables only in certain configs

**Correct Approach**:
- ✅ Remove unused imports completely
- ✅ Investigate why variables aren't used (logic bug?)
- ✅ Use feature gates if code is conditionally compiled
- ✅ Refactor to eliminate the warning's root cause

### Dependency Management

**aws-smithy-http-client Patches: REMOVED**
- Custom patches in `fork-patches/aws-smithy-http-client/` are NOT used by default
- Patches showed no measurable performance benefit
- Removed from `[patch.crates-io]` to avoid forcing downstream users to patch
- Fork preserved for reference/experimentation but not required for builds

### Public API Structure
- **Stable API**: `src/api.rs` - External developers use this via `s3dlio::api`
- **Internal modules**: Everything else may change - mark as implementation details
- **Factory pattern**: `store_for_uri()` creates appropriate backend for any URI scheme

### Python Integration (PyO3/Maturin)
- **Module structure**: `python/s3dlio/` wraps compiled Rust extension `_pymod`
- **Build process**: `./build_pyo3.sh` → `./install_pyo3_wheel.sh`
- **Critical testing rule**: Always test installed package, never development `python/` directory

**CRITICAL: Virtual Environment Check**
- **ALWAYS verify virtual environment is active** before any build/install commands
- Check for `(s3dlio)` prefix in terminal prompt

### Data Generation Algorithm Migration (November 2025)

**Status**: New algorithm active via redirection in `src/data_gen.rs`

**Background**: 
- Original algorithm had cross-block compression bug (compress=1 still gave 7.68:1 ratio)
- New algorithm (`data_gen_alt.rs`) fixes bug using per-block RNG with local back-references
- Performance optimized with Xoshiro256++ (replaces ChaCha20 for 5-10x speedup)
- All existing code now uses new algorithm via transparent redirection

**Temporary Code Preservation**:
The following functions in `src/data_gen.rs` are **COMMENTED OUT** and marked for removal:
- `generate_controlled_data_original()` - lines ~206-250 (old single-pass generator)
- `ObjectGen::new_original()` - lines ~488-533 (old ObjectGen constructor)
- `ObjectGen::fill_chunk_original()` and related methods - lines ~594-710 (old streaming implementation)

**Action Required** (target: December 2025):
1. Run extended validation tests (1 week of production workloads)
2. Verify all downstream projects (sai3-bench, dl-driver) working correctly
3. Remove commented-out code from `src/data_gen.rs`
4. Update documentation to reference only `data_gen_alt.rs`
5. Consider promoting `data_gen_alt.rs` to primary `data_gen.rs` (rename)

**GitHub Issue**: See `.github/ISSUE_TEMPLATE/data_gen_migration.md` for full tracking issue template

**Testing Checklist Before Removal**:
- [ ] All s3dlio tests pass (currently: 162/162 ✓)
- [ ] sai3-bench runs successfully with various compress/dedup settings
- [ ] dl-driver checkpoint save/load works correctly
- [ ] Performance benchmarks show no regression (<5% variance acceptable)
- [ ] Compression ratios match specifications (compress=1 → ratio ~1.0)
- [ ] No user-reported issues with data generation in production

### Future Enhancements - NPY Format Support

**NEEDS IMPLEMENTATION**: Zero-copy in-memory .npy serialization (November 2025)

**Current State**:
- s3dlio has `src/data_formats/npz.rs` (reads NPZ files)
- s3dlio has `src/data_formats/hdf5.rs`, `tfrecord.rs` (format support exists)
- dl-driver implemented custom zero-copy .npy serializer (November 2025)
- ndarray-npy 0.9+ only writes to file paths, not in-memory buffers

**What Needs to be Added**:
- In-memory .npy serialization function (NPY 1.0 format)
- Zero-copy implementation using `ndarray::as_slice_memory_order()`
- Integration with existing `src/data_formats/` module
- Python bindings via PyO3 for numpy interop

**Reference Implementation**:
- See `dl-driver/crates/formats/src/npz.rs::NpzFormat::array_to_npy_bytes()`
- 48 lines, implements NPY 1.0: magic (6B) + version (2B) + header_len (2B) + header (padded dict) + data
- Zero-copy when ndarray is contiguous, one-copy fallback otherwise
- No temp files, pre-allocated buffers

**Why This Belongs in s3dlio**:
- s3dlio is explicitly an AI/ML I/O library
- Already has format support (HDF5, TFRecord, NPZ reading)
- Would enable direct array → storage without intermediate files
- Python bindings would benefit numpy/torch users
- Reusable across dl-driver, sai3-bench, and other tools

**Refactoring Plan**:
1. Add `array_to_npy_bytes()` to s3dlio `src/data_formats/npz.rs`
2. Expose via public API and Python bindings
3. Update dl-driver to use s3dlio implementation
4. Remove duplicate code from dl-driver

**GitHub Issue**: See future enhancement tracking issue (to be filed)
- If not active, run: `source .venv/bin/activate`
- Terminal interrupts (Ctrl-C) may exit the virtual environment
- Re-activate before continuing work

## Key Development Patterns

### URI-based Universal Interface
All backends use consistent URI schemes:
```
s3://bucket/prefix/        # S3 operations
az://container/prefix/     # Azure Blob Storage  
file:///local/path/        # Local filesystem
direct:///local/path/      # DirectIO bypass
```

### Feature-gated Development
When modifying backends, always use feature gates:
```rust
#[cfg(feature = "native-backends")]
// AWS SDK implementation

#[cfg(feature = "arrow-backend")]  
// Arrow object_store implementation
```

### Testing Strategy
- **Backend comparison**: Use `scripts/run_backend_comparison.sh`
- **Python testing**: Must rebuild/reinstall after Rust changes
- **Environment**: Tests require `.env` file with S3 credentials

## Critical Development Workflows

**CRITICAL: Always Check Virtual Environment Before Building**
```bash
# STEP 1: ALWAYS verify virtual environment is active
# Look for (s3dlio) prefix in prompt
# If missing, activate:
source .venv/bin/activate

# STEP 2: Then proceed with builds
cargo build --release
# or
./build_pyo3.sh
```

### Python Extension Development
```bash
# REQUIRED workflow after any Rust changes
# FIRST: Ensure virtual environment is active
source .venv/bin/activate  # If not already active

# Then build and install
./build_pyo3.sh && ./install_pyo3_wheel.sh
python tests/test_functionality.py  # Tests installed package
```

**Never** use `sys.path` manipulation in tests - it imports development Python without compiled Rust.

### UV Package Manager
Project uses UV (not pip) for Python package management:
```bash
# CRITICAL: Always check if you're in the virtual environment
# Virtual environment status: prompt shows (s3dlio) prefix when active

# Activate UV environment (if not already active)
source .venv/bin/activate

# Install packages  
uv pip install package_name  # NOT pip install

# Run Python commands
python -c "import s3dlio; print('Import successful')"  # Works when venv active

# Deactivate when done
deactivate
```

**Important Notes:**
- Terminal interrupts (Ctrl-C) may exit the virtual environment
- Always verify `(s3dlio)` prefix appears in prompt before running Python commands
- If no prefix shown, run `source .venv/bin/activate` to re-enter environment
- Never use exclamation marks in Python print statements (shell escaping issues)

### Backend Development
```bash
# Performance comparison between backends
./scripts/build_performance_variants.sh
./scripts/run_backend_comparison.sh
```

### Search Tools
- **ripgrep (rg)**: Fast code search available in terminal
  ```bash
  # Search for pattern across all files
  rg "pattern" 
  
  # Search in specific file types
  rg "pattern" --type rust
  
  # Case-insensitive search
  rg -i "pattern"
  ```
- **grep_search tool**: Use for exact string or regex searches within files
- **semantic_search tool**: Use for semantic/natural language code searches

### Environment Configuration
Key variables for development/testing:
- `AWS_ENDPOINT_URL`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- `S3DLIO_BACKEND=native|arrow` - Runtime backend selection
- `S3DLIO_USE_OPTIMIZED_HTTP=true` - Performance optimization

### S3 Infrastructure Support
- **Local S3**: MinIO, Vast storage systems with bonded 100 Gb ports
- **Cloud S3**: AWS S3 with local and cloud deployments
- **Multi-target scaling**: Currently handled at process level (different instances target different IP addresses)
- **Future enhancement**: Multi-target addressing within s3dlio for >100 Gb throughput

## Project-Specific Conventions

### Versioning & Releases
- **Current version**: v0.9.5 (check `Cargo.toml` and `pyproject.toml`)
- **Next version**: v0.9.6 (in development on v0.9.6-dev branch)
- **Patch releases**: Increment build number (v0.9.5 → v0.9.6)
- **Minor releases**: For major features (v0.9.x → v0.10.0)
- **Major version 0.x**: Until production-ready quality achieved
- **Documentation**: Update `docs/Changelog.md` and `README.md` for every release

### Logging Framework (v0.8.8+)
- **Framework**: Uses `tracing` crate (not `log`) for observability
- **Dependencies**: `tracing ^0.1`, `tracing-subscriber ^0.3`, `tracing-log ^0.2`
- **Verbosity levels**: 
  - Default: WARN level (quiet)
  - `-v`: INFO level
  - `-vv`: DEBUG level
- **Trace logging**: Operation trace (--op-log) uses separate zstd-compressed TSV format
- **Compatibility**: dl-driver and s3-bench (io-bench) also use tracing
- **Usage**: `tracing::info!()`, `tracing::debug!()`, `tracing::warn!()`, `tracing::error!()`

### Page Cache Optimization (v0.8.8+)
- **Module**: `src/page_cache.rs` - Linux/Unix posix_fadvise() wrapper
- **PageCacheMode**: Sequential, Random, DontNeed, Normal, Auto
- **Auto mode**: Sequential for files ≥64MB, Random for smaller files
- **Integration**: file_store.rs get() and get_range() operations
- **Platform**: Linux/Unix only (no-op on Windows)

### RangeEngine Performance (v0.9.3+, Updated v0.9.6)
- **Multi-backend support**: S3, Azure Blob Storage, Google Cloud Storage, file://, direct://
- **Default status**: **DISABLED** by default as of v0.9.6 (was: enabled in v0.9.3-v0.9.5)
- **Reason for change**: Stat overhead causes up to 50% slowdown on typical workloads
- **Default threshold**: 16 MiB (when explicitly enabled)
- **Configuration**: `DEFAULT_RANGE_ENGINE_THRESHOLD` in `src/constants.rs`
- **Performance gains**: 30-50% throughput improvement on large files (>= 64MB) **when enabled**
- **Must opt-in**: Set `enable_range_engine: true` in backend config for large-file workloads
- **When to enable**: Large-file workloads (>= 64 MiB average), high-bandwidth/high-latency networks
- **When to keep disabled**: Mixed workloads, small objects, local file systems, benchmarks

### Delete Performance (v0.9.5+)
- **Adaptive concurrency**: 10-70x faster delete operations
- **Algorithm**: Scales with workload (10% of total objects, capped at 1,000)
- **Progress tracking**: Batched updates every 50 operations (98% reduction in overhead)
- **Universal support**: Works across all backends (S3, Azure, GCS, file://, direct://)

### Error Handling
- Use `anyhow::Result` for all public APIs
- Convert to `PyResult` at Python boundary using `map_err(py_err)`

### Performance Patterns
- **Streaming operations**: Use `ObjectWriter` trait for large uploads
- **Zero-copy**: Prefer `write_owned_bytes()` over `write_chunk()`
- **Concurrency**: Default 16 PUT / 32 GET concurrent operations

### Module Organization
- **Core storage**: `src/object_store*.rs` files implement backend traits
- **Python API**: Split into `python_core_api.rs`, `python_aiml_api.rs`, etc.
- **Configuration**: `src/config.rs` defines all tunable parameters

## Build System Gotchas

### System Dependencies
- **HDF5**: Required for HDF5 format support
  ```bash
  # Ubuntu/Debian
  sudo apt update && sudo apt install -y libhdf5-dev
  # macOS: brew install hdf5
  # RHEL/CentOS/Fedora: sudo dnf install hdf5-devel
  ```

### PyO3 Extension Module
- **Feature flag**: `extension-module` required for Python builds only
- **Maturin config**: Uses `python-source = "python"` and `module-name = "s3dlio._pymod"`
- **Installation**: Wheel goes to `.venv/lib/python3.12/site-packages/s3dlio/`

### Performance Variants
The project builds multiple CLI variants for benchmarking:
- `target/performance_variants/s3-cli-native`
- `target/performance_variants/s3-cli-arrow`

### Dependency Patches
- Uses forked `aws-smithy-http-client` for connection pool optimization
- Patch applied via `[patch.crates-io]` in `Cargo.toml`

## Common Tasks

### Adding New Storage Backend
1. Implement `ObjectStore` trait in new `src/object_store_<backend>.rs`
2. Add feature flag in `Cargo.toml`
3. Update `store_for_uri()` factory function
4. Add to `scripts/run_backend_comparison.sh`

### Performance Investigation
```bash
# Build with profiling
cargo build --release --features profiling
# Run flamegraph analysis  
cargo run --example simple_flamegraph_test --features profiling
```

### Testing New Python Features
```bash
# Full test workflow
cargo test --release --lib                    # Rust tests
./build_pyo3.sh && ./install_pyo3_wheel.sh   # Build Python
python python/tests/test_modular_api_regression.py  # Python tests
```

### Common Development Commands
```bash
# Backend performance comparison
./scripts/build_performance_variants.sh
./scripts/run_backend_comparison.sh

# Full test suite with S3 credentials
./scripts/test_all.sh

# Profile performance with flamegraph
cargo build --release --features profiling
cargo run --example simple_flamegraph_test --features profiling
```