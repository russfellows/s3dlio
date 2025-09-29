# s3dlio AI Coding Agent Instructions

## Project Overview
s3dlio is a high-performance, multi-protocol storage library built in Rust with Python bindings, designed for AI/ML workloads. It provides universal copy operations across S3, Azure, local file systems, and DirectIO with near line-speed performance.

### Performance Targets
- **Read (GET)**: Minimum 5 GB/s (50 Gb/s) sustained, target higher
- **Write (PUT)**: Minimum 2.5 GB/s (25 Gb/s) sustained, target higher
- **Infrastructure**: Tested against Vast storage systems with bonded 100 Gb ports
- **S3 Compatibility**: MinIO, Vast, AWS S3, and other S3-compatible storage systems

## Core Architecture

### Dual-Backend System
The project uses **mutually exclusive backend selection** at compile-time:
- `native-backends` feature: AWS SDK + Azure SDK (default)
- `arrow-backend` feature: Apache Arrow object_store implementation

```bash
# Build commands
cargo build --no-default-features --features native-backends
cargo build --no-default-features --features arrow-backend
```

**Critical**: These features are mutually exclusive by design (`compile_error!` in `src/lib.rs`).

### Public API Structure
- **Stable API**: `src/api.rs` - External developers use this via `s3dlio::api`
- **Internal modules**: Everything else may change - mark as implementation details
- **Factory pattern**: `store_for_uri()` creates appropriate backend for any URI scheme

### Python Integration (PyO3/Maturin)
- **Module structure**: `python/s3dlio/` wraps compiled Rust extension `_pymod`
- **Build process**: `./build_pyo3.sh` → `./install_pyo3_wheel.sh`
- **Critical testing rule**: Always test installed package, never development `python/` directory

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

### Python Extension Development
```bash
# REQUIRED workflow after any Rust changes
./build_pyo3.sh && ./install_pyo3_wheel.sh
python tests/test_functionality.py  # Tests installed package
```

**Never** use `sys.path` manipulation in tests - it imports development Python without compiled Rust.

### UV Package Manager
Project uses UV (not pip) for Python package management:
```bash
# Activate UV environment
source .venv/bin/activate  # Check for (s3dlio) prefix in prompt

# Install packages  
uv pip install package_name  # NOT pip install

# Virtual environment status
# In environment: prompt shows (s3dlio) prefix
# Outside environment: no prefix shown
```

### Backend Development
```bash
# Performance comparison between backends
./scripts/build_performance_variants.sh
./scripts/run_backend_comparison.sh
```

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
- **Current version**: v0.8.4 (check `Cargo.toml` and `pyproject.toml`)
- **Patch releases**: Increment build number (v0.8.4 → v0.8.5)
- **Minor releases**: For major features (v0.8.x → v0.9.0)
- **Major version 0.x**: Until production-ready quality achieved
- **Documentation**: Update `docs/Changelog.md` and `README.md` for every release

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