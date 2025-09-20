#!/bin/bash
# scripts/build_runtime_selection.sh
#
# Build s3dlio CLI with BOTH backends for runtime selection

set -e

echo "🔄 BUILDING S3DLIO WITH RUNTIME BACKEND SELECTION"
echo "================================================="

# Check if we can enable both backends by modifying feature constraints
echo "🔧 Modifying Cargo.toml for dual-backend support..."

# Backup original Cargo.toml
cp Cargo.toml Cargo.toml.backup

# Create a modified version that allows both backends
cat > Cargo.toml.dual-backend << 'EOF'
[package]
name = "s3dlio"
version = "0.7.10"
edition = "2024"

[lib]
name = "s3dlio"
crate-type = ["cdylib", "rlib"]      # cdylib → Python wheel, rlib → normal Rust

[package.metadata.maturin]
features = ["extension-module"]

[[bin]]
name = "s3-cli"
path = "src/bin/cli.rs"

[dependencies]
aws-config         = "^1"
aws-sdk-s3         = "^1"

# Azure SDK crates (keep minor versions flexible; they evolve together)
azure_core = { version = "0.27.0" }          # Response/Body, paging, errors
azure_identity = { version = "0.27.0" }      # DefaultAzureCredential
azure_storage_blob = { version = "0.4.0" }   # Blob/Container/BlockBlob clients

# Core dependencies for Rust and Python API
anyhow    = "^1"
tokio     = { version = "^1", features = ["full"] }
pyo3 = { version = "^0.25", features = ["extension-module"], optional = true }
pyo3-async-runtimes = { version = "^0.25", optional = true }

# For mypy stub generation
pyo3-stub-gen = { version = "^0.6", optional = true }

serde = { version = "^1", features = ["derive"] }
serde_json = "^1"
bytes = "^1"
flate2 = "^1"
zstd-safe = "^7"
futures-util = "^0.3"
clap = { version = "^4", features = ["derive"] }
log = "^0.4"
env_logger = "^0.11"

# Our features
dotenvy = "^0.15"
rand = { version = "^0.9", features = ["getrandom"] }

# base64  
base64 = "^0.22"

# Rust PS1
tfrecord = { version = "^0.15", features = ["async-tokio"] }

# ML metadata formats - NumPy/HDF5 support
ndarray = "^0.16"
# Disable  openssl-sys = { version = "^0.9", features = ["vendored"] }
hdf5-metno = "^0.10" # Rust's HDF5 bindings via "metno" vendor
ndarray-npy = { version = "^0.8", features = ["npyz"] } # NumPy .npy / .npz

# For thread graph topology
petgraph = "^0.6"

# Futures utils and iterators
futures-core  = "^0.3"
itertools     = "^0.12"

# Gear framework for advanced data structure operations
gstd         = "^1.9"
gear-core-errors = "^1.9"
gcore        = "^1.9"
gprimitives  = "^1.9"
scale-info   = { version = "^2.11", default-features = false, features = ["derive"] }
parity-scale-codec = { version = "^3.7", default-features = false, features = ["derive"] }

# Data compression and utility
io          = "^0.0"
crc32fast   = "^1.4"
glob        = "^0.3"

async-trait  = "^0.1"
async-stream = "^0.3"
futures-core = "^0.3"
futures-util = "^0.3"

zstd         = "^0.13"
humantime    = "^2.2"
tokio-sync   = "^0.1.8"
base64 = "0.22.1"

# --- Arrow backend (optional, enabled via "arrow-backend" feature) -------
object_store = { version = "0.12", default-features = false, features = ["aws", "azure", "http", "gcp"], optional = true }
url = { version = "2.5", optional = true }

# --- Profiling (optional, enabled via "profiling" feature) ---------------
tracing           = { version = "0.1", optional = true }
tracing-subscriber = { version = "0.3", features = ["fmt", "registry", "env-filter"], optional = true }
console-subscriber = { version = "0.4", optional = true }
pprof = { version = "0.13", features = ["flamegraph", "protobuf-codec"], optional = true }

# --- HTTP2 client -----------
hyper        = { version = "1", features = ["client", "http2"] }
hyper-util   = { version = "0.1", features = ["http2", "client", "client-legacy"] }

# --- WebAssembly WASM runtime ----
aws-lc-rs = { version = "^1", features = ["aws-lc-sys"], default-features = false }

[build-dependencies]
pyo3-build-config = "^0.25"

[features]
#default = []
# MODIFIED: Enable both backends for runtime selection
default = ["s3", "native-backends", "arrow-backend"]
s3 = []
native-backends = []                    # Current AWS/Azure implementations  
arrow-backend = ["dep:object_store", "dep:url"]  # Apache Arrow object_store backend
profiling = ["dep:tracing", "dep:tracing-subscriber", "dep:console-subscriber", "dep:pprof"]
extension-module = ["dep:pyo3", "dep:pyo3-async-runtimes"]     # suppresses the cfg warning when building the wheel

[patch.crates-io]
# Use our patched version to expose hyper_builder methods for HTTP connection pool configuration
aws-smithy-http-client = { path = "fork-patches/aws-smithy-http-client" }

[[example]]
name = "s3_backend_comparison"
path = "examples/s3_backend_comparison.rs"

[[bench]]
name = "performance_microbenchmarks"
harness = false

[[bench]]
name = "simple_test"
harness = false
EOF

echo "🚀 Attempting to build with both backends..."
# This will likely fail due to our compile_error! but let's try
cp Cargo.toml.dual-backend Cargo.toml

if cargo build --release --bin s3-cli 2>&1 | tee build.log; then
    echo "✅ SUCCESS: Dual-backend build completed!"
    mkdir -p target/performance_variants
    cp target/release/s3-cli target/performance_variants/s3-cli-dual-backend
    echo "✅ Dual-backend CLI created: target/performance_variants/s3-cli-dual-backend"
else
    echo "❌ EXPECTED: Dual-backend build failed due to compile_error!"
    echo "   This requires modifying the library to support runtime selection."
    echo "   Restoring original Cargo.toml..."
    cp Cargo.toml.backup Cargo.toml
    
    echo ""
    echo "🔧 TO ENABLE RUNTIME SELECTION:"
    echo "   1. Remove compile_error! from src/lib.rs"
    echo "   2. Modify backend selection to use environment variables"
    echo "   3. Create conditional store_for_uri() that chooses backend"
    echo ""
    echo "   This is a more advanced feature that would be great for v0.7.11!"
fi

# Restore original
cp Cargo.toml.backup Cargo.toml
rm -f Cargo.toml.backup Cargo.toml.dual-backend

echo ""
echo "📊 CURRENT AVAILABLE VARIANTS:"
echo "=============================="
ls -la target/performance_variants/ 2>/dev/null || echo "No variants built yet - run build_performance_variants.sh first"