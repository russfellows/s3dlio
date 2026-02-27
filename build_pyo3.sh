#!/bin/bash
# Build wheels for the current platform and Python versions.
#
# ── Platform detection ─────────────────────────────────────────────────────
ARCH=$(uname -m)   # x86_64 | aarch64 | arm64
OS=$(uname -s)     # Linux | Darwin

echo "Building for ${OS}/${ARCH}"

# ── Optional features ─────────────────────────────────────────────────────
# Pass extra features via the EXTRA_FEATURES env var, e.g.:
#   EXTRA_FEATURES="numa,hdf5" ./build_pyo3.sh
#
# Requirements:
#   numa  → sudo apt-get install libhwloc-dev  (or brew install hwloc)
#   hdf5  → sudo apt-get install libhdf5-dev   (or brew install hdf5)
FEATURES="extension-module"
if [ -n "$EXTRA_FEATURES" ]; then
    FEATURES="$FEATURES,$EXTRA_FEATURES"
    echo "Extra Rust features enabled: $EXTRA_FEATURES"
fi

# ── Build for Python 3.12 and 3.13 on the native architecture ──────────────
maturin build --release --features "$FEATURES" \
  -i python3.12 \
  -i python3.13

echo ""
echo "✅ Wheels built for Python 3.12 and 3.13 (${OS}/${ARCH})"
echo "📦 Wheels location: target/wheels/"
echo ""
ls -lh target/wheels/*.whl 2>/dev/null || echo "No wheels found yet"

# ── Cross-compilation to Linux ARM64 (from Linux x86_64) ──────────────────
# Uncomment to produce manylinux aarch64 wheels using maturin's cross-compile
# support. Requires Docker and the maturin zig or cross toolchain:
#
#   pip install maturin[zig]   # easiest cross-compile via zig linker
#
# maturin build --release --features "$FEATURES" \
#   -i python3.12 \
#   -i python3.13 \
#   --target aarch64-unknown-linux-gnu \
#   --zig                        # use zig as cross-linker (no Docker needed)
#
# ── macOS universal2 wheel (x86_64 + arm64 in one fat binary) ─────────────
# Run this on a macOS host (works on both Intel and Apple Silicon):
#
# maturin build --release --features "$FEATURES" \
#   -i python3.12 \
#   -i python3.13 \
#   --target universal2-apple-darwin
