#!/bin/bash
# Build wheels for the current platform and Python versions.
#
# ── Platform detection ─────────────────────────────────────────────────────
ARCH=$(uname -m)   # x86_64 | aarch64 | arm64
OS=$(uname -s)     # Linux | Darwin

echo "Building for ${OS}/${ARCH}"

# ── Backend build profile + optional features ─────────────────────────────
# Positional arg #1 selects which storage backends are compiled into the wheel:
#   default (or slim): AWS + file/direct only (smaller/faster)
#   full: current behavior (AWS + Azure + GCS + file/direct)
#
# Examples:
#   ./build_pyo3.sh
#   ./build_pyo3.sh full
#   ./build_pyo3.sh default
#
# Pass extra features via EXTRA_FEATURES env var, e.g.:
#   EXTRA_FEATURES="numa,hdf5" ./build_pyo3.sh full
#
# Requirements:
#   numa  → sudo apt-get install libhwloc-dev  (or brew install hwloc)
#   hdf5  → sudo apt-get install libhdf5-dev   (or brew install hdf5)
FEATURES="extension-module"

PROFILE_ARG="default"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      echo "Usage: $0 [default|slim|full] [--profile|-p default|slim|full]"
      echo ""
      echo "Profiles:"
      echo "  default|slim  AWS + file/direct only (Azure/GCS excluded)"
      echo "  full          AWS + Azure + GCS + file/direct"
      echo ""
      echo "Examples:"
      echo "  $0"
      echo "  $0 full"
      echo "  $0 --profile full"
      echo "  $0 -p default"
      echo ""
      echo "Optional env vars:"
      echo "  EXTRA_FEATURES=\"numa,hdf5\""
      exit 0
      ;;
    -p|--profile)
      if [[ -z "${2:-}" ]]; then
        echo "ERROR: $1 requires a value (default|slim|full)"
        exit 1
      fi
      PROFILE_ARG="$2"
      shift 2
      ;;
    default|slim|full)
      PROFILE_ARG="$1"
      shift
      ;;
    *)
      echo "ERROR: Invalid argument '$1'"
      echo "       Use: default|slim|full or --profile|-p <profile>"
      echo "       Try: $0 --help"
      exit 1
      ;;
  esac
done

if [[ "$PROFILE_ARG" == "--help" || "$PROFILE_ARG" == "-h" ]]; then
    echo "Usage: $0 [default|slim|full]"
    echo ""
    echo "Profiles:"
    echo "  default|slim  AWS + file/direct only (Azure/GCS excluded)"
    echo "  full          AWS + Azure + GCS + file/direct"
    echo ""
    echo "Optional env vars:"
    echo "  EXTRA_FEATURES=\"numa,hdf5\""
    exit 0
fi

case "$PROFILE_ARG" in
  default|slim)
    echo "Backend profile: default (AWS + file/direct; Azure/GCS excluded)"
    ;;
  full)
    FEATURES="$FEATURES,full-backends"
    echo "Backend profile: full (AWS + Azure + GCS + file/direct)"
    ;;
  *)
    echo "ERROR: Invalid profile '$PROFILE_ARG'"
    echo "       Use: default|slim|full"
    echo "       Try: $0 --help"
    exit 1
    ;;
esac

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
