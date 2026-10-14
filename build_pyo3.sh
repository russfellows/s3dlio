#!/bin/bash
# Build wheels for the current platform and Python versions.
#
# Fails loudly (issue #188): any error stops the script with a non-zero exit
# status. It never reports success unless maturin ran and left a fresh
# cp312 and cp313 wheel behind; wheels already in target/wheels/ from an
# earlier build are never listed as results.
set -euo pipefail

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

if [ -n "${EXTRA_FEATURES:-}" ]; then
    FEATURES="$FEATURES,$EXTRA_FEATURES"
    echo "Extra Rust features enabled: $EXTRA_FEATURES"
fi

# ── Preflight: the tools the build needs must exist ────────────────────────
if ! command -v maturin >/dev/null 2>&1; then
  echo "ERROR: maturin not found on PATH; no wheels were built." >&2
  echo "       Install it (pip install maturin, or uv tool install maturin)," >&2
  echo "       or run: uv run --no-project --with maturin --with patchelf $0" >&2
  exit 1
fi

for PY in python3.12 python3.13; do
  if ! command -v "$PY" >/dev/null 2>&1; then
    echo "ERROR: $PY not found on PATH; no wheels were built." >&2
    echo "       This script builds wheels for Python 3.12 and 3.13; install both." >&2
    exit 1
  fi
done

# ── Build for Python 3.12 and 3.13 on the native architecture ──────────────
WHEEL_DIR="${CARGO_TARGET_DIR:-target}/wheels"
mkdir -p "$WHEEL_DIR"

# Marker file: only wheels newer than this were produced by THIS run.
STAMP=$(mktemp)
trap 'rm -f "$STAMP"' EXIT

if ! maturin build --release --features "$FEATURES" \
  -i python3.12 \
  -i python3.13; then
  echo "ERROR: maturin build failed; no wheels were built." >&2
  exit 1
fi

# ── Verify this run produced the expected wheels before claiming success ───
NEW_WHEELS=()
for TAG in cp312 cp313; do
  FOUND=$(find "$WHEEL_DIR" -maxdepth 1 -name "*-${TAG}-*.whl" -newer "$STAMP" | sort)
  if [ -z "$FOUND" ]; then
    echo "ERROR: maturin exited 0 but produced no new ${TAG} wheel in ${WHEEL_DIR}/." >&2
    exit 1
  fi
  while IFS= read -r WHEEL; do
    NEW_WHEELS+=("$WHEEL")
  done <<< "$FOUND"
done

echo ""
echo "✅ Wheels built for Python 3.12 and 3.13 (${OS}/${ARCH})"
echo "📦 Wheels location: ${WHEEL_DIR}/"
echo ""
ls -lh "${NEW_WHEELS[@]}"

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
