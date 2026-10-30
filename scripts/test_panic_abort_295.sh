#!/usr/bin/env bash
# Regression test for #295 / #267: the file stores must not rely on catching a panic.
#
# Builds examples/abort_localset_295.rs with the `abort-test` profile (panic = "abort", see
# Cargo.toml) and runs it. The program reads and writes through file:// from a tokio LocalSet on
# a multi-thread runtime; with a catch_unwind-based `block_in_place` probe it dies with SIGABRT.
# Passes only if the program exits 0 and prints OK from a panic=abort build.
#
# Usage: scripts/test_panic_abort_295.sh   (S3DLIO_TEST_DIR may name the scratch directory)
set -euo pipefail
cd "$(dirname "$0")/.."
cargo build --profile abort-test --example abort_localset_295 --offline
out="$(target/abort-test/examples/abort_localset_295)"
echo "$out"
grep -qx "panic=abort build" <<<"$out" || { echo "FAIL: not a panic=abort build" >&2; exit 1; }
grep -qx "OK" <<<"$out" || { echo "FAIL: no OK" >&2; exit 1; }
echo "PASS: panic=abort LocalSet read/write"
