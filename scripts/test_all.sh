#!/usr/bin/env bash
# one-time: set env vars in your shell
export AZURE_BLOB_ACCOUNT="egiazurestore1"
export AZURE_BLOB_CONTAINER="s3dlio"

set -euo pipefail

echo "== Core (no cloud) =="
cargo test --release --lib

echo "== File Store Tests =="
cargo test --release --test test_file_store -- --nocapture

echo "== Azure Tests =="
if [[ -n "${AZURE_BLOB_CONTAINER:-}" && ( -n "${AZURE_BLOB_ACCOUNT:-}" || -n "${AZURE_BLOB_ACCOUNT_URL:-}" ) ]]; then
  cargo test --release --features azure --test azure_blob_smoke    -- --nocapture
  cargo test --release --features azure --test azure_blob_multi    -- --nocapture
  cargo test --release --features azure --test azure_blob_sequence -- --nocapture
else
  echo "SKIP Azure: set AZURE_BLOB_CONTAINER and (AZURE_BLOB_ACCOUNT or AZURE_BLOB_ACCOUNT_URL)"
fi

echo "== S3 Tests  =="
# Look for a top-level .env (even if script is run from configs/)
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
DOTENV_FILE="$REPO_ROOT/.env"

MISSING=()
for VAR in AWS_REGION AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY S3DLIO_TEST_BUCKET; do
  [[ -z "${!VAR:-}" ]] && MISSING+=("$VAR")
done

if [[ -f "$DOTENV_FILE" ]]; then
  echo "dotenvy: found $DOTENV_FILE (tests will load it at runtime)"
  HAS_DOTENV=1
else
  HAS_DOTENV=0
fi

if [[ ${#MISSING[@]} -eq 0 || $HAS_DOTENV -eq 1 ]]; then
  cargo test --release --test test_multipart       -- --nocapture
  cargo test --release --test test_dataloader      -- --nocapture
  cargo test --release --test test_performance     -- --nocapture
  cargo test --release --test object_format_tests  -- --nocapture
  cargo test --release --test test_data-gen        -- --nocapture
else
  echo "SKIP S3: missing envs: ${MISSING[*]} and no .env found at $DOTENV_FILE"
  echo "       Set AWS_* vars or create a top-level .env for dotenvy."
fi

