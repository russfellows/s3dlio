#!/usr/bin/env bash
# scripts/test_list_buckets_gcs.sh
#
# Integration test: verify that `s3dlio list-buckets` works against
# Google Cloud Storage via both the S3-compatibility API (issue #121)
# and — if google-cloud auth is configured — the native GCS path.
#
# ----------------------------------------------------------------
# PREREQUISITES
# ----------------------------------------------------------------
#
#  S3-compatibility test (required):
#    1. GCS bucket(s) already exist in your project.
#    2. An HMAC key pair for a service account (or your user account).
#       Create one at: GCS Console → Settings → Interoperability
#       or with: gcloud storage hmac create SERVICE_ACCOUNT_EMAIL
#    3. Export:
#         export AWS_ACCESS_KEY_ID=<HMAC access key>
#         export AWS_SECRET_ACCESS_KEY=<HMAC secret>
#         export AWS_REGION=us-central1          # your bucket region
#         export GOOGLE_CLOUD_PROJECT=my-project # required for gs:// test
#
#  Native GCS test (optional, requires gcs-community or gcs-official feature):
#    Application Default Credentials:
#         gcloud auth application-default login
#         export GOOGLE_CLOUD_PROJECT=my-project
#
# ----------------------------------------------------------------
# USAGE
# ----------------------------------------------------------------
#   chmod +x scripts/test_list_buckets_gcs.sh
#   ./scripts/test_list_buckets_gcs.sh
#
# Optional: specify binary path
#   S3DLIO_BIN=./target/release/s3dlio ./scripts/test_list_buckets_gcs.sh
#
# ----------------------------------------------------------------

set -euo pipefail

# ---- colour helpers ---------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # no colour

pass()  { echo -e "${GREEN}[PASS]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; FAILURES=$((FAILURES+1)); }
skip()  { echo -e "${YELLOW}[SKIP]${NC} $*"; }
info()  { echo -e "       $*"; }

FAILURES=0

# ---- locate binary ----------------------------------------------------
S3DLIO_BIN="${S3DLIO_BIN:-}"
if [[ -z "$S3DLIO_BIN" ]]; then
    # Try release build first, then debug
    for candidate in \
            "$(dirname "$0")/../target/release/s3-cli" \
            "$(dirname "$0")/../target/debug/s3-cli"; do
        if [[ -x "$candidate" ]]; then
            S3DLIO_BIN="$(realpath "$candidate")"
            break
        fi
    done
fi

if [[ -z "$S3DLIO_BIN" || ! -x "$S3DLIO_BIN" ]]; then
    echo "ERROR: s3-cli binary not found. Build with 'cargo build --release' or set S3DLIO_BIN."
    exit 1
fi

info "Using binary: $S3DLIO_BIN"
info ""

# =======================================================================
# TEST 1 — S3-compatibility endpoint (list-buckets → GCS)
# =======================================================================
echo "=== TEST 1: list-buckets via S3-compatibility API ==="
info "Requires: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and GCS HMAC keys"
info ""

if [[ -z "${AWS_ACCESS_KEY_ID:-}" || -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
    skip "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set — skipping S3-compat test"
    info "  To run: export AWS_ACCESS_KEY_ID=<hmac-access-key> AWS_SECRET_ACCESS_KEY=<hmac-secret>"
else
    # Force the GCS S3-compatibility endpoint (XML API)
    export AWS_ENDPOINT_URL="${AWS_ENDPOINT_URL:-https://storage.googleapis.com}"
    export AWS_S3_ADDRESSING_STYLE="${AWS_S3_ADDRESSING_STYLE:-path}"
    # Use provided region (or a sensible default)
    export AWS_REGION="${AWS_REGION:-us-central1}"

    info "Endpoint : $AWS_ENDPOINT_URL"
    info "Region   : $AWS_REGION"
    info "AddrStyle: $AWS_S3_ADDRESSING_STYLE"
    info ""

    OUTPUT=$("$S3DLIO_BIN" list-buckets s3:// 2>&1) || true

    if echo "$OUTPUT" | grep -q "InvalidArgument\|AuthFailure\|InvalidAccessKeyId"; then
        fail "GCS S3-compat list-buckets returned an error:"
        info "  $OUTPUT"
    elif echo "$OUTPUT" | grep -qiE "error|Error"; then
        fail "list-buckets returned unexpected error output:"
        info "  $OUTPUT"
    else
        BUCKET_COUNT=$(echo "$OUTPUT" | grep -c "gs://" 2>/dev/null || true)
        pass "list-buckets via S3-compat endpoint succeeded (found ~${BUCKET_COUNT} bucket URI(s))"
        info "Output:"
        echo "$OUTPUT" | sed 's/^/    /'
    fi
fi

echo ""

# =======================================================================
# TEST 2 — Native GCS path (gs://)
# =======================================================================
echo "=== TEST 2: list-buckets via gs:// (native GCS client) ==="
info "Requires: GOOGLE_CLOUD_PROJECT, gcs-community or gcs-official feature, ADC credentials"
info ""

if [[ -z "${GOOGLE_CLOUD_PROJECT:-}" && -z "${GCLOUD_PROJECT:-}" ]]; then
    skip "GOOGLE_CLOUD_PROJECT not set — skipping native GCS test"
    info "  To run: export GOOGLE_CLOUD_PROJECT=<your-project-id>"
else
    PROJECT="${GOOGLE_CLOUD_PROJECT:-$GCLOUD_PROJECT}"
    info "Project: $PROJECT"
    info ""

    OUTPUT=$("$S3DLIO_BIN" list-buckets gs:// 2>&1) || true

    if echo "$OUTPUT" | grep -q "GCS support is not compiled in"; then
        skip "GCS native client feature not compiled — build with --features gcs-community or gcs-official"
    elif echo "$OUTPUT" | grep -qiE "error|Error|failed|Failed"; then
        fail "list-buckets gs:// returned an error:"
        info "  $OUTPUT"
    else
        BUCKET_COUNT=$(echo "$OUTPUT" | grep -c "gs://" 2>/dev/null || true)
        pass "list-buckets via gs:// succeeded (found ~${BUCKET_COUNT} bucket URI(s))"
        info "Output:"
        echo "$OUTPUT" | sed 's/^/    /'
    fi
fi

echo ""

# =======================================================================
# TEST 3 — Verify list-buckets help shows multi-backend URI arg
# =======================================================================
echo "=== TEST 3: CLI --help shows URI argument ==="

HELP_OUTPUT=$("$S3DLIO_BIN" list-buckets --help 2>&1) || true

if echo "$HELP_OUTPUT" | grep -qiE "gs://|az://|gcs://|scheme|URI"; then
    pass "list-buckets --help documents multi-backend URI argument"
else
    fail "list-buckets --help does not mention gs:// or az:// — check CLI changes"
    info "Help output:"
    echo "$HELP_OUTPUT" | sed 's/^/    /'
fi

echo ""

# =======================================================================
# Summary
# =======================================================================
echo "========================================"
if [[ $FAILURES -eq 0 ]]; then
    echo -e "${GREEN}All tests passed (${FAILURES} failures)${NC}"
else
    echo -e "${RED}${FAILURES} test(s) FAILED${NC}"
    exit 1
fi
