#!/bin/bash
# Comprehensive v0.8.19 Testing - All Backends
# Tests: regex pattern filtering, universal stat, deprecation warnings

set -e

# Live legs are opt-in: there are no default buckets. Each leg runs only when its
# variable names a bucket/container you own (it must hold hello.txt / test.txt):
#   S3  : S3DLIO_TEST_BUCKET  (+ AWS credentials)
#   GCS : S3DLIO_TEST_GCS_BUCKET (+ gcloud auth)
#   Azure: AZURE_BLOB_ACCOUNT + S3DLIO_TEST_AZURE_CONTAINER
S3_BUCKET="${S3DLIO_TEST_BUCKET:-}"
GCS_BUCKET="${S3DLIO_TEST_GCS_BUCKET:-}"

CLI="./target/release/s3-cli"

echo "========================================"
echo "v0.8.19 Multi-Backend Testing"
echo "========================================"
echo ""

# Check if binary exists
if [ ! -f "$CLI" ]; then
    echo "Error: s3-cli binary not found. Run: cargo build --release --bin s3-cli"
    exit 1
fi

# Track failures
FAILURES=0

# Helper function for test results
test_result() {
    if [ $? -eq 0 ]; then
        echo "✅ PASS"
    else
        echo "❌ FAIL"
        ((FAILURES++))
    fi
}

# ====================
# S3 Backend Tests
# ====================
echo "=== S3 Backend Tests ==="
if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$S3_BUCKET" ]; then
    echo "Test: ls s3://$S3_BUCKET/"
    $CLI ls s3://$S3_BUCKET/ > /dev/null
    test_result
    
    echo "Test: ls with pattern (*.txt)"
    $CLI ls s3://$S3_BUCKET/ -p '.*\.txt$' > /dev/null
    test_result
    
    echo "Test: stat s3://$S3_BUCKET/hello.txt"
    $CLI stat s3://$S3_BUCKET/hello.txt > /dev/null
    test_result
else
    echo "⏭️  Skipped (needs AWS credentials and S3DLIO_TEST_BUCKET)"
fi
echo ""

# ====================
# File Backend Tests
# ====================
echo "=== File Backend Tests ==="
TEST_DIR="/tmp/s3dlio-test-$$"
mkdir -p "$TEST_DIR"
echo "test" > "$TEST_DIR/file1.txt"
echo "test" > "$TEST_DIR/file2.txt"
echo "data" > "$TEST_DIR/data.json"

echo "Test: ls file://$TEST_DIR/"
$CLI ls file://$TEST_DIR/ > /dev/null
test_result

echo "Test: ls with pattern (*.txt)"
$CLI ls file://$TEST_DIR/ -p '.*\.txt$' | grep -q "file1.txt"
test_result

echo "Test: stat file://$TEST_DIR/file1.txt"
$CLI stat file://$TEST_DIR/file1.txt > /dev/null
test_result

rm -rf "$TEST_DIR"
echo ""

# ====================
# GCS Backend Tests
# ====================
echo "=== GCS Backend Tests ==="
if [ -n "$GCS_BUCKET" ] && command -v gcloud &> /dev/null && gcloud auth list 2>&1 | grep -q "ACTIVE"; then
    echo "Test: ls gs://$GCS_BUCKET/"
    $CLI ls gs://$GCS_BUCKET/ > /dev/null 2>&1
    test_result
    
    echo "Test: ls with pattern (*.txt)"
    $CLI ls gs://$GCS_BUCKET/ -p '.*\.txt$' > /dev/null 2>&1
    test_result
    
    echo "Test: stat gs://$GCS_BUCKET/test.txt"
    $CLI stat gs://$GCS_BUCKET/test.txt > /dev/null 2>&1
    test_result
else
    echo "⏭️  Skipped (needs gcloud auth and S3DLIO_TEST_GCS_BUCKET)"
fi
echo ""

# ====================
# DirectIO Backend Tests
# ====================
echo "=== DirectIO Backend Tests ==="
TEST_DIR="/tmp/directio-test-$$"
mkdir -p "$TEST_DIR"
echo "directio test" > "$TEST_DIR/file1.txt"
echo "data" > "$TEST_DIR/data.json"

echo "Test: ls direct://$TEST_DIR/"
$CLI ls direct://$TEST_DIR/ > /dev/null
test_result

echo "Test: ls with pattern (*.txt)"
$CLI ls direct://$TEST_DIR/ -p '.*\.txt$' | grep -q "file1.txt"
test_result

echo "Test: stat direct://$TEST_DIR/file1.txt"
$CLI stat direct://$TEST_DIR/file1.txt > /dev/null
test_result

rm -rf "$TEST_DIR"
echo ""

# ====================
# Azure Backend Tests
# ====================
echo "=== Azure Backend Tests ==="
if [ -n "$AZURE_BLOB_ACCOUNT" ] && [ -n "$S3DLIO_TEST_AZURE_CONTAINER" ]; then
    AZURE_URI="az://${AZURE_BLOB_ACCOUNT}/${S3DLIO_TEST_AZURE_CONTAINER}/"
    
    echo "Test: ls $AZURE_URI"
    $CLI ls $AZURE_URI > /dev/null 2>&1
    test_result
    
    echo "Test: ls with pattern (*.txt)"
    $CLI ls $AZURE_URI -p '.*\.txt$' > /dev/null 2>&1
    test_result
    
    echo "Test: stat ${AZURE_URI}test.txt"
    $CLI stat ${AZURE_URI}test.txt > /dev/null 2>&1
    test_result
else
    echo "⏭️  Skipped (needs AZURE_BLOB_ACCOUNT and S3DLIO_TEST_AZURE_CONTAINER)"
fi
echo ""

# ====================
# Deprecation Warning Test
# ====================
echo "=== Deprecation Warning Test ==="
if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$S3_BUCKET" ]; then
    echo "Test: list command shows deprecation warning"
    $CLI list s3://$S3_BUCKET/ 2>&1 | grep -q "WARNING.*deprecated"
    test_result
else
    echo "⏭️  Skipped (needs AWS credentials and S3DLIO_TEST_BUCKET)"
fi
echo ""

# ====================
# Summary
# ====================
echo "========================================"
if [ $FAILURES -eq 0 ]; then
    echo "✅ ALL TESTS PASSED"
    echo "========================================"
    exit 0
else
    echo "❌ $FAILURES TESTS FAILED"
    echo "========================================"
    exit 1
fi
