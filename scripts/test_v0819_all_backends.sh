#!/bin/bash
# Comprehensive v0.8.19 Testing - All Backends
# Tests: regex pattern filtering, universal stat, deprecation warnings

set -e

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
if [ -n "$AWS_ACCESS_KEY_ID" ]; then
    echo "Test: ls s3://signal65-public/"
    $CLI ls s3://signal65-public/ > /dev/null
    test_result
    
    echo "Test: ls with pattern (*.txt)"
    $CLI ls s3://signal65-public/ -p '.*\.txt$' > /dev/null
    test_result
    
    echo "Test: stat s3://signal65-public/hello.txt"
    $CLI stat s3://signal65-public/hello.txt > /dev/null
    test_result
else
    echo "⏭️  Skipped (no AWS credentials)"
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
if command -v gcloud &> /dev/null && gcloud auth list 2>&1 | grep -q "ACTIVE"; then
    echo "Test: ls gs://signal65-russ-b1/"
    $CLI ls gs://signal65-russ-b1/ > /dev/null 2>&1
    test_result
    
    echo "Test: ls with pattern (*.txt)"
    $CLI ls gs://signal65-russ-b1/ -p '.*\.txt$' > /dev/null 2>&1
    test_result
    
    echo "Test: stat gs://signal65-russ-b1/test.txt"
    $CLI stat gs://signal65-russ-b1/test.txt > /dev/null 2>&1
    test_result
else
    echo "⏭️  Skipped (no GCS auth)"
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
if [ -n "$AZURE_BLOB_ACCOUNT" ] && [ -n "$AZURE_BLOB_CONTAINER" ]; then
    AZURE_URI="az://${AZURE_BLOB_ACCOUNT}/${AZURE_BLOB_CONTAINER}/"
    
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
    echo "⏭️  Skipped (no Azure credentials)"
fi
echo ""

# ====================
# Deprecation Warning Test
# ====================
echo "=== Deprecation Warning Test ==="
if [ -n "$AWS_ACCESS_KEY_ID" ]; then
    echo "Test: list command shows deprecation warning"
    $CLI list s3://signal65-public/ 2>&1 | grep -q "WARNING.*deprecated"
    test_result
else
    echo "⏭️  Skipped (no AWS credentials)"
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
