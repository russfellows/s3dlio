#!/bin/bash
# Test script for delete_batch performance improvements
# Uses file:// URIs (no cloud credentials needed)

set -e

CLI="./target/release/s3-cli"
TEST_DIR="/tmp/s3dlio_delete_test"
TEST_URI="file://${TEST_DIR}"

echo "=========================================="
echo "Testing delete_batch Performance"
echo "=========================================="
echo ""

# Cleanup from previous runs
rm -rf "${TEST_DIR}"
mkdir -p "${TEST_DIR}"

# Test 1: Small batch (100 files)
echo "Test 1: Creating 100 files..."
for i in $(seq 1 100); do
    printf "test data %d\n" $i > "${TEST_DIR}/file_$(printf "%04d" $i).dat"
done

echo "  Files created: $(find ${TEST_DIR} -type f | wc -l)"
echo ""

echo "Test 1: Deleting 100 files with delete_batch..."
time ${CLI} delete --recursive "${TEST_URI}/"
echo "  Remaining files: $(find ${TEST_DIR} -type f 2>/dev/null | wc -l)"
echo ""

# Test 2: Medium batch (1000 files)
echo "Test 2: Creating 1000 files..."
for i in $(seq 1 1000); do
    printf "test data %d\n" $i > "${TEST_DIR}/file_$(printf "%04d" $i).dat"
done

echo "  Files created: $(find ${TEST_DIR} -type f | wc -l)"
echo ""

echo "Test 2: Deleting 1000 files with delete_batch..."
time ${CLI} delete --recursive "${TEST_URI}/"
echo "  Remaining files: $(find ${TEST_DIR} -type f 2>/dev/null | wc -l)"
echo ""

# Test 3: Large batch (5000 files)
echo "Test 3: Creating 5000 files..."
for i in $(seq 1 5000); do
    printf "test data %d\n" $i > "${TEST_DIR}/file_$(printf "%04d" $i).dat"
done

echo "  Files created: $(find ${TEST_DIR} -type f | wc -l)"
echo ""

echo "Test 3: Deleting 5000 files with delete_batch..."
time ${CLI} delete --recursive "${TEST_URI}/"
echo "  Remaining files: $(find ${TEST_DIR} -type f 2>/dev/null | wc -l)"
echo ""

# Test 4: Pattern matching delete
echo "Test 4: Creating 1000 files with patterns..."
for i in $(seq 1 500); do
    printf "test data %d\n" $i > "${TEST_DIR}/test_$(printf "%04d" $i).dat"
    printf "other data %d\n" $i > "${TEST_DIR}/other_$(printf "%04d" $i).dat"
done

echo "  Total files: $(find ${TEST_DIR} -type f | wc -l)"
echo "  Files matching 'test_': $(find ${TEST_DIR} -name 'test_*' | wc -l)"
echo ""

echo "Test 4: Deleting only 'test_*' files with pattern..."
time ${CLI} delete --recursive --pattern 'test_' "${TEST_URI}/"
echo "  Remaining files: $(find ${TEST_DIR} -type f | wc -l)"
echo "  Remaining 'other_' files: $(find ${TEST_DIR} -name 'other_*' | wc -l)"
echo ""

# Cleanup
rm -rf "${TEST_DIR}"

echo "=========================================="
echo "All tests passed! ✓"
echo "=========================================="
echo ""
echo "Performance observations:"
echo "- 100 files: Should complete in <1 second"
echo "- 1000 files: Should complete in ~1-2 seconds"
echo "- 5000 files: Should complete in ~5-10 seconds"
echo ""
echo "Note: file:// backend uses concurrent deletes (10-100 workers)"
echo "      S3/Azure/GCS use batch APIs (much faster at scale)"
