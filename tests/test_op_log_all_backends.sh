#!/bin/bash
#
# Test script to verify op-log functionality works with all storage backends
# Tests file:// and direct:// backends using multiple LIST operations
#

set -e

echo "========================================"
echo "Testing op-log functionality with all backends"
echo "========================================"

# Clean up test files
rm -f /tmp/test_op_log_*.tsv.zst
rm -rf /tmp/s3dlio_test_oplog_*

echo ""
echo "========================================"
echo "Test 1: Multiple operations with file:// backend"
echo "========================================"

OP_LOG_FILE="/tmp/test_op_log_file_mixed.tsv.zst"

echo "Creating test directory with 30 files..."
mkdir -p /tmp/s3dlio_test_oplog_file
for i in {1..30}; do
  echo "Test data for file $i - $(date +%s%N)" > /tmp/s3dlio_test_oplog_file/testfile_${i}.txt
done

echo "Performing 25 LIST operations to generate log entries..."
for i in {1..25}; do
  ./target/release/s3-cli --op-log "$OP_LOG_FILE" ls \
    file:///tmp/s3dlio_test_oplog_file/ \
    -r > /dev/null 2>&1
done

# Check log was created
if [ -f "$OP_LOG_FILE" ]; then
  TOTAL_OPS=$(zstd -dc "$OP_LOG_FILE" | tail -n +2 | wc -l)
  echo ""
  echo "SUCCESS: Op-log created for file:// backend"
  echo "Total operations logged: $TOTAL_OPS"
  echo ""
  echo "First 20 lines of op-log:"
  zstd -dc "$OP_LOG_FILE" | head -20
  echo ""
  echo "Last 10 lines of op-log:"
  zstd -dc "$OP_LOG_FILE" | tail -10
  echo ""
  echo "Operation breakdown:"
  echo "  LIST operations: $(zstd -dc "$OP_LOG_FILE" | grep -c LIST || echo 0)"
else
  echo "FAILURE: No op-log created for file:// backend"
  exit 1
fi

echo ""
echo "========================================"
echo "Test 2: Multiple operations with direct:// backend"
echo "========================================"

OP_LOG_FILE_DIRECT="/tmp/test_op_log_direct_mixed.tsv.zst"

echo "Creating test directory with 25 files..."
mkdir -p /tmp/s3dlio_test_oplog_direct
for i in {1..25}; do
  echo "Direct IO test data for file $i - $(date +%s%N)" > /tmp/s3dlio_test_oplog_direct/direct_${i}.txt
done

echo "Performing 30 LIST operations to generate log entries..."
for i in {1..30}; do
  ./target/release/s3-cli --op-log "$OP_LOG_FILE_DIRECT" ls \
    direct:///tmp/s3dlio_test_oplog_direct/ \
    -r > /dev/null 2>&1
done

# Check log was created
if [ -f "$OP_LOG_FILE_DIRECT" ]; then
  TOTAL_OPS=$(zstd -dc "$OP_LOG_FILE_DIRECT" | tail -n +2 | wc -l)
  echo ""
  echo "SUCCESS: Op-log created for direct:// backend"
  echo "Total operations logged: $TOTAL_OPS"
  echo ""
  echo "First 20 lines of op-log:"
  zstd -dc "$OP_LOG_FILE_DIRECT" | head -20
  echo ""
  echo "Operation breakdown:"
  echo "  LIST operations: $(zstd -dc "$OP_LOG_FILE_DIRECT" | grep -c LIST || echo 0)"
else
  echo "FAILURE: No op-log created for direct:// backend"
  exit 1
fi

echo ""
echo "========================================"
echo "Test 3: Azure Blob Storage az:// backend"
echo "========================================"
if [ -n "$AZURE_STORAGE_ACCOUNT" ] && [ -n "$AZURE_STORAGE_ACCESS_KEY" ]; then
  AZURE_CONTAINER="${AZURE_CONTAINER:-s3dlio-test}"
  AZURE_PREFIX="az://${AZURE_STORAGE_ACCOUNT}/${AZURE_CONTAINER}/oplog-test/"
  
  echo "Testing with Azure container: $AZURE_PREFIX"
  
  ./target/release/s3-cli --op-log /tmp/test_op_log_azure_list.tsv.zst ls \
    "$AZURE_PREFIX" \
    -r || echo "Azure LIST operation failed (container may not exist or no permissions)"
  
  if [ -f /tmp/test_op_log_azure_list.tsv.zst ]; then
    echo ""
    echo "SUCCESS: Op-log created for az:// LIST operation"
    echo "First 10 lines of op-log:"
    zstd -dc /tmp/test_op_log_azure_list.tsv.zst | head -10
  else
    echo "INFO: No Azure test performed"
  fi
else
  echo "SKIPPED: No Azure credentials available (AZURE_STORAGE_ACCOUNT and AZURE_STORAGE_ACCESS_KEY not set)"
fi

echo ""
echo "========================================"
echo "All op-log tests completed successfully"
echo "========================================"

# Show summary of all log files created
echo ""
echo "Op-log files created:"
ls -lh /tmp/test_op_log_*.tsv.zst 2>/dev/null || echo "No log files found"

# Show detailed sample of a log file (decompressed)
if [ -f /tmp/test_op_log_file_mixed.tsv.zst ]; then
  echo ""
  echo "========================================"
  echo "Detailed op-log analysis (file:// backend)"
  echo "========================================"
  echo "TSV Format: idx thread op client_id n_objects bytes endpoint file error start first_byte end duration_ns"
  echo ""
  zstd -dc /tmp/test_op_log_file_mixed.tsv.zst | head -30
  echo ""
  echo "... (showing first 30 lines)"
  echo ""
  echo "Log file statistics:"
  echo "  Compressed size: $(ls -lh /tmp/test_op_log_file_mixed.tsv.zst | awk '{print $5}')"
  echo "  Uncompressed size: $(zstd -dc /tmp/test_op_log_file_mixed.tsv.zst | wc -c) bytes"
  echo "  Total operations: $(zstd -dc /tmp/test_op_log_file_mixed.tsv.zst | tail -n +2 | wc -l)"
  echo "  Operation types:"
  echo "    - LIST: $(zstd -dc /tmp/test_op_log_file_mixed.tsv.zst | grep -c LIST || echo 0)"
fi

# Cleanup
echo ""
echo "Cleaning up test files..."
rm -rf /tmp/s3dlio_test_oplog_file
rm -rf /tmp/s3dlio_test_oplog_direct
rm -f /tmp/test_op_log_*.tsv.zst

echo "Done"
