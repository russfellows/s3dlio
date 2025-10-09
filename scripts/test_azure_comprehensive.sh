#!/bin/bash
# scripts/test_azure_comprehensive.sh
#
# Run comprehensive Azure integration tests
# Tests ALL ObjectStore trait methods and zero-copy Bytes API

set -e

echo "=========================================="
echo "Azure Comprehensive Integration Tests"
echo "=========================================="
echo ""

# Check environment
if [ -z "$AZURE_BLOB_ACCOUNT" ] || [ -z "$AZURE_BLOB_CONTAINER" ]; then
    echo "❌ Azure environment variables not set"
    echo ""
    echo "Please set:"
    echo "  export AZURE_BLOB_ACCOUNT=<your-storage-account-name>"
    echo "  export AZURE_BLOB_CONTAINER=<test-container-name>"
    exit 1
fi

echo "📋 Configuration:"
echo "   Storage Account: $AZURE_BLOB_ACCOUNT"
echo "   Container: $AZURE_BLOB_CONTAINER"
echo ""

echo "🧪 Running comprehensive Azure tests..."
echo "   This will test:"
echo "   - Zero-copy get() returns Bytes (v0.9.0 change)"
echo "   - Zero-copy get_range() returns Bytes (v0.9.0 change)"
echo "   - put() with various sizes"
echo "   - put_multipart() for large blobs"
echo "   - list() operations"
echo "   - stat() operations"
echo "   - delete() and delete_prefix()"
echo "   - Edge cases (empty blobs, errors, invalid ranges)"
echo "   - Concurrent operations"
echo "   - Factory function (store_for_uri)"
echo ""
echo "=========================================="
echo ""

cd "$(dirname "$0")/.."

cargo test --release --test test_azure_comprehensive -- --nocapture --test-threads=1

echo ""
echo "=========================================="
echo "✅ All comprehensive Azure tests completed!"
echo "=========================================="
