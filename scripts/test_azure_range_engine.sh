#!/bin/bash
# scripts/test_azure_range_engine.sh
#
# Integration test script for Azure RangeEngine
# Requires: az login and environment variables

set -e

echo "=========================================="
echo "Azure RangeEngine Integration Tests"
echo "=========================================="
echo ""

# Check if Azure CLI is available
if ! command -v az &> /dev/null; then
    echo "❌ Azure CLI (az) not found. Please install it first."
    echo "   Visit: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi

# Check if logged in
echo "🔐 Checking Azure login status..."
if ! az account show &> /dev/null; then
    echo "❌ Not logged in to Azure. Please run: az login"
    exit 1
fi

ACCOUNT_NAME=$(az account show --query name -o tsv)
echo "✅ Logged in to Azure account: $ACCOUNT_NAME"
echo ""

# Check environment variables
if [ -z "$AZURE_BLOB_ACCOUNT" ]; then
    echo "❌ AZURE_BLOB_ACCOUNT environment variable not set"
    echo ""
    echo "Please set the following environment variables:"
    echo "  export AZURE_BLOB_ACCOUNT=<your-storage-account-name>"
    echo "  export AZURE_BLOB_CONTAINER=<test-container-name>"
    echo ""
    echo "Optional:"
    echo "  export AZURE_RUN_LARGE_TESTS=1  # Run 100MB blob test"
    exit 1
fi

if [ -z "$AZURE_BLOB_CONTAINER" ]; then
    echo "❌ AZURE_BLOB_CONTAINER environment variable not set"
    echo ""
    echo "Please set: export AZURE_BLOB_CONTAINER=<test-container-name>"
    exit 1
fi

echo "📋 Configuration:"
echo "   Storage Account: $AZURE_BLOB_ACCOUNT"
echo "   Container: $AZURE_BLOB_CONTAINER"
if [ -n "$AZURE_RUN_LARGE_TESTS" ]; then
    echo "   Large Tests: ENABLED (100MB blobs)"
else
    echo "   Large Tests: DISABLED (set AZURE_RUN_LARGE_TESTS=1 to enable)"
fi
echo ""

# Verify container exists
echo "🔍 Verifying container exists..."
if ! az storage container show --name "$AZURE_BLOB_CONTAINER" --account-name "$AZURE_BLOB_ACCOUNT" --auth-mode login &> /dev/null; then
    echo "⚠️  Container '$AZURE_BLOB_CONTAINER' not found or not accessible"
    echo ""
    read -p "Create container '$AZURE_BLOB_CONTAINER'? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        az storage container create --name "$AZURE_BLOB_CONTAINER" --account-name "$AZURE_BLOB_ACCOUNT" --auth-mode login
        echo "✅ Container created"
    else
        echo "❌ Cannot proceed without container"
        exit 1
    fi
else
    echo "✅ Container exists and is accessible"
fi
echo ""

# Run tests
echo "🧪 Running Azure RangeEngine integration tests..."
echo "=========================================="
echo ""

cd "$(dirname "$0")/.."

cargo test --release --test test_azure_range_engine_integration -- --nocapture --test-threads=1

echo ""
echo "=========================================="
echo "✅ All Azure integration tests completed!"
echo "=========================================="
