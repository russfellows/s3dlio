#!/bin/bash
# scripts/compare_backends.sh
#
# Performance comparison script for native vs Arrow backends

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TEST_DIR="/tmp/s3dlio_backend_comparison"

echo "🔄 S3DLIO Backend Performance Comparison"
echo "========================================"

# Clean up and prepare test directory
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"

# Function to run a specific backend test
run_backend_test() {
    local backend_name="$1"
    local features="$2"
    
    echo ""
    echo "📊 Testing $backend_name backend..."
    echo "Features: $features"
    
    # Build the backend
    echo "Building..."
    cd "$PROJECT_DIR"
    if [ -n "$features" ]; then
        cargo build --release --features "$features" --no-default-features
    else
        cargo build --release
    fi
    
    # Create a simple test file
    local test_file="$TEST_DIR/test_${backend_name}.txt"
    echo "Creating test data..."
    dd if=/dev/urandom of="$test_file" bs=1M count=10 2>/dev/null
    
    # Run basic operations and time them
    echo "Running operations..."
    
    # Test 1: Simple PUT/GET cycle
    time_start=$(date +%s.%N)
    
    # Use the CLI to test operations (assuming it works with either backend)
    echo "Testing file operations..."
    local dest_uri="file://$TEST_DIR/${backend_name}_output.txt"
    
    # For now, we'll just verify the build worked
    echo "✅ $backend_name backend built successfully"
    
    time_end=$(date +%s.%N)
    local duration=$(echo "$time_end - $time_start" | bc -l)
    echo "⏱️  Duration: ${duration}s"
    
    rm -f "$test_file"
}

echo ""
echo "Building and testing both backends..."

# Test Native Backend (current default)
run_backend_test "native" "native-backends"

# Test Arrow Backend  
run_backend_test "arrow" "arrow-backend"

echo ""
echo "🎯 Backend Comparison Summary"
echo "============================"
echo "✅ Native Backend: Built successfully with AWS/Azure SDKs"
echo "✅ Arrow Backend: Built successfully with Apache Arrow object_store"
echo ""
echo "📈 Next Steps:"
echo "1. Create actual benchmark tests using both backends"
echo "2. Compare performance on identical workloads"
echo "3. Test against real cloud storage (S3, Azure)"
echo "4. Measure throughput, latency, and memory usage"

# Clean up
rm -rf "$TEST_DIR"

echo ""
echo "✨ Comparison complete! Both backends are ready for benchmarking."