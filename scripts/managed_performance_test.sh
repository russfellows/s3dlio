#!/bin/bash

# Managed Performance Test with Bucket Creation/Cleanup
# Tests CLI performance against historical benchmarks with proper bucket management

set -e

echo "🔍 MANAGED PERFORMANCE TEST"
echo "=========================="
echo ""
echo "🎯 Goal: Test CLI performance against historical benchmarks"
echo "       Historical: 2.5-3 GB/s PUT, 5+ GB/s GET"
echo ""

# Load AWS environment
if [ -f "aws-env" ]; then
    echo "🔧 Loading AWS environment..."
    source aws-env
else
    echo "⚠️  No aws-env file found, using system environment"
fi

# Create unique test bucket
TEST_BUCKET="s3dlio-perf-test-$(date +%s)"
echo "📦 Creating test bucket: $TEST_BUCKET"

# Test parameters
TEST_PREFIX="perf-test-$(date +%H%M%S)"
OBJECT_SIZE_10MB=$((10 * 1024 * 1024))
NUM_OBJECTS=20  # Quick test - can increase for more comprehensive testing
CONCURRENCY=64  # High concurrency like historical tests

echo "📋 Test Configuration:"
echo "   Bucket: $TEST_BUCKET"
echo "   Objects: $NUM_OBJECTS × 10MB = 200MB total"
echo "   Concurrency: $CONCURRENCY jobs"
echo ""

# CLI binaries
native_cli="./target/performance_variants/s3-cli-native"
arrow_cli="./target/performance_variants/s3-cli-arrow"

# Check binaries exist
if [ ! -f "$native_cli" ] || [ ! -f "$arrow_cli" ]; then
    echo "❌ CLI binaries not found. Run: ./scripts/build_performance_variants.sh"
    exit 1
fi

# Create bucket
if ! $native_cli create-bucket "$TEST_BUCKET"; then
    echo "❌ Failed to create test bucket"
    exit 1
fi

# Cleanup function
cleanup() {
    echo ""
    echo "🗑️ CLEANUP: Removing all objects and bucket..."
    
    # Delete all objects recursively
    echo "   Deleting objects..."
    $native_cli delete "s3://$TEST_BUCKET/" --recursive > /dev/null 2>&1 || true
    
    # Delete bucket
    echo "   Deleting bucket..."
    $native_cli delete-bucket "$TEST_BUCKET" > /dev/null 2>&1 || true
    
    echo "✅ Cleanup completed"
}

# Set trap for cleanup on exit
trap cleanup EXIT

run_performance_test() {
    local backend="$1"
    local cli_binary="$2"
    
    # Display names for user
    local display_name="$backend"
    if [ "$backend" = "native" ]; then
        display_name="Native AWS"
    elif [ "$backend" = "arrow" ]; then
        display_name="Apache Arrow"  
    fi
    
    echo "🎯 Testing $display_name Backend"
    echo "========================="
    
    local test_uri="s3://$TEST_BUCKET/$TEST_PREFIX/$backend/"
    
    echo "📤 PUT Test..."
    local start=$(date +%s.%N)
    
    if $cli_binary put "$test_uri" -n $NUM_OBJECTS -s $OBJECT_SIZE_10MB -j $CONCURRENCY; then
        local end=$(date +%s.%N)
        local duration=$(echo "$end - $start" | bc -l)
        local total_mb=$(echo "$NUM_OBJECTS * 10" | bc)
        local throughput=$(echo "scale=2; $total_mb / $duration" | bc -l)
        
        echo "✅ PUT Results:"
        echo "   Duration: ${duration}s"
        echo "   Throughput: ${throughput} MB/s"
        echo "   Expected: 2500+ MB/s (2.5+ GB/s)"
        
        # Convert to GB/s for comparison
        local throughput_gb=$(echo "scale=3; $throughput / 1000" | bc -l)
        echo "   Throughput: ${throughput_gb} GB/s"
        
        if (( $(echo "$throughput >= 2000" | bc -l) )); then
            echo "   🎉 EXCELLENT - Near historical performance!"
        elif (( $(echo "$throughput >= 1000" | bc -l) )); then
            echo "   ✅ GOOD - Strong performance"
        elif (( $(echo "$throughput >= 500" | bc -l) )); then
            echo "   ⚠️  MODERATE - Below historical expectations"
        else
            echo "   ❌ LOW - Significant performance gap"
        fi
        
        echo ""
        echo "📥 GET Test..."
        local get_start=$(date +%s.%N)
        
        if $cli_binary get "$test_uri" -j $CONCURRENCY > /dev/null; then
            local get_end=$(date +%s.%N)
            local get_duration=$(echo "$get_end - $get_start" | bc -l)
            local get_throughput=$(echo "scale=2; $total_mb / $get_duration" | bc -l)
            local get_throughput_gb=$(echo "scale=3; $get_throughput / 1000" | bc -l)
            
            echo "✅ GET Results:"
            echo "   Duration: ${get_duration}s"  
            echo "   Throughput: ${get_throughput} MB/s"
            echo "   Throughput: ${get_throughput_gb} GB/s"
            echo "   Expected: 5000+ MB/s (5+ GB/s)"
            
            if (( $(echo "$get_throughput >= 4000" | bc -l) )); then
                echo "   🎉 EXCELLENT - Near historical performance!"
            elif (( $(echo "$get_throughput >= 2000" | bc -l) )); then
                echo "   ✅ GOOD - Strong performance"
            elif (( $(echo "$get_throughput >= 1000" | bc -l) )); then
                echo "   ⚠️  MODERATE - Below historical expectations"
            else
                echo "   ❌ LOW - Significant performance gap"
            fi
        else
            echo "❌ GET test failed"
        fi
        
    else
        echo "❌ PUT test failed"
    fi
    
    echo ""
}

# Test both backends
run_performance_test "native" "$native_cli"
run_performance_test "arrow" "$arrow_cli"

echo "🎯 PERFORMANCE SUMMARY"
echo "====================="
echo ""
echo "📊 Historical Benchmarks:"
echo "   PUT: 2.5-3 GB/s (2500-3000 MB/s)"
echo "   GET: 5+ GB/s (5000+ MB/s)"
echo ""
echo "🔍 If performance is significantly below historical:"
echo "   1. Test with more objects: -n 100"
echo "   2. Test with higher concurrency: -j 128 or -j 256"  
echo "   3. Check network conditions"
echo "   4. Profile code paths for bottlenecks"
echo ""
echo "🚀 For comprehensive testing:"
echo "   ./scripts/cli_performance_validation.sh"