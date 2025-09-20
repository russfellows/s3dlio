#!/bin/bash

# Sustained Performance Test - Matches User's Historical Methodology
# Tests CLI performance with proven parameters: 24-48 threads, 10MiB objects, 30+ seconds

set -e

echo "🏁 SUSTAINED PERFORMANCE TEST"
echo "============================="
echo ""
echo "🎯 Goal: Match your proven testing methodology"
echo "   ✓ 24-48 threads (your optimal range)"
echo "   ✓ 10 MiB objects" 
echo "   ✓ 30+ seconds sustained testing"
echo "   ✓ Few thousand objects for proper measurement"
echo ""
echo "📊 Target: Historical 2.5-3 GB/s PUT, 5+ GB/s GET"
echo ""

# Load AWS environment
if [ -f "aws-env" ]; then
    echo "🔧 Loading AWS environment..."
    source aws-env
else
    echo "⚠️  No aws-env file found, using system environment"
fi

# Test parameters matching user's proven methodology
OBJECT_SIZE_10MB=$((10 * 1024 * 1024))  # 10 MiB exactly
NUM_OBJECTS=2500  # Should give us 30+ seconds at realistic speeds
CONCURRENCY_LOW=24  # User's proven sweet spot lower bound
CONCURRENCY_HIGH=48 # User's proven sweet spot upper bound

# Calculate expected test duration
TOTAL_DATA_GB=$(echo "scale=2; $NUM_OBJECTS * 10 / 1024" | bc -l)

echo "📋 Test Configuration (Matching Your Proven Method):"
echo "   Objects: $NUM_OBJECTS × 10 MiB = ${TOTAL_DATA_GB} GiB total"
echo "   Concurrency: $CONCURRENCY_LOW and $CONCURRENCY_HIGH threads"
echo "   Expected Duration: 30+ seconds (at 2+ GB/s)"
echo ""

# Create unique test bucket
TEST_BUCKET="s3dlio-sustained-test-$(date +%s)"
echo "📦 Creating test bucket: $TEST_BUCKET"

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
    echo "   Deleting objects (this may take a moment)..."
    $native_cli delete "s3://$TEST_BUCKET/" --recursive > /dev/null 2>&1 || true
    
    # Delete bucket
    echo "   Deleting bucket..."
    $native_cli delete-bucket "$TEST_BUCKET" > /dev/null 2>&1 || true
    
    echo "✅ Cleanup completed"
}

# Set trap for cleanup on exit
trap cleanup EXIT

run_sustained_test() {
    local backend="$1"
    local cli_binary="$2"
    local concurrency="$3"
    
    # Display names for user
    local display_name="$backend"
    if [ "$backend" = "native" ]; then
        display_name="Native AWS"
    elif [ "$backend" = "arrow" ]; then
        display_name="Apache Arrow"  
    fi
    
    echo "🎯 Testing $display_name Backend (${concurrency} threads)"
    echo "================================================="
    
    local test_uri="s3://$TEST_BUCKET/sustained-test/$backend-j${concurrency}/"
    
    echo "📤 PUT Test (Sustained Load)..."
    echo "   Uploading $NUM_OBJECTS objects with $concurrency threads..."
    
    local start=$(date +%s.%N)
    
    if $cli_binary put "$test_uri" -n $NUM_OBJECTS -s $OBJECT_SIZE_10MB -j $concurrency; then
        local end=$(date +%s.%N)
        local duration=$(echo "$end - $start" | bc -l)
        local total_mb=$(echo "$NUM_OBJECTS * 10" | bc)
        local throughput=$(echo "scale=2; $total_mb / $duration" | bc -l)
        local throughput_gb=$(echo "scale=3; $throughput / 1024" | bc -l)
        
        echo ""
        echo "✅ PUT Results ($concurrency threads):"
        echo "   Duration: ${duration}s"
        echo "   Total Data: ${TOTAL_DATA_GB} GiB"
        echo "   Throughput: ${throughput} MB/s"
        echo "   Throughput: ${throughput_gb} GiB/s"
        echo "   Historical Target: 2.5-3.0 GiB/s"
        
        # Performance assessment 
        if (( $(echo "$throughput_gb >= 2.5" | bc -l) )); then
            echo "   🎉 EXCELLENT - Matches historical performance!"
        elif (( $(echo "$throughput_gb >= 2.0" | bc -l) )); then
            echo "   ✅ VERY GOOD - Close to historical performance"  
        elif (( $(echo "$throughput_gb >= 1.5" | bc -l) )); then
            echo "   ⚠️  GOOD - But below historical peak"
        elif (( $(echo "$throughput_gb >= 1.0" | bc -l) )); then
            echo "   ⚠️  MODERATE - Significant gap from historical"
        else
            echo "   ❌ LOW - Major performance regression"
        fi
        
        # Verify we got proper sustained test duration
        if (( $(echo "$duration >= 30" | bc -l) )); then
            echo "   ✅ Sustained test duration: ${duration}s (30+ seconds achieved)"
        else
            echo "   ⚠️  Short test duration: ${duration}s (target was 30+ seconds)"
        fi
        
        echo ""
        echo "📥 GET Test (Sustained Load)..."
        echo "   Downloading $NUM_OBJECTS objects with $concurrency threads..."
        
        local get_start=$(date +%s.%N)
        
        if $cli_binary get "$test_uri" -j $concurrency > /dev/null; then
            local get_end=$(date +%s.%N)
            local get_duration=$(echo "$get_end - $get_start" | bc -l)
            local get_throughput=$(echo "scale=2; $total_mb / $get_duration" | bc -l)
            local get_throughput_gb=$(echo "scale=3; $get_throughput / 1024" | bc -l)
            
            echo ""
            echo "✅ GET Results ($concurrency threads):"
            echo "   Duration: ${get_duration}s"  
            echo "   Total Data: ${TOTAL_DATA_GB} GiB"
            echo "   Throughput: ${get_throughput} MB/s"
            echo "   Throughput: ${get_throughput_gb} GiB/s"
            echo "   Historical Target: 5.0+ GiB/s"
            
            if (( $(echo "$get_throughput_gb >= 5.0" | bc -l) )); then
                echo "   🎉 EXCELLENT - Matches historical performance!"
            elif (( $(echo "$get_throughput_gb >= 4.0" | bc -l) )); then
                echo "   ✅ VERY GOOD - Close to historical performance"
            elif (( $(echo "$get_throughput_gb >= 3.0" | bc -l) )); then
                echo "   ⚠️  GOOD - But below historical peak"
            elif (( $(echo "$get_throughput_gb >= 2.0" | bc -l) )); then
                echo "   ⚠️  MODERATE - Significant gap from historical"
            else
                echo "   ❌ LOW - Major performance regression"
            fi
            
            # Verify sustained test duration
            if (( $(echo "$get_duration >= 15" | bc -l) )); then
                echo "   ✅ Good sustained download duration: ${get_duration}s"
            fi
        else
            echo "❌ GET test failed"
        fi
        
    else
        echo "❌ PUT test failed"
    fi
    
    echo ""
}

echo "🚀 Starting sustained performance tests..."
echo ""

# Test both backends with both concurrency levels (user's proven sweet spot)
run_sustained_test "native" "$native_cli" "$CONCURRENCY_LOW"

echo "⏳ Waiting 30 seconds before next test..."
sleep 30
echo ""

run_sustained_test "native" "$native_cli" "$CONCURRENCY_HIGH" 

echo "⏳ Waiting 30 seconds before next test..."
sleep 30
echo ""

run_sustained_test "arrow" "$arrow_cli" "$CONCURRENCY_LOW"

echo "⏳ Waiting 30 seconds before next test..."
sleep 30
echo ""

run_sustained_test "arrow" "$arrow_cli" "$CONCURRENCY_HIGH"

echo "🎯 SUSTAINED PERFORMANCE SUMMARY"
echo "================================"
echo ""
echo "📊 Your Historical Benchmarks (Proven Method):"
echo "   PUT: 2.5-3.0 GiB/s with 24-48 threads, 10 MiB objects"
echo "   GET: 5.0+ GiB/s with sustained load"
echo ""
echo "🔍 Results Analysis:"
echo "   • Test used your proven methodology: 24-48 threads, 10 MiB objects"  
echo "   • Sustained load: $NUM_OBJECTS objects over 30+ seconds"
echo "   • Both backends tested at optimal concurrency levels"
echo ""
echo "✅ This test should accurately reflect your historical conditions!"