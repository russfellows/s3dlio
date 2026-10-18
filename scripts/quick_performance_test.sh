#!/bin/bash
# scripts/quick_performance_test.sh  
#
# Quick performance test to investigate the performance discrepancy

set -e

echo "🔍 QUICK PERFORMANCE INVESTIGATION"
echo "================================="
echo ""
echo "🎯 Goal: Investigate why current performance (~350 MB/s) is much lower"  
echo "       than historical performance (2.5-3 GB/s) that you achieved."
echo ""

# Check binaries exist
if [[ ! -f target/performance_variants/s3-cli-native ]] || [[ ! -f target/performance_variants/s3-cli-arrow ]]; then
    echo "❌ Performance variants not found! Building them now..."
    ./scripts/build_performance_variants.sh
fi

# Load environment  
if [[ -z "$AWS_ACCESS_KEY_ID" ]]; then
    echo "🔧 Loading AWS environment..."
    source aws-env
fi

# Quick test parameters
S3_BUCKET="${S3DLIO_TEST_BUCKET:?set S3DLIO_TEST_BUCKET to an S3 bucket you own (no default; refusing to touch a live bucket)}"
TEST_PREFIX="quick-perf-$(date +%H%M%S)"
OBJECT_SIZE_10MB=$((10 * 1024 * 1024))
NUM_OBJECTS=20  # Smaller test for quick feedback
CONCURRENCY=64  # High concurrency like your historical tests

echo "📋 Test Configuration:"
echo "   Objects: $NUM_OBJECTS × 10MB = 200MB total"
echo "   Concurrency: $CONCURRENCY jobs"
echo "   Bucket: $S3_BUCKET"
echo ""

run_quick_test() {
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
    
    local test_uri="s3://$S3_BUCKET/$TEST_PREFIX/$backend/"
    
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
        
        if (( $(echo "$throughput >= 2000" | bc -l) )); then
            echo "   🎉 EXCELLENT - Near historical performance!"
        elif (( $(echo "$throughput >= 1000" | bc -l) )); then
            echo "   ✅ GOOD - Decent performance" 
        else
            echo "   ⚠️  INVESTIGATION NEEDED - Performance below expectations"
            echo "   🔍 Possible causes:"
            echo "      • Network bottleneck to S3 endpoint"
            echo "      • S3 endpoint performance limitations"
            echo "      • Code regression since early versions"
            echo "      • Different test conditions vs historical"
        fi
        
        echo ""
        echo "📥 GET Test..."
        local get_start=$(date +%s.%N)
        
        if $cli_binary get "$test_uri" -j $CONCURRENCY > /dev/null; then
            local get_end=$(date +%s.%N)
            local get_duration=$(echo "$get_end - $get_start" | bc -l)
            local get_throughput=$(echo "scale=2; $total_mb / $get_duration" | bc -l)
            
            echo "✅ GET Results:"
            echo "   Duration: ${get_duration}s"  
            echo "   Throughput: ${get_throughput} MB/s"
            echo "   Expected: 5000+ MB/s (5+ GB/s)"
            
            if (( $(echo "$get_throughput >= 4000" | bc -l) )); then
                echo "   🎉 EXCELLENT - Near historical performance!"
            elif (( $(echo "$get_throughput >= 2000" | bc -l) )); then
                echo "   ✅ GOOD - Decent performance"
            else
                echo "   ⚠️  INVESTIGATION NEEDED - GET performance below expectations"
            fi
        fi
        
        # Cleanup
        echo "🗑️ Cleaning up..."
        $cli_binary delete "$test_uri" > /dev/null 2>&1 || true
        
    else
        echo "❌ PUT test failed"
    fi
    
    echo ""
}

# Test both backends
run_quick_test "native" "./target/performance_variants/s3-cli-native"
run_quick_test "arrow" "./target/performance_variants/s3-cli-arrow"

echo "🎯 INVESTIGATION SUMMARY"
echo "======================="
echo ""
echo "If performance is significantly below historical levels:"
echo ""
echo "🔍 Next Steps:"
echo "   1. Test with different concurrency levels (try -j 128, -j 256)"
echo "   2. Test with larger datasets (100+ objects)"
echo "   3. Compare network conditions vs historical setup"
echo "   4. Profile code paths to identify bottlenecks"
echo ""
echo "💡 Your historical 2.5-3 GB/s PUT performance was excellent!"
echo "   Current results will help identify if there's been regression."
echo ""
echo "🚀 For comprehensive testing:"
echo "   ./scripts/cli_performance_validation.sh"