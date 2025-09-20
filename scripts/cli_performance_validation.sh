#!/bin/bash
# scripts/cli_performance_validation.sh
#
# Validate CLI performance against historical benchmarks
# Expected: 2.5-3 GB/s PUT, 5 GB/s GET with 10MB objects

set -e

# Configuration
S3_BUCKET="${S3_BUCKET:-my-bucket2}"
ENDPOINT_URL="${AWS_ENDPOINT_URL:-http://10.9.0.21}"
TEST_PREFIX="perf-validation-$(date +%Y%m%d_%H%M%S)"

# Test parameters that match historical testing
OBJECT_SIZE_10MB=$((10 * 1024 * 1024))  # 10MB objects  
NUM_OBJECTS=100                          # 100 objects = 1GB total
CONCURRENCY_LEVELS=(16 32 64 128 256)    # Test different concurrency levels

echo "🚀 CLI PERFORMANCE VALIDATION"
echo "============================="
echo ""
echo "Testing against historical benchmarks:"
echo "  Expected PUT: 2.5-3 GB/s with 10MB objects"
echo "  Expected GET: ~5 GB/s with 10MB objects"
echo ""
echo "Configuration:"
echo "  S3 Bucket: $S3_BUCKET"
echo "  Endpoint: $ENDPOINT_URL"
echo "  Object Size: 10MB"
echo "  Object Count: $NUM_OBJECTS"
echo "  Total Data: 1GB per test"
echo ""

# Check if variants exist
if [[ ! -f target/performance_variants/s3-cli-native ]] || [[ ! -f target/performance_variants/s3-cli-arrow ]]; then
    echo "❌ Performance variants not found!"
    echo "   Run: ./scripts/build_performance_variants.sh"
    exit 1
fi

# Ensure environment is loaded
if [[ -z "$AWS_ACCESS_KEY_ID" ]] || [[ -z "$AWS_SECRET_ACCESS_KEY" ]]; then
    echo "🔧 Loading AWS environment..."
    source aws-env || {
        echo "❌ Failed to load aws-env. Please ensure your S3 credentials are configured."
        exit 1
    }
fi

echo "✅ Environment loaded - AWS credentials configured"
echo ""

# Function to run performance test
run_perf_test() {
    local backend_name="$1"
    local cli_binary="$2"
    local concurrency="$3"
    local test_prefix="$4"
    
    echo "🎯 Testing $backend_name backend (concurrency: $concurrency)"
    echo "===========================================" 
    
    local put_prefix="s3://$S3_BUCKET/$test_prefix/$backend_name/j$concurrency"
    
    echo "📤 PUT Test: $NUM_OBJECTS objects × 10MB with $concurrency concurrent jobs"
    echo "URI: $put_prefix/"
    
    # Run PUT test and capture timing
    local start_time=$(date +%s.%N)
    if $cli_binary put "$put_prefix/" -n $NUM_OBJECTS -s $OBJECT_SIZE_10MB -j $concurrency; then
        local end_time=$(date +%s.%N)
        local duration=$(echo "$end_time - $start_time" | bc -l)
        local total_mb=$(echo "$NUM_OBJECTS * 10" | bc)
        local throughput_mbps=$(echo "scale=2; $total_mb / $duration" | bc -l)
        local throughput_gbps=$(echo "scale=3; $throughput_mbps / 1024" | bc -l)
        
        echo "✅ PUT Results:"
        echo "   Duration: ${duration}s"
        echo "   Throughput: ${throughput_mbps} MB/s (${throughput_gbps} GB/s)"
        
        # Check against historical benchmark  
        local threshold_mbps=2560  # 2.5 GB/s minimum
        if (( $(echo "$throughput_mbps >= $threshold_mbps" | bc -l) )); then
            echo "   🎉 MEETS HISTORICAL BENCHMARK (≥2.5 GB/s)"
        else
            echo "   ⚠️  BELOW HISTORICAL BENCHMARK (expected ≥2.5 GB/s)"
        fi
        
        # Brief pause before GET test
        sleep 5
        
        echo ""
        echo "📥 GET Test: Reading back $NUM_OBJECTS × 10MB objects"
        
        # Run GET test
        local get_start=$(date +%s.%N)
        if $cli_binary get "$put_prefix/" -j $concurrency > /dev/null; then
            local get_end=$(date +%s.%N)
            local get_duration=$(echo "$get_end - $get_start" | bc -l)
            local get_throughput_mbps=$(echo "scale=2; $total_mb / $get_duration" | bc -l)
            local get_throughput_gbps=$(echo "scale=3; $get_throughput_mbps / 1024" | bc -l)
            
            echo "✅ GET Results:"
            echo "   Duration: ${get_duration}s"
            echo "   Throughput: ${get_throughput_mbps} MB/s (${get_throughput_gbps} GB/s)"
            
            # Check against GET benchmark
            local get_threshold_mbps=5120  # 5 GB/s minimum
            if (( $(echo "$get_throughput_mbps >= $get_threshold_mbps" | bc -l) )); then
                echo "   🎉 MEETS HISTORICAL BENCHMARK (≥5 GB/s)"
            else
                echo "   ⚠️  BELOW HISTORICAL BENCHMARK (expected ≥5 GB/s)"
            fi
            
            # Store results for comparison
            echo "$backend_name,$concurrency,$throughput_mbps,$get_throughput_mbps" >> results/performance_comparison.csv
            
        else
            echo "❌ GET test failed"
        fi
        
        # Cleanup test objects
        echo "🗑️  Cleaning up test objects..."
        $cli_binary delete "$put_prefix/" > /dev/null 2>&1 || echo "   (cleanup failed - objects may remain)"
        
    else
        echo "❌ PUT test failed"
    fi
    
    echo ""
}

# Create results directory  
mkdir -p results
echo "Backend,Concurrency,PUT_MBps,GET_MBps" > results/performance_comparison.csv

echo "🔥 STARTING COMPREHENSIVE PERFORMANCE VALIDATION"
echo "==============================================="
echo ""

# Test both backends at different concurrency levels
for concurrency in "${CONCURRENCY_LEVELS[@]}"; do
    echo "🎯 CONCURRENCY LEVEL: $concurrency"
    echo "================================="
    
    # Test Native backend
    run_perf_test "Native" "./target/performance_variants/s3-cli-native" "$concurrency" "$TEST_PREFIX"
    
    # Test Arrow backend  
    run_perf_test "Arrow" "./target/performance_variants/s3-cli-arrow" "$concurrency" "$TEST_PREFIX"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
done

echo "📊 PERFORMANCE VALIDATION COMPLETE"
echo "=================================="
echo ""
echo "📈 Results saved to: results/performance_comparison.csv"
echo ""
echo "🔍 Quick Summary:"
cat results/performance_comparison.csv | column -t -s','
echo ""
echo "🎯 Key Findings:"
echo "   • Compare results against historical benchmarks:"
echo "   • Expected PUT: ≥2.5 GB/s (2560 MB/s)" 
echo "   • Expected GET: ≥5 GB/s (5120 MB/s)"
echo ""
echo "💡 If performance is below expectations, possible causes:"
echo "   1. Network/hardware differences vs historical setup"
echo "   2. S3 endpoint performance characteristics"  
echo "   3. Code changes affecting performance since early versions"
echo "   4. Concurrency level needs optimization"
echo ""
echo "✨ Performance validation complete!"