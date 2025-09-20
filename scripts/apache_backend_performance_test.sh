#!/bin/bash
#
# apache_backend_performance_test.sh
# Same tests as AWS SDK but with Apache Arrow object_store backend

set -euo pipefail

# Configuration - same as AWS tests
OBJECT_COUNT=5000
OBJECT_SIZE_MB=10
OBJECT_SIZE_BYTES=$((OBJECT_SIZE_MB * 1024 * 1024))
MIN_TEST_DURATION=60
TEST_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Load AWS environment (same credentials work for Apache backend)
source aws-env

echo "=========================================="
echo "S3DLIO Apache Backend Performance Test"
echo "=========================================="
TOTAL_MB=$((OBJECT_COUNT * OBJECT_SIZE_MB))
TOTAL_GB=$(echo "scale=1; $TOTAL_MB / 1024" | bc -l)
echo "Objects: $OBJECT_COUNT × ${OBJECT_SIZE_MB} MiB"
echo "Total data: $TOTAL_MB MiB ($TOTAL_GB GiB)"
echo "Backend: Apache Arrow object_store"
echo "Minimum test duration: ${MIN_TEST_DURATION}s"
echo "Timestamp: $TEST_TIMESTAMP"
echo "=========================================="

# Set enhanced performance environment (same settings)
export S3DLIO_CONCURRENCY=48
export S3DLIO_TARGET_GBPS=3.0
export S3DLIO_ENABLE_HTTP2=true
export S3DLIO_ENABLE_IO_URING=true
export S3DLIO_PART_SIZE_MB=10

# Test configurations for Apache backend
declare -a APACHE_CONFIGS=(
    "arrow-backend:Apache baseline"
    "arrow-backend,enhanced-http:Apache + HTTP/2"
    "arrow-backend,io-uring:Apache + io_uring"
    "arrow-backend,enhanced-http,io-uring:Apache full enhanced"
)

# Results
RESULTS_FILE="apache_backend_results_${TEST_TIMESTAMP}.csv"
echo "config,direction,duration_s,throughput_gbps,objects_processed,avg_latency_ms" > "$RESULTS_FILE"

# Function for proper bucket cleanup
cleanup_bucket() {
    local bucket="$1"
    local binary="$2"
    echo "🧹 Cleaning up bucket $bucket..."
    
    # Delete all objects recursively
    echo "  Deleting all objects..."
    "$binary" delete "s3://$bucket/" --recursive --jobs 1000 || {
        echo "  ⚠️  Object deletion failed, continuing..."
    }
    
    # Delete the bucket
    echo "  Deleting bucket..."
    "$binary" delete-bucket "$bucket" || {
        echo "  ⚠️  Bucket deletion failed"
    }
    echo "  ✅ Cleanup complete"
}

# Function to run Apache backend performance test
run_apache_test() {
    local config="$1"
    local description="$2"
    local bucket="s3dlio-apache-${config//,/-}-$(date +%H%M%S)"
    local binary="./target/release/s3-cli"
    
    echo ""
    echo "🧪 Testing: $description"
    echo "📊 Configuration: $config"
    echo "🪣 Bucket: $bucket"
    
    # Build with Apache backend features
    echo "  Building with Apache features: $config"
    cargo build --release --no-default-features --features "$config" --bin s3-cli
    
    # Create bucket
    echo "🪣 Creating bucket..."
    "$binary" create-bucket "$bucket"
    
    # PUT Test (Upload)
    echo ""
    echo "⬆️  PUT Test - Uploading $OBJECT_COUNT objects..."
    put_start=$(date +%s.%N)
    
    "$binary" put "s3://$bucket/perf-test/" \
        --num "$OBJECT_COUNT" \
        --size "${OBJECT_SIZE_BYTES}" \
        --jobs 48 \
        --template "object_{}_of_{}.bin" \
        --object-type RAW
    
    put_end=$(date +%s.%N)
    put_duration=$(echo "$put_end - $put_start" | bc -l)
    put_throughput=$(echo "scale=3; ($OBJECT_COUNT * $OBJECT_SIZE_BYTES) / ($put_duration * 1000000000)" | bc -l)
    put_latency=$(echo "scale=2; $put_duration * 1000 / $OBJECT_COUNT" | bc -l)
    
    echo "✅ PUT Results:"
    echo "   Duration: ${put_duration}s"
    echo "   Throughput: ${put_throughput} GB/s"
    echo "   Objects: $OBJECT_COUNT"
    echo "   Avg Latency: ${put_latency} ms per object"
    
    # Record PUT results
    echo "$config,PUT,$put_duration,$put_throughput,$OBJECT_COUNT,$put_latency" >> "$RESULTS_FILE"
    
    echo "💤 Sleeping 30s between PUT and GET..."
    sleep 30
    
    # GET Test (Download to memory)
    echo ""
    echo "⬇️  GET Test - Fetching $OBJECT_COUNT objects to memory..."
    get_start=$(date +%s.%N)
    
    "$binary" get "s3://$bucket/perf-test/" \
        --jobs 48 \
        --recursive
    
    get_end=$(date +%s.%N)
    get_duration=$(echo "$get_end - $get_start" | bc -l)
    get_throughput=$(echo "scale=3; ($OBJECT_COUNT * $OBJECT_SIZE_BYTES) / ($get_duration * 1000000000)" | bc -l)
    get_latency=$(echo "scale=2; $get_duration * 1000 / $OBJECT_COUNT" | bc -l)
    
    echo "✅ GET Results:"
    echo "   Duration: ${get_duration}s"
    echo "   Throughput: ${get_throughput} GB/s"
    echo "   Objects: $OBJECT_COUNT"
    echo "   Avg Latency: ${get_latency} ms per object"
    
    # Record GET results
    echo "$config,GET,$get_duration,$get_throughput,$OBJECT_COUNT,$get_latency" >> "$RESULTS_FILE"
    
    # Summary for this configuration
    echo ""
    echo "📊 $description Summary:"
    echo "   PUT: ${put_throughput} GB/s (${put_duration}s)"
    echo "   GET: ${get_throughput} GB/s (${get_duration}s)"
    echo "   Total data: $((OBJECT_COUNT * OBJECT_SIZE_MB)) MiB"
    
    # Mandatory cleanup
    cleanup_bucket "$bucket" "$binary"
    
    echo "💤 Sleeping 30s before next configuration..."
    sleep 30
}

# Main execution
echo "Starting Apache backend performance tests..."

for config_entry in "${APACHE_CONFIGS[@]}"; do
    IFS=':' read -r config description <<< "$config_entry"
    run_apache_test "$config" "$description"
done

# Generate summary report
echo ""
echo "🎉 All Apache backend tests completed!"
echo ""
echo "📊 Performance Summary:"
echo "----------------------------------------"

# Display results
column -t -s',' "$RESULTS_FILE" | head -1
echo "----------------------------------------"
column -t -s',' "$RESULTS_FILE" | tail -n +2

echo ""
echo "📄 Detailed results saved to: $RESULTS_FILE"
echo ""
echo "🔍 Analysis:"
echo "  - Tests ran with Apache Arrow object_store backend"
echo "  - Tests ran with $OBJECT_COUNT objects ($TOTAL_GB GiB total)"
echo "  - Each test duration was measured for sustained throughput"
echo "  - Compare with AWS SDK results from long_duration_results_*.csv"

# Check if we met minimum duration
min_duration=$(awk -F',' 'NR>1 {if($3 > max || max=="") max=$3} END {print max}' "$RESULTS_FILE")
duration_check=$(echo "$min_duration >= $MIN_TEST_DURATION" | bc -l)
if (( duration_check == 1 )); then
    echo "  ✅ Tests met minimum ${MIN_TEST_DURATION}s duration requirement"
else
    echo "  ⚠️  Consider increasing object count for longer test duration"
fi

echo ""
echo "Next steps:"
echo "  1. Compare with AWS SDK results: long_duration_results_20250920_120557.csv"
echo "  2. Analyze backend-specific performance characteristics"
echo "  3. Evaluate which backend + feature combination works best for your use case"