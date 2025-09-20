#!/bin/bash
# scripts/test_enhanced_performance.sh
#
# Test script for enhanced performance features
# Tests all three enhancement phases with your proven methodology

set -euo pipefail

# Configuration
BUCKET_PREFIX="${S3DLIO_BUCKET_PREFIX:-s3dlio-perf-test}"
NUM_OBJECTS="${S3DLIO_NUM_OBJECTS:-2500}"
OBJECT_SIZE_MB="${S3DLIO_OBJECT_SIZE_MB:-10}"
TEST_DURATION="${S3DLIO_TEST_DURATION:-30}"
CONCURRENCY_LEVELS="${S3DLIO_CONCURRENCY_LEVELS:-24 32 48 64}"

# Feature flags to test
FEATURE_COMBINATIONS=(
    "baseline"                      # No enhanced features
    "enhanced-http"                 # HTTP/2 support only
    "io-uring"                      # io_uring only (Linux)
    "enhanced-http,io-uring"        # Both HTTP/2 and io_uring
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

warn() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check for Rust toolchain
    if ! command -v cargo >/dev/null 2>&1; then
        error "Cargo not found. Please install Rust toolchain."
        exit 1
    fi
    
    # Source AWS environment if aws-env file exists
    if [[ -f "aws-env" ]]; then
        log "Loading AWS environment from aws-env file"
        source aws-env
    fi
    
    # Check for AWS credentials
    if [[ -z "${AWS_ACCESS_KEY_ID:-}" ]] || [[ -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
        error "AWS credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY or provide aws-env file."
        exit 1
    fi
    
    # Check for S3 endpoint
    if [[ -z "${AWS_ENDPOINT_URL:-}" ]] && [[ -z "${S3_ENDPOINT:-}" ]]; then
        warn "No S3 endpoint specified. Using default AWS S3."
    fi
    
    success "Prerequisites check passed"
}

# Build binary with specific features
build_binary() {
    local features="$1"
    local binary_name="s3-cli-${features//,/-}"
    
    log "Building binary with features: ${features}"
    
    if [[ "$features" == "baseline" ]]; then
        cargo build --release --bin s3-cli
        cp target/release/s3-cli "target/release/${binary_name}"
    else
        cargo build --release --bin s3-cli --features "$features"
        cp target/release/s3-cli "target/release/${binary_name}"
    fi
    
    if [[ ! -f "target/release/${binary_name}" ]]; then
        error "Failed to build binary: ${binary_name}"
        return 1
    fi
    
    success "Built binary: ${binary_name}"
    echo "target/release/${binary_name}"
}

# Create test bucket
create_test_bucket() {
    local bucket_name="$1"
    local binary="$2"
    
    log "Creating test bucket: ${bucket_name}"
    
    # Try to create bucket (may already exist)
    if "${binary}" create-bucket "${bucket_name}" 2>/dev/null; then
        success "Created bucket: ${bucket_name}"
    else
        log "Bucket may already exist, continuing..."
    fi
}

# Cleanup test bucket
cleanup_bucket() {
    local bucket_name="$1"
    local binary="$2"
    
    log "Cleaning up bucket: ${bucket_name}"
    
    # Delete all objects first
    "${binary}" delete "${bucket_name}" --all --force 2>/dev/null || true
    
    # Delete bucket
    "${binary}" delete-bucket "${bucket_name}" 2>/dev/null || true
    
    log "Cleanup completed for: ${bucket_name}"
}

# Run performance test
run_performance_test() {
    local features="$1"
    local concurrency="$2"
    local binary="$3"
    local bucket_name="$4"
    
    log "Running performance test: ${features} (concurrency=${concurrency})"
    
    # Environment variables for enhanced features
    export S3DLIO_CONCURRENCY="$concurrency"
    export S3DLIO_PART_SIZE_MB="$OBJECT_SIZE_MB"
    export S3DLIO_ENABLE_OPTIMIZATIONS="true"
    
    # Feature-specific environment variables
    case "$features" in
        *enhanced-http*)
            export S3DLIO_ENABLE_HTTP2="true"
            ;;
        *io-uring*)
            export S3DLIO_URING_QUEUE_DEPTH="256"
            ;;
    esac
    
    local results_file="results_${features//,/-}_c${concurrency}.log"
    
    # Upload test
    log "Starting upload test (${NUM_OBJECTS} objects, ${OBJECT_SIZE_MB}MB each)"
    local upload_start=$(date +%s)
    
    "${binary}" put "${bucket_name}" \
        --count "$NUM_OBJECTS" \
        --size "${OBJECT_SIZE_MB}MB" \
        --concurrency "$concurrency" \
        --format npz \
        > "$results_file" 2>&1 &
    
    local upload_pid=$!
    
    # Monitor for test duration
    sleep "$TEST_DURATION"
    
    # Get throughput measurement
    local upload_end=$(date +%s)
    local upload_duration=$((upload_end - upload_start))
    
    if kill -0 "$upload_pid" 2>/dev/null; then
        # Still running, get current stats
        local uploaded_objects=$(ls -1 /tmp/s3dlio-* 2>/dev/null | wc -l || echo "0")
        local upload_throughput_mbps=$(echo "scale=2; $uploaded_objects * $OBJECT_SIZE_MB / $upload_duration" | bc -l)
        local upload_throughput_gbps=$(echo "scale=3; $upload_throughput_mbps / 1024" | bc -l)
        
        # Stop the process
        kill "$upload_pid" 2>/dev/null || true
        wait "$upload_pid" 2>/dev/null || true
    else
        # Process completed
        wait "$upload_pid"
        local uploaded_objects="$NUM_OBJECTS"
        local upload_throughput_mbps=$(echo "scale=2; $uploaded_objects * $OBJECT_SIZE_MB / $upload_duration" | bc -l)
        local upload_throughput_gbps=$(echo "scale=3; $upload_throughput_mbps / 1024" | bc -l)
    fi
    
    # Brief pause between tests
    sleep 3
    
    # Download test
    log "Starting download test"
    local download_start=$(date +%s)
    
    "${binary}" get "${bucket_name}" \
        --concurrency "$concurrency" \
        --output-dir "/tmp/s3dlio-download-$$" \
        >> "$results_file" 2>&1 &
    
    local download_pid=$!
    
    # Monitor for test duration
    sleep "$TEST_DURATION"
    
    # Get throughput measurement
    local download_end=$(date +%s)
    local download_duration=$((download_end - download_start))
    
    if kill -0 "$download_pid" 2>/dev/null; then
        # Still running, estimate from time
        local downloaded_objects=$(find "/tmp/s3dlio-download-$$" -name "*.npz" 2>/dev/null | wc -l || echo "0")
        local download_throughput_mbps=$(echo "scale=2; $downloaded_objects * $OBJECT_SIZE_MB / $download_duration" | bc -l)
        local download_throughput_gbps=$(echo "scale=3; $download_throughput_mbps / 1024" | bc -l)
        
        # Stop the process
        kill "$download_pid" 2>/dev/null || true
        wait "$download_pid" 2>/dev/null || true
    else
        # Process completed
        wait "$download_pid"
        local downloaded_objects=$(find "/tmp/s3dlio-download-$$" -name "*.npz" 2>/dev/null | wc -l || echo "$uploaded_objects")
        local download_throughput_mbps=$(echo "scale=2; $downloaded_objects * $OBJECT_SIZE_MB / $download_duration" | bc -l)
        local download_throughput_gbps=$(echo "scale=3; $download_throughput_mbps / 1024" | bc -l)
    fi
    
    # Cleanup download directory
    rm -rf "/tmp/s3dlio-download-$$" 2>/dev/null || true
    
    # Output results
    echo "${features},${concurrency},${upload_throughput_gbps},${download_throughput_gbps}" >> "performance_results.csv"
    
    log "Results for ${features} (concurrency=${concurrency}):"
    log "  Upload: ${upload_throughput_gbps} GB/s (${uploaded_objects} objects in ${upload_duration}s)"
    log "  Download: ${download_throughput_gbps} GB/s (${downloaded_objects} objects in ${download_duration}s)"
    
    # Unset feature-specific environment variables
    unset S3DLIO_ENABLE_HTTP2 S3DLIO_URING_QUEUE_DEPTH
}

# Generate performance report
generate_report() {
    log "Generating performance report..."
    
    if [[ ! -f "performance_results.csv" ]]; then
        error "No results file found"
        return 1
    fi
    
    local report_file="performance_report_$(date +%Y%m%d_%H%M%S).md"
    
    cat > "$report_file" << EOF
# S3DLIO Enhanced Performance Test Results

Test Configuration:
- Objects: ${NUM_OBJECTS} × ${OBJECT_SIZE_MB} MB
- Test Duration: ${TEST_DURATION} seconds per operation
- Concurrency Levels: ${CONCURRENCY_LEVELS}
- Date: $(date)

## Results Summary

| Feature Set | Concurrency | Upload (GB/s) | Download (GB/s) | Notes |
|-------------|-------------|---------------|-----------------|-------|
EOF
    
    # Process results
    while IFS=',' read -r features concurrency upload_gbps download_gbps; do
        echo "| ${features} | ${concurrency} | ${upload_gbps} | ${download_gbps} | |" >> "$report_file"
    done < performance_results.csv
    
    cat >> "$report_file" << EOF

## Analysis

### Historical Comparison
- Target Performance: 2.5-3.0 GB/s PUT, 5+ GB/s GET
- Previous Results: ~1.5-1.8 GB/s sustained

### Feature Impact
- **HTTP/2**: Expected improvement for non-AWS S3 implementations
- **io_uring**: Expected improvement on Linux systems with high I/O load
- **Adaptive Concurrency**: Expected optimization of concurrency levels based on throughput

### Recommendations
EOF
    
    # Find best performing configuration
    local best_upload=$(sort -t',' -k3 -nr performance_results.csv | head -1)
    local best_download=$(sort -t',' -k4 -nr performance_results.csv | head -1)
    
    echo "- Best Upload: $(echo "$best_upload" | cut -d',' -f1) at concurrency $(echo "$best_upload" | cut -d',' -f2)" >> "$report_file"
    echo "- Best Download: $(echo "$best_download" | cut -d',' -f1) at concurrency $(echo "$best_download" | cut -d',' -f2)" >> "$report_file"
    
    success "Report generated: ${report_file}"
}

# Main execution
main() {
    log "Starting enhanced performance test suite"
    
    check_prerequisites
    
    # Initialize results file
    echo "features,concurrency,upload_gbps,download_gbps" > performance_results.csv
    
    # Test each feature combination
    for features in "${FEATURE_COMBINATIONS[@]}"; do
        log "Testing feature combination: ${features}"
        
        # Skip io_uring on non-Linux systems
        if [[ "$features" == *io-uring* ]] && [[ "$(uname)" != "Linux" ]]; then
            warn "Skipping io_uring test on non-Linux system"
            continue
        fi
        
        # Build binary for this feature set
        local binary
        if ! binary=$(build_binary "$features"); then
            error "Failed to build binary for features: ${features}"
            continue
        fi
        
        # Create unique bucket for this test
        local bucket_name="${BUCKET_PREFIX}-${features//,/-}-$(date +%s)"
        create_test_bucket "$bucket_name" "$binary"
        
        # Test each concurrency level
        for concurrency in $CONCURRENCY_LEVELS; do
            log "Testing concurrency level: ${concurrency}"
            
            if ! run_performance_test "$features" "$concurrency" "$binary" "$bucket_name"; then
                error "Performance test failed for ${features} at concurrency ${concurrency}"
            fi
            
            # Brief pause between concurrency tests
            sleep 5
        done
        
        # Cleanup bucket
        cleanup_bucket "$bucket_name" "$binary"
        
        # Pause between feature combinations
        log "Completed testing ${features}, pausing..."
        sleep 10
    done
    
    generate_report
    
    success "Enhanced performance test suite completed!"
    log "Check performance_report_*.md for detailed results"
}

# Handle script interruption
trap 'error "Test interrupted"; exit 1' INT TERM

# Run main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi