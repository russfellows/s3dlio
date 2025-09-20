#!/bin/bash
# scripts/run_backend_comparison.sh
#
# Script to run the same performance tests on both backends for comparison

set -e

echo "🚀 BACKEND PERFORMANCE COMPARISON SUITE"
echo "========================================"
echo ""
echo "This script will run identical performance tests on both:"
echo "1. Apache Arrow object_store backend"  
echo "2. Native AWS SDK backend"
echo ""
echo "Each test runs for 2+ minutes per operation (PUT/GET) for each object size (1MB, 10MB)"
echo "Total estimated time: ~16+ minutes"
echo ""

# Function to run test with specific backend
run_backend_test() {
    local backend_name="$1"
    local features="$2"
    local log_file="$3"
    
    echo "🎯 Testing $backend_name Backend"
    echo "================================"
    echo "Features: $features"
    echo "Log file: $log_file"
    echo ""
    
    # Run the test and capture both stdout and stderr
    if [[ "$backend_name" == "Apache Arrow" ]]; then
        test_name="test_arrow_backend_performance"
    else
        test_name="test_native_backend_performance"
    fi
    
    if cargo test $features "$test_name" -- --nocapture > "$log_file" 2>&1; then
        echo "✅ $backend_name test completed successfully"
        echo ""
        # Extract and display summary
        echo "📊 $backend_name Results Summary:"
        echo "--------------------------------"
        grep -A 20 "PERFORMANCE SUMMARY TABLE" "$log_file" | head -25 || echo "Summary extraction failed"
        echo ""
    else
        echo "❌ $backend_name test failed"
        echo "Error details:"
        tail -20 "$log_file"
        echo ""
    fi
}

# Create results directory
mkdir -p results
timestamp=$(date +"%Y%m%d_%H%M%S")

# Test Arrow Backend
echo "Phase 1: Apache Arrow Backend"
echo "=============================="
run_backend_test "Apache Arrow" "--no-default-features --features s3,arrow-backend" "results/arrow_performance_$timestamp.log"

# Brief pause between tests
sleep 10

# Test Native AWS Backend  
echo "Phase 2: Native AWS SDK Backend"
echo "==============================="
run_backend_test "Native AWS SDK" "--no-default-features --features s3,native-backends" "results/native_performance_$timestamp.log"

echo "🏁 COMPARISON COMPLETE"
echo "====================="
echo ""
echo "Results saved to:"
echo "- Arrow:  results/arrow_performance_$timestamp.log"
echo "- Native: results/native_performance_$timestamp.log"
echo ""

# Try to generate a comparison summary
echo "📊 BACKEND COMPARISON SUMMARY"
echo "============================="
echo ""

echo "Arrow Backend Results:"
echo "---------------------"
grep -A 10 "PERFORMANCE SUMMARY TABLE" "results/arrow_performance_$timestamp.log" | grep -E "MB|TOTALS" | head -5 || echo "Could not extract Arrow summary"
echo ""

echo "Native Backend Results:"  
echo "----------------------"
grep -A 10 "PERFORMANCE SUMMARY TABLE" "results/native_performance_$timestamp.log" | grep -E "MB|TOTALS" | head -5 || echo "Could not extract Native summary"
echo ""

echo "💡 For detailed results, see the log files above."
echo "💡 Look for 'PERFORMANCE SUMMARY TABLE' section in each log."