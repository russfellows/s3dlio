#!/bin/bash
# Test runner for TFRecord indexing implementation
# Runs all Rust and Python tests

set -e  # Exit on first error

echo "========================================================================"
echo "TFRecord Indexing - Comprehensive Test Suite"
echo "========================================================================"
echo ""

# Color codes
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Track results
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# Function to print test result
print_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✓ PASS${NC}: $2"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        echo -e "${RED}✗ FAIL${NC}: $2"
        FAILED_TESTS=$((FAILED_TESTS + 1))
    fi
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
}

# Test 1: Rust unit tests
echo -e "${BLUE}[1/4] Running Rust Unit Tests...${NC}"
echo "--------------------------------------------------------------------"
if cargo test --lib tfrecord_index::tests --release --quiet 2>&1 | grep -q "9 passed"; then
    print_result 0 "Rust unit tests (9 tests)"
else
    print_result 1 "Rust unit tests"
fi
echo ""

# Test 2: Check for compilation warnings
echo -e "${BLUE}[2/4] Checking for Compilation Warnings...${NC}"
echo "--------------------------------------------------------------------"
if cargo build --release 2>&1 | grep -qi "warning"; then
    print_result 1 "Zero warnings policy"
    cargo build --release 2>&1 | grep -i warning
else
    print_result 0 "Zero warnings policy"
fi
echo ""

# Test 3: Python integration tests
echo -e "${BLUE}[3/4] Running Python Integration Tests...${NC}"
echo "--------------------------------------------------------------------"
if python tests/test_tfrecord_index_python.py >/dev/null 2>&1; then
    print_result 0 "Python API tests (4 tests)"
else
    print_result 1 "Python API tests"
    python tests/test_tfrecord_index_python.py
fi
echo ""

# Test 4: DALI compatibility tests
echo -e "${BLUE}[4/4] Running DALI Compatibility Tests...${NC}"
echo "--------------------------------------------------------------------"
if python tests/test_dali_compatibility.py >/dev/null 2>&1; then
    print_result 0 "DALI compatibility tests (4 tests)"
else
    print_result 1 "DALI compatibility tests"
    python tests/test_dali_compatibility.py
fi
echo ""

# Summary
echo "========================================================================"
echo "TEST SUMMARY"
echo "========================================================================"
echo "Total test suites: $TOTAL_TESTS"
echo -e "Passed: ${GREEN}$PASSED_TESTS${NC}"
if [ $FAILED_TESTS -gt 0 ]; then
    echo -e "Failed: ${RED}$FAILED_TESTS${NC}"
fi
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
    echo "========================================================================"
    echo ""
    echo "TFRecord indexing implementation is ready for use:"
    echo "  - 9 Rust unit tests passing"
    echo "  - 4 Python integration tests passing"
    echo "  - 4 DALI compatibility tests passing"
    echo "  - Zero compilation warnings"
    echo "  - Format validated against NVIDIA DALI spec"
    echo ""
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED${NC}"
    echo "========================================================================"
    exit 1
fi
