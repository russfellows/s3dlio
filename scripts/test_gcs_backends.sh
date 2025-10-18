#!/bin/bash
# scripts/test_gcs_backends.sh
#
# Test both GCS backends (community vs official) against real Google Cloud Storage.
#
# Login to GCS using gcloud or service account before running.
# Command is: "gcloud auth application-default login"
#
# PREREQUISITES:
# 1. Set GCS_TEST_BUCKET environment variable
# 2. Authenticate via one of:
#    - gcloud auth application-default login
#    - GOOGLE_APPLICATION_CREDENTIALS env var
#    - GCE/GKE metadata server
#
# USAGE:
#   export GCS_TEST_BUCKET=your-test-bucket
#   ./scripts/test_gcs_backends.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if GCS_TEST_BUCKET is set
if [ -z "$GCS_TEST_BUCKET" ]; then
    echo -e "${RED}ERROR: GCS_TEST_BUCKET environment variable not set${NC}"
    echo ""
    echo "Please set it to your test bucket name:"
    echo "  export GCS_TEST_BUCKET=your-test-bucket"
    echo ""
    exit 1
fi

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  GCS Backend Comparison Test Suite                          ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Test Bucket: $GCS_TEST_BUCKET${NC}"
echo -e "${GREEN}Test Prefix: s3dlio-test/${NC}"
echo ""
echo -e "${YELLOW}Note: Tests will run single-threaded to ensure deterministic behavior${NC}"
echo -e "${YELLOW}Note: All test objects use the 's3dlio-test/' prefix and are cleaned up${NC}"
echo ""

# Check authentication
echo -e "${YELLOW}Checking GCS authentication...${NC}"
if [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo -e "${GREEN}✓ Using service account: $GOOGLE_APPLICATION_CREDENTIALS${NC}"
elif command -v gcloud &> /dev/null && gcloud auth application-default print-access-token &> /dev/null; then
    echo -e "${GREEN}✓ Using gcloud CLI authentication${NC}"
else
    echo -e "${RED}ERROR: No GCS authentication found${NC}"
    echo ""
    echo "Please authenticate using one of:"
    echo "  1. gcloud auth application-default login"
    echo "  2. export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json"
    echo ""
    exit 1
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Test 1: gcs-community (gcloud-storage) Backend${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Build with community backend
echo -e "${YELLOW}Building with gcs-community feature...${NC}"
cargo build --release --features gcs-community

# Run tests with community backend
echo ""
echo -e "${YELLOW}Running functional tests with gcs-community...${NC}"
echo -e "${YELLOW}(Running single-threaded to avoid race conditions)${NC}"
cargo test --test test_gcs_functional --release --features gcs-community -- --test-threads=1 --nocapture

COMMUNITY_EXIT_CODE=$?

if [ $COMMUNITY_EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ gcs-community tests PASSED${NC}"
else
    echo ""
    echo -e "${RED}✗ gcs-community tests FAILED${NC}"
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Test 2: gcs-official (google-cloud-storage) Backend${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

# Build with official backend
echo -e "${YELLOW}Building with gcs-official feature...${NC}"
cargo build --release --no-default-features --features native-backends,s3,gcs-official

# Run tests with official backend
echo ""
echo -e "${YELLOW}Running functional tests with gcs-official...${NC}"
echo -e "${YELLOW}(Running single-threaded to avoid race conditions)${NC}"
echo ""

set +e  # Don't exit on error for official backend (may still have issues)
cargo test --test test_gcs_functional --release --no-default-features --features native-backends,s3,gcs-official -- --test-threads=1 --nocapture
OFFICIAL_EXIT_CODE=$?
set -e

if [ $OFFICIAL_EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ gcs-official tests PASSED${NC}"
else
    echo ""
    echo -e "${RED}✗ gcs-official tests FAILED${NC}"
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Test Summary${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
echo ""

if [ $COMMUNITY_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ gcs-community (gcloud-storage):       PASSED${NC}"
else
    echo -e "${RED}✗ gcs-community (gcloud-storage):       FAILED${NC}"
fi

if [ $OFFICIAL_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ gcs-official (google-cloud-storage): PASSED${NC}"
else
    echo -e "${RED}✗ gcs-official (google-cloud-storage): FAILED${NC}"
fi

echo ""

# Overall result
if [ $COMMUNITY_EXIT_CODE -eq 0 ] && [ $OFFICIAL_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✓ BOTH BACKENDS PASSED                                      ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    exit 0
elif [ $COMMUNITY_EXIT_CODE -eq 0 ]; then
    echo -e "${YELLOW}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${YELLOW}║  Working Backend: gcs-community (gcs-official has issues)    ║${NC}"
    echo -e "${YELLOW}╚══════════════════════════════════════════════════════════════╝${NC}"
    exit 1
else
    echo -e "${RED}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  TESTS FAILED                                                ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════════╝${NC}"
    exit 1
fi
