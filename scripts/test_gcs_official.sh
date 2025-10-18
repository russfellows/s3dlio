#!/bin/bash
# Test script for gcs-official backend
# Run GCS functional tests using google-cloud-storage crate

set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  GCS Official Backend Test Suite                            ║${NC}"
echo -e "${BLUE}║  Backend: google-cloud-storage (gcs-official)                ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo

# Check environment
if [ -z "$GCS_TEST_BUCKET" ]; then
    echo -e "${RED}✗ GCS_TEST_BUCKET environment variable not set${NC}"
    echo "  Set it to a valid GCS bucket name you have access to:"
    echo "  export GCS_TEST_BUCKET=your-bucket-name"
    exit 1
fi

echo -e "Test Bucket: ${GREEN}${GCS_TEST_BUCKET}${NC}"
echo "Test Prefix: s3dlio-test/"
echo

# Check authentication
echo -e "${YELLOW}Checking GCS authentication...${NC}"
if gcloud auth application-default print-access-token &>/dev/null; then
    echo -e "${GREEN}✓ Using gcloud CLI authentication${NC}"
elif [ -n "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo -e "${GREEN}✓ Using service account: $GOOGLE_APPLICATION_CREDENTIALS${NC}"
else
    echo -e "${RED}✗ No GCS authentication found${NC}"
    echo "  Run: gcloud auth application-default login"
    exit 1
fi
echo

# Build with gcs-official
echo -e "${YELLOW}Building with gcs-official feature...${NC}"
cargo build --release --no-default-features --features native-backends,gcs-official

echo
echo -e "${YELLOW}Running functional tests with gcs-official...${NC}"
echo "(Running single-threaded to avoid race conditions)"
echo

cargo test --release --test test_gcs_official --no-default-features \
    --features native-backends,gcs-official -- --test-threads=1

if [ $? -eq 0 ]; then
    echo
    echo -e "${GREEN}✓ gcs-official tests PASSED${NC}"
    echo
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║  ✓ ALL TESTS PASSED                                          ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
    exit 0
else
    echo
    echo -e "${RED}✗ gcs-official tests FAILED${NC}"
    exit 1
fi
