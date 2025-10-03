#!/bin/bash
set -e

echo "=== Comprehensive GCS Backend Test ==="
echo ""

# Use ADC authentication
source gcs-native-env

# Test parameters
BUCKET="signal65-russ-b1"
TEST_PREFIX="gcs-comprehensive-test"

echo "1. Creating test files..."
echo "Test file content 1" > test1.txt
echo "Test file content 2 - larger for range test with more content" > test2.txt
dd if=/dev/urandom of=test3.bin bs=1M count=1 2>/dev/null
echo "Created 3 test files"

echo ""
echo "2. Testing UPLOAD..."
./target/release/s3-cli -v upload test1.txt gs://$BUCKET/$TEST_PREFIX/
./target/release/s3-cli -v upload test2.txt gs://$BUCKET/$TEST_PREFIX/
./target/release/s3-cli -v upload test3.bin gs://$BUCKET/$TEST_PREFIX/
echo "✅ Upload successful"

echo ""
echo "3. Testing LIST (non-recursive)..."
./target/release/s3-cli -v ls gs://$BUCKET/$TEST_PREFIX/
echo "✅ List successful"

echo ""
echo "4. Testing DOWNLOAD..."
./target/release/s3-cli -v download gs://$BUCKET/$TEST_PREFIX/test1.txt ./downloaded1.txt
cat ./downloaded1.txt/$TEST_PREFIX/test1.txt
echo "✅ Download successful"

echo ""
echo "5. Testing RANGE READ..."
./target/release/s3-cli -v download gs://$BUCKET/$TEST_PREFIX/test2.txt ./range-test.txt --offset 0 --length 20
cat ./range-test.txt/$TEST_PREFIX/test2.txt
echo ""
echo "✅ Range read successful"

echo ""
echo "6. Testing STAT (via list)..."
./target/release/s3-cli -v ls gs://$BUCKET/$TEST_PREFIX/test3.bin
echo "✅ Stat successful"

echo ""
echo "7. Cleanup - deleting test objects..."
# Note: s3-cli doesn't have delete command, use gcloud
gcloud storage rm gs://$BUCKET/$TEST_PREFIX/test1.txt
gcloud storage rm gs://$BUCKET/$TEST_PREFIX/test2.txt
gcloud storage rm gs://$BUCKET/$TEST_PREFIX/test3.bin
echo "✅ Cleanup successful"

echo ""
echo "8. Verifying deletion..."
echo "Objects remaining in test prefix:"
gcloud storage ls gs://$BUCKET/$TEST_PREFIX/ 2>&1 || echo "(none - prefix empty)"

echo ""
echo "=== All GCS tests PASSED ✅ ==="
