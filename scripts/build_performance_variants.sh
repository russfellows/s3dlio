#!/bin/bash
# scripts/build_performance_variants.sh
#
# Build multiple variants of s3dlio for performance comparison

set -e

echo "🏗️  BUILDING S3DLIO PERFORMANCE VARIANTS"
echo "========================================"

# Clean previous builds
echo "🧹 Cleaning previous builds..."
cargo clean

# Create output directory for binaries
mkdir -p target/performance_variants

echo ""
echo "🔥 Building NATIVE AWS SDK variant..."
echo "====================================="
cargo build --release --no-default-features --features native-backends --bin s3-cli
cp target/release/s3-cli target/performance_variants/s3-cli-native
echo "✅ Native variant built: target/performance_variants/s3-cli-native"

echo ""
echo "🏹 Building APACHE ARROW variant..."
echo "=================================="
cargo build --release --no-default-features --features arrow-backend --bin s3-cli  
cp target/release/s3-cli target/performance_variants/s3-cli-arrow
echo "✅ Arrow variant built: target/performance_variants/s3-cli-arrow"

echo ""
echo "📊 PERFORMANCE VARIANTS READY"
echo "============================="
echo ""
echo "Native AWS SDK CLI: ./target/performance_variants/s3-cli-native"
echo "Apache Arrow CLI:   ./target/performance_variants/s3-cli-arrow"
echo ""
echo "Example usage for PUT performance testing:"
echo "# Native backend - 100 objects, 10MB each, 64 concurrent jobs"
echo "./target/performance_variants/s3-cli-native put s3://my-bucket2/perf-test/native/ -n 100 -s 10485760 -j 64"
echo ""
echo "# Arrow backend - same test"
echo "./target/performance_variants/s3-cli-arrow put s3://my-bucket2/perf-test/arrow/ -n 100 -s 10485760 -j 64"
echo ""
echo "Example usage for GET performance testing:"
echo "./target/performance_variants/s3-cli-native get s3://my-bucket2/perf-test/native/ -j 64"
echo "./target/performance_variants/s3-cli-arrow get s3://my-bucket2/perf-test/arrow/ -j 64"
echo ""
echo "🎯 Ready for CLI-based performance validation!"