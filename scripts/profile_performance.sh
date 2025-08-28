#!/bin/bash
# Performance profiling script for s3dlio
# This script demonstrates various profiling approaches for identifying bottlenecks

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🔬 s3dlio Performance Profiling Suite"
echo "======================================"
echo

# Check if profiling feature is available
if ! grep -q 'profiling.*=' "$PROJECT_ROOT/Cargo.toml"; then
    echo "❌ Profiling feature not found in Cargo.toml"
    exit 1
fi

# Set up environment variables for profiling
export RUST_LOG="s3dlio=debug,info"
export S3DLIO_PROFILE_FREQ="${S3DLIO_PROFILE_FREQ:-100}"  # 100Hz sampling
export S3DLIO_HTTP_MAX_CONNECTIONS="${S3DLIO_HTTP_MAX_CONNECTIONS:-200}"
export S3DLIO_RUNTIME_THREADS="${S3DLIO_RUNTIME_THREADS:-}"  # Auto-detect

# Build with profiling enabled
echo "🔨 Building with profiling enabled..."
RUSTFLAGS="--cfg tokio_unstable" cargo build --release --features profiling

# Run microbenchmarks to establish baseline
echo "📊 Running microbenchmarks..."
cargo bench --bench s3_microbenchmarks -- --output-format html

echo "📈 Benchmark results saved to target/criterion/"

# Example: Profile a specific operation with standalone pprof
echo "🔥 Demonstrating standalone pprof profiling..."
cat > /tmp/profile_example.rs << 'EOF'
use s3dlio::profiling::*;
use s3dlio::data_gen::generate_controlled_data;
use std::fs::File;

fn main() -> anyhow::Result<()> {
    // Profile data generation operation
    let profiler = profile_section("data_generation")?;
    
    // Generate some test data (CPU-intensive operation)
    for _ in 0..100 {
        let _data = generate_controlled_data(1024 * 1024, 10, 1); // 1MB
    }
    
    // Save flamegraph
    profiler.save_flamegraph("target/data_generation_profile.svg")?;
    println!("📊 Flamegraph saved to: target/data_generation_profile.svg");
    
    Ok(())
}
EOF

echo "   Example profiling code created in /tmp/profile_example.rs"

# Demonstrate tracing profiling initialization
echo "🔍 Demonstrating tracing-based profiling..."
cat > /tmp/tracing_example.rs << 'EOF'
use s3dlio::profiling::*;
use s3dlio::data_gen::generate_controlled_data;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize comprehensive profiling
    init_profiling()?;
    
    // Use profiling macros
    let _span = profile_span!("main_operation", size = 1024*1024);
    
    // Profile an async operation
    let result = profile_async!("data_generation", async {
        generate_controlled_data(1024 * 1024, 10, 1)
    }).await;
    
    println!("✅ Generated {} bytes of test data", result.len());
    println!("📊 Check logs for tracing output");
    
    Ok(())
}
EOF

echo "   Example tracing code created in /tmp/tracing_example.rs"

# Performance testing recommendations
echo
echo "🚀 Performance Testing Recommendations:"
echo "======================================="
echo
echo "1. CPU Profiling (Flamegraphs):"
echo "   • Set S3DLIO_PROFILE_FREQ=100 (or higher for detail)"
echo "   • Build with: RUSTFLAGS='--cfg tokio_unstable' cargo build --release --features profiling"
echo "   • Use init_profiling() at start of your application"
echo "   • Flamegraphs show time distribution across functions"
echo
echo "2. Async Task Monitoring (tokio-console):"
echo "   • Set S3DLIO_TOKIO_CONSOLE=1"
echo "   • Run: tokio-console in another terminal"
echo "   • Monitor task scheduling, contention, and waker churn"
echo
echo "3. Microbenchmarks (criterion):"
echo "   • Run: cargo bench"
echo "   • Focus on inner loops and buffer operations"
echo "   • Compare before/after optimization changes"
echo
echo "4. System-level Profiling:"
echo "   • Linux: cargo flamegraph --release --features profiling"
echo "   • Monitor: perf stat for cache misses, IPC, page faults"
echo "   • Check: htop/top for CPU/memory usage patterns"
echo
echo "📋 Environment Variables for Tuning:"
echo "   S3DLIO_PROFILE_FREQ=100           # Sampling frequency (Hz)"
echo "   S3DLIO_TOKIO_CONSOLE=1            # Enable tokio-console"
echo "   S3DLIO_HTTP_MAX_CONNECTIONS=200   # HTTP pool size"
echo "   S3DLIO_RUNTIME_THREADS=16         # Tokio worker threads"
echo "   S3DLIO_RANGE_THRESHOLD_MB=4       # Concurrent range threshold"
echo "   S3DLIO_RANGE_CHUNK_SIZE_MB=8      # Range chunk size"
echo "   RUST_LOG=s3dlio=debug,info        # Logging level"
echo

# Performance testing workflow
echo "📝 Typical Performance Investigation Workflow:"
echo "============================================="
echo
echo "1. Establish Baseline:"
echo "   cargo bench                                    # Microbenchmarks"
echo "   time your_s3_workload                         # Wall-clock timing"
echo
echo "2. Profile CPU Hotspots:"
echo "   RUSTFLAGS='--cfg tokio_unstable' cargo run --release --features profiling"
echo "   # Check flamegraph for time distribution"
echo
echo "3. Profile Async Behavior:"
echo "   S3DLIO_TOKIO_CONSOLE=1 cargo run --release --features profiling &"
echo "   tokio-console                                  # In another terminal"
echo
echo "4. Tune and Retest:"
echo "   # Adjust environment variables based on findings"
echo "   # Re-run benchmarks to measure improvement"
echo
echo "5. System-level Analysis:"
echo "   perf stat -e cache-misses,cache-references,page-faults,context-switches your_program"
echo "   # Look for excessive cache misses or page faults"
echo

echo "✅ Profiling suite setup complete!"
echo "📁 Next steps:"
echo "   • Run 'cargo bench' to see microbenchmark results"
echo "   • Try the example profiling code in /tmp/"
echo "   • Instrument your own workloads with the profiling macros"
echo
