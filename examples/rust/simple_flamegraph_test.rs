#[cfg(feature = "profiling")]
use s3dlio::profiling::{init_profiling, profile_section};
#[cfg(feature = "profiling")]
use std::time::{Duration, Instant};
use tokio;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(not(feature = "profiling"))]
    {
        eprintln!("❌ This example requires the 'profiling' feature to be enabled.");
        eprintln!("   Run with: cargo run --example simple_flamegraph_test --features profiling");
        return Ok(());
    }
    
    #[cfg(feature = "profiling")]
    {
        // Initialize profiling
        init_profiling();
        
        println!("🔥 Starting simple flamegraph test...");
        
        // Start CPU profiling 
        let guard = pprof::ProfilerGuardBuilder::default()
            .frequency(1000)
            .blocklist(&["libc", "libgcc", "pthread", "vdso"])
            .build()
            .map_err(|e| format!("Failed to start profiler: {}", e))?;
    
    {
        let _upload_profile = profile_section("simulated_upload");
        simulate_work("Upload simulation", Duration::from_millis(200)).await;
        
        {
            let _nested_work = profile_section("nested_processing");
            simulate_work("Nested processing", Duration::from_millis(100)).await;
        }
        
        simulate_work("Upload finalization", Duration::from_millis(150)).await;
    }
    
    {
        let _download_profile = profile_section("simulated_download");
        simulate_work("Download simulation", Duration::from_millis(300)).await;
        
        for i in 0..3 {
            let _batch = profile_section(&format!("batch_{}", i));
            simulate_work(&format!("Batch {} processing", i), Duration::from_millis(50)).await;
        }
    }
    
    // Stop profiling and save flamegraph
    match guard.report().build() {
        Ok(report) => {
            // Save the flamegraph directly using pprof
            let file = std::fs::File::create("profiles/simple_test_profile.svg")?;
            report.flamegraph(file)?;
            println!("✅ Flamegraph saved to: profiles/simple_test_profile.svg");
        }
        Err(e) => {
            eprintln!("❌ Failed to generate profiling report: {}", e);
        }
    }
    
    println!("🎯 Simple flamegraph test completed!");
    }
}

#[cfg(feature = "profiling")]
async fn simulate_work(name: &str, duration: Duration) {
    let start = Instant::now();
    println!("  ⚡ {}: Starting work...", name);
    
    // Simulate CPU-intensive work
    let mut sum = 0u64;
    let end_time = start + duration;
    while Instant::now() < end_time {
        for i in 0..1000 {
            sum = sum.wrapping_add(i);
        }
        tokio::task::yield_now().await;
    }
    
    println!("  ✅ {}: Completed in {:?} (sum: {})", name, start.elapsed(), sum);
}
