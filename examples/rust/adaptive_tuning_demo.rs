// examples/adaptive_tuning_demo.rs
//
// Demonstrates optional adaptive tuning in s3dlio
// Shows how explicit settings ALWAYS override adaptive behavior

use s3dlio::api::{AdaptiveConfig, WorkloadType, AdaptiveParams, WriterOptions, LoaderOptions};

fn main() {
    println!("=== s3dlio Adaptive Tuning Demo ===\n");
    
    // Example 1: Adaptive tuning DISABLED (default)
    println!("1. Default WriterOptions (adaptive DISABLED):");
    let opts_default = WriterOptions::new();
    let part_size = opts_default.effective_part_size(Some(500 * 1024 * 1024));
    println!("   Part size for 500 MB file: {} MB", part_size / (1024 * 1024));
    println!("   Result: Uses default (16 MB)\n");
    
    // Example 2: Adaptive tuning ENABLED
    println!("2. WriterOptions with adaptive ENABLED:");
    let opts_adaptive = WriterOptions::new().with_adaptive();
    let part_size_small = opts_adaptive.effective_part_size(Some(10 * 1024 * 1024));
    let part_size_large = opts_adaptive.effective_part_size(Some(1024 * 1024 * 1024));
    println!("   Part size for 10 MB file: {} MB", part_size_small / (1024 * 1024));
    println!("   Part size for 1 GB file: {} MB", part_size_large / (1024 * 1024));
    println!("   Result: Adaptive picks 8 MB for small, 32 MB for large\n");
    
    // Example 3: Explicit setting OVERRIDES adaptive
    println!("3. Explicit part_size ALWAYS overrides adaptive:");
    let opts_explicit = WriterOptions::new()
        .with_adaptive()
        .with_part_size(20 * 1024 * 1024); // Explicit 20 MB
    let part_size_override = opts_explicit.effective_part_size(Some(1024 * 1024 * 1024));
    println!("   Part size for 1 GB file (explicit 20 MB set): {} MB", part_size_override / (1024 * 1024));
    println!("   Result: Explicit 20 MB used, adaptive ignored\n");
    
    // Example 4: LoaderOptions with adaptive concurrency
    println!("4. LoaderOptions with adaptive concurrency:");
    let loader_adaptive = LoaderOptions::default().with_adaptive();
    let concurrency = loader_adaptive.effective_concurrency(Some(WorkloadType::SmallFile));
    println!("   Concurrency for small files: {}", concurrency);
    println!("   Result: Adaptive picks high concurrency (8x CPU count)\n");
    
    // Example 5: Explicit num_workers OVERRIDES adaptive
    println!("5. Explicit num_workers ALWAYS overrides adaptive:");
    let loader_explicit = LoaderOptions::default()
        .with_adaptive()
        .num_workers(16); // Explicit 16 workers
    let concurrency_override = loader_explicit.effective_concurrency(Some(WorkloadType::SmallFile));
    println!("   Concurrency for small files (explicit 16 set): {}", concurrency_override);
    println!("   Result: Explicit 16 used, adaptive ignored\n");
    
    // Example 6: Custom adaptive configuration
    println!("6. Custom adaptive configuration with bounds:");
    let custom_adaptive = AdaptiveConfig::enabled()
        .with_part_size_bounds(10 * 1024 * 1024, 40 * 1024 * 1024) // 10-40 MB
        .with_concurrency_bounds(4, 32); // 4-32 workers
    let opts_custom = WriterOptions::new().with_adaptive_config(custom_adaptive);
    let part_size_bounded = opts_custom.effective_part_size(Some(2 * 1024 * 1024 * 1024));
    println!("   Part size for 2 GB file (max bound 40 MB): {} MB", part_size_bounded / (1024 * 1024));
    println!("   Result: Would be 64 MB, but clamped to max bound (40 MB)\n");
    
    // Example 7: Direct use of AdaptiveParams
    println!("7. Direct use of AdaptiveParams:");
    let adaptive_cfg = AdaptiveConfig::enabled();
    let params = AdaptiveParams::new(adaptive_cfg);
    
    // Compute part sizes for different file sizes
    let sizes = vec![
        (5 * 1024 * 1024, "5 MB"),
        (50 * 1024 * 1024, "50 MB"),
        (500 * 1024 * 1024, "500 MB"),
    ];
    
    println!("   Adaptive part sizes:");
    for (size, label) in sizes {
        let part = params.compute_part_size(Some(size), None);
        println!("     {} file -> {} MB parts", label, part / (1024 * 1024));
    }
    
    // Compute concurrency for different workload types
    println!("\n   Adaptive concurrency:");
    let workloads = vec![
        (WorkloadType::SmallFile, "Small files"),
        (WorkloadType::MediumFile, "Medium files"),
        (WorkloadType::LargeFile, "Large files"),
        (WorkloadType::Batch, "Batch operations"),
    ];
    
    for (workload, label) in workloads {
        let concurrency = params.compute_concurrency(Some(workload), None);
        println!("     {} -> {} workers", label, concurrency);
    }
    
    println!("\n=== Key Takeaways ===");
    println!("✓ Adaptive tuning is OPTIONAL (disabled by default)");
    println!("✓ Explicit settings ALWAYS override adaptive behavior");
    println!("✓ Use .with_adaptive() to enable adaptive tuning");
    println!("✓ Adaptive picks optimal values based on workload characteristics");
    println!("✓ You maintain full control - adaptive is a convenience, not a requirement");
}
