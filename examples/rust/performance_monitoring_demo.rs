//! Performance monitoring and caching demonstration
//!
//! This example demonstrates the enhanced performance monitoring and caching capabilities
//! using HDR histograms for precise tail latency analysis in AI/ML workloads.

use anyhow::Result;
use std::time::Instant;
use s3dlio::metrics::enhanced::{
    MetricsConfig, HistogramConfig, EnhancedMetricsCollector,
    init_global_metrics, record_operation, record_error, print_global_report
};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    
    // Initialize global metrics with high throughput support for all demos
    let global_config = MetricsConfig {
        histogram_config: HistogramConfig {
            significant_digits: 3,
            max_value: 1_000_000_000_000, // Support up to 1TB/s throughput measurements
        },
        per_operation_tracking: true,
        sampling_rate: 1.0,
        batch_size: 100,
    };
    init_global_metrics(global_config);
    
    println!("🚀 s3dlio Performance Monitoring Demo");
    println!("=====================================\n");
    
    // Demo 1: HDR histogram performance monitoring
    demo_hdr_performance_monitoring().await?;
    
    // Demo 2: AI/ML workload simulation with realistic access patterns
    demo_aiml_workload_simulation().await?;
    
    println!("✅ All demos completed successfully!");
    Ok(())
}

/// Demonstrate HDR histogram performance monitoring
async fn demo_hdr_performance_monitoring() -> Result<()> {
    println!("📊 Demo 1: HDR Histogram Performance Monitoring");
    println!("-----------------------------------------------");
    
    // Create custom metrics configuration for AI/ML training with high throughput support
    let custom_config = MetricsConfig {
        histogram_config: HistogramConfig {
            significant_digits: 3,
            max_value: 1_000_000_000_000, // Support up to 1TB/s throughput measurements
        },
        per_operation_tracking: true,
        sampling_rate: 1.0,
        batch_size: 100,
    };
    let mut collector = EnhancedMetricsCollector::new(custom_config);
    
    println!("Recording simulated S3 operations with various latencies...");
    
    // Simulate different operation types with realistic latencies
    let operations = [
        ("s3_get_small", 100, 1000, 1024),        // Small file: 100μs, 1KB
        ("s3_get_medium", 500, 5000, 1024*1024),  // Medium file: 500μs, 1MB  
        ("s3_get_large", 2000, 10000, 100*1024*1024), // Large file: 2ms, 100MB
        ("s3_put_small", 200, 1000, 1024),        // Small upload: 200μs
        ("s3_put_medium", 800, 5000, 1024*1024),  // Medium upload: 800μs
        ("s3_put_large", 5000, 10000, 100*1024*1024), // Large upload: 5ms
    ];
    
    // Record operations with some variance
    for _ in 0..1000 {
        for (op_name, base_latency, request_size, response_size) in &operations {
            // Add realistic variance to latencies
            let variance = (rand::random::<f64>() - 0.5) * 0.3; // ±15% variance
            let latency = (*base_latency as f64 * (1.0 + variance)) as u64;
            
            collector.record_operation(op_name, latency, *request_size, *response_size)?;
            
            // Occasionally record errors (5% error rate)
            if rand::random::<f64>() < 0.05 {
                collector.record_error(op_name);
            }
        }
    }
    
    // Generate and display performance report
    let report = collector.generate_report();
    report.print_summary();
    
    // Demonstrate percentile analysis for a specific operation
    if let Some(latency_summary) = collector.get_latency_summary("s3_get_large") {
        println!("🎯 Detailed Analysis for 's3_get_large' operation:");
        println!("   P50 (median): {}μs", latency_summary.p50);
        println!("   P90: {}μs", latency_summary.p90);
        println!("   P95: {}μs", latency_summary.p95);
        println!("   P99: {}μs", latency_summary.p99);
        println!("   P99.9: {}μs", latency_summary.p999);
        println!("   P99.99: {}μs", latency_summary.p9999);
        println!("   Mean: {:.2}μs", latency_summary.mean);
        println!("   StdDev: {:.2}μs", latency_summary.stdev);
    }
    
    println!("\n✅ HDR performance monitoring demo complete!\n");
    Ok(())
}

/// Demonstrate intelligent caching with pattern learning
/// Demonstrate AI/ML workload simulation with realistic patterns
async fn demo_aiml_workload_simulation() -> Result<()> {
    println!("🤖 Demo 2: AI/ML Workload Simulation");
    println!("------------------------------------");
    
    println!("Simulating AI/ML training and inference workloads...");
    
    // Simulate typical AI/ML operations with realistic latencies
    let operations = [
        ("data_loading", 50, 10*1024*1024, 10*1024*1024),      // 10MB dataset loading
        ("preprocessing", 1000, 5*1024*1024, 5*1024*1024),     // Data preprocessing  
        ("model_forward", 2000, 1024*1024, 1024*1024),         // Forward pass
        ("gradient_compute", 3000, 1024*1024, 1024*1024),      // Gradient computation
        ("weight_update", 500, 1024*1024, 1024*1024),          // Weight updates
        ("checkpoint_save", 5000, 50*1024*1024, 50*1024*1024), // Model checkpointing
    ];
    
    println!("  Simulating training epoch...");
    let start_time = Instant::now();
    
    // Simulate 100 training steps
    for step in 0..100 {
        for (op_name, base_latency, request_size, response_size) in &operations {
            // Add realistic variance (±20%)
            let variance = (rand::random::<f64>() - 0.5) * 0.4;
            let latency = (*base_latency as f64 * (1.0 + variance)) as u64;
            
            record_operation(op_name, latency, *request_size, *response_size)?;
            
            // Occasionally record errors (2% error rate)
            if rand::random::<f64>() < 0.02 {
                record_error(op_name);
            }
        }
        
        if step % 20 == 0 {
            println!("    Completed {} training steps...", step + 1);
        }
    }
    
    let training_time = start_time.elapsed();
    
    // Simulate inference workload
    println!("  Simulating inference workload...");
    let inference_start = Instant::now();
    
    for request_id in 0..500 {
        // Simulate inference latency with batch processing benefits
        let batch_size = if request_id % 10 == 0 { 32 } else { 1 };
        let base_latency = if batch_size > 1 { 800 } else { 1200 }; // Batching is more efficient
        
        let variance = (rand::random::<f64>() - 0.5) * 0.3;
        let latency = (base_latency as f64 * (1.0 + variance)) as u64;
        
        record_operation("ai_inference", latency, 1024*1024, 1024)?;
        
        // Simulate occasional errors (1% error rate)
        if rand::random::<f64>() < 0.01 {
            record_error("ai_inference");
        }
        
        if request_id % 100 == 0 && request_id > 0 {
            println!("    Processed {} inference requests...", request_id);
        }
    }
    
    let inference_time = inference_start.elapsed();
    let total_time = start_time.elapsed();
    
    println!("\n🎯 AI/ML Workload Results:");
    println!("   Training Time: {:.2}s", training_time.as_secs_f64());
    println!("   Inference Time: {:.2}s", inference_time.as_secs_f64()); 
    println!("   Total Time: {:.2}s", total_time.as_secs_f64());
    println!("   Inference Requests/Second: {:.2}", 500.0 / inference_time.as_secs_f64());
    
    // Display global performance report
    print_global_report();
    
    println!("\n✅ AI/ML workload simulation complete!\n");
    Ok(())
}