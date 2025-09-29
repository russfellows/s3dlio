//! Performance metrics and monitoring infrastructure
//! 
//! This module provides comprehensive metrics collection for AI/ML workloads
//! using HDR histograms for precise tail latency analysis.

pub mod enhanced;

pub use enhanced::{
    MetricsConfig, HistogramConfig, EnhancedMetricsCollector,
    PerformanceHistogram, PerformanceReport, HistogramSummary,
    OperationReport, init_global_metrics, record_operation, record_error,
    get_global_report, print_global_report
};

// Re-export convenience functions for backward compatibility
// The actual implementations are in the enhanced module

/// Convenience function to create metrics config for AI/ML workloads
pub fn aiml_metrics_config() -> MetricsConfig {
    MetricsConfig::training_optimized()
}

/// Convenience function to create metrics config for high-frequency operations  
pub fn high_frequency_metrics_config() -> MetricsConfig {
    MetricsConfig::high_frequency()
}

/// Convenience function to create metrics config for inference workloads
pub fn inference_metrics_config() -> MetricsConfig {
    MetricsConfig::inference_optimized()
}