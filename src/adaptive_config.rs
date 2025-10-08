// src/adaptive_config.rs
//
// Optional adaptive tuning for s3dlio operations
// Provides smart defaults based on workload characteristics
// CRITICAL: Adaptive behavior is OPTIONAL - explicit user settings always override

use serde::{Deserialize, Serialize};
use std::num::NonZeroUsize;

use crate::constants::{
    DEFAULT_S3_MULTIPART_PART_SIZE,
    MIN_S3_MULTIPART_PART_SIZE,
    DEFAULT_CONCURRENT_UPLOADS,
};

/// Adaptive tuning mode
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AdaptiveMode {
    /// No adaptive tuning - use explicit settings or defaults
    Disabled,
    /// Enable adaptive tuning for performance optimization
    Enabled,
}

impl Default for AdaptiveMode {
    fn default() -> Self {
        // Default is DISABLED - users opt-in to adaptive behavior
        Self::Disabled
    }
}

/// Workload characteristics for adaptive tuning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WorkloadType {
    /// Small files, low latency priority (< 16 MB)
    SmallFile,
    /// Medium files, balanced performance (16-256 MB)
    MediumFile,
    /// Large files, throughput priority (> 256 MB)
    LargeFile,
    /// Batch operations with many small objects
    Batch,
    /// Unknown or mixed workload
    Unknown,
}

impl WorkloadType {
    /// Classify workload based on file size
    pub fn from_file_size(size: usize) -> Self {
        match size {
            0..=16_777_216 => Self::SmallFile,          // < 16 MB
            16_777_217..=268_435_456 => Self::MediumFile, // 16-256 MB
            _ => Self::LargeFile,                        // > 256 MB
        }
    }
}

/// Adaptive configuration for s3dlio operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    /// Adaptive tuning mode (default: Disabled)
    pub mode: AdaptiveMode,
    
    /// Workload type hint for optimization
    pub workload_type: Option<WorkloadType>,
    
    /// Minimum part size for uploads (default: 5 MB - AWS minimum)
    pub min_part_size: usize,
    
    /// Maximum part size for uploads (default: 64 MB - good for most workloads)
    pub max_part_size: usize,
    
    /// Minimum concurrency (default: 8)
    pub min_concurrency: usize,
    
    /// Maximum concurrency (default: 128)
    pub max_concurrency: usize,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            mode: AdaptiveMode::Disabled,
            workload_type: None,
            min_part_size: MIN_S3_MULTIPART_PART_SIZE,     // 5 MB
            max_part_size: 64 * 1024 * 1024,                // 64 MB
            min_concurrency: 8,
            max_concurrency: 128,
        }
    }
}

impl AdaptiveConfig {
    /// Enable adaptive tuning
    pub fn enabled() -> Self {
        Self {
            mode: AdaptiveMode::Enabled,
            ..Default::default()
        }
    }
    
    /// Set workload type hint
    pub fn with_workload_type(mut self, workload: WorkloadType) -> Self {
        self.workload_type = Some(workload);
        self
    }
    
    /// Set part size bounds
    pub fn with_part_size_bounds(mut self, min: usize, max: usize) -> Self {
        self.min_part_size = min.max(MIN_S3_MULTIPART_PART_SIZE); // Enforce AWS minimum
        self.max_part_size = max;
        self
    }
    
    /// Set concurrency bounds
    pub fn with_concurrency_bounds(mut self, min: usize, max: usize) -> Self {
        self.min_concurrency = min.max(1); // At least 1
        self.max_concurrency = max;
        self
    }
}

/// Adaptive parameter computation
pub struct AdaptiveParams {
    config: AdaptiveConfig,
}

impl AdaptiveParams {
    /// Create new adaptive parameter computer
    pub fn new(config: AdaptiveConfig) -> Self {
        Self { config }
    }
    
    /// Compute optimal part size for upload
    /// 
    /// CRITICAL: If explicit_part_size is provided, it is ALWAYS used
    /// Adaptive tuning only applies when explicit_part_size is None
    /// 
    /// # Arguments
    /// * `file_size` - Total file size in bytes (if known)
    /// * `explicit_part_size` - Explicitly requested part size (overrides adaptive)
    /// 
    /// # Returns
    /// Part size in bytes
    pub fn compute_part_size(
        &self,
        file_size: Option<usize>,
        explicit_part_size: Option<usize>,
    ) -> usize {
        // RULE 1: Explicit settings ALWAYS override adaptive behavior
        if let Some(size) = explicit_part_size {
            return size.max(MIN_S3_MULTIPART_PART_SIZE); // Enforce AWS minimum
        }
        
        // RULE 2: If adaptive is disabled, use default
        if self.config.mode == AdaptiveMode::Disabled {
            return DEFAULT_S3_MULTIPART_PART_SIZE; // 16 MB
        }
        
        // RULE 3: Adaptive tuning based on file size
        match file_size {
            None => DEFAULT_S3_MULTIPART_PART_SIZE, // Unknown size: use default
            Some(size) => {
                let workload = self.config.workload_type
                    .unwrap_or_else(|| WorkloadType::from_file_size(size));
                
                match workload {
                    // Small files: Use smaller parts (8 MB) to avoid unnecessary chunking
                    WorkloadType::SmallFile => {
                        (8 * 1024 * 1024)
                            .max(self.config.min_part_size)
                            .min(self.config.max_part_size)
                    }
                    // Medium files: Use default parts (16 MB) - proven performance
                    WorkloadType::MediumFile => {
                        DEFAULT_S3_MULTIPART_PART_SIZE
                            .max(self.config.min_part_size)
                            .min(self.config.max_part_size)
                    }
                    // Large files: Use larger parts (32 MB) for better throughput
                    WorkloadType::LargeFile => {
                        (32 * 1024 * 1024)
                            .max(self.config.min_part_size)
                            .min(self.config.max_part_size)
                    }
                    // Batch: Use medium parts (16 MB) for balanced performance
                    WorkloadType::Batch => {
                        DEFAULT_S3_MULTIPART_PART_SIZE
                            .max(self.config.min_part_size)
                            .min(self.config.max_part_size)
                    }
                    // Unknown: Use default
                    WorkloadType::Unknown => {
                        DEFAULT_S3_MULTIPART_PART_SIZE
                            .max(self.config.min_part_size)
                            .min(self.config.max_part_size)
                    }
                }
            }
        }
    }
    
    /// Compute optimal concurrency for operations
    /// 
    /// CRITICAL: If explicit_concurrency is provided, it is ALWAYS used
    /// Adaptive tuning only applies when explicit_concurrency is None
    /// 
    /// # Arguments
    /// * `workload_type` - Type of workload (if known)
    /// * `explicit_concurrency` - Explicitly requested concurrency (overrides adaptive)
    /// 
    /// # Returns
    /// Number of concurrent operations
    pub fn compute_concurrency(
        &self,
        workload_type: Option<WorkloadType>,
        explicit_concurrency: Option<usize>,
    ) -> usize {
        // RULE 1: Explicit settings ALWAYS override adaptive behavior
        if let Some(concurrency) = explicit_concurrency {
            return concurrency.max(1); // At least 1
        }
        
        // RULE 2: If adaptive is disabled, use default
        if self.config.mode == AdaptiveMode::Disabled {
            return DEFAULT_CONCURRENT_UPLOADS; // 32
        }
        
        // RULE 3: Adaptive tuning based on workload and system characteristics
        let num_cpus = std::thread::available_parallelism()
            .map(NonZeroUsize::get)
            .unwrap_or(8); // Default to 8 if detection fails
        
        let workload = workload_type.or(self.config.workload_type).unwrap_or(WorkloadType::Unknown);
        
        match workload {
            // Small files: Higher concurrency for better throughput
            WorkloadType::SmallFile => {
                (num_cpus * 8) // 8x CPU count
                    .max(self.config.min_concurrency)
                    .min(self.config.max_concurrency)
            }
            // Medium files: Balanced concurrency
            WorkloadType::MediumFile => {
                (num_cpus * 4) // 4x CPU count
                    .max(self.config.min_concurrency)
                    .min(self.config.max_concurrency)
            }
            // Large files: Lower concurrency, higher per-transfer throughput
            WorkloadType::LargeFile => {
                (num_cpus * 2) // 2x CPU count
                    .max(self.config.min_concurrency)
                    .min(self.config.max_concurrency)
            }
            // Batch: High concurrency for parallelism
            WorkloadType::Batch => {
                (num_cpus * 8) // 8x CPU count
                    .max(self.config.min_concurrency)
                    .min(self.config.max_concurrency)
            }
            // Unknown: Use default
            WorkloadType::Unknown => {
                DEFAULT_CONCURRENT_UPLOADS
                    .max(self.config.min_concurrency)
                    .min(self.config.max_concurrency)
            }
        }
    }
    
    /// Compute optimal buffer size for operations
    /// 
    /// # Arguments
    /// * `operation_type` - Type of operation (upload/download)
    /// * `explicit_buffer_size` - Explicitly requested buffer size (overrides adaptive)
    /// 
    /// # Returns
    /// Buffer size in bytes
    pub fn compute_buffer_size(
        &self,
        operation_type: &str,
        explicit_buffer_size: Option<usize>,
    ) -> usize {
        // RULE 1: Explicit settings ALWAYS override adaptive behavior
        if let Some(size) = explicit_buffer_size {
            return size;
        }
        
        // RULE 2: If adaptive is disabled, use default
        if self.config.mode == AdaptiveMode::Disabled {
            return 1024 * 1024; // 1 MB default
        }
        
        // RULE 3: Adaptive tuning based on operation type
        match operation_type {
            "upload" => 2 * 1024 * 1024,   // 2 MB for uploads
            "download" => 4 * 1024 * 1024, // 4 MB for downloads (read-ahead)
            _ => 1024 * 1024,              // 1 MB default
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_adaptive_mode_default_disabled() {
        let config = AdaptiveConfig::default();
        assert_eq!(config.mode, AdaptiveMode::Disabled);
    }
    
    #[test]
    fn test_explicit_part_size_overrides_adaptive() {
        let config = AdaptiveConfig::enabled();
        let params = AdaptiveParams::new(config);
        
        // Explicit size should ALWAYS be used
        let explicit = Some(10 * 1024 * 1024); // 10 MB
        let result = params.compute_part_size(Some(100 * 1024 * 1024), explicit);
        assert_eq!(result, 10 * 1024 * 1024);
    }
    
    #[test]
    fn test_adaptive_disabled_uses_default() {
        let config = AdaptiveConfig::default(); // Disabled by default
        let params = AdaptiveParams::new(config);
        
        // Should use default even with file size hint
        let result = params.compute_part_size(Some(500 * 1024 * 1024), None);
        assert_eq!(result, DEFAULT_S3_MULTIPART_PART_SIZE);
    }
    
    #[test]
    fn test_adaptive_part_size_for_large_files() {
        let config = AdaptiveConfig::enabled();
        let params = AdaptiveParams::new(config);
        
        // Large file should get 32 MB parts
        let result = params.compute_part_size(Some(1024 * 1024 * 1024), None);
        assert_eq!(result, 32 * 1024 * 1024);
    }
    
    #[test]
    fn test_adaptive_part_size_for_small_files() {
        let config = AdaptiveConfig::enabled();
        let params = AdaptiveParams::new(config);
        
        // Small file should get 8 MB parts
        let result = params.compute_part_size(Some(10 * 1024 * 1024), None);
        assert_eq!(result, 8 * 1024 * 1024);
    }
    
    #[test]
    fn test_explicit_concurrency_overrides_adaptive() {
        let config = AdaptiveConfig::enabled();
        let params = AdaptiveParams::new(config);
        
        // Explicit concurrency should ALWAYS be used
        let result = params.compute_concurrency(Some(WorkloadType::SmallFile), Some(42));
        assert_eq!(result, 42);
    }
    
    #[test]
    fn test_adaptive_disabled_concurrency_uses_default() {
        let config = AdaptiveConfig::default(); // Disabled
        let params = AdaptiveParams::new(config);
        
        // Should use default even with workload hint
        let result = params.compute_concurrency(Some(WorkloadType::LargeFile), None);
        assert_eq!(result, DEFAULT_CONCURRENT_UPLOADS);
    }
    
    #[test]
    fn test_workload_type_from_file_size() {
        assert_eq!(WorkloadType::from_file_size(1024 * 1024), WorkloadType::SmallFile);
        assert_eq!(WorkloadType::from_file_size(100 * 1024 * 1024), WorkloadType::MediumFile);
        assert_eq!(WorkloadType::from_file_size(500 * 1024 * 1024), WorkloadType::LargeFile);
    }
    
    #[test]
    fn test_min_part_size_enforced() {
        let config = AdaptiveConfig::enabled();
        let params = AdaptiveParams::new(config);
        
        // Explicit part size below AWS minimum should be clamped
        let result = params.compute_part_size(None, Some(1024 * 1024)); // 1 MB
        assert_eq!(result, MIN_S3_MULTIPART_PART_SIZE); // 5 MB
    }
    
    #[test]
    fn test_adaptive_respects_bounds() {
        let config = AdaptiveConfig::enabled()
            .with_part_size_bounds(10 * 1024 * 1024, 20 * 1024 * 1024); // 10-20 MB
        let params = AdaptiveParams::new(config);
        
        // Large file would want 32 MB, but should be clamped to max
        let result = params.compute_part_size(Some(1024 * 1024 * 1024), None);
        assert_eq!(result, 20 * 1024 * 1024);
    }
}
