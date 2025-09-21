// src/performance/config.rs
//
// Performance configuration management

use anyhow::Result;
use crate::concurrency::{ConcurrencyMode, S3PerformanceProfile, Throughput};

#[cfg(feature = "enhanced-http")]
use crate::http::HttpClientConfig;

/// Comprehensive performance configuration
#[derive(Debug, Clone)]
pub struct PerformanceConfig {
    /// Concurrency management configuration
    pub concurrency: ConcurrencyConfiguration,
    
    /// HTTP client configuration
    #[cfg(feature = "enhanced-http")]
    pub http_client: HttpClientConfig,
    
    /// S3 performance profile
    pub s3_profile: S3PerformanceProfile,
    
    /// Part size configuration
    pub part_size: PartSizeConfig,
    
    /// Enable performance optimizations
    pub enable_optimizations: bool,
}

#[derive(Debug, Clone)]
pub struct ConcurrencyConfiguration {
    pub mode: ConcurrencyMode,
    /// Whether to enable adaptive concurrency adjustment
    pub adaptive: bool,
    /// Minimum concurrency level
    pub min_concurrency: usize,
    /// Maximum concurrency level  
    pub max_concurrency: usize,
}

#[derive(Debug, Clone)]
pub struct PartSizeConfig {
    /// Target part size for uploads
    pub target_part_size: u64,
    /// Minimum part size
    pub min_part_size: u64,
    /// Maximum part size
    pub max_part_size: u64,
    /// Enable automatic part size optimization
    pub auto_optimize: bool,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            concurrency: ConcurrencyConfiguration::default(),
            
            #[cfg(feature = "enhanced-http")]
            http_client: HttpClientConfig::default(),
            
            s3_profile: S3PerformanceProfile::default(),
            part_size: PartSizeConfig::default(),
            enable_optimizations: true,
        }
    }
}

impl Default for ConcurrencyConfiguration {
    fn default() -> Self {
        Self {
            mode: ConcurrencyMode::Auto,
            adaptive: true,
            min_concurrency: 8,
            max_concurrency: 256,
        }
    }
}

impl Default for PartSizeConfig {
    fn default() -> Self {
        Self {
            target_part_size: 10 * 1024 * 1024, // 10 MiB - your proven optimum
            min_part_size: 5 * 1024 * 1024,     // S3 minimum
            max_part_size: 5 * 1024 * 1024 * 1024, // 5 GiB maximum
            auto_optimize: true,
        }
    }
}

impl PerformanceConfig {
    /// Create configuration optimized for your testing methodology
    pub fn for_sustained_testing() -> Self {
        Self {
            concurrency: ConcurrencyConfiguration {
                mode: ConcurrencyMode::Explicit(48), // Your proven sweet spot
                adaptive: true,
                min_concurrency: 24,  // Your minimum tested range
                max_concurrency: 64,  // Reasonable maximum
            },
            
            #[cfg(feature = "enhanced-http")]
            http_client: HttpClientConfig::high_performance(),
            
            s3_profile: S3PerformanceProfile::high_performance(),
            part_size: PartSizeConfig {
                target_part_size: 10 * 1024 * 1024, // 10 MiB objects
                min_part_size: 5 * 1024 * 1024,
                max_part_size: 100 * 1024 * 1024,   // 100 MiB max for testing
                auto_optimize: true,
            },
            enable_optimizations: true,
        }
    }
    
    /// Create configuration targeting specific throughput
    pub fn for_target_throughput(target_gbps: f64) -> Self {
        let target = Throughput::new_bytes_per_sec((target_gbps * 1_000_000_000.0) as u64);
        
        Self {
            concurrency: ConcurrencyConfiguration {
                mode: ConcurrencyMode::TargetThroughput(target),
                adaptive: true,
                min_concurrency: 8,
                max_concurrency: 256,
            },
            
            #[cfg(feature = "enhanced-http")]
            http_client: HttpClientConfig::high_performance(),
            
            s3_profile: S3PerformanceProfile::high_performance(),
            part_size: PartSizeConfig::default(),
            enable_optimizations: true,
        }
    }
    
    /// Auto-detect configuration based on endpoint
    pub fn auto_detect(endpoint: &str) -> Self {
        let is_aws = endpoint.contains("amazonaws.com") || endpoint.contains("s3.");
        
        let s3_profile = if is_aws {
            S3PerformanceProfile::aws_s3()
        } else {
            S3PerformanceProfile::high_performance()
        };
        
        Self {
            concurrency: ConcurrencyConfiguration {
                mode: ConcurrencyMode::Auto,
                adaptive: true,
                min_concurrency: if is_aws { 16 } else { 8 },
                max_concurrency: if is_aws { 128 } else { 256 },
            },
            
            #[cfg(feature = "enhanced-http")]
            http_client: HttpClientConfig::auto_detect(endpoint),
            
            s3_profile,
            part_size: PartSizeConfig::default(),
            enable_optimizations: true,
        }
    }
    
    /// Load configuration from environment variables
    pub fn from_environment() -> Result<Self> {
        let mut config = Self::default();
        
        // Concurrency settings
        if let Ok(target_gbps) = std::env::var("S3DLIO_TARGET_GBPS") {
            if let Ok(gbps) = target_gbps.parse::<f64>() {
                let target = Throughput::new_bytes_per_sec((gbps * 1_000_000_000.0) as u64);
                config.concurrency.mode = ConcurrencyMode::TargetThroughput(target);
            }
        } else if let Ok(concurrency) = std::env::var("S3DLIO_CONCURRENCY") {
            if let Ok(n) = concurrency.parse::<usize>() {
                config.concurrency.mode = ConcurrencyMode::Explicit(n);
            }
        }
        
        // Part size settings
        if let Ok(part_size) = std::env::var("S3DLIO_PART_SIZE_MB") {
            if let Ok(mb) = part_size.parse::<u64>() {
                config.part_size.target_part_size = mb * 1024 * 1024;
            }
        }
        
        // Enable/disable optimizations
        if let Ok(opt) = std::env::var("S3DLIO_ENABLE_OPTIMIZATIONS") {
            config.enable_optimizations = matches!(opt.to_lowercase().as_str(), 
                "true" | "1" | "yes" | "on" | "enable");
        }
        
        // HTTP/2 settings
        #[cfg(feature = "enhanced-http")]
        {
            if let Ok(http2) = std::env::var("S3DLIO_ENABLE_HTTP2") {
                config.http_client.enable_http2 = matches!(http2.to_lowercase().as_str(),
                    "true" | "1" | "yes" | "on" | "enable");
            }
        }
        
        Ok(config)
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate concurrency bounds
        if self.concurrency.min_concurrency > self.concurrency.max_concurrency {
            anyhow::bail!("Minimum concurrency cannot be greater than maximum");
        }
        
        // Validate part sizes
        if self.part_size.min_part_size > self.part_size.max_part_size {
            anyhow::bail!("Minimum part size cannot be greater than maximum");
        }
        
        if self.part_size.target_part_size < self.part_size.min_part_size {
            anyhow::bail!("Target part size cannot be less than minimum");
        }
        
        Ok(())
    }
}

/// Environment variable keys for configuration
pub mod env_keys {
    pub const TARGET_GBPS: &str = "S3DLIO_TARGET_GBPS";
    pub const CONCURRENCY: &str = "S3DLIO_CONCURRENCY";
    pub const PART_SIZE_MB: &str = "S3DLIO_PART_SIZE_MB";
    pub const ENABLE_OPTIMIZATIONS: &str = "S3DLIO_ENABLE_OPTIMIZATIONS";
    pub const ENABLE_HTTP2: &str = "S3DLIO_ENABLE_HTTP2";
    pub const PERFORMANCE_PROFILE: &str = "S3DLIO_PERFORMANCE_PROFILE";
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = PerformanceConfig::default();
        assert!(config.enable_optimizations);
        assert_eq!(config.part_size.target_part_size, 10 * 1024 * 1024);
    }
    
    #[test]
    fn test_sustained_testing_config() {
        let config = PerformanceConfig::for_sustained_testing();
        if let ConcurrencyMode::Explicit(n) = config.concurrency.mode {
            assert_eq!(n, 48);
        } else {
            panic!("Expected explicit concurrency mode");
        }
    }
    
    #[test]
    fn test_target_throughput_config() {
        let config = PerformanceConfig::for_target_throughput(2.5);
        if let ConcurrencyMode::TargetThroughput(throughput) = config.concurrency.mode {
            assert_eq!(throughput.as_gbps(), 2.5);
        } else {
            panic!("Expected target throughput mode");
        }
    }
    
    #[test]
    fn test_validation() {
        let config = PerformanceConfig::default();
        assert!(config.validate().is_ok());
        
        let mut invalid_config = config.clone();
        invalid_config.concurrency.min_concurrency = 100;
        invalid_config.concurrency.max_concurrency = 50;
        assert!(invalid_config.validate().is_err());
    }
}