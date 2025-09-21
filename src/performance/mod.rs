// src/performance/mod.rs
//
// Performance optimization module

pub mod config;

pub use config::{
    PerformanceConfig, 
    ConcurrencyConfiguration, 
    PartSizeConfig,
    env_keys,
};

use anyhow::Result;
use crate::concurrency::AdaptiveScheduler;

#[cfg(feature = "enhanced-http")]
use crate::http::EnhancedHttpClient;

/// Performance-optimized S3 client factory
pub struct PerformanceOptimizedClient {
    config: PerformanceConfig,
    scheduler: Option<AdaptiveScheduler>,
    
    #[cfg(feature = "enhanced-http")]
    http_client: Option<EnhancedHttpClient>,
}

impl PerformanceOptimizedClient {
    /// Create a new performance-optimized client
    pub fn new(config: PerformanceConfig) -> Result<Self> {
        config.validate()?;
        
        let scheduler = if config.enable_optimizations {
            Some(AdaptiveScheduler::new(
                config.concurrency.mode.clone(),
                config.s3_profile.clone(),
            ))
        } else {
            None
        };
        
        #[cfg(feature = "enhanced-http")]
        let http_client = if config.enable_optimizations {
            Some(EnhancedHttpClient::new(config.http_client.clone())?)
        } else {
            None
        };
        
        Ok(Self {
            config,
            scheduler,
            
            #[cfg(feature = "enhanced-http")]
            http_client,
        })
    }
    
    /// Create client optimized for your testing methodology
    pub fn for_sustained_testing() -> Result<Self> {
        Self::new(PerformanceConfig::for_sustained_testing())
    }
    
    /// Create client targeting specific throughput
    pub fn for_target_throughput(target_gbps: f64) -> Result<Self> {
        Self::new(PerformanceConfig::for_target_throughput(target_gbps))
    }
    
    /// Auto-detect optimal configuration
    pub fn auto_detect(endpoint: &str) -> Result<Self> {
        Self::new(PerformanceConfig::auto_detect(endpoint))
    }
    
    /// Create client from environment variables
    pub fn from_environment() -> Result<Self> {
        Self::new(PerformanceConfig::from_environment()?)
    }
    
    /// Get the configuration
    pub fn config(&self) -> &PerformanceConfig {
        &self.config
    }
    
    /// Get the adaptive scheduler (if enabled)
    pub fn scheduler(&self) -> Option<&AdaptiveScheduler> {
        self.scheduler.as_ref()
    }
    
    /// Get mutable scheduler for statistics updates
    pub fn scheduler_mut(&mut self) -> Option<&mut AdaptiveScheduler> {
        self.scheduler.as_mut()
    }
    
    /// Get the HTTP client (if enabled and feature available)
    #[cfg(feature = "enhanced-http")]
    pub fn http_client(&self) -> Option<&EnhancedHttpClient> {
        self.http_client.as_ref()
    }
    
    /// Update configuration dynamically
    pub fn update_config(&mut self, new_config: PerformanceConfig) -> Result<()> {
        new_config.validate()?;
        
        // Update scheduler if mode changed
        if let Some(ref mut scheduler) = self.scheduler {
            if !std::ptr::eq(&self.config.concurrency.mode, &new_config.concurrency.mode) {
                scheduler.update_mode(new_config.concurrency.mode.clone())?;
            }
            
            if !std::ptr::eq(&self.config.s3_profile, &new_config.s3_profile) {
                scheduler.update_profile(new_config.s3_profile.clone())?;
            }
        }
        
        self.config = new_config;
        Ok(())
    }
    
    /// Get optimal part size for given object size
    pub fn optimal_part_size(&self, object_size: u64) -> u64 {
        if let Some(scheduler) = &self.scheduler {
            scheduler.optimal_part_size(object_size)
        } else {
            self.config.part_size.target_part_size
        }
    }
    
    /// Get optimal concurrency for current conditions
    pub fn optimal_concurrency(&self) -> usize {
        if let Some(scheduler) = &self.scheduler {
            scheduler.current_concurrency()
        } else {
            match &self.config.concurrency.mode {
                crate::concurrency::ConcurrencyMode::Explicit(n) => *n,
                crate::concurrency::ConcurrencyMode::Auto => 32,
                crate::concurrency::ConcurrencyMode::TargetThroughput(_) => 32,
            }
        }
    }
    
    /// Check if HTTP/2 should be used for endpoint
    pub fn should_use_http2(&self, endpoint: &str) -> bool {
        #[cfg(feature = "enhanced-http")]
        {
            if let Some(client) = &self.http_client {
                return client.should_use_http2(endpoint);
            }
        }
        
        // Fallback: use HTTP/2 for non-AWS endpoints
        !endpoint.contains("amazonaws.com")
    }
    
    /// Get performance statistics
    pub fn performance_stats(&self) -> PerformanceStats {
        let mut stats = PerformanceStats::default();
        
        if let Some(scheduler) = &self.scheduler {
            let scheduler_stats = scheduler.statistics();
            stats.current_concurrency = scheduler_stats.current_concurrency;
            stats.avg_throughput_gbps = scheduler_stats.average_throughput.as_gbps();
            stats.operations_completed = scheduler_stats.total_operations;
        }
        
        #[cfg(feature = "enhanced-http")]
        {
            if let Some(client) = &self.http_client {
                stats.http2_connections = client.http2_connection_count();
                stats.http1_connections = client.http1_connection_count();
            }
        }
        
        stats
    }
}

/// Performance statistics
#[derive(Debug, Default)]
pub struct PerformanceStats {
    pub current_concurrency: usize,
    pub avg_throughput_gbps: f64,
    pub operations_completed: u64,
    pub http2_connections: usize,
    pub http1_connections: usize,
}

impl std::fmt::Display for PerformanceStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Performance Statistics:")?;
        writeln!(f, "  Current Concurrency: {}", self.current_concurrency)?;
        writeln!(f, "  Average Throughput: {:.2} GB/s", self.avg_throughput_gbps)?;
        writeln!(f, "  Operations Completed: {}", self.operations_completed)?;
        
        if self.http2_connections > 0 || self.http1_connections > 0 {
            writeln!(f, "  HTTP/2 Connections: {}", self.http2_connections)?;
            writeln!(f, "  HTTP/1.1 Connections: {}", self.http1_connections)?;
        }
        
        Ok(())
    }
}