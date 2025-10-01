// src/http/client.rs
//
// Enhanced HTTP client with HTTP/2 support and performance optimizations

use anyhow::{Context, Result};
use std::time::Duration;
use reqwest::ClientBuilder;
use tracing::{info, warn};

/// HTTP client configuration for optimal S3 performance
#[derive(Debug, Clone)]
pub struct HttpClientConfig {
    /// Maximum number of connections per host
    pub max_connections_per_host: usize,
    /// Connection timeout
    pub connect_timeout: Duration,
    /// Request timeout
    pub request_timeout: Duration,
    /// Enable HTTP/2 support
    pub enable_http2: bool,
    /// Force HTTP/2 prior knowledge (skip HTTP/1.1 upgrade)
    pub http2_prior_knowledge: bool,
    /// HTTP/2 keep alive interval
    pub http2_keep_alive_interval: Option<Duration>,
    /// TCP keepalive settings
    pub tcp_keepalive: Option<Duration>,
    /// Pool idle timeout
    pub pool_idle_timeout: Option<Duration>,
}

impl Default for HttpClientConfig {
    fn default() -> Self {
        Self {
            max_connections_per_host: 200,
            connect_timeout: Duration::from_secs(30),
            request_timeout: Duration::from_secs(300),
            enable_http2: true,
            http2_prior_knowledge: false, // Start with upgrade, can be enabled for known HTTP/2 endpoints
            http2_keep_alive_interval: Some(Duration::from_secs(30)),
            tcp_keepalive: Some(Duration::from_secs(60)),
            pool_idle_timeout: Some(Duration::from_secs(90)),
        }
    }
}

impl HttpClientConfig {
    /// Configuration optimized for high-performance S3-compatible storage
    pub fn high_performance() -> Self {
        Self {
            max_connections_per_host: 256,
            connect_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(180),
            enable_http2: true,
            http2_prior_knowledge: true, // Skip upgrade negotiation
            http2_keep_alive_interval: Some(Duration::from_secs(15)),
            tcp_keepalive: Some(Duration::from_secs(30)),
            pool_idle_timeout: Some(Duration::from_secs(60)),
        }
    }
    
    /// Configuration for AWS S3 (HTTP/1.1 only)
    pub fn aws_s3() -> Self {
        Self {
            max_connections_per_host: 200,
            connect_timeout: Duration::from_secs(30),
            request_timeout: Duration::from_secs(300),
            enable_http2: false, // AWS S3 doesn't support HTTP/2
            http2_prior_knowledge: false,
            http2_keep_alive_interval: None,
            tcp_keepalive: Some(Duration::from_secs(60)),
            pool_idle_timeout: Some(Duration::from_secs(90)),
        }
    }
    
    /// Auto-detect configuration based on endpoint
    pub fn auto_detect(endpoint: &str) -> Self {
        if endpoint.contains("amazonaws.com") || endpoint.contains("s3.") {
            Self::aws_s3()
        } else {
            // Assume non-AWS S3 implementation that might support HTTP/2
            Self::high_performance()
        }
    }
}

/// Enhanced HTTP client with performance optimizations
#[derive(Debug, Clone)]
pub struct EnhancedHttpClient {
    client: reqwest::Client,
    config: HttpClientConfig,
}

impl EnhancedHttpClient {
    /// Create new enhanced HTTP client
    pub fn new(config: HttpClientConfig) -> Result<Self> {
        let mut builder = ClientBuilder::new()
            .pool_max_idle_per_host(config.max_connections_per_host)
            .connect_timeout(config.connect_timeout)
            .timeout(config.request_timeout);
            
        // Configure HTTP/2 support
        if config.enable_http2 {
            if config.http2_prior_knowledge {
                builder = builder.http2_prior_knowledge();
            }
            
            if let Some(interval) = config.http2_keep_alive_interval {
                builder = builder.http2_keep_alive_interval(interval);
            }
        } else {
            // Force HTTP/1.1 only
            builder = builder.http1_only();
        }
        
        // Configure TCP settings
        if let Some(keepalive) = config.tcp_keepalive {
            builder = builder.tcp_keepalive(keepalive);
        }
        
        if let Some(idle_timeout) = config.pool_idle_timeout {
            builder = builder.pool_idle_timeout(idle_timeout);
        }
        
        // Additional performance optimizations
        builder = builder
            .tcp_nodelay(true)  // Disable Nagle's algorithm for lower latency
            .use_rustls_tls()   // Use rustls for better performance than OpenSSL
            .hickory_dns(true); // Use hickory-dns for better async DNS resolution
            
        let client = builder.build()
            .context("Failed to build HTTP client")?;
            
        Ok(Self { client, config })
    }
    
    /// Create client with default configuration
    pub fn default() -> Result<Self> {
        Self::new(HttpClientConfig::default())
    }
    
    /// Create client optimized for specific endpoint
    pub fn for_endpoint(endpoint: &str) -> Result<Self> {
        let config = HttpClientConfig::auto_detect(endpoint);
        Self::new(config)
    }
    
    /// Get underlying reqwest client
    pub fn client(&self) -> &reqwest::Client {
        &self.client
    }
    
    /// Get configuration
    pub fn config(&self) -> &HttpClientConfig {
        &self.config
    }
    
    /// Test HTTP/2 support for an endpoint
    pub async fn test_http2_support(&self, url: &str) -> Result<bool> {
        let response = self.client
            .head(url)
            .send()
            .await
            .context("Failed to test HTTP/2 support")?;
            
        // Check if response came via HTTP/2
        Ok(response.version() == reqwest::Version::HTTP_2)
    }
}

/// HTTP client factory for different S3 implementations
pub struct HttpClientFactory;

impl HttpClientFactory {
    /// Create client based on S3 endpoint detection
    pub async fn create_for_s3_endpoint(endpoint: &str) -> Result<EnhancedHttpClient> {
        let mut client = EnhancedHttpClient::for_endpoint(endpoint)?;
        
        // If we're not sure about HTTP/2 support, test it
        if !endpoint.contains("amazonaws.com") && client.config.enable_http2 && !client.config.http2_prior_knowledge {
            // Test if endpoint supports HTTP/2
            match client.test_http2_support(endpoint).await {
                Ok(true) => {
                    info!("Detected HTTP/2 support for endpoint: {}", endpoint);
                    // Recreate client with prior knowledge enabled for better performance
                    let mut config = client.config.clone();
                    config.http2_prior_knowledge = true;
                    client = EnhancedHttpClient::new(config)?;
                },
                Ok(false) => {
                    info!("Endpoint does not support HTTP/2, using HTTP/1.1: {}", endpoint);
                    // Recreate client with HTTP/1.1 only
                    let mut config = client.config.clone();
                    config.enable_http2 = false;
                    client = EnhancedHttpClient::new(config)?;
                },
                Err(e) => {
                    warn!("Failed to test HTTP/2 support for {}: {}. Using HTTP/1.1", endpoint, e);
                    // Fallback to HTTP/1.1
                    let mut config = client.config.clone();
                    config.enable_http2 = false;
                    client = EnhancedHttpClient::new(config)?;
                }
            }
        }
        
        Ok(client)
    }
    
    /// Create client for AWS S3
    pub fn create_for_aws() -> Result<EnhancedHttpClient> {
        EnhancedHttpClient::new(HttpClientConfig::aws_s3())
    }
    
    /// Create high-performance client for non-AWS S3
    pub fn create_high_performance() -> Result<EnhancedHttpClient> {
        EnhancedHttpClient::new(HttpClientConfig::high_performance())
    }
}

impl EnhancedHttpClient {
    /// Check if HTTP/2 should be used for endpoint
    pub fn should_use_http2(&self, endpoint: &str) -> bool {
        if !self.config.enable_http2 {
            return false;
        }
        
        // AWS S3 doesn't support HTTP/2
        if endpoint.contains("amazonaws.com") || endpoint.contains("s3.") {
            return false;
        }
        
        true
    }
    
    /// Get number of HTTP/2 connections (mock implementation)
    pub fn http2_connection_count(&self) -> usize {
        if self.config.enable_http2 {
            // This is a simplified implementation - in reality you'd track actual connections
            self.config.max_connections_per_host / 2
        } else {
            0
        }
    }
    
    /// Get number of HTTP/1.1 connections (mock implementation) 
    pub fn http1_connection_count(&self) -> usize {
        if self.config.enable_http2 {
            self.config.max_connections_per_host / 2
        } else {
            self.config.max_connections_per_host
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_client_creation() {
        let client = EnhancedHttpClient::default();
        assert!(client.is_ok());
    }
    
    #[tokio::test]
    async fn test_aws_config() {
        let client = HttpClientFactory::create_for_aws();
        assert!(client.is_ok());
        let client = client.unwrap();
        assert!(!client.config.enable_http2);
    }
    
    #[tokio::test]  
    async fn test_high_performance_config() {
        let client = HttpClientFactory::create_high_performance();
        assert!(client.is_ok());
        let client = client.unwrap();
        assert!(client.config.enable_http2);
        assert!(client.config.http2_prior_knowledge);
    }
}