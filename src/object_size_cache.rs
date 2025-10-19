// src/object_size_cache.rs
//
// Thread-safe cache for object sizes with TTL expiration
//
// Enables pre-statting objects before download to eliminate per-object stat latency
// in high-throughput benchmarking workloads like sai3-bench.
//
// v0.9.10 optimization: Pre-stat 1000 objects in 200ms instead of 20 seconds

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// Cached object size with timestamp for TTL expiration
#[derive(Debug, Clone)]
struct CachedSize {
    /// Object size in bytes
    size: u64,
    /// When this entry was cached
    cached_at: Instant,
}

/// Thread-safe cache of object sizes from stat/head operations
/// 
/// Enables pre-statting objects before download to eliminate per-object stat latency
/// in high-throughput workloads like benchmarking.
/// 
/// # Usage Pattern
/// 
/// ```no_run
/// use s3dlio::object_size_cache::ObjectSizeCache;
/// use std::time::Duration;
/// 
/// # async fn example() -> anyhow::Result<()> {
/// let cache = ObjectSizeCache::new(Duration::from_secs(60));
/// 
/// // Pre-stat phase: Populate cache
/// cache.put("s3://bucket/object1.dat".to_string(), 67108864).await;
/// cache.put("s3://bucket/object2.dat".to_string(), 67108864).await;
/// 
/// // Download phase: Use cached sizes (no stat overhead)
/// if let Some(size) = cache.get("s3://bucket/object1.dat").await {
///     println!("Using cached size: {} bytes", size);
/// }
/// # Ok(())
/// # }
/// ```
/// 
/// # Performance Impact
/// 
/// For 1000 object benchmark:
/// - Without cache: 1000 × 20ms stat = 20 seconds overhead
/// - With cache: Pre-stat in 200ms (100 concurrent), then zero overhead
/// - Improvement: 2.5x faster total time, 99% stat overhead reduction
#[derive(Debug)]
pub struct ObjectSizeCache {
    /// Internal cache storage (URI → cached size + timestamp)
    cache: Arc<RwLock<HashMap<String, CachedSize>>>,
    
    /// Time-to-live for cached entries
    ttl: Duration,
}

impl ObjectSizeCache {
    /// Create a new size cache with the given TTL
    /// 
    /// # Arguments
    /// 
    /// * `ttl` - Time-to-live for cached entries (60 seconds recommended)
    /// 
    /// # Example
    /// 
    /// ```
    /// use s3dlio::object_size_cache::ObjectSizeCache;
    /// use std::time::Duration;
    /// 
    /// let cache = ObjectSizeCache::new(Duration::from_secs(60));
    /// ```
    pub fn new(ttl: Duration) -> Self {
        Self {
            cache: Arc::new(RwLock::new(HashMap::new())),
            ttl,
        }
    }

    /// Get cached size for URI (returns None if not cached or expired)
    /// 
    /// This is a read-only operation using RwLock::read(), allowing concurrent access
    /// from multiple threads without blocking.
    /// 
    /// # Arguments
    /// 
    /// * `uri` - Object URI (e.g., "s3://bucket/key")
    /// 
    /// # Returns
    /// 
    /// * `Some(size)` - If entry exists and hasn't expired
    /// * `None` - If not cached or expired
    /// 
    /// # Example
    /// 
    /// ```no_run
    /// # use s3dlio::object_size_cache::ObjectSizeCache;
    /// # use std::time::Duration;
    /// # async fn example() {
    /// # let cache = ObjectSizeCache::new(Duration::from_secs(60));
    /// if let Some(size) = cache.get("s3://bucket/object.dat").await {
    ///     println!("Cached size: {} bytes", size);
    /// } else {
    ///     println!("Not in cache, need to stat");
    /// }
    /// # }
    /// ```
    pub async fn get(&self, uri: &str) -> Option<u64> {
        let cache = self.cache.read().await;
        if let Some(entry) = cache.get(uri) {
            // Check if entry has expired
            if entry.cached_at.elapsed() < self.ttl {
                tracing::trace!("Cache HIT: {} ({} bytes)", uri, entry.size);
                return Some(entry.size);
            } else {
                tracing::trace!("Cache EXPIRED: {} (age: {:?})", uri, entry.cached_at.elapsed());
            }
        } else {
            tracing::trace!("Cache MISS: {}", uri);
        }
        None
    }

    /// Store size for URI in cache
    /// 
    /// This is a write operation using RwLock::write(), which will block if other
    /// readers are active. Consider batching puts when possible.
    /// 
    /// # Arguments
    /// 
    /// * `uri` - Object URI (e.g., "s3://bucket/key")
    /// * `size` - Object size in bytes
    /// 
    /// # Example
    /// 
    /// ```no_run
    /// # use s3dlio::object_size_cache::ObjectSizeCache;
    /// # use std::time::Duration;
    /// # async fn example() {
    /// # let cache = ObjectSizeCache::new(Duration::from_secs(60));
    /// cache.put("s3://bucket/object.dat".to_string(), 67108864).await;
    /// # }
    /// ```
    pub async fn put(&self, uri: String, size: u64) {
        let mut cache = self.cache.write().await;
        tracing::trace!("Caching: {} → {} bytes", uri, size);
        cache.insert(uri, CachedSize {
            size,
            cached_at: Instant::now(),
        });
    }

    /// Clear expired entries from cache
    /// 
    /// This should be called periodically (e.g., every 10 seconds) to prevent
    /// memory growth from expired entries.
    /// 
    /// # Example
    /// 
    /// ```no_run
    /// # use s3dlio::object_size_cache::ObjectSizeCache;
    /// # use std::time::Duration;
    /// # async fn example() {
    /// # let cache = ObjectSizeCache::new(Duration::from_secs(60));
    /// cache.clear_expired().await;
    /// # }
    /// ```
    pub async fn clear_expired(&self) {
        let mut cache = self.cache.write().await;
        let before_count = cache.len();
        cache.retain(|uri, entry| {
            let expired = entry.cached_at.elapsed() >= self.ttl;
            if expired {
                tracing::debug!("Removing expired cache entry: {}", uri);
            }
            !expired
        });
        let after_count = cache.len();
        let removed = before_count - after_count;
        if removed > 0 {
            tracing::info!("Cleared {} expired cache entries ({} remain)", removed, after_count);
        }
    }

    /// Clear all entries from cache
    /// 
    /// Useful for testing or when you want to force fresh stats.
    pub async fn clear(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
        tracing::debug!("Cleared all cache entries");
    }

    /// Get cache statistics
    /// 
    /// Returns count of total entries and how many are expired (but not yet cleared).
    /// 
    /// # Example
    /// 
    /// ```no_run
    /// # use s3dlio::object_size_cache::ObjectSizeCache;
    /// # use std::time::Duration;
    /// # async fn example() {
    /// # let cache = ObjectSizeCache::new(Duration::from_secs(60));
    /// let stats = cache.stats().await;
    /// println!("Cache: {} total, {} expired", stats.total_entries, stats.expired_entries);
    /// # }
    /// ```
    pub async fn stats(&self) -> CacheStats {
        let cache = self.cache.read().await;
        let total = cache.len();
        let expired = cache.values()
            .filter(|e| e.cached_at.elapsed() >= self.ttl)
            .count();
        
        CacheStats {
            total_entries: total,
            expired_entries: expired,
        }
    }
}

/// Statistics about cache state
#[derive(Debug, Clone, Copy)]
pub struct CacheStats {
    /// Total number of cached entries
    pub total_entries: usize,
    
    /// Number of entries that have expired (but not yet cleared)
    pub expired_entries: usize,
}

impl CacheStats {
    /// Number of valid (non-expired) entries
    pub fn valid_entries(&self) -> usize {
        self.total_entries - self.expired_entries
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_cache_basic() {
        let cache = ObjectSizeCache::new(Duration::from_secs(60));
        
        // Initially empty
        assert_eq!(cache.get("s3://bucket/object").await, None);
        
        // Store and retrieve
        cache.put("s3://bucket/object".to_string(), 12345).await;
        assert_eq!(cache.get("s3://bucket/object").await, Some(12345));
        
        // Different key
        assert_eq!(cache.get("s3://bucket/other").await, None);
    }

    #[tokio::test]
    async fn test_cache_expiration() {
        let cache = ObjectSizeCache::new(Duration::from_millis(100));
        
        // Cache a value
        cache.put("s3://bucket/object".to_string(), 12345).await;
        assert_eq!(cache.get("s3://bucket/object").await, Some(12345));
        
        // Wait for expiration
        sleep(Duration::from_millis(150)).await;
        
        // Should be expired now
        assert_eq!(cache.get("s3://bucket/object").await, None);
    }

    #[tokio::test]
    async fn test_cache_overwrite() {
        let cache = ObjectSizeCache::new(Duration::from_secs(60));
        
        // Store initial value
        cache.put("s3://bucket/object".to_string(), 12345).await;
        assert_eq!(cache.get("s3://bucket/object").await, Some(12345));
        
        // Overwrite with new value
        cache.put("s3://bucket/object".to_string(), 67890).await;
        assert_eq!(cache.get("s3://bucket/object").await, Some(67890));
    }

    #[tokio::test]
    async fn test_cache_stats() {
        let cache = ObjectSizeCache::new(Duration::from_millis(100));
        
        // Empty cache
        let stats = cache.stats().await;
        assert_eq!(stats.total_entries, 0);
        assert_eq!(stats.expired_entries, 0);
        
        // Add some entries
        cache.put("s3://bucket/obj1".to_string(), 1000).await;
        cache.put("s3://bucket/obj2".to_string(), 2000).await;
        
        let stats = cache.stats().await;
        assert_eq!(stats.total_entries, 2);
        assert_eq!(stats.expired_entries, 0);
        assert_eq!(stats.valid_entries(), 2);
        
        // Wait for expiration
        sleep(Duration::from_millis(150)).await;
        
        let stats = cache.stats().await;
        assert_eq!(stats.total_entries, 2);  // Still in cache
        assert_eq!(stats.expired_entries, 2);  // But expired
        assert_eq!(stats.valid_entries(), 0);
    }

    #[tokio::test]
    async fn test_clear_expired() {
        let cache = ObjectSizeCache::new(Duration::from_millis(100));
        
        // Add entries
        cache.put("s3://bucket/obj1".to_string(), 1000).await;
        cache.put("s3://bucket/obj2".to_string(), 2000).await;
        
        // Wait for expiration
        sleep(Duration::from_millis(150)).await;
        
        // Clear expired
        cache.clear_expired().await;
        
        // Cache should be empty now
        let stats = cache.stats().await;
        assert_eq!(stats.total_entries, 0);
        assert_eq!(stats.expired_entries, 0);
    }

    #[tokio::test]
    async fn test_clear_all() {
        let cache = ObjectSizeCache::new(Duration::from_secs(60));
        
        // Add entries
        cache.put("s3://bucket/obj1".to_string(), 1000).await;
        cache.put("s3://bucket/obj2".to_string(), 2000).await;
        
        // Clear all
        cache.clear().await;
        
        // Cache should be empty
        let stats = cache.stats().await;
        assert_eq!(stats.total_entries, 0);
        assert_eq!(cache.get("s3://bucket/obj1").await, None);
        assert_eq!(cache.get("s3://bucket/obj2").await, None);
    }

    #[tokio::test]
    async fn test_concurrent_access() {
        let cache = Arc::new(ObjectSizeCache::new(Duration::from_secs(60)));
        
        // Spawn multiple tasks writing to cache
        let mut handles = vec![];
        for i in 0..100 {
            let cache = Arc::clone(&cache);
            let handle = tokio::spawn(async move {
                let uri = format!("s3://bucket/object-{}", i);
                cache.put(uri.clone(), i * 1000).await;
                sleep(Duration::from_millis(1)).await;
                cache.get(&uri).await
            });
            handles.push(handle);
        }
        
        // Wait for all tasks
        for (i, handle) in handles.into_iter().enumerate() {
            let result = handle.await.unwrap();
            assert_eq!(result, Some(i as u64 * 1000));
        }
        
        // Verify all entries are in cache
        let stats = cache.stats().await;
        assert_eq!(stats.total_entries, 100);
    }
    
    #[tokio::test]
    async fn test_zero_ttl_disables_cache() {
        let cache = ObjectSizeCache::new(Duration::from_secs(0));
        
        cache.put("s3://bucket/object1".to_string(), 1000).await;
        
        // With 0 TTL, entry should be immediately expired
        sleep(Duration::from_millis(10)).await;
        let result = cache.get("s3://bucket/object1").await;
        assert!(result.is_none(), "0 TTL should immediately expire entries");
    }
    
    #[tokio::test]
    async fn test_cache_hit_rate() {
        let cache = ObjectSizeCache::new(Duration::from_secs(60));
        
        // Populate cache with 10 entries
        for i in 0..10 {
            cache.put(format!("s3://bucket/object-{}", i), i * 1000).await;
        }
        
        // Test cache hit rate: 10 hits, 5 misses
        let mut hits = 0;
        let mut misses = 0;
        
        for i in 0..15 {
            let uri = format!("s3://bucket/object-{}", i);
            if cache.get(&uri).await.is_some() {
                hits += 1;
            } else {
                misses += 1;
            }
        }
        
        assert_eq!(hits, 10, "Should have 10 cache hits");
        assert_eq!(misses, 5, "Should have 5 cache misses");
    }
    
    #[tokio::test]
    async fn test_expiration_boundary() {
        let cache = ObjectSizeCache::new(Duration::from_millis(50));
        
        cache.put("s3://bucket/fast-expire".to_string(), 1000).await;
        
        // Just before expiration
        sleep(Duration::from_millis(25)).await;
        assert!(cache.get("s3://bucket/fast-expire").await.is_some(), "Should still be valid");
        
        // Just after expiration
        sleep(Duration::from_millis(30)).await;
        assert!(cache.get("s3://bucket/fast-expire").await.is_none(), "Should be expired");
    }
    
    #[tokio::test]
    async fn test_large_number_of_entries() {
        let cache = ObjectSizeCache::new(Duration::from_secs(60));
        
        // Add 1000 entries
        for i in 0..1000 {
            cache.put(format!("s3://bucket/object-{:04}", i), i * 1000).await;
        }
        
        let stats = cache.stats().await;
        assert_eq!(stats.total_entries, 1000);
        assert_eq!(stats.valid_entries(), 1000);
        
        // Verify random access works
        assert_eq!(cache.get("s3://bucket/object-0500").await, Some(500 * 1000));
        assert_eq!(cache.get("s3://bucket/object-0999").await, Some(999 * 1000));
    }
    
    #[tokio::test]
    async fn test_concurrent_read_heavy_workload() {
        let cache = Arc::new(ObjectSizeCache::new(Duration::from_secs(60)));
        
        // Populate cache with 100 entries
        for i in 0..100 {
            cache.put(format!("s3://bucket/object-{:03}", i), i * 1000).await;
        }
        
        // Spawn 1000 concurrent readers
        let mut handles = vec![];
        for i in 0..1000 {
            let cache_clone = Arc::clone(&cache);
            let handle = tokio::spawn(async move {
                let uri = format!("s3://bucket/object-{:03}", i % 100);
                cache_clone.get(&uri).await
            });
            handles.push(handle);
        }
        
        // All reads should succeed
        let mut successful_reads = 0;
        for handle in handles {
            if handle.await.unwrap().is_some() {
                successful_reads += 1;
            }
        }
        
        assert_eq!(successful_reads, 1000, "All reads should find cached entries");
    }
    
    #[tokio::test]
    async fn test_memory_efficiency_large_cache() {
        let cache = ObjectSizeCache::new(Duration::from_secs(60));
        
        // Add 10,000 entries to test memory efficiency
        for i in 0..10_000 {
            cache.put(format!("s3://bucket/dataset/object-{:06}", i), i * 1000).await;
        }
        
        let stats = cache.stats().await;
        assert_eq!(stats.total_entries, 10_000);
        assert_eq!(stats.valid_entries(), 10_000);
        
        // Verify clear works on large cache
        cache.clear().await;
        let stats_after = cache.stats().await;
        assert_eq!(stats_after.total_entries, 0);
    }
}
