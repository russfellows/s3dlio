// src/object_size_cache.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use std::collections::{HashMap, VecDeque};
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
    /// Insertion sequence number: identifies this entry's record in
    /// [`Inner::order`] (a newer `put` of the same URI gets a newer number).
    seq: u64,
}

/// The map plus the order in which entries were (last) written.
///
/// `order` holds one `(uri, seq)` record per `put`, oldest first. A record is
/// *live* when the map still holds that URI with the same `seq`; any other
/// record (the entry was overwritten, expired or evicted) is stale and is
/// skipped when it reaches the front. Because every entry has the same TTL,
/// oldest-written is also first-to-expire, so the front of `order` is always
/// both the best eviction candidate and the next entry to expire.
#[derive(Debug, Default)]
struct Inner {
    map: HashMap<Arc<str>, CachedSize>,
    order: VecDeque<(Arc<str>, u64)>,
    next_seq: u64,
}

impl Inner {
    /// Is `(uri, seq)` the current record of a live map entry?
    fn is_live(&self, uri: &str, seq: u64) -> bool {
        self.map.get(uri).is_some_and(|e| e.seq == seq)
    }

    /// Drop stale records when they outnumber live entries 2:1, so overwrites
    /// and expiry-removals cannot grow `order` without bound. O(len) once per
    /// at least `max_entries` writes: amortised O(1) per `put`.
    fn compact_order_if_needed(&mut self, max_entries: usize) {
        if self.order.len() > 2 * max_entries.max(self.map.len()) {
            let map = &self.map;
            self.order
                .retain(|(uri, seq)| map.get(uri.as_ref()).is_some_and(|e| e.seq == *seq));
        }
    }

    /// Evict the oldest live entry. Returns false when the cache is empty.
    fn evict_oldest(&mut self) -> bool {
        while let Some((uri, seq)) = self.order.pop_front() {
            if self.is_live(&uri, seq) {
                self.map.remove(uri.as_ref());
                return true;
            }
        }
        false
    }
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
///
/// # Bounded size
///
/// The cache never holds more than `max_entries` entries (default
/// [`DEFAULT_SIZE_CACHE_MAX_ENTRIES`](crate::constants::DEFAULT_SIZE_CACHE_MAX_ENTRIES),
/// override with `S3DLIO_SIZE_CACHE_MAX_ENTRIES` or
/// [`ObjectSizeCache::with_max_entries`]), and an expired entry is removed as
/// soon as `get` finds it. When a new URI arrives at the limit the entry that
/// was written longest ago is evicted. All entries share one TTL, so that is
/// also the entry closest to expiry: expired entries always go first, and
/// no scan is needed (O(1) amortised per `put`). Overwriting a URI refreshes
/// its timestamp and moves it to the newest position.
#[derive(Debug)]
pub struct ObjectSizeCache {
    /// Internal cache storage (URI → cached size + timestamp) and write order
    cache: Arc<RwLock<Inner>>,

    /// Time-to-live for cached entries
    ttl: Duration,

    /// Maximum number of entries held at once (>= 1)
    max_entries: usize,
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
        Self::with_max_entries(ttl, crate::constants::size_cache_max_entries())
    }

    /// Create a size cache with the given TTL and an explicit entry limit
    /// (clamped to a minimum of 1). [`ObjectSizeCache::new`] takes the limit
    /// from `S3DLIO_SIZE_CACHE_MAX_ENTRIES`.
    pub fn with_max_entries(ttl: Duration, max_entries: usize) -> Self {
        Self {
            cache: Arc::new(RwLock::new(Inner::default())),
            ttl,
            max_entries: max_entries.max(1),
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
        {
            // Fast path: a hit or a miss needs only the shared read lock.
            let cache = self.cache.read().await;
            match cache.map.get(uri) {
                Some(entry) if entry.cached_at.elapsed() < self.ttl => {
                    tracing::trace!("Cache HIT: {} ({} bytes)", uri, entry.size);
                    return Some(entry.size);
                }
                Some(entry) => {
                    tracing::trace!(
                        "Cache EXPIRED: {} (age: {:?})",
                        uri,
                        entry.cached_at.elapsed()
                    );
                    // Expired: fall through and remove it under the write lock.
                }
                None => {
                    tracing::trace!("Cache MISS: {}", uri);
                    return None;
                }
            }
        }

        // Slow path (expired entry): remove it. Re-check under the write lock,
        // because a concurrent `put` may have refreshed the entry meanwhile.
        let mut cache = self.cache.write().await;
        match cache.map.get(uri) {
            Some(entry) if entry.cached_at.elapsed() < self.ttl => Some(entry.size),
            Some(_) => {
                cache.map.remove(uri);
                // Its record in `order` is now stale and is skipped lazily.
                None
            }
            None => None,
        }
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

        // Make room for a new URI. Overwriting an existing URI does not grow
        // the map, so it never evicts anything.
        if !cache.map.contains_key(uri.as_str()) {
            while cache.map.len() >= self.max_entries {
                if !cache.evict_oldest() {
                    break;
                }
            }
        }

        let seq = cache.next_seq;
        cache.next_seq += 1;
        let key: Arc<str> = Arc::from(uri);
        cache.map.insert(
            Arc::clone(&key),
            CachedSize {
                size,
                cached_at: Instant::now(),
                seq,
            },
        );
        cache.order.push_back((key, seq));
        cache.compact_order_if_needed(self.max_entries);
    }

    /// Clear expired entries from cache
    ///
    /// Expired entries are also removed when `get` finds them, and the cache
    /// is bounded by its entry limit, so calling this is optional; it frees
    /// expired entries that nobody asks for again.
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
        let before_count = cache.map.len();
        let ttl = self.ttl;
        cache.map.retain(|uri, entry| {
            let expired = entry.cached_at.elapsed() >= ttl;
            if expired {
                tracing::debug!("Removing expired cache entry: {}", uri);
            }
            !expired
        });
        let after_count = cache.map.len();
        cache.compact_order_if_needed(self.max_entries);
        let removed = before_count - after_count;
        if removed > 0 {
            tracing::info!(
                "Cleared {} expired cache entries ({} remain)",
                removed,
                after_count
            );
        }
    }

    /// Clear all entries from cache
    ///
    /// Useful for testing or when you want to force fresh stats.
    pub async fn clear(&self) {
        let mut cache = self.cache.write().await;
        cache.map.clear();
        cache.order.clear();
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
        let total = cache.map.len();
        let expired = cache
            .map
            .values()
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
        assert_eq!(stats.total_entries, 2); // Still in cache
        assert_eq!(stats.expired_entries, 2); // But expired
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
            cache
                .put(format!("s3://bucket/object-{}", i), i * 1000)
                .await;
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
        assert!(
            cache.get("s3://bucket/fast-expire").await.is_some(),
            "Should still be valid"
        );

        // Just after expiration
        sleep(Duration::from_millis(30)).await;
        assert!(
            cache.get("s3://bucket/fast-expire").await.is_none(),
            "Should be expired"
        );
    }

    #[tokio::test]
    async fn test_large_number_of_entries() {
        let cache = ObjectSizeCache::new(Duration::from_secs(60));

        // Add 1000 entries
        for i in 0..1000 {
            cache
                .put(format!("s3://bucket/object-{:04}", i), i * 1000)
                .await;
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
            cache
                .put(format!("s3://bucket/object-{:03}", i), i * 1000)
                .await;
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

        assert_eq!(
            successful_reads, 1000,
            "All reads should find cached entries"
        );
    }

    #[tokio::test]
    async fn test_memory_efficiency_large_cache() {
        let cache = ObjectSizeCache::new(Duration::from_secs(60));

        // Add 10,000 entries to test memory efficiency
        for i in 0..10_000 {
            cache
                .put(format!("s3://bucket/dataset/object-{:06}", i), i * 1000)
                .await;
        }

        let stats = cache.stats().await;
        assert_eq!(stats.total_entries, 10_000);
        assert_eq!(stats.valid_entries(), 10_000);

        // Verify clear works on large cache
        cache.clear().await;
        let stats_after = cache.stats().await;
        assert_eq!(stats_after.total_entries, 0);
    }

    // ── Issue #241: the cache is bounded and prunes expired entries ──────

    /// The limit comes from `S3DLIO_SIZE_CACHE_MAX_ENTRIES` for `new()`: the
    /// map must stop at the limit however many distinct URIs are written.
    #[tokio::test]
    async fn test_env_limit_bounds_the_cache() {
        let _guard = crate::test_env::lock_async().await;
        let _env =
            crate::test_env::EnvRestore::capture(&[crate::constants::ENV_SIZE_CACHE_MAX_ENTRIES]);
        std::env::set_var(crate::constants::ENV_SIZE_CACHE_MAX_ENTRIES, "50");
        let cache = ObjectSizeCache::new(Duration::from_secs(60));

        for i in 0..200u64 {
            cache.put(format!("s3://bucket/object-{i}"), i).await;
        }

        let stats = cache.stats().await;
        assert_eq!(
            stats.total_entries, 50,
            "200 distinct URIs written with a limit of 50 must leave exactly 50"
        );
        // The oldest were dropped, the newest kept.
        assert_eq!(cache.get("s3://bucket/object-0").await, None);
        assert_eq!(cache.get("s3://bucket/object-149").await, None);
        assert_eq!(cache.get("s3://bucket/object-150").await, Some(150));
        assert_eq!(cache.get("s3://bucket/object-199").await, Some(199));
    }

    /// `get` removes an expired entry it finds; before the fix it only
    /// reported a miss and left the entry in the map.
    #[tokio::test]
    async fn test_get_removes_expired_entry() {
        let cache = ObjectSizeCache::new(Duration::from_millis(50));
        cache.put("s3://bucket/a".to_string(), 1).await;
        cache.put("s3://bucket/b".to_string(), 2).await;
        sleep(Duration::from_millis(100)).await;
        assert_eq!(cache.stats().await.total_entries, 2);

        assert_eq!(cache.get("s3://bucket/a").await, None);
        assert_eq!(
            cache.stats().await.total_entries,
            1,
            "the expired entry that get() found must be gone"
        );
        assert_eq!(cache.get("s3://bucket/b").await, None);
        assert_eq!(cache.stats().await.total_entries, 0);
    }

    /// A URI removed on expiry can be cached again and is served normally.
    #[tokio::test]
    async fn test_reinsert_after_expiry_removal() {
        let cache = ObjectSizeCache::new(Duration::from_millis(50));
        cache.put("s3://bucket/a".to_string(), 1).await;
        sleep(Duration::from_millis(100)).await;
        assert_eq!(cache.get("s3://bucket/a").await, None);

        cache.put("s3://bucket/a".to_string(), 7).await;
        assert_eq!(cache.get("s3://bucket/a").await, Some(7));
        assert_eq!(cache.stats().await.total_entries, 1);
    }

    #[tokio::test]
    async fn test_with_max_entries_bounds_and_evicts_oldest_first() {
        let cache = ObjectSizeCache::with_max_entries(Duration::from_secs(60), 100);
        for i in 0..1000u64 {
            cache.put(format!("s3://bucket/object-{i:04}"), i).await;
        }
        assert_eq!(cache.stats().await.total_entries, 100);
        assert_eq!(cache.get("s3://bucket/object-0899").await, None);
        assert_eq!(cache.get("s3://bucket/object-0900").await, Some(900));
        assert_eq!(cache.get("s3://bucket/object-0999").await, Some(999));
    }

    #[tokio::test]
    async fn test_overwrite_refreshes_position_and_never_evicts() {
        let cache = ObjectSizeCache::with_max_entries(Duration::from_secs(60), 3);
        for name in ["a", "b", "c"] {
            cache.put(format!("s3://bucket/{name}"), 1).await;
        }
        // Overwriting a cached URI at the limit evicts nothing...
        cache.put("s3://bucket/a".to_string(), 2).await;
        assert_eq!(cache.stats().await.total_entries, 3);
        // ...and makes it the newest, so the next new URI evicts "b".
        cache.put("s3://bucket/d".to_string(), 4).await;
        assert_eq!(cache.get("s3://bucket/b").await, None);
        assert_eq!(cache.get("s3://bucket/a").await, Some(2));
        assert_eq!(cache.get("s3://bucket/c").await, Some(1));
        assert_eq!(cache.get("s3://bucket/d").await, Some(4));
    }

    /// Expired entries are the first to go when a new URI needs room.
    #[tokio::test]
    async fn test_expired_entries_are_evicted_before_live_ones() {
        let cache = ObjectSizeCache::with_max_entries(Duration::from_millis(100), 2);
        cache.put("s3://bucket/old".to_string(), 1).await;
        sleep(Duration::from_millis(150)).await;
        cache.put("s3://bucket/live".to_string(), 2).await;
        cache.put("s3://bucket/new".to_string(), 3).await;
        assert_eq!(cache.get("s3://bucket/live").await, Some(2));
        assert_eq!(cache.get("s3://bucket/new").await, Some(3));
        assert_eq!(cache.stats().await.total_entries, 2);
    }

    /// Repeated overwrites and expiry removals must not grow the internal
    /// write-order queue without bound either.
    #[tokio::test]
    async fn test_write_order_queue_stays_bounded() {
        let cache = ObjectSizeCache::with_max_entries(Duration::from_secs(60), 10);
        for i in 0..10_000u64 {
            cache.put("s3://bucket/same".to_string(), i).await;
        }
        for i in 0..10_000u64 {
            cache.put(format!("s3://bucket/object-{}", i % 25), i).await;
        }
        let inner = cache.cache.read().await;
        assert!(inner.map.len() <= 10);
        assert!(
            inner.order.len() <= 2 * 10 + 1,
            "write-order queue grew to {}",
            inner.order.len()
        );
    }

    #[tokio::test]
    async fn test_clear_resets_write_order() {
        let cache = ObjectSizeCache::with_max_entries(Duration::from_secs(60), 2);
        cache.put("s3://bucket/a".to_string(), 1).await;
        cache.put("s3://bucket/b".to_string(), 2).await;
        cache.clear().await;
        assert_eq!(cache.cache.read().await.order.len(), 0);
        cache.put("s3://bucket/c".to_string(), 3).await;
        assert_eq!(cache.get("s3://bucket/c").await, Some(3));
    }
}
