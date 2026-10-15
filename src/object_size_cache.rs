// src/object_size_cache.rs
//
// SPDX-License-Identifier: Apache-2.0 OR MIT
// SPDX-FileCopyrightText: 2025 Russ Fellows <russ.fellows@gmail.com>

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;

/// "No slot" marker for the intrusive list links.
const NIL: usize = usize::MAX;

/// Cached object size with timestamp for TTL expiration
#[derive(Debug, Clone)]
struct CachedSize {
    /// Object size in bytes
    size: u64,
    /// When this entry was (last) written
    cached_at: Instant,
    /// Index of this entry's node in [`Inner::nodes`]
    slot: usize,
}

/// One link of the write-order list. `key` is `None` while the slot is free.
#[derive(Debug)]
struct Node {
    key: Option<Arc<str>>,
    prev: usize,
    next: usize,
}

/// The map plus the order in which entries were (last) written.
///
/// The order is a doubly-linked list stored in a slab (`nodes`, with a free
/// list), oldest write at `head`, newest at `tail`. Every map entry has exactly
/// one node and every node belongs to exactly one entry, so there are never
/// stale records and nothing ever needs a compaction pass: insert, overwrite
/// (unlink + relink at the tail), expiry removal and eviction are all O(1).
/// Because each write stamps `cached_at` while the node moves to the tail, the
/// list is sorted by `cached_at`: with one shared TTL the head is both the
/// entry written longest ago and the first to expire, and the expired entries
/// are always a prefix of the list.
#[derive(Debug)]
struct Inner {
    map: HashMap<Arc<str>, CachedSize>,
    nodes: Vec<Node>,
    free: Vec<usize>,
    head: usize,
    tail: usize,
}

impl Inner {
    fn new() -> Self {
        Self {
            map: HashMap::new(),
            nodes: Vec::new(),
            free: Vec::new(),
            head: NIL,
            tail: NIL,
        }
    }

    /// Detach `slot` from the list (it stays allocated).
    fn unlink(&mut self, slot: usize) {
        let (prev, next) = (self.nodes[slot].prev, self.nodes[slot].next);
        if prev == NIL {
            self.head = next;
        } else {
            self.nodes[prev].next = next;
        }
        if next == NIL {
            self.tail = prev;
        } else {
            self.nodes[next].prev = prev;
        }
        self.nodes[slot].prev = NIL;
        self.nodes[slot].next = NIL;
    }

    /// Attach a detached `slot` at the tail (newest).
    fn push_back(&mut self, slot: usize) {
        self.nodes[slot].prev = self.tail;
        self.nodes[slot].next = NIL;
        if self.tail == NIL {
            self.head = slot;
        } else {
            self.nodes[self.tail].next = slot;
        }
        self.tail = slot;
    }

    /// Allocate a node for `key` (reusing a free slot) and attach it at the tail.
    fn alloc_back(&mut self, key: Arc<str>) -> usize {
        let slot = match self.free.pop() {
            Some(slot) => {
                self.nodes[slot].key = Some(key);
                slot
            }
            None => {
                self.nodes.push(Node {
                    key: Some(key),
                    prev: NIL,
                    next: NIL,
                });
                self.nodes.len() - 1
            }
        };
        self.push_back(slot);
        slot
    }

    /// Detach `slot` and put it on the free list.
    fn release(&mut self, slot: usize) {
        self.unlink(slot);
        self.nodes[slot].key = None;
        self.free.push(slot);
    }

    /// Remove the entry for `uri` from map and list. O(1).
    fn remove(&mut self, uri: &str) -> Option<CachedSize> {
        let entry = self.map.remove(uri)?;
        self.release(entry.slot);
        Some(entry)
    }

    /// Evict the entry written longest ago. Returns false when empty. O(1).
    fn evict_oldest(&mut self) -> bool {
        if self.head == NIL {
            return false;
        }
        let slot = self.head;
        if let Some(key) = self.nodes[slot].key.take() {
            self.map.remove(key.as_ref());
        }
        self.unlink(slot);
        self.free.push(slot);
        true
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
/// also the entry closest to expiry: expired entries always go first. Every
/// operation is O(1) (a slab-backed linked list gives the write order; no
/// scan and no compaction pass exists). Overwriting a URI updates it in place
/// (no allocation), refreshes its timestamp and moves it to the newest
/// position.
///
/// Eviction is oldest-written-first, not "least recently read": a loop that
/// cycles over more distinct URIs than the limit evicts each entry just
/// before it is needed again and gets almost no hits. Raise the limit
/// (`S3DLIO_SIZE_CACHE_MAX_ENTRIES`) above the dataset's object count.
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
            cache: Arc::new(RwLock::new(Inner::new())),
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
                cache.remove(uri);
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
        let now = Instant::now();

        // Overwrite: update in place -- one hash, no allocation, no new list
        // node -- and move the node to the newest position. (`HashMap::entry`
        // would need an owned `Arc<str>` key up front, i.e. an allocation on
        // every overwrite, which is the common case in the skip-HEAD path.)
        if let Some(entry) = cache.map.get_mut(uri.as_str()) {
            entry.size = size;
            entry.cached_at = now;
            let slot = entry.slot;
            cache.unlink(slot);
            cache.push_back(slot);
            return;
        }

        // New URI: make room, then insert.
        while cache.map.len() >= self.max_entries {
            if !cache.evict_oldest() {
                break;
            }
        }
        let key: Arc<str> = Arc::from(uri);
        let slot = cache.alloc_back(Arc::clone(&key));
        cache.map.insert(
            key,
            CachedSize {
                size,
                cached_at: now,
                slot,
            },
        );
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
        // The list is ordered by write time, so the expired entries are a
        // prefix: pop from the head until the first live entry.
        while cache.head != NIL {
            let slot = cache.head;
            let Some(key) = cache.nodes[slot].key.clone() else {
                break;
            };
            let expired = cache
                .map
                .get(key.as_ref())
                .is_some_and(|e| e.cached_at.elapsed() >= self.ttl);
            if !expired {
                break;
            }
            tracing::debug!("Removing expired cache entry: {}", key);
            cache.remove(key.as_ref());
        }
        let after_count = cache.map.len();
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
        *cache = Inner::new();
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

    /// Walk the write-order list and check it against the map: same size,
    /// every node's key maps back to that node's slot, `cached_at`
    /// non-decreasing head to tail, and the slab has no leaked slots.
    fn assert_consistent(inner: &Inner) {
        let mut seen = 0usize;
        let mut slot = inner.head;
        let mut prev = NIL;
        let mut last_at: Option<Instant> = None;
        while slot != NIL {
            let node = &inner.nodes[slot];
            assert_eq!(node.prev, prev, "broken back link at slot {slot}");
            let key = node.key.as_ref().expect("listed node has a key");
            let entry = inner.map.get(key.as_ref()).expect("listed key is in map");
            assert_eq!(entry.slot, slot, "map slot disagrees with list");
            if let Some(t) = last_at {
                assert!(entry.cached_at >= t, "list not sorted by write time");
            }
            last_at = Some(entry.cached_at);
            seen += 1;
            prev = slot;
            slot = node.next;
        }
        assert_eq!(prev, inner.tail, "tail disagrees with the list");
        assert_eq!(seen, inner.map.len(), "list and map sizes differ");
        assert_eq!(
            inner.nodes.len(),
            inner.map.len() + inner.free.len(),
            "slab leaked a slot"
        );
    }

    /// Overwriting a handful of keys a million times (the skip-HEAD pattern:
    /// every GET re-puts a cached size) creates no stale records, grows
    /// nothing, and never needs an O(limit) pass: the structure stays
    /// exactly one node per entry.
    #[tokio::test]
    async fn test_million_overwrites_keep_structure_bounded() {
        let cache = ObjectSizeCache::with_max_entries(Duration::from_secs(60), 10);
        for i in 0..1_000_000u64 {
            cache.put(format!("s3://bucket/object-{}", i % 3), i).await;
        }
        let inner = cache.cache.read().await;
        assert_eq!(inner.map.len(), 3);
        assert_eq!(inner.nodes.len(), 3, "overwrites must reuse their node");
        assert!(inner.free.is_empty());
        assert_consistent(&inner);
    }

    /// Expiry removal followed by re-insertion (a new epoch after the TTL)
    /// recycles slots instead of growing the slab.
    #[tokio::test]
    async fn test_expiry_churn_recycles_slots() {
        let cache = ObjectSizeCache::with_max_entries(Duration::from_millis(20), 100);
        for _ in 0..10 {
            for i in 0..50u64 {
                cache.put(format!("s3://bucket/object-{i}"), i).await;
            }
            sleep(Duration::from_millis(30)).await;
            for i in 0..50u64 {
                assert_eq!(cache.get(&format!("s3://bucket/object-{i}")).await, None);
            }
        }
        let inner = cache.cache.read().await;
        assert_eq!(inner.map.len(), 0);
        assert!(
            inner.nodes.len() <= 50,
            "slab grew to {}",
            inner.nodes.len()
        );
        assert_consistent(&inner);
    }

    /// clear_expired removes exactly the expired prefix; an entry refreshed
    /// by an overwrite is newer and survives.
    #[tokio::test]
    async fn test_clear_expired_spares_refreshed_entries() {
        let cache = ObjectSizeCache::with_max_entries(Duration::from_millis(120), 10);
        cache.put("s3://bucket/a".to_string(), 1).await;
        cache.put("s3://bucket/b".to_string(), 2).await;
        sleep(Duration::from_millis(80)).await;
        cache.put("s3://bucket/a".to_string(), 3).await; // refresh a
        sleep(Duration::from_millis(80)).await; // b is 160 ms old, a is 80 ms
        cache.clear_expired().await;
        assert_eq!(cache.get("s3://bucket/a").await, Some(3));
        assert_eq!(cache.get("s3://bucket/b").await, None);
        assert_consistent(&*cache.cache.read().await);
    }

    /// A deterministic pseudo-random mix of put / get / clear_expired keeps
    /// the list, map and slab consistent and the size within the limit.
    #[tokio::test]
    async fn test_random_operations_keep_invariants() {
        let cache = ObjectSizeCache::with_max_entries(Duration::from_millis(5), 16);
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        for step in 0..5_000u32 {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let key = format!("s3://bucket/k{}", (x >> 33) % 40);
            match (x >> 20) % 5 {
                0..=2 => cache.put(key, step as u64).await,
                3 => {
                    let _ = cache.get(&key).await;
                }
                _ => cache.clear_expired().await,
            }
            if step % 500 == 0 {
                sleep(Duration::from_millis(6)).await;
            }
            if step % 97 == 0 {
                let inner = cache.cache.read().await;
                assert!(inner.map.len() <= 16);
                assert_consistent(&inner);
            }
        }
        assert_consistent(&*cache.cache.read().await);
    }

    #[tokio::test]
    async fn test_clear_resets_write_order() {
        let cache = ObjectSizeCache::with_max_entries(Duration::from_secs(60), 2);
        cache.put("s3://bucket/a".to_string(), 1).await;
        cache.put("s3://bucket/b".to_string(), 2).await;
        cache.clear().await;
        assert_eq!(cache.cache.read().await.nodes.len(), 0);
        cache.put("s3://bucket/c".to_string(), 3).await;
        assert_eq!(cache.get("s3://bucket/c").await, Some(3));
    }
}
