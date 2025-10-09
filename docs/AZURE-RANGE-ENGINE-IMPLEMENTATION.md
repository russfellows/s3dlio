# Azure Backend RangeEngine Integration - Implementation Plan

**Date**: October 9, 2025  
**Branch**: feat/v0.9.3-range-engine-backends  
**Status**: In Progress

---

## Analysis Summary

### Current Azure Backend State

**Strengths**:
- ✅ `AzureBlob` client with working `get()` and `get_range()` methods
- ✅ Returns `Bytes` for zero-copy efficiency
- ✅ `get_range()` already implements HTTP range requests
- ✅ ObjectStore trait fully implemented

**Limitations**:
- ❌ No configuration struct (unit struct only)
- ❌ Not Clone (required for RangeEngine closure)
- ❌ No range engine integration
- ❌ Always uses simple sequential downloads

### Comparison with FileSystemObjectStore (Reference)

| Aspect | FileSystemObjectStore | AzureObjectStore |
|--------|----------------------|------------------|
| Structure | `struct { config: FileSystemConfig }` | `struct` (unit) |
| Clone | ✅ Implemented | ❌ Not needed (unit) |
| Configuration | `enable_range_engine`, `range_engine` | None |
| get() Strategy | Size-based (threshold check) | Always simple |
| RangeEngine | ✅ Integrated | ❌ Not integrated |
| Client Pattern | Self-contained | External `AzureBlob` |

---

## Implementation Strategy

### Option A: Add Configuration (CHOSEN)

Make AzureObjectStore stateful with configuration, matching FileSystemObjectStore pattern.

**Advantages**:
- ✅ Consistent with FileSystemObjectStore
- ✅ Clean, testable design
- ✅ Easy to configure per-instance
- ✅ Follows established pattern

**Changes Required**:
1. Create `AzureConfig` struct
2. Add config field to `AzureObjectStore`
3. Implement `Clone` for `AzureObjectStore`
4. Add `get_with_range_engine()` helper
5. Modify `get()` to use size-based strategy
6. Update factory functions

---

## Implementation Plan

### Step 1: Add AzureConfig Struct

```rust
/// Configuration for Azure Blob Storage backend with RangeEngine support
#[derive(Debug, Clone)]
pub struct AzureConfig {
    /// Enable RangeEngine for concurrent range downloads
    /// Default: true (network storage benefits from parallel ranges)
    pub enable_range_engine: bool,
    
    /// RangeEngine configuration
    /// Network-optimized defaults: 4MB threshold, 32 concurrent ranges
    pub range_engine: RangeEngineConfig,
}

impl Default for AzureConfig {
    fn default() -> Self {
        Self {
            enable_range_engine: true,
            range_engine: RangeEngineConfig {
                chunk_size: DEFAULT_RANGE_ENGINE_CHUNK_SIZE,  // 64MB chunks
                max_concurrent_ranges: DEFAULT_RANGE_ENGINE_MAX_CONCURRENT,  // 32 parallel
                min_split_size: DEFAULT_NETWORK_RANGE_ENGINE_THRESHOLD,  // 4MB threshold
                range_timeout: Duration::from_secs(DEFAULT_RANGE_TIMEOUT_SECS),  // 30s
            },
        }
    }
}
```

**New Constant Needed**:
```rust
// src/constants.rs
/// Threshold for network storage (Azure, GCS, S3) - lower than file storage
/// Network storage benefits from concurrent ranges even for smaller files
pub const DEFAULT_NETWORK_RANGE_ENGINE_THRESHOLD: u64 = 4 * 1024 * 1024; // 4MB
```

---

### Step 2: Update AzureObjectStore Structure

```rust
// Before
pub struct AzureObjectStore;

impl AzureObjectStore {
    pub fn new() -> Self { Self }
    #[inline]
    pub fn boxed() -> Box<dyn ObjectStore> { Box::new(Self) }
}

// After
#[derive(Clone)]
pub struct AzureObjectStore {
    config: AzureConfig,
}

impl AzureObjectStore {
    pub fn new() -> Self {
        Self {
            config: AzureConfig::default(),
        }
    }
    
    pub fn with_config(config: AzureConfig) -> Self {
        Self { config }
    }
    
    #[inline]
    pub fn boxed() -> Box<dyn ObjectStore> {
        Box::new(Self::new())
    }
}
```

---

### Step 3: Add get_with_range_engine() Helper

```rust
impl AzureObjectStore {
    /// Download using RangeEngine for concurrent range requests
    async fn get_with_range_engine(&self, uri: &str, object_size: u64) -> Result<Bytes> {
        let engine = RangeEngine::new(self.config.range_engine.clone());
        
        // Create closure that captures uri for get_range calls
        let uri_owned = uri.to_string();
        let self_clone = self.clone();
        
        let get_range_fn = move |offset: u64, length: u64| {
            let uri = uri_owned.clone();
            let store = self_clone.clone();
            async move {
                store.get_range(&uri, offset, Some(length)).await
            }
        };
        
        let (bytes, stats) = engine.download(object_size, get_range_fn, None).await?;
        
        info!(
            "RangeEngine (Azure) downloaded {} bytes in {} ranges: {:.2} MB/s ({:.2} Gbps)",
            stats.bytes_downloaded,
            stats.ranges_processed,
            stats.throughput_mbps(),
            stats.throughput_gbps()
        );
        
        Ok(bytes)
    }
}
```

---

### Step 4: Modify get() Method

```rust
#[async_trait]
impl ObjectStore for AzureObjectStore {
    async fn get(&self, uri: &str) -> Result<Bytes> {
        let (cli, _acct, _cont, key) = Self::client_for_uri(uri)?;
        
        // Get blob size to decide download strategy
        let props = cli.stat(&key).await?;
        let object_size = props.content_length;
        
        // Use RangeEngine for large blobs if enabled
        if self.config.enable_range_engine && object_size >= self.config.range_engine.min_split_size {
            trace!(
                "Blob size {} >= threshold {}, using RangeEngine for {}",
                object_size,
                self.config.range_engine.min_split_size,
                uri
            );
            return self.get_with_range_engine(uri, object_size).await;
        }
        
        // Simple sequential download for small blobs
        trace!(
            "Blob size {} < threshold {}, using simple download for {}",
            object_size,
            self.config.range_engine.min_split_size,
            uri
        );
        
        let b = cli.get(&key).await?;
        Ok(b)
    }
    
    // get_range() unchanged - already works correctly
    async fn get_range(&self, uri: &str, offset: u64, length: Option<u64>) -> Result<Bytes> {
        let (cli, _acct, _cont, key) = Self::client_for_uri(uri)?;
        let end = length.map(|len| offset + len - 1);
        let b = cli.get_range(&key, offset, end).await?;
        Ok(b)
    }
}
```

---

### Step 5: Update Import Statements

```rust
// Add to imports at top of src/object_store.rs
use crate::range_engine_generic::{RangeEngine, RangeEngineConfig};
use crate::constants::{
    DEFAULT_RANGE_ENGINE_CHUNK_SIZE,
    DEFAULT_RANGE_ENGINE_MAX_CONCURRENT,
    DEFAULT_NETWORK_RANGE_ENGINE_THRESHOLD,
    DEFAULT_RANGE_TIMEOUT_SECS,
};
use std::time::Duration;
use tracing::{trace, info};
```

---

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_azure_config_defaults() {
        let config = AzureConfig::default();
        assert!(config.enable_range_engine);
        assert_eq!(config.range_engine.min_split_size, 4 * 1024 * 1024); // 4MB
        assert_eq!(config.range_engine.max_concurrent_ranges, 32);
    }
    
    #[test]
    fn test_azure_store_creation() {
        let store = AzureObjectStore::new();
        assert!(store.config.enable_range_engine);
        
        let custom_config = AzureConfig {
            enable_range_engine: false,
            ..Default::default()
        };
        let store2 = AzureObjectStore::with_config(custom_config);
        assert!(!store2.config.enable_range_engine);
    }
}
```

### Integration Tests

Create `tests/test_azure_range_engine.rs`:

```rust
#[tokio::test]
async fn test_azure_small_blob_simple_download() {
    // Test that small blobs (< 4MB) use simple download
    // Requires Azure credentials and test container
}

#[tokio::test]
async fn test_azure_large_blob_range_engine() {
    // Test that large blobs (> 4MB) use RangeEngine
    // Verify concurrent range requests
}

#[tokio::test]
async fn test_azure_range_engine_performance() {
    // Benchmark: RangeEngine vs simple download for large blobs
    // Expect 30-50% improvement
}
```

---

## Performance Expectations

### Network Storage Characteristics

**Why Azure Benefits from RangeEngine**:
- Network latency: 10-100ms per request
- Concurrent ranges hide latency
- No seek overhead (unlike local files)
- No disk contention
- No page cache benefits to preserve

**Expected Improvements**:
- **Small blobs (< 4MB)**: No change (uses simple download)
- **Medium blobs (4-64MB)**: 20-40% faster
- **Large blobs (> 64MB)**: 30-50% faster
- **Huge blobs (> 1GB)**: 40-60% faster

**Configuration Recommendations**:
```rust
// Default (good for most cases)
AzureConfig::default()  // 4MB threshold, 32 parallel

// Aggressive (high bandwidth, low latency)
AzureConfig {
    enable_range_engine: true,
    range_engine: RangeEngineConfig {
        chunk_size: 32 * 1024 * 1024,  // 32MB chunks
        max_concurrent_ranges: 64,     // 64 parallel
        min_split_size: 2 * 1024 * 1024,  // 2MB threshold
        ..Default::default()
    },
}

// Conservative (limited bandwidth or cost concerns)
AzureConfig {
    enable_range_engine: true,
    range_engine: RangeEngineConfig {
        max_concurrent_ranges: 16,     // 16 parallel
        min_split_size: 8 * 1024 * 1024,  // 8MB threshold
        ..Default::default()
    },
}
```

---

## Implementation Checklist

### Code Changes
- [ ] Add `DEFAULT_NETWORK_RANGE_ENGINE_THRESHOLD` constant
- [ ] Create `AzureConfig` struct with documentation
- [ ] Update `AzureObjectStore` structure (add config field)
- [ ] Implement `Clone` for `AzureObjectStore`
- [ ] Add `get_with_range_engine()` helper method
- [ ] Modify `get()` to use size-based strategy
- [ ] Update factory functions (`new()`, `with_config()`, `boxed()`)
- [ ] Add import statements

### Testing
- [ ] Add unit tests for AzureConfig
- [ ] Add unit tests for AzureObjectStore creation
- [ ] Create integration test file
- [ ] Test small blob simple download
- [ ] Test large blob RangeEngine download
- [ ] Performance benchmark tests

### Documentation
- [ ] Add inline documentation for AzureConfig
- [ ] Update get() method documentation
- [ ] Add example usage in comments
- [ ] Update TESTING-GUIDE.md with Azure tests

### Verification
- [ ] `cargo build --release` (zero warnings)
- [ ] `cargo test --lib` (all tests pass)
- [ ] `cargo clippy` (zero warnings)
- [ ] Integration tests with real Azure (if available)
- [ ] Performance validation (30-50% improvement)

---

## Notes

**Key Differences from File Backend**:
1. **Lower threshold**: 4MB instead of 64MB (network benefits more)
2. **Higher concurrency**: Default 32 instead of 16 (no disk contention)
3. **No page cache**: Don't need to worry about page cache pollution
4. **Network latency**: Concurrent ranges specifically hide network latency

**Compatibility**:
- Zero breaking changes (existing code continues to work)
- Default behavior enables RangeEngine (transparent optimization)
- Users can disable via `AzureConfig { enable_range_engine: false }`

**Next Steps**:
After Azure integration complete:
1. Apply same pattern to GCS backend
2. Evaluate S3 backend (may already be optimized)
3. Comprehensive benchmarking across all backends
4. Update documentation and examples

---

**Implementation Start**: October 9, 2025  
**Expected Completion**: 2-3 hours  
**Status**: Ready to begin coding
