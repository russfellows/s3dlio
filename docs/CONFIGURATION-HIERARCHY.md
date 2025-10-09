# s3dlio Configuration Hierarchy

## Conceptual Levels (Aligned with AI/ML Training Concepts)

### Level 1: LoaderOptions (User-Facing, Training-Centric)
**Analogy**: PyTorch `DataLoader` constructor parameters

**Purpose**: Controls WHAT batches to create and HOW to iterate the dataset

**User Mental Model**: "I'm training a model, I need batches of data"

**Key Fields**:
```rust
pub struct LoaderOptions {
    // Batch construction
    batch_size: usize,           // PyTorch: batch_size
    drop_last: bool,             // PyTorch: drop_last
    shuffle: bool,               // PyTorch: shuffle
    
    // Dataset iteration
    num_workers: usize,          // PyTorch: num_workers
    prefetch: usize,             // PyTorch: prefetch_factor
    
    // Training optimizations
    pin_memory: bool,            // PyTorch: pin_memory
    persistent_workers: bool,    // PyTorch: persistent_workers
    
    // ... other training-related options
}
```

**Who Uses It**: ML practitioners, data scientists, model trainers

**Visibility**: **ALWAYS VISIBLE** - This is the primary API

---

### Level 2: PoolConfig (Performance Tuning)
**Analogy**: Internal worker pool management in PyTorch (hidden behind `num_workers`)

**Purpose**: Controls HOW data is fetched efficiently to fill batches

**User Mental Model**: "I want to tune download performance without changing training behavior"

**Key Fields**:
```rust
pub struct PoolConfig {
    pool_size: usize,           // Concurrent download workers
    readahead_batches: usize,   // Batch prefetch depth
    batch_timeout: Duration,    // Request timeout
    max_inflight: usize,        // Global request limit
}
```

**Who Uses It**: Performance engineers, infrastructure teams

**Visibility**: **OPTIONAL** - Good defaults exist, advanced users can tune

**Current API Pattern** (GOOD):
```rust
// Simple case: Use defaults
let stream = dataloader.stream();  // Uses PoolConfig::default()

// Advanced case: Tune performance
let pool_config = PoolConfig { pool_size: 128, ..Default::default() };
let stream = dataloader.stream_with_pool(pool_config);
```

---

### Level 3: RangeEngineConfig (Internal Optimization)
**Analogy**: File I/O internals in PyTorch Dataset implementations (buffering, caching, etc.)

**Purpose**: Controls storage-layer optimizations for large objects

**User Mental Model**: "How does s3dlio split large files into parallel range requests?"

**Key Fields**:
```rust
pub struct RangeEngineConfig {
    chunk_size: usize,              // Range chunk size (64MB default)
    max_concurrent_ranges: usize,   // Parallel ranges per object (32 default)
    min_split_size: u64,            // Threshold to trigger splitting (4MB)
    range_timeout: Duration,        // Per-range timeout
}
```

**Who Uses It**: Storage engineers, infrastructure experts debugging edge cases

**Visibility**: **MOSTLY HIDDEN** - Exposed only in specialized APIs

**Current Implementation** (GOOD):
- Used internally by `ObjectStore` implementations
- Backend-specific thresholds via constants (file: 4MB, directio: 16MB, S3: 4MB)
- Not exposed in high-level DataLoader APIs

---

## Comparison with PyTorch DataLoader

| s3dlio | PyTorch Equivalent | Visibility |
|--------|-------------------|------------|
| `LoaderOptions` | `DataLoader(batch_size, num_workers, ...)` | ✅ Always visible |
| `PoolConfig` | Worker pool internals (managed by `num_workers`) | 🟡 Optional tuning |
| `RangeEngineConfig` | Dataset file I/O internals | ❌ Hidden |

### PyTorch Example:
```python
# User only configures high-level options
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,        # Level 1: Batch construction
    num_workers=4,        # Level 1: Parallelism
    prefetch_factor=2,    # Level 1: Prefetching
    pin_memory=True,      # Level 1: GPU optimization
)

# Level 2 (worker pool) is automatically managed
# Level 3 (file I/O) is hidden in Dataset implementation
```

### s3dlio Equivalent:
```rust
// Simple case: Like PyTorch
let options = LoaderOptions {
    batch_size: 32,
    num_workers: 4,
    prefetch: 2,
    pin_memory: true,
    ..Default::default()
};
let loader = DataLoader::new(dataset, options);
let stream = loader.stream();

// Advanced case: Tune download pool (NOT in PyTorch)
let pool_config = PoolConfig { pool_size: 128, ..Default::default() };
let loader = AsyncPoolDataLoader::new(dataset, options);
let stream = loader.stream_with_pool(pool_config);
```

---

## Current State Analysis

### ✅ What Works Well:

1. **LoaderOptions is comprehensive** - Covers all training use cases
2. **PoolConfig has good defaults** - `stream()` method uses `PoolConfig::default()`
3. **RangeEngine is properly hidden** - Not exposed in DataLoader APIs
4. **Two-tier API** - Simple `stream()` vs advanced `stream_with_pool()`

### 🟡 Potential Improvements:

#### Option A: Derive PoolConfig from LoaderOptions
Map existing LoaderOptions fields to PoolConfig:

```rust
impl LoaderOptions {
    /// Convert to PoolConfig using existing fields as hints
    pub fn to_pool_config(&self) -> PoolConfig {
        PoolConfig {
            pool_size: self.num_workers * 16,  // Scale from num_workers
            readahead_batches: self.prefetch.max(2),  // Use prefetch field
            ..Default::default()
        }
    }
}

// Usage:
let pool_config = options.to_pool_config();
let stream = dataloader.stream_with_pool(pool_config);
```

**Pros**: Reuses existing LoaderOptions fields, maintains conceptual hierarchy
**Cons**: Heuristic mapping may not suit all use cases

#### Option B: Embed PoolConfig in LoaderOptions
Add pool-related fields directly to LoaderOptions:

```rust
pub struct LoaderOptions {
    // ... existing fields ...
    
    // Async pool configuration (None = use defaults)
    pub async_pool_size: Option<usize>,
    pub async_readahead_batches: Option<usize>,
}
```

**Pros**: Single configuration surface
**Cons**: Pollutes LoaderOptions with implementation details

#### Option C: Keep Current Design (RECOMMENDED)
Maintain separation but improve documentation:

**Pros**:
- Clean separation of concerns
- Good defaults via `stream()`
- Advanced tuning via `stream_with_pool()`
- Matches current usage patterns

**Cons**: 
- Requires understanding two configs for advanced use

---

## Recommendations

### 1. **Keep Current API Structure** ✅
The existing pattern is sound:
```rust
dataloader.stream()                    // Level 1: Simple, like PyTorch
dataloader.stream_with_pool(config)    // Level 2: Advanced tuning
```

### 2. **Improve Documentation** 📚
Make the hierarchy explicit in:
- README.md: "Configuration Levels" section
- API docs: Cross-reference LoaderOptions ↔ PoolConfig
- Examples: Show both simple and advanced usage

### 3. **Add Convenience Constructor** 🛠️
```rust
impl PoolConfig {
    /// Create PoolConfig with sensible scaling from LoaderOptions
    pub fn from_loader_options(opts: &LoaderOptions) -> Self {
        Self {
            pool_size: opts.num_workers * 16,
            readahead_batches: opts.prefetch.max(2),
            ..Default::default()
        }
    }
}
```

### 4. **Keep RangeEngineConfig Hidden** ❌👁️
- Continue using internal constants
- Don't expose in LoaderOptions or PoolConfig
- Only surface in specialized debugging/profiling APIs

---

## Summary Table

| Config | Level | User Visibility | PyTorch Equivalent | Status |
|--------|-------|----------------|-------------------|---------|
| **LoaderOptions** | 1 | Always visible | `DataLoader(...)` | ✅ Good |
| **PoolConfig** | 2 | Optional tuning | Internal workers | ✅ Good (has defaults) |
| **RangeEngineConfig** | 3 | Hidden | Dataset I/O | ✅ Good (properly hidden) |

**Conclusion**: Current design aligns well with AI/ML training concepts. Minor documentation improvements recommended, but no major refactoring needed.
