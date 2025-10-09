# Post v0.9.1 TODO - Priority Work Items

## Status Check (October 2025)

### ✅ Completed in v0.9.1
- Zero-copy Python API implementation
- Universal get_many() across all backends
- Range request support (CLI --offset/--length + Python get_range())
- Comprehensive testing and documentation
- Version bump to 0.9.1

### 🎯 Next Priority Items

## 1. GCS Backend Completion Testing 🔥
**Priority:** HIGH  
**Effort:** Medium  
**Status:** Backend works, needs validation

### What's Already Working
Per `GCS-TESTING-SUMMARY.md`, the GCS backend is FULLY FUNCTIONAL:
- ✅ Upload (PUT) works
- ✅ Download (GET) works  
- ✅ List works
- ✅ Delete works
- ✅ Authentication (ADC) works
- ✅ CLI integration complete
- ✅ ObjectStore trait implementation complete

### What Needs Validation

#### A. Python API Testing
**File:** `src/python_api/python_core_api.rs`

Test that Python bindings work with GCS:
```python
import s3dlio

# Test basic operations
s3dlio.upload(['test.txt'], 'gs://bucket/prefix/')
s3dlio.download('gs://bucket/test.txt', './local/')
s3dlio.list('gs://bucket/prefix/')

# Test zero-copy get
view = s3dlio.get('gs://bucket/file.bin')
data = bytes(view.memoryview())

# Test zero-copy get_range
view = s3dlio.get_range('gs://bucket/file.bin', 0, 1024)

# Test get_many
views = s3dlio.get_many(['gs://bucket/file1', 'gs://bucket/file2'])

# Test with PyTorch
from s3dlio import S3IterableDataset
dataset = S3IterableDataset("gs://bucket/data/", batch_size=32)
```

**Action Items:**
- [ ] Create `python/tests/test_gcs_api.py`
- [ ] Test all Python functions with gs:// URIs
- [ ] Verify zero-copy works with GCS
- [ ] Test PyTorch/TensorFlow integration
- [ ] Document any GCS-specific quirks

#### B. Performance Benchmarking
**Scripts:** `scripts/run_backend_comparison.sh`, `scripts/compare_backends_performance.sh`

Benchmark GCS against other backends:
- [ ] Upload throughput (target: 2.5+ GB/s)
- [ ] Download throughput (target: 5+ GB/s)
- [ ] Compare: S3 vs Azure vs GCS vs File vs DirectIO
- [ ] Test concurrent operations scaling
- [ ] Test large file handling (>10GB)

**Action Items:**
- [ ] Update `scripts/build_performance_variants.sh` to include GCS
- [ ] Run sustained performance tests
- [ ] Document performance characteristics
- [ ] Identify any optimization opportunities

#### C. Error Handling Edge Cases
Test comprehensive error scenarios:
- [ ] Network failures during upload/download
- [ ] Authentication token expiration
- [ ] Invalid bucket/object names
- [ ] Permission denied errors
- [ ] Quota exceeded scenarios

**Action Items:**
- [ ] Create `tests/test_gcs_errors.rs`
- [ ] Test error propagation to Python
- [ ] Verify error messages are clear
- [ ] Document error handling

---

## 2. Stage 3: Backend-Agnostic Range Engine ⚡
**Priority:** HIGH  
**Effort:** High  
**Status:** Deferred from v0.9.0

### Background
Per `STAGE3-DEFERRAL.md`, this was deferred from v0.9.0 to v0.9.1, but we focused on zero-copy instead.

### Current State
- ✅ S3 has high-performance concurrent range GET operations
- ❌ File/DirectIO/Azure/GCS use simple sequential reads
- ✅ API supports range requests: `store.get_range(uri, offset, length)`
- ✅ CLI supports range flags: `--offset` and `--length`

### Goal
Enable ALL backends to use concurrent range downloads for large files:
- Split large files into chunks
- Download chunks concurrently
- Reassemble in correct order
- 30-50% expected throughput improvement

### Implementation Plan

#### Phase 1: Range Engine Core
**File:** `src/range_engine.rs` (already exists!)

Check current implementation:
```bash
grep -n "struct RangeEngine" src/range_engine.rs
```

The range engine exists but may only be used by S3. Need to:
- [ ] Review current RangeEngine implementation
- [ ] Identify S3-specific assumptions
- [ ] Extract generic range splitting logic
- [ ] Make engine backend-agnostic

#### Phase 2: Backend Integration
**Files:** 
- `src/file_store.rs` - File backend
- `src/file_store_direct.rs` - DirectIO backend  
- `src/object_store.rs` - Azure/GCS backends

For each backend:
- [ ] Implement concurrent range downloads
- [ ] Use RangeEngine for file size threshold (e.g., >4MB)
- [ ] Add configuration for chunk size and concurrency
- [ ] Ensure zero-copy throughout

#### Phase 3: Testing
- [ ] Unit tests for RangeEngine
- [ ] Integration tests for each backend
- [ ] Performance benchmarks (before/after)
- [ ] Verify correctness (checksum validation)

#### Phase 4: Documentation
- [ ] Update API docs with range engine behavior
- [ ] Document performance characteristics
- [ ] Add configuration examples
- [ ] Update Changelog

**Expected Outcome:**
- 30-50% throughput improvement for large files on File/Azure/GCS/DirectIO
- Zero API changes (transparent optimization)
- Configurable thresholds and chunk sizes

---

## 3. Additional GCS Features (Lower Priority)

### A. List Buckets Support ⏳
**Priority:** LOW  
**Effort:** Medium

GCS doesn't expose project-level bucket listing easily.

**Options:**
1. Check if gcloud-storage SDK supports list_buckets() with project ID
2. Add --project flag to CLI if needed
3. Document workaround: `gcloud storage buckets list`
4. **Recommendation:** Document workaround, defer indefinitely

### B. Resumable Upload Implementation ⏳
**Priority:** LOW  
**Effort:** High

GCS supports resumable uploads for large files.

**Current State:**
- GcsBufferedWriter exists
- May not use resumable upload API
- Would benefit large uploads (>100MB)

**Action Items:**
- [ ] Check if gcloud-storage SDK exposes resumable upload APIs
- [ ] Implement session token management
- [ ] Add retry logic with resume capability
- [ ] Test interruption/resume scenarios

**Blocker:** Complexity, SDK API availability

---

## 4. Python DataLoader Enhancements 🔄
**Priority:** MEDIUM  
**Effort:** Medium

### Background
Dataset trait uses `Vec<u8>` (not zero-copy) - noted in v0.9.1 docs.

### Options
1. **Breaking Change (v0.10.0):** Update Dataset trait to use `Bytes`
2. **Non-Breaking:** Add new trait methods returning Bytes
3. **Hybrid:** Deprecate Vec<u8> methods, add Bytes versions

**Impact:**
- DataLoader iterators would benefit from zero-copy
- Reduces memory allocations in training loops
- Better integration with framework zero-copy APIs

**Decision Needed:** When to tackle this (v0.9.2 or v0.10.0?)

---

## 5. Documentation Updates 📚
**Priority:** MEDIUM  
**Effort:** Low

### Immediate Updates
- [ ] Update GCS_TODO.md to reflect current status
- [ ] Mark completed items in STAGE3-DEFERRAL.md
- [ ] Create v0.9.2 planning document
- [ ] Update README with GCS examples

### API Documentation
- [ ] Add GCS examples to API guides
- [ ] Document GCS authentication (ADC)
- [ ] Add GCS to backend comparison table
- [ ] Update performance documentation

---

## Recommended Priority Order

### Sprint 1: GCS Validation (1-2 days)
1. ✅ Python API testing with GCS
2. ✅ Performance benchmarking
3. ✅ Error handling edge cases
4. ✅ Documentation updates

### Sprint 2: Range Engine (3-5 days)
1. 🔄 Review and refactor RangeEngine
2. 🔄 Integrate with File/DirectIO backends
3. 🔄 Integrate with Azure/GCS backends
4. 🔄 Testing and benchmarking
5. 🔄 Documentation

### Sprint 3: Polish (1-2 days)
1. 📝 Update all documentation
2. 📝 Create v0.9.2 or v0.10.0 planning
3. 📝 Decide on Dataset trait zero-copy strategy
4. 📝 Prepare release notes

---

## Questions to Resolve

1. **Version Number:** Should Stage 3 be v0.9.2 or v0.10.0?
   - v0.9.2: Pure performance enhancement, non-breaking
   - v0.10.0: If we also update Dataset trait (breaking)

2. **GCS Priority:** How critical is Python API validation?
   - If needed for production: Sprint 1 first
   - If exploratory: Can defer to Sprint 2

3. **Range Engine Scope:** Full implementation or MVP?
   - MVP: Just File backend first, prove concept
   - Full: All backends simultaneously

---

## Notes
- All Rust tests pass: 91/91 ✅
- All Python tests pass: 27/27 ✅
- Zero-copy tests pass: 6/6 ✅
- Range request tests pass: 4/4 ✅
- Build warnings: 0 ✅
- Current version: 0.9.1
