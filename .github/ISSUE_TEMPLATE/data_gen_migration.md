---
name: Data Generation Algorithm Migration Completion
about: Track the completion of migration from old data_gen to data_gen_alt
title: 'Complete migration to new data generation algorithm (data_gen_alt)'
labels: enhancement, performance, tech-debt
assignees: ''
---

## Background

In November 2025, we discovered and fixed a critical bug in the original data generation algorithm where `compress=1` (which should produce incompressible data) was actually producing 7.68:1 compression ratio due to cross-block pattern reuse.

**Root Cause**: The original algorithm used a shared `BASE_BLOCK` template across all unique blocks, allowing zstd to find patterns across block boundaries.

**Solution**: New algorithm (`data_gen_alt.rs`) generates each unique block with its own Xoshiro256++ RNG keystream, ensuring true incompressibility when `compress=1`. Compressibility is achieved via local back-references within each block only.

## Current Status

✅ **Completed**:
- New algorithm implemented in `src/data_gen_alt.rs`
- Performance optimized with Xoshiro256++ (5-10x faster than ChaCha20)
- All existing code redirected via `src/data_gen.rs` to use new algorithm
- All 162 library tests passing
- Compression bug fixed: compress=1 now produces ratio ~1.0000 (correct!)
- Old algorithm preserved as commented-out code for reference

## Remaining Work

### Phase 1: Extended Validation (Target: December 2025)

- [ ] Run production workloads for 1 week minimum
- [ ] Verify sai3-bench with various compress/dedup settings (1-6 for both)
- [ ] Verify dl-driver checkpoint save/load operations
- [ ] Performance benchmarking across different data sizes (1MB - 1GB)
- [ ] Compression ratio validation for compress=1,2,3,4,5,6
- [ ] Deduplication ratio validation for dedup=1,2,3,4,5,6

### Phase 2: Code Cleanup (After validation passes)

- [ ] Remove commented-out code from `src/data_gen.rs`:
  - `generate_controlled_data_original()` (lines ~206-250)
  - `ObjectGen::new_original()` (lines ~488-533)
  - `ObjectGen::*_original()` methods (lines ~594-710)
- [ ] Update inline documentation to reference only new algorithm
- [ ] Consider renaming `data_gen_alt.rs` → `data_gen.rs` (major refactor)
- [ ] Update external documentation and examples

### Phase 3: Optimization Opportunities

- [ ] Profile Xoshiro256++ performance across different platforms
- [ ] Evaluate alternative back-reference strategies for better compression
- [ ] Benchmark against industry-standard synthetic data generators
- [ ] Consider SIMD optimizations for block filling
- [ ] Evaluate streaming performance with different chunk sizes

## Testing Checklist

### Correctness Tests
- [x] compress=1 produces ratio ~1.0 (incompressible) ✓
- [x] compress=2,3,4 produce increasing compression ratios ✓
- [x] dedup=2 produces exactly 50% unique blocks ✓
- [x] Streaming generator matches single-pass output ✓
- [ ] Extended compress values (5,6,7,8) work correctly
- [ ] Extended dedup values (5,6,7,8) work correctly
- [ ] Large datasets (>1GB) generate correctly
- [ ] Concurrent generation is thread-safe

### Performance Tests
- [x] Basic performance: 1-7 GB/s ✓
- [ ] Performance stability over extended runs
- [ ] Memory usage within acceptable bounds
- [ ] No performance degradation with high compress/dedup values
- [ ] Streaming performance comparable to single-pass

### Integration Tests
- [ ] sai3-bench workloads complete successfully
- [ ] dl-driver data generation works correctly
- [ ] Python bindings work with new algorithm
- [ ] All downstream project tests pass

## Performance Baseline

Current performance (new algorithm with Xoshiro256++):
- **1MB dataset**: 954 MB/s (3.76x faster than old algorithm)
- **16MB dataset**: 2,816 MB/s (1.47x slower than old algorithm)
- **64MB dataset**: 7,351 MB/s (1.38x slower than old algorithm)
- **Streaming (16MB)**: 1,374 MB/s

Note: Old algorithm performance numbers are INVALID due to compression bug. Performance comparison should focus on correctness first, acceptable performance second.

## Documentation Updates Needed

- [ ] Update `README.md` with new algorithm details
- [ ] Update `docs/Changelog.md` with v0.9.17 entry
- [ ] Update `.github/copilot-instructions.md` (remove migration section)
- [ ] Add design document for new algorithm
- [ ] Update code examples in documentation

## Success Criteria

1. **Correctness**: compress=1 maintains ratio ~1.0 in all scenarios
2. **Performance**: Within 25% of old algorithm on large datasets (>16MB)
3. **Compatibility**: All downstream projects work without code changes
4. **Testing**: All tests pass including new extended validation suite
5. **Production**: No user-reported issues after 1 week of production use

## Related Files

- `src/data_gen.rs` - Redirection wrapper (to be simplified)
- `src/data_gen_alt.rs` - New algorithm implementation
- `tests/test_data_gen_alt.rs` - Comprehensive test suite
- `.github/copilot-instructions.md` - Migration tracking

## Timeline

- **November 2025**: Algorithm implemented and redirected ✓
- **December 2025**: Extended validation period
- **January 2026**: Code cleanup and optimization
- **February 2026**: Complete migration, remove old code

## Notes

The old algorithm's performance appeared better but was fundamentally broken. The new algorithm trades a small amount of performance (on large datasets) for correctness and proper compression control. This is the right trade-off for a production tool.
