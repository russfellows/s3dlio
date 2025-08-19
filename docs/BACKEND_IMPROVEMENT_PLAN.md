# s3dlio Backend Equivalence and Interface Improvement Plan

## Issues Identified

### 1. URI Scheme Inconsistency
- Need to support `direct://` for O_DIRECT file I/O vs `file://` for regular buffered I/O
- Current scheme detection doesn't differentiate between direct and buffered file access

### 2. Hard-coded Constants
- Multiple instances of hardcoded "4096" values in file_store_direct.rs
- Need centralized constants for maintainability

### 3. Backend Feature Parity
- All backends have basic ObjectStore trait implementation
- Container create/delete operations exist but need validation
- Azure multipart uploads use stream-based approach vs S3's explicit block management

### 4. Interface Rationalization
- Factory functions need to support new direct:// scheme
- URI parsing and scheme detection needs enhancement

## Implementation Plan

### Phase 1: Constants and Code Quality (Priority: High)
1. Create centralized constants module
2. Replace all hardcoded 4096 values with named constants
3. Add other common constants (buffer sizes, thresholds, etc.)

### Phase 2: URI Scheme Enhancement (Priority: High)
1. Add `Direct` variant to `Scheme` enum
2. Update `infer_scheme()` to recognize `direct://` URIs
3. Modify factory functions to route `direct://` to O_DIRECT implementation
4. Update all URI validation throughout codebase

### Phase 3: Backend Equivalence Validation (Priority: Medium)
1. Create comprehensive backend equivalence test suite
2. Validate container create/delete operations across all backends
3. Test multipart upload behavior consistency
4. Verify error handling consistency

### Phase 4: Azure Backend Enhancement (Priority: Medium)
1. Investigate explicit block management for Azure multipart uploads
2. Add configuration options for Azure upload behavior
3. Ensure Azure performance parity with S3

### Phase 5: Interface Rationalization (Priority: Low)
1. Unify configuration structs where possible
2. Standardize error mapping across backends
3. Add comprehensive documentation

## Files to Modify

### Core Files
- `src/object_store.rs` - URI scheme detection, factory functions
- `src/lib.rs` - Constants module export
- `src/file_store_direct.rs` - Replace hardcoded values

### New Files
- `src/constants.rs` - Centralized constants

### Test Files  
- `tests/test_backend_equivalence.rs` - Comprehensive backend testing

### Documentation
- Update all documentation to reflect new URI schemes
- Add backend equivalence documentation

## Success Criteria

1. ✅ All hardcoded values replaced with named constants
2. ✅ `direct://` URIs properly route to O_DIRECT implementation  
3. ✅ `file://` URIs route to regular buffered I/O
4. ✅ All backends pass equivalence tests
5. ✅ Container operations work across all backends
6. ✅ Multipart uploads consistent across backends
7. ✅ Error handling standardized
8. ✅ Performance characteristics documented

## Dependencies

- No new external dependencies required
- All changes use existing crate functionality
- Maintains backward compatibility where possible
