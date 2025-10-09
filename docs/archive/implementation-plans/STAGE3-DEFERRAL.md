# v0.9.0 Implementation Status

## Stage Execution Order

**COMPLETED**:
- ✅ **Stage 1**: Python loader concurrent batching (commit 0994a1a)
- ✅ **Stage 2**: Zero-copy Bytes migration - BREAKING (commit d214dfc)
- ✅ **Stage 4**: Optional adaptive tuning (commit b4fd8b3)

**DEFERRED**:
- ⏸️ **Stage 3**: Backend-agnostic range engine → **v0.9.1**

## Why Stage 3 Was Deferred

**Stage 3 Scope**: Enable all backends (File/DirectIO/Azure/GCS) to use high-performance concurrent range GET operations (currently only S3 has this).

**Deferral Rationale**:
1. **Non-Breaking**: Stage 3 is an internal performance optimization with no API changes
2. **Scope Management**: v0.9.0 focuses on API stability + breaking changes
3. **Release Strategy**: Stage 3 can ship in v0.9.1 as pure performance enhancement
4. **User Impact**: Stage 3 benefits are transparent to users (no code changes required)

## v0.9.0 Focus

**API-Stable Beta with Breaking Changes**:
- Stage 1: User-facing performance (3-8x batch loading)
- Stage 2: **BREAKING** API change (Vec<u8> → Bytes)
- Stage 4: Optional user-facing feature (adaptive tuning)

## v0.9.1 Plan

**Performance Enhancement (Non-Breaking)**:
- Stage 3: Backend-agnostic range engine
- Expected: 30-50% throughput improvement for File/Azure/GCS large files
- Zero API changes, fully backward compatible
