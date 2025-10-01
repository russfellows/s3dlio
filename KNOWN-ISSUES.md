# Known Issues and Future Enhancements

This document tracks known limitations, bugs, and planned enhancements for s3dlio.

## Active Issues

### 1. Progress Bars Don't Update in Real-Time
**Priority:** Medium  
**Status:** Tracked in GitHub Issues  
**Discovered:** Pre-v0.8.8

Progress bars in CLI commands remain at 0% during operation execution and only update to 100% after the command completes. This defeats the purpose of progress indicators.

**Current Behavior:**
```bash
s3-cli download s3://bucket/large-file /local/dest
# Progress bar shows 0% for entire duration
# [████████████████████] 0%
# ... operation completes ...
# [████████████████████] 100% (instantly jumps to complete)
```

**Expected Behavior:**
Progress bars should update continuously during operation execution, showing:
- Current bytes transferred vs total
- Current throughput (MB/s, GB/s)
- Estimated time remaining (ETA)
- Real-time percentage completion

**Impact:** Users cannot monitor operation progress or estimate completion time for long-running transfers.

**Technical Notes:**
- Likely issue with progress callback not being invoked during streaming operations
- May be related to async/await boundaries or buffering
- Progress tracking code exists in `src/progress.rs`
- CLI uses `S3ProgressTracker` and `ProgressCallback` structs

**Related Files:**
- `src/progress.rs` - Progress tracking implementation
- `src/bin/cli.rs` - CLI progress bar integration

**Reference:** See GitHub Issues for additional discussion and proposed fixes.

---

### 2. Op-Log Facility Limited to S3 Backend
**Priority:** Medium  
**Status:** Documented, Not Started  
**Discovered:** v0.8.8 (October 2025)

The `--op-log` trace logging facility only records operations for S3-specific commands (`s3://` URIs). Generic storage commands using other backends (`file://`, `az://`, `direct://`) create the trace file but don't write operation records.

**Impact:** Users cannot profile or trace operations for non-S3 storage backends.

**Workaround:** Use S3-specific commands (`get`, `put`, `list`) instead of generic commands (`download`, `upload`, `ls`) when trace logging is needed.

**Documentation:**
- Detailed analysis: `docs/ISSUE-op-log-backend-support.md`
- GitHub issue format: `docs/GITHUB-ISSUE-op-log-backend-support.md`

**Proposed Solution:** Implement `LoggedObjectStore` wrapper pattern to instrument all backends uniformly.

---

## Future Enhancements

### Multi-Target S3 Addressing
**Priority:** Low  
**Status:** Design Phase

Currently, s3dlio handles >100 Gb/s throughput by running multiple process instances, each targeting different S3 endpoint IP addresses. Future enhancement would support multi-target addressing within a single s3dlio instance.

**Benefits:**
- Simplified deployment (single process)
- Better resource management
- Coordinated load balancing

**Technical Notes:**
- Most S3 deployments (MinIO, Vast) use bonded 100 Gb ports
- Multi-process approach works well currently
- Enhancement useful for cloud deployments with DNS round-robin

---

## Resolved Issues

### ✅ Logging Framework Migration
**Resolved in:** v0.8.8 (October 2025)

Migrated from `log` crate to `tracing` crate for compatibility with dl-driver and s3-bench projects.

**Changes:**
- Updated dependencies in Cargo.toml
- Converted ~99 log macro calls to tracing equivalents
- Added tracing-subscriber with env filter
- Added tracing-log bridge for backward compatibility

---

## Issue Submission Guidelines

When documenting a new issue:

1. **Create detailed analysis:** `docs/ISSUE-<short-name>.md`
2. **Create GitHub format:** `docs/GITHUB-ISSUE-<short-name>.md`
3. **Add to this tracking file** with priority and status
4. **Include:**
   - Clear description of current vs expected behavior
   - Root cause analysis
   - Proposed solution(s)
   - Impact assessment
   - Workarounds (if available)
   - Related files and code references

## Priority Definitions

- **Critical:** Blocks core functionality, data loss risk, security issue
- **High:** Major feature broken, significant performance regression
- **Medium:** Feature limitation, minor performance issue, usability problem
- **Low:** Enhancement, nice-to-have, cosmetic issue

## Status Definitions

- **Documented:** Issue identified and documented, not started
- **Design Phase:** Solution being designed, not implemented
- **In Progress:** Actively being worked on
- **Testing:** Implementation complete, under testing
- **Resolved:** Fixed and merged to main branch
- **Won't Fix:** Decided not to implement, documented for reference
