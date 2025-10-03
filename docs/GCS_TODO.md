# GCS Backend - TODO and Next Steps

**Last Updated:** October 2025  
**Current Version:** v0.8.18  
**Status:** Phase 2 Complete, Additional Work Pending

## Not Yet Completed

### 1. Python API Testing ⏳
**Priority:** HIGH  
**Effort:** Medium

The Rust GCS backend is complete, but Python bindings haven't been tested with GCS URIs.

**What needs testing:**
- [ ] `s3dlio.upload(['file.txt'], 'gs://bucket/prefix/')`
- [ ] `s3dlio.download('gs://bucket/*.txt', './local/')`
- [ ] `S3IterableDataset("gs://bucket/data/", ...)`
- [ ] PyTorch DataLoader integration with GCS
- [ ] TensorFlow dataset integration with GCS
- [ ] Error handling in Python layer

**How to test:**
```python
# Rebuild Python extension
./build_pyo3.sh && ./install_pyo3_wheel.sh

# Test basic upload/download
python -c "import s3dlio; s3dlio.upload(['test.txt'], 'gs://bucket/')"

# Test with PyTorch (if available)
from s3dlio import S3IterableDataset
dataset = S3IterableDataset("gs://bucket/data/", batch_size=32)
```

**Blockers:** None - Rust implementation complete

---

### 2. Performance Benchmarking ⏳
**Priority:** HIGH  
**Effort:** High

No performance testing has been done for GCS backend.

**What needs benchmarking:**
- [ ] Upload throughput (target: 2.5+ GB/s sustained)
- [ ] Download throughput (target: 5+ GB/s sustained)
- [ ] Comparison: GCS vs S3 vs Azure vs File vs DirectIO
- [ ] Multipart upload performance tuning
- [ ] Concurrent operation scaling
- [ ] Large file handling (>10GB files)

**How to benchmark:**
```bash
# Use existing performance test scripts
./scripts/run_backend_comparison.sh

# Add GCS-specific tests
./scripts/compare_backends_performance.sh \
  --backends s3,azure,gcs \
  --file-sizes 100MB,1GB,10GB \
  --operations upload,download,list
```

**Expected modifications:**
- Update `scripts/build_performance_variants.sh` to include GCS
- Add GCS endpoints to comparison scripts
- Create GCS-specific sustained performance tests

**Blockers:** Need GCS bucket with adequate bandwidth allocation

---

### 3. Range Reads via CLI ⏳
**Priority:** MEDIUM  
**Effort:** Low

The ObjectStore API supports range reads (`get_range()`), but the CLI doesn't expose this.

**Current gap:**
- API works: `store.get_range("gs://bucket/obj", 0, 1024).await`
- CLI missing: No `--offset` / `--length` flags

**What needs implementation:**
```bash
# Desired functionality:
s3-cli download gs://bucket/large-file.bin ./output.bin --offset 0 --length 1048576
```

**How to implement:**
1. Add `offset: Option<u64>` and `length: Option<u64>` to Download command struct
2. Update `download_cmd` to use `get_range()` when flags present
3. Update help text and documentation

**File to modify:** `src/bin/cli.rs` (Download command handler)

**Blockers:** None - straightforward enhancement

---

### 4. List Buckets Support ⏳
**Priority:** LOW  
**Effort:** Medium

GCS doesn't expose project-level bucket listing through the current SDK.

**Current limitation:**
- `s3-cli list-buckets` only works for S3
- GCS requires project ID for bucket listing
- gcloud-storage SDK may not support this operation

**Workaround:**
```bash
# Use gcloud CLI instead
gcloud storage buckets list
```

**Potential implementation:**
- Check if `gcloud-storage` SDK supports `list_buckets()` with project ID
- Add `--project` flag to CLI if needed
- Or document workaround and defer indefinitely

**Blockers:** SDK limitations, unclear if feasible

---

### 5. Error Scenario Testing ⏳
**Priority:** MEDIUM  
**Effort:** Medium

Need comprehensive error handling testing.

**Scenarios to test:**
- [ ] Network failures during upload/download
- [ ] Authentication token expiration
- [ ] Invalid bucket/object names
- [ ] Permission denied errors
- [ ] Quota exceeded scenarios
- [ ] Partial upload failures
- [ ] Concurrent write conflicts

**How to test:**
- Simulate network interruptions
- Test with expired credentials
- Test with read-only buckets
- Use invalid URIs and verify error messages

**Blockers:** None - good practice to add

---

### 6. Large File Stress Testing ⏳
**Priority:** MEDIUM  
**Effort:** Medium

Multipart logic exists but hasn't been stress-tested with very large files.

**What needs testing:**
- [ ] Files >1GB
- [ ] Files >10GB
- [ ] Files >100GB (if infrastructure supports)
- [ ] Multipart chunk size optimization
- [ ] Memory usage during large transfers
- [ ] Progress reporting accuracy

**How to test:**
```bash
# Create large test file
dd if=/dev/urandom of=test-10gb.bin bs=1M count=10240

# Upload to GCS
s3-cli upload test-10gb.bin gs://bucket/large-files/

# Monitor memory and throughput
```

**Blockers:** Infrastructure capacity (bandwidth, storage)

---

### 7. Resumable Upload Implementation ⏳
**Priority:** LOW  
**Effort:** High

GCS supports resumable uploads, but current implementation doesn't use them.

**Current state:**
- `GcsBufferedWriter` exists but may not use resumable upload API
- Would benefit large uploads (can resume after failure)

**What needs implementation:**
- Check if gcloud-storage SDK exposes resumable upload APIs
- Implement session token management
- Add retry logic with resume capability
- Test interruption/resume scenarios

**Blockers:** SDK API availability, complexity

---

### 8. Custom Metadata Support ⏳
**Priority:** LOW  
**Effort:** Low-Medium

GCS supports custom metadata, but we don't expose it.

**Desired functionality:**
```bash
s3-cli upload file.txt gs://bucket/obj --metadata "key1=value1,key2=value2"
```

**What needs implementation:**
- Add metadata fields to CLI
- Update `put()` method to accept metadata map
- Update `stat()` to return custom metadata
- Document metadata usage

**Blockers:** None - enhancement only

---

### 9. Signed URL Generation ⏳
**Priority:** LOW  
**Effort:** Medium

GCS supports signed URLs for temporary public access.

**Use case:**
- Share objects without making bucket public
- Time-limited access for external users

**What needs implementation:**
```bash
s3-cli generate-url gs://bucket/private-file.txt --expires 3600
# Output: https://storage.googleapis.com/...?signature=...
```

**Blockers:** SDK API research needed

---

### 10. Storage Class Selection ⏳
**Priority:** LOW  
**Effort:** Low

GCS supports multiple storage classes (Standard, Nearline, Coldline, Archive).

**Current state:**
- Buckets created with default storage class
- Objects inherit bucket storage class

**What needs implementation:**
- Add `--storage-class` flag to create-bucket
- Add `--storage-class` flag to upload (object-level override)
- Document cost implications

**Blockers:** None - straightforward

---

## Documentation Gaps

### Missing Documentation
- [ ] GCS authentication setup guide (ADC walkthrough)
- [ ] Troubleshooting guide for common GCS errors
- [ ] GCS-specific performance tuning guide
- [ ] Python API examples with GCS
- [ ] Comparison table: GCS vs S3 vs Azure features

### To Be Updated
- [ ] Main README: Add GCS to supported backends table
- [ ] API documentation: Add GCS examples
- [ ] Tutorial: Add GCS walkthrough

---

## Future Enhancements (v0.9.0+)

### Advanced Features
- [ ] **Bucket lifecycle policies** - Automatic object deletion/archival
- [ ] **Object versioning** - Track and retrieve object versions
- [ ] **Cross-region replication** - Multi-region data redundancy
- [ ] **Customer-managed encryption keys (CMEK)** - Enhanced security
- [ ] **Requester pays** - Charge data transfer to requester

### Performance Optimizations
- [ ] **Parallel chunk uploads** - Split large files across connections
- [ ] **Connection pooling tuning** - Optimize for GCS endpoints
- [ ] **Compression optimization** - Test gzip vs zstd for GCS
- [ ] **Read-ahead caching** - Prefetch for sequential reads

### Monitoring & Observability
- [ ] **GCS operation metrics** - Track latency, throughput, errors
- [ ] **Cost tracking** - Monitor storage and transfer costs
- [ ] **Operation logging** - Detailed GCS API call logs
- [ ] **Health checks** - Verify GCS connectivity and auth

---

## Priority Matrix

| Task | Priority | Effort | Blockers | Target Version |
|------|----------|--------|----------|----------------|
| Python API Testing | HIGH | Medium | None | v0.8.19 |
| Performance Benchmarking | HIGH | High | GCS infra | v0.8.19 |
| Range Reads via CLI | MEDIUM | Low | None | v0.8.19 |
| Error Scenario Testing | MEDIUM | Medium | None | v0.8.19 |
| Large File Testing | MEDIUM | Medium | Infra | v0.8.19 |
| List Buckets Support | LOW | Medium | SDK limits | v0.9.0 |
| Resumable Uploads | LOW | High | SDK API | v0.9.0 |
| Custom Metadata | LOW | Low-Med | None | v0.9.0 |
| Signed URLs | LOW | Medium | SDK research | v0.9.0 |
| Storage Class Selection | LOW | Low | None | v0.9.0 |

---

## Recommended Next Release (v0.8.19)

**Focus:** Testing and Performance

1. ✅ Python API testing with GCS
2. ✅ Performance benchmarking suite
3. ✅ Range reads CLI support
4. ✅ Error scenario coverage
5. ✅ Documentation updates

**Estimated effort:** 1-2 weeks

---

## Deferred to v0.9.0

**Focus:** Advanced Features

- Resumable uploads
- Signed URLs
- Storage class management
- Advanced monitoring

**Rationale:** Core functionality is complete and working. These are enhancements that can be added based on user demand.

---

## How to Contribute

If you want to tackle any of these items:

1. Check this TODO list for status
2. Create a feature branch: `git checkout -b feature/gcs-<item-name>`
3. Implement and test
4. Update this TODO with progress
5. Submit PR with documentation updates

---

## Questions & Discussions

For questions about GCS implementation:
- See `docs/GCS_Phase2_0-8-18.md` for implementation details
- See `docs/GCS-TESTING-SUMMARY.md` for test results
- Check `src/gcs_client.rs` and `src/object_store.rs` for code

---

**Last Review:** October 2025  
**Next Review:** After v0.8.19 release
