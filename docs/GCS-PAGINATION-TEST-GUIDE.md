# GCS Pagination Testing Guide

## Issue Fixed in v0.8.22
**Bug**: GCS list and delete operations were limited to 1000 objects  
**Cause**: Missing pagination loop in `GcsClient::list_objects()`  
**Fix**: Implemented pagination using GCS `next_page_token` response field

## Root Cause Analysis

The GCS API returns a maximum of 1000 objects per call by default. The previous implementation made a single API call and returned, missing any objects beyond the first page.

### Code Changes
**File**: `src/gcs_client.rs`  
**Method**: `GcsClient::list_objects()`

**Before** (single page):
```rust
let response = self.client.list_objects(&request).await?;
let result: Vec<String> = response.items.unwrap_or_default()...
```

**After** (paginated):
```rust
let mut all_objects = Vec::new();
let mut page_token: Option<String> = None;

loop {
    let mut request = ListObjectsRequest {
        page_token: page_token.clone(),
        ...
    };
    
    let response = self.client.list_objects(&request).await?;
    all_objects.extend(response.items...);
    
    if let Some(next_token) = response.next_page_token {
        page_token = Some(next_token);
    } else {
        break;
    }
}
```

## Impact

### Fixed Operations
1. **List operations** (`gs://bucket/prefix*`) - Now returns ALL matching objects
2. **Delete prefix** (`delete_prefix("gs://bucket/prefix")`) - Now deletes ALL matching objects
3. **CLI list command** - Now shows all objects in bucket/prefix
4. **CLI delete command** - Now deletes all objects under prefix

### No Impact
- **Get/Put operations** - Single object operations not affected
- **S3 operations** - Already had correct pagination
- **Azure operations** - Use different API, not affected

## Testing Strategy

### Prerequisites
- GCS bucket with >1000 objects (ideally >2000 for multi-page testing)
- Valid GCS credentials (GOOGLE_APPLICATION_CREDENTIALS or ADC)

### Test 1: List Operation with >1000 Objects

```bash
# Create test bucket with many objects
export TEST_BUCKET="gs://test-pagination-bucket"
export TEST_PREFIX="test-prefix/"

# Upload 1500+ test objects (using gsutil or script)
for i in {1..1500}; do
    echo "test data $i" | gsutil cp - "${TEST_BUCKET}${TEST_PREFIX}file-$(printf "%04d" $i).txt"
done

# Test list operation
./target/release/s3-cli list "${TEST_BUCKET}${TEST_PREFIX}" --recursive

# Expected: Should show all 1500 objects, not just 1000
# Verify: Count the output lines
./target/release/s3-cli list "${TEST_BUCKET}${TEST_PREFIX}" --recursive | wc -l
# Should be >= 1500
```

### Test 2: Delete Operation with >1000 Objects

```bash
# Verify objects exist
gsutil ls "${TEST_BUCKET}${TEST_PREFIX}" | wc -l
# Should show 1500

# Delete using s3dlio
./target/release/s3-cli delete "${TEST_BUCKET}${TEST_PREFIX}"

# Verify all deleted
gsutil ls "${TEST_BUCKET}${TEST_PREFIX}" | wc -l
# Should show 0 (or error if empty)
```

### Test 3: Python API List Test

```python
import s3dlio

# List all objects
store = s3dlio.api.store_for_uri("gs://test-pagination-bucket/test-prefix/")
objects = store.list("gs://test-pagination-bucket/test-prefix/", recursive=True)

print(f"Total objects: {len(objects)}")
# Should show 1500+, not capped at 1000

assert len(objects) >= 1500, f"Expected 1500+ objects, got {len(objects)}"
```

### Test 4: Edge Cases

```bash
# Test exactly 1000 objects (single page)
# Should work without regression

# Test 1001 objects (requires pagination)
# Should return all 1001

# Test 0 objects (empty prefix)
# Should return empty list without error

# Test non-recursive with >1000 prefixes
# Should handle delimiter + pagination
```

## Debugging Pagination

### Enable Debug Logging
```bash
export RUST_LOG=s3dlio=debug,gcs_client=debug

./target/release/s3-cli list gs://bucket/prefix/
```

### Expected Log Output
```
DEBUG GCS LIST: bucket=bucket, prefix=Some("prefix/"), recursive=true
DEBUG GCS LIST page request: page_token=None
DEBUG GCS LIST page received: 1000 objects
DEBUG GCS LIST continuing to next page
DEBUG GCS LIST page request: page_token="CAE..."
DEBUG GCS LIST page received: 500 objects
DEBUG GCS LIST no more pages, breaking
DEBUG GCS LIST success: 1500 total objects
```

### Verification Points
1. ✅ Multiple "page request" log entries for >1000 objects
2. ✅ "continuing to next page" message between pages
3. ✅ Final count matches actual object count
4. ✅ "no more pages, breaking" when done

## Comparison with S3

The S3 implementation already had correct pagination:

```rust
// S3: Uses continuation_token (src/s3_utils.rs:244)
loop {
    if let Some(token) = &cont {
        req_builder = req_builder.continuation_token(token);
    }
    
    let resp = req_builder.send().await?;
    // collect objects...
    
    if let Some(next) = resp.next_continuation_token() {
        cont = Some(next.to_string());
    } else {
        break;
    }
}
```

The GCS fix follows the same pattern:
- **S3**: `continuation_token` → `next_continuation_token`
- **GCS**: `page_token` → `next_page_token`

## Manual Verification (Without Test Setup)

If you cannot create 1000+ objects for testing, you can verify the code correctness by:

1. **Code Review**: Compare with S3 implementation (known working)
2. **API Documentation**: Verify against [GCS API docs](https://cloud.google.com/storage/docs/json_api/v1/objects/list)
3. **Crate Source**: Check gcloud-storage v1.1.1 source code for field names
4. **Build Test**: Ensure code compiles without warnings
5. **Small Scale Test**: Test with <1000 objects to ensure no regression

## API Documentation References

- **GCS Objects.list**: https://cloud.google.com/storage/docs/json_api/v1/objects/list
- **Page Token**: "A previously-returned page token representing part of the larger set of results"
- **Next Page Token**: "The continuation token... If this is the last page of results, then no continuation token is returned"
- **Max Results**: "The recommended upper value for maxResults is 1000 objects in a single response"

## Success Criteria

✅ **Build**: Code compiles without warnings  
✅ **Logic**: Pagination loop matches S3 pattern  
✅ **API**: Uses correct GCS field names (`page_token`, `next_page_token`)  
✅ **Completeness**: Loop continues until `next_page_token` is `None`  
✅ **Integration**: Both `list()` and `delete_prefix()` use the fixed method  

## Future Enhancements

Consider adding in future versions:
1. **Max results control**: Allow caller to set `max_results` for memory control
2. **Streaming API**: Return iterator/stream instead of Vec for very large result sets
3. **Progress callback**: Report pagination progress for long-running lists
4. **Parallel pagination**: Fetch multiple pages concurrently (if order doesn't matter)
