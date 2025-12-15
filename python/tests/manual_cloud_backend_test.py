#!/usr/bin/env python3
"""
Manual Integration Test for Cloud Backends

This script tests s3dlio's Python API against REAL cloud storage backends.
Run this when you have valid credentials configured for the backend you want to test.

PREREQUISITES:

For S3 (s3://):
    export AWS_ACCESS_KEY_ID="your-key"
    export AWS_SECRET_ACCESS_KEY="your-secret"
    export AWS_REGION="us-east-1"
    # OR use: export AWS_PROFILE="your-profile"
    # For MinIO/Ceph: export AWS_ENDPOINT_URL="http://localhost:9000"

For Azure Blob (az://):
    # Option 1: Azure CLI login
    az login
    
    # Option 2: Service Principal (set these env vars)
    export AZURE_TENANT_ID="your-tenant"
    export AZURE_CLIENT_ID="your-client"
    export AZURE_CLIENT_SECRET="your-secret"

For GCS (gs://):
    export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
    # For emulator: export STORAGE_EMULATOR_HOST="localhost:9023"

USAGE:
    # Test S3 backend
    python manual_cloud_backend_test.py s3://your-bucket/test-prefix/
    
    # Test Azure backend
    python manual_cloud_backend_test.py az://your-container/test-prefix/
    
    # Test GCS backend
    python manual_cloud_backend_test.py gs://your-bucket/test-prefix/
    
    # Test local filesystem (baseline)
    python manual_cloud_backend_test.py file:///tmp/s3dlio-test/

The script will:
1. Create test files in the specified location
2. Test all Python API functions
3. Clean up test files
4. Report results
"""

import s3dlio
from s3dlio import (
    ObjectStoreIterableDataset,
    ObjectStoreMapDataset,
    list_keys,
    list_full_uris,
    get_object,
    stat_object,
)
import sys
import os
import uuid
import time


def parse_uri(uri: str) -> tuple:
    """Parse URI into (scheme, bucket/container, path)."""
    if "://" not in uri:
        raise ValueError(f"Invalid URI: {uri} (missing scheme)")
    
    scheme, rest = uri.split("://", 1)
    
    if scheme in ("file", "direct"):
        return scheme, None, rest
    
    parts = rest.split("/", 1)
    bucket = parts[0]
    path = parts[1] if len(parts) > 1 else ""
    
    return scheme, bucket, path


def create_test_file_uri(base_uri: str, filename: str) -> str:
    """Create a full URI for a test file."""
    base = base_uri.rstrip("/")
    return f"{base}/{filename}"


def run_cloud_test(base_uri: str):
    """Run comprehensive tests against a cloud backend."""
    
    scheme, bucket, path = parse_uri(base_uri)
    test_id = uuid.uuid4().hex[:8]
    test_prefix = f"s3dlio-test-{test_id}"
    
    if scheme in ("file", "direct"):
        test_dir = os.path.join(path.rstrip("/"), test_prefix)
        os.makedirs(test_dir, exist_ok=True)
        test_base = f"{scheme}://{test_dir}/"
    else:
        base_path = path.rstrip("/")
        if base_path:
            test_base = f"{scheme}://{bucket}/{base_path}/{test_prefix}/"
        else:
            test_base = f"{scheme}://{bucket}/{test_prefix}/"
    
    print("=" * 70)
    print(f"s3dlio Cloud Backend Integration Test")
    print(f"Version: {s3dlio.__version__}")
    print("=" * 70)
    print(f"\nBackend: {scheme}://")
    print(f"Test location: {test_base}")
    print(f"Test ID: {test_id}")
    
    # Create test files
    test_files = {
        "small.bin": b"Hello from s3dlio!",
        "medium.bin": os.urandom(1024),  # 1 KB
        "large.bin": os.urandom(1024 * 100),  # 100 KB
    }
    
    created_uris = []
    
    print("\n--- Phase 1: Create Test Files ---")
    for filename, content in test_files.items():
        uri = create_test_file_uri(test_base, filename)
        try:
            s3dlio.put_bytes(uri, content)
            created_uris.append(uri)
            print(f"  ✓ Created: {filename} ({len(content)} bytes)")
        except Exception as e:
            print(f"  ✗ FAILED to create {filename}: {e}")
            cleanup(created_uris, scheme)
            return 1
    
    print("\n--- Phase 2: Test list_keys() ---")
    try:
        keys = list_keys(test_base)
        print(f"  Keys found: {keys}")
        expected_keys = set(test_files.keys())
        found_keys = set(keys)
        if expected_keys == found_keys:
            print(f"  ✓ All expected keys found")
        else:
            missing = expected_keys - found_keys
            extra = found_keys - expected_keys
            if missing:
                print(f"  ✗ Missing keys: {missing}")
            if extra:
                print(f"  ⚠ Extra keys: {extra}")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        cleanup(created_uris, scheme)
        return 1
    
    print("\n--- Phase 3: Test list_full_uris() ---")
    try:
        uris = list_full_uris(test_base)
        print(f"  URIs found: {uris}")
        # Verify scheme is preserved
        for uri in uris:
            if not uri.startswith(f"{scheme}://"):
                print(f"  ✗ Scheme not preserved: {uri}")
                cleanup(created_uris, scheme)
                return 1
        print(f"  ✓ All URIs have correct scheme ({scheme}://)")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        cleanup(created_uris, scheme)
        return 1
    
    print("\n--- Phase 4: Test get_object() ---")
    for filename, expected_content in test_files.items():
        uri = create_test_file_uri(test_base, filename)
        try:
            data = get_object(uri)
            if bytes(data) == expected_content:
                print(f"  ✓ {filename}: content matches")
            else:
                print(f"  ✗ {filename}: content MISMATCH!")
                cleanup(created_uris, scheme)
                return 1
        except Exception as e:
            print(f"  ✗ {filename} FAILED: {e}")
            cleanup(created_uris, scheme)
            return 1
    
    print("\n--- Phase 5: Test stat_object() ---")
    for filename, expected_content in test_files.items():
        uri = create_test_file_uri(test_base, filename)
        try:
            meta = stat_object(uri)
            print(f"  ✓ {filename}: size={meta.get('size', 'N/A')}, "
                  f"last_modified={meta.get('last_modified', 'N/A')}")
        except Exception as e:
            print(f"  ✗ {filename} FAILED: {e}")
            cleanup(created_uris, scheme)
            return 1
    
    print("\n--- Phase 6: Test ObjectStoreMapDataset ---")
    try:
        # ObjectStoreMapDataset takes a PREFIX, not a list of URIs
        dataset = ObjectStoreMapDataset.from_prefix(test_base)
        print(f"  Dataset length: {len(dataset)}")
        
        if len(dataset) != len(test_files):
            print(f"  ⚠ Expected {len(test_files)} items, got {len(dataset)}")
        
        # Read first item
        data = dataset[0]
        print(f"  ✓ dataset[0] returned {len(data)} bytes")
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        cleanup(created_uris, scheme)
        return 1
    
    print("\n--- Phase 7: Test ObjectStoreIterableDataset ---")
    try:
        # ObjectStoreIterableDataset takes a PREFIX, not a list of URIs
        dataset = ObjectStoreIterableDataset.from_prefix(test_base)
        count = 0
        total_bytes = 0
        for data in dataset:
            count += 1
            total_bytes += len(data)
        print(f"  ✓ Iterated {count} items, {total_bytes} total bytes")
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        cleanup(created_uris, scheme)
        return 1
    
    # Cleanup
    print("\n--- Phase 8: Cleanup ---")
    cleanup(created_uris, scheme)
    
    print("\n" + "=" * 70)
    print(f"ALL TESTS PASSED for {scheme}:// backend!")
    print("=" * 70)
    return 0


def cleanup(uris: list, scheme: str):
    """Clean up test files."""
    for uri in uris:
        try:
            s3dlio.delete(uri)
            print(f"  Deleted: {uri}")
        except Exception as e:
            print(f"  ⚠ Could not delete {uri}: {e}")
    
    # For local filesystem, try to remove the directory
    if scheme in ("file", "direct"):
        try:
            # Extract directory from first URI
            if uris:
                first_uri = uris[0]
                _, _, path = parse_uri(first_uri)
                test_dir = os.path.dirname(path)
                if os.path.isdir(test_dir) and "s3dlio-test" in test_dir:
                    os.rmdir(test_dir)
                    print(f"  Removed directory: {test_dir}")
        except Exception as e:
            print(f"  ⚠ Could not remove test directory: {e}")


def print_usage():
    """Print usage information."""
    print(__doc__)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)
    
    base_uri = sys.argv[1]
    
    if not any(base_uri.startswith(s) for s in ["file://", "direct://", "s3://", "az://", "gs://"]):
        print(f"ERROR: Unsupported URI scheme in: {base_uri}")
        print("Supported schemes: file://, direct://, s3://, az://, gs://")
        sys.exit(1)
    
    try:
        sys.exit(run_cloud_test(base_uri))
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
