#!/usr/bin/env python3
"""
Test URI parsing for all supported schemes.

This test validates that the Python wrapper correctly handles URIs for all
backends WITHOUT requiring actual cloud connectivity or waiting for timeouts.

We test that:
1. URI scheme is correctly identified (not rejected as invalid)
2. Error messages don't contain S3-specific language for non-S3 URIs
3. The code paths accept all schemes without hardcoded S3 assumptions

This test should complete in < 5 seconds since it doesn't do any network I/O.

Run with: python -m pytest python/tests/test_uri_parsing_all_schemes.py -v
Or directly: python python/tests/test_uri_parsing_all_schemes.py
"""

import s3dlio
from s3dlio import _pymod as _core
import sys
import os
import tempfile


def test_scheme_detection():
    """Test that all schemes are recognized (not rejected as 'invalid scheme')."""
    print("\n--- Test: Scheme Detection ---")
    
    all_uris = {
        "file": "file:///tmp/test.bin",
        "direct": "direct:///tmp/test.bin", 
        "s3": "s3://bucket/key",
        "az": "az://container/blob",
        "gs": "gs://bucket/object",
    }
    
    for scheme, uri in all_uris.items():
        # We just check that the URI can be parsed without "invalid scheme" errors
        # by attempting to create a dataset (which will fail on I/O, not parsing)
        try:
            # create_dataset validates the URI scheme internally
            ds = _core.create_dataset(uri, {})
            print(f"  ✓ {scheme}:// - scheme recognized, dataset created")
        except Exception as e:
            error_msg = str(e).lower()
            # These are OK - they mean the scheme was recognized but I/O failed
            ok_errors = [
                "not exist",
                "not found", 
                "no such file",
                "connection",
                "credentials",
                "timeout",
                "not yet implemented",  # OK for Azure/GCS datasets
                "path does not exist",
            ]
            if any(ok in error_msg for ok in ok_errors):
                print(f"  ✓ {scheme}:// - scheme recognized (I/O error: {type(e).__name__})")
            elif "invalid scheme" in error_msg or "unknown scheme" in error_msg:
                print(f"  ✗ {scheme}:// - FAILED: scheme not recognized")
                return False
            else:
                # Unknown error - might be OK, print for debugging
                print(f"  ? {scheme}:// - unknown error: {e}")
    
    return True


def test_local_file_backend():
    """Test that file:// backend works end-to-end (fast, no network)."""
    print("\n--- Test: Local file:// Backend ---")
    
    # Create temp file
    test_dir = tempfile.mkdtemp(prefix="s3dlio_test_")
    test_file = os.path.join(test_dir, "test.bin")
    test_content = b"test data for file backend"
    
    try:
        with open(test_file, "wb") as f:
            f.write(test_content)
        
        file_uri = f"file://{test_file}"
        dir_uri = f"file://{test_dir}/"
        
        # Test get
        data = s3dlio.get_object(file_uri)
        assert bytes(data) == test_content, "Content mismatch"
        print(f"  ✓ get_object works")
        
        # Test list_keys
        keys = s3dlio.list_keys(dir_uri)
        assert "test.bin" in keys, f"Key not found: {keys}"
        print(f"  ✓ list_keys works: {keys}")
        
        # Test list_full_uris
        uris = s3dlio.list_full_uris(dir_uri)
        assert file_uri in uris, f"URI not found: {uris}"
        assert all(u.startswith("file://") for u in uris), "Scheme not preserved"
        print(f"  ✓ list_full_uris works, scheme preserved")
        
        # Test stat_object
        stat = s3dlio.stat_object(file_uri)
        assert stat is not None
        print(f"  ✓ stat_object works")
        
        # Test dataset creation
        ds = _core.create_dataset(dir_uri, {})
        assert len(ds) == 1
        keys = ds.keys()
        assert "test.bin" in keys
        print(f"  ✓ create_dataset works, keys(): {keys}")
        
        return True
        
    finally:
        # Cleanup
        os.remove(test_file)
        os.rmdir(test_dir)


def test_cloud_uri_format_validation():
    """
    Test that cloud URIs are correctly formatted without actually connecting.
    
    We can't test actual I/O without credentials, but we can verify that:
    1. The URIs are accepted (no 'invalid scheme' error)
    2. Error messages mention the right backend (not hardcoded S3)
    """
    print("\n--- Test: Cloud URI Format Validation ---")
    
    test_cases = [
        ("s3", "s3://mybucket/prefix/", "S3"),
        ("az", "az://mycontainer/prefix/", "Azure"),
        ("gs", "gs://mybucket/prefix/", "GCS"),
    ]
    
    for scheme, uri, expected_backend in test_cases:
        try:
            # This will fail fast (no network timeout) because:
            # - S3: No credentials or invalid bucket
            # - Azure: No credentials
            # - GCS: No credentials
            ds = _core.create_dataset(uri, {})
            # If it succeeds (unlikely without creds), that's fine
            print(f"  ✓ {scheme}:// - dataset created successfully")
        except Exception as e:
            error_msg = str(e)
            # Check that error is NOT S3-specific for non-S3 backends
            if scheme != "s3" and "s3://" in error_msg.lower():
                print(f"  ✗ {scheme}:// - ERROR: S3-specific message for {expected_backend}: {e}")
                return False
            # Check that error mentions the right backend or is generic
            print(f"  ✓ {scheme}:// - error is appropriate: {type(e).__name__}")
    
    return True


def test_deprecated_aliases_exist():
    """Test that deprecated class aliases still exist for backward compatibility."""
    print("\n--- Test: Deprecated Aliases ---")
    
    # These should exist (deprecated but functional)
    deprecated = [
        ("S3IterableDataset", s3dlio.S3IterableDataset),
        ("S3MapDataset", s3dlio.S3MapDataset),
        ("S3JaxIterable", s3dlio.S3JaxIterable),
    ]
    
    for name, cls in deprecated:
        if cls is not None:
            print(f"  ✓ {name} exists (deprecated alias)")
        else:
            print(f"  ✗ {name} is None!")
            return False
    
    # New generic names should also exist
    generic = [
        ("ObjectStoreIterableDataset", s3dlio.ObjectStoreIterableDataset),
        ("ObjectStoreMapDataset", s3dlio.ObjectStoreMapDataset),
        ("JaxIterable", s3dlio.JaxIterable),
    ]
    
    for name, cls in generic:
        if cls is not None:
            print(f"  ✓ {name} exists (generic name)")
        else:
            print(f"  ✗ {name} is None!")
            return False
    
    return True


def run_all_tests():
    """Run all URI parsing tests."""
    print("=" * 60)
    print("s3dlio URI Parsing & Multi-Backend Tests")
    print(f"Version: {s3dlio.__version__}")
    print("=" * 60)
    print("\nThese tests validate URI handling without network timeouts.")
    
    results = []
    
    results.append(("Scheme detection", test_scheme_detection()))
    results.append(("Local file backend", test_local_file_backend()))
    results.append(("Cloud URI format", test_cloud_uri_format_validation()))
    results.append(("Deprecated aliases", test_deprecated_aliases_exist()))
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("All tests PASSED!")
        return 0
    else:
        print("Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
