#!/usr/bin/env python3
"""
Test v0.8.19 Python API fixes across all backends
Tests:
1. Universal list() function with all backends
2. Universal stat() function with all backends  
3. Regex pattern support in list()
"""

import s3dlio
import sys

def test_s3():
    """Test S3 backend"""
    print("=" * 60)
    print("S3 Backend Testing")
    print("=" * 60)
    
    # Test list
    print("\n[S3] list() - baseline")
    keys = s3dlio.list("s3://signal65-public/")
    print(f"  Found {len(keys)} objects")
    
    # Test list with pattern
    print("\n[S3] list() with pattern (.*\\.txt$)")
    keys = s3dlio.list("s3://signal65-public/", pattern=r'.*\.txt$')
    print(f"  Filtered to {len(keys)} .txt files")
    for key in keys:
        print(f"    - {key}")
    
    # Test stat
    print("\n[S3] stat() - hello.txt")
    stat = s3dlio.stat("s3://signal65-public/hello.txt")
    print(f"  Size: {stat['size']}")
    print(f"  ETag: {stat['etag']}")
    print(f"  Content-Type: {stat['content_type']}")
    
    print("\n✅ S3 tests PASSED\n")

def test_gcs():
    """Test GCS backend"""
    print("=" * 60)
    print("GCS Backend Testing")
    print("=" * 60)
    
    # Test list
    print("\n[GCS] list() - baseline")
    keys = s3dlio.list("gs://signal65-russ-b1/")
    print(f"  Found {len(keys)} objects")
    
    # Test list with pattern
    print("\n[GCS] list() with pattern (.*\\.txt$)")
    keys = s3dlio.list("gs://signal65-russ-b1/", pattern=r'.*\.txt$')
    print(f"  Filtered to {len(keys)} .txt files")
    for key in keys:
        print(f"    - {key}")
    
    # Test stat
    print("\n[GCS] stat() - test.txt")
    stat = s3dlio.stat("gs://signal65-russ-b1/test.txt")
    print(f"  Size: {stat['size']}")
    print(f"  ETag: {stat['etag']}")
    print(f"  Last Modified: {stat['last_modified']}")
    
    print("\n✅ GCS tests PASSED\n")

def test_azure():
    """Test Azure backend"""
    print("=" * 60)
    print("Azure Backend Testing")
    print("=" * 60)
    
    # Test list
    print("\n[Azure] list() - baseline")
    keys = s3dlio.list("az://egiazurestore1/s3dlio/")
    print(f"  Found {len(keys)} objects")
    
    # Test list with pattern
    print("\n[Azure] list() with pattern (.*\\.txt$)")
    keys = s3dlio.list("az://egiazurestore1/s3dlio/", pattern=r'.*\.txt$')
    print(f"  Filtered to {len(keys)} .txt files")
    for key in keys:
        print(f"    - {key}")
    
    # Test stat
    print("\n[Azure] stat() - test.txt")
    stat = s3dlio.stat("az://egiazurestore1/s3dlio/test.txt")
    print(f"  Size: {stat['size']}")
    print(f"  ETag: {stat['etag']}")
    print(f"  Last Modified: {stat['last_modified']}")
    
    print("\n✅ Azure tests PASSED\n")

def test_file():
    """Test File backend"""
    print("=" * 60)
    print("File Backend Testing")
    print("=" * 60)
    
    # Create test files
    import os
    import tempfile
    
    test_dir = "/tmp/s3dlio-python-test"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create some test files
    for i in range(3):
        with open(f"{test_dir}/file{i}.txt", "w") as f:
            f.write(f"Test file {i}")
    with open(f"{test_dir}/data.json", "w") as f:
        f.write('{"test": true}')
    
    # Test list
    print(f"\n[File] list() - {test_dir}")
    keys = s3dlio.list(f"file://{test_dir}/")
    print(f"  Found {len(keys)} objects")
    
    # Test list with pattern
    print("\n[File] list() with pattern (.*\\.txt$)")
    keys = s3dlio.list(f"file://{test_dir}/", pattern=r'.*\.txt$')
    print(f"  Filtered to {len(keys)} .txt files")
    for key in keys:
        print(f"    - {key}")
    
    # Test stat
    print(f"\n[File] stat() - {test_dir}/file0.txt")
    stat = s3dlio.stat(f"file://{test_dir}/file0.txt")
    print(f"  Size: {stat['size']}")
    print(f"  Storage Class: {stat['storage_class']}")
    
    print("\n✅ File tests PASSED\n")

def test_directio():
    """Test DirectIO backend"""
    print("=" * 60)
    print("DirectIO Backend Testing")
    print("=" * 60)
    
    test_dir = "/tmp/directio-test"
    
    # Test list
    print(f"\n[DirectIO] list() - {test_dir}")
    keys = s3dlio.list(f"direct://{test_dir}/")
    print(f"  Found {len(keys)} objects")
    
    # Test list with pattern
    print("\n[DirectIO] list() with pattern (.*\\.txt$)")
    keys = s3dlio.list(f"direct://{test_dir}/", pattern=r'.*\.txt$')
    print(f"  Filtered to {len(keys)} .txt files")
    for key in keys:
        print(f"    - {key}")
    
    # Test stat
    print(f"\n[DirectIO] stat() - {test_dir}/testfile1.txt")
    stat = s3dlio.stat(f"direct://{test_dir}/testfile1.txt")
    print(f"  Size: {stat['size']}")
    print(f"  Storage Class: {stat['storage_class']}")
    
    print("\n✅ DirectIO tests PASSED\n")

def main():
    """Run all backend tests"""
    print("\n" + "=" * 60)
    print("Python API v0.8.19 - Universal Backend Testing")
    print("=" * 60)
    print()
    
    failed = []
    
    # Test each backend
    for test_func, name in [
        (test_s3, "S3"),
        (test_gcs, "GCS"),
        (test_azure, "Azure"),
        (test_file, "File"),
        (test_directio, "DirectIO"),
    ]:
        try:
            test_func()
        except Exception as e:
            print(f"\n❌ {name} tests FAILED: {e}\n")
            import traceback
            traceback.print_exc()
            failed.append(name)
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if not failed:
        print("\n✅ ALL BACKEND TESTS PASSED!")
        print("\nResults:")
        print("  - S3: PASS")
        print("  - GCS: PASS")
        print("  - Azure: PASS")
        print("  - File: PASS")
        print("  - DirectIO: PASS")
        print("\n100% backend coverage achieved!")
        return 0
    else:
        print(f"\n❌ {len(failed)} backend(s) failed:")
        for name in failed:
            print(f"  - {name}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
