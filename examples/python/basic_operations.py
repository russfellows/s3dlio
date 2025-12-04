#!/usr/bin/env python3
"""
s3dlio Basic Operations Example

Demonstrates the core storage operations available in s3dlio:
- put(): Create objects with generated data
- get(): Read entire objects
- get_range(): Read partial objects (byte ranges)
- list(): List objects under a prefix
- stat(): Get object metadata
- delete(): Remove objects

All operations work with ANY backend:
- file://   Local filesystem (buffered I/O)
- direct:// Local filesystem (O_DIRECT, bypasses page cache)
- s3://     Amazon S3 or S3-compatible (MinIO, Ceph, etc.)
- az://     Azure Blob Storage
- gs://     Google Cloud Storage

Usage:
    python examples/python/basic_operations.py
    python examples/python/basic_operations.py file:///tmp/mytest/
    python examples/python/basic_operations.py s3://mybucket/prefix/
"""

import sys
import tempfile
import s3dlio


def main():
    # Determine base URI from command line or use temp directory
    if len(sys.argv) > 1:
        base_uri = sys.argv[1]
        if not base_uri.endswith('/'):
            base_uri += '/'
    else:
        temp_dir = tempfile.mkdtemp(prefix="s3dlio_basic_")
        base_uri = f"file://{temp_dir}/"
        print(f"Using temporary directory: {base_uri}")
    
    print("=" * 60)
    print("s3dlio Basic Operations Example")
    print("=" * 60)
    print(f"Base URI: {base_uri}\n")

    # =========================================================================
    # PUT - Create objects
    # =========================================================================
    print("1. PUT - Creating objects...")
    
    # Create 5 objects with random data, 4KB each
    s3dlio.put(
        prefix=base_uri,
        num=5,
        template="data_{}.bin",
        size=4096,
        object_type="random",
        max_in_flight=4,
    )
    print(f"   Created 5 objects (4KB each)")

    # =========================================================================
    # LIST - List objects
    # =========================================================================
    print("\n2. LIST - Listing objects...")
    
    keys = s3dlio.list(base_uri, recursive=False)
    print(f"   Found {len(keys)} objects:")
    for key in keys[:5]:  # Show first 5
        print(f"     - {key}")
    if len(keys) > 5:
        print(f"     ... and {len(keys) - 5} more")

    # =========================================================================
    # STAT - Get object metadata
    # =========================================================================
    print("\n3. STAT - Getting object metadata...")
    
    if keys:
        first_key = keys[0]
        metadata = s3dlio.stat(first_key)
        print(f"   Object: {first_key}")
        print(f"   Size: {metadata['size']} bytes")
        print(f"   Last modified: {metadata.get('last_modified', 'N/A')}")

    # =========================================================================
    # GET - Read entire objects
    # =========================================================================
    print("\n4. GET - Reading objects...")
    
    total_bytes = 0
    for key in keys[:3]:
        data = s3dlio.get(key)
        total_bytes += len(bytes(data))
    print(f"   Read {total_bytes} bytes from 3 objects")

    # =========================================================================
    # GET_RANGE - Read partial objects
    # =========================================================================
    print("\n5. GET_RANGE - Reading byte ranges...")
    
    if keys:
        # Read first 1KB
        data = s3dlio.get_range(keys[0], offset=0, length=1024)
        print(f"   Read first 1024 bytes: {len(bytes(data))} bytes")
        
        # Read middle section
        data = s3dlio.get_range(keys[0], offset=1024, length=512)
        print(f"   Read bytes 1024-1536: {len(bytes(data))} bytes")

    # =========================================================================
    # GET_MANY - Parallel reads
    # =========================================================================
    print("\n6. GET_MANY - Parallel object reads...")
    
    results = s3dlio.get_many(keys, max_in_flight=4)
    print(f"   Read {len(results)} objects in parallel")
    total = sum(len(bytes(data)) for _, data in results)
    print(f"   Total bytes: {total}")

    # =========================================================================
    # DELETE - Remove objects
    # =========================================================================
    print("\n7. DELETE - Removing objects...")
    
    for key in keys:
        s3dlio.delete(key, recursive=False)
    print(f"   Deleted {len(keys)} objects")
    
    # Verify deletion
    remaining = s3dlio.list(base_uri, recursive=False)
    print(f"   Remaining objects: {len(remaining)}")

    print("\n" + "=" * 60)
    print("✅ Basic operations example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
