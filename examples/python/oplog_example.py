#!/usr/bin/env python3
"""
s3dlio Operation Log (Op-Log) Example

This example demonstrates how to create an I/O trace log (op-log) using s3dlio.
The op-log records all storage operations (GET, PUT, LIST, DELETE) with timing
information, which is useful for performance analysis, debugging, and replay.

The op-log is written as a compressed TSV file (using zstd) with the following columns:
    timestamp, operation, uri, offset, size, duration_us, status, first_byte_us, client_id

Supported backends:
    - s3://    - Amazon S3 or S3-compatible (MinIO, Ceph, etc.)
    - az://    - Azure Blob Storage
    - gs://    - Google Cloud Storage
    - file://  - Local filesystem (buffered)
    - direct://- Local filesystem (O_DIRECT bypass)

Usage:
    # Using local filesystem (no cloud credentials needed)
    python examples/oplog_example.py
    
    # Using S3 (requires AWS credentials)
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    python examples/oplog_example.py s3://mybucket/test-prefix/
    
    # Using Azure (requires Azure credentials)
    export AZURE_STORAGE_ACCOUNT=...
    export AZURE_STORAGE_KEY=...
    python examples/oplog_example.py az://mycontainer/test-prefix/
    
    # Using GCS (requires GCS credentials)
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
    python examples/oplog_example.py gs://mybucket/test-prefix/

After running, view the op-log with:
    zstd -dc /tmp/oplog_example.tsv.zst | less
    
Or analyze with Python:
    import pandas as pd
    df = pd.read_csv('/tmp/oplog_example.tsv.zst', sep='\\t', compression='zstd')
    print(df.groupby('op')['duration_us'].describe())
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path

# Import s3dlio - make sure you've built and installed the wheel
# Build: ./build_pyo3.sh && ./install_pyo3_wheel.sh
try:
    import s3dlio
except ImportError:
    print("ERROR: s3dlio module not found.")
    print("Build and install with: ./build_pyo3.sh && ./install_pyo3_wheel.sh")
    sys.exit(1)


def setup_test_data(base_uri: str, num_objects: int = 10, object_size: int = 4096) -> list[str]:
    """
    Create test objects for the demo.
    
    Args:
        base_uri: Base URI prefix (e.g., "file:///tmp/test/" or "s3://bucket/prefix/")
        num_objects: Number of test objects to create
        object_size: Size of each object in bytes
        
    Returns:
        List of created object URIs
    """
    print(f"\n📦 Creating {num_objects} test objects ({object_size} bytes each)...")
    
    # Use s3dlio.put() to create objects with random data
    # This works with ALL backends (S3, Azure, GCS, file, direct)
    s3dlio.put(
        prefix=base_uri,
        num=num_objects,
        template="testfile_{}.dat",
        size=object_size,
        object_type="random",  # Use random data
        max_in_flight=16,
    )
    
    # Return the URIs we just created
    return [f"{base_uri}testfile_{i}.dat" for i in range(num_objects)]


def demonstrate_operations(base_uri: str, object_uris: list[str]):
    """
    Perform various storage operations to generate op-log entries.
    
    All ObjectStore operations are logged via the LoggedObjectStore wrapper:
        - PUT operations
        - GET operations  
        - GET_RANGE operations
        - LIST operations
        - HEAD/stat operations
        - DELETE operations
        
    Args:
        base_uri: Base URI prefix for listing
        object_uris: List of object URIs to read
    """
    print("\n🔄 Performing storage operations (all logged via ObjectStore)...")
    
    # 1. LIST operations
    print("  📋 LIST operations...")
    for i in range(3):
        keys = s3dlio.list(base_uri, recursive=False)
        if i == 0:
            print(f"     Found {len(keys)} objects")
    
    # 2. STAT/HEAD operations
    print("  📊 STAT/HEAD operations...")
    for uri in object_uris[:3]:
        info = s3dlio.stat(uri)
        if uri == object_uris[0]:
            print(f"     First object size: {info['size']} bytes")
    
    # 3. GET operations (full object reads)
    print("  📥 GET operations...")
    total_bytes = 0
    for uri in object_uris[:5]:
        data = s3dlio.get(uri)
        total_bytes += len(bytes(data))
    print(f"     Read {total_bytes} bytes from 5 objects")
    
    # 4. GET_RANGE operations (partial reads)
    print("  📥 GET_RANGE operations...")
    for uri in object_uris[:3]:
        # Read first 1KB
        data = s3dlio.get_range(uri, offset=0, length=1024)
        # Read a middle chunk
        data = s3dlio.get_range(uri, offset=512, length=512)
    print(f"     Performed 6 partial range reads")
    
    # 5. DELETE operations (delete half the objects)
    print("  🗑️  DELETE operations...")
    delete_count = len(object_uris) // 2
    for uri in object_uris[:delete_count]:
        s3dlio.delete(uri, recursive=False)
    print(f"     Deleted {delete_count} objects")
    
    # 6. Final LIST to confirm deletions
    print("  📋 Final LIST (verify deletions)...")
    remaining = s3dlio.list(base_uri, recursive=True)
    print(f"     {len(remaining)} objects remaining")


def cleanup_test_data(base_uri: str):
    """
    Clean up any remaining test objects.
    
    Args:
        base_uri: Base URI prefix to delete
    """
    print("\n🧹 Cleaning up remaining test data...")
    try:
        # This delete operation is also logged!
        s3dlio.delete(base_uri, recursive=True)
        print("     Done")
    except Exception as e:
        print(f"     Warning: cleanup failed: {e}")


def analyze_oplog(oplog_path: str):
    """
    Display a summary of the op-log.
    
    Args:
        oplog_path: Path to the op-log file (.tsv.zst)
    """
    print(f"\n📊 Op-Log Analysis")
    print(f"   File: {oplog_path}")
    
    # Check if file exists
    if not os.path.exists(oplog_path):
        print("   ERROR: Op-log file not found!")
        return
    
    # Get file size
    file_size = os.path.getsize(oplog_path)
    print(f"   Size: {file_size:,} bytes (compressed)")
    
    # Decompress and analyze
    try:
        result = subprocess.run(
            ["zstd", "-dc", oplog_path],
            capture_output=True,
            text=True,
            check=True
        )
        lines = result.stdout.strip().split('\n')
        
        if len(lines) < 2:
            print("   No operations logged")
            return
        
        header = lines[0]
        header_fields = header.split('\t')
        operations = lines[1:]
        
        print(f"   Total operations: {len(operations)}")
        
        # Find the 'op' column index
        op_idx = header_fields.index('op') if 'op' in header_fields else 2
        
        # Count by operation type
        op_counts = {}
        for line in operations:
            fields = line.split('\t')
            if len(fields) > op_idx:
                op = fields[op_idx]
                op_counts[op] = op_counts.get(op, 0) + 1
        
        print("   Operation breakdown:")
        for op, count in sorted(op_counts.items()):
            print(f"     {op}: {count}")
        
        # Show header and first few entries
        print("\n   Op-log columns:")
        print(f"   {header}")
        print("\n   Sample entries (first 5):")
        for line in operations[:5]:
            # Truncate long fields for display
            fields = line.split('\t')
            display_fields = []
            for i, field in enumerate(fields):
                if len(field) > 40:
                    field = field[:37] + "..."
                display_fields.append(field)
            print(f"   {chr(9).join(display_fields)}")
        
        if len(operations) > 5:
            print(f"   ... and {len(operations) - 5} more entries")
            
    except FileNotFoundError:
        print("   Note: Install 'zstd' to analyze the log file")
        print(f"   View manually: zstd -dc {oplog_path} | head -20")
    except subprocess.CalledProcessError as e:
        print(f"   Error decompressing: {e}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("s3dlio Op-Log Example")
    print("=" * 60)
    
    # Determine base URI from command line or use local filesystem
    if len(sys.argv) > 1:
        base_uri = sys.argv[1]
        # Ensure URI ends with /
        if not base_uri.endswith('/'):
            base_uri += '/'
    else:
        # Default: use local filesystem in temp directory
        temp_dir = tempfile.mkdtemp(prefix="s3dlio_oplog_example_")
        base_uri = f"file://{temp_dir}/"
        print(f"\nℹ️  No URI specified, using local filesystem: {base_uri}")
        print("   Pass a URI as argument to use a different backend:")
        print("   python examples/oplog_example.py s3://mybucket/prefix/")
    
    # Op-log file path
    oplog_path = "/tmp/oplog_example.tsv.zst"
    
    print(f"\n📝 Op-Log Configuration:")
    print(f"   Output file: {oplog_path}")
    print(f"   Target URI:  {base_uri}")
    
    # =========================================================================
    # STEP 1: Initialize the op-logger
    # =========================================================================
    print("\n🔧 Initializing op-logger...")
    s3dlio.init_op_log(oplog_path)
    print(f"   Op-log active: {s3dlio.is_op_log_active()}")
    
    try:
        # =====================================================================
        # STEP 2: Perform storage operations (these will be logged)
        # =====================================================================
        
        # Create test data
        object_uris = setup_test_data(base_uri, num_objects=10, object_size=4096)
        
        # Perform various operations
        demonstrate_operations(base_uri, object_uris)
        
        # Cleanup test data (also logged!)
        cleanup_test_data(base_uri)
        
        # =====================================================================
        # STEP 3: Finalize the op-logger (IMPORTANT!)
        # =====================================================================
        # This flushes all buffered entries and closes the file.
        # Without this, you may lose data!
        print("\n🔒 Finalizing op-logger...")
        s3dlio.finalize_op_log()
        print(f"   Op-log active: {s3dlio.is_op_log_active()}")
        
        # =====================================================================
        # STEP 4: Analyze the results
        # =====================================================================
        analyze_oplog(oplog_path)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        # Still try to finalize to save partial log
        try:
            s3dlio.finalize_op_log()
        except:
            pass
        # Try to cleanup on error
        try:
            cleanup_test_data(base_uri)
        except:
            pass
        raise
    
    finally:
        # Clean up temp directory if we created one (data already deleted above)
        if base_uri.startswith("file://") and "s3dlio_oplog_example_" in base_uri:
            temp_dir = base_uri.replace("file://", "").rstrip('/')
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                pass
    
    print("\n" + "=" * 60)
    print("✅ Example completed successfully!")
    print("=" * 60)
    print(f"\nView the full op-log with:")
    print(f"  zstd -dc {oplog_path} | less")
    print(f"\nOr load in Python/Pandas:")
    print(f"  import pandas as pd")
    print(f"  df = pd.read_csv('{oplog_path}', sep='\\t', compression='zstd')")
    print(f"  print(df.groupby('op')['duration_us'].describe())")


if __name__ == "__main__":
    main()
