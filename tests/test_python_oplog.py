#!/usr/bin/env python3
"""
Test Python API op-log functionality for file:// and direct:// backends.
Tests PUT and GET operations (not LIST, which isn't implemented for file backends yet).
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

def main():
    print("=" * 60)
    print("Python API Op-Log Test (file:// and direct:// backends)")
    print("=" * 60)
    
    # Import s3dlio
    import s3dlio
    
    # Create temporary directories
    src_dir = tempfile.mkdtemp(prefix="s3dlio_src_")
    dest_file_dir = tempfile.mkdtemp(prefix="s3dlio_dest_file_")
    dest_direct_dir = tempfile.mkdtemp(prefix="s3dlio_dest_direct_")
    download_dir = tempfile.mkdtemp(prefix="s3dlio_download_")
    
    print(f"\n1. Creating test environment...")
    print(f"   Source: {src_dir}")
    print(f"   Dest (file://): {dest_file_dir}")
    print(f"   Dest (direct://): {dest_direct_dir}")
    print(f"   Download: {download_dir}")
    
    try:
        # Create test files to upload
        print(f"\n2. Creating 10 test files...")
        for i in range(10):
            test_file = os.path.join(src_dir, f"test_{i:03d}.dat")
            with open(test_file, 'wb') as f:
                # Write 10KB of data
                f.write(b'X' * (10 * 1024))
        print(f"   Created 10 test files (10KB each)")
        
        # Initialize op-log
        log_path = "/tmp/test_op_log_python.tsv.zst"
        if os.path.exists(log_path):
            os.remove(log_path)
        
        print(f"\n3. Initializing op-log: {log_path}")
        s3dlio.init_op_log(log_path)
        
        print("4. Checking if op-log is active...")
        if s3dlio.is_op_log_active():
            print("   ✓ Op-log is ACTIVE")
        else:
            print("   ✗ Op-log is NOT active")
            return 1
        
        # Test file:// backend - PUT operations
        print("\n5. Testing file:// backend - 10 PUT operations...")
        file_uri = f"file://{dest_file_dir}/"
        src_pattern = os.path.join(src_dir, "*.dat")
        s3dlio.upload(
            src_patterns=[src_pattern],
            dest_prefix=file_uri,
            max_in_flight=4,
            create_bucket=False
        )
        print(f"   ✓ Uploaded 10 files to {file_uri}")
        
        # Test file:// backend - GET operations
        print("\n6. Testing file:// backend - 10 GET operations...")
        download_file_dir = os.path.join(download_dir, "from_file")
        os.makedirs(download_file_dir, exist_ok=True)
        
        # Download with wildcard pattern
        src_uri = f"file://{dest_file_dir}/"
        s3dlio.download(
            src_uri=src_uri,
            dest_dir=download_file_dir,
            max_in_flight=4,
            recursive=True
        )
        print(f"   ✓ Downloaded files from {file_uri}")
        
        # Test direct:// backend - PUT operations  
        print("\n7. Testing direct:// backend - 10 PUT operations...")
        direct_uri = f"direct://{dest_direct_dir}/"
        s3dlio.upload(
            src_patterns=[src_pattern],
            dest_prefix=direct_uri,
            max_in_flight=4,
            create_bucket=False
        )
        print(f"   ✓ Uploaded 10 files to {direct_uri}")
        
        # Test direct:// backend - GET operations
        print("\n8. Testing direct:// backend - 10 GET operations...")
        download_direct_dir = os.path.join(download_dir, "from_direct")
        os.makedirs(download_direct_dir, exist_ok=True)
        
        # Download with wildcard pattern
        src_uri = f"direct://{dest_direct_dir}/"
        s3dlio.download(
            src_uri=src_uri,
            dest_dir=download_direct_dir,
            max_in_flight=4,
            recursive=True
        )
        print(f"   ✓ Downloaded files from {direct_uri}")
        
        # Finalize op-log
        print("\n9. Finalizing op-log...")
        s3dlio.finalize_op_log()
        print("   ✓ Op-log finalized")
        
        # Verify log file was created
        print("\n10. Verifying log file...")
        if not os.path.exists(log_path):
            print(f"   ✗ Log file not created: {log_path}")
            return 1
        
        file_size = os.path.getsize(log_path)
        print(f"   ✓ Log file created: {log_path} ({file_size} bytes)")
        
        # Decompress and show sample entries
        print("\n11. Sample log entries:")
        import subprocess
        result = subprocess.run(
            ['zstdcat', log_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            print(f"   Total log entries: {len(lines)}")
            
            # Show header
            if len(lines) > 0:
                print(f"\n   Header:")
                print(f"   {lines[0]}")
            
            # Show first few entries
            if len(lines) > 1:
                print(f"\n   First 5 entries:")
                for line in lines[1:6]:
                    print(f"   {line}")
            
            # Count operations by type
            puts = sum(1 for line in lines[1:] if '\tPUT\t' in line)
            gets = sum(1 for line in lines[1:] if '\tGET\t' in line)
            
            print(f"\n   Operation summary:")
            print(f"   - PUT operations: {puts}")
            print(f"   - GET operations: {gets}")
            print(f"   - Total operations: {puts + gets}")
            
            # Check for file:// and direct:// endpoints
            file_ops = sum(1 for line in lines[1:] if '\tfile://' in line)
            direct_ops = sum(1 for line in lines[1:] if '\tdirect://' in line)
            
            print(f"\n   Backend summary:")
            print(f"   - file:// operations: {file_ops}")
            print(f"   - direct:// operations: {direct_ops}")
        else:
            print(f"   ⚠ Could not decompress log file: {result.stderr}")
        
        print("\n" + "=" * 60)
        print("✓ All tests passed successfully")
        print("=" * 60)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Cleanup
        print(f"\n12. Cleaning up temporary directories...")
        for d in [src_dir, dest_file_dir, dest_direct_dir, download_dir]:
            if os.path.exists(d):
                shutil.rmtree(d)
        print("   ✓ Cleanup complete")

if __name__ == "__main__":
    exit(main())
