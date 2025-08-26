#!/usr/bin/env python3
"""
Verify that compression actually reduces file size
"""

import s3dlio
import tempfile
import os
from pathlib import Path

def test_compression_effectiveness():
    """Test that compression actually reduces file size"""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create highly compressible test data
        test_data = b"A" * 10000  # 10KB of repeated 'A' characters
        
        # Test without compression
        uri1 = f"file://{temp_dir}/no_compression"
        store1 = s3dlio.PyCheckpointStore(uri1)
        manifest1 = store1.save(100, 1, "test", test_data, None)
        
        # Test with high compression
        uri2 = f"file://{temp_dir}/with_compression"
        store2 = s3dlio.PyCheckpointStore(uri2, compression_level=9)
        manifest2 = store2.save(100, 1, "test", test_data, None)
        
        # Check file sizes (look for .zst extension for compressed files)
        uncompressed_dir = Path(temp_dir) / "no_compression"
        compressed_dir = Path(temp_dir) / "with_compression"
        
        print(f"Directory contents:")
        print(f"Uncompressed: {list(uncompressed_dir.rglob('*'))}")
        print(f"Compressed: {list(compressed_dir.rglob('*'))}")
        
        # Find the shard files (they are named rank-*.bin)
        uncompressed_files = list(uncompressed_dir.rglob("rank-*.bin"))
        compressed_files = list(compressed_dir.rglob("rank-*.bin*"))  # Include .zst extension
        
        if uncompressed_files and compressed_files:
            uncompressed_size = uncompressed_files[0].stat().st_size
            compressed_size = compressed_files[0].stat().st_size
            
            print(f"Uncompressed shard size: {uncompressed_size} bytes")
            print(f"Compressed shard size: {compressed_size} bytes")
            print(f"Compression ratio: {compressed_size/uncompressed_size:.2f}")
            
            if compressed_size < uncompressed_size:
                print("✓ Compression is working - compressed file is smaller!")
                return True
            else:
                print("✗ Compression not effective - files are same size")
                return False
        else:
            print("✗ Could not find shard files to compare")
            return False

if __name__ == "__main__":
    print("Testing compression effectiveness...")
    if test_compression_effectiveness():
        print("\n🎉 Compression verification successful!")
    else:
        print("\n❌ Compression verification failed!")
