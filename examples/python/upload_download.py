#!/usr/bin/env python3
"""
s3dlio Upload and Download Example

Demonstrates how to upload local files to storage and download
objects back to local filesystem:
- upload(): Upload local files/directories to any backend
- download(): Download objects from any backend to local files

This is useful for:
- Backing up data to cloud storage
- Distributing datasets
- Moving data between storage systems

All operations work with ANY backend:
- file://   Local filesystem
- s3://     Amazon S3 or S3-compatible
- az://     Azure Blob Storage
- gs://     Google Cloud Storage

Usage:
    python examples/python/upload_download.py
    python examples/python/upload_download.py file:///tmp/storage/
"""

import os
import sys
import tempfile
import shutil
import s3dlio


def create_test_files(local_dir: str, num_files: int = 5, file_size: int = 4096):
    """Create test files for upload demonstration."""
    os.makedirs(local_dir, exist_ok=True)
    
    files_created = []
    for i in range(num_files):
        filepath = os.path.join(local_dir, f"testfile_{i:03d}.dat")
        with open(filepath, 'wb') as f:
            # Write some identifiable data
            header = f"File {i} - s3dlio test data\n".encode()
            padding = bytes([i % 256] * (file_size - len(header)))
            f.write(header + padding)
        files_created.append(filepath)
    
    return files_created


def example_upload_files(local_dir: str, dest_uri: str):
    """Upload local files to storage."""
    print("\n1. Upload Files")
    print("-" * 40)
    
    # Create test files
    print(f"   Creating test files in {local_dir}...")
    files = create_test_files(local_dir, num_files=5, file_size=4096)
    print(f"   Created {len(files)} files")
    
    # Upload to storage
    print(f"   Uploading to {dest_uri}...")
    s3dlio.upload(
        src_patterns=files,       # List of file paths
        dest_prefix=dest_uri,     # Destination URI prefix
        max_in_flight=4,          # Concurrent uploads
        create_bucket=False,      # Don't try to create bucket
    )
    
    # Verify upload
    keys = s3dlio.list(dest_uri, recursive=True)
    print(f"   Uploaded {len(keys)} files")
    for key in keys[:3]:
        print(f"     - {key}")
    if len(keys) > 3:
        print(f"     ... and {len(keys) - 3} more")
    
    return keys


def example_download_files(src_uri: str, local_dir: str):
    """Download objects from storage to local filesystem."""
    print("\n2. Download Files")
    print("-" * 40)
    
    print(f"   Downloading from {src_uri}...")
    print(f"   Destination: {local_dir}")
    
    os.makedirs(local_dir, exist_ok=True)
    
    s3dlio.download(
        src_uri=src_uri,          # Source URI prefix
        dest_dir=local_dir,       # Local destination directory
        max_in_flight=4,          # Concurrent downloads
        recursive=True,           # Include subdirectories
    )
    
    # Verify download
    downloaded = []
    for root, dirs, files in os.walk(local_dir):
        for f in files:
            downloaded.append(os.path.join(root, f))
    
    print(f"   Downloaded {len(downloaded)} files")
    for path in downloaded[:3]:
        size = os.path.getsize(path)
        print(f"     - {os.path.basename(path)} ({size} bytes)")
    if len(downloaded) > 3:
        print(f"     ... and {len(downloaded) - 3} more")
    
    return downloaded


def example_verify_roundtrip(original_dir: str, downloaded_dir: str):
    """Verify that downloaded files match originals."""
    print("\n3. Verify Round-Trip Integrity")
    print("-" * 40)
    
    original_files = sorted(os.listdir(original_dir))
    downloaded_files = sorted(os.listdir(downloaded_dir))
    
    if len(original_files) != len(downloaded_files):
        print(f"   ❌ File count mismatch: {len(original_files)} vs {len(downloaded_files)}")
        return False
    
    all_match = True
    for orig_name, dl_name in zip(original_files, downloaded_files):
        orig_path = os.path.join(original_dir, orig_name)
        dl_path = os.path.join(downloaded_dir, dl_name)
        
        with open(orig_path, 'rb') as f1, open(dl_path, 'rb') as f2:
            orig_data = f1.read()
            dl_data = f2.read()
            
            if orig_data == dl_data:
                print(f"   ✓ {orig_name} matches")
            else:
                print(f"   ❌ {orig_name} MISMATCH!")
                all_match = False
    
    if all_match:
        print(f"   ✅ All {len(original_files)} files verified!")
    
    return all_match


def main():
    # Determine storage URI
    if len(sys.argv) > 1:
        storage_uri = sys.argv[1]
        if not storage_uri.endswith('/'):
            storage_uri += '/'
    else:
        storage_temp = tempfile.mkdtemp(prefix="s3dlio_storage_")
        storage_uri = f"file://{storage_temp}/"
    
    # Create local temp directories
    upload_dir = tempfile.mkdtemp(prefix="s3dlio_upload_")
    download_dir = tempfile.mkdtemp(prefix="s3dlio_download_")
    
    print("=" * 60)
    print("s3dlio Upload/Download Example")
    print("=" * 60)
    print(f"Storage URI:   {storage_uri}")
    print(f"Upload from:   {upload_dir}")
    print(f"Download to:   {download_dir}")

    try:
        # Upload files
        keys = example_upload_files(upload_dir, storage_uri)
        
        # Download files
        downloaded = example_download_files(storage_uri, download_dir)
        
        # Verify integrity
        example_verify_roundtrip(upload_dir, download_dir)
        
        # Cleanup storage
        print("\n4. Cleanup")
        print("-" * 40)
        for key in keys:
            s3dlio.delete(key)
        print(f"   Deleted {len(keys)} objects from storage")
        
        print("\n" + "=" * 60)
        print("✅ Upload/download example completed!")
        print("=" * 60)
        
    finally:
        # Cleanup temp directories
        shutil.rmtree(upload_dir, ignore_errors=True)
        shutil.rmtree(download_dir, ignore_errors=True)
        if "s3dlio_storage_" in storage_uri:
            storage_temp = storage_uri.replace("file://", "").rstrip('/')
            shutil.rmtree(storage_temp, ignore_errors=True)


if __name__ == "__main__":
    main()
