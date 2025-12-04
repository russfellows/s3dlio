#!/usr/bin/env python3
"""
s3dlio Streaming Writer Example

Demonstrates how to use the streaming writer API to efficiently upload
large files in chunks. This is useful when:
- Data is generated incrementally (e.g., from a ML training loop)
- You don't want to buffer the entire file in memory
- You need compression during upload

The streaming writer supports:
- Buffered writes with automatic flushing
- Optional zstd compression
- Works with all backends (file://, s3://, az://, gs://, direct://)

Usage:
    python examples/python/streaming_writer.py
    python examples/python/streaming_writer.py file:///tmp/output/
"""

import sys
import tempfile
import s3dlio


def example_basic_streaming_write(base_uri: str):
    """Basic streaming write - write data in chunks."""
    print("\n1. Basic Streaming Write")
    print("-" * 40)
    
    output_uri = f"{base_uri}streamed_output.bin"
    print(f"   Writing to: {output_uri}")
    
    # Create a filesystem writer (works with file:// and direct://)
    # For S3, use create_s3_writer(); for Azure, use create_azure_writer()
    if base_uri.startswith("file://"):
        writer = s3dlio.create_filesystem_writer(output_uri, None)
    elif base_uri.startswith("direct://"):
        writer = s3dlio.create_direct_filesystem_writer(output_uri, None)
    elif base_uri.startswith("s3://"):
        writer = s3dlio.create_s3_writer(output_uri, None)
    elif base_uri.startswith("az://"):
        writer = s3dlio.create_azure_writer(output_uri, None)
    else:
        print(f"   Unsupported URI scheme: {base_uri}")
        return
    
    # Write data in chunks
    chunk_size = 4096
    num_chunks = 10
    
    for i in range(num_chunks):
        # Generate some data (in real use, this could come from anywhere)
        chunk = bytes([i % 256] * chunk_size)
        writer.write_chunk(chunk)
    
    # Finalize the writer (flushes buffers and completes upload)
    bytes_written, compressed_bytes = writer.finalize()
    
    print(f"   Wrote {num_chunks} chunks of {chunk_size} bytes")
    print(f"   Total bytes written: {bytes_written}")
    print(f"   Compressed bytes: {compressed_bytes}")
    
    # Verify by reading back
    data = s3dlio.get(output_uri)
    print(f"   Verified: read back {len(bytes(data))} bytes")
    
    # Cleanup
    s3dlio.delete(output_uri)
    print("   Cleaned up")


def example_compressed_streaming_write(base_uri: str):
    """Streaming write with zstd compression."""
    print("\n2. Compressed Streaming Write (zstd)")
    print("-" * 40)
    
    output_uri = f"{base_uri}compressed_output.bin.zst"
    print(f"   Writing to: {output_uri}")
    
    # Create writer options with compression
    options = s3dlio.PyWriterOptions()
    options.with_compression("zstd", level=3)  # zstd level 1-22, 3 is default
    options.with_buffer_size(64 * 1024)  # 64KB buffer
    
    # Create writer - for this example, we use filesystem
    # Note: Compression works with all backends
    if base_uri.startswith("file://"):
        writer = s3dlio.create_filesystem_writer(output_uri, options)
    elif base_uri.startswith("direct://"):
        writer = s3dlio.create_direct_filesystem_writer(output_uri, options)
    else:
        print(f"   Skipping (compression example uses file:// backend)")
        return
    
    # Write repetitive data (compresses well)
    chunk_size = 8192
    num_chunks = 100
    
    for i in range(num_chunks):
        # Repetitive data compresses very well
        chunk = b"Hello, this is test data that repeats! " * (chunk_size // 40)
        writer.write_chunk(chunk[:chunk_size])
    
    bytes_written, compressed_bytes = writer.finalize()
    
    uncompressed_size = chunk_size * num_chunks
    print(f"   Uncompressed size: {uncompressed_size} bytes")
    print(f"   Bytes written: {bytes_written}")
    
    # Cleanup
    s3dlio.delete(output_uri)
    print("   Cleaned up")


def example_incremental_data_generation(base_uri: str):
    """Simulate incremental data generation (like from a ML training loop)."""
    print("\n3. Incremental Data Generation (ML Simulation)")
    print("-" * 40)
    
    output_uri = f"{base_uri}training_log.bin"
    print(f"   Writing to: {output_uri}")
    
    if base_uri.startswith("file://"):
        writer = s3dlio.create_filesystem_writer(output_uri, None)
    elif base_uri.startswith("direct://"):
        writer = s3dlio.create_direct_filesystem_writer(output_uri, None)
    else:
        print(f"   Skipping (using file:// for this example)")
        return
    
    # Simulate a training loop that generates data incrementally
    import struct
    
    num_epochs = 5
    batches_per_epoch = 20
    
    print(f"   Simulating {num_epochs} epochs, {batches_per_epoch} batches each...")
    
    for epoch in range(num_epochs):
        for batch in range(batches_per_epoch):
            # Simulate metrics from training
            loss = 1.0 / (epoch * batches_per_epoch + batch + 1)
            accuracy = 1.0 - loss
            
            # Pack as binary data
            record = struct.pack('iiff', epoch, batch, loss, accuracy)
            writer.write_chunk(record)
    
    bytes_written, _ = writer.finalize()
    
    total_records = num_epochs * batches_per_epoch
    print(f"   Wrote {total_records} training records")
    print(f"   Total bytes: {bytes_written}")
    
    # Read back and verify
    data = bytes(s3dlio.get(output_uri))
    records_read = len(data) // struct.calcsize('iiff')
    print(f"   Verified: {records_read} records readable")
    
    # Cleanup
    s3dlio.delete(output_uri)
    print("   Cleaned up")


def main():
    # Determine base URI
    if len(sys.argv) > 1:
        base_uri = sys.argv[1]
        if not base_uri.endswith('/'):
            base_uri += '/'
    else:
        temp_dir = tempfile.mkdtemp(prefix="s3dlio_streaming_")
        base_uri = f"file://{temp_dir}/"
        print(f"Using temporary directory: {base_uri}")
    
    print("=" * 60)
    print("s3dlio Streaming Writer Example")
    print("=" * 60)
    print(f"Base URI: {base_uri}")

    try:
        example_basic_streaming_write(base_uri)
        example_compressed_streaming_write(base_uri)
        example_incremental_data_generation(base_uri)
        
        print("\n" + "=" * 60)
        print("✅ Streaming writer example completed!")
        print("=" * 60)
        
    finally:
        # Cleanup temp directory if we created one
        if "s3dlio_streaming_" in base_uri:
            import shutil
            temp_dir = base_uri.replace("file://", "").rstrip('/')
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
