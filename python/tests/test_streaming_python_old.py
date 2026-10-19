#!/usr/bin/env python3
"""
Python tests for s3dlio v0.7.9 streaming functionality.

This test suite verifies that the Python bindings for our streaming API work correctly
across all backends: Filesystem, Azure Blob Storage, and Direct I/O Filesystem.
"""

import os
import tempfile
import sys
import uuid
from pathlib import Path

import pytest

# Import our s3dlio module
import s3dlio

# Explicit opt-in for the live-backend tests: no default bucket, account or container.
S3_BUCKET = os.environ.get("S3DLIO_TEST_BUCKET", "").strip()
AZURE_ACCOUNT = os.environ.get("AZURE_BLOB_ACCOUNT", "").strip()
AZURE_CONTAINER = os.environ.get("S3DLIO_TEST_AZURE_CONTAINER", "").strip()


def test_writer_options():
    """Test PyWriterOptions creation and configuration."""
    print("🧪 Testing WriterOptions...")

    # Test basic creation
    options = s3dlio.PyWriterOptions()
    print("  ✅ Created basic WriterOptions")

    # Test compression setting
    options.with_compression("zstd", 5)
    print("  ✅ Set zstd compression with level 5")

    # Test buffer size setting
    options.with_buffer_size(8192)
    print("  ✅ Set buffer size to 8192")

    # Test invalid compression type (commented out for now - needs API update)
    # try:
    #     options.with_compression("invalid", None)
    #     print("  ❌ Should have failed with invalid compression type")
    #     return False
    # except RuntimeError as e:
    #     print(f"  ✅ Correctly rejected invalid compression: {e}")
    print("✅ WriterOptions tests passed!")


def test_filesystem_streaming():
    """Test filesystem streaming writer functionality."""
    print("🧪 Testing Filesystem streaming...")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_file = Path(tmp_dir) / "streaming_test.bin"
        test_uri = f"file://{test_file}"

        # Test basic streaming without compression
        print("  Testing basic streaming...")
        options = s3dlio.PyWriterOptions()
        writer = s3dlio.create_filesystem_writer(test_uri, options)

        # Write some test data
        chunk1 = b"Hello, " * 1000  # 7000 bytes
        chunk2 = b"World!" * 500  # 3000 bytes

        writer.write_owned_bytes(chunk1)
        writer.write_owned_bytes(chunk2)

        # Check bytes written
        bytes_written = writer.bytes_written()
        expected_bytes = len(chunk1) + len(chunk2)
        assert bytes_written == expected_bytes, (
            f"Expected {expected_bytes}, got {bytes_written}"
        )
        print(f"  ✅ Wrote {bytes_written} bytes correctly")

        # Finalize the writer
        stats = writer.finalize()
        print(f"  ✅ Finalized with stats: {stats}")

        # Verify file was created
        assert test_file.exists(), f"File should exist: {test_file}"
        file_size = test_file.stat().st_size
        assert file_size == expected_bytes, (
            f"File size mismatch: expected {expected_bytes}, got {file_size}"
        )
        print(f"  ✅ File created with correct size: {file_size} bytes")

    print("✅ Filesystem streaming tests passed!")


def test_filesystem_streaming_with_compression():
    """Test filesystem streaming with compression."""
    print("🧪 Testing Filesystem streaming with compression...")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_file = (
            Path(tmp_dir) / "streaming_compressed"
        )  # No .zst extension - it will be added
        test_uri = f"file://{test_file}"

        # Test streaming with compression
        print("  Testing streaming with zstd compression...")
        options = s3dlio.PyWriterOptions()
        options.with_compression("zstd", 3)
        writer = s3dlio.create_filesystem_writer(test_uri, options)

        # Write highly compressible data
        chunk1 = b"A" * 4096  # Highly compressible
        chunk2 = b"B" * 4096  # Highly compressible

        writer.write_chunk(chunk1)
        writer.write_chunk(chunk2)

        # Check bytes written (uncompressed size)
        bytes_written = writer.bytes_written()
        expected_bytes = len(chunk1) + len(chunk2)
        assert bytes_written == expected_bytes, (
            f"Expected {expected_bytes}, got {bytes_written}"
        )
        print(f"  ✅ Uncompressed bytes written: {bytes_written}")

        # Finalize the writer
        writer.finalize()

        # Check compressed bytes
        compressed_bytes = writer.compressed_bytes()
        print(f"  ✅ Compressed size: {compressed_bytes} bytes")
        assert compressed_bytes < bytes_written, (
            "Compressed size should be smaller than original"
        )
        compression_ratio = (1 - compressed_bytes / bytes_written) * 100
        print(f"  ✅ Compression ratio: {compression_ratio:.1f}%")

        # Verify compressed file was created (with .zst extension added)
        compressed_file = test_file.with_suffix(".zst")
        assert compressed_file.exists(), (
            f"Compressed file should exist: {compressed_file}"
        )
        actual_file_size = compressed_file.stat().st_size
        print(
            f"  ✅ Compressed file created: {compressed_file} ({actual_file_size} bytes)"
        )

    print("✅ Filesystem compression streaming tests passed!")


def test_write_owned_bytes():
    """Test the write_owned_bytes optimization."""
    print("🧪 Testing write_owned_bytes optimization...")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_file = Path(tmp_dir) / "owned_bytes_test.bin"
        test_uri = f"file://{test_file}"

        options = s3dlio.PyWriterOptions()
        writer = s3dlio.create_filesystem_writer(test_uri, options)

        # Test write_owned_bytes
        data = b"Zero-copy optimization test data!" * 100
        writer.write_owned_bytes(data)

        bytes_written = writer.bytes_written()
        assert bytes_written == len(data), f"Expected {len(data)}, got {bytes_written}"
        print(f"  ✅ write_owned_bytes wrote {bytes_written} bytes")

        writer.finalize()

        # Verify file was created correctly
        assert test_file.exists()
        file_size = test_file.stat().st_size
        assert file_size == len(data)
        print(f"  ✅ File created with correct size: {file_size} bytes")

    print("✅ write_owned_bytes tests passed!")


@pytest.mark.skipif(
    not S3_BUCKET,
    reason="live S3 test: set S3DLIO_TEST_BUCKET to a writable bucket to run",
)
def test_s3_streaming():
    """Test S3 streaming writer against an explicitly opted-in bucket."""
    print("🧪 Testing S3 streaming...")

    test_uri = (
        f"s3://{S3_BUCKET}/s3dlio-pytest-{uuid.uuid4().hex[:12]}/streaming-test.bin"
    )
    options = s3dlio.PyWriterOptions()
    data = b"Python S3 streaming test data!" * 50
    try:
        writer = s3dlio.create_s3_writer(test_uri, options)
        writer.write_chunk(data)

        bytes_written = writer.bytes_written()
        assert bytes_written == len(data)
        print(f"  ✅ S3 streaming wrote {bytes_written} bytes")

        writer.finalize()
        print("  ✅ S3 streaming finalized successfully")

        # The object must really be in the bucket, with the bytes we wrote.
        assert s3dlio.stat(test_uri)["size"] == len(data)
        assert bytes(s3dlio.get(test_uri)) == data
    finally:
        try:
            s3dlio.delete(test_uri)
        except Exception as exc:
            print(f"cleanup warning for {test_uri}: {exc}")

    print("✅ S3 streaming tests passed!")


@pytest.mark.skipif(
    not (AZURE_ACCOUNT and AZURE_CONTAINER),
    reason="live Azure test: set AZURE_BLOB_ACCOUNT and S3DLIO_TEST_AZURE_CONTAINER to run",
)
def test_azure_streaming():
    """Test Azure streaming writer against an explicitly opted-in container."""
    print("🧪 Testing Azure streaming...")

    blob = f"s3dlio-pytest-{uuid.uuid4().hex[:12]}/streaming-test.bin"
    test_uri = f"https://{AZURE_ACCOUNT}.blob.core.windows.net/{AZURE_CONTAINER}/{blob}"
    az_uri = f"az://{AZURE_ACCOUNT}/{AZURE_CONTAINER}/{blob}"
    options = s3dlio.PyWriterOptions()
    data = b"Python Azure streaming test data!" * 50
    try:
        writer = s3dlio.create_azure_writer(test_uri, options)
        writer.write_chunk(data)

        bytes_written = writer.bytes_written()
        assert bytes_written == len(data)
        print(f"  ✅ Azure streaming wrote {bytes_written} bytes")

        writer.finalize()
        print("  ✅ Azure streaming finalized successfully")

        # The blob must really exist, with the bytes we wrote.
        assert s3dlio.stat(az_uri)["size"] == len(data)
        assert bytes(s3dlio.get(az_uri)) == data
    finally:
        try:
            s3dlio.delete(az_uri)
        except Exception as exc:
            print(f"cleanup warning for {az_uri}: {exc}")

    print("✅ Azure streaming tests passed!")


def test_writer_checksum():
    """Test writer checksum functionality."""
    print("🧪 Testing writer checksum...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        test_file = Path(tmp_dir) / "checksum_test.bin"
        test_uri = f"file://{test_file}"

        options = s3dlio.PyWriterOptions()
        writer = s3dlio.create_filesystem_writer(test_uri, options)

        # Write some data
        data = b"Checksum test data"
        writer.write_chunk(data)

        # Get checksum before finalizing
        checksum = writer.checksum()
        print(f"  ✅ Checksum: {checksum}")
        assert checksum is not None, "Checksum should not be None"
        assert checksum.startswith("crc32c:"), "Checksum should be CRC32C format"

        writer.finalize()

    print("✅ Checksum tests passed!")


def test_error_handling():
    """Test error handling for writer operations."""
    print("🧪 Testing error handling...")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_file = Path(tmp_dir) / "error_test.bin"
        test_uri = f"file://{test_file}"

        options = s3dlio.PyWriterOptions()
        writer = s3dlio.create_filesystem_writer(test_uri, options)

        # Write some data and finalize
        writer.write_chunk(b"test")
        writer.finalize()

        # Try to write after finalization (should error)
        with pytest.raises(RuntimeError):
            writer.write_chunk(b"should fail")
        print("  ✅ Correctly rejected write after finalization")

        # Try to finalize again (should error)
        with pytest.raises(RuntimeError):
            writer.finalize()
        print("  ✅ Correctly rejected double finalization")

        # Try to get info after finalization (should now work with stored stats)
        bytes_written = writer.bytes_written()
        print(
            f"  ✅ Successfully got bytes_written after finalization: {bytes_written}"
        )

    print("✅ Error handling tests passed!")


def main():
    """Run all Python streaming tests (standalone-script mode)."""
    print("🚀 Starting Python Phase 2 Streaming API Tests")
    print("=" * 60)

    tests = [
        test_writer_options,
        test_filesystem_streaming,
        test_filesystem_streaming_with_compression,
        test_write_owned_bytes,
        test_writer_checksum,
        test_error_handling,
    ]
    # Live-backend tests run only when explicitly opted in via environment.
    if S3_BUCKET:
        tests.append(test_s3_streaming)
    if AZURE_ACCOUNT and AZURE_CONTAINER:
        tests.append(test_azure_streaming)

    passed = 0
    total = len(tests)
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {test.__name__}: {e}")
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
        print()  # Add spacing between tests

    # Summary
    print("=" * 60)
    print("🏁 TEST SUMMARY")

    if passed == total:
        print(f"🎉 ALL TESTS PASSED! ({passed}/{total})")
        print("✅ Python Phase 2 Streaming API is working correctly!")
        return True
    else:
        print(f"❌ SOME TESTS FAILED! ({passed}/{total} passed)")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
