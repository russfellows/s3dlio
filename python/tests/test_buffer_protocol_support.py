#!/usr/bin/env python3
"""
Test buffer protocol support in s3dlio.put_bytes() and s3dlio.put_bytes_async()

Verifies that s3dlio is as permissive as boto3/minio/s3torch in accepting
various buffer-like objects without type errors.

Design philosophy: Be liberal in what you accept, optimize based on what you get.
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path
import asyncio

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import s3dlio


class CustomBytesObject:
    """Custom object that implements __bytes__() but not buffer protocol."""
    
    def __init__(self, data):
        self._data = data
    
    def __bytes__(self):
        return self._data
    
    def __len__(self):
        return len(self._data)


class TestBufferProtocolSupport:
    """Test that put_bytes accepts all buffer-like objects."""
    
    def test_put_bytes_accepts_regular_bytes(self):
        """Test that regular bytes objects work (basic case)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test_bytes.bin"
            uri = f"file://{test_file}"
            
            data = b"Hello, bytes!"
            s3dlio.put_bytes(uri, data)
            
            # Verify write
            result = s3dlio.get(uri)
            assert bytes(result) == data
            print("✅ Regular bytes accepted")
    
    def test_put_bytes_accepts_bytearray(self):
        """Test that bytearray objects work (buffer protocol)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test_bytearray.bin"
            uri = f"file://{test_file}"
            
            data = bytearray(b"Hello, bytearray!")
            s3dlio.put_bytes(uri, data)
            
            # Verify write
            result = s3dlio.get(uri)
            assert bytes(result) == bytes(data)
            print("✅ bytearray accepted")
    
    def test_put_bytes_accepts_memoryview(self):
        """Test that memoryview objects work (buffer protocol)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test_memoryview.bin"
            uri = f"file://{test_file}"
            
            original = b"Hello, memoryview!"
            data = memoryview(original)
            s3dlio.put_bytes(uri, data)
            
            # Verify write
            result = s3dlio.get(uri)
            assert bytes(result) == original
            print("✅ memoryview accepted")
    
    def test_put_bytes_accepts_bytesview(self):
        """Test that s3dlio's own BytesView works (from get())."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "source.bin"
            file2 = Path(tmpdir) / "dest.bin"
            uri1 = f"file://{file1}"
            uri2 = f"file://{file2}"
            
            # Write original data
            original = b"Hello, BytesView!"
            s3dlio.put_bytes(uri1, original)
            
            # Get returns BytesView
            bytes_view = s3dlio.get(uri1)
            
            # Put should accept BytesView
            s3dlio.put_bytes(uri2, bytes_view)
            
            # Verify round-trip
            result = s3dlio.get(uri2)
            assert bytes(result) == original
            print("✅ s3dlio.BytesView accepted")
    
    def test_put_bytes_accepts_custom_bytes_object(self):
        """Test that any object with __bytes__() method works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test_custom.bin"
            uri = f"file://{test_file}"
            
            original = b"Hello, custom object!"
            custom_obj = CustomBytesObject(original)
            s3dlio.put_bytes(uri, custom_obj)
            
            # Verify write
            result = s3dlio.get(uri)
            assert bytes(result) == original
            print("✅ Custom object with __bytes__() accepted")
    
    @pytest.mark.asyncio
    async def test_put_bytes_async_accepts_regular_bytes(self):
        """Test async version with regular bytes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test_async_bytes.bin"
            uri = f"file://{test_file}"
            
            data = b"Hello, async bytes!"
            await s3dlio.put_bytes_async(uri, data)
            
            # Verify write
            result = s3dlio.get(uri)
            assert bytes(result) == data
            print("✅ Async: Regular bytes accepted")
    
    @pytest.mark.asyncio
    async def test_put_bytes_async_accepts_bytearray(self):
        """Test async version with bytearray."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test_async_bytearray.bin"
            uri = f"file://{test_file}"
            
            data = bytearray(b"Hello, async bytearray!")
            await s3dlio.put_bytes_async(uri, data)
            
            # Verify write
            result = s3dlio.get(uri)
            assert bytes(result) == bytes(data)
            print("✅ Async: bytearray accepted")
    
    @pytest.mark.asyncio
    async def test_put_bytes_async_accepts_memoryview(self):
        """Test async version with memoryview."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test_async_memoryview.bin"
            uri = f"file://{test_file}"
            
            original = b"Hello, async memoryview!"
            data = memoryview(original)
            await s3dlio.put_bytes_async(uri, data)
            
            # Verify write
            result = s3dlio.get(uri)
            assert bytes(result) == original
            print("✅ Async: memoryview accepted")
    
    @pytest.mark.asyncio
    async def test_put_bytes_async_accepts_bytesview(self):
        """Test async version with BytesView."""
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "async_source.bin"
            file2 = Path(tmpdir) / "async_dest.bin"
            uri1 = f"file://{file1}"
            uri2 = f"file://{file2}"
            
            # Write original data
            original = b"Hello, async BytesView!"
            await s3dlio.put_bytes_async(uri1, original)
            
            # Get returns BytesView
            bytes_view = s3dlio.get(uri1)
            
            # Async put should accept BytesView
            await s3dlio.put_bytes_async(uri2, bytes_view)
            
            # Verify round-trip
            result = s3dlio.get(uri2)
            assert bytes(result) == original
            print("✅ Async: s3dlio.BytesView accepted")
    
    @pytest.mark.asyncio
    async def test_put_bytes_async_accepts_custom_bytes_object(self):
        """Test async version with custom __bytes__() object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test_async_custom.bin"
            uri = f"file://{test_file}"
            
            original = b"Hello, async custom object!"
            custom_obj = CustomBytesObject(original)
            await s3dlio.put_bytes_async(uri, custom_obj)
            
            # Verify write
            result = s3dlio.get(uri)
            assert bytes(result) == original
            print("✅ Async: Custom object with __bytes__() accepted")
    
    def test_put_bytes_rejects_invalid_data(self):
        """Test that completely invalid data is rejected with clear error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test_invalid.bin"
            uri = f"file://{test_file}"
            
            # These should fail with TypeError
            invalid_objects = [
                42,  # int
                3.14,  # float
                "string",  # str (not bytes-like)
                ["list"],  # list
                {"dict": "value"},  # dict
            ]
            
            for obj in invalid_objects:
                with pytest.raises((TypeError, Exception)) as exc_info:
                    s3dlio.put_bytes(uri, obj)
                assert "bytes" in str(exc_info.value).lower() or "buffer" in str(exc_info.value).lower()
            
            print("✅ Invalid data properly rejected with clear error")


class TestDgenPyCompatibility:
    """Test compatibility with dgen_py.BytesView specifically."""
    
    def test_dgen_py_bytesview_acceptance(self):
        """Test that dgen_py.BytesView is accepted without type errors."""
        try:
            import dgen_py
        except ImportError:
            pytest.skip("dgen_py not installed")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test_dgen.bin"
            uri = f"file://{test_file}"
            
            # Generate data with dgen_py
            size = 16 * 1024  # 16 KB
            data = dgen_py.generate_buffer(size, dedup_ratio=1.0, compress_ratio=1.0)
            
            # This should NOT raise "data must be BytesView or bytes" error
            s3dlio.put_bytes(uri, data)
            
            # Verify write
            result = s3dlio.get(uri)
            assert len(result) == size
            print("✅ dgen_py.BytesView accepted (main test case)")
    
    @pytest.mark.asyncio
    async def test_dgen_py_bytesview_acceptance_async(self):
        """Test that dgen_py.BytesView is accepted in async version."""
        try:
            import dgen_py
        except ImportError:
            pytest.skip("dgen_py not installed")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test_dgen_async.bin"
            uri = f"file://{test_file}"
            
            # Generate data with dgen_py
            size = 16 * 1024  # 16 KB
            data = dgen_py.generate_buffer(size, dedup_ratio=1.0, compress_ratio=1.0)
            
            # This should NOT raise "data must be BytesView or bytes" error
            await s3dlio.put_bytes_async(uri, data)
            
            # Verify write
            result = s3dlio.get(uri)
            assert len(result) == size
            print("✅ Async: dgen_py.BytesView accepted (main test case)")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "-s"])
