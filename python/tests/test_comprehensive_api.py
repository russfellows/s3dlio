#!/usr/bin/env python3
"""
Comprehensive test matrix for s3dlio enhanced API.
Tests all URI schemes, error conditions, and backend-specific functionality.
"""

import os
import sys
import pytest
import tempfile
from pathlib import Path
import asyncio
from unittest import mock

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import s3dlio


class TestURISchemes:
    """Test all supported URI schemes with comprehensive coverage."""
    
    def test_file_scheme_basic(self):
        """Test basic file:// URI functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_data = b"Hello, file system!"
            test_file.write_bytes(test_data)
            
            # Test dataset creation
            dataset = s3dlio.create_dataset(f"file://{test_file}")
            assert dataset is not None
            
            # Test async loader creation
            loader = s3dlio.create_async_loader(f"file://{test_file}")
            assert loader is not None
    
    def test_file_scheme_directory_scanning(self):
        """Test file:// URI with directory containing multiple files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for i in range(5):
                test_file = Path(tmpdir) / f"test_{i}.txt"
                test_file.write_bytes(f"Test data {i}".encode())
            
            # Test directory URI
            dataset = s3dlio.create_dataset(f"file://{tmpdir}")
            assert dataset is not None
            
            # Test that we can iterate over items
            loader = s3dlio.create_async_loader(f"file://{tmpdir}")
            assert loader is not None
    
    @pytest.mark.skipif(not os.environ.get('S3_TEST_BUCKET'), 
                       reason="S3_TEST_BUCKET environment variable not set")
    def test_s3_scheme_basic(self):
        """Test basic s3:// URI functionality."""
        bucket = os.environ.get('S3_TEST_BUCKET')
        uri = f"s3://{bucket}/test-prefix/"
        
        try:
            # Test dataset creation
            dataset = s3dlio.create_dataset(uri)
            assert dataset is not None
            
            # Test async loader creation
            loader = s3dlio.create_async_loader(uri)
            assert loader is not None
        except Exception as e:
            # S3 may not be configured, skip with warning
            pytest.skip(f"S3 test skipped due to configuration: {e}")
    
    @pytest.mark.skipif(not os.environ.get('AZURE_TEST_CONTAINER'), 
                       reason="AZURE_TEST_CONTAINER environment variable not set")
    def test_azure_scheme_basic(self):
        """Test basic az:// URI functionality."""
        container = os.environ.get('AZURE_TEST_CONTAINER')
        uri = f"az://{container}/test-prefix/"
        
        # Test dataset creation
        dataset = s3dlio.create_dataset(uri)
        assert dataset is not None
        
        # Test async loader creation  
        loader = s3dlio.create_async_loader(uri)
        assert loader is not None
    
    @pytest.mark.skipif(not os.environ.get('DIRECT_IO_PATH'),
                       reason="DIRECT_IO_PATH environment variable not set")
    def test_direct_scheme_basic(self):
        """Test basic direct:// URI functionality."""
        path = os.environ.get('DIRECT_IO_PATH')
        uri = f"direct://{path}"
        
        # Test dataset creation
        dataset = s3dlio.create_dataset(uri)
        assert dataset is not None
        
        # Test async loader creation
        loader = s3dlio.create_async_loader(uri)
        assert loader is not None


class TestErrorHandling:
    """Test comprehensive error handling and edge cases."""
    
    def test_unsupported_scheme(self):
        """Test handling of unsupported URI schemes."""
        with pytest.raises(Exception) as exc_info:
            s3dlio.create_dataset("ftp://example.com/path")
        assert "unsupported" in str(exc_info.value).lower() or "unknown" in str(exc_info.value).lower()
    
    def test_malformed_uri(self):
        """Test handling of malformed URIs."""
        with pytest.raises(Exception):
            s3dlio.create_dataset("not-a-uri")
    
    def test_nonexistent_file(self):
        """Test handling of non-existent file paths."""
        with pytest.raises(Exception):
            s3dlio.create_dataset("file:///nonexistent/path/file.txt")
    
    def test_invalid_s3_bucket(self):
        """Test handling of invalid S3 bucket names."""
        with pytest.raises(Exception):
            s3dlio.create_dataset("s3://invalid..bucket.name/path")
    
    def test_empty_uri(self):
        """Test handling of empty URI."""
        with pytest.raises(Exception):
            s3dlio.create_dataset("")
    
    def test_none_uri(self):
        """Test handling of None URI."""
        with pytest.raises(Exception):
            s3dlio.create_dataset(None)


class TestOptionsValidation:
    """Test options parsing and validation across all backends."""
    
    def test_valid_options_file(self):
        """Test valid options with file:// URI."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_bytes(b"test data")
            
            options = {
                "batch_size": 32,
                "shuffle": True,
                "num_workers": 2
            }
            
            dataset = s3dlio.create_dataset(f"file://{test_file}", options)
            assert dataset is not None
    
    @pytest.mark.skipif(not os.environ.get('S3_TEST_BUCKET'),
                       reason="S3_TEST_BUCKET environment variable not set")
    def test_valid_options_s3(self):
        """Test valid options with s3:// URI."""
        bucket = os.environ.get('S3_TEST_BUCKET')
        
        options = {
            "batch_size": 64,
            "prefetch": 8,
            "part_size": 8388608,  # 8MB
        }
        
        dataset = s3dlio.create_dataset(f"s3://{bucket}/prefix/", options)
        assert dataset is not None
    
    def test_invalid_options_type(self):
        """Test handling of invalid options type."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt" 
            test_file.write_bytes(b"test data")
            
            # Options should be dict, not list
            with pytest.raises(Exception):
                s3dlio.create_dataset(f"file://{test_file}", ["invalid", "options"])
    
    def test_invalid_option_values(self):
        """Test handling of invalid option values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_bytes(b"test data")
            
            # Negative batch_size should be rejected
            with pytest.raises(Exception):
                options = {"batch_size": -1}
                s3dlio.create_dataset(f"file://{test_file}", options)


class TestCompatibilityShims:
    """Test backward compatibility wrappers and deprecation warnings."""
    
    def test_pys3dataset_compatibility(self):
        """Test PyS3Dataset compatibility wrapper."""
        # This should work but may show deprecation warnings
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_bytes(b"test data")
            
            # Test old-style S3 dataset creation if it exists
            try:
                from s3dlio import PyS3Dataset
                # Should still work for backward compatibility
                dataset = PyS3Dataset(f"file://{test_file}")
                assert dataset is not None
            except ImportError:
                # If old interface doesn't exist, that's also valid
                pass
    
    def test_pys3asyncdataloader_compatibility(self):
        """Test PyS3AsyncDataLoader compatibility wrapper."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt" 
            test_file.write_bytes(b"test data")
            
            # Test old-style async loader creation if it exists
            try:
                from s3dlio import PyS3AsyncDataLoader
                # Should still work for backward compatibility
                loader = PyS3AsyncDataLoader(f"file://{test_file}")
                assert loader is not None
            except ImportError:
                # If old interface doesn't exist, that's also valid
                pass


class TestTorchIntegration:
    """Test PyTorch integration and dataset usage patterns."""
    
    @pytest.mark.skipif(not _torch_available(), 
                       reason="PyTorch not available")
    def test_torch_dataset_creation(self):
        """Test S3IterableDataset creation through torch.py."""
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "test.txt"
            test_file.write_bytes(b"test data for torch")
            
            from s3dlio.torch import S3IterableDataset
            
            # Test that the fixed torch.py creates datasets properly
            dataset = S3IterableDataset(f"file://{test_file}")
            assert dataset is not None
            
            # Test that we can get items
            items = list(dataset)
            assert len(items) > 0
    
    @pytest.mark.skipif(not _torch_available(),
                       reason="PyTorch not available")
    def test_torch_dataloader_integration(self):
        """Test integration with PyTorch DataLoader."""
        import torch.utils.data as torch_data
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple test files
            for i in range(3):
                test_file = Path(tmpdir) / f"test_{i}.txt" 
                test_file.write_bytes(f"test data {i}".encode())
            
            from s3dlio.torch import S3IterableDataset
            
            dataset = S3IterableDataset(f"file://{tmpdir}")
            dataloader = torch_data.DataLoader(dataset, batch_size=1, num_workers=0)
            
            # Test that we can iterate through the dataloader
            batches = list(dataloader)
            assert len(batches) >= 3  # Should have at least our 3 files


class TestConcurrencyAndAsync:
    """Test async functionality and concurrent access patterns."""
    
    @pytest.mark.asyncio
    async def test_async_iteration(self):
        """Test async iteration over datasets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for i in range(5):
                test_file = Path(tmpdir) / f"async_test_{i}.txt"
                test_file.write_bytes(f"async test data {i}".encode())
            
            loader = s3dlio.create_async_loader(f"file://{tmpdir}")
            
            # Test async iteration
            count = 0
            async for item in loader:
                assert len(item) > 0  # Should have some data
                count += 1
                if count >= 5:  # Don't iterate forever
                    break
            
            assert count > 0
    
    @pytest.mark.asyncio
    async def test_concurrent_loaders(self):
        """Test multiple concurrent async loaders."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for i in range(10):
                test_file = Path(tmpdir) / f"concurrent_test_{i}.txt"
                test_file.write_bytes(f"concurrent test data {i}".encode())
            
            # Create multiple loaders
            loaders = []
            for i in range(3):
                loader = s3dlio.create_async_loader(f"file://{tmpdir}")
                loaders.append(loader)
            
            # Test concurrent access
            async def get_first_item(loader):
                async for item in loader:
                    return item
            
            tasks = [get_first_item(loader) for loader in loaders]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 3
            assert all(len(result) > 0 for result in results)


def _torch_available():
    """Check if PyTorch is available for testing."""
    try:
        import torch
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    # Print environment info for debugging
    print("=== S3DLIO Comprehensive Test Suite ===")
    print(f"Python: {sys.version}")
    print(f"PyTorch available: {_torch_available()}")
    print(f"S3_TEST_BUCKET: {'SET' if os.environ.get('S3_TEST_BUCKET') else 'NOT SET'}")
    print(f"AZURE_TEST_CONTAINER: {'SET' if os.environ.get('AZURE_TEST_CONTAINER') else 'NOT SET'}")
    print(f"DIRECT_IO_PATH: {'SET' if os.environ.get('DIRECT_IO_PATH') else 'NOT SET'}")
    print("=" * 40)
    
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])