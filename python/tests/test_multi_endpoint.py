#!/usr/bin/env python3
"""
Multi-endpoint store tests for Python bindings.

Tests the multi-endpoint functionality including:
- Round-robin and least-connections load balancing
- Zero-copy buffer protocol
- Statistics tracking
- Error handling
"""

import os
import tempfile
from pathlib import Path

import s3dlio
import pytest


class TestMultiEndpointCreation:
    """Tests for creating multi-endpoint stores"""
    
    def test_create_from_uris(self, tmp_path):
        """Test creating store from URI list"""
        # Create test directories
        dirs = [tmp_path / f"endpoint{i}" for i in range(3)]
        for d in dirs:
            d.mkdir()
        
        # Create multi-endpoint store with file:// URIs
        uris = [f"file://{d}" for d in dirs]
        store = s3dlio.create_multi_endpoint_store(
            uris=uris,
            strategy="round_robin"
        )
        
        assert store is not None
        
    def test_create_from_template(self, tmp_path):
        """Test creating store from URI template with range expansion"""
        # Create test directories
        for i in range(1, 4):
            (tmp_path / f"endpoint{i}").mkdir()
        
        # Use template expansion: {1...3} -> endpoint1, endpoint2, endpoint3
        template = f"file://{tmp_path}/endpoint{{1...3}}"
        store = s3dlio.create_multi_endpoint_store_from_template(
            uri_template=template,
            strategy="round_robin"
        )
        
        assert store is not None
        
    def test_create_from_file(self, tmp_path):
        """Test creating store from configuration file"""
        # Create test directories
        dirs = [tmp_path / f"endpoint{i}" for i in range(3)]
        for d in dirs:
            d.mkdir()
        
        # Create config file with URIs
        config_file = tmp_path / "endpoints.txt"
        with open(config_file, 'w') as f:
            for d in dirs:
                f.write(f"file://{d}\n")
        
        store = s3dlio.create_multi_endpoint_store_from_file(
            file_path=str(config_file),
            strategy="least_connections"
        )
        
        assert store is not None
        
    def test_invalid_strategy(self, tmp_path):
        """Test that invalid strategy raises error"""
        dir1 = tmp_path / "endpoint1"
        dir1.mkdir()
        
        with pytest.raises(Exception):
            s3dlio.create_multi_endpoint_store(
                uris=[f"file://{dir1}"],
                strategy="invalid-strategy"
            )


class TestMultiEndpointOperations:
    """Tests for multi-endpoint CRUD operations"""
    
    @pytest.fixture
    def multi_store(self, tmp_path):
        """Create a multi-endpoint store for testing"""
        # Create 3 endpoint directories
        dirs = [tmp_path / f"endpoint{i}" for i in range(3)]
        for d in dirs:
            d.mkdir()
        
        uris = [f"file://{d}" for d in dirs]
        return s3dlio.create_multi_endpoint_store(
            uris=uris,
            strategy="round_robin"
        )
    
    @pytest.mark.asyncio
    async def test_put_and_get(self, multi_store, tmp_path):
        """Test basic put and get operations."""
        test_data = b"Hello from multi-endpoint store!"
        uri = f"file://{tmp_path}/endpoint0/test.txt"
        
        # Put data (async operation)
        await multi_store.put(uri, test_data)
        
        # Get data back (async operation)
        result = await multi_store.get(uri)
        assert bytes(result) == test_data
        
    @pytest.mark.asyncio
    async def test_get_range(self, multi_store, tmp_path):
        """Test range get operations."""
        test_data = b"0123456789" * 10  # 100 bytes
        uri = f"file://{tmp_path}/endpoint0/range_test.txt"
        
        await multi_store.put(uri, test_data)
        
        # Get a range (offset=10, length=10)
        result = await multi_store.get_range(uri, 10, 10)
        assert len(result) == 10
        assert bytes(result) == test_data[10:20]
        
    @pytest.mark.asyncio
    async def test_list_objects(self, multi_store, tmp_path):
        """Test listing objects"""
        # Create test files
        endpoint_dir = tmp_path / "endpoint0"
        for i in range(5):
            uri = f"file://{endpoint_dir}/file{i}.txt"
            await multi_store.put(uri, f"data{i}".encode())
        
        # List objects (async operation)
        prefix = f"file://{endpoint_dir}/"
        objects = await multi_store.list(prefix, recursive=False)
        
        assert isinstance(objects, list)
        assert len(objects) >= 5
        
    @pytest.mark.asyncio
    async def test_delete_object(self, multi_store, tmp_path):
        """Test deleting objects"""
        test_data = b"temporary data"
        uri = f"file://{tmp_path}/endpoint0/delete_me.txt"
        
        # Put and verify (async operations)
        await multi_store.put(uri, test_data)
        result = await multi_store.get(uri)
        assert bytes(result) == test_data
        
        # Delete (async operation)
        await multi_store.delete(uri)
        
        # Verify deleted (should raise error)
        with pytest.raises(Exception):
            await multi_store.get(uri)


class TestZeroCopyBehavior:
    """Tests to verify zero-copy data access"""
    
    @pytest.fixture
    def multi_store(self, tmp_path):
        """Create a multi-endpoint store for testing"""
        dir1 = tmp_path / "endpoint1"
        dir1.mkdir()
        
        return s3dlio.create_multi_endpoint_store(
            uris=[f"file://{dir1}"],
            strategy="round_robin"
        )
    
    @pytest.mark.asyncio
    async def test_memoryview_access(self, multi_store, tmp_path):
        """Test that returned data supports memoryview (zero-copy)"""
        # Create test data (1 MB)
        test_data = b"x" * (1024 * 1024)
        uri = f"file://{tmp_path}/endpoint1/large.bin"
        
        # Put data (async operation)
        await multi_store.put(uri, test_data)
        
        # Get data as BytesView (async operation)
        result = await multi_store.get(uri)
        
        # Should support buffer protocol (zero-copy access)
        mv = result.memoryview()
        assert len(mv) == len(test_data)
        assert bytes(mv) == test_data
        
    @pytest.mark.asyncio
    async def test_numpy_integration(self, multi_store, tmp_path):
        """Test zero-copy integration with numpy (if available)"""
        try:
            import numpy as np
        except ImportError:
            pytest.skip("numpy not available")
        
        # Create binary data
        original_array = np.arange(1000, dtype=np.float64)
        test_data = original_array.tobytes()
        uri = f"file://{tmp_path}/endpoint1/array.bin"
        
        # Store data (async operation)
        await multi_store.put(uri, test_data)
        
        # Retrieve and create numpy array from memoryview (zero-copy, async operation)
        result = await multi_store.get(uri)
        mv = result.memoryview()
        retrieved_array = np.frombuffer(mv, dtype=np.float64)
        
        assert np.array_equal(retrieved_array, original_array)
        
    @pytest.mark.asyncio
    async def test_large_object_performance(self, multi_store, tmp_path):
        """Test that large objects don't cause excessive memory copies"""
        # Create 10 MB test data
        size_mb = 10
        test_data = b"A" * (size_mb * 1024 * 1024)
        uri = f"file://{tmp_path}/endpoint1/large_perf.bin"
        
        # Put data (async operation)
        await multi_store.put(uri, test_data)
        
        # Get data - should be zero-copy via BytesView (async operation)
        result = await multi_store.get(uri)
        
        # Verify size without copying to bytes
        mv = result.memoryview()
        assert len(mv) == len(test_data)
        
        # Accessing via memoryview should not copy
        assert mv[0] == ord('A')
        assert mv[-1] == ord('A')


class TestLoadBalancing:
    """Tests for load balancing behavior"""
    
    @pytest.mark.asyncio
    async def test_round_robin_distribution(self, tmp_path):
        """Test that round-robin distributes requests evenly"""
        # Create 3 endpoints
        dirs = [tmp_path / f"endpoint{i}" for i in range(3)]
        for d in dirs:
            d.mkdir()
        
        uris = [f"file://{d}" for d in dirs]
        store = s3dlio.create_multi_endpoint_store(
            uris=uris,
            strategy="round_robin"
        )
        
        # Perform multiple operations (async)
        for i in range(9):
            uri = f"file://{dirs[i % 3]}/file{i}.txt"
            await store.put(uri, f"data{i}".encode())
        
        # Get statistics
        stats = store.get_endpoint_stats()
        assert isinstance(stats, list)
        assert len(stats) == 3
        
        # Each endpoint should have some requests
        for stat in stats:
            assert 'total_requests' in stat
            assert stat['total_requests'] > 0
            
    @pytest.mark.asyncio
    async def test_least_connections_strategy(self, tmp_path):
        """Test least-connections strategy"""
        # Create 2 endpoints
        dirs = [tmp_path / f"endpoint{i}" for i in range(2)]
        for d in dirs:
            d.mkdir()
        
        uris = [f"file://{d}" for d in dirs]
        store = s3dlio.create_multi_endpoint_store(
            uris=uris,
            strategy="least_connections"
        )
        
        # Perform operations (async)
        for i in range(4):
            uri = f"file://{dirs[0]}/file{i}.txt"
            await store.put(uri, f"data{i}".encode())
        
        stats = store.get_endpoint_stats()
        assert len(stats) == 2
        
    @pytest.mark.asyncio
    async def test_get_total_stats(self, tmp_path):
        """Test retrieving total statistics across all endpoints"""
        dir1 = tmp_path / "endpoint1"
        dir1.mkdir()
        
        store = s3dlio.create_multi_endpoint_store(
            uris=[f"file://{dir1}"],
            strategy="round_robin"
        )
        
        # Perform operations (async)
        uri = f"file://{dir1}/stats_test.txt"
        test_data = b"statistics test"
        await store.put(uri, test_data)
        result = await store.get(uri)
        
        # Get total stats
        total_stats = store.get_total_stats()
        assert isinstance(total_stats, dict)
        assert 'total_requests' in total_stats
        assert total_stats['total_requests'] >= 2  # At least put + get


class TestErrorHandling:
    """Tests for error handling"""
    
    def test_empty_uri_list(self):
        """Test that empty URI list raises error"""
        with pytest.raises(Exception):
            s3dlio.create_multi_endpoint_store(
                uris=[],
                strategy="round_robin"
            )
            
    def test_invalid_uri_scheme(self, tmp_path):
        """Test that invalid URI scheme raises error"""
        dir1 = tmp_path / "endpoint1"
        dir1.mkdir()
        
        # Try to create with mixed schemes (should fail validation)
        with pytest.raises(Exception):
            s3dlio.create_multi_endpoint_store(
                uris=[
                    f"file://{dir1}",
                    "s3://bucket/key",  # Mixed schemes not allowed
                ],
                strategy="round_robin"
            )
            
    def test_get_nonexistent_object(self, tmp_path):
        """Test that getting non-existent object raises error"""
        dir1 = tmp_path / "endpoint1"
        dir1.mkdir()
        
        store = s3dlio.create_multi_endpoint_store(
            uris=[f"file://{dir1}"],
            strategy="round_robin"
        )
        
        with pytest.raises(Exception):
            run_async(store.get(f"file://{dir1}/does_not_exist.txt"))
            
    def test_invalid_file_path(self):
        """Test that invalid config file path raises error"""
        with pytest.raises(Exception):
            s3dlio.create_multi_endpoint_store_from_file(
                file_path="/nonexistent/path/config.txt",
                strategy="round_robin"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
