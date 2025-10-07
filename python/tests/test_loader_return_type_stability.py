"""
Test to verify PyBytesAsyncDataLoader return type stability.

This test ensures that the async data loader ALWAYS returns list[bytes],
regardless of batch_size. This provides a stable type contract for ML frameworks
like PyTorch, JAX, and TensorFlow.

Previously, batch_size=1 returned bytes, and batch_size>1 returned list[bytes].
Now it's consistently list[bytes] in all cases.
"""

import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from s3dlio import PyDataset, PyBytesAsyncDataLoader
except ImportError:
    print("s3dlio not installed. Run: ./build_pyo3.sh && ./install_pyo3_wheel.sh")
    sys.exit(1)


async def test_loader_return_type_batch_size_1():
    """Test that batch_size=1 returns list[bytes] with length 1."""
    print("Testing batch_size=1...")
    
    # Create a simple in-memory dataset
    dataset = PyDataset.from_bytes_list([b"item1", b"item2", b"item3"])
    
    # Create loader with batch_size=1
    loader = PyBytesAsyncDataLoader(dataset, {"batch_size": 1})
    
    items = []
    async for batch in loader:
        # Verify it's a list
        assert isinstance(batch, list), f"Expected list, got {type(batch)}"
        # Verify it has length 1
        assert len(batch) == 1, f"Expected length 1, got {len(batch)}"
        # Verify the item is bytes
        assert isinstance(batch[0], bytes), f"Expected bytes, got {type(batch[0])}"
        items.append(batch[0])
    
    # Verify we got all items
    assert len(items) == 3, f"Expected 3 items, got {len(items)}"
    assert items == [b"item1", b"item2", b"item3"]
    print("✓ batch_size=1 returns list[bytes] correctly")


async def test_loader_return_type_batch_size_2():
    """Test that batch_size=2 returns list[bytes] with length 2."""
    print("Testing batch_size=2...")
    
    dataset = PyDataset.from_bytes_list([b"item1", b"item2", b"item3", b"item4"])
    
    # Create loader with batch_size=2
    loader = PyBytesAsyncDataLoader(dataset, {"batch_size": 2})
    
    batches = []
    async for batch in loader:
        # Verify it's a list
        assert isinstance(batch, list), f"Expected list, got {type(batch)}"
        # Verify each item is bytes
        for item in batch:
            assert isinstance(item, bytes), f"Expected bytes, got {type(item)}"
        batches.append(batch)
    
    # Verify we got 2 batches
    assert len(batches) == 2, f"Expected 2 batches, got {len(batches)}"
    # First batch should have 2 items
    assert len(batches[0]) == 2, f"Expected batch length 2, got {len(batches[0])}"
    # Second batch should have 2 items
    assert len(batches[1]) == 2, f"Expected batch length 2, got {len(batches[1])}"
    
    print("✓ batch_size=2 returns list[bytes] correctly")


async def test_loader_return_type_stability():
    """Test that return type is stable across different batch sizes."""
    print("Testing return type stability across batch sizes...")
    
    dataset = PyDataset.from_bytes_list([b"a", b"b", b"c", b"d", b"e"])
    
    for batch_size in [1, 2, 3, 5]:
        loader = PyBytesAsyncDataLoader(dataset, {"batch_size": batch_size})
        
        async for batch in loader:
            # ALWAYS list, regardless of batch_size
            assert isinstance(batch, list), \
                f"batch_size={batch_size}: Expected list, got {type(batch)}"
            
            # All items should be bytes
            for item in batch:
                assert isinstance(item, bytes), \
                    f"batch_size={batch_size}: Expected bytes, got {type(item)}"
            
            # Length should be <= batch_size
            assert len(batch) <= batch_size, \
                f"batch_size={batch_size}: Batch length {len(batch)} exceeds batch_size"
    
    print("✓ Return type is stable across all batch sizes")


async def test_ml_framework_compatibility():
    """Test that the API is compatible with typical ML framework patterns."""
    print("Testing ML framework compatibility patterns...")
    
    dataset = PyDataset.from_bytes_list([b"data1", b"data2", b"data3"])
    loader = PyBytesAsyncDataLoader(dataset, {"batch_size": 1})
    
    # Pattern 1: Direct indexing (common in PyTorch)
    async for batch in loader:
        first_item = batch[0]  # Should always work
        assert isinstance(first_item, bytes)
    
    # Pattern 2: Iteration (common in TensorFlow)
    loader2 = PyBytesAsyncDataLoader(dataset, {"batch_size": 2})
    async for batch in loader2:
        for item in batch:  # Should always work
            assert isinstance(item, bytes)
    
    # Pattern 3: List operations (common in JAX)
    loader3 = PyBytesAsyncDataLoader(dataset, {"batch_size": 1})
    async for batch in loader3:
        batch_size = len(batch)  # Should always work
        assert batch_size >= 1
    
    print("✓ ML framework compatibility patterns work correctly")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("PyBytesAsyncDataLoader Return Type Stability Tests")
    print("=" * 60)
    
    try:
        await test_loader_return_type_batch_size_1()
        await test_loader_return_type_batch_size_2()
        await test_loader_return_type_stability()
        await test_ml_framework_compatibility()
        
        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print("\nReturn type is now stable: ALWAYS list[bytes]")
        print("This ensures compatibility with PyTorch, JAX, and TensorFlow")
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
