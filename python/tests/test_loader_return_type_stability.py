"""
Test to verify PyBytesAsyncDataLoader return type stability.

This test ensures that the async data loader ALWAYS returns a list of byte buffers
(zero-copy ``BytesView`` objects; ``bytes(view)`` copies out), regardless of batch_size.
This provides a stable type contract for ML frameworks like PyTorch, JAX, and TensorFlow.

Previously, batch_size=1 returned bytes, and batch_size>1 returned a list of bytes.
Now it is consistently ``list[BytesView]`` in all cases.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

import pytest

# Test the INSTALLED package as-is (no sys.path manipulation): putting python/ on the path
# would shadow the wheel with the source tree and its stale native module.
from s3dlio import PyDataset, PyBytesAsyncDataLoader


def _make_dataset(root, payloads):
    """Write each payload to a file under ``root`` and return a PyDataset over them.

    There is no in-memory ``PyDataset`` constructor (``PyDataset.from_bytes_list`` was
    never implemented); ``PyDataset.from_uris`` over local ``file://`` objects exercises
    the same loader code path without any cloud endpoint. Zero-padded names keep the
    dataset order equal to the payload order.
    """
    uris = []
    for i, payload in enumerate(payloads):
        path = Path(root) / f"item{i:03d}.bin"
        path.write_bytes(payload)
        uris.append(f"file://{path}")
    return PyDataset.from_uris(uris)


def _as_bytes(item):
    """The loader yields zero-copy ``BytesView`` objects (buffer protocol), not ``bytes``.

    Changelog: loaders return ``BytesView``; ``bytes(view)`` is the documented copy-out.
    """
    assert not isinstance(item, (str, int, list)), (
        f"Expected a buffer, got {type(item)}"
    )
    assert type(item).__name__ == "BytesView", f"Expected BytesView, got {type(item)}"
    return bytes(item)


@pytest.mark.asyncio
async def test_loader_return_type_batch_size_1(tmp_path):
    """Test that batch_size=1 returns list[BytesView] with length 1."""
    print("Testing batch_size=1...")

    # Create a simple in-memory dataset
    dataset = _make_dataset(tmp_path, [b"item1", b"item2", b"item3"])

    # Create loader with batch_size=1
    loader = PyBytesAsyncDataLoader(dataset, {"batch_size": 1})

    items = []
    async for batch in loader:
        # Verify it's a list
        assert isinstance(batch, list), f"Expected list, got {type(batch)}"
        # Verify it has length 1
        assert len(batch) == 1, f"Expected length 1, got {len(batch)}"
        # Verify the item is a BytesView (copied out as bytes)
        items.append(_as_bytes(batch[0]))

    # Verify we got all items
    assert len(items) == 3, f"Expected 3 items, got {len(items)}"
    assert items == [b"item1", b"item2", b"item3"]
    print("✓ batch_size=1 returns list[BytesView] correctly")


@pytest.mark.asyncio
async def test_loader_return_type_batch_size_2(tmp_path):
    """Test that batch_size=2 returns list[BytesView] with length 2."""
    print("Testing batch_size=2...")

    dataset = _make_dataset(tmp_path, [b"item1", b"item2", b"item3", b"item4"])

    # Create loader with batch_size=2
    loader = PyBytesAsyncDataLoader(dataset, {"batch_size": 2})

    batches = []
    async for batch in loader:
        # Verify it's a list
        assert isinstance(batch, list), f"Expected list, got {type(batch)}"
        # Verify each item is a BytesView (copied out as bytes)
        batches.append([_as_bytes(item) for item in batch])

    # Verify we got 2 batches
    assert len(batches) == 2, f"Expected 2 batches, got {len(batches)}"
    # First batch should have 2 items
    assert len(batches[0]) == 2, f"Expected batch length 2, got {len(batches[0])}"
    # Second batch should have 2 items
    assert len(batches[1]) == 2, f"Expected batch length 2, got {len(batches[1])}"
    assert batches == [[b"item1", b"item2"], [b"item3", b"item4"]]

    print("✓ batch_size=2 returns list[BytesView] correctly")


@pytest.mark.asyncio
async def test_loader_return_type_stability(tmp_path):
    """Test that return type is stable across different batch sizes."""
    print("Testing return type stability across batch sizes...")

    dataset = _make_dataset(tmp_path, [b"a", b"b", b"c", b"d", b"e"])

    for batch_size in [1, 2, 3, 5]:
        loader = PyBytesAsyncDataLoader(dataset, {"batch_size": batch_size})
        n_batches = 0
        n_items = 0

        async for batch in loader:
            n_batches += 1
            n_items += len(batch)
            # ALWAYS list, regardless of batch_size
            assert isinstance(batch, list), (
                f"batch_size={batch_size}: Expected list, got {type(batch)}"
            )

            # All items should be BytesView
            for item in batch:
                _as_bytes(item)

            # Length should be <= batch_size
            assert len(batch) <= batch_size, (
                f"batch_size={batch_size}: Batch length {len(batch)} exceeds batch_size"
            )

        # A loader that yields nothing must not pass: expect every item, in ceil(5/bs) batches
        assert n_batches >= 1, f"batch_size={batch_size}: loader yielded no batches"
        assert n_items == 5, f"batch_size={batch_size}: expected 5 items, got {n_items}"
        assert n_batches == -(-5 // batch_size), (
            f"batch_size={batch_size}: unexpected batch count {n_batches}"
        )

    print("✓ Return type is stable across all batch sizes")


@pytest.mark.asyncio
async def test_ml_framework_compatibility(tmp_path):
    """Test that the API is compatible with typical ML framework patterns."""
    print("Testing ML framework compatibility patterns...")

    dataset = _make_dataset(tmp_path, [b"data1", b"data2", b"data3"])
    loader = PyBytesAsyncDataLoader(dataset, {"batch_size": 1})

    # Pattern 1: Direct indexing (common in PyTorch)
    seen = 0
    async for batch in loader:
        first_item = batch[0]  # Should always work
        _as_bytes(first_item)
        seen += 1
    assert seen == 3, f"Pattern 1: expected 3 batches, got {seen}"

    # Pattern 2: Iteration (common in TensorFlow)
    loader2 = PyBytesAsyncDataLoader(dataset, {"batch_size": 2})
    seen = 0
    async for batch in loader2:
        for item in batch:  # Should always work
            _as_bytes(item)
            seen += 1
    assert seen == 3, f"Pattern 2: expected 3 items, got {seen}"

    # Pattern 3: List operations (common in JAX)
    loader3 = PyBytesAsyncDataLoader(dataset, {"batch_size": 1})
    seen = 0
    async for batch in loader3:
        batch_size = len(batch)  # Should always work
        assert batch_size >= 1
        seen += 1
    assert seen == 3, f"Pattern 3: expected 3 batches, got {seen}"

    print("✓ ML framework compatibility patterns work correctly")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("PyBytesAsyncDataLoader Return Type Stability Tests")
    print("=" * 60)

    try:
        with tempfile.TemporaryDirectory() as tmp:
            for test in (
                test_loader_return_type_batch_size_1,
                test_loader_return_type_batch_size_2,
                test_loader_return_type_stability,
                test_ml_framework_compatibility,
            ):
                sub = Path(tmp) / test.__name__
                sub.mkdir()
                await test(sub)

        print("=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        print("\nReturn type is now stable: ALWAYS list[BytesView]")
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
