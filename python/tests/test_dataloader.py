# test_dataloader.py

import asyncio

import pytest
from s3dlio import PyVecDataset, PyAsyncDataLoader


def test_sync_batches():
    print("=== Sync DataLoader ===")
    # Create a small in-memory dataset
    ds = PyVecDataset([10, 20, 30, 40, 50])

    # Use sync loader under the hood (really returns an async stream, but you can collect synchronously
    # via async.run, or just reuse the async example below). For pure sync testing you can do:
    async def collect():
        loader = PyAsyncDataLoader(
            ds, {"batch_size": 2, "drop_last": False, "shuffle": False}
        )
        batches = []
        async for batch in loader:
            batches.append(batch)
        return batches

    batches = asyncio.run(collect())
    print("Batches:", batches)
    assert [list(b) for b in batches] == [[10, 20], [30, 40], [50]]


@pytest.mark.asyncio
async def test_async_batches():
    print("\n=== Async DataLoader ===")
    ds = PyVecDataset(list(range(1, 11)))
    # Try with shuffle on
    opts = {"batch_size": 3, "shuffle": True, "seed": 42}
    loader = PyAsyncDataLoader(ds, opts)
    batches = []
    async for batch in loader:
        print(batch)
        batches.append(list(batch))
    # 10 items in batches of 3 (the last is partial); shuffling must neither lose nor repeat any
    assert [len(b) for b in batches] == [3, 3, 3, 1]
    assert sorted(x for b in batches for x in b) == list(range(1, 11))


if __name__ == "__main__":
    test_sync_batches()
    asyncio.run(test_async_batches())
