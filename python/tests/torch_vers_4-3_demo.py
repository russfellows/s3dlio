#!/usr/bin/env python3
"""
Torch Stage-0 demo: exercise s3dlio.torch S3IterableDataset/S3MapDataset
+ built-in selftests for return_type = {tensor, bytes, reader}.

Examples
--------
# Seed 128 tiny objects under a test prefix, then run iterable+range mode
python torch_stage0_demo.py \
  --prefix s3://my-bucket/s3dlio-test/torch/ \
  --setup 128 --size 4096 \
  --iterable --reader range --prefetch 16 --batch-size 4 --num-workers 0

# Map-style with default (sequential) reader
python torch_stage0_demo.py --prefix s3://my-bucket/s3dlio-test/torch/ --map --take 10

# Run built-in selftests (validates tensor/bytes/reader)
python torch_stage0_demo.py --prefix s3://my-bucket/s3dlio-test/torch/ --selftest --take 32
"""
import argparse, time
from typing import Dict, Any, Optional, List

import torch
from torch.utils.data import DataLoader

from s3dlio import _pymod as _core
from s3dlio.torch import S3IterableDataset, S3MapDataset


def _identity_collate(batch):  # returns list as-is (needed for bytes/reader modes)
    return batch


def seed(prefix: str, n: int, size: int, jobs: int) -> None:
    print(f"[seed] Writing {n} objects of {size} bytes to {prefix} (jobs={jobs})...")
    _core.put(prefix, n, "obj_{}_of_{}.bin", max_in_flight=jobs, size=size,
              should_create_bucket=False, object_type="zeros",
              dedup_factor=1, compress_factor=1)
    print("[seed] Done.")


def run_iterable(prefix: str, args) -> None:
    print(f"[iterable] reader={args.reader} part_size={args.part_size} inflight={args.inflight} "
          f"prefetch={args.prefetch} enable_sharding={args.enable_sharding}")

    ds = S3IterableDataset.from_prefix(
        prefix,
        batch_size=1, drop_last=False,
        shuffle=args.shuffle, seed=args.seed,
        num_workers=0, prefetch=args.prefetch,
        reader_mode=args.reader, part_size=args.part_size, max_inflight_parts=args.inflight,
        enable_sharding=args.enable_sharding,
        writable=args.writable, suppress_nonwritable_warning=True,
        return_type="tensor",  # demo path keeps default tensor behavior
    )
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers)

    total = 0
    total_bytes = 0
    t0 = time.perf_counter()
    for batch in loader:
        if isinstance(batch, list):
            items = batch
        else:
            items = [t for t in batch]  # split stacked tensor along dim 0
        total += len(items)
        total_bytes += sum(int(t.numel()) for t in items)
        if args.take and total >= args.take:
            break
    dt = time.perf_counter() - t0
    mb = total_bytes / (1024*1024)
    print(f"[iterable] items={total} bytes={total_bytes} ({mb:.2f} MiB) "
          f"time={dt:.3f}s -> {total/dt:.1f} it/s, {mb/dt:.2f} MiB/s")


def run_map(prefix: str, args) -> None:
    print(f"[map] reader={args.reader} part_size={args.part_size} inflight={args.inflight} "
          f"take={args.take or 'ALL'} writable={args.writable}")

    ds = S3MapDataset.from_prefix(
        prefix,
        reader_mode=args.reader, part_size=args.part_size, max_inflight_parts=args.inflight,
        writable=args.writable, suppress_nonwritable_warning=True,
        return_type="tensor",
    )
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=args.shuffle)

    total = 0
    total_bytes = 0
    t0 = time.perf_counter()
    for batch in loader:
        if isinstance(batch, list):
            items = batch
        else:
            items = [t for t in batch]
        total += len(items)
        total_bytes += sum(int(t.numel()) for t in items)
        if args.take and total >= args.take:
            break
    dt = time.perf_counter() - t0
    mb = total_bytes / (1024*1024)
    print(f"[map] items={total} bytes={total_bytes} ({mb:.2f} MiB) "
          f"time={dt:.3f}s -> {total/dt:.1f} it/s, {mb/dt:.2f} MiB/s")


# -------------------------
# Selftests for return_type
# -------------------------
def _assert(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def _loader_for(prefix: str, args, return_type: str) -> DataLoader:
    ds = S3IterableDataset.from_prefix(
        prefix,
        batch_size=1, drop_last=False,
        shuffle=False, seed=0,
        num_workers=0, prefetch=args.prefetch,
        reader_mode=args.reader, part_size=args.part_size, max_inflight_parts=args.inflight,
        enable_sharding=False, return_type=return_type,
    )
    if return_type == "tensor":
        return DataLoader(ds, batch_size=args.batch_size, num_workers=0)
    else:
        # bytes/reader need identity collate (matches s3torchconnector behavior)
        return DataLoader(ds, batch_size=args.batch_size, num_workers=0, collate_fn=_identity_collate)


def _flatten_batch(batch) -> List:
    # batch can be list[...] (identity collate) or a stacked tensor
    if isinstance(batch, list):
        return batch
    return [t for t in batch]  # split stacked tensor along dim 0


def selftest_return_types(prefix: str, args) -> None:
    print(f"[selftest] prefix={prefix} reader={args.reader} part={args.part_size} inflight={args.inflight}")
    want = args.take if args.take else None  # stop when reached

    # 1) Tensor mode
    total, total_bytes = 0, 0
    for batch in _loader_for(prefix, args, "tensor"):
        items = _flatten_batch(batch)
        for t in items:
            _assert(isinstance(t, torch.Tensor), "tensor mode: item is not a torch.Tensor")
            _assert(t.dtype == torch.uint8 and t.dim() == 1, "tensor mode: must be 1-D uint8")
            total_bytes += int(t.numel())
        total += len(items)
        if want and total >= want:
            break
    _assert(total > 0, "tensor mode: no items read")
    print(f"[PASS] tensor: items={total}, bytes={total_bytes}")

    # 2) Bytes mode
    total, total_bytes = 0, 0
    for batch in _loader_for(prefix, args, "bytes"):
        for b in _flatten_batch(batch):
            _assert(isinstance(b, (bytes, bytearray, memoryview)), "bytes mode: item is not bytes-like")
            total_bytes += len(b)
        total += len(_flatten_batch(batch))
        if want and total >= want:
            break
    _assert(total > 0, "bytes mode: no items read")
    print(f"[PASS] bytes:  items={total}, bytes={total_bytes}")

    # 3) Reader mode
    total, total_bytes = 0, 0
    for batch in _loader_for(prefix, args, "reader"):
        for r in _flatten_batch(batch):
            _assert(hasattr(r, "read") and callable(r.read), "reader mode: item has no .read()")
            data = r.read()  # consume
            # best-effort close support
            if hasattr(r, "close"):
                try: r.close()
                except Exception: pass
            _assert(isinstance(data, (bytes, bytearray)), "reader mode: .read() did not return bytes")
            total_bytes += len(data)
        total += len(_flatten_batch(batch))
        if want and total >= want:
            break
    _assert(total > 0, "reader mode: no items read")
    print(f"[PASS] reader: items={total}, bytes={total_bytes}")

    print("[selftest] OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="s3://bucket/prefix/")
    ap.add_argument("--setup", type=int, default=0, help="Create N test objects under prefix before running.")
    ap.add_argument("--size", type=int, default=4096, help="Size of each test object for --setup.")
    ap.add_argument("--jobs", type=int, default=64, help="Max inflight for setup writer.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--iterable", action="store_true")
    g.add_argument("--map", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="Validate return_type: tensor, bytes, reader.")
    ap.add_argument("--reader", choices=["sequential","range"], default="sequential")
    ap.add_argument("--part-size", dest="part_size", type=int, default=8<<20)
    ap.add_argument("--inflight", type=int, default=4, help="Max inflight range parts (range mode).")
    ap.add_argument("--prefetch", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--take", type=int, default=0, help="Stop after N items (0 = all)")
    ap.add_argument("--writable", action="store_true")
    ap.add_argument("--enable-sharding", action="store_true", help="Shard across ranks/workers (iterable only).")
    args = ap.parse_args()

    if args.setup > 0:
        seed(args.prefix, n=args.setup, size=args.size, jobs=args.jobs)

    if args.selftest:
        selftest_return_types(args.prefix, args)
        return

    if args.iterable or not args.map:
        run_iterable(args.prefix, args)
    if args.map:
        run_map(args.prefix, args)


if __name__ == "__main__":
    main()

