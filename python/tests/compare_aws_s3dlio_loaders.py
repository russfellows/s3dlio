#!/usr/bin/env python3
"""
Compare AWS s3torchconnector vs s3dlio (iterable mode) on the same prefix.

Examples
--------
python python/tests/compare_aws_s3dlio_loaders.py \
  --prefix s3://my-bucket/s3dlio-test/torch/ --region us-east-1 \
  --reader range --prefetch 16 --batch-size 4 --take 128
"""
import argparse, time, importlib.util, sys
from typing import Any, Tuple

import torch
from torch.utils.data import DataLoader

from s3dlio.torch import S3IterableDataset as S3DLIOIterable


def have_s3torchconnector() -> bool:
    return importlib.util.find_spec("s3torchconnector") is not None


def _identity_collate(batch):  # returns list as-is
    return batch


def _bytes_len_from_any(x: Any) -> int:
    """
    Return the payload size in bytes without forcing extra copies when possible.
    - s3dlio items: torch.Tensor([L], dtype=uint8) -> use numel()
    - bytes-like: len(...)
    - AWS connector items: reader with .read()/.close() -> read once to get bytes length
    """
    # s3dlio: torch tensor
    if isinstance(x, torch.Tensor):
        return int(x.numel())

    # Native bytes-like
    if isinstance(x, (bytes, bytearray, memoryview)):
        return len(x)

    # AWS connector: reader object (duck-typed)
    if hasattr(x, "read") and callable(x.read):
        try:
            data = x.read()
        finally:
            try:
                if hasattr(x, "close"):
                    x.close()
            except Exception:
                pass
        return len(data)

    if hasattr(x, "readall") and callable(x.readall):
        data = x.readall()
        return len(data)

    if hasattr(x, "read_all") and callable(x.read_all):
        data = x.read_all()
        return len(data)

    raise TypeError(f"Cannot determine size for item of type {type(x)}")


def run_s3dlio(prefix: str, args) -> Tuple[int, int, float]:
    ds = S3DLIOIterable.from_prefix(
        prefix,
        batch_size=1, drop_last=False,
        shuffle=False, seed=0,
        num_workers=0, prefetch=args.prefetch,
        reader_mode=args.reader, part_size=args.part_size, max_inflight_parts=args.inflight,
        enable_sharding=False,
    )
    # s3dlio yields torch.Tensors; use identity collate and just count numel()
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0, collate_fn=_identity_collate)
    total, total_bytes = 0, 0
    t0 = time.perf_counter()
    for batch in loader:
        for item in batch:
            total += 1
            total_bytes += _bytes_len_from_any(item)
        if args.take and total >= args.take:
            break
    dt = time.perf_counter() - t0
    return total, total_bytes, dt


def run_aws(prefix: str, region: str, args) -> Tuple[int, int, float]:
    from s3torchconnector import S3IterableDataset as AWSIterable  # type: ignore
    ds = AWSIterable.from_prefix(prefix, region=region)
    # AWS yields reader objects; identity collate + size via read()
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=0, collate_fn=_identity_collate)
    total, total_bytes = 0, 0
    t0 = time.perf_counter()
    for batch in loader:
        for item in batch:
            total += 1
            total_bytes += _bytes_len_from_any(item)
        if args.take and total >= args.take:
            break
    dt = time.perf_counter() - t0
    return total, total_bytes, dt


def fmt(total, total_bytes, dt):
    mb = total_bytes / (1024 * 1024)
    ips = total / dt if dt > 0 else 0.0
    mbps = mb / dt if dt > 0 else 0.0
    return f"items={total} bytes={total_bytes} ({mb:.2f} MiB) time={dt:.3f}s  ->  {ips:.1f} it/s, {mbps:.2f} MiB/s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--region", default="us-east-1", help="AWS region for s3torchconnector")
    ap.add_argument("--reader", choices=["sequential", "range"], default="sequential", help="s3dlio reader mode")
    ap.add_argument("--part-size", dest="part_size", type=int, default=8 << 20)
    ap.add_argument("--inflight", type=int, default=4)
    ap.add_argument("--prefetch", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--take", type=int, default=0)
    args = ap.parse_args()

    print(f"[s3dlio] reader={args.reader} part_size={args.part_size} inflight={args.inflight} prefetch={args.prefetch}")
    d_total, d_bytes, d_dt = run_s3dlio(args.prefix, args)
    print("s3dlio:", fmt(d_total, d_bytes, d_dt))

    if not have_s3torchconnector():
        print("AWS s3torchconnector not installed; skipping that half of the comparison.")
        sys.exit(0)

    print(f"[aws]    region={args.region}")
    a_total, a_bytes, a_dt = run_aws(args.prefix, args.region, args)
    print("aws   :", fmt(a_total, a_bytes, a_dt))

    if d_total and a_total:
        print(f"\nThroughput ratio (s3dlio/aws): "
              f"items/s = {d_total/d_dt:.2f}/{a_total/a_dt:.2f}  "
              f"MiB/s = {d_bytes/(1024*1024)/d_dt:.2f}/{a_bytes/(1024*1024)/a_dt:.2f}")


if __name__ == "__main__":
    main()

