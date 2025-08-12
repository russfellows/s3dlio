#!/usr/bin/env python3
"""
Compare s3dlio vs AWS s3torchconnector: throughput, integrity, and reader semantics.

Examples
--------
# Throughput + integrity in 'reader' mode (closest to s3torchconnector)
python compare_parity_suite.py \
  --prefix s3://my-bucket/s3dlio-test/torch/ --region us-east-1 \
  --return-type reader --reader sequential --batch-size 4 --prefetch 16 --take 128 --integrity

# Tensor mode (s3dlio default) vs AWS (readers)
python compare_parity_suite.py \
  --prefix s3://my-bucket/s3dlio-test/torch/ --region us-east-1 \
  --return-type tensor --reader sequential --batch-size 4 --prefetch 16 --take 128 --integrity

# Reader semantics probe (partial reads, EOF)
python compare_parity_suite.py \
  --prefix s3://my-bucket/s3dlio-test/torch/ --region us-east-1 \
  --return-type reader --reader sequential --semantics --take 16
"""
import argparse, time, importlib.util, sys, hashlib
from typing import Any, Tuple, Iterable, List

import torch
from torch.utils.data import DataLoader

from s3dlio.torch import S3IterableDataset as S3DLIOIterable

def have_s3torchconnector() -> bool:
    return importlib.util.find_spec("s3torchconnector") is not None

def _identity_collate(batch):  # return list as-is
    return batch

def _materialize_bytes(item: Any, rt_hint: str) -> bytes:
    """Convert a single item into bytes for hashing/integrity."""
    # s3dlio tensor mode
    if isinstance(item, torch.Tensor):
        t = item
        assert t.dtype == torch.uint8 and t.dim() == 1, "tensor item must be 1-D uint8"
        return bytes(memoryview(t.contiguous().cpu().numpy()))
    # bytes-like
    if isinstance(item, (bytes, bytearray, memoryview)):
        return bytes(item)
    # reader-like (AWS or s3dlio return_type='reader')
    if hasattr(item, "read") and callable(item.read):
        try:
            data = item.read()
        finally:
            try:
                if hasattr(item, "close"):
                    item.close()
            except Exception:
                pass
        return data
    raise TypeError(f"Cannot turn {type(item)} into bytes")

def _count_bytes(item: Any) -> int:
    """Fast byte count without necessarily materializing whole byte payload."""
    if isinstance(item, torch.Tensor):
        return int(item.numel())
    if isinstance(item, (bytes, bytearray, memoryview)):
        return len(item)
    if hasattr(item, "read") and callable(item.read):
        try:
            data = item.read()
        finally:
            try:
                if hasattr(item, "close"):
                    item.close()
            except Exception:
                pass
        return len(data)
    return len(_materialize_bytes(item, "unknown"))

def _hashes_from_loader(loader: Iterable, rt_hint: str, limit: int) -> List[str]:
    """Consume up to `limit` items from loader and return SHA256 hex digests."""
    out: List[str] = []
    for batch in loader:
        items = batch if isinstance(batch, list) else [t for t in batch]
        for it in items:
            out.append(hashlib.sha256(_materialize_bytes(it, rt_hint)).hexdigest())
            if limit and len(out) >= limit:
                return out
    return out

def build_s3dlio_loader(prefix: str, args) -> Tuple[Iterable, str]:
    ds = S3DLIOIterable.from_prefix(
        prefix,
        batch_size=1, drop_last=False,
        shuffle=False, seed=0,
        num_workers=0, prefetch=args.prefetch,
        reader_mode=args.reader, part_size=args.part_size, max_inflight_parts=args.inflight,
        enable_sharding=False,
        return_type=args.return_type,
    )
    if args.return_type == "tensor":
        return DataLoader(ds, batch_size=args.batch_size, num_workers=0), "tensor"
    return DataLoader(ds, batch_size=args.batch_size, num_workers=0, collate_fn=_identity_collate), "reader_or_bytes"

def build_aws_loader(prefix: str, region: str, batch_size: int) -> Iterable:
    from s3torchconnector import S3IterableDataset as AWSIterable  # type: ignore
    ds = AWSIterable.from_prefix(prefix, region=region)
    return DataLoader(ds, batch_size=batch_size, num_workers=0, collate_fn=_identity_collate)

def throughput(loader: Iterable, rt_hint: str, take: int) -> Tuple[int, int, float]:
    total, total_bytes = 0, 0
    t0 = time.perf_counter()
    for batch in loader:
        items = batch if isinstance(batch, list) else [t for t in batch]
        for it in items:
            total += 1
            total_bytes += _count_bytes(it)
            if take and total >= take:
                dt = time.perf_counter() - t0
                return total, total_bytes, dt
    dt = time.perf_counter() - t0
    return total, total_bytes, dt

def integrity_compare(l1: Iterable, l2: Iterable, rt1: str, take: int) -> Tuple[int, int]:
    """Compare first N items by SHA256 multiset equality (order-agnostic)."""
    # To keep it practical, just consume N items from each side
    h1 = _hashes_from_loader(l1, rt1, take)
    h2 = _hashes_from_loader(l2, "reader_or_bytes", take)
    from collections import Counter
    c1, c2 = Counter(h1), Counter(h2)
    # Return (#matched, #expected)
    matched = sum((c1 & c2).values())
    expected = max(len(h1), len(h2))
    return matched, expected

def reader_semantics_probe(loader: Iterable, count: int, chunk: int = 1024) -> Tuple[int, int]:
    """Probe first `count` items: read chunk, then rest, then EOF; return (#ok, #tested)."""
    ok = 0
    tested = 0
    for batch in loader:
        items = batch if isinstance(batch, list) else [t for t in batch]
        for r in items:
            if not (hasattr(r, "read") and callable(r.read)):
                # skip non-readers silently
                continue
            tested += 1
            try:
                a = r.read(chunk)
                b = r.read()  # rest
                c = r.read()  # EOF should be empty
                if hasattr(r, "close"):
                    try: r.close()
                    except Exception: pass
                if isinstance(a, (bytes, bytearray)) and isinstance(b, (bytes, bytearray)) and isinstance(c, (bytes, bytearray)) and len(c) == 0 and (len(a) <= chunk):
                    ok += 1
            except Exception:
                pass
            if tested >= count:
                return ok, tested
    return ok, tested

def fmt(total, total_bytes, dt):
    mb = total_bytes / (1024*1024)
    ips = total / dt if dt > 0 else 0.0
    mbps = mb / dt if dt > 0 else 0.0
    return f"items={total} bytes={total_bytes} ({mb:.2f} MiB) time={dt:.3f}s  ->  {ips:.1f} it/s, {mbps:.2f} MiB/s"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--region", default="us-east-1")
    ap.add_argument("--reader", choices=["sequential","range"], default="sequential", help="s3dlio reader mode")
    ap.add_argument("--return-type", choices=["tensor","bytes","reader"], default="tensor", help="s3dlio return type")
    ap.add_argument("--part-size", dest="part_size", type=int, default=8<<20)
    ap.add_argument("--inflight", type=int, default=4)
    ap.add_argument("--prefetch", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--take", type=int, default=128)
    ap.add_argument("--integrity", action="store_true", help="Compare multiset of SHA256 hashes for first N items.")
    ap.add_argument("--semantics", action="store_true", help="Probe reader .read()/.close() semantics on first items.")
    args = ap.parse_args()

    # s3dlio
    print(f"[s3dlio] return_type={args.return_type} reader={args.reader} part_size={args.part_size} inflight={args.inflight} prefetch={args.prefetch}")
    lio_loader, rt_hint = build_s3dlio_loader(args.prefix, args)
    d_total, d_bytes, d_dt = throughput(lio_loader, rt_hint, args.take)
    print("s3dlio:", fmt(d_total, d_bytes, d_dt))

    # aws
    if not have_s3torchconnector():
        print("AWS s3torchconnector not installed; skipping comparison.")
        sys.exit(0)
    print(f"[aws]    region={args.region}")
    aws_loader = build_aws_loader(args.prefix, args.region, args.batch_size)
    a_total, a_bytes, a_dt = throughput(aws_loader, "reader_or_bytes", args.take)
    print("aws   :", fmt(a_total, a_bytes, a_dt))

    if d_total and a_total:
        print(f"\nThroughput ratio (s3dlio/aws): "
              f"items/s = {d_total/d_dt:.2f}/{a_total/a_dt:.2f}  "
              f"MiB/s = {d_bytes/(1024*1024)/d_dt:.2f}/{a_bytes/(1024*1024)/a_dt:.2f}")

    # Integrity (order-agnostic hash compare)
    if args.integrity:
        lio_loader, rt_hint = build_s3dlio_loader(args.prefix, args)
        aws_loader = build_aws_loader(args.prefix, args.region, args.batch_size)
        matched, expected = integrity_compare(lio_loader, aws_loader, rt_hint, args.take)
        status = "PASS" if matched == expected else f"PARTIAL ({matched}/{expected})"
        print(f"\n[integrity] {status}: matched={matched}, expected={expected} (order-agnostic SHA256)")

    # Reader semantics (only meaningful with return_type=reader; we’ll probe AWS anyway)
    if args.semantics:
        if args.return_type != "reader":
            print("\n[semantics] Note: s3dlio not in 'reader' mode; probing AWS only.")
        else:
            lio_loader, _ = build_s3dlio_loader(args.prefix, args)
            ok, tested = reader_semantics_probe(lio_loader, min(args.take, 16))
            print(f"[semantics] s3dlio reader: ok={ok}/{tested}")
        aws_loader = build_aws_loader(args.prefix, args.region, args.batch_size)
        ok, tested = reader_semantics_probe(aws_loader, min(args.take, 16))
        print(f"[semantics] aws reader:    ok={ok}/{tested}")

if __name__ == "__main__":
    main()


