#!/usr/bin/env python3
"""
bench_s3-torch_v8.py  (updated)

– Loads AWS_* creds + endpoint from .env
– Configures the CRT client for high throughput via S3ClientConfig:
     • throughput_target_gbps (1 Tbps hint)
     • part_size = 32 MiB
     • max_connections = 64
– Parallel PUT of N objects via S3Checkpoint.writer
– Parallel GET (one‐task‐per‐key)
– Sharded GET (each thread loops a shard sequentially)
"""

import os
import time
import argparse
from dotenv import load_dotenv

# 1) Load .env first
load_dotenv()

# 2) Imports
from s3torchconnector import S3Checkpoint, S3ClientConfig
import torch
import concurrent.futures

# 3) Helpers
def make_keys(prefix: str, n: int):
    return [f"{prefix}/obj_{i:06d}.pt" for i in range(n)]

def parallel_put(cp: S3Checkpoint, bucket: str, keys: list[str], size_mb: int, workers: int):
    payload = torch.randint(0, 256, size=(size_mb * 1024**2 // 4,), dtype=torch.int32)
    uri_base = f"s3://{bucket}"

    def upload_one(key: str):
        with cp.writer(f"{uri_base}/{key}") as w:
            torch.save(payload, w)

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(upload_one, keys))
    dt = time.perf_counter() - t0

    total_bytes = len(keys) * size_mb * 1024**2
    print(f"PUT  ({workers} threads) "
          f"{total_bytes/1e6:.1f} MB in {dt:.2f}s → "
          f"{total_bytes/dt/1024/1024:.1f} MiB/s, {len(keys)/dt:.1f} ops/s")

def threaded_get(cp: S3Checkpoint, bucket: str, keys: list[str], workers: int):
    client = cp._client

    def fetch_one(key: str) -> int:
        reader = client.get_object(bucket, key)
        data = reader.read()
        return len(data)

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        sizes = list(ex.map(fetch_one, keys))
    dt = time.perf_counter() - t0

    total_bytes = sum(sizes)
    print(f"GET  ({workers} threads) "
          f"{total_bytes/1e6:.1f} MB in {dt:.2f}s → "
          f"{total_bytes/dt/1024/1024:.1f} MiB/s, {len(keys)/dt:.1f} ops/s")

def sharded_get(cp: S3Checkpoint, bucket: str, keys: list[str], workers: int):
    client = cp._client
    shards = [keys[i::workers] for i in range(workers)]

    def fetch_shard(shard_keys: list[str]) -> tuple[int,int]:
        total_bytes = 0
        count = 0
        for key in shard_keys:
            reader = client.get_object(bucket, key)
            buf = reader.read()
            total_bytes += len(buf)
            count += 1
        return total_bytes, count

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(fetch_shard, shards))
    dt = time.perf_counter() - t0

    total_bytes = sum(r[0] for r in results)
    total_objs  = sum(r[1] for r in results)
    print(f"SHARDED-GET ({workers} threads) "
          f"{total_bytes/1e6:.1f} MB in {dt:.2f}s → "
          f"{total_bytes/dt/1024/1024:.1f} MiB/s, {total_objs/dt:.1f} ops/s")


# 4) Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="End-to-end PUT/GET benchmark for s3-connector-for-pytorch"
    )
    parser.add_argument("--bucket",   required=True, help="S3 bucket name")
    parser.add_argument("--prefix",   default="bench",   help="object key prefix")
    parser.add_argument("-n",         type=int,   default=100, help="number of objects")
    parser.add_argument("--mb",       type=int,   default=16,  help="size per object (MiB)")
    parser.add_argument("--threads",  type=int,   default=16,  help="parallel threads")
    parser.add_argument("--gbps",     type=float, default=100.0,
                        help="throughput_target_gbps for CRT (Gbps)")
    args = parser.parse_args()

    # ── Build a high-throughput CRT client config (no other made-up params) ──────────
    client_cfg = S3ClientConfig(
        throughput_target_gbps=args.gbps,
        part_size=32 * 1024**2,       # 32 MiB
        force_path_style=False,       # set True if your endpoint needs path-style
        max_attempts=3                # standard retry setting
    )

    cp = S3Checkpoint(
        region=os.getenv("AWS_REGION"),
        s3client_config=client_cfg
    )

    # Prepare keys
    keys = make_keys(args.prefix, args.n)

    print(f"\n→ UPLOADING {args.n}×{args.mb} MiB objects "
          f"to {args.bucket}/{args.prefix} using {args.threads} threads")
    parallel_put(cp, args.bucket, keys, args.mb, args.threads)

    print(f"\n→ DOWNLOADING {args.n} objects "
          f"from {args.bucket}/{args.prefix} using {args.threads} threads")
    threaded_get(cp, args.bucket, keys, args.threads)

    print(f"\n→ DOWNLOADING {args.n} objects "
          f"from {args.bucket}/{args.prefix} using {args.threads} threads "
          f"(sharded per thread)")
    sharded_get(cp, args.bucket, keys, args.threads)

