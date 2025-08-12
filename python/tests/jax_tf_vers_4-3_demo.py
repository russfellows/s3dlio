#!/usr/bin/env python3
"""
JAX/TF Stage-0 demo: exercise s3dlio async loader via small adapters.

Examples
--------
# TensorFlow dataset (range mode)
python jax_tf_stage0_demo.py --prefix s3://my-bucket/s3dlio-test/tf/ --tf --reader range --take 32

# JAX sample (first 16 items)
python jax_tf_stage0_demo.py --prefix s3://my-bucket/s3dlio-test/jax/ --jax --take 16
"""
import argparse, time
from typing import Dict, Any, Optional, Iterator

import numpy as np

from s3dlio import _pymod as _core

# Minimal copy of the torch-side async bridge (no torch dependency).
import asyncio, threading, queue

class _AsyncBytesSource:
    def __init__(self, uri: str, opts: Dict[str, Any]):
        self._uri = uri
        self._opts = opts
        cap = max(32, 2 * opts.get("prefetch", 8) * max(1, opts.get("batch_size", 1)))
        self._q: "queue.Queue[Optional[bytes]]" = queue.Queue(cap)
        self._thread: Optional[threading.Thread] = None
    def start(self) -> "_AsyncBytesSource":
        def runner():
            async def run():
                loader = _core.PyS3AsyncDataLoader(self._uri, self._opts)
                try:
                    async for batch in loader:
                        for b in batch:
                            self._q.put(b, block=True)
                finally:
                    self._q.put(None)
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(run())
            finally:
                try: loop.close()
                except Exception: pass
        t = threading.Thread(target=runner, daemon=True)
        t.start()
        self._thread = t
        return self
    def __iter__(self):
        while True:
            item = self._q.get()
            if item is None: break
            yield item
    def join(self, timeout: Optional[float] = None):
        if self._thread is not None:
            self._thread.join(timeout=timeout)


def _normalize_opts(**kwargs) -> Dict[str, Any]:
    return {
        "batch_size": 1,
        "drop_last": False,
        "shuffle": False,
        "seed": 0,
        "num_workers": 0,
        "prefetch": kwargs.get("prefetch", 8),
        "auto_tune": False,
        "reader_mode": kwargs.get("reader_mode", "sequential"),
        "part_size": kwargs.get("part_size", 8 << 20),
        "max_inflight_parts": kwargs.get("max_inflight_parts", 4),
    }


def run_tf(prefix: str, args):
    import tensorflow as tf

    opts = _normalize_opts(
        prefetch=args.prefetch,
        reader_mode=args.reader,
        part_size=args.part_size,
        max_inflight_parts=args.inflight,
    )

    def gen():
        src = _AsyncBytesSource(prefix, opts).start()
        try:
            n = 0
            for b in src:
                yield np.frombuffer(b, dtype=np.uint8)
                n += 1
                if args.take and n >= args.take:
                    break
        finally:
            src.join(timeout=1.0)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=tf.TensorSpec(shape=(None,), dtype=tf.uint8),
    ).batch(args.batch_size)

    t0 = time.perf_counter()
    total = 0
    total_bytes = 0
    for batch in ds:
        total += int(batch.shape[0])
        total_bytes += int(tf.size(batch).numpy())
    dt = time.perf_counter() - t0
    mb = total_bytes / (1024*1024)
    print(f"[tf] items={total} bytes={total_bytes} ({mb:.2f} MiB) "
          f"time={dt:.3f}s -> {total/dt:.1f} it/s, {mb/dt:.2f} MiB/s")


def run_jax(prefix: str, args):
    import jax
    import jax.numpy as jnp

    opts = _normalize_opts(
        prefetch=args.prefetch,
        reader_mode=args.reader,
        part_size=args.part_size,
        max_inflight_parts=args.inflight,
    )

    src = _AsyncBytesSource(prefix, opts).start()
    t0 = time.perf_counter()
    total = 0
    total_bytes = 0
    try:
        for b in src:
            arr = jnp.asarray(np.frombuffer(b, dtype=np.uint8))
            total += 1
            total_bytes += int(arr.size)
            if args.take and total >= args.take:
                break
    finally:
        src.join(timeout=1.0)
    dt = time.perf_counter() - t0
    mb = total_bytes / (1024*1024)
    print(f"[jax] items={total} bytes={total_bytes} ({mb:.2f} MiB) "
          f"time={dt:.3f}s -> {total/dt:.1f} it/s, {mb/dt:.2f} MiB/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--tf", action="store_true")
    ap.add_argument("--jax", action="store_true")
    ap.add_argument("--reader", choices=["sequential","range"], default="sequential")
    ap.add_argument("--part-size", dest="part_size", type=int, default=8<<20)
    ap.add_argument("--inflight", type=int, default=4)
    ap.add_argument("--prefetch", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--take", type=int, default=0)
    args = ap.parse_args()

    if not args.tf and not args.jax:
        args.tf = True  # default to TF

    if args.tf:
        run_tf(args.prefix, args)
    if args.jax:
        run_jax(args.prefix, args)


if __name__ == "__main__":
    main()

