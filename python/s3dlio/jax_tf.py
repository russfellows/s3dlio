"""
JAX / TensorFlow helpers that stream from the Rust S3 DataLoader.

• Uses ONLY the Rust async dataloader (`PyS3AsyncDataLoader`).
• JAX path yields NumPy uint8 arrays (zero-copy view over bytes).
• TF path builds a tf.data.Dataset from a Python generator fed by the loader.
• NEW: `writable` option — when True, wrap each `bytes` in `bytearray` so the
  NumPy view is writeable (one extra copy). Default is False (zero extra copy).
"""

from __future__ import annotations

import asyncio
import threading
import queue
from typing import Optional, Dict, Any, Iterable, Callable

import numpy as np
import jax.numpy as jnp

#import _pymod as _core  # native module built by maturin
from . import _pymod as _core


def _normalize_opts(**kwargs) -> Dict[str, Any]:
    return {
        "batch_size":  kwargs.get("batch_size", 1),
        "drop_last":  kwargs.get("drop_last", False),
        "shuffle":    kwargs.get("shuffle", False),
        "seed":       kwargs.get("seed", 0),
        "num_workers":kwargs.get("num_workers", 0),
        "prefetch":   kwargs.get("prefetch", 8),
        "auto_tune":  kwargs.get("auto_tune", False),
    }


class _AsyncBytesSource:
    def __init__(
        self,
        uri: str,
        opts: Dict[str, Any],
        *,
        writable: bool = False,
        suppress_nonwritable_warning: bool = True,
    ):
        self._uri = uri
        self._opts = opts
        self._writable = writable
        self._q: "queue.Queue[Optional[bytes]]" = queue.Queue(
            max(32, 2 * opts.get("prefetch", 8) * max(1, opts.get("batch_size", 1)))
        )
        self._thread: Optional[threading.Thread] = None
        self._writable = writable
        self._suppress_nonwritable_warning = suppress_nonwritable_warning

    def start(self) -> "_AsyncBytesSource":
        def runner():
            async def run():
                loader = _core.PyS3AsyncDataLoader(self._uri, self._opts)
                try:
                    async for batch in loader:     # batch: List[bytes]
                        for b in batch:
                            self._q.put(b, block=True)
                finally:
                    self._q.put(None)

            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                loop.run_until_complete(run())
            finally:
                try:
                    loop.close()
                except Exception:
                    pass

        t = threading.Thread(target=runner, daemon=True)
        t.start()
        self._thread = t
        return self

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is None:
                break
            yield item

    def join(self, timeout: Optional[float] = None):
        if self._thread is not None:
            self._thread.join(timeout=timeout)


class S3JaxIterable:
    """
    JAX-friendly iterator yielding NumPy uint8 arrays (caller can jnp.asarray).

    >>> it = S3JaxIterable.from_prefix("s3://bucket/prefix/", prefetch=8, num_workers=32)
    >>> import jax.numpy as jnp
    >>> for a in it:
    ...     x = jnp.asarray(a)
    ...     ...
    """

    def __init__(
        self,
        uri: str,
        *,
        loader_opts: Dict[str, Any],
        transform: Optional[Callable] = None,
        writable: bool = False,                    # NEW
        suppress_nonwritable_warning: bool = True, # NEW
    ):
        self._uri = uri
        self._opts = loader_opts
        self._xfm = transform
        self._writable = writable           # NEW
        self._suppress_nonwritable_warning = suppress_nonwritable_warning

    @classmethod
    def from_prefix(
        cls,
        uri: str,
        *,
        prefetch: int = 8,
        num_workers: int = 0,
        batch_size: int = 1,
        drop_last: bool = False,
        shuffle: bool = False,
        seed: int = 0,
        auto_tune: bool = False,
        transform: Optional[Callable] = None,
        writable: bool = False,             # NEW
        suppress_nonwritable_warning: bool = True,
    ) -> "S3JaxIterable":
        opts = _normalize_opts(
            batch_size=batch_size, drop_last=drop_last, shuffle=shuffle, seed=seed,
            num_workers=num_workers, prefetch=prefetch, auto_tune=auto_tune
        )
        return cls( uri, loader_opts=opts, transform=transform, writable=writable,
            suppress_nonwritable_warning=suppress_nonwritable_warning,
        )

    def __iter__(self):
        src = _AsyncBytesSource( self._uri, self._opts, writable=self._writable,
            suppress_nonwritable_warning=self._suppress_nonwritable_warning,
        ).start()
        try:
            for b in src:
                if self._writable:
                    # one copy → writable backing store
                    arr = np.frombuffer(memoryview(bytearray(b)), dtype=np.uint8)
                else:
                    # zero-copy readonly view
                    arr = np.frombuffer(memoryview(b), dtype=np.uint8)

                yield self._xfm(arr) if self._xfm is not None else arr
        finally:
            src.join(timeout=1.0)


def make_tf_dataset(
    uri: str,
    *,
    prefetch: int = 8,
    num_workers: int = 0,
    batch_size: int = 1,
    drop_last: bool = False,
    shuffle: bool = False,
    seed: int = 0,
    auto_tune: bool = False,
    map_fn: Optional[Callable] = None,
    writable: bool = False,                      # NEW (parity; TF will still copy)
    suppress_nonwritable_warning: bool = True,   # NEW
):
    """
    Build a tf.data.Dataset that streams uint8 tensors from the Rust loader.
    """
    import tensorflow as tf  # lazy import

    opts = _normalize_opts(
        batch_size=batch_size, drop_last=drop_last, shuffle=shuffle, seed=seed,
        num_workers=num_workers, prefetch=prefetch, auto_tune=auto_tune
    )

    def gen():
        src = _AsyncBytesSource( uri, opts, writable=writable,
            suppress_nonwritable_warning=suppress_nonwritable_warning,
        ).start()
        try:
            for b in src:
                if writable:
                    # one copy → writable buffer
                    yield np.frombuffer(memoryview(bytearray(b)), dtype=np.uint8)
                else:
                    # zero-copy readonly view
                    yield np.frombuffer(memoryview(b), dtype=np.uint8)
        finally:
            src.join(timeout=1.0)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=tf.TensorSpec(shape=(None,), dtype=tf.uint8),
    )
    if map_fn is not None:
        ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds

