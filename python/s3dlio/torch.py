"""
PyTorch wrappers that stream bytes from the Rust S3 DataLoader.

Notes
-----
• This uses ONLY the Rust async dataloader (`PyS3AsyncDataLoader`).
• Set concurrency via Rust `LoaderOptions` (prefetch, num_workers, shuffle…).
• Recommended: use PyTorch DataLoader with `num_workers=0` and let Rust do I/O.
  If you want batching, either:
    - set batch_size on the *PyTorch* DataLoader (keep Rust batch_size=1), or
    - set Rust `batch_size>1` and keep PyTorch DataLoader batch_size=1.
  This wrapper FLATTENS Rust batches into per-sample tensors.
"""

from __future__ import annotations

import asyncio
import threading
import queue
from typing import Iterable, Dict, Any, Sequence, Optional

import torch
from torch.utils.data import IterableDataset

#import _pymod as _core  # native module built by maturin
from . import _pymod as _core


def _normalize_opts(**kwargs) -> Dict[str, Any]:
    """
    Build the LoaderOptions dict expected by the Rust side.

    Supported keys (mirrors `LoaderOptions` in Rust):
      batch_size: int (default 1)
      drop_last: bool
      shuffle: bool
      seed: int
      num_workers: int
      prefetch: int
      auto_tune: bool
    """
    opts: Dict[str, Any] = {
        "batch_size":  kwargs.get("batch_size", 1),
        "drop_last":  kwargs.get("drop_last", False),
        "shuffle":    kwargs.get("shuffle", False),
        "seed":       kwargs.get("seed", 0),
        "num_workers":kwargs.get("num_workers", 0),
        "prefetch":   kwargs.get("prefetch", 8),
        "auto_tune":  kwargs.get("auto_tune", False),
    }
    return opts


class _AsyncBytesSource:
    """
    Drives the Rust `PyS3AsyncDataLoader` in a background thread +
    event loop, pushing *samples* (bytes) into a Queue for synchronous
    consumption on the main thread.
    """

    def __init__(self, uri: str, opts: Dict[str, Any]):
        self._uri = uri
        self._opts = opts
        self._q: "queue.Queue[Optional[bytes]]" = queue.Queue(
            max(32, 2 * opts.get("prefetch", 8) * max(1, opts.get("batch_size", 1)))
        )
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "_AsyncBytesSource":
        def runner():
            async def run():
                loader = _core.PyS3AsyncDataLoader(self._uri, self._opts)
                try:
                    async for batch in loader:           # batch: List[bytes]
                        for b in batch:
                            self._q.put(b, block=True)
                finally:
                    # Signal end of stream
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


class S3IterableDataset(IterableDataset):
    """
    Stream objects as 1D uint8 tensors directly from S3 via the Rust DataLoader.

    Usage
    -----
    >>> ds = S3IterableDataset.from_prefix(
    ...     "s3://bucket/prefix/",
    ...     prefetch=8, num_workers=32, shuffle=True, seed=123
    ... )
    >>> loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0)
    >>> for batch in loader:
    ...     # batch is [N, L] after PyTorch collate (variable L if JSON etc.)
    ...     ...
    """

#    def __init__(self, uri: str, *, loader_opts: Dict[str, Any]):
    def __init__(
        self,
        uri: str,
        *,
        loader_opts: Dict[str, Any],
        writable: bool = False,                     # NEW
        suppress_nonwritable_warning: bool = True,  # NEW
    ):
        self._uri = uri
        self._opts = loader_opts
        self._writable = writable                                   # NEW
        self._suppress_nonwritable_warning = suppress_nonwritable_warning  # NEW

    @classmethod
    def from_prefix(
        cls,
        uri: str,
        *,
        batch_size: int = 1,
        drop_last: bool = False,
        shuffle: bool = False,
        seed: int = 0,
        num_workers: int = 0,
        prefetch: int = 8,
        auto_tune: bool = False,
        writable: bool = False,                     # NEW
        suppress_nonwritable_warning: bool = True,  # NEW
    ) -> "S3IterableDataset":
        opts = _normalize_opts(
            batch_size=batch_size, drop_last=drop_last, shuffle=shuffle, seed=seed,
            num_workers=num_workers, prefetch=prefetch, auto_tune=auto_tune
        )
        return cls( uri, loader_opts=opts, writable=writable, 
            suppress_nonwritable_warning=suppress_nonwritable_warning,)

    def __iter__(self) -> Iterable[torch.Tensor]:
        src = _AsyncBytesSource(self._uri, self._opts).start()
        try:
            for b in src:
                # Default: zero-copy read-only; Optional: writable with one copy
                if self._writable:
                    buf = bytearray(b)  # one copy → writable backing store
                    yield torch.frombuffer(memoryview(buf), dtype=torch.uint8)
                else:
                    mv = memoryview(b)  # zero-copy, read-only
                    if self._suppress_nonwritable_warning:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings(
                                "ignore",
                                message=(
                                    "The given buffer is not writable, and PyTorch does not "
                                    "support non-writable tensors.*"
                                ),
                                category=UserWarning,
                            )
                            yield torch.frombuffer(mv, dtype=torch.uint8)
                    else:
                        yield torch.frombuffer(mv, dtype=torch.uint8)
        finally:
            src.join(timeout=1.0)

