"""
PyTorch wrappers that stream bytes from the Rust ObjectStore DataLoader.

Supports all URI schemes: file://, s3://, az://, gs://, direct://

Notes
-----
• Uses the generic Rust async dataloader (`create_async_loader`).
• Set concurrency via Rust `LoaderOptions` (prefetch, num_workers, shuffle…).
• Recommended: use PyTorch DataLoader with `num_workers=0` and let Rust do I/O.
  If you want batching, either:
    - set batch_size on the *PyTorch* DataLoader (keep Rust batch_size=1), or
    - set Rust `batch_size>1` and keep PyTorch DataLoader batch_size=1.
  This wrapper FLATTENS Rust batches into per-sample tensors.

Stage-0 additions
-----------------
• Reader strategy controls (reader_mode="sequential"|"range", part_size, max_inflight_parts).
• Optional sharding across ranks × workers for iterable datasets.
• return_type toggle:
    - "tensor" (default): yields torch.Tensor([L], dtype=uint8)
    - "bytes"           : yields Python bytes
    - "reader"          : yields a file-like object with .read(), .close(), etc.
"""

from __future__ import annotations

import asyncio
import threading
import queue
from typing import Iterable, Dict, Any, Sequence, Optional, Iterator, Tuple

import torch
from torch.utils.data import IterableDataset, Dataset, get_worker_info

# import _pymod as _core  # native module built by maturin
from . import _pymod as _core


# -----------------------------------------------------------------------------
# Small reader shim for AWS-style compatibility
# -----------------------------------------------------------------------------
class _BytesReader:
    """
    Minimal file-like wrapper around an in-memory object payload.

    Supported API (covers common s3torchconnector usage):
      .read(size=-1) -> bytes
      .readall() / .read_all() -> bytes
      .close() (no-op)
      __enter__/__exit__ (context manager)
      __len__() -> int (payload length)
    """
    __slots__ = ("_buf", "_pos", "_closed")

    def __init__(self, data: bytes):
        self._buf = memoryview(data)  # avoid a copy; reads will slice -> bytes
        self._pos = 0
        self._closed = False

    def read(self, size: int = -1) -> bytes:
        if self._closed:
            raise ValueError("I/O operation on closed _BytesReader")
        if size is None or size < 0:
            start, end = self._pos, len(self._buf)
            self._pos = end
            return bytes(self._buf[start:end])
        size = int(size)
        start, end = self._pos, min(len(self._buf), self._pos + size)
        self._pos = end
        return bytes(self._buf[start:end])

    def readall(self) -> bytes:
        return self.read(-1)

    def read_all(self) -> bytes:
        return self.read(-1)

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "_BytesReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self._buf)


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

    Stage 0 additions:
      reader_mode: str ("sequential" | "range")
      part_size: int (bytes; for range mode)
      max_inflight_parts: int (>=1; for range mode)
      # Sharding fields are injected automatically when enable_sharding=True
      #   shard_rank, shard_world_size, worker_id, num_workers_pytorch
    """
    opts: Dict[str, Any] = {
        "batch_size":   kwargs.get("batch_size", 1),
        "drop_last":    kwargs.get("drop_last", False),
        "shuffle":      kwargs.get("shuffle", False),
        "seed":         kwargs.get("seed", 0),
        "num_workers":  kwargs.get("num_workers", 0),
        "prefetch":     kwargs.get("prefetch", 8),
        "auto_tune":    kwargs.get("auto_tune", False),

        # NEW: reader strategy & tuning
        "reader_mode":        kwargs.get("reader_mode", "sequential"),
        "part_size":          kwargs.get("part_size", 8 << 20),  # 8 MiB
        "max_inflight_parts": kwargs.get("max_inflight_parts", 4),
    }
    return opts


def _dist_info() -> Tuple[int, int]:
    """Best-effort torch.distributed rank/world_size (works if not initialized)."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
    except Exception:
        pass
    return 0, 1


class _AsyncBytesSource:
    """
    Drives the Rust async dataloader in a background thread + event loop,
    pushing *samples* (bytes) into a Queue for synchronous consumption.
    
    Works with any URI scheme supported by the ObjectStore interface.
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
                # Use the new generic create_async_loader function
                loader = _core.create_async_loader(self._uri, self._opts)
                try:
                    async for batch in loader:  # batch: List[bytes]
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


class ObjectStoreIterableDataset(IterableDataset):
    """
    Stream objects as 1D uint8 tensors (or bytes/reader) via the Rust ObjectStore DataLoader.
    
    Supports all URI schemes: file://, s3://, az://, gs://, direct://

    Usage
    -----
    >>> ds = ObjectStoreIterableDataset.from_prefix(
    ...     "file:///data/prefix/",
    ...     prefetch=8, num_workers=32, shuffle=True, seed=123
    ... )
    >>> loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0)
    >>> for batch in loader:
    ...     # batch is [N, L] after PyTorch collate when return_type='tensor'
    ...     ...
    """

    def __init__(
        self,
        uri: str,
        *,
        loader_opts: Dict[str, Any],
        writable: bool = False,
        suppress_nonwritable_warning: bool = True,
        enable_sharding: bool = False,
        return_type: str = "tensor",  # "tensor" | "bytes" | "reader"
    ):
        self._uri = uri
        self._opts = loader_opts
        self._writable = writable
        self._suppress_nonwritable_warning = suppress_nonwritable_warning
        self._enable_sharding = enable_sharding
        rt = str(return_type).lower()
        if rt not in ("tensor", "bytes", "reader"):
            raise ValueError("return_type must be 'tensor', 'bytes', or 'reader'")
        self._return_type = rt

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
        # reader controls
        reader_mode: str = "sequential",
        part_size: int = 8 << 20,
        max_inflight_parts: int = 4,
        # sharding toggle
        enable_sharding: bool = False,
        # tensor controls
        writable: bool = False,
        suppress_nonwritable_warning: bool = True,
        # return type
        return_type: str = "tensor",
    ) -> "ObjectStoreIterableDataset":
        opts = _normalize_opts(
            batch_size=batch_size, drop_last=drop_last, shuffle=shuffle, seed=seed,
            num_workers=num_workers, prefetch=prefetch, auto_tune=auto_tune,
            reader_mode=reader_mode, part_size=part_size, max_inflight_parts=max_inflight_parts,
        )
        return cls(
            uri,
            loader_opts=opts,
            writable=writable,
            suppress_nonwritable_warning=suppress_nonwritable_warning,
            enable_sharding=enable_sharding,
            return_type=return_type,
        )

    def __iter__(self) -> Iterable[torch.Tensor]:
        # Inject sharding fields only at iteration time (worker info is known here).
        opts = dict(self._opts)
        if self._enable_sharding:
            rank, world = _dist_info()
            wi = get_worker_info()
            worker_id = 0 if wi is None else wi.id
            num_workers = 1 if wi is None else wi.num_workers
            opts.update({
                "shard_rank": rank,
                "shard_world_size": max(1, world),
                "worker_id": worker_id,
                "num_workers_pytorch": max(1, num_workers),
            })

        src = _AsyncBytesSource(self._uri, opts).start()
        try:
            for b in src:
                # Return-style switch BEFORE tensor conversion for full compatibility
                if self._return_type == "reader":
                    yield _BytesReader(b)
                    continue
                if self._return_type == "bytes":
                    yield b
                    continue

                # Default: zero-copy read-only tensor; Optional: writable with one copy
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


# ---------------------------------------------------------------------------
# Map-style dataset for PyTorch parity
# ---------------------------------------------------------------------------
class ObjectStoreMapDataset(Dataset):
    """
    Map-style Dataset over objects under a URI prefix.
    
    Supports all URI schemes: file://, s3://, az://, gs://, direct://

    Yields:
      return_type = "tensor" -> torch.Tensor([L], dtype=uint8)
                   "bytes"   -> bytes
                   "reader"  -> file-like object with .read()/.close()

    By default returns 1-D uint8 tensors backed by a read-only buffer. To make
    them writable in tensor mode, set `writable=True` (incurs one copy per item).
    """
    def __init__(
        self,
        uri: str,
        *,
        reader_mode: str = "sequential",
        part_size: int = 8 << 20,
        max_inflight_parts: int = 4,
        writable: bool = False,
        suppress_nonwritable_warning: bool = True,
        return_type: str = "tensor",
    ) -> None:
        self._uri = uri
        self._opts = _normalize_opts(
            reader_mode=reader_mode, part_size=part_size, max_inflight_parts=max_inflight_parts
        )
        self._writable = writable
        self._suppress_nonwritable_warning = suppress_nonwritable_warning
        rt = str(return_type).lower()
        if rt not in ("tensor", "bytes", "reader"):
            raise ValueError("return_type must be 'tensor', 'bytes', or 'reader'")
        self._return_type = rt
        
        # Parse URI to extract scheme and base path for later reconstruction
        if "://" in uri:
            self._scheme, rest = uri.split("://", 1)
            # Keep the full path as the base (strip trailing slash for consistent joining)
            self._base = rest.rstrip("/")
        else:
            self._scheme = "file"
            self._base = uri.rstrip("/")
        
        # Use the generic Rust dataset - works with any URI scheme
        self._core_ds = _core.create_dataset(uri, self._opts)
        self._keys: Sequence[str] = self._core_ds.keys()

    @classmethod
    def from_prefix(cls, uri: str, **kwargs) -> "ObjectStoreMapDataset":
        return cls(uri, **kwargs)

    def __len__(self) -> int:
        return len(self._keys)

    def __getitem__(self, index: int):
        if index < 0 or index >= len(self._keys):
            raise IndexError(index)
        from . import _pymod as _core_mod
        
        # Reconstruct full URI preserving the original scheme
        key = self._keys[index]
        full_uri = f"{self._scheme}://{self._base}/{key}"
        
        # get() returns BytesView which supports zero-copy memoryview()
        bytesview = _core_mod.get(full_uri)

        # Return-style switch
        if self._return_type == "reader":
            # Reader needs bytes, not BytesView
            return _BytesReader(bytes(bytesview))
        if self._return_type == "bytes":
            return bytes(bytesview)

        # Tensor mode - use zero-copy memoryview from BytesView
        mv = bytesview.memoryview()  # TRUE zero-copy!
        
        if self._writable:
            # If writable is requested, we must copy to a bytearray
            buf = bytearray(mv)
            return torch.frombuffer(memoryview(buf), dtype=torch.uint8)
        
        # Zero-copy path: memoryview directly to tensor
        import warnings
        if self._suppress_nonwritable_warning:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        "The given buffer is not writable, and PyTorch does not "
                        "support non-writable tensors.*"
                    ),
                    category=UserWarning,
                )
                return torch.frombuffer(mv, dtype=torch.uint8)
        return torch.frombuffer(mv, dtype=torch.uint8)


# -----------------------------------------------------------------------------
# Deprecated aliases for backward compatibility
# -----------------------------------------------------------------------------
import warnings as _warnings

class S3IterableDataset(ObjectStoreIterableDataset):
    """
    DEPRECATED: Use ObjectStoreIterableDataset instead. 
    
    This alias exists for backward compatibility. ObjectStoreIterableDataset 
    supports all URI schemes (file://, s3://, az://, gs://, direct://).
    """
    def __init__(self, *args, **kwargs):
        _warnings.warn(
            "S3IterableDataset is deprecated. Use ObjectStoreIterableDataset instead - it supports all URI schemes.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


class S3MapDataset(ObjectStoreMapDataset):
    """
    DEPRECATED: Use ObjectStoreMapDataset instead.
    
    This alias exists for backward compatibility. ObjectStoreMapDataset
    supports all URI schemes (file://, s3://, az://, gs://, direct://).
    """
    def __init__(self, *args, **kwargs):
        _warnings.warn(
            "S3MapDataset is deprecated. Use ObjectStoreMapDataset instead - it supports all URI schemes.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(*args, **kwargs)


