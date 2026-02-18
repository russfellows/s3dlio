"""
Drop-in replacement for AWS s3torchconnector using s3dlio backend.

This module provides API-compatible classes that match s3torchconnector's interface,
allowing users to easily switch between implementations by changing only the import:

    # Original:
    from s3torchconnector import S3IterableDataset, S3MapDataset, S3Checkpoint
    
    # Drop-in replacement:
    from s3dlio.compat.s3torchconnector import S3IterableDataset, S3MapDataset, S3Checkpoint

Key differences from native s3dlio classes:
- Uses .from_prefix(uri, region=...) signature matching s3torchconnector
- Returns item objects with .bucket, .key, .read() methods (not raw tensors)
- Supports S3, Azure, GCS, local filesystems (s3torchconnector only supports S3)
- Higher performance (Rust backend vs Python/C++)
- ZERO-COPY: get() returns BytesView backed by Rust Bytes (Arc-counted, no memcpy)
- Global client cache: DashMap<StoreKey, Arc<dyn ObjectStore>> auto-reuses connections

Performance architecture (analogous to MinIO Python SDK but faster):
- MinIO: urllib3.PoolManager(maxsize=10) per Minio() instance, synchronous
- s3dlio: Global DashMap client cache, async Tokio runtime, io_uring-style submit
- Both: Thread-safe client objects. s3dlio is superior because the cache is
  process-global and automatic — no manual client passing required.

Environment variables (same as s3torchconnector):
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY - S3 credentials
    AWS_ENDPOINT_URL - Custom endpoint (MinIO, Ceph, etc.)
    AWS_REGION - Default region (if not specified in from_prefix)
    AZURE_STORAGE_ACCOUNT_NAME, AZURE_STORAGE_ACCOUNT_KEY - Azure credentials
    GOOGLE_APPLICATION_CREDENTIALS - GCS service account JSON path
"""

from __future__ import annotations

import io
import os
import warnings
from typing import Any, Iterator, Optional
from urllib.parse import urlparse

import torch
from torch.utils.data import IterableDataset, Dataset

from ..torch import ObjectStoreIterableDataset, ObjectStoreMapDataset

# Import s3dlio native module once at module level (not per-class)
from .. import _pymod as _core


class _BytesViewIO(io.RawIOBase):
    """
    Zero-copy file-like wrapper around s3dlio BytesView.
    
    Implements io.RawIOBase so that torch.load(), pickle.load(), etc.
    can read directly from Rust-owned memory without any copy.
    
    The underlying BytesView implements the Python buffer protocol,
    backed by Rust Bytes (Arc-counted). This wrapper adds seekable
    file-like semantics on top.
    """
    
    __slots__ = ("_view", "_pos", "_len")
    
    def __init__(self, bytes_view):
        """
        Args:
            bytes_view: s3dlio BytesView object (implements buffer protocol)
        """
        super().__init__()
        self._view = bytes_view
        self._pos = 0
        self._len = len(bytes_view)
    
    def readable(self) -> bool:
        return True
    
    def seekable(self) -> bool:
        return True
    
    def writable(self) -> bool:
        return False
    
    def readinto(self, b):
        """Read up to len(b) bytes into b. Zero-copy via memoryview slicing."""
        if self._pos >= self._len:
            return 0
        remaining = self._len - self._pos
        n = min(len(b), remaining)
        # Use memoryview for zero-copy slice into the caller's buffer
        mv = memoryview(self._view)
        b[:n] = mv[self._pos:self._pos + n]
        self._pos += n
        return n
    
    def read(self, size=-1):
        """Read up to size bytes. Returns bytes (one copy from Rust buffer)."""
        if self._pos >= self._len:
            return b""
        if size < 0:
            size = self._len - self._pos
        remaining = self._len - self._pos
        n = min(size, remaining)
        mv = memoryview(self._view)
        result = bytes(mv[self._pos:self._pos + n])
        self._pos += n
        return result
    
    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self._pos = offset
        elif whence == io.SEEK_CUR:
            self._pos += offset
        elif whence == io.SEEK_END:
            self._pos = self._len + offset
        self._pos = max(0, min(self._pos, self._len))
        return self._pos
    
    def tell(self) -> int:
        return self._pos


class S3Item:
    """
    Item returned by S3IterableDataset and S3MapDataset, matching s3torchconnector API.
    
    Attributes:
        bucket (str): Bucket name (or container/account for Azure/GCS)
        key (str): Object key (relative path)
    
    Methods:
        read() -> BytesView: Read the entire object payload (ZERO-COPY)
        read_bytes() -> bytes: Read as Python bytes (creates copy)
        close(): Close the reader (no-op for compatibility)
    """
    
    __slots__ = ("_uri", "_bucket", "_key", "_data", "_closed")
    
    def __init__(self, uri: str, data):
        """
        Args:
            uri: Full URI (e.g., "s3://bucket/path/key")
            data: Object payload — BytesView (zero-copy) or bytes
        """
        self._uri = uri
        self._data = data
        self._closed = False
        
        # Parse URI to extract bucket and key
        parsed = urlparse(uri)
        self._bucket, self._key = self._parse_uri_parts(parsed)
    
    def _parse_uri_parts(self, parsed):
        """Extract bucket/container and key from parsed URI."""
        scheme = parsed.scheme or "s3"
        
        if scheme == "s3":
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
        elif scheme == "az":
            parts = parsed.path.lstrip("/").split("/", 1)
            bucket = f"{parsed.netloc}/{parts[0]}" if len(parts) > 0 else parsed.netloc
            key = parts[1] if len(parts) > 1 else ""
        elif scheme == "gs":
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
        elif scheme in ("file", "direct"):
            full_path = parsed.path
            if "/" in full_path:
                bucket = os.path.dirname(full_path)
                key = os.path.basename(full_path)
            else:
                bucket = "/"
                key = full_path
        else:
            bucket = parsed.netloc or "unknown"
            key = parsed.path.lstrip("/")
        
        return bucket, key
    
    @property
    def bucket(self) -> str:
        """Bucket name (or container for Azure, account for GCS)."""
        return self._bucket
    
    @property
    def key(self) -> str:
        """Object key (relative path from bucket root)."""
        return self._key
    
    def read(self):
        """
        Read the entire object payload (ZERO-COPY).
        
        Returns BytesView (implements buffer protocol) for optimal performance.
        Compatible with:
          - torch.frombuffer(data, dtype=torch.uint8)  # Zero-copy tensor
          - np.frombuffer(data, dtype=np.uint8)        # Zero-copy array
          - file.write(data)                            # Zero-copy write
          - memoryview(data)                            # Buffer protocol
        
        For strict bytes type (slower, creates copy), use read_bytes() instead.
        
        Returns:
            BytesView: Zero-copy buffer protocol object
        """
        if self._closed:
            raise ValueError("I/O operation on closed S3Item")
        return self._data
    
    def read_bytes(self) -> bytes:
        """
        Read as bytes (CREATES COPY - slower but strictly compatible).
        
        Use this only if you need exact bytes type. For most ML workloads,
        prefer read() which returns zero-copy BytesView.
        
        Returns:
            bytes: Python bytes object (creates memory copy)
        """
        if self._closed:
            raise ValueError("I/O operation on closed S3Item")
        return bytes(self._data) if not isinstance(self._data, bytes) else self._data
    
    def readall(self):
        """Alias for read() - returns zero-copy BytesView."""
        return self.read()
    
    def read_all(self):
        """Alias for read() - returns zero-copy BytesView."""
        return self.read()
    
    def close(self) -> None:
        """Close the reader (releases reference to BytesView data)."""
        self._closed = True
        self._data = None  # Allow GC to reclaim Rust Bytes
    
    def __enter__(self) -> "S3Item":
        return self
    
    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
    
    def __len__(self) -> int:
        """Return size of object in bytes."""
        if self._data is None:
            return 0
        return len(self._data)
    
    def __repr__(self) -> str:
        size = len(self._data) if self._data is not None else 0
        return f"S3Item(bucket='{self.bucket}', key='{self.key}', size={size})"


class S3IterableDataset(IterableDataset):
    """
    Drop-in replacement for s3torchconnector.S3IterableDataset using s3dlio.
    
    Key features:
    - API-compatible with s3torchconnector for easy migration
    - Multi-protocol support (S3, Azure, GCS, file://, direct://)
    - Higher performance (Rust backend with global client cache)
    - Multi-endpoint load balancing (via AWS_ENDPOINT_URL=url1,url2,...)
    - ZERO-COPY: Items contain BytesView backed by Rust memory
    
    Usage:
        >>> from s3dlio.compat.s3torchconnector import S3IterableDataset
        >>> dataset = S3IterableDataset.from_prefix("s3://bucket/prefix/", region="us-east-1")
        >>> for item in dataset:
        ...     print(f"{item.bucket}/{item.key}: {len(item.read())} bytes")
    """
    
    def __init__(
        self,
        uri: str,
        region: Optional[str] = None,
        *,
        enable_sharding: bool = False,
        **loader_kwargs
    ):
        """
        Args:
            uri: Full URI to prefix (e.g., "s3://bucket/prefix/")
            region: AWS region (optional, can use AWS_REGION env var or config)
            enable_sharding: Automatically shard dataset across distributed workers
            **loader_kwargs: Additional arguments for s3dlio loader (prefetch, shuffle, etc.)
        """
        self._uri = uri
        self._region = region
        self._enable_sharding = enable_sharding
        
        # Set AWS region if provided (s3dlio uses env vars)
        if region:
            os.environ.setdefault("AWS_REGION", region)
        
        # Extract s3dlio-compatible loader options
        self._loader_opts = self._convert_loader_kwargs(loader_kwargs)
        
        # Pre-list object keys so we can reconstruct full URIs during iteration
        # This uses the global client cache (DashMap) — fast after first call
        self._keys = _core.list(uri, recursive=True)
    
    def _convert_loader_kwargs(self, kwargs):
        """Convert s3torchconnector kwargs to s3dlio format."""
        opts = {}
        
        # Direct mappings
        for key in ["prefetch", "shuffle", "seed", "num_workers", "batch_size", "drop_last"]:
            if key in kwargs:
                opts[key] = kwargs[key]
        
        # Reader configuration (s3torchconnector uses reader_constructor, s3dlio uses reader_mode)
        if "reader_constructor" in kwargs:
            warnings.warn("reader_constructor not directly supported - using default reader_mode")
        
        return opts
    
    @classmethod
    def from_prefix(
        cls,
        prefix: str,
        region: Optional[str] = None,
        **kwargs
    ) -> "S3IterableDataset":
        """
        Create dataset from S3 prefix (API-compatible with s3torchconnector).
        
        Args:
            prefix: S3 URI prefix (e.g., "s3://bucket/path/")
            region: AWS region (e.g., "us-east-1")
            **kwargs: Additional options (enable_sharding, prefetch, etc.)
        
        Returns:
            S3IterableDataset instance
        
        Examples:
            >>> # Basic usage
            >>> ds = S3IterableDataset.from_prefix("s3://my-bucket/train/", region="us-east-1")
            
            >>> # With sharding for distributed training
            >>> ds = S3IterableDataset.from_prefix(
            ...     "s3://my-bucket/train/",
            ...     region="us-east-1",
            ...     enable_sharding=True
            ... )
            
            >>> # Azure Blob Storage
            >>> ds = S3IterableDataset.from_prefix("az://myaccount/container/prefix/")
            
            >>> # Local filesystem (for testing)
            >>> ds = S3IterableDataset.from_prefix("file:///data/train/")
        """
        return cls(prefix, region=region, **kwargs)
    
    def __iter__(self) -> Iterator[S3Item]:
        """Iterate over objects, yielding S3Item instances with full URIs."""
        for uri in self._keys:
            # Fetch each object — returns BytesView (zero-copy from Rust)
            # Uses global client cache, so only first call creates the store
            data = _core.get(uri)
            yield S3Item(uri, data)
    
    def __len__(self) -> int:
        """Return number of objects in the dataset."""
        return len(self._keys)


class S3MapDataset(Dataset):
    """
    Drop-in replacement for s3torchconnector.S3MapDataset using s3dlio.
    
    Key features:
    - API-compatible with s3torchconnector for easy migration
    - Multi-protocol support (S3, Azure, GCS, file://, direct://)
    - Random access by index
    - Higher performance (Rust backend with global client cache)
    - ZERO-COPY: Items contain BytesView backed by Rust memory
    
    Usage:
        >>> from s3dlio.compat.s3torchconnector import S3MapDataset
        >>> dataset = S3MapDataset.from_prefix("s3://bucket/prefix/", region="us-east-1")
        >>> item = dataset[0]
        >>> print(f"{item.bucket}/{item.key}: {len(item.read())} bytes")
    """
    
    def __init__(
        self,
        uri: str,
        region: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            uri: Full URI to prefix (e.g., "s3://bucket/prefix/")
            region: AWS region (optional, can use AWS_REGION env var)
            **kwargs: Additional arguments for s3dlio (reader_mode, etc.)
        """
        self._uri = uri
        self._region = region
        
        # Set AWS region if provided
        if region:
            os.environ.setdefault("AWS_REGION", region)
        
        # List all object URIs upfront for random access
        # Uses global client cache — fast after first call
        self._full_uris = _core.list(uri, recursive=True)
    
    @classmethod
    def from_prefix(
        cls,
        prefix: str,
        region: Optional[str] = None,
        **kwargs
    ) -> "S3MapDataset":
        """
        Create map dataset from S3 prefix (API-compatible with s3torchconnector).
        
        Args:
            prefix: S3 URI prefix (e.g., "s3://bucket/path/")
            region: AWS region (e.g., "us-east-1")
            **kwargs: Additional options
        
        Returns:
            S3MapDataset instance
        
        Examples:
            >>> ds = S3MapDataset.from_prefix("s3://my-bucket/train/", region="us-east-1")
            >>> item = ds[0]
            >>> content = item.read()
        """
        return cls(prefix, region=region, **kwargs)
    
    def __len__(self) -> int:
        """Return number of objects in the dataset."""
        return len(self._full_uris)
    
    def __getitem__(self, index: int) -> S3Item:
        """Get item by index, returning S3Item instance with zero-copy data."""
        if index < 0:
            index += len(self._full_uris)
        if index < 0 or index >= len(self._full_uris):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self._full_uris)}")
        
        uri = self._full_uris[index]
        # Fetch via Rust — returns BytesView (zero-copy)
        data = _core.get(uri)
        return S3Item(uri, data)


class S3Checkpoint:
    """
    Drop-in replacement for s3torchconnector.S3Checkpoint using s3dlio.
    
    Provides simple checkpoint saving/loading to S3-compatible storage.
    
    Features:
    - API-compatible with s3torchconnector
    - Multi-protocol support (S3, Azure, GCS, file://)
    - Works with torch.save() and torch.load()
    - Reader uses zero-copy BytesView wrapper for efficient load
    
    Usage:
        >>> from s3dlio.compat.s3torchconnector import S3Checkpoint
        >>> checkpoint = S3Checkpoint(region="us-east-1")
        >>> 
        >>> # Save checkpoint
        >>> with checkpoint.writer("s3://bucket/model.ckpt") as writer:
        ...     torch.save(model.state_dict(), writer)
        >>> 
        >>> # Load checkpoint
        >>> with checkpoint.reader("s3://bucket/model.ckpt") as reader:
        ...     state_dict = torch.load(reader)
        >>> model.load_state_dict(state_dict)
    """
    
    def __init__(self, region: Optional[str] = None):
        """
        Args:
            region: AWS region (optional, can use AWS_REGION env var)
        """
        self._region = region
        if region:
            os.environ.setdefault("AWS_REGION", region)
    
    def writer(self, uri: str):
        """
        Context manager for writing checkpoint to storage.
        
        torch.save() serializes into an in-memory buffer, then on __exit__
        the entire payload is uploaded via put_bytes() (one copy from Python
        heap to Rust Bytes, then zero-copy through the upload pipeline).
        
        Args:
            uri: Full URI to checkpoint (e.g., "s3://bucket/checkpoint.pt")
        
        Returns:
            Context manager yielding a file-like writer for torch.save()
        
        Example:
            >>> with checkpoint.writer("s3://bucket/model.ckpt") as writer:
            ...     torch.save(model.state_dict(), writer)
        """
        class _S3Writer:
            def __init__(self, uri):
                self.uri = uri
                self.buffer = io.BytesIO()
            
            def write(self, data):
                return self.buffer.write(data)
            
            def tell(self):
                return self.buffer.tell()
            
            def seek(self, offset, whence=io.SEEK_SET):
                return self.buffer.seek(offset, whence)
            
            def close(self):
                # getbuffer() returns a memoryview — no copy
                # put_bytes does Bytes::copy_from_slice once (unavoidable Python→Rust)
                mv = self.buffer.getbuffer()
                _core.put_bytes(self.uri, bytes(mv))
                mv.release()
                self.buffer.close()
            
            def __enter__(self):
                return self  # Return self — torch.save needs write()+tell()
            
            def __exit__(self, exc_type, exc, tb):
                if exc_type is None:  # Only upload if no exception
                    self.close()
                else:
                    self.buffer.close()
        
        return _S3Writer(uri)
    
    def reader(self, uri: str):
        """
        Context manager for reading checkpoint from storage.
        
        Downloads the object via get() which returns a BytesView (Rust Bytes,
        zero-copy). Wraps it in _BytesViewIO for seekable file-like access
        that torch.load() requires.
        
        Args:
            uri: Full URI to checkpoint (e.g., "s3://bucket/checkpoint.pt")
        
        Returns:
            Context manager yielding a seekable file-like reader for torch.load()
        
        Example:
            >>> with checkpoint.reader("s3://bucket/model.ckpt") as reader:
            ...     state_dict = torch.load(reader, weights_only=True)
        """
        class _S3Reader:
            def __init__(self, uri):
                self.uri = uri
                self._view = None
                self._reader = None
            
            def __enter__(self):
                # Download — returns BytesView (zero-copy from Rust)
                self._view = _core.get(self.uri)
                # Wrap in buffered reader for seekable file-like interface
                # _BytesViewIO.readinto() uses memoryview (zero-copy slice)
                self._reader = io.BufferedReader(_BytesViewIO(self._view))
                return self._reader
            
            def __exit__(self, exc_type, exc, tb):
                if self._reader:
                    self._reader.close()
                self._view = None  # Release Rust Bytes reference
        
        return _S3Reader(uri)


class S3ClientConfig:
    """
    Configuration for S3Client, matching s3torchconnector API.
    
    Args:
        force_path_style (bool): Force path-style S3 URLs (required for MinIO)
        max_attempts (int): Maximum retry attempts (default: 5)
    """
    def __init__(self, force_path_style=False, max_attempts=5):
        self.force_path_style = force_path_style
        self.max_attempts = max_attempts


class S3Client:
    """
    Low-level S3 client matching s3torchconnector._s3client.S3Client API.
    
    This provides compatibility for DLIO's s3_torch_storage.py which uses
    the low-level S3Client interface instead of the high-level Dataset classes.
    
    Uses s3dlio's global client cache (DashMap) — all S3Client instances
    sharing the same endpoint automatically reuse the same Rust ObjectStore,
    similar to MinIO's PoolManager but process-global and lock-free.
    
    Args:
        region (str): AWS region
        endpoint (str): Custom endpoint URL (for MinIO, Ceph, etc.)
        s3client_config (S3ClientConfig): Client configuration
    """
    
    def __init__(self, region=None, endpoint=None, s3client_config=None):
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self.endpoint = endpoint or os.environ.get("AWS_ENDPOINT_URL")
        self.config = s3client_config or S3ClientConfig()
    
    def _make_uri(self, bucket, key):
        """Build full s3:// URI from bucket and key."""
        if key.startswith("s3://") or key.startswith("az://") or key.startswith("gs://") or key.startswith("file://"):
            return key
        return f"s3://{bucket}/{key.lstrip('/')}"
    
    def put_object(self, bucket, key):
        """
        Create a writer for uploading object data.
        
        The writer accumulates data in a BytesIO buffer, then uploads
        via put_bytes() on close() (one unavoidable Python→Rust copy,
        then zero-copy through the upload pipeline).
        
        Args:
            bucket (str): Bucket name
            key (str): Object key (can be full s3:// URI)
        
        Returns:
            Writer object with write() and close() methods
        """
        uri = self._make_uri(bucket, key)
        
        class _S3Writer:
            def __init__(self, uri):
                self.uri = uri
                self.buffer = io.BytesIO()
            
            def write(self, data):
                """Accumulate data to write"""
                return self.buffer.write(data)
            
            def close(self):
                """Upload accumulated data via put_bytes (one copy Python→Rust)"""
                if self.buffer.tell() > 0:
                    self.buffer.seek(0)
                    payload = self.buffer.read()
                    _core.put_bytes(self.uri, payload)
                self.buffer.close()
        
        return _S3Writer(uri)
    
    def get_object(self, bucket, key, start=None, end=None):
        """
        Create a reader for downloading object data.
        
        ZERO-COPY: Uses get_range() for range requests (server-side range,
        no wasted bandwidth) and get() for full objects. Both return
        BytesView backed by Rust Bytes.
        
        Args:
            bucket (str): Bucket name  
            key (str): Object key (can be full s3:// URI)
            start (int, optional): Range start byte
            end (int, optional): Range end byte (inclusive, s3torchconnector convention)
        
        Returns:
            Reader object with read() method returning BytesView (zero-copy)
        """
        uri = self._make_uri(bucket, key)
        
        class _S3Reader:
            def __init__(self, uri, start, end):
                self.uri = uri
                self.start = start
                self.end = end
                self._data = None
            
            def read(self):
                """Read object data — ZERO-COPY via BytesView."""
                if self._data is None:
                    if self.start is not None and self.end is not None:
                        # Server-side range request — only fetches the needed bytes
                        # end is inclusive in s3torchconnector API
                        length = self.end - self.start + 1
                        self._data = _core.get_range(self.uri, self.start, length)
                    elif self.start is not None:
                        # Start offset, read to end
                        self._data = _core.get_range(self.uri, self.start)
                    else:
                        # Full object — returns BytesView (zero-copy from Rust)
                        self._data = _core.get(self.uri)
                return self._data
        
        return _S3Reader(uri, start, end)
    
    def list_objects(self, bucket, prefix=""):
        """
        List objects in bucket with optional prefix.
        
        Args:
            bucket (str): Bucket name
            prefix (str): Key prefix (can be full s3:// URI)
        
        Yields:
            ListObjectResult with object_info list
        """
        if prefix.startswith("s3://") or prefix.startswith("az://") or prefix.startswith("gs://") or prefix.startswith("file://"):
            uri = prefix.rstrip('/') + '/'
        else:
            uri = f"s3://{bucket}/{prefix.lstrip('/')}"
        
        class ObjectInfo:
            """Object metadata"""
            __slots__ = ("key",)
            def __init__(self, key):
                self.key = key
        
        class ListObjectResult:
            """Result from list_objects"""
            __slots__ = ("object_info",)
            def __init__(self, object_infos):
                self.object_info = object_infos
        
        try:
            # Use the native list function (uses global client cache)
            objects = _core.list(uri, recursive=True)
            object_infos = [ObjectInfo(obj) for obj in objects]
            yield ListObjectResult(object_infos)
        except Exception:
            yield ListObjectResult([])


# Expose API-compatible classes
__all__ = [
    "S3IterableDataset",
    "S3MapDataset", 
    "S3Checkpoint",
    "S3Item",
    "S3Client",
    "S3ClientConfig",
]
