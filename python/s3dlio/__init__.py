from __future__ import annotations

"""
Top-level s3dlio package:
• Re-exports the Rust PyO3 module `_pymod` (installed top-level by maturin).
• Adds small helpers and the high-level loaders for Torch / JAX / TF.
• Supports all URI schemes: file://, s3://, az://, gs://, direct://
"""

import warnings as _warnings

# Version - automatically read from package metadata (set by pyproject.toml during build)
try:
    from importlib.metadata import version
    __version__ = version("s3dlio")
except Exception:
    # Fallback for development or if metadata not available
    __version__ = "0.9.14+dev"

from importlib import import_module
import sys as _sys
from typing import List  # avoid confusion with Rust-exported `list` symbol

# PyTorch integration (optional - only import if torch is available)
try:
    from .torch import (
        ObjectStoreMapDataset, ObjectStoreIterableDataset,
        S3MapDataset, S3IterableDataset  # Deprecated aliases
    )
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    ObjectStoreMapDataset = None
    ObjectStoreIterableDataset = None
    S3MapDataset = None
    S3IterableDataset = None

# ------------------------------------------------------------------
# 1) Import the native module and re-export *public* names
#     Import the native extension from inside this package
# ------------------------------------------------------------------

#
# We don't need these now, I think ?
#
#_native = import_module(__name__ + "._pymod")

# Expose it as a submodule (optional but handy)
#_sys.modules[__name__ + "._pymod"] = _native

# Keep a stable handle to the native module.
_core = import_module(__name__ + "._pymod")   # this is the compiled extension

# Re-export all public symbols from the native module at package top-level
for _name in dir(_core):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_core, _name)

#del _sys, _name, import_module, _native
del _name

# ------------------------------------------------------------------
# 2) convenience helpers - generic versions that work with all URI schemes
# ------------------------------------------------------------------
def list_keys(uri: str) -> List[str]:
    """
    Return object keys (relative paths) under the given URI prefix.
    
    Works with all URI schemes: file://, s3://, az://, gs://, direct://
    
    Example:
        keys = list_keys("file:///data/prefix/")  # Returns ['file1.bin', 'file2.bin']
        keys = list_keys("s3://bucket/prefix/")   # Returns ['key1', 'key2']
    """
    full_uris = _core.list(uri)
    
    # Extract the base path to strip from results
    if "://" in uri:
        scheme, rest = uri.split("://", 1)
        # For file://, the path starts with /; for cloud, it's bucket/path
        if scheme in ('file', 'direct'):
            base = rest.rstrip('/')
        else:
            base = rest.rstrip('/')
    else:
        base = uri.rstrip('/')
    
    keys = []
    for full_uri in full_uris:
        if "://" in full_uri:
            _, path = full_uri.split("://", 1)
        else:
            path = full_uri
        
        # Strip the base path to get relative key
        path = path.lstrip('/')
        base_stripped = base.lstrip('/')
        if path.startswith(base_stripped):
            key = path[len(base_stripped):].lstrip('/')
        else:
            # Fallback: just get the filename
            key = path.rsplit('/', 1)[-1] if '/' in path else path
        
        if key:
            keys.append(key)
    
    return keys


def list_full_uris(uri: str) -> List[str]:
    """
    Return fully-qualified URIs under the given prefix.
    
    Works with all URI schemes: file://, s3://, az://, gs://, direct://
    Preserves the original scheme in the returned URIs.
    
    Example:
        uris = list_full_uris("file:///data/prefix/")  # Returns file:// URIs
        uris = list_full_uris("s3://bucket/prefix/")   # Returns s3:// URIs
    """
    return _core.list(uri)


def get_object(uri: str) -> bytes:
    """
    Fetch one object by full URI and return its bytes.
    
    Works with all URI schemes: file://, s3://, az://, gs://, direct://
    
    Example:
        data = get_object("file:///data/file.bin")
        data = get_object("s3://bucket/key")
    """
    return _core.get(uri)


def stat_object(uri: str) -> dict:
    """
    Return a Python dict of object metadata for the given URI.
    
    Works with all URI schemes: file://, s3://, az://, gs://, direct://
    
    Example:
        meta = stat_object("file:///data/file.bin")
        meta = stat_object("s3://bucket/key")
    """
    return stat(uri)


# ------------------------------------------------------------------
# Deprecated helpers - kept for backward compatibility
# ------------------------------------------------------------------
def list_keys_from_s3(uri: str) -> List[str]:
    """
    DEPRECATED: Use list_keys() instead - it supports all URI schemes.
    
    Return object keys (strings) under the given URI prefix.
    """
    _warnings.warn(
        "list_keys_from_s3() is deprecated. Use list_keys() instead - it supports all URI schemes.",
        DeprecationWarning,
        stacklevel=2
    )
    return list_keys(uri)


def list_uris(uri: str) -> List[str]:
    """
    DEPRECATED: Use list_full_uris() instead - it supports all URI schemes.
    
    Return fully-qualified URIs under the given prefix.
    """
    _warnings.warn(
        "list_uris() is deprecated. Use list_full_uris() instead - it supports all URI schemes.",
        DeprecationWarning,
        stacklevel=2
    )
    return list_full_uris(uri)


def get_by_key(bucket: str, key: str) -> bytes:
    """
    DEPRECATED: Use get_object(uri) instead - it supports all URI schemes.
    
    Fetch one S3 object by bucket/key and return its bytes.
    This function only works with S3.
    """
    _warnings.warn(
        "get_by_key() is deprecated and S3-only. Use get_object(uri) instead - it supports all URI schemes.",
        DeprecationWarning,
        stacklevel=2
    )
    return _core.get(f"s3://{bucket}/{key}")


def stat_key(bucket: str, key: str) -> dict:
    """
    DEPRECATED: Use stat_object(uri) instead - it supports all URI schemes.
    
    Return a Python dict of S3 object metadata for bucket/key.
    This function only works with S3.
    """
    _warnings.warn(
        "stat_key() is deprecated and S3-only. Use stat_object(uri) instead - it supports all URI schemes.",
        DeprecationWarning,
        stacklevel=2
    )
    return stat(f"s3://{bucket}/{key}")

# ------------------------------------------------------------------
# 3) high-level loaders (optional - only if frameworks are available)
# ------------------------------------------------------------------
# PyTorch loaders - imported at top with try/except

# JAX/TensorFlow loaders (optional)
try:
    from .jax_tf import (
        JaxIterable, make_tf_dataset,
        S3JaxIterable  # Deprecated alias
    )
    _HAS_JAX_TF = True
except ImportError:
    _HAS_JAX_TF = False
    JaxIterable = None
    S3JaxIterable = None
    make_tf_dataset = None

