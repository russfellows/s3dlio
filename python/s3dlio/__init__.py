from __future__ import annotations

"""
Top-level s3dlio package:
• Re-exports the Rust PyO3 module `_pymod` (installed top-level by maturin).
• Adds small helpers and the high-level loaders for Torch / JAX / TF.
"""

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
    from .torch import S3MapDataset, S3IterableDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
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
# 2) convenience helpers
# ------------------------------------------------------------------
def list_keys_from_s3(uri: str) -> List[str]:
    """Return S3 keys (strings) under the given s3://bucket/prefix/."""
    return _core.list(uri)
    #return _native.list(uri)

def list_uris(uri: str) -> List[str]:
    """Return fully-qualified s3:// URIs under the given prefix."""
    bucket, _ = uri.replace("s3://", "", 1).split("/", 1)
    return [f"s3://{bucket}/{k}" for k in list_keys_from_s3(uri)]

def get_by_key(bucket: str, key: str) -> bytes:
    """Fetch one object by bucket/key and return its bytes."""
    return _core.get(f"s3://{bucket}/{key}")
    #return _native.get(f"s3://{bucket}/{key}")

def stat_key(bucket: str, key: str) -> dict:
    """Return a Python dict of S3 object metadata for bucket/key."""
    return stat(f"s3://{bucket}/{key}")

# ------------------------------------------------------------------
# 3) high-level loaders (optional - only if frameworks are available)
# ------------------------------------------------------------------
# PyTorch loaders - imported at top with try/except (S3MapDataset, S3IterableDataset)

# JAX/TensorFlow loaders (optional)
try:
    from .jax_tf import S3JaxIterable, make_tf_dataset  # noqa: E402
    _HAS_JAX_TF = True
except ImportError:
    _HAS_JAX_TF = False
    S3JaxIterable = None
    make_tf_dataset = None

