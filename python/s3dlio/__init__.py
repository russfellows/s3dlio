from __future__ import annotations

"""
Top-level s3dlio package:
• Re-exports the Rust PyO3 module `_pymod` (installed top-level by maturin).
• Adds small helpers and the high-level loaders for Torch / JAX / TF.
• Supports all URI schemes: file://, s3://, az://, gs://, direct://
"""

import warnings as _warnings  # noqa: E402 -- must follow the bare module docstring above, which itself must follow `from __future__ import annotations`

# Version - automatically read from package metadata (set by pyproject.toml during build)
try:
    from importlib.metadata import version

    __version__ = version("s3dlio")
except Exception:
    # Fallback for development or if metadata not available
    __version__ = "0.9.14+dev"

from importlib import import_module  # noqa: E402 -- must follow the version-detection try/except above
from typing import List  # noqa: E402 -- avoid confusion with Rust-exported `list` symbol; must follow the version-detection try/except above

# The native module is imported FIRST, outside any try/except. Its initialisation applies
# the `.env` policy (S3DLIO_LOAD_DOTENV, issue #243) before anything else runs, and if it
# fails, the original error must reach the caller: a PyO3 module can be initialised only
# once per process, so a retry (as the torch block below would otherwise cause, by
# swallowing the ImportError and importing again further down) replaces the real
# failure with PyO3's "initialized once" message.
_core = import_module(__name__ + "._pymod")  # this is the compiled extension

# PyTorch integration (optional - only import if torch is available)
try:
    from .torch import (
        ObjectStoreMapDataset,
        ObjectStoreIterableDataset,
        S3MapDataset,
        S3IterableDataset,  # Deprecated aliases
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
# _native = import_module(__name__ + "._pymod")

# Expose it as a submodule (optional but handy)
# _sys.modules[__name__ + "._pymod"] = _native

# `_core` (the native module) was imported at the top of this file.

# Re-export all public symbols from the native module at package top-level
for _name in dir(_core):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_core, _name)

# Diagnostics for the borrowed-buffer completion barrier of put_bytes()/put_bytes_async():
# counters (quiesce_timeouts, queued, internal_errors, shutdown_attach_waits, ...). Underscore names
# are not re-exported by the loop above.
_put_export_stats = _core._put_export_stats
# Test-only hook (issue #294): occupies every async-runtime worker. It exists ONLY in a wheel built with the
# `test-hooks` cargo feature (CI test wheel); a release wheel has no such symbol and this stays None.
_hold_async_runtime = getattr(_core, "_hold_async_runtime", None)
# Hidden test hooks of the barrier (used only by python/tests/test_put_release_barrier.py).
_put_test_queue_export = _core._put_test_queue_export
_put_test_retain_next_clone = _core._put_test_retain_next_clone
# Hidden test hook (issue #304): the next guarded put closes the exit gate right after it spawned. Only in a
# `test-hooks` cargo feature build; a release wheel has no such symbol and this stays None.
_test_close_exit_gate_after_next_spawn = getattr(_core, "_test_close_exit_gate_after_next_spawn", None)
_put_test_retained = _core._put_test_retained
_put_test_release_retained = _core._put_test_release_retained

# del _sys, _name, import_module, _native
del _name


# ------------------------------------------------------------------
# 2) convenience helpers - generic versions that work with all URI schemes
# ------------------------------------------------------------------
def _after_scheme(uri: str) -> str:
    """The part of a URI after ``scheme://`` (the whole string if it has no scheme)."""
    return uri.split("://", 1)[1] if "://" in uri else uri


def list_keys(uri: str) -> List[str]:
    """
    Return object keys (relative paths) for the objects the URI selects.

    On ``s3://`` the key part of the URI selects an exact name, a directory or a ``*``
    glob, not a bare key prefix: ``s3://b/dir/`` lists what is in the directory, while
    ``s3://b/dir`` selects the object ``dir`` and the directory entry ``dir/`` (and never
    ``dir2``); ``s3://b/dir*`` selects by prefix. This lists one level (non-recursive).

    Works with all URI schemes: file://, s3://, az://, gs://, direct://

    Keys are used exactly as written, like every other s3dlio operation (issue #172):
    the prefix is the URI minus its scheme, and a returned key is the listed object's
    key minus that prefix, with no other normalisation. A leading ``/`` or an empty
    segment (``a//b``) in a key is part of the key, so ``uri + key`` (for a ``uri`` that
    ends in ``/``) is the URI of the object listed. One separator is dropped when the
    URI does not end in ``/``.

    Example:
        keys = list_keys("file:///data/prefix/")  # Returns ['file1.bin', 'file2.bin']
        keys = list_keys("s3://bucket/prefix/")   # Returns ['key1', 'key2']
    """
    full_uris = _core.list(uri)

    prefix = _after_scheme(uri)

    keys = []
    for full_uri in full_uris:
        path = _after_scheme(full_uri)

        if path.startswith(prefix):
            key = path[len(prefix) :]
            # The "/" between a directory-style prefix and the key is a separator, not
            # part of the key.
            if prefix and not prefix.endswith("/") and key.startswith("/"):
                key = key[1:]
        else:
            # Fallback (a listed entry that does not start with the listed prefix): just get the filename
            key = path.rsplit("/", 1)[-1] if "/" in path else path

        if key:
            keys.append(key)

    return keys


def list_full_uris(uri: str) -> List[str]:
    """
    Return fully-qualified URIs of the objects the URI selects.

    On ``s3://`` the key part selects an exact name (the object ``K`` and the directory
    entry ``K/``), a directory (``K/``) or a ``*`` / ``**`` glob; it is never a bare key
    prefix (``s3://b/foo`` does not select ``foobar``; use ``s3://b/foo*``). This lists
    one level (non-recursive).

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
    return stat(uri)  # noqa: F821 -- `stat` is injected into globals() by the `dir(_core)` re-export loop above, which runs before this function is ever called


# ------------------------------------------------------------------
# Deprecated helpers - kept for backward compatibility
# TODO: Remove all functions in this section entirely.
# ------------------------------------------------------------------

# TODO: Remove list_keys_from_s3 — use list_keys() instead.
# def list_keys_from_s3(uri: str) -> List[str]:
#     _warnings.warn(
#         "list_keys_from_s3() is deprecated. Use list_keys() instead.",
#         DeprecationWarning, stacklevel=2
#     )
#     return list_keys(uri)


# TODO: Remove list_uris — use list_full_uris() instead.
# def list_uris(uri: str) -> List[str]:
#     _warnings.warn(
#         "list_uris() is deprecated. Use list_full_uris() instead.",
#         DeprecationWarning, stacklevel=2
#     )
#     return list_full_uris(uri)


def get_by_key(bucket: str, key: str) -> bytes:
    """
    DEPRECATED: Use get_object(uri) instead - it supports all URI schemes.

    Fetch one S3 object by bucket/key and return its bytes.
    This function only works with S3.
    """
    _warnings.warn(
        "get_by_key() is deprecated and S3-only. Use get_object(uri) instead - it supports all URI schemes.",
        DeprecationWarning,
        stacklevel=2,
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
        stacklevel=2,
    )
    return stat(f"s3://{bucket}/{key}")  # noqa: F821 -- see the noqa above in stat_object(): `stat` comes from the `dir(_core)` re-export loop


# ------------------------------------------------------------------
# 3) high-level loaders (optional - only if frameworks are available)
# ------------------------------------------------------------------
# PyTorch loaders - imported at top with try/except

# JAX/TensorFlow loaders (optional)
try:
    from .jax_tf import (
        JaxIterable,
        make_tf_dataset,
        S3JaxIterable,  # Deprecated alias
    )

    _HAS_JAX_TF = True
except ImportError:
    _HAS_JAX_TF = False
    JaxIterable = None
    S3JaxIterable = None
    make_tf_dataset = None
