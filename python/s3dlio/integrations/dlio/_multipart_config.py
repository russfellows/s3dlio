"""
Shared multipart-upload configuration for the DLIO storage integrations
(s3dlio_storage.py, s3_torch_storage.py).

mlcommons/storage#715 (s3dlio issue #153 bug 3.1 / B10): both storage
backends hardcoded their single-PUT-vs-multipart size threshold (and part
size, and max-in-flight part count) as module-level constants with no way
to override them without editing source. That's wrong for two reasons:
different backends (MinIO, AWS S3, on-prem S3-compatible appliances) have
different optimal thresholds, and mlperf-storage benchmark configs need to
sweep these values without a code change per run.

This module centralizes the env-var contract so both storage classes read
identical, identically-named variables:

- `S3DLIO_MULTIPART_THRESHOLD_MB` reuses the SAME name already documented
  and read by DLIO's own `ObjStoreLibStorage`
  (`dlio_benchmark/storage/obj_store_lib.py`, see
  docs/Environment_Variables.md "Multipart Upload" section) -- a
  different file, in the DLIO_local_changes repo, solving the same
  problem for a different storage backend. Reusing the name means a
  single env var setting affects both backends consistently rather than
  needing two different variables for the same knob.
- `S3DLIO_MULTIPART_PART_SIZE_MB` and `S3DLIO_MULTIPART_MAX_IN_FLIGHT` are
  new names, following the same `S3DLIO_MULTIPART_*` prefix.
- `S3DLIO_DISABLE_MULTIPART` is an explicit switch (matches the
  `1/true/yes/on` case-insensitive boolean convention used by
  `S3DLIO_PUT_VERIFY` etc.) so callers don't have to know the
  "set the threshold above any real object size" trick to force
  single-PUT-only behavior.
"""

import os

# Defaults preserved from the pre-B10 hardcoded constants in
# s3dlio_storage.py. Match s3dlio's DEFAULT_S3_MULTIPART_THRESHOLD
# (src/constants.rs) -- NOT the same default as DLIO's own
# obj_store_lib.py (16 MiB there), since that's a different call site
# with its own historical default; both are independently overridable
# via the same env var name.
DEFAULT_MULTIPART_THRESHOLD_BYTES = 32 * 1024 * 1024  # 32 MiB
DEFAULT_MULTIPART_PART_SIZE_BYTES = 32 * 1024 * 1024  # 32 MiB per part
DEFAULT_MULTIPART_MAX_IN_FLIGHT = 8  # concurrent parts per object


def multipart_threshold_bytes() -> int:
    """S3DLIO_MULTIPART_THRESHOLD_MB (MiB) -> bytes.

    Unset / non-numeric / negative falls back to the default. `0` is
    valid and meaningful: it means "always use multipart" (matches the
    documented obj_store_lib.py contract for this same env var name).
    """
    raw = os.environ.get("S3DLIO_MULTIPART_THRESHOLD_MB")
    if raw is None:
        return DEFAULT_MULTIPART_THRESHOLD_BYTES
    try:
        mb = int(raw)
    except ValueError:
        return DEFAULT_MULTIPART_THRESHOLD_BYTES
    if mb < 0:
        return DEFAULT_MULTIPART_THRESHOLD_BYTES
    return mb * 1024 * 1024


def multipart_part_size_bytes() -> int:
    """S3DLIO_MULTIPART_PART_SIZE_MB (MiB) -> bytes.

    Unset / non-numeric / non-positive falls back to the default.
    Unlike the threshold var, `0` has no valid meaning here -- it would
    make the part-splitting loop never advance -- so it also falls back.
    """
    raw = os.environ.get("S3DLIO_MULTIPART_PART_SIZE_MB")
    if raw is None:
        return DEFAULT_MULTIPART_PART_SIZE_BYTES
    try:
        mb = int(raw)
    except ValueError:
        return DEFAULT_MULTIPART_PART_SIZE_BYTES
    if mb < 1:
        return DEFAULT_MULTIPART_PART_SIZE_BYTES
    return mb * 1024 * 1024


def multipart_max_in_flight() -> int:
    """S3DLIO_MULTIPART_MAX_IN_FLIGHT -> int.

    Unset / non-numeric / non-positive falls back to the default.
    """
    raw = os.environ.get("S3DLIO_MULTIPART_MAX_IN_FLIGHT")
    if raw is None:
        return DEFAULT_MULTIPART_MAX_IN_FLIGHT
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MULTIPART_MAX_IN_FLIGHT
    if value < 1:
        return DEFAULT_MULTIPART_MAX_IN_FLIGHT
    return value


def multipart_disabled() -> bool:
    """S3DLIO_DISABLE_MULTIPART: explicit switch to force put_bytes() for
    every write regardless of size. Matches the `1/true/yes/on`
    (case-insensitive) boolean convention used by S3DLIO_PUT_VERIFY etc.
    """
    raw = os.environ.get("S3DLIO_DISABLE_MULTIPART", "")
    return raw.strip().lower() in ("1", "true", "yes", "on")
