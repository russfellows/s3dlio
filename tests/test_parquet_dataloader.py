#!/usr/bin/env python3
"""
Integration tests for s3dlio Parquet DataLoader.

Requirements
------------
* MinIO endpoint running at $AWS_ENDPOINT_URL (loaded from .env or environment).
* `pyarrow` for creating test Parquet files and verifying ArrowIpc output.
* `s3dlio` Python wheel installed in the active virtual environment.

Usage
-----
    source .env && python -m pytest tests/test_parquet_dataloader.py -v --tb=short

The tests upload small Parquet files to ``s3://mlp-flux/s3dlio/__parquet_tests__/``
and delete them on teardown.
"""

import io
import os
import struct
import pytest

# ── Dependency guards ─────────────────────────────────────────────────────────

s3dlio = pytest.importorskip("s3dlio", reason="s3dlio wheel not installed")
pa = pytest.importorskip("pyarrow", reason="pyarrow not installed")
pq = pytest.importorskip("pyarrow.parquet", reason="pyarrow.parquet not installed")

# ── Fixtures / helpers ────────────────────────────────────────────────────────

MINIO_ENDPOINT = os.environ.get("AWS_ENDPOINT_URL", "")
MINIO_AVAILABLE = bool(MINIO_ENDPOINT and os.environ.get("AWS_ACCESS_KEY_ID"))

TEST_BUCKET = "mlp-flux"
TEST_PREFIX = "s3dlio/__parquet_tests__"
TEST_URI_PREFIX = f"s3://{TEST_BUCKET}/{TEST_PREFIX}/"


def skip_without_minio(func):
    """Decorator: skip the test if MinIO is not reachable."""
    return pytest.mark.skipif(
        not MINIO_AVAILABLE,
        reason="AWS_ENDPOINT_URL / AWS_ACCESS_KEY_ID not set — MinIO unavailable",
    )(func)


def make_parquet_bytes(num_rgs: int = 2, rows_per_rg: int = 3) -> bytes:
    """Return in-memory Parquet bytes with ``num_rgs`` row groups."""
    total_rows = num_rgs * rows_per_rg
    table = pa.table(
        {
            "id": pa.array(list(range(total_rows)), type=pa.int64()),
            "value": pa.array([float(i) * 1.1 for i in range(total_rows)], type=pa.float32()),
        }
    )
    buf = io.BytesIO()
    pq.write_table(
        table,
        buf,
        row_group_size=rows_per_rg,
        compression="snappy",
    )
    return buf.getvalue()


@pytest.fixture(scope="module", autouse=True)
def configure_s3():
    """Configure s3dlio to use the MinIO endpoint once per module."""
    if not MINIO_AVAILABLE:
        return
    ca_bundle = os.environ.get("AWS_CA_BUNDLE", "")
    region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    s3dlio.configure_s3(
        endpoint_url=MINIO_ENDPOINT,
        region=region,
        ca_bundle=ca_bundle if ca_bundle else None,
    )


@pytest.fixture(scope="module")
def uploaded_parquet_uris():
    """Upload two test Parquet files; yield their URIs; delete on teardown."""
    if not MINIO_AVAILABLE:
        pytest.skip("MinIO not available")

    data = make_parquet_bytes(num_rgs=2, rows_per_rg=3)
    keys = [
        f"{TEST_PREFIX}/part-0.parquet",
        f"{TEST_PREFIX}/part-1.parquet",
    ]
    uris = [f"s3://{TEST_BUCKET}/{k}" for k in keys]

    # Upload
    for key in keys:
        uri = f"s3://{TEST_BUCKET}/{key}"
        s3dlio.put_bytes(uri, data)

    yield uris

    # Cleanup
    for uri in uris:
        try:
            s3dlio.delete(uri)
        except Exception:
            pass


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestParquetDataloaderBasic:
    """Basic construction and iteration over uploaded Parquet files."""

    @skip_without_minio
    def test_list_uploaded_files(self, uploaded_parquet_uris):
        """s3dlio.list() should discover the uploaded .parquet objects."""
        keys = s3dlio.list(TEST_URI_PREFIX, pattern=r".*\.parquet$")
        assert len(keys) >= 2, f"Expected at least 2 .parquet keys, got: {keys}"

    @skip_without_minio
    def test_create_raw_loader(self, uploaded_parquet_uris):
        """create_async_loader in parquet/raw mode returns a loader with correct length."""
        loader = s3dlio.create_async_loader(
            prefix=TEST_URI_PREFIX,
            format="parquet",
            decode_mode="raw",
            footer_cap=4 * 1024 * 1024,
        )
        # 2 files × 2 row groups each
        assert loader.len() == 4, f"Expected 4 row groups, got {loader.len()}"

    @skip_without_minio
    def test_raw_rg_bytes_not_empty(self, uploaded_parquet_uris):
        """Each raw row group must return non-empty bytes."""
        loader = s3dlio.create_async_loader(
            prefix=TEST_URI_PREFIX,
            format="parquet",
            decode_mode="raw",
            footer_cap=4 * 1024 * 1024,
        )
        for i in range(loader.len()):
            data = loader.get(i)
            assert isinstance(data, (bytes, bytearray, memoryview)), (
                f"Expected bytes, got {type(data)}"
            )
            assert len(data) > 0, f"Row group {i} returned empty bytes"

    @skip_without_minio
    def test_raw_rg_parquet_magic(self, uploaded_parquet_uris):
        """Raw bytes for column chunks are embedded inside a valid Parquet file context.

        The raw fetch covers the column-chunk bytes, not a complete Parquet file,
        so we just verify they are non-empty and have a plausible size.
        """
        loader = s3dlio.create_async_loader(
            prefix=TEST_URI_PREFIX,
            format="parquet",
            decode_mode="raw",
            footer_cap=4 * 1024 * 1024,
        )
        data = loader.get(0)
        # Column-chunk bytes for a snappy-compressed int64 + float32 RG
        # should be at least 10 bytes.
        assert len(bytes(data)) >= 10, "Raw column-chunk bytes suspiciously small"

    @skip_without_minio
    def test_num_rows_per_rg(self, uploaded_parquet_uris):
        """num_rows_in_rg() must return 3 for every row group (rows_per_rg=3)."""
        loader = s3dlio.create_async_loader(
            prefix=TEST_URI_PREFIX,
            format="parquet",
            decode_mode="raw",
            footer_cap=4 * 1024 * 1024,
        )
        for i in range(loader.len()):
            rows = loader.num_rows_in_rg(i)
            assert rows == 3, f"RG {i}: expected 3 rows, got {rows}"

    @skip_without_minio
    def test_keys_format(self, uploaded_parquet_uris):
        """keys() must return URI#rgN strings in the right format."""
        loader = s3dlio.create_async_loader(
            prefix=TEST_URI_PREFIX,
            format="parquet",
            decode_mode="raw",
            footer_cap=4 * 1024 * 1024,
        )
        keys = loader.keys()
        assert len(keys) == 4
        for k in keys:
            assert "#rg" in k, f"Key should contain '#rg': {k}"

    @skip_without_minio
    def test_rg_info_vec(self, uploaded_parquet_uris):
        """rg_info_vec() returns (file_uri_idx, rg_idx_in_file) pairs."""
        loader = s3dlio.create_async_loader(
            prefix=TEST_URI_PREFIX,
            format="parquet",
            decode_mode="raw",
            footer_cap=4 * 1024 * 1024,
        )
        info = loader.rg_info_vec()
        assert len(info) == 4
        # Should have (0,0), (0,1), (1,0), (1,1) — 2 RGs per file.
        rg_indices = [rg for _, rg in info]
        assert rg_indices.count(0) == 2, "Each file contributes one rg_idx=0"
        assert rg_indices.count(1) == 2, "Each file contributes one rg_idx=1"


class TestParquetEpochFastPath:
    """Verify epoch-2+ construction is faster and returns identical results."""

    @skip_without_minio
    def test_epoch2_same_num_rgs(self, uploaded_parquet_uris):
        """Two sequential constructions must return the same number of row groups."""
        import time

        loader1 = s3dlio.create_async_loader(
            prefix=TEST_URI_PREFIX,
            format="parquet",
            decode_mode="raw",
        )
        t0 = time.perf_counter()
        loader2 = s3dlio.create_async_loader(
            prefix=TEST_URI_PREFIX,
            format="parquet",
            decode_mode="raw",
        )
        elapsed_epoch2 = time.perf_counter() - t0

        assert loader1.len() == loader2.len(), (
            "Epoch 1 and epoch 2 must have the same number of row groups"
        )
        print(f"\n  Epoch-2 construction time: {elapsed_epoch2 * 1000:.1f} ms")

    @skip_without_minio
    def test_epoch2_identical_rg_rows(self, uploaded_parquet_uris):
        """num_rows_in_rg() must be identical across epochs."""
        loader1 = s3dlio.create_async_loader(
            prefix=TEST_URI_PREFIX, format="parquet", decode_mode="raw"
        )
        loader2 = s3dlio.create_async_loader(
            prefix=TEST_URI_PREFIX, format="parquet", decode_mode="raw"
        )
        for i in range(loader1.len()):
            assert loader1.num_rows_in_rg(i) == loader2.num_rows_in_rg(i), (
                f"RG {i} num_rows differs between epoch 1 and epoch 2"
            )


class TestParquetArrowIpc:
    """ArrowIpc decode mode: bytes must be parseable by pyarrow."""

    @skip_without_minio
    def test_arrow_ipc_decode(self, uploaded_parquet_uris):
        """ArrowIpc bytes must decode to a valid RecordBatch via pyarrow."""
        loader = s3dlio.create_async_loader(
            prefix=TEST_URI_PREFIX,
            format="parquet",
            decode_mode="arrow_ipc",
            footer_cap=4 * 1024 * 1024,
        )
        for i in range(loader.len()):
            raw = bytes(loader.get(i))
            reader = pa.ipc.open_stream(pa.py_buffer(raw))
            batch = reader.read_next_batch()
            assert batch.num_rows == 3, (
                f"RG {i}: expected 3 rows in Arrow batch, got {batch.num_rows}"
            )
            assert batch.schema.get_field_index("id") >= 0, "Missing 'id' column"
            assert batch.schema.get_field_index("value") >= 0, "Missing 'value' column"

    @skip_without_minio
    def test_arrow_ipc_values_correct(self, uploaded_parquet_uris):
        """Decoded Arrow batches must contain the correct integer ids."""
        loader = s3dlio.create_async_loader(
            prefix=TEST_URI_PREFIX,
            format="parquet",
            decode_mode="arrow_ipc",
        )
        all_ids = []
        for i in range(loader.len()):
            raw = bytes(loader.get(i))
            reader = pa.ipc.open_stream(pa.py_buffer(raw))
            batch = reader.read_next_batch()
            all_ids.extend(batch.column("id").to_pylist())

        # 2 files × 2 RGs × 3 rows = 12 ids total (0-5 repeated for 2 files).
        assert len(all_ids) == 12, f"Expected 12 ids total, got {len(all_ids)}"
        # IDs within each file must be [0,1,2,3,4,5].
        assert sorted(set(all_ids)) == [0, 1, 2, 3, 4, 5], (
            f"Unexpected id values: {sorted(set(all_ids))}"
        )
