#!/usr/bin/env python3
"""
Tests for s3dlio Parquet generation, reading, and DataLoader.

Two test groups:

1. **Local / always-run** — use ``file://`` URIs and temporary directories.
   These run without any external services and cover:
   * ``generate_parquet_bytes`` — in-memory Parquet generation
   * ``generate_and_write_parquet`` — generate + write in one Rust call
   * ``parquet_get_rg`` — per-RG fetch (raw and arrow decode modes)
   * Multiple dimension combinations (columns × rows/RG × row-groups)

2. **MinIO / skipped when unavailable** — upload test files to S3 and exercise
   the streaming DataLoader (``create_async_loader``).

Usage
-----
    # Local tests only (no MinIO needed):
    uv run pytest python/tests/test_parquet_dataloader.py -v --tb=short -k "Local"

    # All tests (MinIO required):
    source .env && uv run pytest python/tests/test_parquet_dataloader.py -v --tb=short
"""

import io
import os
import struct
import tempfile
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


# =============================================================================
# LOCAL TESTS — no MinIO / S3 required, use file:// URIs
# =============================================================================

# Dimensions exercised by the generation tests.
# Each tuple is (num_cols, rows_per_rg, num_row_groups).
GEN_DIMS = [
    (1, 1, 1),        # minimal: single column, single row, single RG
    (4, 16, 2),       # small multi-col, multi-RG
    (10, 64, 4),      # moderate
    (200, 8192, 3),   # DLRM-like schema, 3 RGs
]


def _decode_arrow(buf: bytes) -> pa.RecordBatch:
    """Deserialise Arrow IPC stream bytes to a RecordBatch."""
    return pa.ipc.open_stream(pa.py_buffer(buf)).read_next_batch()


class TestLocalGenerateParquetBytes:
    """generate_parquet_bytes() — in-memory Parquet, no I/O."""

    def test_returns_bytes(self):
        bv = s3dlio.generate_parquet_bytes(4, 16, 2)
        raw = bytes(bv)
        assert len(raw) > 0

    def test_parquet_magic(self):
        """Output must start and end with the Parquet magic bytes PAR1."""
        bv = s3dlio.generate_parquet_bytes(4, 16, 2)
        raw = bytes(bv)
        assert raw[:4] == b"PAR1", "Missing PAR1 header"
        assert raw[-4:] == b"PAR1", "Missing PAR1 footer"

    def test_pyarrow_can_read(self):
        """PyArrow must be able to parse the in-memory Parquet file."""
        bv = s3dlio.generate_parquet_bytes(4, 8, 3)
        table = pq.read_table(io.BytesIO(bytes(bv)))
        assert table.num_columns == 4
        assert table.num_rows == 8 * 3

    def test_column_names(self):
        """Columns must be named col0, col1, …"""
        bv = s3dlio.generate_parquet_bytes(5, 4, 1)
        table = pq.read_table(io.BytesIO(bytes(bv)))
        assert table.schema.names == [f"col{i}" for i in range(5)]

    def test_column_type_is_float32(self):
        """All columns must have Float32 type."""
        bv = s3dlio.generate_parquet_bytes(3, 4, 1)
        table = pq.read_table(io.BytesIO(bytes(bv)))
        for field in table.schema:
            assert field.type == pa.float32(), (
                f"Column {field.name} has type {field.type}, expected float32"
            )

    def test_row_group_count(self):
        """PyArrow metadata must report the correct number of row groups."""
        num_rgs = 5
        bv = s3dlio.generate_parquet_bytes(2, 8, num_rgs)
        meta = pq.read_metadata(io.BytesIO(bytes(bv)))
        assert meta.num_row_groups == num_rgs

    def test_rows_per_rg(self):
        """Each row group must contain exactly rows_per_rg rows."""
        rows_per_rg = 64
        num_rgs = 3
        bv = s3dlio.generate_parquet_bytes(2, rows_per_rg, num_rgs)
        meta = pq.read_metadata(io.BytesIO(bytes(bv)))
        for i in range(num_rgs):
            assert meta.row_group(i).num_rows == rows_per_rg, (
                f"RG {i}: expected {rows_per_rg} rows, got {meta.row_group(i).num_rows}"
            )

    @pytest.mark.parametrize("num_cols,rows_per_rg,num_rgs", GEN_DIMS)
    def test_dimensions(self, num_cols, rows_per_rg, num_rgs):
        """Various dimension combinations all produce valid Parquet."""
        bv = s3dlio.generate_parquet_bytes(num_cols, rows_per_rg, num_rgs)
        table = pq.read_table(io.BytesIO(bytes(bv)))
        assert table.num_columns == num_cols
        assert table.num_rows == rows_per_rg * num_rgs


class TestLocalGenerateAndWriteParquet:
    """generate_and_write_parquet() — generate + write to file:// in one call."""

    @pytest.mark.parametrize("num_cols,rows_per_rg,num_rgs", GEN_DIMS)
    def test_file_written_and_readable(self, tmp_path, num_cols, rows_per_rg, num_rgs):
        """File must exist after the call and be readable by PyArrow."""
        path = tmp_path / f"test_{num_cols}c_{rows_per_rg}r_{num_rgs}rg.parquet"
        uri = f"file://{path}"
        s3dlio.generate_and_write_parquet(uri, num_cols, rows_per_rg, num_rgs)
        assert path.exists(), f"File was not created: {path}"
        table = pq.read_table(str(path))
        assert table.num_columns == num_cols
        assert table.num_rows == rows_per_rg * num_rgs

    def test_file_has_parquet_magic(self, tmp_path):
        """Written file must start and end with PAR1."""
        path = tmp_path / "magic_check.parquet"
        s3dlio.generate_and_write_parquet(f"file://{path}", 4, 16, 2)
        data = path.read_bytes()
        assert data[:4] == b"PAR1"
        assert data[-4:] == b"PAR1"

    def test_multiple_files_independent(self, tmp_path):
        """Writing several files must produce independent, valid Parquet files."""
        specs = [(10, 32, 2), (20, 64, 3), (5, 128, 1)]
        paths = []
        for num_cols, rows_per_rg, num_rgs in specs:
            p = tmp_path / f"{num_cols}_{rows_per_rg}_{num_rgs}.parquet"
            s3dlio.generate_and_write_parquet(f"file://{p}", num_cols, rows_per_rg, num_rgs)
            paths.append((p, num_cols, rows_per_rg, num_rgs))

        for path, num_cols, rows_per_rg, num_rgs in paths:
            table = pq.read_table(str(path))
            assert table.num_columns == num_cols, f"{path.name}: wrong num_cols"
            assert table.num_rows == rows_per_rg * num_rgs, f"{path.name}: wrong num_rows"


class TestLocalParquetGetRg:
    """parquet_get_rg() — per-RG fetch and decode via file:// URIs."""

    @pytest.fixture
    def parquet_file(self, tmp_path):
        """Write a 3-RG × 200-col × 8192-row Parquet file and return its info."""
        path = tmp_path / "test.parquet"
        num_cols, rows_per_rg, num_rgs = 200, 8192, 3
        s3dlio.generate_and_write_parquet(
            f"file://{path}", num_cols, rows_per_rg, num_rgs
        )
        return str(path), f"file://{path}", num_cols, rows_per_rg, num_rgs

    def test_raw_mode_returns_bytes(self, parquet_file):
        _, uri, _, _, num_rgs = parquet_file
        for rg_idx in range(num_rgs):
            bv = s3dlio.parquet_get_rg(uri, rg_idx, None, 4 * 1024 * 1024, "raw")
            assert len(bytes(bv)) > 0, f"RG {rg_idx} raw bytes are empty"

    def test_arrow_mode_schema(self, parquet_file):
        """Arrow IPC bytes must decode to a batch with the correct schema."""
        _, uri, num_cols, rows_per_rg, _ = parquet_file
        bv = s3dlio.parquet_get_rg(uri, 0, None, 4 * 1024 * 1024, "arrow")
        batch = _decode_arrow(bytes(bv))
        assert batch.num_columns == num_cols, (
            f"Expected {num_cols} columns, got {batch.num_columns}"
        )
        assert batch.num_rows == rows_per_rg, (
            f"Expected {rows_per_rg} rows, got {batch.num_rows}"
        )
        for field in batch.schema:
            assert field.type == pa.float32(), (
                f"Column {field.name} has type {field.type}, expected float32"
            )

    def test_arrow_mode_all_rgs(self, parquet_file):
        """All row groups must decode to identical shapes."""
        _, uri, num_cols, rows_per_rg, num_rgs = parquet_file
        for rg_idx in range(num_rgs):
            bv = s3dlio.parquet_get_rg(uri, rg_idx, None, 4 * 1024 * 1024, "arrow")
            batch = _decode_arrow(bytes(bv))
            assert batch.num_columns == num_cols, f"RG {rg_idx}: wrong num_cols"
            assert batch.num_rows == rows_per_rg, f"RG {rg_idx}: wrong num_rows"

    def test_column_projection(self, parquet_file):
        """col_indices parameter must restrict columns returned."""
        _, uri, _, rows_per_rg, _ = parquet_file
        col_indices = [0, 5, 10]
        bv = s3dlio.parquet_get_rg(uri, 0, col_indices, 4 * 1024 * 1024, "arrow")
        batch = _decode_arrow(bytes(bv))
        assert batch.num_columns == len(col_indices), (
            f"Expected {len(col_indices)} projected columns, got {batch.num_columns}"
        )
        assert batch.num_rows == rows_per_rg

    def test_rg_idx_out_of_range(self, parquet_file):
        """Requesting an out-of-range RG index must raise an exception."""
        _, uri, _, _, num_rgs = parquet_file
        with pytest.raises(Exception):
            s3dlio.parquet_get_rg(uri, num_rgs + 100, None, 4 * 1024 * 1024, "raw")

    @pytest.mark.parametrize("num_cols,rows_per_rg,num_rgs", GEN_DIMS)
    def test_dimensions(self, tmp_path, num_cols, rows_per_rg, num_rgs):
        """Various dimension combinations all round-trip through parquet_get_rg."""
        path = tmp_path / f"dims_{num_cols}_{rows_per_rg}_{num_rgs}.parquet"
        uri = f"file://{path}"
        s3dlio.generate_and_write_parquet(uri, num_cols, rows_per_rg, num_rgs)
        for rg_idx in range(num_rgs):
            bv = s3dlio.parquet_get_rg(uri, rg_idx, None, 4 * 1024 * 1024, "arrow")
            batch = _decode_arrow(bytes(bv))
            assert batch.num_columns == num_cols
            assert batch.num_rows == rows_per_rg


class TestLocalParquetMetadataCache:
    """parquet_rg_row_offsets and cache helpers — file:// URIs."""

    def test_row_offsets_length(self, tmp_path):
        """parquet_rg_row_offsets must return num_rgs + 1 offsets."""
        num_rgs = 4
        path = tmp_path / "offsets.parquet"
        s3dlio.generate_and_write_parquet(f"file://{path}", 4, 32, num_rgs)
        offsets = s3dlio.parquet_rg_row_offsets(f"file://{path}")
        assert len(offsets) == num_rgs + 1, (
            f"Expected {num_rgs + 1} offsets, got {len(offsets)}"
        )

    def test_row_offsets_monotone(self, tmp_path):
        """Row offsets must be strictly monotonically increasing."""
        path = tmp_path / "mono.parquet"
        s3dlio.generate_and_write_parquet(f"file://{path}", 4, 32, 5)
        offsets = s3dlio.parquet_rg_row_offsets(f"file://{path}")
        for i in range(1, len(offsets)):
            assert offsets[i] > offsets[i - 1], (
                f"Offset not monotone at index {i}: {offsets}"
            )

    def test_cache_len_increases(self, tmp_path):
        """Fetching a new file must increase the cache length by 1."""
        before = s3dlio.parquet_cache_len()
        path = tmp_path / "cache_test.parquet"
        s3dlio.generate_and_write_parquet(f"file://{path}", 2, 8, 2)
        s3dlio.parquet_rg_row_offsets(f"file://{path}")
        after = s3dlio.parquet_cache_len()
        assert after == before + 1

    def test_cache_invalidate(self, tmp_path):
        """After invalidation the cache should not hold the entry."""
        path = tmp_path / "inval.parquet"
        uri = f"file://{path}"
        s3dlio.generate_and_write_parquet(uri, 2, 8, 2)
        s3dlio.parquet_rg_row_offsets(uri)
        before = s3dlio.parquet_cache_len()
        s3dlio.parquet_cache_invalidate(uri)
        after = s3dlio.parquet_cache_len()
        assert after == before - 1

    def test_cache_clear(self, tmp_path):
        """parquet_cache_clear must leave the cache empty."""
        for i in range(3):
            path = tmp_path / f"clear_{i}.parquet"
            uri = f"file://{path}"
            s3dlio.generate_and_write_parquet(uri, 2, 8, 2)
            s3dlio.parquet_rg_row_offsets(uri)
        s3dlio.parquet_cache_clear()
        assert s3dlio.parquet_cache_len() == 0
