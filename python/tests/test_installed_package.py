"""
S3DLIO installed-package test suite (pytest).

Tests the s3dlio package that is actually installed in the active environment, using
only ``file://`` URIs, so no live endpoint or credentials are needed.

Run it with pytest (``pytest python/tests/test_installed_package.py``) or directly as a
script (``python python/tests/test_installed_package.py``), which hands the file to pytest.

It used to be a standalone script whose functions took a ``results`` argument that the
script supplied itself, so pytest reported "fixture 'results' not found" errors for every
function (issue #280). Every check is now an ordinary pytest test with real assertions.
"""

import asyncio
import sys
from pathlib import Path

import pytest

# DO NOT modify sys.path - use the installed package as-is.
import s3dlio


def _write_files(directory: Path, names_and_content):
    for name, content in names_and_content:
        (directory / name).write_bytes(content)


# ---------------------------------------------------------------------------
# Package installation and import
# ---------------------------------------------------------------------------
SOURCE_TREE = Path(__file__).resolve().parents[1]  # <repo>/python


def _is_inside_source_tree(module_file, source_tree):
    return Path(source_tree).resolve() in Path(module_file).resolve().parents


def test_source_tree_detection():
    """The check below really tells a source-tree import from an installed one."""
    assert _is_inside_source_tree(SOURCE_TREE / "s3dlio" / "__init__.py", SOURCE_TREE)
    assert not _is_inside_source_tree(
        SOURCE_TREE.parent
        / ".venv"
        / "lib"
        / "site-packages"
        / "s3dlio"
        / "__init__.py",
        SOURCE_TREE,
    )


def test_package_is_not_imported_from_the_source_checkout():
    """Skip (with the reason) when s3dlio is imported from the repository's python/ tree.

    An editable install (maturin's python-source = "python") also imports from that
    tree; this suite is meant for the built wheel, so the run is reported as skipped
    rather than failed.  With an installed wheel the assertion below is the real check."""
    module_path = Path(s3dlio.__file__).resolve()
    if _is_inside_source_tree(module_path, SOURCE_TREE):
        pytest.skip(
            f"s3dlio is imported from the source checkout ({module_path}) - an editable/"
            "source install, not the built wheel; install the wheel to run this check"
        )
    assert not _is_inside_source_tree(module_path, SOURCE_TREE), module_path


def test_module_has_public_attributes():
    public = [name for name in dir(s3dlio) if not name.startswith("_")]
    assert public, "s3dlio exports no public attributes"


# ---------------------------------------------------------------------------
# Enhanced API availability
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["create_dataset", "create_async_loader"])
def test_enhanced_api_function_is_callable(name):
    assert callable(getattr(s3dlio, name, None)), (
        f"s3dlio.{name} is missing or not callable"
    )


def test_pydataset_class_is_exported():
    assert isinstance(getattr(s3dlio, "PyDataset", None), type)


# ---------------------------------------------------------------------------
# Dataset creation
# ---------------------------------------------------------------------------
def test_file_dataset_creation(tmp_path):
    test_file = tmp_path / "functional_test.txt"
    test_file.write_bytes(b"Functional test data for s3dlio")
    dataset = s3dlio.create_dataset(f"file://{test_file}")
    assert isinstance(dataset, s3dlio.PyDataset)


def test_directory_dataset_creation(tmp_path):
    _write_files(
        tmp_path,
        [(f"extra_{i}.txt", f"Extra test data {i}".encode()) for i in range(3)],
    )
    dataset = s3dlio.create_dataset(f"file://{tmp_path}")
    assert isinstance(dataset, s3dlio.PyDataset)


# ---------------------------------------------------------------------------
# Async loader
# ---------------------------------------------------------------------------
def test_async_loader_creation(tmp_path):
    _write_files(
        tmp_path,
        [
            (f"async_functional_{i}.txt", f"Async functional test {i}".encode())
            for i in range(3)
        ],
    )
    loader = s3dlio.create_async_loader(f"file://{tmp_path}")
    assert hasattr(loader, "__aiter__"), "async loader is not async-iterable"


def test_async_iteration_yields_items(tmp_path):
    _write_files(
        tmp_path,
        [
            (f"iteration_{i:02d}.txt", f"Iteration test content {i}".encode())
            for i in range(4)
        ],
    )
    loader = s3dlio.create_async_loader(f"file://{tmp_path}")

    async def drain():
        items = 0
        async for _item in loader:
            items += 1
            # Safety limit to avoid an endless loop.
            if items >= 10:
                break
        return items

    assert asyncio.run(drain()) > 0, "async iteration produced no items"


# ---------------------------------------------------------------------------
# PyTorch integration (the original "PyS3AsyncDataLoader not found" bug fix)
# ---------------------------------------------------------------------------
def test_pytorch_integration(tmp_path):
    pytest.importorskip("torch", reason="PyTorch is not installed")
    from s3dlio.torch import S3IterableDataset
    from torch.utils.data import DataLoader

    _write_files(
        tmp_path,
        [
            (f"train_{i:03d}.txt", f"Training sample {i} - ".encode() + b"x" * 50)
            for i in range(5)
        ],
    )

    # This was broken in v0.7.x - "PyS3AsyncDataLoader not found".
    with pytest.warns(DeprecationWarning, match="ObjectStoreIterableDataset"):
        dataset = S3IterableDataset(f"file://{tmp_path}", loader_opts={})
    dataloader = DataLoader(dataset, batch_size=2, num_workers=0)
    batch = next(iter(dataloader), None)
    assert batch is not None, "the DataLoader produced no batch from 5 files"


# ---------------------------------------------------------------------------
# Legacy and helper API
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name",
    [
        "get",
        "put",
        "list",
        "stat",
        "get_by_key",
        "stat_key",
        "list_keys",
        "list_full_uris",
    ],
)
def test_api_function_is_callable(name):
    # list_keys_from_s3 and list_uris were retired in favour of list_keys and
    # list_full_uris (see python/s3dlio/__init__.py), so the current names are checked.
    assert callable(getattr(s3dlio, name, None)), (
        f"s3dlio.{name} is missing or not callable"
    )


def test_list_keys_returns_relative_keys(tmp_path):
    _write_files(tmp_path, [("a.txt", b"a"), ("b.txt", b"b")])
    keys = s3dlio.list_keys(f"file://{tmp_path}/")
    assert sorted(keys) == ["a.txt", "b.txt"]


def test_list_full_uris_returns_object_uris(tmp_path):
    _write_files(tmp_path, [("a.txt", b"a"), ("b.txt", b"b")])
    uris = s3dlio.list_full_uris(f"file://{tmp_path}/")
    assert sorted(uris) == [f"file://{tmp_path}/a.txt", f"file://{tmp_path}/b.txt"]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("uri", "message"),
    [
        ("ftp://invalid.example.com/path", "Unable to infer dataset type from URI"),
        ("file:///absolutely/nonexistent/path/file.txt", "Path does not exist"),
        ("not-even-a-uri", "Unable to infer dataset type from URI"),
        ("", "Unable to infer dataset type from URI"),
    ],
    ids=["invalid-scheme", "nonexistent-file", "malformed-uri", "empty-uri"],
)
def test_create_dataset_rejects_bad_uri(uri, message):
    with pytest.raises(RuntimeError, match=message):
        s3dlio.create_dataset(uri)


# ---------------------------------------------------------------------------
# Options handling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "options",
    [{"batch_size": 16, "shuffle": False}, {}, None],
    ids=["valid-options", "empty-options", "none-options"],
)
def test_create_dataset_accepts_options(tmp_path, options):
    test_file = tmp_path / "options_test.txt"
    test_file.write_bytes(b"Options test data")
    dataset = s3dlio.create_dataset(f"file://{test_file}", options)
    assert isinstance(dataset, s3dlio.PyDataset)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
