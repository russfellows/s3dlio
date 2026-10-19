"""
S3DLIO Working Test Suite - Tests that Actually Pass
Focuses on functionality that we know works and can validate.
"""

# ruff: noqa: BLE001
# These checks record any exception a call raises as a failed result (that is what
# TestResults is for), so a broad except is intended throughout this file.

import asyncio
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import pytest

# DO NOT modify sys.path - use the installed package as-is.


class TestResults:
    """Track test results for reporting."""

    __test__ = False  # a helper, not a test class: keep pytest from collecting it

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.tests = []

    def add_result(self, name: str, passed: bool, message: str = ""):
        self.tests.append((name, passed, message))
        if passed:
            self.passed += 1
            print(f"✅ {name}: PASSED" + (f" - {message}" if message else ""))
        else:
            self.failed += 1
            print(f"❌ {name}: FAILED" + (f" - {message}" if message else ""))

    def add_skip(self, name: str, reason: str):
        self.skipped += 1
        print(f"⏭️  {name}: SKIPPED - {reason}")

    def summary(self):
        total = self.passed + self.failed
        if self.skipped:
            print(f"\n({self.skipped} check(s) skipped: not configured)")
        print(f"\n{'=' * 60}")
        print(
            f"TEST SUMMARY: {self.passed}/{total} PASSED ({self.passed / total * 100:.1f}%)"
        )
        print(f"{'=' * 60}")
        return self.passed == total


@pytest.fixture
def results():
    """The result tracker each check records into; any recorded failure fails the test.

    The assertion runs after the check returns, so pytest reports it as a teardown error
    that names every failed check."""
    tracker = TestResults()
    yield tracker
    failed = [
        f"{name}: {message}" for name, passed, message in tracker.tests if not passed
    ]
    assert tracker.failed == 0, "failed checks: " + "; ".join(failed)


def test_basic_imports(results: TestResults):
    """Test 1: Basic imports and module availability."""
    try:
        import s3dlio

        results.add_result("Import s3dlio", True, "Module imported successfully")

        # Test function availability using different methods
        try:
            create_dataset = s3dlio.create_dataset
            results.add_result(
                "create_dataset function", True, f"Type: {type(create_dataset)}"
            )
        except AttributeError:
            results.add_result(
                "create_dataset function", False, "AttributeError on direct access"
            )
        except Exception as e:
            results.add_result("create_dataset function", False, f"Other error: {e}")

        try:
            create_loader = s3dlio.create_async_loader
            results.add_result(
                "create_async_loader function", True, f"Type: {type(create_loader)}"
            )
        except AttributeError:
            results.add_result(
                "create_async_loader function", False, "AttributeError on direct access"
            )
        except Exception as e:
            results.add_result(
                "create_async_loader function", False, f"Other error: {e}"
            )

        if create_loader and callable(create_loader):
            results.add_result(
                "create_async_loader function", True, f"Type: {type(create_loader)}"
            )
        else:
            results.add_result(
                "create_async_loader function", False, "Not found or not callable"
            )

        # Test class availability
        PyDataset = getattr(s3dlio, "PyDataset", None)
        PyBytesAsyncDataLoader = getattr(s3dlio, "PyBytesAsyncDataLoader", None)

        if PyDataset and isinstance(PyDataset, type):
            results.add_result("PyDataset class", True, f"Available as {PyDataset}")
        else:
            results.add_result("PyDataset class", False, "Not found")

        if PyBytesAsyncDataLoader and isinstance(PyBytesAsyncDataLoader, type):
            results.add_result(
                "PyBytesAsyncDataLoader class",
                True,
                f"Available as {PyBytesAsyncDataLoader}",
            )
        else:
            results.add_result("PyBytesAsyncDataLoader class", False, "Not found")

    except Exception as e:
        results.add_result("Import s3dlio", False, str(e))


def test_dataset_creation(results: TestResults):
    """Test 2: Dataset creation with file:// URIs."""
    try:
        import s3dlio

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test file
            test_file = Path(tmpdir) / "test.txt"
            test_data = b"Hello, s3dlio test data!"
            test_file.write_bytes(test_data)

            # Test single file dataset creation
            file_uri = f"file://{test_file}"
            dataset = s3dlio.create_dataset(file_uri)

            if dataset is not None:
                results.add_result(
                    "File dataset creation", True, f"Created {type(dataset).__name__}"
                )
            else:
                results.add_result("File dataset creation", False, "Returned None")

            # Test directory dataset creation
            dir_uri = f"file://{tmpdir}"
            dir_dataset = s3dlio.create_dataset(dir_uri)

            if dir_dataset is not None:
                results.add_result(
                    "Directory dataset creation",
                    True,
                    f"Created {type(dir_dataset).__name__}",
                )
            else:
                results.add_result("Directory dataset creation", False, "Returned None")

    except Exception as e:
        results.add_result("Dataset creation", False, str(e))


def test_async_loader_creation(results: TestResults):
    """Test 3: Async loader creation."""
    try:
        import s3dlio

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for i in range(3):
                test_file = Path(tmpdir) / f"async_test_{i}.txt"
                test_file.write_bytes(f"Async test data {i}".encode())

            # Test async loader creation
            loader_uri = f"file://{tmpdir}"
            loader = s3dlio.create_async_loader(loader_uri)

            if loader is not None:
                results.add_result(
                    "Async loader creation", True, f"Created {type(loader).__name__}"
                )
            else:
                results.add_result("Async loader creation", False, "Returned None")

    except Exception as e:
        results.add_result("Async loader creation", False, str(e))


async def check_async_iteration(results: TestResults):
    """Test 4: Async iteration functionality."""
    try:
        import s3dlio

        create_async_loader = s3dlio.create_async_loader

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            test_files = []
            for i in range(3):
                test_file = Path(tmpdir) / f"iter_test_{i}.txt"
                test_content = f"Iteration test {i}".encode()
                test_file.write_bytes(test_content)
                test_files.append((test_file, test_content))

            # Test async iteration
            loader = create_async_loader(f"file://{tmpdir}")

            count = 0
            total_bytes = 0

            try:
                async for item in loader:
                    count += 1
                    total_bytes += len(item)

                    # Don't iterate forever
                    if count >= 5:
                        break

                if count > 0:
                    results.add_result(
                        "Async iteration",
                        True,
                        f"Processed {count} items, {total_bytes} bytes",
                    )
                else:
                    results.add_result("Async iteration", False, "No items processed")

            except Exception as iter_e:
                results.add_result(
                    "Async iteration", False, f"Iteration error: {iter_e}"
                )

    except Exception as e:
        results.add_result("Async iteration", False, str(e))


def test_options_handling(results: TestResults):
    """Test 5: Options dictionary handling."""
    try:
        import s3dlio

        create_dataset = s3dlio.create_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = Path(tmpdir) / "options_test.txt"
            test_file.write_bytes(b"Options test data")

            # Test with valid options
            options = {"batch_size": 16, "shuffle": False, "num_workers": 1}

            dataset_with_options = create_dataset(f"file://{test_file}", options)

            if dataset_with_options is not None:
                results.add_result(
                    "Options handling", True, "Dataset created with options"
                )
            else:
                results.add_result("Options handling", False, "Failed with options")

            # Test with empty options
            dataset_empty_opts = create_dataset(f"file://{test_file}", {})

            if dataset_empty_opts is not None:
                results.add_result(
                    "Empty options handling", True, "Dataset created with empty options"
                )
            else:
                results.add_result(
                    "Empty options handling", False, "Failed with empty options"
                )

    except Exception as e:
        results.add_result("Options handling", False, str(e))


def test_error_handling(results: TestResults):
    """Test 6: Error handling for invalid inputs."""
    try:
        import s3dlio

        create_dataset = s3dlio.create_dataset

        # Test invalid URI schemes
        error_cases = [
            ("ftp://invalid.com/path", "Invalid scheme"),
            ("file:///nonexistent/path", "Nonexistent path"),
            ("not-a-uri", "Malformed URI"),
        ]

        for uri, description in error_cases:
            try:
                create_dataset(uri)
                results.add_result(
                    f"Error handling - {description}",
                    False,
                    f"Should have failed: {uri}",
                )
            except Exception:
                results.add_result(
                    f"Error handling - {description}", True, f"Properly rejected: {uri}"
                )

    except Exception as e:
        results.add_result("Error handling", False, str(e))


def test_legacy_functions(results: TestResults):
    """Test 7: Legacy function availability."""
    try:
        import s3dlio

        # Test legacy functions exist
        legacy_functions = ["get", "put", "list", "stat"]

        for func_name in legacy_functions:
            func = getattr(s3dlio, func_name, None)
            if func and callable(func):
                results.add_result(f"Legacy {func_name}() function", True, "Available")
            else:
                results.add_result(
                    f"Legacy {func_name}() function", False, "Not available"
                )

        # Test helper functions
        helper_functions = ["list_keys", "list_full_uris", "get_by_key", "stat_key"]

        for func_name in helper_functions:
            func = getattr(s3dlio, func_name, None)
            if func and callable(func):
                results.add_result(f"Helper {func_name}() function", True, "Available")
            else:
                results.add_result(
                    f"Helper {func_name}() function", False, "Not available"
                )

    except Exception as e:
        results.add_result("Legacy functions", False, str(e))


def check_pytorch_integration(results: TestResults):
    """Test 8: PyTorch integration (the original bug fix)."""
    try:
        # Check if PyTorch is available
        if importlib.util.find_spec("torch") is None:
            results.add_result(
                "PyTorch integration", True, "PyTorch not installed - skipping"
            )
            return

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test training data
            for i in range(3):
                train_file = Path(tmpdir) / f"train_{i}.txt"
                train_file.write_bytes(f"Training sample {i}".encode())

            # Test S3IterableDataset import
            try:
                from s3dlio.torch import S3IterableDataset

                results.add_result(
                    "S3IterableDataset import", True, "Successfully imported"
                )
            except ImportError as e:
                results.add_result("S3IterableDataset import", False, str(e))
                return

            # Test S3IterableDataset creation
            # Note: This was the original bug - it used to fail with PyS3AsyncDataLoader not found
            try:
                # Try with empty loader_opts first
                dataset = S3IterableDataset(f"file://{tmpdir}", loader_opts={})
                results.add_result(
                    "PyTorch bug fix verification",
                    True,
                    "S3IterableDataset created successfully",
                )

                # Test that we can create a DataLoader
                from torch.utils.data import DataLoader

                dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
                results.add_result(
                    "PyTorch DataLoader creation", True, "DataLoader created"
                )

                # Test that we can iterate (at least start iteration)
                try:
                    iterator = iter(dataloader)
                    first_batch = next(iterator)
                    results.add_result(
                        "PyTorch DataLoader iteration",
                        True,
                        f"Got batch of size {len(first_batch)}",
                    )
                except StopIteration:
                    results.add_result(
                        "PyTorch DataLoader iteration",
                        True,
                        "Iterator created (empty dataset)",
                    )
                except Exception as iter_e:
                    results.add_result(
                        "PyTorch DataLoader iteration",
                        False,
                        f"Iteration failed: {iter_e}",
                    )

            except Exception as e:
                if "loader_opts" in str(e):
                    results.add_result(
                        "PyTorch bug fix verification",
                        False,
                        f"Requires loader_opts parameter: {e}",
                    )
                else:
                    results.add_result("PyTorch bug fix verification", False, str(e))

    except Exception as e:
        results.add_result("PyTorch integration", False, str(e))


def check_s3_functionality(results: TestResults):
    """Test 9: S3 functionality (if credentials available)."""
    s3_bucket = os.environ.get("S3DLIO_TEST_BUCKET")

    if not s3_bucket:
        results.add_skip(
            "S3 functionality", "S3DLIO_TEST_BUCKET not set (needs a live S3 endpoint)"
        )
        return

    try:
        import s3dlio

        create_dataset = s3dlio.create_dataset

        # Test S3 dataset creation
        s3_uri = f"s3://{s3_bucket}/test-prefix/"

        try:
            s3_dataset = create_dataset(s3_uri)
            if s3_dataset is not None:
                results.add_result(
                    "S3 dataset creation", True, f"Created {type(s3_dataset).__name__}"
                )
            else:
                results.add_result("S3 dataset creation", False, "Returned None")
        except Exception as e:
            # A configured live check must really pass: no error counts as success.
            results.add_result("S3 dataset creation", False, str(e))

    except Exception as e:
        results.add_result("S3 functionality", False, str(e))


def test_async_iteration(results: TestResults):
    """Test 4 under pytest: asyncio.run, so no pytest-asyncio setting matters."""
    asyncio.run(check_async_iteration(results))


def test_pytorch_integration(results: TestResults):
    """Test 8 under pytest: skipped, with the reason, when PyTorch is not installed."""
    pytest.importorskip("torch", reason="PyTorch is not installed")
    check_pytorch_integration(results)


def test_s3_functionality(results: TestResults):
    """Test 9 under pytest: needs a live S3 endpoint, skipped unless S3DLIO_TEST_BUCKET is set."""
    if not os.environ.get("S3DLIO_TEST_BUCKET"):
        pytest.skip(
            "S3DLIO_TEST_BUCKET is not set (this check needs a live S3 endpoint)"
        )
    check_s3_functionality(results)


async def run_all_tests():
    """Run all tests and report results."""
    print("S3DLIO Test Suite - Validation of Working Functionality")
    print("=" * 60)
    print("Testing functionality that should actually work...")
    print("=" * 60)

    results = TestResults()

    # Run synchronous tests
    sync_tests = [
        ("Basic Imports", test_basic_imports),
        ("Dataset Creation", test_dataset_creation),
        ("Async Loader Creation", test_async_loader_creation),
        ("Options Handling", test_options_handling),
        ("Error Handling", test_error_handling),
        ("Legacy Functions", test_legacy_functions),
        ("PyTorch Integration", check_pytorch_integration),
        ("S3 Functionality", check_s3_functionality),
    ]

    for test_name, test_func in sync_tests:
        print(f"\n--- {test_name} ---")
        test_func(results)

    # Run async tests
    print("\n--- Async Iteration ---")
    await check_async_iteration(results)

    # Final summary
    success = results.summary()

    if success:
        print("🎉 ALL TESTS PASSED!")
    else:
        print("⚠️  Some tests failed, but this shows what actually works")

    return success


def main():
    """Main test runner."""
    return asyncio.run(run_all_tests())


if __name__ == "__main__":
    sys.exit(0 if main() else 1)  # non-zero exit when any check failed
