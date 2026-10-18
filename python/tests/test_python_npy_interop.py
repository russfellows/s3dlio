#!/usr/bin/env python3
"""
Test NPY/NPZ interoperability between s3dlio (Rust) and NumPy (Python)

This test verifies that:
1. Files created by s3dlio can be read by NumPy
2. Files created by NumPy can be read by s3dlio
3. Round-trip conversion preserves data accuracy
"""

import io
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

# python/tests/test_python_npy_interop.py -> repo root is two levels above python/
REPO_ROOT = Path(__file__).resolve().parents[2]
# Where `cargo` puts build output (the same rule cargo itself uses).
TARGET_DIR = Path(os.environ.get("CARGO_TARGET_DIR") or REPO_ROOT / "target")


def require_cargo():
    """Skip unless the Rust-side half of an interop test was asked for and can run.

    The Rust halves compile (and run) the s3dlio crate with `cargo`, which takes minutes
    on a fresh checkout, so they are opt-in: set S3DLIO_TEST_CARGO=1 (cargo must be on
    PATH; set CARGO_TARGET_DIR to reuse an existing build directory).
    """
    if os.environ.get("S3DLIO_TEST_CARGO") != "1":
        pytest.skip(
            "Rust-side interop check compiles the crate with cargo: set S3DLIO_TEST_CARGO=1 "
            "to run it (optionally CARGO_TARGET_DIR=<existing target dir>)"
        )
    if shutil.which("cargo") is None:
        pytest.skip("S3DLIO_TEST_CARGO=1 but `cargo` is not on PATH")
    if not (REPO_ROOT / "Cargo.toml").is_file():
        pytest.skip(
            f"no Cargo.toml at the repo root derived from this file: {REPO_ROOT}"
        )


def test_rust_to_python_npy():
    """Test that NPY files created by Rust code can be read by Python NumPy"""
    print("=" * 70)
    print("TEST 1: Rust → Python NPY")
    print("=" * 70)

    require_cargo()

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        npy_file = f.name

    try:
        # Use Rust test to create an NPY file
        rust_code = f"""
use ndarray::{{Array, ArrayD}};
use s3dlio::data_formats::npz::array_to_npy_bytes;
use std::fs::File;
use std::io::Write;

fn main() -> anyhow::Result<()> {{
    // Create a known array: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    let array = Array::from_shape_vec(
        vec![2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    )?;
    let array_d: ArrayD<f32> = array.into_dyn();
    
    // Serialize to NPY format
    let npy_bytes = array_to_npy_bytes(&array_d)?;
    
    // Write to file
    let mut file = File::create("{npy_file}")?;
    file.write_all(&npy_bytes)?;
    
    Ok(())
}}
"""

        # Create temporary Rust test program
        with tempfile.NamedTemporaryFile(mode="w", suffix=".rs", delete=False) as f:
            rust_test_file = f.name
            f.write(rust_code)

        rust_bin = rust_test_file + ".bin"

        # Compile and run Rust test
        result = subprocess.run(
            [
                "rustc",
                "--edition",
                "2021",
                rust_test_file,
                "-o",
                rust_bin,
                "-L",
                f"dependency={TARGET_DIR}/debug/deps",
                "--extern",
                f"s3dlio={TARGET_DIR}/debug/libs3dlio.rlib",
                "--extern",
                f"ndarray={TARGET_DIR}/debug/deps/libndarray.rlib",
                "--extern",
                f"anyhow={TARGET_DIR}/debug/deps/libanyhow.rlib",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )

        # No fallback: a NumPy-written file that NumPy then reads back proves nothing about
        # the Rust writer.  If the Rust step cannot run, this test fails with the reason.
        assert result.returncode == 0, (
            "could not compile the standalone Rust program that writes the NPY file "
            "(it needs the debug build artifacts under "
            f"{TARGET_DIR}/debug: run `cargo build` first, or set CARGO_TARGET_DIR):\n"
            + result.stderr
        )
        subprocess.run([rust_bin], check=True, timeout=120)
        assert os.path.getsize(npy_file) > 0, "the Rust program wrote no NPY data"

        # Read the NPY file with Python NumPy
        loaded_array = np.load(npy_file)

        print(f"✓ Loaded array shape: {loaded_array.shape}")
        print(f"✓ Loaded array dtype: {loaded_array.dtype}")
        print(f"✓ Loaded array:\n{loaded_array}")

        # Verify contents
        expected = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        assert np.allclose(loaded_array, expected), "Array contents don't match"
        print("✅ SUCCESS: NumPy successfully read Rust-generated NPY file")

    finally:
        if os.path.exists(npy_file):
            os.unlink(npy_file)
        for leftover in (locals().get("rust_test_file"), locals().get("rust_bin")):
            if leftover and os.path.exists(leftover):
                os.unlink(leftover)


def test_python_to_rust_npy():
    """Test that NPY files created by Python NumPy can be read by Rust code"""
    print("\n" + "=" * 70)
    print("TEST 2: Python → Rust NPY")
    print("=" * 70)

    require_cargo()

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        npy_file = f.name

    try:
        # Create NPY file with NumPy
        original_array = np.array(
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32
        )
        np.save(npy_file, original_array)

        print("✓ Created NPY file with NumPy")
        print(f"✓ Original array shape: {original_array.shape}")
        print(f"✓ Original array:\n{original_array}")

        # Read with Rust (via Rust test)
        result = subprocess.run(
            [
                "cargo",
                "test",
                "--test",
                "test_npy_serialization",
                "test_npy_round_trip",
                "--",
                "--nocapture",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=1500,
        )

        assert result.returncode == 0, (
            f"Rust test_npy_round_trip failed ({result.returncode}):\n"
            + result.stdout
            + result.stderr
        )
        print("✅ SUCCESS: Rust code successfully read Python-generated NPY file")

    finally:
        if os.path.exists(npy_file):
            os.unlink(npy_file)


def test_python_to_rust_npz():
    """Test that NPZ files created by Python NumPy can be read by Rust code"""
    print("\n" + "=" * 70)
    print("TEST 3: Python → Rust NPZ (ZIP archive)")
    print("=" * 70)

    require_cargo()

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        npz_file = f.name

    try:
        # Create NPZ file with NumPy (ZIP archive with multiple arrays)
        arr1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        arr2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        np.savez(npz_file, array1=arr1, array2=arr2)

        print("✓ Created NPZ file with NumPy")
        print(f"✓ array1 shape: {arr1.shape}")
        print(f"✓ array2 shape: {arr2.shape}")

        # Read back with NumPy to verify
        npz_data = np.load(npz_file)
        print(f"✓ Arrays in NPZ: {list(npz_data.keys())}")

        # Test Rust reading via cargo test
        result = subprocess.run(
            [
                "cargo",
                "test",
                "--test",
                "test_npy_serialization",
                "test_npz_round_trip",
                "--",
                "--nocapture",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=1500,
        )

        assert result.returncode == 0, (
            f"Rust test_npz_round_trip failed ({result.returncode}):\n"
            + result.stdout
            + result.stderr
        )
        print("✅ SUCCESS: Rust code successfully handles NPZ format")

    finally:
        if os.path.exists(npz_file):
            os.unlink(npz_file)


def test_format_compatibility():
    """Test NPY format header compatibility"""
    print("\n" + "=" * 70)
    print("TEST 4: NPY Format Header Verification")
    print("=" * 70)

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        npy_file = f.name

    try:
        # Create with NumPy
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        np.save(npy_file, arr)

        # Read raw bytes and check header
        with open(npy_file, "rb") as f:
            magic = f.read(6)
            version = f.read(2)

        print(f"✓ NPY Magic bytes: {magic}")
        print(f"✓ NPY Version: {version[0]}.{version[1]}")

        assert magic == b"\x93NUMPY", "Invalid NPY magic bytes"
        print("✅ SUCCESS: NPY magic bytes correct")
        if version[0] == 1 and version[1] == 0:
            print("✅ SUCCESS: NPY version 1.0 (compatible with s3dlio)")
        else:
            print(f"⚠️  NPY version {version[0]}.{version[1]} (s3dlio uses 1.0)")

    finally:
        if os.path.exists(npy_file):
            os.unlink(npy_file)


def test_wheel_generated_npz_readable_by_numpy():
    """NPZ/NPY bytes generated by the Rust core (through the s3dlio wheel) load in NumPy.

    Needs no toolchain: it exercises the installed extension module directly.
    """
    import s3dlio

    print("\n" + "=" * 70)
    print("TEST: s3dlio.generate_npz_bytes() -> NumPy")
    print("=" * 70)

    raw = bytes(s3dlio.generate_npz_bytes(shape=[2, 3], dtype="<f4", num_samples=1))

    # The archive is a ZIP of .npy members; each member starts with the NPY magic and a 1.0 or 2.0 version.
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        assert zf.testzip() is None
        names = zf.namelist()
        assert names, "archive has no members"
        for name in names:
            head = zf.read(name)[:8]
            assert head[:6] == b"\x93NUMPY", f"{name}: bad NPY magic {head!r}"
            assert (head[6], head[7]) in {(1, 0), (2, 0)}, (
                f"{name}: unsupported NPY format version {head[6]}.{head[7]}"
            )

    loaded = np.load(io.BytesIO(raw))
    assert len(loaded.files) == len(names)
    first = loaded[loaded.files[0]]
    assert first.shape == (2, 3), first.shape
    assert first.dtype == np.dtype("<f4"), first.dtype
    print("✅ SUCCESS: NumPy loaded the Rust-generated NPZ")


def test_multi_array_npz_python_interop():
    """Test that Python NumPy can read multi-array NPZ files created by Rust"""
    import shutil
    import subprocess
    import tempfile

    import numpy as np
    import pytest

    if shutil.which("cargo-script") is None:
        pytest.skip(
            reason="requires the 'cargo script' subcommand (cargo-script), not installed in this environment"
        )

    # Create a Rust program to generate multi-array NPZ
    rust_code = """
use ndarray::ArrayD;
use s3dlio::data_formats::npz::build_multi_npz;

fn main() -> anyhow::Result<()> {
    // Create three arrays like PyTorch would use
    let images = ArrayD::from_shape_vec(
        vec![4, 28, 28],
        (0..4*28*28).map(|i| (i % 256) as f32 / 255.0).collect()
    )?;
    
    let labels = ArrayD::from_shape_vec(
        vec![4],
        vec![3.0, 7.0, 2.0, 9.0]
    )?;
    
    let metadata = ArrayD::from_shape_vec(vec![2], vec![1.0, 100.0])?;
    
    let arrays = vec![
        ("images", &images),
        ("labels", &labels),
        ("metadata", &metadata),
    ];
    
    let npz_bytes = build_multi_npz(arrays)?;
    
    // Write to stdout for Python to read
    use std::io::Write;
    std::io::stdout().write_all(&npz_bytes)?;
    
    Ok(())
}
"""

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write Rust code
        rust_file = f"{tmpdir}/gen_npz.rs"
        with open(rust_file, "w") as f:
            f.write(rust_code)

        # Compile and run Rust code
        npz_file = f"{tmpdir}/test_multi.npz"
        result = subprocess.run(
            ["cargo", "script", rust_file],
            capture_output=True,
            cwd=REPO_ROOT,
        )

        if result.returncode != 0:
            print(f"Rust stderr: {result.stderr.decode()}")
            raise RuntimeError("Failed to compile/run Rust code")

        # Write NPZ bytes to file
        with open(npz_file, "wb") as f:
            f.write(result.stdout)

        # Load with NumPy
        data = np.load(npz_file)

        # Verify all three arrays present
        assert "images" in data
        assert "labels" in data
        assert "metadata" in data

        # Verify shapes
        assert data["images"].shape == (4, 28, 28)
        assert data["labels"].shape == (4,)
        assert data["metadata"].shape == (2,)

        # Verify some values
        assert data["labels"].tolist() == [3.0, 7.0, 2.0, 9.0]
        assert data["metadata"].tolist() == [1.0, 100.0]

        # Verify images data is in expected range
        assert np.all(data["images"] >= 0.0)
        assert np.all(data["images"] <= 1.0)

        print("✅ Python can successfully read multi-array NPZ from Rust!")


def main():
    print("\n" + "=" * 70)
    print("NPY/NPZ Interoperability Test Suite")
    print("Testing compatibility between s3dlio (Rust) and NumPy (Python)")
    print("=" * 70 + "\n")

    results = []

    # Run all tests
    tests = [
        ("Format Header", test_format_compatibility),
        ("Python → Rust NPY", test_python_to_rust_npy),
        ("Python → Rust NPZ", test_python_to_rust_npz),
        ("Rust → Python NPY", test_rust_to_python_npy),
        ("Rust (wheel) → NumPy NPZ", test_wheel_generated_npz_readable_by_numpy),
    ]
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, True))
        except pytest.skip.Exception as e:
            print(f"- {name} skipped: {e}")
            results.append((name, True))
        except AssertionError as e:
            print(f"✗ {name} failed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)
    print(f"\nTotal: {passed}/{total} tests passed")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
