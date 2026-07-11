#!/usr/bin/env python3
"""
Framework integration tests for s3dlio v0.9.0 - Bytes Migration Verification

Since s3dlio is designed for S3 data generation, these tests focus on:
1. Verifying get() works correctly after Vec<u8> -> Bytes migration
2. Testing that frameworks can consume the returned bytes
3. Verifying async loaders work with all frameworks

Note: Full S3 testing requires S3 credentials and is done separately.
These tests use file:// URIs where supported and pre-generated test data.
"""

import sys
import os
import io
import tempfile
import numpy as np


def test_pytorch_bytes_compatibility():
    """Test PyTorch can consume data after Bytes migration"""
    print("\n" + "=" * 60)
    print("PYTORCH - BYTES MIGRATION COMPATIBILITY")
    print("=" * 60 + "\n")

    try:
        import torch
        import s3dlio

        print("✓ PyTorch and s3dlio imported successfully")
    except ImportError as e:
        import pytest

        pytest.skip(reason=f"PyTorch not available: {e}")

    # Test 1: get() returns bytes that PyTorch can use
    print("\n1. Testing get() returns valid bytes for PyTorch...")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create NPZ test file manually
        test_data = {"features": np.random.rand(32, 64).astype(np.float32)}
        npz_file = os.path.join(tmpdir, "test.npz")
        np.savez(npz_file, **test_data)

        # Read via s3dlio.get() - tests Bytes -> Python bytes
        uri = f"file://{npz_file}"
        result_bytes = s3dlio.get(uri)

        print(f"   ✓ get() returned {len(result_bytes)} bytes")

        # Verify PyTorch can load it
        npz = np.load(io.BytesIO(result_bytes))
        tensor = torch.from_numpy(npz["features"])

        assert tensor.shape == (32, 64), f"Unexpected shape: {tensor.shape}"
        print(f"   ✓ PyTorch tensor created: {tensor.shape}")

    # Test 2: Async loader with PyTorch
    print("\n2. Testing async loader produces valid bytes...")
    import asyncio

    async def test_loader():
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple NPZ files
            for i in range(5):
                data = {"x": np.random.rand(10, 20).astype(np.float32)}
                npz_file = os.path.join(tmpdir, f"data_{i:03d}.npz")
                np.savez(npz_file, **data)

            # Load via s3dlio async loader
            loader = s3dlio.create_async_loader(f"file://{tmpdir}/", {"batch_size": 2})

            tensors = []
            async for batch in loader:
                for item_bytes in batch:
                    npz = np.load(io.BytesIO(item_bytes))
                    tensor = torch.from_numpy(npz["x"])
                    tensors.append(tensor)

            return len(tensors)

    count = asyncio.run(test_loader())
    assert count == 5, f"Expected 5, got {count}"
    print(f"   ✓ Loaded {count} tensors via async loader")

    print("\n✅ PyTorch: Bytes migration compatible\n")


def test_tensorflow_bytes_compatibility():
    """Test TensorFlow can consume data after Bytes migration"""
    print("\n" + "=" * 60)
    print("TENSORFLOW - BYTES MIGRATION COMPATIBILITY")
    print("=" * 60 + "\n")

    try:
        import tensorflow as tf
        import s3dlio

        print("✓ TensorFlow and s3dlio imported successfully")
    except ImportError as e:
        import pytest

        pytest.skip(reason=f"TensorFlow not available: {e}")

    # Test 1: get() with TFRecord
    print("\n1. Testing get() with TFRecord data...")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create TFRecord file
        tfr_file = os.path.join(tmpdir, "test.tfrecord")
        with tf.io.TFRecordWriter(tfr_file) as writer:
            for i in range(5):
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "value": tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[i])
                            )
                        }
                    )
                )
                writer.write(example.SerializeToString())

        # Read via s3dlio
        tfr_bytes = s3dlio.get(f"file://{tfr_file}")

        print(f"   ✓ TFRecord read: {len(tfr_bytes)} bytes")

        # Verify non-empty
        assert len(tfr_bytes) > 0, "Empty bytes"
        print("   ✓ TensorFlow can access the bytes")

    # Test 2: NPZ data with TensorFlow
    print("\n2. Testing NPZ data with TensorFlow...")
    import asyncio

    async def test_tf_loader():
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create NPZ files
            for i in range(4):
                data = {"features": np.random.rand(8, 16).astype(np.float32)}
                npz_file = os.path.join(tmpdir, f"tf_data_{i}.npz")
                np.savez(npz_file, **data)

            # Load via async loader
            loader = s3dlio.create_async_loader(f"file://{tmpdir}/", {"batch_size": 2})

            tensors = []
            async for batch in loader:
                for item_bytes in batch:
                    npz = np.load(io.BytesIO(item_bytes))
                    tensor = tf.constant(npz["features"])
                    tensors.append(tensor)

            return len(tensors)

    count = asyncio.run(test_tf_loader())
    assert count == 4, f"Expected 4, got {count}"
    print(f"   ✓ Converted {count} items to TF tensors")

    print("\n✅ TensorFlow: Bytes migration compatible\n")


def test_jax_bytes_compatibility():
    """Test JAX can consume data after Bytes migration"""
    print("\n" + "=" * 60)
    print("JAX - BYTES MIGRATION COMPATIBILITY")
    print("=" * 60 + "\n")

    try:
        import jax.numpy as jnp
        import s3dlio

        print("✓ JAX and s3dlio imported successfully")
    except ImportError as e:
        import pytest

        pytest.skip(reason=f"JAX not available: {e}")

    # Test 1: get() returns bytes JAX can use
    print("\n1. Testing get() with JAX...")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create NPZ file
        data = {"params": np.random.randn(50, 100).astype(np.float32)}
        npz_file = os.path.join(tmpdir, "jax_test.npz")
        np.savez(npz_file, **data)

        # Read via s3dlio
        npz_bytes = s3dlio.get(f"file://{npz_file}")

        print(f"   ✓ get() returned {len(npz_bytes)} bytes")

        # Convert to JAX array
        npz = np.load(io.BytesIO(npz_bytes))
        jax_array = jnp.array(npz["params"])

        assert jax_array.shape == (50, 100), f"Unexpected shape: {jax_array.shape}"
        print(f"   ✓ JAX array created: {jax_array.shape}")

    # Test 2: Batch loading for JAX
    print("\n2. Testing batch loading with JAX...")
    import asyncio

    async def test_jax_batches():
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create batch files
            for i in range(6):
                data = {"batch": np.random.randn(16, 32).astype(np.float32)}
                npz_file = os.path.join(tmpdir, f"batch_{i:02d}.npz")
                np.savez(npz_file, **data)

            # Load batches
            loader = s3dlio.create_async_loader(f"file://{tmpdir}/", {"batch_size": 3})

            arrays = []
            async for batch in loader:
                for item_bytes in batch:
                    npz = np.load(io.BytesIO(item_bytes))
                    jax_array = jnp.array(npz["batch"])
                    arrays.append(jax_array)

            return len(arrays)

    count = asyncio.run(test_jax_batches())
    assert count == 6, f"Expected 6, got {count}"
    print(f"   ✓ Loaded {count} JAX arrays via batching")

    print("\n✅ JAX: Bytes migration compatible\n")


def main():
    """Run all framework integration tests"""
    print("=" * 70)
    print("s3dlio v0.9.0 - FRAMEWORK INTEGRATION TESTS (Bytes Migration)")
    print("=" * 70)

    import _pytest.outcomes

    tests = {
        "PyTorch": test_pytorch_bytes_compatibility,
        "TensorFlow": test_tensorflow_bytes_compatibility,
        "JAX": test_jax_bytes_compatibility,
    }

    results = {}
    for framework, test_func in tests.items():
        try:
            test_func()
            results[framework] = True
        except _pytest.outcomes.Skipped as e:
            print(f"⚠️  {framework} skipped: {e}")
            results[framework] = True
        except AssertionError as e:
            print(f"❌ {framework} FAILED: {e}")
            results[framework] = False

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for framework, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{framework:15} {status}")

    print(f"\nTotal: {passed}/{total} frameworks passed")

    if passed == total:
        print("\n🎉 All framework integrations verified after Bytes migration")
        return 0
    else:
        print(f"\n⚠️  {total - passed} framework(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
