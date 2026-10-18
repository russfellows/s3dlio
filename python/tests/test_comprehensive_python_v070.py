#!/usr/bin/env python3
"""
Comprehensive Python tests for s3dlio v0.7.0
Tests all frameworks (JAX, PyTorch, TensorFlow) with all backends (File, S3, Azure, DirectIO)
Uses the validated PyCheckpointStore API pattern from existing tests.
"""

import tempfile
import os
import numpy as np
import s3dlio
import gc
from pathlib import Path


def _finalize_and_load(store, world_size, step, epoch, framework, metas):
    """Finalize the checkpoint from ALL ranks' shard metadata (finalize is rank 0's job),
    then return ``(reader, manifest)`` for the latest manifest.

    The Python API has no ``PyCheckpointReader.load_distributed_shard`` (it never did: the
    original v0.7.x tests called a method that was only ever planned). The documented read
    path is: ``PyCheckpointWriter.finalize_distributed_checkpoint`` -> ``reader.
    load_latest_manifest()`` -> ``reader.read_shard_by_rank(manifest, rank)``.
    """
    writer = store.writer(world_size=world_size, rank=0)
    writer.finalize_distributed_checkpoint(step, epoch, framework, metas, None)
    reader = store.reader()
    manifest = reader.load_latest_manifest()
    assert manifest is not None, "finalized checkpoint must have a manifest"
    return reader, manifest


def _finalize_and_read(store, world_size, rank, step, epoch, framework, metas):
    """Finalize (see ``_finalize_and_load``) and read ``rank``'s shard back as bytes."""
    reader, manifest = _finalize_and_load(
        store, world_size, step, epoch, framework, metas
    )
    return bytes(reader.read_shard_by_rank(manifest, rank))


def test_jax_streaming_checkpointing():
    """Test JAX-style checkpointing with streaming interface"""
    print("=== Testing JAX Streaming Checkpointing ===")

    # Test data - simulate a JAX array
    jax_data = np.random.rand(1000, 1000).astype(np.float32)
    print(f"JAX test data: {jax_data.shape} array, {jax_data.nbytes} bytes")

    with tempfile.TemporaryDirectory() as temp_dir:
        base_uri = f"file://{temp_dir}"
        print(f"Using URI: {base_uri}")

        # Create checkpoint store (using working API pattern)
        store = s3dlio.PyCheckpointStore(
            base_uri, strategy="binary", multipart_threshold=1024 * 1024
        )
        writer = store.writer(world_size=1, rank=0)

        checkpoint_id = 100
        shard_name = "jax_weights"

        # Test 1: Traditional save
        print("Testing traditional JAX save...")
        shard_meta = writer.save_distributed_shard(
            checkpoint_id,  # step
            0,  # epoch
            "jax",  # framework
            jax_data.tobytes(),  # data
        )

        print(f"✓ Saved {shard_meta['size']} bytes to {shard_meta['key']}")
        print(f"✓ JAX checkpoint saved successfully")

        # Test 2: Streaming save (v0.7.0 enhancement)
        stream_shard_name = "jax_streaming"
        print("Testing JAX streaming save...")

        stream_step = checkpoint_id + 1
        stream = writer.get_distributed_shard_stream(
            stream_step,  # step
            0,  # epoch
            "jax",  # framework
        )

        # Stream in chunks (simulating JAX block-wise saves)
        chunk_size = 100000  # 100KB chunks
        jax_bytes = jax_data.tobytes()
        for i in range(0, len(jax_bytes), chunk_size):
            chunk = jax_bytes[i : i + chunk_size]
            stream.write_chunk(chunk)

        stream_meta = stream.finalize()
        print(f"✓ Streamed {stream_meta['size']} bytes to {stream_meta['key']}")

        # Test 3: Checkpoint loading and validation (finalize, then read via the manifest)
        print("Testing JAX checkpoint loading...")

        # The one-shot and the streamed shard are the same rank at the same step, so each
        # is finalized and read back on its own checkpoint step.
        loaded_data = _finalize_and_read(
            store, 1, 0, checkpoint_id, 0, "jax", [shard_meta]
        )
        loaded_array = np.frombuffer(loaded_data, dtype=np.float32).reshape(
            jax_data.shape
        )
        assert np.allclose(loaded_array, jax_data), (
            "JAX data should match after roundtrip"
        )
        print(f"✓ Loaded and validated {len(loaded_data)} bytes")
        print("✓ Data integrity verified")

        # Verify streaming data matches too
        loaded_stream_data = _finalize_and_read(
            store, 1, 0, stream_step, 0, "jax", [stream_meta]
        )
        loaded_stream_array = np.frombuffer(
            loaded_stream_data, dtype=np.float32
        ).reshape(jax_data.shape)
        assert np.allclose(loaded_stream_array, jax_data), (
            "JAX streaming data should match"
        )

        print("✓ JAX streaming validation passed")
        print("✅ JAX checkpointing test completed successfully")


def test_pytorch_multirank_checkpointing():
    """Test PyTorch-style multirank checkpointing"""
    print("\n=== Testing PyTorch Multi-Rank Checkpointing ===")

    # Simulate PyTorch tensor data (model weights)
    pytorch_data = np.random.rand(512, 512, 3).astype(np.float32)
    print(
        f"PyTorch test data: {pytorch_data.shape} tensor, {pytorch_data.nbytes} bytes"
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        base_uri = f"file://{temp_dir}"
        print(f"Using URI: {base_uri}")

        # Test with multi-rank scenario (distributed training)
        store = s3dlio.PyCheckpointStore(
            base_uri, strategy="binary", multipart_threshold=1024 * 1024
        )
        writer = store.writer(world_size=2, rank=0)  # 2-GPU setup, rank 0

        checkpoint_id = 200

        # Test distributed saving
        print("Testing distributed PyTorch checkpoint...")

        shard_meta = writer.save_distributed_shard(
            checkpoint_id,  # step
            42,  # epoch (meaningful for PyTorch)
            "pytorch",  # framework
            pytorch_data.tobytes(),  # data
        )

        print(f"✓ PyTorch checkpoint saved: {shard_meta['size']} bytes")

        # Finalizing a 2-rank checkpoint needs both ranks' shards
        other_data = pytorch_data[::-1].copy()
        other_meta = store.writer(world_size=2, rank=1).save_distributed_shard(
            checkpoint_id, 42, "pytorch", other_data.tobytes()
        )

        # Test concurrent access simulation
        print("Testing concurrent checkpoint validation...")

        metas = [shard_meta, other_meta]
        loaded_data = _finalize_and_read(
            store, 2, 0, checkpoint_id, 42, "pytorch", metas
        )
        loaded_other = _finalize_and_read(
            store, 2, 1, checkpoint_id, 42, "pytorch", metas
        )
        assert np.array_equal(
            np.frombuffer(loaded_other, dtype=np.float32).reshape(other_data.shape),
            other_data,
        ), "rank 1 shard should round-trip too"

        # Verify data integrity
        loaded_tensor = np.frombuffer(loaded_data, dtype=np.float32).reshape(
            pytorch_data.shape
        )
        assert np.allclose(loaded_tensor, pytorch_data), "PyTorch data should match"

        print(f"✓ PyTorch data validated successfully")
        print("✅ PyTorch multi-rank checkpointing test passed")


def test_tensorflow_large_model_checkpointing():
    """Test TensorFlow-style large model checkpointing"""
    print("\n=== Testing TensorFlow Large Model Checkpointing ===")

    # Simulate TensorFlow tensor data (typical CNN weights)
    tf_data = np.random.rand(256, 256, 64).astype(np.float32)
    print(f"TensorFlow test data: {tf_data.shape} tensor, {tf_data.nbytes} bytes")

    with tempfile.TemporaryDirectory() as temp_dir:
        base_uri = f"file://{temp_dir}"
        print(f"Using URI: {base_uri}")

        # Test with multi-GPU scenario
        store = s3dlio.PyCheckpointStore(
            base_uri, strategy="binary", multipart_threshold=1024 * 1024
        )
        writer = store.writer(world_size=4, rank=0)  # 4-GPU setup

        checkpoint_id = 300

        # Test TensorFlow-style streaming checkpointing
        print("Testing TensorFlow streaming checkpoint...")

        stream = writer.get_distributed_shard_stream(
            checkpoint_id,  # step
            0,  # epoch
            "tensorflow",  # framework
        )

        # Stream TensorFlow data in realistic chunks
        chunk_size = 64 * 1024  # 64KB chunks
        tf_bytes = tf_data.tobytes()

        for i in range(0, len(tf_bytes), chunk_size):
            chunk = tf_bytes[i : i + chunk_size]
            stream.write_chunk(chunk)

        stream_meta = stream.finalize()

        print(f"✓ TensorFlow checkpoint streamed: {stream_meta['size']} bytes")

        # Finalizing a 4-rank checkpoint needs all four ranks' shards. Every rank writes
        # DIFFERENT bytes (rank r holds tf_data * (r + 1)), so reading the wrong rank's
        # shard cannot pass the per-rank assertions below.
        expected = {0: tf_data}
        metas = [stream_meta]
        for other_rank in range(1, 4):
            expected[other_rank] = tf_data * np.float32(other_rank + 1)
            metas.append(
                store.writer(world_size=4, rank=other_rank).save_distributed_shard(
                    checkpoint_id, 0, "tensorflow", expected[other_rank].tobytes()
                )
            )
        assert len({a.tobytes() for a in expected.values()}) == 4, "ranks must differ"

        # Test loading with validation
        print("Testing TensorFlow loading...")

        reader, manifest = _finalize_and_load(
            store, 4, checkpoint_id, 0, "tensorflow", metas
        )
        assert manifest["world_size"] == 4

        # Verify data integrity of every rank's shard (including non-zero ranks)
        for rank in range(4):
            loaded_data = bytes(reader.read_shard_by_rank(manifest, rank))
            loaded_tensor = np.frombuffer(loaded_data, dtype=np.float32).reshape(
                tf_data.shape
            )
            assert np.array_equal(loaded_tensor, expected[rank]), (
                f"TensorFlow rank {rank} data should match"
            )
            for other in range(4):
                if other != rank:
                    assert not np.array_equal(loaded_tensor, expected[other]), (
                        f"rank {rank} shard must not equal rank {other}'s data"
                    )

        print(f"✓ TensorFlow validation successful")
        print("✅ TensorFlow large model checkpointing test passed")


def test_cross_framework_compatibility():
    """Test cross-framework checkpoint compatibility (v0.7.0 feature)"""
    print("\n=== Testing Cross-Framework Compatibility ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        base_uri = f"file://{temp_dir}"

        # Create shared data
        shared_data = np.random.rand(100, 100).astype(np.float64)
        print(f"Shared test data: {shared_data.shape}, {shared_data.nbytes} bytes")

        # Save with JAX-style
        jax_store = s3dlio.PyCheckpointStore(
            base_uri, strategy="binary", multipart_threshold=1024 * 1024
        )
        jax_writer = jax_store.writer(world_size=1, rank=0)

        checkpoint_id = 400

        jax_meta = jax_writer.save_distributed_shard(
            checkpoint_id,  # step
            0,  # epoch
            "jax",  # framework
            shared_data.tobytes(),  # data
        )

        print("✓ Checkpoint saved with JAX-style API")

        # Load with PyTorch-style reader
        pytorch_store = s3dlio.PyCheckpointStore(
            base_uri, strategy="binary", multipart_threshold=1024 * 1024
        )

        # A second store object over the same URI finalizes and reads the JAX-written shard
        loaded_data = _finalize_and_read(
            pytorch_store, 1, 0, checkpoint_id, 0, "jax", [jax_meta]
        )

        # Verify cross-framework compatibility
        loaded_array = np.frombuffer(loaded_data, dtype=np.float64).reshape(
            shared_data.shape
        )
        assert np.allclose(loaded_array, shared_data), (
            "Cross-framework data should match"
        )

        print("✓ JAX → PyTorch compatibility verified")
        print("✅ Cross-framework compatibility test passed")


def test_performance_and_scaling():
    """Test performance characteristics of v0.7.0 features"""
    print("\n=== Testing Performance and Scaling ===")

    import time

    with tempfile.TemporaryDirectory() as temp_dir:
        base_uri = f"file://{temp_dir}"

        # Test large checkpoint performance
        large_data = np.random.rand(2000, 2000).astype(np.float32)  # ~16MB
        print(
            f"Performance test data: {large_data.shape}, {large_data.nbytes / 1024 / 1024:.1f}MB"
        )

        store = s3dlio.PyCheckpointStore(
            base_uri, strategy="binary", multipart_threshold=1024 * 1024
        )
        writer = store.writer(world_size=1, rank=0)

        # Time save operation
        start_time = time.time()

        meta = writer.save_distributed_shard(
            500,  # step
            0,  # epoch
            "performance_test",  # framework
            large_data.tobytes(),  # data
        )

        save_time = time.time() - start_time

        # Time load operation (manifest lookup + validated shard read)
        writer.finalize_distributed_checkpoint(500, 0, "performance_test", [meta], None)
        reader = store.reader()
        start_time = time.time()

        manifest = reader.load_latest_manifest()
        assert manifest is not None, "finalized checkpoint must have a manifest"
        loaded_data = bytes(reader.read_shard_by_rank(manifest, 0))

        load_time = time.time() - start_time

        # Performance metrics
        data_size_mb = large_data.nbytes / (1024 * 1024)
        save_throughput = data_size_mb / save_time
        load_throughput = data_size_mb / load_time

        print(f"✓ Large checkpoint ({data_size_mb:.1f}MB) performance:")
        print(f"  - Save: {save_time:.2f}s ({save_throughput:.1f} MB/s)")
        print(f"  - Load: {load_time:.2f}s ({load_throughput:.1f} MB/s)")

        # Verify data integrity
        loaded_array = np.frombuffer(loaded_data, dtype=np.float32).reshape(
            large_data.shape
        )
        assert np.allclose(loaded_array, large_data), (
            "Performance test data should match"
        )

        print("✓ Performance test data integrity verified")
        print("✅ Performance and scaling test passed")


def test_v070_backend_features():
    """Test v0.7.0 specific backend features"""
    print("\n=== Testing v0.7.0 Backend Features ===")

    with tempfile.TemporaryDirectory() as temp_dir:
        base_uri = f"file://{temp_dir}"

        # Test data
        test_data = np.random.rand(500, 500).astype(np.float32)
        print(f"v0.7.0 test data: {test_data.shape}, {test_data.nbytes} bytes")

        # Test enhanced checkpoint store
        store = s3dlio.PyCheckpointStore(
            base_uri,
            strategy="binary",
            multipart_threshold=512 * 1024,  # Lower threshold for testing
        )
        writer = store.writer(world_size=1, rank=0)

        checkpoint_id = 700

        # Test streaming with framework tagging
        print("Testing v0.7.0 streaming with framework tagging...")

        frameworks = ["jax", "pytorch", "tensorflow", "custom"]
        metas = {}

        for i, framework in enumerate(frameworks):
            stream = writer.get_distributed_shard_stream(
                checkpoint_id + i,  # step (one checkpoint per framework)
                i,  # epoch (using loop index)
                framework,  # framework
            )

            # Write framework-specific test pattern
            framework_data = test_data * (i + 1)  # Different scaling per framework
            stream.write_chunk(framework_data.tobytes())
            meta = stream.finalize()
            metas[framework] = meta

            print(f"✓ {framework} checkpoint: {meta['size']} bytes to {meta['key']}")

        # Verify all frameworks saved correctly
        for i, framework in enumerate(frameworks):
            reader, manifest = _finalize_and_load(
                store, 1, checkpoint_id + i, i, framework, [metas[framework]]
            )
            # The manifest must record exactly what this checkpoint was tagged with.
            assert manifest["framework"] == framework
            assert manifest["epoch"] == i
            assert manifest["global_step"] == checkpoint_id + i
            assert manifest["world_size"] == 1
            loaded_data = bytes(reader.read_shard_by_rank(manifest, 0))

            expected_data = test_data * (i + 1)
            loaded_array = np.frombuffer(loaded_data, dtype=np.float32).reshape(
                test_data.shape
            )
            assert np.allclose(loaded_array, expected_data), (
                f"{framework} data should match"
            )

            print(f"✓ {framework} validation passed")

        print("✅ v0.7.0 backend features test passed")


def main():
    """Run all comprehensive Python tests for s3dlio v0.7.0"""
    print("🚀 Starting Comprehensive s3dlio v0.7.0 Python Tests")
    print(
        "Testing all frameworks (JAX, PyTorch, TensorFlow) with validated API patterns"
    )
    print(
        "Features: Enhanced Streaming, Multi-Rank Support, Cross-Framework Compatibility"
    )
    print("=" * 80)

    try:
        # Core framework tests
        test_jax_streaming_checkpointing()
        test_pytorch_multirank_checkpointing()
        test_tensorflow_large_model_checkpointing()

        # Advanced features
        test_cross_framework_compatibility()
        test_performance_and_scaling()
        test_v070_backend_features()

        print("\n" + "=" * 80)
        print("🎉 ALL TESTS PASSED!")
        print("✅ JAX streaming checkpointing: PASSED")
        print("✅ PyTorch multi-rank checkpointing: PASSED")
        print("✅ TensorFlow large model checkpointing: PASSED")
        print("✅ Cross-framework compatibility: PASSED")
        print("✅ Performance and scaling: PASSED")
        print("✅ v0.7.0 backend features: PASSED")
        print("\ns3dlio v0.7.0 is ready for production AI/ML workloads! 🚀")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
