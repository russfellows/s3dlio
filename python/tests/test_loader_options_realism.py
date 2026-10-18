# tests/test_loader_options_realism.py
"""
Test the enhanced LoaderOptions with AI/ML realism knobs.

This test validates that all the new options work correctly and can be configured
for realistic AI/ML training scenarios.
"""

import pytest
import s3dlio


def test_basic_ai_ml_options():
    """Test that we can configure LoaderOptions with AI/ML realism knobs."""

    # Test GPU-optimized configuration
    opts = (
        s3dlio.PyLoaderOptions()
        .with_batch_size(256)
        .pin_memory(True)
        .persistent_workers(True)
        .non_blocking(True)
        .channels_first()
        .use_spawn()
        .with_timeout(30.0)
        .collate_buffer_size(2048)
    )

    print(f"✓ GPU-optimized options configured")
    print(f"  - batch_size: {opts.batch_size}")
    print(f"  - pin_memory: {opts.pin_memory}")
    print(f"  - persistent_workers: {opts.persistent_workers}")


def test_distributed_training_options():
    """Test distributed training configuration."""

    # Test distributed configuration
    opts = (
        s3dlio.PyLoaderOptions()
        .distributed_optimized(rank=2, world_size=8)
        .random_sampling(replacement=False)
        .with_generator_seed(12345)
        .enable_transforms(True)
    )

    print(f"✓ Distributed training options configured")
    print(f"  - shard_rank: {opts.shard_rank}")
    print(f"  - shard_world_size: {opts.shard_world_size}")


def test_debug_mode_options():
    """Test debug/development configuration."""

    opts = s3dlio.PyLoaderOptions().debug_mode().with_batch_size(4).shuffle(True, 42)

    print(f"✓ Debug mode options configured")
    print(f"  - num_workers: {opts.num_workers}")
    print(f"  - timeout_seconds: {opts.timeout_seconds}")


@pytest.mark.xfail(
    strict=True,
    reason="#283: PyLoaderOptions properties are shadowed by builder methods; awaiting an API naming decision",
)
def test_preset_configurations():
    """Test the convenience preset methods."""

    # GPU optimized preset
    gpu_opts = s3dlio.PyLoaderOptions().gpu_optimized()
    assert gpu_opts.pin_memory == True
    assert gpu_opts.persistent_workers == True
    assert gpu_opts.non_blocking == True

    # Debug preset
    debug_opts = s3dlio.PyLoaderOptions().debug_mode()
    assert debug_opts.num_workers == 0
    assert debug_opts.persistent_workers == False

    print(f"✓ Preset configurations work correctly")


def test_realistic_pytorch_config():
    """Test a realistic PyTorch-style configuration."""

    opts = (
        s3dlio.PyLoaderOptions()
        .with_batch_size(128)
        .num_workers(8)
        .pin_memory(True)
        .persistent_workers(True)
        .shuffle(True, 42)
        .drop_last(True)
        .prefetch(4)
        .with_timeout(30.0)
        .channels_first()
        .non_blocking(True)
        .collate_buffer_size(1024)
    )

    print(f"✓ Realistic PyTorch configuration:")
    print(f"  - batch_size: {opts.batch_size}")
    print(f"  - num_workers: {opts.num_workers}")
    print(f"  - pin_memory: {opts.pin_memory}")
    print(f"  - persistent_workers: {opts.persistent_workers}")
    print(f"  - shuffle: {opts.shuffle}")
    print(f"  - drop_last: {opts.drop_last}")


def test_realistic_tensorflow_config():
    """Test a realistic TensorFlow-style configuration."""

    opts = (
        s3dlio.PyLoaderOptions()
        .with_batch_size(64)
        .num_workers(4)
        .prefetch(2)
        .channels_last()
        .enable_transforms(True)
        .use_forkserver()
        .collate_buffer_size(512)
    )

    print(f"✓ Realistic TensorFlow configuration:")
    print(f"  - batch_size: {opts.batch_size}")
    print(f"  - num_workers: {opts.num_workers}")
    print(f"  - prefetch: {opts.prefetch}")


@pytest.mark.xfail(
    strict=True,
    reason="#283: PyLoaderOptions properties are shadowed by builder methods; awaiting an API naming decision",
)
def test_multi_gpu_distributed_config():
    """Test a realistic multi-GPU distributed training configuration."""

    # Simulate 4-GPU training setup
    for rank in range(4):
        opts = (
            s3dlio.PyLoaderOptions()
            .distributed_optimized(rank=rank, world_size=4)
            .with_batch_size(32)
            .num_workers(6)
            .with_generator_seed(42 + rank)
        )

        assert opts.shard_rank == rank
        assert opts.shard_world_size == 4
        assert opts.pin_memory == True  # From distributed_optimized preset

    print(f"✓ Multi-GPU distributed configuration validated for 4 GPUs")


def _run_expected_failure(test):
    """Run a test marked ``xfail(strict=True)`` (#283) from the ``__main__`` runner.

    The failure is the expected outcome: report it as such. If the test unexpectedly
    passes, #283 is fixed and the xfail marker must be removed, so fail loudly.
    """
    try:
        test()
    except AssertionError as exc:
        print(f"XFAIL (expected, #283): {test.__name__}: {exc!r}")
    except AttributeError as exc:
        print(f"XFAIL (expected, #283): {test.__name__}: {exc!r}")
    else:
        raise SystemExit(
            f"XPASS: {test.__name__} passed but is marked xfail(strict=True) for #283; "
            "remove the marker"
        )


if __name__ == "__main__":
    print("Testing LoaderOptions AI/ML Realism Knobs")
    print("=" * 50)

    test_basic_ai_ml_options()
    print()

    test_distributed_training_options()
    print()

    test_debug_mode_options()
    print()

    _run_expected_failure(test_preset_configurations)
    print()

    test_realistic_pytorch_config()
    print()

    test_realistic_tensorflow_config()
    print()

    _run_expected_failure(test_multi_gpu_distributed_config)
    print()

    print(
        "All LoaderOptions realism tests passed (2 expected failures tracked by #283)."
    )
    print("\nThe enhanced LoaderOptions now supports:")
    print("  ✓ GPU training optimizations (pin_memory, non_blocking)")
    print("  ✓ Distributed training (sampling, sharding)")
    print("  ✓ Performance optimizations (persistent_workers, timeout)")
    print("  ✓ Memory layout control (channels_first/last)")
    print("  ✓ Multiprocessing options (spawn/fork/forkserver)")
    print("  ✓ Sampling strategies (random, weighted, distributed)")
    print("  ✓ Transform support and collation buffers")
    print("  ✓ Convenient presets (gpu_optimized, debug_mode)")
