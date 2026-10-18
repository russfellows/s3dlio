# tests/test_loader_options_realism.py
"""
Test the enhanced LoaderOptions with AI/ML realism knobs.

This test validates that all the new options work correctly and can be configured
for realistic AI/ML training scenarios.
"""

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

    print("✓ GPU-optimized options configured")
    print(f"  - batch_size: {opts.batch_size}")
    print(f"  - pin_memory: {opts.pin_memory_value}")
    print(f"  - persistent_workers: {opts.persistent_workers_value}")


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

    print("✓ Distributed training options configured")
    print(f"  - shard_rank: {opts.shard_rank}")
    print(f"  - shard_world_size: {opts.shard_world_size}")


def test_debug_mode_options():
    """Test debug/development configuration."""

    opts = s3dlio.PyLoaderOptions().debug_mode().with_batch_size(4).shuffle(True, 42)

    print("✓ Debug mode options configured")
    print(f"  - num_workers: {opts.num_workers_value}")
    print(f"  - timeout_seconds: {opts.timeout_seconds}")


def test_preset_configurations():
    """Test the convenience preset methods."""

    # GPU optimized preset
    gpu_opts = s3dlio.PyLoaderOptions().gpu_optimized()
    assert gpu_opts.pin_memory_value == True
    assert gpu_opts.persistent_workers_value == True
    assert gpu_opts.non_blocking_value == True

    # Debug preset
    debug_opts = s3dlio.PyLoaderOptions().debug_mode()
    assert debug_opts.num_workers_value == 0
    assert debug_opts.persistent_workers_value == False

    print("✓ Preset configurations work correctly")


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

    print("✓ Realistic PyTorch configuration:")
    print(f"  - batch_size: {opts.batch_size}")
    print(f"  - num_workers: {opts.num_workers_value}")
    print(f"  - pin_memory: {opts.pin_memory_value}")
    print(f"  - persistent_workers: {opts.persistent_workers_value}")
    print(f"  - shuffle: {opts.shuffle_value}")
    print(f"  - drop_last: {opts.drop_last_value}")


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

    print("✓ Realistic TensorFlow configuration:")
    print(f"  - batch_size: {opts.batch_size}")
    print(f"  - num_workers: {opts.num_workers_value}")
    print(f"  - prefetch: {opts.prefetch_value}")


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
        assert opts.pin_memory_value == True  # From distributed_optimized preset

    print("✓ Multi-GPU distributed configuration validated for 4 GPUs")


def test_shadowed_values_readable_via_value_properties():
    """#283: the seven values shadowed by builder methods are read via ``*_value``."""

    opts = (
        s3dlio.PyLoaderOptions()
        .pin_memory(True)
        .num_workers(8)
        .shuffle(True, 42)
        .drop_last(True)
        .prefetch(4)
        .persistent_workers(True)
        .non_blocking(True)
    )

    assert opts.pin_memory_value is True
    assert opts.num_workers_value == 8
    assert opts.shuffle_value is True
    assert opts.drop_last_value is True
    assert opts.prefetch_value == 4
    assert opts.persistent_workers_value is True
    assert opts.non_blocking_value is True
    assert isinstance(opts.num_workers_value, int)
    assert isinstance(opts.prefetch_value, int)

    off = opts.pin_memory(False).drop_last(False).shuffle(False, 42).num_workers(0)
    assert off.pin_memory_value is False
    assert off.drop_last_value is False
    assert off.shuffle_value is False
    assert off.num_workers_value == 0


def test_builders_still_chain_and_remain_methods():
    """#283 option A: the same-named builders keep their name and return the options."""

    opts = s3dlio.PyLoaderOptions()
    for name, args in [
        ("pin_memory", (True,)),
        ("num_workers", (2,)),
        ("shuffle", (True, 1)),
        ("drop_last", (True,)),
        ("prefetch", (3,)),
        ("persistent_workers", (True,)),
        ("non_blocking", (True,)),
    ]:
        assert callable(getattr(opts, name)), name
        result = getattr(opts, name)(*args)
        assert isinstance(result, s3dlio.PyLoaderOptions), name


def test_to_dict_matches_properties():
    """#283: ``to_dict()`` reports every settable option, matching the property values."""

    expected_keys = {
        "batch_size",
        "drop_last",
        "shuffle",
        "seed",
        "num_workers",
        "prefetch",
        "auto_tune",
        "pin_memory",
        "persistent_workers",
        "timeout_seconds",
        "non_blocking",
        "shard_rank",
        "shard_world_size",
        "generator_seed",
        "enable_transforms",
        "collate_buffer_size",
        "multiprocessing_context",
        "memory_format",
        "sampler_type",
        "sampler_replacement",
        "sampler_rank",
        "sampler_world_size",
    }

    def check(opts):
        d = opts.to_dict()
        assert isinstance(d, dict)
        assert set(d) == expected_keys
        assert d["batch_size"] == opts.batch_size
        assert d["seed"] == opts.seed
        assert d["timeout_seconds"] == opts.timeout_seconds
        assert d["shard_rank"] == opts.shard_rank
        assert d["shard_world_size"] == opts.shard_world_size
        for name in (
            "drop_last",
            "shuffle",
            "num_workers",
            "prefetch",
            "pin_memory",
            "persistent_workers",
            "non_blocking",
        ):
            assert d[name] == getattr(opts, name + "_value"), name
        return d

    d = check(s3dlio.PyLoaderOptions())
    assert d["batch_size"] == 32

    d = check(
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
    assert d["batch_size"] == 128
    assert d["timeout_seconds"] == 30.0
    assert d["memory_format"] == "channels_first"
    assert d["collate_buffer_size"] == 1024

    d = check(s3dlio.PyLoaderOptions().gpu_optimized())
    assert d["pin_memory"] is True and d["persistent_workers"] is True

    d = check(s3dlio.PyLoaderOptions().distributed_optimized(rank=2, world_size=8))
    assert d["shard_rank"] == 2 and d["shard_world_size"] == 8

    d = check(s3dlio.PyLoaderOptions().debug_mode())
    assert d["num_workers"] == 0 and d["persistent_workers"] is False


def test_to_dict_sampler_parameters_are_lossless():
    """#283: ``to_dict()`` exposes sampler parameters, not just the sampler name."""

    def sampler(opts):
        d = opts.to_dict()
        return (
            d["sampler_type"],
            d["sampler_replacement"],
            d["sampler_rank"],
            d["sampler_world_size"],
        )

    base = s3dlio.PyLoaderOptions()
    assert sampler(base) == ("sequential", None, None, None)
    assert sampler(base.random_sampling(True)) == ("random", True, None, None)
    assert sampler(base.random_sampling(False)) == ("random", False, None, None)
    assert sampler(base.distributed_sampling(1, 4)) == (
        "distributed_random",
        None,
        1,
        4,
    )
    # The receiver of a sampler builder is unchanged.
    assert sampler(base) == ("sequential", None, None, None)


def test_builders_return_new_object_and_leave_original_unchanged():
    """Builders return a NEW options object; discarding the result has no effect."""
    a = s3dlio.PyLoaderOptions()
    b = a.pin_memory(True)
    assert b is not a
    assert a.pin_memory_value is False
    assert b.pin_memory_value is True

    c = a.num_workers(8)
    assert c is not a
    assert a.num_workers_value != 8
    assert c.num_workers_value == 8

    # Calling a builder and discarding the result changes nothing.
    a.shuffle(True, 7)
    assert a.shuffle_value is False


if __name__ == "__main__":
    print("Testing LoaderOptions AI/ML Realism Knobs")
    print("=" * 50)

    test_basic_ai_ml_options()
    print()

    test_distributed_training_options()
    print()

    test_debug_mode_options()
    print()

    test_preset_configurations()
    print()

    test_realistic_pytorch_config()
    print()

    test_realistic_tensorflow_config()
    print()

    test_multi_gpu_distributed_config()
    print()

    test_shadowed_values_readable_via_value_properties()
    print()

    test_builders_still_chain_and_remain_methods()
    print()

    test_to_dict_matches_properties()
    print()

    test_to_dict_sampler_parameters_are_lossless()
    print()

    test_builders_return_new_object_and_leave_original_unchanged()
    print()

    print("All LoaderOptions realism tests passed.")
    print("\nThe enhanced LoaderOptions now supports:")
    print("  ✓ GPU training optimizations (pin_memory, non_blocking)")
    print("  ✓ Distributed training (sampling, sharding)")
    print("  ✓ Performance optimizations (persistent_workers, timeout)")
    print("  ✓ Memory layout control (channels_first/last)")
    print("  ✓ Multiprocessing options (spawn/fork/forkserver)")
    print("  ✓ Sampling strategies (random, weighted, distributed)")
    print("  ✓ Transform support and collation buffers")
    print("  ✓ Convenient presets (gpu_optimized, debug_mode)")
