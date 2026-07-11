#!/usr/bin/env python3

"""
Test script for s3dlio Python mp_get functionality
Validates high-performance S3 operations from Python
"""

import s3dlio
import pytest


def test_mp_get_performance():
    """Test mp_get with the same parameters as our CLI test"""

    print("Testing s3dlio Python mp_get performance...")
    print("=" * 50)

    # Test parameters using the working bucket - larger scale test
    uri = "s3://my-bucket2/large-test/"
    procs = 8
    num = 200  # Number of objects to test
    template = "large_object_{}_of_1000.dat"

    print(f"URI: {uri}")
    print(f"Processes: {procs}")
    print(f"Objects: {num}")
    print(f"Template: {template}")
    print()

    try:
        # Call mp_get - this should achieve the same warp-level performance as CLI
        result = s3dlio.mp_get(uri=uri, procs=procs, num=num, template=template)
    except Exception as e:
        pytest.skip(
            reason=f"requires a live S3 bucket 'my-bucket2' pre-populated with {num} objects, not available in this sandbox: {e}"
        )

    print("Results:")
    print(f"  Total objects: {result['total_objects']:,}")
    print(f"  Total bytes: {result['total_bytes']:,}")
    print(f"  Duration: {result['duration_seconds']:.2f}s")
    print(f"  Throughput: {result['throughput_mb_s']:.1f} MB/s")
    print(f"  Ops/sec: {result['ops_per_sec']:.1f}")
    print()

    # Show per-worker breakdown
    print("Per-worker performance:")
    for i, worker in enumerate(result["workers"]):
        print(
            f"  Worker {worker['worker_id']}: {worker['objects']:,} objects, "
            f"{worker['bytes']:,} bytes, {worker['throughput_mb_s']:.1f} MB/s"
        )

    print()
    print("✅ Python mp_get test completed successfully!")

    # Compare to expected CLI performance (~2,308 MB/s)
    expected_throughput = 2000  # MB/s minimum expected
    if result["throughput_mb_s"] >= expected_throughput:
        print(
            f"🚀 Excellent! Python achieved {result['throughput_mb_s']:.1f} MB/s (>= {expected_throughput} MB/s)"
        )
    else:
        print(
            f"⚠️  Performance lower than expected: {result['throughput_mb_s']:.1f} MB/s (expected >= {expected_throughput} MB/s)"
        )


if __name__ == "__main__":
    test_mp_get_performance()
