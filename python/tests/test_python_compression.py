#!/usr/bin/env python3
"""
Test script to verify Python compression integration is working
"""

import sys
import tempfile

try:
    from s3dlio import PyCheckpointStore as CheckpointStore

    print("✓ Successfully imported CheckpointStore")
except ImportError as e:
    print(f"✗ Failed to import CheckpointStore: {e}")
    sys.exit(1)


def test_compression_integration():
    """Test that compression_level parameter is accepted and works"""

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test without compression
        uri = f"file://{temp_dir}/test_no_compression"
        CheckpointStore(uri)
        print("✓ Created CheckpointStore without compression")

        # Test with compression level 3
        uri2 = f"file://{temp_dir}/test_with_compression"
        store2 = CheckpointStore(uri2, compression_level=3)
        print("✓ Created CheckpointStore with compression_level=3")

        # Test with compression level 9
        uri3 = f"file://{temp_dir}/test_high_compression"
        CheckpointStore(uri3, compression_level=9)
        print("✓ Created CheckpointStore with compression_level=9")

        # Test simple save/load with compression
        test_data = (
            b"Hello, compression world! " * 100
        )  # Repeating data that compresses well

        # Save with compression
        manifest_key = store2.save(100, 10, "test", test_data, None)
        print(f"✓ Saved data with compression: {manifest_key}")

        # Load it back
        loaded_data = store2.load_latest()
        assert bytes(loaded_data) == test_data, "Loaded data doesn't match original"
        print("✓ Successfully loaded compressed data")


if __name__ == "__main__":
    print("Testing Python compression integration...")
    try:
        test_compression_integration()
        print("\n🎉 All compression integration tests passed!")
    except AssertionError as e:
        print(f"\n❌ Compression integration test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
