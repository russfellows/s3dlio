"""
Test checkpoint functionality through Python bindings
"""
import os
import sys
import tempfile
import pytest

# Add the s3dlio module to the path
sys.path.insert(0, '/home/eval/Documents/Rust-Devel/s3dlio/python')

try:
    import s3dlio
    HAS_S3DLIO = True
except ImportError:
    HAS_S3DLIO = False

@pytest.mark.skipif(not HAS_S3DLIO, reason="s3dlio module not available")
def test_basic_checkpoint_operations():
    """Test basic checkpoint write and read operations"""
    with tempfile.TemporaryDirectory() as temp_dir:
        uri = f"file://{temp_dir}"
        
        # Create checkpoint store
        store = s3dlio.PyCheckpointStore.open_auto_config(uri)
        
        # Test data to checkpoint
        test_data = b"test checkpoint data for rank 0"
        user_metadata = {
            "model_type": "transformer",
            "training_step": 42
        }
        
        # Write a checkpoint using the writer
        writer = s3dlio.PyCheckpointWriter(store, 1, 0)  # world_size=1, rank=0
        shard_key = writer.put_shard(test_data, user_metadata)
        manifest_key = writer.write_manifest(1, "pytorch", [], user_metadata)
        
        print(f"✓ Wrote checkpoint: manifest={manifest_key}, shard={shard_key}")
        
        # Read back the checkpoint
        reader = s3dlio.PyCheckpointReader(store)
        latest_manifest = reader.load_latest_manifest()
        assert latest_manifest is not None, "Should have found a checkpoint"
        
        # Verify metadata
        assert latest_manifest["epoch"] == 1
        assert latest_manifest["framework"] == "pytorch"
        assert len(latest_manifest["shards"]) == 1
        assert latest_manifest["shards"][0]["rank"] == 0
        
        # Read the shard data
        loaded_data = reader.read_shard_by_rank(latest_manifest, 0)
        assert loaded_data == test_data
        
        print("✓ Successfully read back checkpoint data")

@pytest.mark.skipif(not HAS_S3DLIO, reason="s3dlio module not available")
def test_checkpoint_versioning():
    """Test checkpoint versioning and epoch management"""
    with tempfile.TemporaryDirectory() as temp_dir:
        uri = f"file://{temp_dir}"
        
        store = s3dlio.PyCheckpointStore.open(uri)
        
        # Write multiple checkpoint epochs
        epochs = [1, 3, 2, 5]  # Out of order to test sorting
        framework = "pytorch"
        
        for i, epoch in enumerate(epochs):
            test_data = f"data for epoch {epoch}".encode()
            user_meta = {"epoch": epoch}
            
            writer = s3dlio.PyCheckpointWriter(store, 1, 0)
            shard_key = writer.put_shard(test_data, None)
            manifest_key = writer.write_manifest(epoch, framework, [], user_meta)
            
            print(f"✓ Wrote checkpoint for epoch {epoch}")
        
        # Verify latest checkpoint is epoch 5 (highest)
        reader = s3dlio.PyCheckpointReader(store)
        latest = reader.load_latest_manifest()
        assert latest is not None
        assert latest["epoch"] == 5
        
        data = reader.read_shard_by_rank(latest, 0)
        assert data == b"data for epoch 5"
        
        print(f"✓ Latest checkpoint correctly identified as epoch {latest['epoch']}")

@pytest.mark.skipif(not HAS_S3DLIO, reason="s3dlio module not available")
def test_checkpoint_validation():
    """Test checkpoint validation and error handling"""
    with tempfile.TemporaryDirectory() as temp_dir:
        uri = f"file://{temp_dir}"
        
        store = s3dlio.PyCheckpointStore.open(uri)
        
        # Test missing checkpoint
        reader = s3dlio.PyCheckpointReader(store)
        result = reader.load_latest_manifest()
        assert result is None, "Should return None when no checkpoint exists"
        
        # Write a simple valid checkpoint
        writer = s3dlio.PyCheckpointWriter(store, 1, 0)
        shard_key = writer.put_shard(b"test data", None)
        manifest_key = writer.write_manifest(1, "pytorch", [], None)
        
        # Now we should be able to read it
        manifest = reader.load_latest_manifest()
        assert manifest is not None
        assert manifest["epoch"] == 1
        assert manifest["framework"] == "pytorch"
        assert len(manifest["shards"]) == 1
        
        print("✓ Validation test completed")

if __name__ == "__main__":
    # Run tests directly if executed as script
    if HAS_S3DLIO:
        test_basic_checkpoint_operations()
        test_checkpoint_versioning() 
        test_checkpoint_validation()
        print("✓ All Python checkpoint tests passed!")
    else:
        print("❌ s3dlio module not available - install the wheel first")
