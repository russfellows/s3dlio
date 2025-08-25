#!/usr/bin/env python3
"""
Framework integration tests for s3dlio checkpoint functionality.
Tests checkpoint integration with PyTorch, JAX, and TensorFlow.
"""

import tempfile
import os
import numpy as np

def test_pytorch_integration():
    """Test checkpoint integration with PyTorch tensors"""
    print("Testing PyTorch integration...")
    
    try:
        import torch
        import s3dlio
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = f"file://{temp_dir}/pytorch_checkpoint"
            store = s3dlio.PyCheckpointStore(base_path, None, None)
            
            # Create some PyTorch tensors
            model_state = {
                'weight': torch.randn(10, 5),
                'bias': torch.randn(10),
                'param': torch.tensor([1.0, 2.0, 3.0])
            }
            
            # Serialize to bytes
            import pickle
            serialized_data = pickle.dumps(model_state)
            
            # Save checkpoint
            writer = store.writer(world_size=1, rank=0)
            shard_key = writer.save_distributed_shard(
                step=100,
                epoch=10,
                framework="pytorch",
                data=serialized_data
            )
            
            manifest_key = writer.finalize_distributed_checkpoint(
                step=100,
                epoch=10,
                framework="pytorch",
                shard_metas=[shard_key],
                user_meta=None
            )
            
            print(f"✓ Saved PyTorch checkpoint: {manifest_key}")
            
            # Load checkpoint
            reader = store.reader()
            manifest = reader.load_latest_manifest()
            loaded_data = reader.read_shard_by_rank(manifest, 0)
            
            # Deserialize and verify
            loaded_state = pickle.loads(loaded_data)
            
            # Check that tensors match
            assert torch.allclose(model_state['weight'], loaded_state['weight'])
            assert torch.allclose(model_state['bias'], loaded_state['bias'])
            assert torch.allclose(model_state['param'], loaded_state['param'])
            
            print("✓ PyTorch checkpoint data verified successfully")
            return True
            
    except ImportError:
        print("✗ PyTorch not available")
        return False
    except Exception as e:
        print(f"✗ PyTorch test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_jax_integration():
    """Test checkpoint integration with JAX arrays"""
    print("\nTesting JAX integration...")
    
    try:
        import jax
        import jax.numpy as jnp
        import s3dlio
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = f"file://{temp_dir}/jax_checkpoint"
            store = s3dlio.PyCheckpointStore(base_path, None, None)
            
            # Create some JAX arrays
            key = jax.random.PRNGKey(42)
            model_params = {
                'weights': jax.random.normal(key, (20, 10)),
                'biases': jax.random.normal(key, (10,)),
                'scalars': jnp.array([0.1, 0.2, 0.3])
            }
            
            # Convert to numpy for serialization
            numpy_params = jax.tree.map(lambda x: np.asarray(x), model_params)
            
            # Serialize to bytes
            import pickle
            serialized_data = pickle.dumps(numpy_params)
            
            # Save checkpoint
            writer = store.writer(world_size=1, rank=0)
            shard_key = writer.save_distributed_shard(
                step=200,
                epoch=20,
                framework="jax",
                data=serialized_data
            )
            
            manifest_key = writer.finalize_distributed_checkpoint(
                step=200,
                epoch=20,
                framework="jax",
                shard_metas=[shard_key],
                user_meta=None
            )
            
            print(f"✓ Saved JAX checkpoint: {manifest_key}")
            
            # Load checkpoint
            reader = store.reader()
            manifest = reader.load_latest_manifest()
            loaded_data = reader.read_shard_by_rank(manifest, 0)
            
            # Deserialize and verify
            loaded_numpy = pickle.loads(loaded_data)
            loaded_jax = jax.tree.map(lambda x: jnp.asarray(x), loaded_numpy)
            
            # Check that arrays match
            assert jnp.allclose(model_params['weights'], loaded_jax['weights'])
            assert jnp.allclose(model_params['biases'], loaded_jax['biases'])
            assert jnp.allclose(model_params['scalars'], loaded_jax['scalars'])
            
            print("✓ JAX checkpoint data verified successfully")
            return True
            
    except ImportError:
        print("✗ JAX not available")
        return False
    except Exception as e:
        print(f"✗ JAX test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tensorflow_integration():
    """Test checkpoint integration with TensorFlow tensors"""
    print("\nTesting TensorFlow integration...")
    
    try:
        import tensorflow as tf
        import s3dlio
        
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = f"file://{temp_dir}/tensorflow_checkpoint"
            store = s3dlio.PyCheckpointStore(base_path, None, None)
            
            # Create some TensorFlow tensors
            model_vars = {
                'dense_weight': tf.Variable(tf.random.normal([15, 8])),
                'dense_bias': tf.Variable(tf.zeros([8])),
                'learning_rate': tf.Variable(0.001)
            }
            
            # Convert to numpy for serialization
            numpy_vars = {k: v.numpy() for k, v in model_vars.items()}
            
            # Serialize to bytes
            import pickle
            serialized_data = pickle.dumps(numpy_vars)
            
            # Save checkpoint
            writer = store.writer(world_size=1, rank=0)
            shard_key = writer.save_distributed_shard(
                step=300,
                epoch=30,
                framework="tensorflow",
                data=serialized_data
            )
            
            manifest_key = writer.finalize_distributed_checkpoint(
                step=300,
                epoch=30,
                framework="tensorflow",
                shard_metas=[shard_key],
                user_meta=None
            )
            
            print(f"✓ Saved TensorFlow checkpoint: {manifest_key}")
            
            # Load checkpoint
            reader = store.reader()
            manifest = reader.load_latest_manifest()
            loaded_data = reader.read_shard_by_rank(manifest, 0)
            
            # Deserialize and verify
            loaded_numpy = pickle.loads(loaded_data)
            
            # Check that arrays match
            assert np.allclose(model_vars['dense_weight'].numpy(), loaded_numpy['dense_weight'])
            assert np.allclose(model_vars['dense_bias'].numpy(), loaded_numpy['dense_bias'])
            assert np.allclose(model_vars['learning_rate'].numpy(), loaded_numpy['learning_rate'])
            
            print("✓ TensorFlow checkpoint data verified successfully")
            return True
            
    except ImportError:
        print("✗ TensorFlow not available")
        return False
    except Exception as e:
        print(f"✗ TensorFlow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multi_backend_checkpointing():
    """Test checkpointing across different storage backends"""
    print("\nTesting multi-backend checkpointing...")
    
    try:
        import s3dlio
        
        # Test with different strategies
        strategies = [None, "flat", "round_robin"]
        
        for strategy in strategies:
            with tempfile.TemporaryDirectory() as temp_dir:
                base_path = f"file://{temp_dir}/strategy_{strategy or 'default'}"
                
                # Test with different strategies
                store = s3dlio.PyCheckpointStore(base_path, strategy, None)
                
                # Save a simple checkpoint
                writer = store.writer(world_size=1, rank=0)
                test_data = f"Strategy test: {strategy}".encode()
                
                shard_key = writer.save_distributed_shard(
                    step=1,
                    epoch=1,
                    framework="test",
                    data=test_data
                )
                
                manifest_key = writer.finalize_distributed_checkpoint(
                    step=1,
                    epoch=1,
                    framework="test",
                    shard_metas=[shard_key],
                    user_meta=None
                )
                
                # Verify we can read it back
                reader = store.reader()
                manifest = reader.load_latest_manifest()
                loaded_data = reader.read_shard_by_rank(manifest, 0)
                
                assert loaded_data == test_data
                print(f"✓ Strategy '{strategy or 'default'}' works correctly")
        
        print("✓ All backend strategies work correctly")
        return True
        
    except Exception as e:
        print(f"✗ Multi-backend test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all framework integration tests"""
    print("=" * 70)
    print("S3DLIO Framework Integration Tests")
    print("=" * 70)
    
    tests = [
        test_pytorch_integration,
        test_jax_integration,
        test_tensorflow_integration,
        test_multi_backend_checkpointing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 70)
    print(f"Framework Integration Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All framework integration tests PASSED!")
        return True
    else:
        print("❌ Some framework integration tests FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
