# S3DLIO Checkpoint Features

## Overview

S3DLIO v0.6.0 introduces comprehensive checkpointing capabilities modeled after the AWS S3 PyTorch Connector, providing high-performance distributed checkpointing across all supported storage backends. The checkpoint system is implemented primarily in Rust for optimal performance and safety, with Python bindings for seamless integration with ML frameworks.

## Key Features

### ✅ **Multi-Backend Support**
- **S3**: Amazon S3 with hot-spot avoidance strategies
- **Azure Blob**: Azure Blob Storage with optimized partitioning
- **File System**: Local and network file systems
- **Direct I/O**: High-performance O_DIRECT file operations

### ✅ **Framework Integration**
- **PyTorch**: Native tensor serialization and state dict management
- **JAX**: JAX array and parameter tree checkpointing
- **TensorFlow**: Variable and model state preservation
- **Framework Agnostic**: JSON manifest format supports any ML framework

### ✅ **Distributed Checkpointing**
- **Multi-rank coordination**: Distributed training checkpoint coordination
- **Shard management**: Automatic shard partitioning and metadata tracking
- **Manifest-based**: JSON manifests for checkpoint discovery and validation
- **Latest pointer management**: Automatic latest checkpoint tracking

### ✅ **Storage Optimization**
- **Path strategies**: Flat, RoundRobin, and Binary strategies for S3/Azure performance
- **Hot-spot avoidance**: Intelligent key distribution to prevent storage bottlenecks
- **Multipart uploads**: Efficient large checkpoint handling
- **Concurrent operations**: Parallel shard uploads and downloads

## Architecture

The checkpoint system follows a manifest-based architecture:

```
checkpoint_base/
├── manifests/
│   ├── ckpt-{step}-{timestamp}.json     # Checkpoint manifests
│   └── latest.json                      # Latest checkpoint pointer
└── shards/
    └── ckpt-{step}-{timestamp}/
        ├── rank-0.bin                   # Shard data files
        ├── rank-1.bin
        └── ...
```

### Components

1. **CheckpointStore**: Main interface for checkpoint operations
2. **CheckpointWriter**: Handles distributed checkpoint writing
3. **CheckpointReader**: Loads and validates checkpoints
4. **Manifest**: JSON metadata describing checkpoint structure
5. **Strategy**: Path organization for storage optimization

## API Reference

### Rust API

```rust
use s3dlio::checkpoint::{CheckpointStore, CheckpointConfig};

// Create checkpoint store
let store = CheckpointStore::open("s3://bucket/checkpoints")?;

// Write distributed checkpoint
let writer = store.writer(world_size, rank)?;
let shard_key = writer.put_shard(&data, None).await?;
let manifest_key = writer.write_manifest(epoch, "pytorch", shard_metas, None).await?;

// Read checkpoint
let reader = store.reader()?;
let manifest = reader.load_latest_manifest().await?;
let data = reader.read_shard_by_rank(&manifest, rank).await?;
```

### Python API

```python
import s3dlio

# Create checkpoint store
store = s3dlio.PyCheckpointStore("s3://bucket/checkpoints", None, None)

# Write distributed checkpoint
writer = store.writer(world_size=4, rank=0)
shard_key = writer.save_distributed_shard(
    step=100, epoch=10, framework="pytorch", data=checkpoint_data
)
manifest_key = writer.finalize_distributed_checkpoint(
    step=100, epoch=10, framework="pytorch", 
    shard_metas=[shard_key], user_meta=None
)

# Read checkpoint
reader = store.reader()
manifest = reader.load_latest_manifest()
data = reader.read_shard_by_rank(manifest, rank=0)
```

## Configuration

### Storage Strategies

- **Flat**: Simple flat key structure (default for file://)
- **RoundRobin**: Distributes keys across prefixes to avoid hot-spots
- **Binary**: Binary tree structure for optimal S3/Azure performance

```rust
let config = CheckpointConfig::new()
    .with_strategy(Strategy::RoundRobin)
    .with_multipart_threshold(50 * 1024 * 1024); // 50MB

let store = CheckpointStore::open_with_config("s3://bucket/checkpoints", config)?;
```

### Python Configuration

```python
store = s3dlio.PyCheckpointStore(
    uri="s3://bucket/checkpoints",
    strategy="round_robin",
    multipart_threshold=50 * 1024 * 1024
)
```

## Testing

### Test Coverage

The checkpoint system includes comprehensive test coverage across both Rust and Python:

**Total Tests: 28 tests (all passing)**

### Rust Tests (in `tests/` directory)

#### `test_checkpoint_integration.rs`
- Basic checkpoint operations
- Manifest validation and serialization
- Latest pointer management
- Single-rank checkpoint workflows

```bash
# Run basic checkpoint integration tests
cargo test --test test_checkpoint_integration --features extension-module
```

#### `test_checkpoint_advanced.rs` 
- Concurrent checkpoint operations
- Distributed multi-rank coordination
- Multiple epoch scenarios
- Strategy validation

```bash
# Run advanced checkpoint tests
cargo test --test test_checkpoint_advanced --features extension-module
```

### Python Tests (in `python/tests/` directory)

#### `test_checkpoint_basic_python.py`
- **Basic operations**: Store creation, shard saving, manifest writing
- **Multi-rank distributed**: 3-rank distributed checkpoint simulation
- **Versioning**: Multiple epoch checkpoints and latest management

```bash
# Run basic Python checkpoint tests
python ./python/tests/test_checkpoint_basic_python.py
```

#### `test_checkpoint_framework_integration.py`
- **PyTorch integration**: Tensor serialization and state dict handling
- **JAX integration**: JAX array and parameter tree checkpointing  
- **TensorFlow integration**: Variable and model state preservation
- **Multi-backend strategies**: Testing all storage strategies

```bash
# Run framework integration tests
python ./python/tests/test_checkpoint_framework_integration.py
```

### Running All Tests

```bash
# Run all Rust tests
cargo test --features extension-module

# Run all Python tests
python ./python/tests/test_checkpoint_basic_python.py
python ./python/tests/test_checkpoint_framework_integration.py

# Or run both together
echo "===== BASIC TESTS =====" && \
python ./python/tests/test_checkpoint_basic_python.py && \
echo "===== FRAMEWORK INTEGRATION TESTS =====" && \
python ./python/tests/test_checkpoint_framework_integration.py
```

## Examples

### PyTorch Distributed Training

```python
import torch
import s3dlio

# Setup checkpoint store
store = s3dlio.PyCheckpointStore("s3://bucket/checkpoints", "round_robin", None)

# During training loop
def save_checkpoint(model, optimizer, epoch, step):
    # Serialize model state
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step
    }
    
    import pickle
    serialized = pickle.dumps(checkpoint_data)
    
    # Save distributed checkpoint
    writer = store.writer(world_size=torch.distributed.get_world_size(), 
                         rank=torch.distributed.get_rank())
    
    shard_key = writer.save_distributed_shard(
        step=step, epoch=epoch, framework="pytorch", data=serialized
    )
    
    # Rank 0 finalizes the checkpoint
    if torch.distributed.get_rank() == 0:
        # Collect shard metadata from all ranks
        shard_metas = [shard_key]  # In practice, gather from all ranks
        
        manifest_key = writer.finalize_distributed_checkpoint(
            step=step, epoch=epoch, framework="pytorch", 
            shard_metas=shard_metas, user_meta=None
        )
        print(f"Checkpoint saved: {manifest_key}")

def load_checkpoint(model, optimizer):
    reader = store.reader()
    manifest = reader.load_latest_manifest()
    
    if manifest:
        data = reader.read_shard_by_rank(manifest, torch.distributed.get_rank())
        checkpoint = pickle.loads(data)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint['epoch'], checkpoint['step']
    return 0, 0
```

### JAX Training

```python
import jax
import jax.numpy as jnp
import s3dlio

store = s3dlio.PyCheckpointStore("file:///tmp/jax_checkpoints", None, None)

def save_jax_checkpoint(params, epoch, step):
    # Convert JAX params to numpy for serialization
    numpy_params = jax.tree.map(lambda x: np.asarray(x), params)
    
    import pickle
    serialized = pickle.dumps(numpy_params)
    
    writer = store.writer(world_size=1, rank=0)
    shard_key = writer.save_distributed_shard(
        step=step, epoch=epoch, framework="jax", data=serialized
    )
    
    manifest_key = writer.finalize_distributed_checkpoint(
        step=step, epoch=epoch, framework="jax", 
        shard_metas=[shard_key], user_meta=None
    )
    
    return manifest_key

def load_jax_checkpoint():
    reader = store.reader()
    manifest = reader.load_latest_manifest()
    
    if manifest:
        data = reader.read_shard_by_rank(manifest, 0)
        numpy_params = pickle.loads(data)
        
        # Convert back to JAX arrays
        jax_params = jax.tree.map(lambda x: jnp.asarray(x), numpy_params)
        return jax_params, manifest['epoch']
    
    return None, 0
```

## Performance Considerations

### S3/Azure Optimization

1. **Use RoundRobin or Binary strategies** for distributed workloads
2. **Set appropriate multipart thresholds** (50MB+ recommended)
3. **Distribute ranks across availability zones** when possible

### File System Optimization

1. **Use Flat strategy** for local file systems
2. **Enable O_DIRECT** for high-performance scenarios
3. **Consider NFS/Lustre tuning** for shared file systems

### Memory Management

1. **Stream large checkpoints** to avoid memory spikes
2. **Use compression** for network-bound scenarios
3. **Implement checksum validation** for data integrity

## Migration Guide

### From Manual Checkpointing

Replace manual save/load logic with s3dlio checkpoint APIs:

```python
# Before: Manual file handling
torch.save(model.state_dict(), f's3://bucket/model_{epoch}.pt')

# After: s3dlio checkpointing
store = s3dlio.PyCheckpointStore("s3://bucket/checkpoints", "round_robin", None)
writer = store.writer(world_size, rank)
# ... use checkpoint API
```

### From AWS S3 PyTorch Connector

The s3dlio checkpoint API provides similar functionality with enhanced performance:

```python
# AWS S3 PyTorch Connector equivalent
# checkpoint = S3Checkpoint.save(model, "s3://bucket/checkpoint")

# s3dlio equivalent
store = s3dlio.PyCheckpointStore("s3://bucket/checkpoints", "round_robin", None)
writer = store.writer(world_size, rank)
# Enhanced distributed coordination and multi-backend support
```

## Best Practices

1. **Use timestamp-based checkpointing** for production workloads
2. **Implement periodic cleanup** of old checkpoints
3. **Validate checkpoint integrity** after loading
4. **Use appropriate storage strategies** for your backend
5. **Monitor checkpoint sizes** and adjust multipart thresholds
6. **Test checkpoint/restore workflows** before production deployment

## Troubleshooting

### Common Issues

**"PyCheckpointStore not found"**: Ensure s3dlio wheel is installed with checkpoint features
```bash
pip install s3dlio[checkpoint]  # If packaged separately
# or
pip install ./target/wheels/s3dlio-*.whl --force-reinstall
```

**"Permission denied" on S3**: Verify AWS credentials and bucket permissions

**"Manifest not found"**: Check that checkpoint writing completed successfully

**"Shard missing"**: Verify all ranks completed their shard uploads

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Rust logs via env variable
import os
os.environ['RUST_LOG'] = 'debug'
```

## Changelog

### v0.6.0 - Checkpoint System

**Added:**
- Complete distributed checkpointing system
- Multi-backend support (S3, Azure, file://, direct://)
- PyTorch, JAX, and TensorFlow integration
- Manifest-based checkpoint coordination
- Storage optimization strategies
- Python bindings with PyO3
- Comprehensive test coverage (28 tests)

**Performance:**
- Hot-spot avoidance for S3/Azure
- Concurrent shard operations
- Multipart upload support
- Zero-copy optimizations where possible

**Compatibility:**
- Drop-in replacement for basic checkpoint workflows
- Enhanced distributed coordination
- Framework-agnostic design
