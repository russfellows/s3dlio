# Multi-Endpoint Storage Guide

**Version:** 0.9.14  
**Date:** November 2025

## Overview

The multi-endpoint feature in s3dlio allows you to distribute I/O operations across multiple storage endpoints for improved performance, scalability, and fault tolerance. This is particularly valuable for:

- **High-throughput workloads**: Aggregate bandwidth across multiple storage servers
- **Load distribution**: Balance requests across replicated datasets
- **AI/ML training**: Access distributed training data from multiple sources
- **Benchmarking**: Test storage systems with distributed load patterns
- **Fault tolerance**: Continue operations if individual endpoints fail

## Table of Contents

1. [Architecture](#architecture)
2. [Load Balancing Strategies](#load-balancing-strategies)
3. [Rust API](#rust-api)
4. [Python API](#python-api)
5. [Configuration Methods](#configuration-methods)
6. [Performance Tuning](#performance-tuning)
7. [Best Practices](#best-practices)
8. [Use Cases](#use-cases)

---

## Architecture

### Core Components

The multi-endpoint system consists of three main components:

1. **MultiEndpointStore**: A wrapper around multiple `ObjectStore` instances that implements the full `ObjectStore` trait
2. **LoadBalanceStrategy**: Determines how requests are distributed across endpoints
3. **EndpointConfig**: Configuration for individual endpoints including thread count and process affinity

### Design Principles

- **Zero-copy**: Data returned from Python interface uses buffer protocol (no memory copies)
- **Thread-safe**: All operations use atomic counters for statistics
- **Schema validation**: All endpoints must use the same URI scheme (s3://, az://, gs://, file://, direct://)
- **Transparent**: Implements full ObjectStore trait - can be used anywhere a single store is expected

---

## Load Balancing Strategies

### Round Robin

Distributes requests sequentially across endpoints in circular order.

**Characteristics:**
- Simple and predictable
- Even distribution for uniform workloads
- No overhead for tracking connections
- Best for homogeneous endpoints

**Use when:**
- All endpoints have similar performance characteristics
- Workload is uniform (similar object sizes)
- Simplicity is preferred

**Example:**
```rust
let config = MultiEndpointStoreConfig {
    endpoints: vec![
        EndpointConfig { uri: "s3://bucket1".to_string(), ..Default::default() },
        EndpointConfig { uri: "s3://bucket2".to_string(), ..Default::default() },
        EndpointConfig { uri: "s3://bucket3".to_string(), ..Default::default() },
    ],
    strategy: LoadBalanceStrategy::RoundRobin,
    ..Default::default()
};
```

**Request pattern:**
```
Request 1 → Endpoint 0
Request 2 → Endpoint 1
Request 3 → Endpoint 2
Request 4 → Endpoint 0
Request 5 → Endpoint 1
...
```

### Least Connections

Routes requests to the endpoint with the fewest active connections.

**Characteristics:**
- Adapts to varying endpoint performance
- Better for heterogeneous environments
- Tracks active connections per endpoint
- Self-balancing under varying load

**Use when:**
- Endpoints have different performance characteristics
- Object sizes vary significantly
- Some endpoints may be slower than others
- Dynamic load balancing is needed

**Example:**
```rust
let config = MultiEndpointStoreConfig {
    endpoints: vec![
        EndpointConfig { uri: "s3://fast-bucket".to_string(), ..Default::default() },
        EndpointConfig { uri: "s3://slow-bucket".to_string(), ..Default::default() },
    ],
    strategy: LoadBalanceStrategy::LeastConnections,
    ..Default::default()
};
```

**Request pattern:**
```
Fast endpoint completes quickly → receives more requests
Slow endpoint has pending connections → receives fewer requests
```

---

## Rust API

### Creating a Multi-Endpoint Store

#### From URI List

```rust
use s3dlio::{MultiEndpointStore, MultiEndpointStoreConfig, EndpointConfig, LoadBalanceStrategy};

// Create configuration
let config = MultiEndpointStoreConfig {
    endpoints: vec![
        EndpointConfig {
            uri: "s3://bucket1/prefix".to_string(),
            thread_count: Some(8),
            process_affinity: None,
        },
        EndpointConfig {
            uri: "s3://bucket2/prefix".to_string(),
            thread_count: Some(8),
            process_affinity: None,
        },
    ],
    strategy: LoadBalanceStrategy::RoundRobin,
    default_thread_count: 4,
};

// Create store
let store = MultiEndpointStore::new(config).await?;
```

#### From URI Template

Use `{start...end}` syntax for range expansion:

```rust
use s3dlio::uri_utils::expand_uri_template;

// Expand template
let uris = expand_uri_template("s3://bucket-{1...10}/data")?;
// Results in: s3://bucket-1/data, s3://bucket-2/data, ..., s3://bucket-10/data

// Create endpoints
let endpoints: Vec<EndpointConfig> = uris.iter()
    .map(|uri| EndpointConfig {
        uri: uri.clone(),
        thread_count: Some(4),
        process_affinity: None,
    })
    .collect();

let config = MultiEndpointStoreConfig {
    endpoints,
    strategy: LoadBalanceStrategy::LeastConnections,
    default_thread_count: 4,
};

let store = MultiEndpointStore::new(config).await?;
```

#### From Configuration File

```rust
use s3dlio::uri_utils::load_uris_from_file;

// Load URIs from file (one per line)
let uris = load_uris_from_file("endpoints.txt")?;

let endpoints: Vec<EndpointConfig> = uris.iter()
    .map(|uri| EndpointConfig {
        uri: uri.clone(),
        ..Default::default()
    })
    .collect();

let config = MultiEndpointStoreConfig {
    endpoints,
    strategy: LoadBalanceStrategy::RoundRobin,
    default_thread_count: 4,
};

let store = MultiEndpointStore::new(config).await?;
```

### Basic Operations

All operations implement the standard `ObjectStore` trait:

```rust
use bytes::Bytes;

// Put object
let data = Bytes::from("Hello, multi-endpoint storage!");
store.put("s3://bucket1/file.txt", data).await?;

// Get object (zero-copy)
let data = store.get("s3://bucket1/file.txt").await?;

// Get byte range (zero-copy)
let range_data = store.get_range("s3://bucket1/file.txt", 0, Some(100)).await?;

// List objects
let objects = store.list("s3://bucket1/", true).await?;

// Delete object
store.delete("s3://bucket1/file.txt").await?;
```

### Statistics and Monitoring

```rust
// Get per-endpoint statistics
let endpoint_stats = store.get_endpoint_stats();
for (i, stats) in endpoint_stats.iter().enumerate() {
    println!("Endpoint {}: {} requests, {} bytes, {} active",
             i, stats.requests, stats.bytes_transferred, stats.active_connections);
}

// Get total statistics
let total = store.get_total_stats();
println!("Total: {} requests, {} bytes, {} errors",
         total.total_requests, total.total_bytes, total.total_errors);
```

---

## Python API

The Python interface provides simplified access with automatic async handling and zero-copy data access.

### Installation

```bash
# Build and install Python bindings
cd s3dlio
./build_pyo3.sh
./install_pyo3_wheel.sh
```

### Creating a Multi-Endpoint Store

#### From URI List

```python
import s3dlio

# Create store with multiple S3 buckets
store = s3dlio.create_multi_endpoint_store(
    uris=[
        "s3://bucket1/prefix",
        "s3://bucket2/prefix",
        "s3://bucket3/prefix",
    ],
    strategy="round_robin"  # or "least_connections"
)
```

#### From URI Template

```python
# Automatically expands {1...10} to 10 URIs
store = s3dlio.create_multi_endpoint_store_from_template(
    template="s3://data-bucket-{1...10}/dataset",
    strategy="least_connections"
)
```

#### From Configuration File

```python
# Load URIs from file (one per line)
store = s3dlio.create_multi_endpoint_store_from_file(
    file_path="endpoints.txt",
    strategy="round_robin"
)
```

### Basic Operations

```python
# Put object
data = b"Training data for model"
store.put("s3://bucket1/model/data.bin", data)

# Get object (zero-copy via BytesView)
result = store.get("s3://bucket1/model/data.bin")
assert result == data

# Get byte range
chunk = store.get_range("s3://bucket1/model/large.bin", offset=1024, length=512)

# List objects
objects = store.list("s3://bucket1/model/", recursive=True)
print(f"Found {len(objects)} objects")

# Delete object
store.delete("s3://bucket1/model/data.bin")
```

### Zero-Copy Data Access

The Python interface returns `BytesView` objects that support the buffer protocol for zero-copy access:

```python
import numpy as np

# Get large object (no copy!)
data = store.get("s3://bucket1/large_model.bin")

# Access via memoryview (zero-copy)
mv = memoryview(data)
print(f"Size: {len(mv)} bytes")

# Convert to numpy array (zero-copy if possible)
array = np.frombuffer(mv, dtype=np.float32)

# Use with PyTorch (zero-copy)
import torch
tensor = torch.frombuffer(mv, dtype=torch.float32)
```

### Statistics

```python
# Get per-endpoint statistics
stats = store.get_endpoint_stats()
for i, stat in enumerate(stats):
    print(f"Endpoint {i}:")
    print(f"  Requests: {stat['requests']}")
    print(f"  Bytes: {stat['bytes_transferred']}")
    print(f"  Errors: {stat['errors']}")
    print(f"  Active: {stat['active_connections']}")

# Get total statistics
total = store.get_total_stats()
print(f"Total requests: {total['total_requests']}")
print(f"Total bytes: {total['total_bytes']}")
print(f"Total errors: {total['total_errors']}")
```

---

## Configuration Methods

### URI Template Expansion

Templates support numeric range expansion with optional zero-padding:

```rust
// Simple range
"s3://bucket-{1...5}/data"
→ ["s3://bucket-1/data", "s3://bucket-2/data", ..., "s3://bucket-5/data"]

// Zero-padded range
"s3://bucket-{01...10}/data"
→ ["s3://bucket-01/data", "s3://bucket-02/data", ..., "s3://bucket-10/data"]

// IP addresses
"direct://192.168.1.{1...10}:9000"
→ ["direct://192.168.1.1:9000", "direct://192.168.1.2:9000", ...]

// Multiple ranges in one template
"s3://rack{1...3}-node{1...4}"
→ 12 URIs (3 racks × 4 nodes)
```

### Configuration File Format

Create a text file with one URI per line:

```text
# endpoints.txt - Multi-endpoint configuration
s3://bucket1/dataset
s3://bucket2/dataset
s3://bucket3/dataset

# Comments and blank lines are ignored
az://container1/data
az://container2/data
```

Load in Rust:
```rust
let uris = load_uris_from_file("endpoints.txt")?;
```

Load in Python:
```python
store = s3dlio.create_multi_endpoint_store_from_file(
    file_path="endpoints.txt",
    strategy="round_robin"
)
```

---

## Performance Tuning

### Thread Count Configuration

Control concurrency per endpoint:

```rust
EndpointConfig {
    uri: "s3://high-performance-bucket".to_string(),
    thread_count: Some(16),  // 16 concurrent operations
    process_affinity: None,
}
```

**Guidelines:**
- **High-throughput endpoints**: 8-16 threads for NVMe-backed S3 buckets
- **Standard storage**: 4-8 threads for typical cloud storage
- **Shared endpoints**: 2-4 threads to avoid overwhelming shared resources

### Process Affinity (NUMA-aware)

For multi-socket systems, assign endpoints to specific NUMA nodes:

```rust
let endpoints = vec![
    EndpointConfig {
        uri: "direct://nvme0/data".to_string(),
        thread_count: Some(8),
        process_affinity: Some(0),  // NUMA node 0
    },
    EndpointConfig {
        uri: "direct://nvme1/data".to_string(),
        thread_count: Some(8),
        process_affinity: Some(1),  // NUMA node 1
    },
];
```

### Choosing a Load Balancing Strategy

| Scenario | Strategy | Reason |
|----------|----------|--------|
| Identical endpoints | Round Robin | Simple, predictable, minimal overhead |
| Varying performance | Least Connections | Adapts to endpoint speed differences |
| Mixed object sizes | Least Connections | Prevents slow requests from blocking fast ones |
| Testing/benchmarking | Round Robin | Easier to analyze and predict behavior |
| Production ML training | Least Connections | Handles varying network/disk conditions |

---

## Best Practices

### 1. Schema Consistency

All endpoints must use the same URI scheme:

```rust
// ✅ GOOD: All S3
vec![
    "s3://bucket1/data",
    "s3://bucket2/data",
    "s3://bucket3/data",
]

// ❌ BAD: Mixed schemes (will fail validation)
vec![
    "s3://bucket1/data",
    "az://container/data",  // Different scheme!
    "file:///local/data",   // Different scheme!
]
```

### 2. Endpoint Homogeneity

For best results with Round Robin, use similar endpoints:

```rust
// ✅ GOOD: Similar performance characteristics
vec![
    "s3://us-west-1-bucket/data",
    "s3://us-west-2-bucket/data",
    "s3://us-west-3-bucket/data",
]

// ⚠️ SUBOPTIMAL: Mixed performance (use LeastConnections instead)
vec![
    "s3://premium-nvme-bucket/data",    // 10 GB/s
    "s3://standard-bucket/data",        // 1 GB/s
    "s3://glacier-instant-bucket/data", // 250 MB/s
]
```

### 3. Monitor Statistics

Regularly check endpoint statistics to detect imbalances:

```python
def check_balance(store, threshold=0.2):
    """Check if load is balanced within threshold"""
    stats = store.get_endpoint_stats()
    requests = [s['requests'] for s in stats]
    
    avg = sum(requests) / len(requests)
    max_deviation = max(abs(r - avg) / avg for r in requests)
    
    if max_deviation > threshold:
        print(f"Warning: Load imbalance detected ({max_deviation:.1%})")
        for i, s in enumerate(stats):
            print(f"  Endpoint {i}: {s['requests']} requests")
```

### 4. Error Handling

Always handle endpoint failures gracefully:

```python
try:
    data = store.get("s3://bucket1/file.txt")
except Exception as e:
    print(f"Failed to retrieve object: {e}")
    # Fallback logic or retry
```

### 5. Configuration File Management

Use environment-specific configuration files:

```bash
# Development
endpoints-dev.txt:
  file:///tmp/dev-data1
  file:///tmp/dev-data2

# Staging
endpoints-staging.txt:
  s3://staging-bucket-1/data
  s3://staging-bucket-2/data

# Production
endpoints-prod.txt:
  s3://prod-us-west-1/data
  s3://prod-us-west-2/data
  s3://prod-us-east-1/data
```

---

## Use Cases

### Use Case 1: Distributed ML Training Data

**Scenario:** Training a large language model with data distributed across multiple S3 buckets.

```python
import s3dlio
import torch
from torch.utils.data import IterableDataset

class DistributedS3Dataset(IterableDataset):
    def __init__(self, bucket_template, num_shards):
        # Create multi-endpoint store
        self.store = s3dlio.create_multi_endpoint_store_from_template(
            template=f"s3://training-data-shard-{{1...{num_shards}}}/",
            strategy="least_connections"
        )
        self.num_shards = num_shards
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        shard = worker_info.id if worker_info else 0
        
        # List training files from assigned shard
        prefix = f"s3://training-data-shard-{shard + 1}/"
        files = self.store.list(prefix, recursive=True)
        
        for file_uri in files:
            # Zero-copy data access
            data = self.store.get(file_uri)
            
            # Convert to tensor without copying
            tensor = torch.frombuffer(memoryview(data), dtype=torch.float32)
            yield tensor.clone()  # Clone for safety

# Create dataset with 10 sharded S3 buckets
dataset = DistributedS3Dataset("training-data-shard", num_shards=10)
dataloader = torch.utils.data.DataLoader(dataset, num_workers=4)
```

### Use Case 2: High-Throughput Benchmarking

**Scenario:** Testing storage system performance with distributed load.

```python
import s3dlio
import time
from concurrent.futures import ThreadPoolExecutor

# Create multi-endpoint store for benchmarking
store = s3dlio.create_multi_endpoint_store(
    uris=[
        "s3://benchmark-bucket-1/test",
        "s3://benchmark-bucket-2/test",
        "s3://benchmark-bucket-3/test",
        "s3://benchmark-bucket-4/test",
    ],
    strategy="round_robin"
)

def benchmark_throughput(num_objects=1000, object_size_mb=10):
    """Benchmark write throughput"""
    data = b"X" * (object_size_mb * 1024 * 1024)
    
    start = time.time()
    
    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = []
        for i in range(num_objects):
            bucket = (i % 4) + 1
            uri = f"s3://benchmark-bucket-{bucket}/test/object-{i}.bin"
            future = executor.submit(store.put, uri, data)
            futures.append(future)
        
        # Wait for completion
        for future in futures:
            future.result()
    
    elapsed = time.time() - start
    throughput = (num_objects * object_size_mb) / elapsed
    
    print(f"Wrote {num_objects} objects ({object_size_mb} MB each)")
    print(f"Throughput: {throughput:.2f} MB/s")
    
    # Show per-endpoint statistics
    stats = store.get_endpoint_stats()
    for i, s in enumerate(stats):
        print(f"Endpoint {i}: {s['requests']} requests, "
              f"{s['bytes_transferred'] / 1024**3:.2f} GB")

benchmark_throughput()
```

### Use Case 3: Fault-Tolerant Data Pipeline

**Scenario:** Processing pipeline that continues despite individual endpoint failures.

```python
import s3dlio
import logging

class FaultTolerantPipeline:
    def __init__(self, endpoints):
        self.store = s3dlio.create_multi_endpoint_store(
            uris=endpoints,
            strategy="least_connections"
        )
        self.logger = logging.getLogger(__name__)
    
    def process_dataset(self, file_list):
        """Process files with automatic retry on failure"""
        results = []
        
        for uri in file_list:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Get data (automatically uses available endpoint)
                    data = self.store.get(uri)
                    
                    # Process data
                    result = self.process_data(data)
                    results.append(result)
                    
                    break  # Success
                    
                except Exception as e:
                    self.logger.warning(f"Attempt {attempt + 1} failed for {uri}: {e}")
                    
                    if attempt == max_retries - 1:
                        self.logger.error(f"Failed to process {uri} after {max_retries} attempts")
                        # Continue to next file instead of failing entire pipeline
        
        return results
    
    def process_data(self, data):
        """Process raw data (placeholder)"""
        # Your processing logic here
        return len(data)
    
    def get_health_status(self):
        """Check endpoint health"""
        stats = self.store.get_endpoint_stats()
        total = self.store.get_total_stats()
        
        print(f"Pipeline Health Report:")
        print(f"  Total requests: {total['total_requests']}")
        print(f"  Total errors: {total['total_errors']}")
        print(f"  Error rate: {total['total_errors'] / max(total['total_requests'], 1) * 100:.2f}%")
        
        for i, s in enumerate(stats):
            error_rate = s['errors'] / max(s['requests'], 1) * 100
            status = "HEALTHY" if error_rate < 5 else "DEGRADED"
            print(f"  Endpoint {i}: {status} ({error_rate:.1f}% errors)")

# Create pipeline with replica endpoints
pipeline = FaultTolerantPipeline([
    "s3://replica-1/data",
    "s3://replica-2/data",
    "s3://replica-3/data",
])
```

### Use Case 4: Geographic Data Distribution

**Scenario:** Accessing data from geographically distributed buckets for low latency.

```python
import s3dlio

# Create store with regional buckets
store = s3dlio.create_multi_endpoint_store(
    uris=[
        "s3://data-us-west-2/dataset",      # Oregon
        "s3://data-us-east-1/dataset",      # Virginia
        "s3://data-eu-west-1/dataset",      # Ireland
        "s3://data-ap-northeast-1/dataset", # Tokyo
    ],
    strategy="least_connections"  # Automatically routes to fastest endpoint
)

# Access data - automatically uses closest/fastest endpoint
data = store.get("s3://data-us-west-2/model-weights.bin")

# The least-connections strategy will naturally favor:
# - Endpoints with lower latency (fewer pending connections)
# - Geographically closer endpoints (complete requests faster)
# - Higher bandwidth endpoints (process requests quicker)
```

---

## Advanced Topics

### Custom Thread Pool Configuration

For advanced control, configure thread counts per endpoint based on backend type:

```rust
fn optimize_thread_count(uri: &str) -> usize {
    if uri.starts_with("direct://") {
        // Direct I/O: High concurrency for NVMe
        16
    } else if uri.starts_with("s3://") {
        // S3: Moderate concurrency
        8
    } else if uri.starts_with("file://") {
        // Local file: Low concurrency (single disk)
        4
    } else {
        // Default
        4
    }
}

let endpoints: Vec<EndpointConfig> = uris.iter()
    .map(|uri| EndpointConfig {
        uri: uri.clone(),
        thread_count: Some(optimize_thread_count(uri)),
        process_affinity: None,
    })
    .collect();
```

### Monitoring and Alerting

Implement monitoring for production deployments:

```python
import time

class EndpointMonitor:
    def __init__(self, store, check_interval=60):
        self.store = store
        self.check_interval = check_interval
        self.baseline = None
    
    def establish_baseline(self, duration=300):
        """Collect baseline statistics"""
        print(f"Collecting baseline for {duration}s...")
        time.sleep(duration)
        self.baseline = self.store.get_total_stats()
        print(f"Baseline: {self.baseline['total_requests']} requests")
    
    def monitor(self):
        """Continuous monitoring loop"""
        while True:
            time.sleep(self.check_interval)
            
            current = self.store.get_total_stats()
            endpoint_stats = self.store.get_endpoint_stats()
            
            # Check for anomalies
            if self.baseline:
                error_rate = current['total_errors'] / max(current['total_requests'], 1)
                baseline_error_rate = self.baseline['total_errors'] / max(self.baseline['total_requests'], 1)
                
                if error_rate > baseline_error_rate * 2:
                    print(f"ALERT: Error rate increased to {error_rate:.2%}")
            
            # Check endpoint balance
            requests = [s['requests'] for s in endpoint_stats]
            if len(requests) > 1:
                std_dev = (sum((r - sum(requests)/len(requests))**2 for r in requests) / len(requests))**0.5
                if std_dev > sum(requests) / len(requests) * 0.5:
                    print(f"ALERT: Load imbalance detected (std dev: {std_dev:.0f})")

# Usage
monitor = EndpointMonitor(store)
monitor.establish_baseline(duration=300)
monitor.monitor()
```

---

## Troubleshooting

### Issue: Uneven load distribution with Round Robin

**Symptoms:** Some endpoints receive significantly more requests than others.

**Possible causes:**
- Object sizes vary widely
- Some endpoints are slower than others
- Requests have different processing times

**Solution:** Switch to Least Connections strategy:
```python
store = s3dlio.create_multi_endpoint_store(
    uris=endpoints,
    strategy="least_connections"  # Changed from "round_robin"
)
```

### Issue: High error rate on specific endpoint

**Symptoms:** One endpoint shows many errors in statistics.

**Diagnosis:**
```python
stats = store.get_endpoint_stats()
for i, s in enumerate(stats):
    if s['errors'] > 0:
        error_rate = s['errors'] / s['requests'] * 100
        print(f"Endpoint {i}: {error_rate:.1f}% error rate")
```

**Solutions:**
1. Check endpoint connectivity/credentials
2. Verify bucket/container exists and is accessible
3. Consider removing faulty endpoint from configuration
4. Check for rate limiting or quota issues

### Issue: Memory usage increasing with large objects

**Symptoms:** High memory consumption when processing large files.

**Cause:** Not using zero-copy access properly.

**Solution:** Use memoryview for zero-copy access:
```python
# ❌ BAD: Copies data to bytes
data = store.get(uri)
bytes_data = bytes(data)  # Creates copy!

# ✅ GOOD: Zero-copy via memoryview
data = store.get(uri)
mv = memoryview(data)  # No copy
# Process via memoryview or convert directly to numpy/torch
```

---

## Migration Guide

### Migrating from Single Endpoint

**Before:**
```python
import s3dlio

# Old: Single endpoint
data = s3dlio.get("s3://my-bucket/file.txt")
```

**After:**
```python
import s3dlio

# New: Multi-endpoint with single endpoint (compatible)
store = s3dlio.create_multi_endpoint_store(
    uris=["s3://my-bucket/"],
    strategy="round_robin"
)

data = store.get("s3://my-bucket/file.txt")
```

### Adding Additional Endpoints

Once you have a multi-endpoint store, you can gradually add endpoints:

```python
# Start with 1 endpoint
endpoints = ["s3://bucket-1/data"]

# Later, expand to 3 endpoints
endpoints = [
    "s3://bucket-1/data",
    "s3://bucket-2/data",
    "s3://bucket-3/data",
]

# Recreate store with new configuration
store = s3dlio.create_multi_endpoint_store(
    uris=endpoints,
    strategy="round_robin"
)
```

---

## Performance Characteristics

### Throughput Scaling

Expected throughput scaling with multiple endpoints:

| Endpoints | Expected Throughput | Use Case |
|-----------|---------------------|----------|
| 1 | 1.0× baseline | Single source |
| 2 | 1.8-2.0× baseline | Simple replication |
| 4 | 3.5-4.0× baseline | Standard distributed setup |
| 8 | 6.5-8.0× baseline | High-performance workloads |
| 16+ | 12-16× baseline | Extreme throughput requirements |

**Note:** Actual scaling depends on:
- Network bandwidth
- Storage backend performance
- Object size distribution
- Request pattern (read vs. write heavy)

### Overhead

Multi-endpoint overhead is minimal:

- **Round Robin**: ~10ns per request (atomic counter increment)
- **Least Connections**: ~50ns per request (atomic counter read + comparison)
- **Statistics**: Lock-free atomics (no contention)

For typical S3 requests (1-100ms latency), overhead is negligible (<0.01%).

---

## See Also

- [User Guide](USER_GUIDE.md) - General s3dlio usage
- [API Documentation](README.md) - Core API reference
- [Changelog](Changelog.md) - Version history and features
- [Performance Guide](PERFORMANCE.md) - Optimization techniques

---

## Support

For issues or questions:
- **GitHub Issues**: [s3dlio issues](https://github.com/russfellows/s3dlio/issues)
- **Documentation**: [docs directory](.)
- **Examples**: [examples directory](../examples/)

---

**Version History:**
- v0.9.14 (November 2025): Initial multi-endpoint support
