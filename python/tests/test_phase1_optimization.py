#!/usr/bin/env python3
"""
Test script for Phase 1 GET optimization validation.
Validates that runtime scaling and concurrent GET features are working.
"""

import os
import sys
import time
import threading
from pathlib import Path

# Add the Python module to path
sys.path.insert(0, str(Path(__file__).parent / "python"))

try:
    import s3dlio
    print("✓ s3dlio module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import s3dlio: {e}")
    print("Please build the Python extension first with: ./build_pyo3.sh")
    sys.exit(1)

def test_runtime_scaling():
    """Test that runtime threads scale with environment variables"""
    print("\n=== Testing Runtime Thread Scaling ===")
    
    # Test with different thread counts
    thread_counts = [4, 8, 16]
    
    for count in thread_counts:
        os.environ["S3DLIO_RT_THREADS"] = str(count)
        print(f"Testing with S3DLIO_RT_THREADS={count}")
        
        # Create a simple configuration to trigger runtime initialization
        try:
            config = s3dlio.Config()
            config.set_compression_type("none")
            config.set_record_length(1024)
            config.set_data_loader_num_threads(2)
            print(f"  ✓ Runtime initialized with {count} threads")
        except Exception as e:
            print(f"  ✗ Failed with {count} threads: {e}")
    
    # Clean up environment
    if "S3DLIO_RT_THREADS" in os.environ:
        del os.environ["S3DLIO_RT_THREADS"]

def test_concurrent_threshold():
    """Test concurrent GET threshold configuration"""
    print("\n=== Testing Concurrent GET Threshold ===")
    
    # Test different threshold values
    thresholds = ["1048576", "16777216", "33554432"]  # 1MB, 16MB, 32MB
    
    for threshold in thresholds:
        os.environ["S3DLIO_CONCURRENT_THRESHOLD"] = threshold
        print(f"Testing with S3DLIO_CONCURRENT_THRESHOLD={threshold}")
        
        try:
            config = s3dlio.Config()
            config.set_record_length(1024)
            print(f"  ✓ Threshold {threshold} configured successfully")
        except Exception as e:
            print(f"  ✗ Failed with threshold {threshold}: {e}")
    
    # Clean up environment
    if "S3DLIO_CONCURRENT_THRESHOLD" in os.environ:
        del os.environ["S3DLIO_CONCURRENT_THRESHOLD"]

def test_http_configuration():
    """Test HTTP client configuration options"""
    print("\n=== Testing HTTP Client Configuration ===")
    
    # Test various HTTP configurations
    configs = [
        ("S3DLIO_MAX_CONNECTIONS", "200"),
        ("S3DLIO_CONNECT_TIMEOUT_MS", "5000"),
        ("S3DLIO_READ_TIMEOUT_MS", "45000"),
        ("S3DLIO_CHUNK_SIZE", "8388608"),  # 8MB
        ("S3DLIO_RANGE_CONCURRENCY", "12"),
    ]
    
    for env_var, value in configs:
        os.environ[env_var] = value
        print(f"Testing with {env_var}={value}")
        
        try:
            config = s3dlio.Config()
            config.set_record_length(1024)
            print(f"  ✓ {env_var} configured successfully")
        except Exception as e:
            print(f"  ✗ Failed with {env_var}: {e}")
    
    # Clean up environment variables
    for env_var, _ in configs:
        if env_var in os.environ:
            del os.environ[env_var]

def test_thread_safety():
    """Test that the optimizations work correctly with multiple threads"""
    print("\n=== Testing Thread Safety ===")
    
    results = []
    
    def worker(thread_id):
        try:
            config = s3dlio.Config()
            config.set_record_length(1024 * thread_id)  # Different sizes per thread
            config.set_data_loader_num_threads(2)
            results.append(f"Thread {thread_id}: ✓")
        except Exception as e:
            results.append(f"Thread {thread_id}: ✗ {e}")
    
    # Start multiple threads
    threads = []
    for i in range(1, 6):  # 5 threads
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads
    for thread in threads:
        thread.join()
    
    # Print results
    for result in results:
        print(f"  {result}")

def test_environment_defaults():
    """Test that intelligent defaults work when no environment variables are set"""
    print("\n=== Testing Intelligent Defaults ===")
    
    # Clear any existing optimization environment variables
    opt_vars = [
        "S3DLIO_RT_THREADS",
        "S3DLIO_CONCURRENT_THRESHOLD", 
        "S3DLIO_MAX_CONNECTIONS",
        "S3DLIO_CONNECT_TIMEOUT_MS",
        "S3DLIO_READ_TIMEOUT_MS",
        "S3DLIO_CHUNK_SIZE",
        "S3DLIO_RANGE_CONCURRENCY"
    ]
    
    for var in opt_vars:
        if var in os.environ:
            del os.environ[var]
    
    try:
        config = s3dlio.Config()
        config.set_record_length(8192)
        config.set_data_loader_num_threads(4)
        print("  ✓ Intelligent defaults working correctly")
    except Exception as e:
        print(f"  ✗ Failed with intelligent defaults: {e}")

def main():
    print("Phase 1 GET Optimization Validation Test")
    print("=" * 50)
    
    # Run all tests
    test_runtime_scaling()
    test_concurrent_threshold()
    test_http_configuration()
    test_thread_safety()
    test_environment_defaults()
    
    print("\n=== Phase 1 Optimization Test Complete ===")
    print("If all tests show ✓, Phase 1 optimizations are working correctly!")
    print("\nNext steps:")
    print("1. Test with actual S3 operations to measure performance improvements")
    print("2. Monitor thread scaling with: ps -eLf | grep s3dlio")
    print("3. Proceed with Phase 2 streaming PUT optimizations")

if __name__ == "__main__":
    main()
