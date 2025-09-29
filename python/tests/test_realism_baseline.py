#!/usr/bin/env python3
"""
Test script to verify Python multi-backend factory functionality.
This script tests the current state before implementing realism enhancements.
"""

import os
import tempfile
import asyncio
from pathlib import Path

# Test installed package (no sys.path manipulation!)
import s3dlio

class TestResults:
    def __init__(self):
        self.results = []
    
    def add_result(self, test_name, passed, details=""):
        status = "✅ PASS" if passed else "❌ FAIL"
        self.results.append((test_name, passed, details))
        print(f"{status}: {test_name} - {details}")
    
    def summary(self):
        passed = sum(1 for _, p, _ in self.results if p)
        total = len(self.results)
        print(f"\nSummary: {passed}/{total} tests passed")
        return passed == total

def test_current_python_api():
    """Test current Python API multi-backend support"""
    print("=== Current Python API Multi-Backend Test ===\n")
    results = TestResults()
    
    # Test available functions
    required_functions = ['create_dataset', 'create_async_loader', 'list', 'stat', 'get']
    available_functions = [name for name in dir(s3dlio) if not name.startswith('_')]
    
    for func in required_functions:
        if func in available_functions:
            results.add_result(f"Function '{func}' available", True, f"Found in module")
        else:
            results.add_result(f"Function '{func}' available", False, f"Missing from module")
    
    # Test backend support with temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test_data.txt"
        test_file.write_bytes(b"Test data for backend compatibility")
        
        # Test file:// backend
        try:
            file_uri = f"file://{test_file}"
            dataset = s3dlio.create_dataset(file_uri)
            if dataset is not None:
                results.add_result("file:// backend - create_dataset", True, f"Type: {type(dataset).__name__}")
            else:
                results.add_result("file:// backend - create_dataset", False, "Returned None")
        except Exception as e:
            results.add_result("file:// backend - create_dataset", False, str(e))
        
        try:
            loader = s3dlio.create_async_loader(file_uri)
            if loader is not None:
                results.add_result("file:// backend - create_async_loader", True, f"Type: {type(loader).__name__}")
            else:
                results.add_result("file:// backend - create_async_loader", False, "Returned None")
        except Exception as e:
            results.add_result("file:// backend - create_async_loader", False, str(e))
        
        # Test direct:// backend
        try:
            direct_uri = f"direct://{test_file}"
            dataset = s3dlio.create_dataset(direct_uri)
            if dataset is not None:
                results.add_result("direct:// backend - create_dataset", True, f"Type: {type(dataset).__name__}")
            else:
                results.add_result("direct:// backend - create_dataset", False, "Returned None")
        except Exception as e:
            results.add_result("direct:// backend - create_dataset", False, str(e))
            
    # Test options parsing
    test_options = {
        "batch_size": 16,
        "shuffle": True,
        "num_workers": 2,
        "prefetch": 4,
        "reader_mode": "sequential",
        "part_size": 1024*1024,
        "max_inflight_parts": 2
    }
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp:
            tmp.write(b"Options test data")
            tmp_path = tmp.name
        
        dataset_with_opts = s3dlio.create_dataset(f"file://{tmp_path}", test_options)
        if dataset_with_opts is not None:
            results.add_result("Options parsing", True, "All current options accepted")
        else:
            results.add_result("Options parsing", False, "Failed with options")
            
        os.unlink(tmp_path)
    except Exception as e:
        results.add_result("Options parsing", False, str(e))
    
    return results.summary()

def test_current_async_functionality():
    """Test current async functionality"""
    print("\n=== Current Async Functionality Test ===\n")
    results = TestResults()
    
    async def async_test():
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for i in range(3):
                test_file = Path(tmpdir) / f"async_test_{i}.txt" 
                test_file.write_bytes(f"Async test data {i} ".encode() * 10)
            
            try:
                loader = s3dlio.create_async_loader(f"file://{tmpdir}")
                results.add_result("Async loader creation", True, "Created successfully")
                
                # Test async iteration
                count = 0
                total_bytes = 0
                async for item in loader:
                    count += 1
                    total_bytes += len(item)
                    if count >= 5:  # Limit iterations for test
                        break
                
                if count > 0:
                    results.add_result("Async iteration", True, f"Processed {count} items, {total_bytes} bytes")
                else:
                    results.add_result("Async iteration", False, "No items processed")
                    
            except Exception as e:
                results.add_result("Async functionality", False, str(e))
    
    # Run async test
    try:
        asyncio.run(async_test())
    except Exception as e:
        results.add_result("Async test runner", False, str(e))
    
    return results.summary()

def main():
    """Run all current functionality tests"""
    print("🚀 Testing Current s3dlio Python API for Realism Enhancement Baseline")
    print("=" * 80)
    
    all_passed = True
    
    # Test current API
    all_passed &= test_current_python_api()
    
    # Test async functionality 
    all_passed &= test_current_async_functionality()
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 All baseline tests passed! Ready for realism enhancements.")
    else:
        print("⚠️  Some baseline tests failed. Fix these before proceeding with enhancements.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())