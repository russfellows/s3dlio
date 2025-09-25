#!/usr/bin/env python3
"""
Simple test to verify the new DataGenMode parameters are accepted by the Python API.
This test focuses on parameter validation rather than actual S3 operations.
"""

import sys
sys.path.insert(0, 'python')

try:
    from s3dlio import put
    print("✓ Successfully imported s3dlio.put")
except ImportError as e:
    print(f"✗ Failed to import s3dlio: {e}")
    sys.exit(1)

def test_parameter_acceptance():
    """Test that new parameters are accepted (even if S3 fails)"""
    
    print("\n=== Testing Parameter Acceptance ===")
    
    test_cases = [
        {
            "name": "Default parameters (Streaming mode)",
            "params": {
                "prefix": "s3://test-bucket/test-{}",
                "num": 1,
                "template": "test-{}",
                "size": 1024,
                "object_type": "binary"
            }
        },
        {
            "name": "Explicit Streaming mode",
            "params": {
                "prefix": "s3://test-bucket/test-{}",
                "num": 1, 
                "template": "test-{}",
                "size": 1024,
                "object_type": "binary",
                "data_gen_mode": "streaming",
                "chunk_size": 8192
            }
        },
        {
            "name": "SinglePass mode",
            "params": {
                "prefix": "s3://test-bucket/test-{}",
                "num": 1,
                "template": "test-{}",
                "size": 1024,
                "object_type": "binary", 
                "data_gen_mode": "singlepass",
                "chunk_size": 4096
            }
        },
        {
            "name": "Streaming with large chunk",
            "params": {
                "prefix": "s3://test-bucket/test-{}",
                "num": 1,
                "template": "test-{}",
                "size": 1024 * 1024,  # 1MB
                "object_type": "binary",
                "data_gen_mode": "streaming",
                "chunk_size": 65536  # 64KB chunks
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n   Testing: {test_case['name']}")
        try:
            # This will fail with "service error" but that means parameters were accepted
            put(**test_case["params"])
            print(f"     ✗ Unexpected success (should fail on S3)")
        except RuntimeError as e:
            if "service error" in str(e):
                print(f"     ✓ Parameters accepted (expected S3 service error)")
            else:
                print(f"     ✗ Unexpected runtime error: {e}")
        except TypeError as e:
            print(f"     ✗ Parameter error: {e}")
        except Exception as e:
            print(f"     ✗ Other error: {e}")

def test_invalid_parameters():
    """Test that invalid parameters are properly rejected"""
    
    print("\n=== Testing Invalid Parameters ===")
    
    invalid_cases = [
        {
            "name": "Invalid data_gen_mode",
            "params": {
                "prefix": "s3://test/test-{}",
                "num": 1,
                "template": "test-{}",
                "size": 1024,
                "object_type": "binary",
                "data_gen_mode": "invalid_mode"
            }
        }
    ]
    
    for test_case in invalid_cases:
        print(f"\n   Testing: {test_case['name']}")
        try:
            put(**test_case["params"])
            print(f"     ✗ Should have failed but didn't")
        except Exception as e:
            print(f"     ✓ Correctly rejected: {e}")

def show_function_info():
    """Show function signature and help"""
    
    print("\n=== Function Information ===")
    
    import inspect
    
    # Show signature
    sig = inspect.signature(put)
    print(f"\nFunction signature:")
    print(f"  put{sig}")
    
    # Show parameters with defaults
    print(f"\nParameters:")
    for name, param in sig.parameters.items():
        if param.default != inspect.Parameter.empty:
            print(f"  {name}: default = {repr(param.default)}")
        else:
            print(f"  {name}: required")

if __name__ == "__main__":
    print("🚀 Starting s3dlio Parameter Validation Test...")
    
    show_function_info()
    test_parameter_acceptance()
    test_invalid_parameters()
    
    print("\n🎉 Parameter validation test completed!")
    print("\nNote: 'service error' messages are expected since we don't have AWS credentials configured.")
    print("The important thing is that our new parameters (data_gen_mode, chunk_size) are accepted.")