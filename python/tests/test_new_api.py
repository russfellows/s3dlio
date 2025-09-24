#!/usr/bin/env python3

"""
Test script to verify the s3dlio bug fix and new generic API functionality.

This tests:
1. The specific bug fix: PyS3AsyncDataLoader -> create_async_loader 
2. The new generic API with multiple URI schemes
3. All major API improvements
"""

import asyncio
import sys
import tempfile
import os
from pathlib import Path

# Test the Python API imports
try:
    import s3dlio
    print("✓ Successfully imported s3dlio")
except ImportError as e:
    print(f"✗ Failed to import s3dlio: {e}")
    sys.exit(1)

# Test generic dataset creation and async loader functionality
async def test_generic_api():
    """Test the new generic create_async_loader functionality"""
    print("\n=== Testing Generic API ===")
    
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp:
        tmp.write(b"Hello, this is test data!\nLine 2\nLine 3")
        tmp_path = tmp.name
    
    try:
        # Test 1: file:// URI scheme
        file_uri = f"file://{tmp_path}"
        print(f"Testing file URI: {file_uri}")
        
        # Test create_async_loader (this was the source of the bug!)
        try:
            loader = s3dlio.create_async_loader(file_uri)
            print("✓ create_async_loader works with file:// URI")
        except Exception as e:
            print(f"✗ create_async_loader failed: {e}")
            return False
        
        # Test create_dataset
        try:
            dataset = s3dlio.create_dataset(file_uri)
            print("✓ create_dataset works with file:// URI")
        except Exception as e:
            print(f"✗ create_dataset failed: {e}")
            return False
        
        # Test convenience functions
        try:
            # Test list function
            parent_dir = f"file://{os.path.dirname(tmp_path)}"
            files = await s3dlio.list(parent_dir)
            print(f"✓ list function works, found {len(files)} files")
            
            # Test stat function
            stat_result = await s3dlio.stat(file_uri)
            print(f"✓ stat function works, size: {stat_result.get('size', 'unknown')}")
            
            # Test get function
            data = await s3dlio.get(file_uri)
            print(f"✓ get function works, retrieved {len(data)} bytes")
            
        except Exception as e:
            print(f"✗ Convenience functions failed: {e}")
            return False
            
        return True
        
    finally:
        # Cleanup
        try:
            os.unlink(tmp_path)
        except:
            pass

def test_pytorch_integration():
    """Test that the PyTorch wrapper works with the fixed API"""
    print("\n=== Testing PyTorch Integration ===")
    
    try:
        # This should not fail anymore - the bug was here!
        from s3dlio.torch import S3DataLoader
        print("✓ Successfully imported S3DataLoader from torch module")
        
        # Test creating with file URI (should use the fixed create_async_loader)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.txt') as tmp:
            tmp.write(b"test data")
            tmp_path = tmp.name
        
        try:
            # This was the line that failed before the fix!
            loader = S3DataLoader(f"file://{tmp_path}")
            print("✓ S3DataLoader creation works with fixed API")
            return True
        except Exception as e:
            print(f"✗ S3DataLoader failed: {e}")
            return False
        finally:
            try:
                os.unlink(tmp_path)
            except:
                pass
                
    except Exception as e:
        print(f"✗ PyTorch integration test failed: {e}")
        return False

def test_compatibility_shims():
    """Test that compatibility wrappers still work"""
    print("\n=== Testing Compatibility Shims ===")
    
    try:
        # Test that old API names still exist (with deprecation warnings)
        if hasattr(s3dlio, 'PyS3Dataset'):
            print("✓ PyS3Dataset compatibility shim exists")
        else:
            print("⚠ PyS3Dataset compatibility shim missing")
            
        if hasattr(s3dlio, 'PyS3AsyncDataLoader'):
            print("✓ PyS3AsyncDataLoader compatibility shim exists") 
        else:
            print("⚠ PyS3AsyncDataLoader compatibility shim missing")
            
        return True
        
    except Exception as e:
        print(f"✗ Compatibility test failed: {e}")
        return False

async def main():
    """Main test function"""
    print("S3DLIO Bug Fix and API Enhancement Test")
    print("=" * 50)
    
    # Run all tests
    results = []
    
    # Test the core functionality that was broken
    results.append(await test_generic_api())
    
    # Test that PyTorch integration works 
    results.append(test_pytorch_integration())
    
    # Test compatibility
    results.append(test_compatibility_shims())
    
    # Summary
    print(f"\n=== Test Results ===")
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✓ All tests passed ({passed}/{total})")
        print("✓ Bug fix successful!")
        print("✓ Generic API enhancement successful!")
        return 0
    else:
        print(f"✗ Some tests failed ({passed}/{total})")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))