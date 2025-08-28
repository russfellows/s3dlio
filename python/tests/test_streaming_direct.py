#!/usr/bin/env python3
"""
Direct test of streaming functionality without going through s3dlio package.
This bypasses the torch.py import issue.
"""

import asyncio
import tempfile
from pathlib import Path

# Import the core module directly
import s3dlio._pymod as core

def test_available_functions():
    """Test what functions are available in the core module."""
    print("🔍 Available functions in core module:")
    all_funcs = [x for x in dir(core) if not x.startswith('_')]
    print(f"  Total functions: {len(all_funcs)}")
    
    # Look for streaming-related functions
    streaming_funcs = [x for x in all_funcs if 'writer' in x.lower() or 'create' in x.lower()]
    print(f"  Streaming-related functions: {streaming_funcs}")
    
    return len(streaming_funcs) > 1  # More than just create_bucket

async def main():
    """Test what we can with the current state."""
    print("🚀 Testing s3dlio core module functionality")
    print("=" * 50)
    
    # Test available functions
    if test_available_functions():
        print("✅ Streaming functions found!")
        # Here we would test the actual streaming functionality
    else:
        print("❌ Streaming functions not found in core module")
        print("📝 Current available functions:", [x for x in dir(core) if not x.startswith('_')])
        return False
    
    print("✅ Direct core module test completed!")
    return True

if __name__ == "__main__":
    success = asyncio.run(main())
    print(f"\n🏁 Test result: {'PASSED' if success else 'FAILED'}")
