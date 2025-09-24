# S3DLIO Testing Guide - Critical Issues and Solutions

## 🚨 **CRITICAL TESTING GOTCHA #1: Python Path Import Issues**

### **Problem**: Tests import development code instead of installed package

**Symptom**: Tests fail with "module 's3dlio' has no attribute 'create_dataset'" even though the functions exist when tested manually.

**Root Cause**: 
- Tests use `sys.path.insert(0, 'python')` to import from development directory
- Development `python/s3dlio/` contains pure Python wrappers but NOT the compiled Rust module
- Installed package contains both Python wrappers AND compiled Rust module (`_pymod`)
- Test imports development version which is missing the Rust functions

**Example of the problem**:
```python
# ❌ WRONG: This imports from development directory (missing Rust module)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))
import s3dlio
s3dlio.create_dataset("file://...")  # AttributeError: not found

# ✅ CORRECT: This imports from installed package (has Rust module)  
import s3dlio  # No sys.path manipulation
s3dlio.create_dataset("file://...")  # Works perfectly
```

### **Solutions**:

#### **Option 1: Use Installed Package (Recommended)**
```python
#!/usr/bin/env python3
"""
Correct test that uses installed s3dlio package.
"""
# NO sys.path manipulation - use installed package!
import s3dlio

def test_functionality():
    # This works because it uses the installed package with Rust module
    dataset = s3dlio.create_dataset("file:///path/to/data")
    assert dataset is not None
```

#### **Option 2: Rebuild and Install Before Testing**
```bash
# Always rebuild and install before testing
./build_pyo3.sh && ./install_pyo3_wheel.sh
python tests/your_test.py
```

#### **Option 3: Test Both Versions (Advanced)**
```python
#!/usr/bin/env python3
"""
Test that validates both development and installed versions.
"""
import sys
import importlib

def test_installed_version():
    """Test the installed package."""
    # Import without path manipulation
    import s3dlio
    assert hasattr(s3dlio, 'create_dataset')
    
def test_development_version():
    """Test development version limitations."""
    # Import from development path
    sys.path.insert(0, 'python')
    import s3dlio_dev as s3dlio
    
    # Should have Python wrappers but NOT Rust functions
    assert hasattr(s3dlio, '__version__')  # Python wrapper
    assert not hasattr(s3dlio, 'create_dataset')  # Rust function missing
```

### **How to Identify This Problem**:

1. **Manual testing works but automated tests fail**
2. **Functions exist in Python REPL but not in test scripts**  
3. **Error**: `AttributeError: module 's3dlio' has no attribute 'create_dataset'`
4. **Test has `sys.path.insert()` or similar path manipulation**

### **Quick Debug Commands**:
```bash
# Check what's actually installed
python -c "import s3dlio; print(s3dlio.__file__); print(hasattr(s3dlio, 'create_dataset'))"

# Check development version  
python -c "import sys; sys.path.insert(0, 'python'); import s3dlio; print(s3dlio.__file__); print(hasattr(s3dlio, 'create_dataset'))"

# Compare module contents
python -c "import s3dlio; print('Installed functions:', [x for x in dir(s3dlio) if 'create' in x])"
```

---

## 🧪 **Testing Best Practices for s3dlio**

### **Rule #1: Always Use Installed Package for Functional Tests**
```python
# ✅ GOOD: Uses installed package
import s3dlio

# ❌ BAD: Imports development version
sys.path.insert(0, 'python')
import s3dlio
```

### **Rule #2: Rebuild Before Testing New Changes**
```bash
# Required workflow for testing changes
./build_pyo3.sh        # Compile Rust changes
./install_pyo3_wheel.sh  # Install new version
python tests/test_functionality.py  # Now test
```

### **Rule #3: Test Installation Status**
```python
def verify_installation():
    """Verify s3dlio is properly installed with Rust module."""
    import s3dlio
    
    # Check basic import
    assert hasattr(s3dlio, '__version__')
    
    # Check Rust functions are available  
    rust_functions = ['create_dataset', 'create_async_loader', 'get', 'put', 'list']
    for func_name in rust_functions:
        assert hasattr(s3dlio, func_name), f"Missing Rust function: {func_name}"
    
    # Check Rust classes are available
    rust_classes = ['PyDataset', 'PyBytesAsyncDataLoader']  
    for class_name in rust_classes:
        assert hasattr(s3dlio, class_name), f"Missing Rust class: {class_name}"
    
    print("✅ Installation verified - all Rust components available")
```

### **Rule #4: Use Proper Test Structure**
```python
#!/usr/bin/env python3
"""
Template for s3dlio tests that actually work.
"""
import os
import tempfile
from pathlib import Path

# NO PATH MANIPULATION - use installed package!
import s3dlio

def test_core_functionality():
    """Test core s3dlio functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data
        test_file = Path(tmpdir) / "test.txt"  
        test_file.write_bytes(b"test data")
        
        # Test dataset creation (uses installed Rust module)
        dataset = s3dlio.create_dataset(f"file://{test_file}")
        assert dataset is not None
        assert type(dataset).__name__ == 'PyDataset'
        
        # Test async loader creation
        loader = s3dlio.create_async_loader(f"file://{tmpdir}")
        assert loader is not None
        assert type(loader).__name__ == 'PyBytesAsyncDataLoader'

if __name__ == "__main__":
    test_core_functionality()
    print("✅ All tests passed!")
```

---

## 🔧 **Development Workflow**

### **Correct Development Cycle**:
1. Make changes to Rust code
2. `./build_pyo3.sh` - Compile changes
3. `./install_pyo3_wheel.sh` - Install new wheel  
4. Run tests (no path manipulation!)
5. Repeat as needed

### **Testing Workflow Commands**:
```bash
# Full development cycle
cd /path/to/s3dlio

# 1. Make your changes to src/ files
vim src/python_api/python_aiml_api.rs

# 2. Build and install
./build_pyo3.sh && ./install_pyo3_wheel.sh

# 3. Test (uses installed package automatically)
python tests/test_working_functionality.py

# 4. Verify specific functionality
python -c "
import s3dlio
import tempfile
from pathlib import Path

with tempfile.TemporaryDirectory() as tmpdir:
    test_file = Path(tmpdir) / 'test.txt'
    test_file.write_bytes(b'test')
    
    dataset = s3dlio.create_dataset(f'file://{test_file}')
    print(f'SUCCESS: Created {type(dataset).__name__}')
"
```

---

## 🐛 **Common Testing Problems and Solutions**

### **Problem**: "Functions exist manually but not in tests"
**Solution**: Check for `sys.path.insert()` in test files - remove it!

### **Problem**: "Tests pass sometimes but fail other times"  
**Solution**: Always rebuild/install before testing changes

### **Problem**: "Import errors in test files"
**Solution**: Make sure UV virtual environment is activated

### **Problem**: "Functions show as 'not found' in linter but work at runtime"
**Solution**: This is normal - linter can't see dynamically exported Rust functions

### **Problem**: "PyTorch tests fail with loader_opts missing"
**Solution**: Use `S3IterableDataset(uri, loader_opts={})` format

---

## 📋 **Testing Checklist**

Before running any s3dlio test:

- [ ] UV virtual environment is activated
- [ ] Latest code is built: `./build_pyo3.sh`  
- [ ] Latest wheel is installed: `./install_pyo3_wheel.sh`
- [ ] Test imports s3dlio WITHOUT path manipulation
- [ ] Test verifies installation before running functionality tests

---

## 📚 **Reference Files**

- **Working Test Example**: `tests/test_working_functionality_fixed.py`
- **Build Script**: `build_pyo3.sh`  
- **Install Script**: `install_pyo3_wheel.sh`
- **Python Package**: `python/s3dlio/` (development)
- **Installed Package**: `.venv/lib/python3.12/site-packages/s3dlio/` (testing)

---

**💡 REMEMBER**: The golden rule is "test what users will actually install and use" - always use the installed package for functional testing!