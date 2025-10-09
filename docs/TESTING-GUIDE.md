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

## 🧪 **Cancellation Testing (v0.9.2)**

### **Overview**

s3dlio v0.9.2 introduces `CancellationToken` support for graceful shutdown of async data loading operations. The test suite in `tests/test_cancellation.rs` validates this behavior across all DataLoader components.

### **Test Coverage**

**File**: `tests/test_cancellation.rs` (9 comprehensive tests)

#### **DataLoader Tests** (5 tests):
1. **Cancellation before start** - Token cancelled before streaming begins
2. **Cancellation during streaming** - Token cancelled mid-stream with in-flight requests
3. **Cancellation with delay** - Simulates Ctrl-C after some batches processed
4. **Without cancellation token** - Validates normal operation when no token provided
5. **Multiple loaders, shared token** - Single token cancels multiple loaders simultaneously

#### **AsyncPoolDataLoader Tests** (4 tests):
1. **Cancellation before start** - Pre-cancelled token prevents pool startup
2. **Cancellation during streaming** - Pool drains in-flight requests cleanly
3. **Cancellation stops new requests** - Verifies no new requests after cancellation
4. **Idempotent cancellation** - Multiple cancel() calls don't cause issues

### **Running Cancellation Tests**

```bash
# Run all cancellation tests
cargo test --test test_cancellation

# Run specific test
cargo test --test test_cancellation test_dataloader_cancellation_during_streaming

# Run with output
cargo test --test test_cancellation -- --nocapture
```

### **Expected Behavior**

#### **On Cancellation**:
- ✅ Workers exit cleanly without submitting new requests
- ✅ In-flight requests are allowed to complete (drain pattern)
- ✅ MPSC channels properly closed
- ✅ No orphaned background tasks
- ✅ Stream ends with `None` within timeout (2 seconds)

#### **Without Cancellation**:
- ✅ Normal operation continues indefinitely
- ✅ All batches processed until dataset exhausted
- ✅ Clean completion without hanging

### **Test Patterns**

#### **Basic Cancellation Test**:
```rust
#[tokio::test]
async fn test_cancellation_example() {
    let token = CancellationToken::new();
    
    let opts = LoaderOptions::default()
        .with_batch_size(32)
        .with_cancellation_token(token.clone());
    
    let loader = DataLoader::new(dataset, opts);
    let mut stream = loader.stream();
    
    // Get some batches
    let _ = stream.next().await;
    
    // Cancel
    token.cancel();
    
    // Should complete quickly
    let result = tokio::time::timeout(
        Duration::from_secs(2),
        async {
            while let Some(_) = stream.next().await {}
        }
    ).await;
    
    assert!(result.is_ok(), "Stream should end after cancellation");
}
```

#### **Ctrl-C Handler Pattern** (for applications):
```rust
let cancel_token = CancellationToken::new();

let opts = LoaderOptions::default()
    .with_cancellation_token(cancel_token.clone());

// Spawn Ctrl-C handler
tokio::spawn(async move {
    tokio::signal::ctrl_c().await.unwrap();
    println!("Received Ctrl-C, shutting down...");
    cancel_token.cancel();
});

// Training loop
let mut stream = loader.stream();
while let Some(batch_result) = stream.next().await {
    train_step(batch_result?).await?;
}
```

### **Mock Dataset Pattern**

The tests use non-existent `file://` URIs for testing cancellation logic:

```rust
fn create_test_dataset(size: usize) -> MultiBackendDataset {
    let uris: Vec<String> = (0..size)
        .map(|i| format!("file:///tmp/test_data_{}.bin", i))
        .collect();
    
    MultiBackendDataset::from_uris(uris).expect("Failed to create test dataset")
}
```

**Why mock URIs work**:
- Cancellation logic doesn't depend on successful data fetches
- Tests verify control flow, not data correctness
- Timeouts prevent hanging on missing files
- Focuses testing on cancellation behavior, not I/O

### **Configuration Hierarchy and Cancellation**

Cancellation is part of **Level 1: LoaderOptions** (user-facing configuration):

```rust
// Level 1: Training-focused configuration
let options = LoaderOptions {
    batch_size: 32,
    cancellation_token: Some(token),  // NEW in v0.9.2
    ..Default::default()
};

// Level 2: PoolConfig (performance tuning) inherits cancellation
let loader = AsyncPoolDataLoader::new(dataset, options);
let stream = loader.stream_with_pool(pool_config);
// Cancellation applies regardless of pool configuration

// Level 3: RangeEngine (internal) respects cancellation automatically
```

See [Configuration Hierarchy](CONFIGURATION-HIERARCHY.md) for full details.

### **Testing Checklist for Cancellation**

When adding new DataLoader components:

- [ ] Add cancellation_token parameter to constructor/options
- [ ] Check `token.is_cancelled()` before async operations
- [ ] Use clean exit (break/return) when cancelled
- [ ] Allow in-flight requests to drain naturally
- [ ] Add test case to `tests/test_cancellation.rs`
- [ ] Verify timeout behavior (should complete within 2s)
- [ ] Test both with and without token

### **Known Limitations**

1. **Mock URIs**: Tests use non-existent files, so actual I/O isn't tested
2. **Timeout-based**: Tests rely on timeouts rather than explicit completion signals
3. **Single-threaded runtime**: Tests run in tokio single-threaded mode

**For production testing**: Use real S3/Azure URIs with `.env` configuration.

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