#!/usr/bin/env python3
"""
Comprehensive regression tests for the modular s3dlio Python API.

This test suite verifies:
1. Module imports and basic functionality
2. NumPy function availability and basic operation
3. Dataset classes functionality
4. API consistency after refactoring

Created to ensure no regressions as we modularize the codebase.
"""

import sys
import unittest
import traceback


class TestModularAPI(unittest.TestCase):
    """Test suite for the modular s3dlio Python API."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            import s3dlio
            self.s3dlio = s3dlio
        except ImportError as e:
            self.fail(f"Failed to import s3dlio: {e}")
    
    def test_module_import(self):
        """Test that s3dlio module imports successfully."""
        # This should already pass from setUp, but explicit test
        self.assertIsNotNone(self.s3dlio)
        # Note: __version__ may not be available in all builds
        # This is informational, not required
    
    def test_core_functions_available(self):
        """Test that core S3 functions are available."""
        core_functions = [
            'get', 'put', 'list', 'delete', 'stat',
            'get_many', 'get_many_stats', 
            'upload', 'download'
        ]
        
        for func_name in core_functions:
            with self.subTest(function=func_name):
                self.assertTrue(
                    hasattr(self.s3dlio, func_name),
                    f"Core function '{func_name}' not available"
                )
    
    def test_aiml_functions_available(self):
        """Test that AI/ML functions are available."""
        aiml_functions = [
            'read_npz'
            # Note: save_numpy_array and load_numpy_array are permanently removed
            # They were disabled in v0.7.x and removed from v0.9.0 API
        ]
        
        for func_name in aiml_functions:
            with self.subTest(function=func_name):
                self.assertTrue(
                    hasattr(self.s3dlio, func_name),
                    f"AI/ML function '{func_name}' not available"
                )
    
    def test_advanced_functions_available(self):
        """Test that advanced functions are available."""
        advanced_functions = [
            'save_checkpoint', 'load_checkpoint',
            'load_checkpoint_with_validation'
        ]
        
        for func_name in advanced_functions:
            with self.subTest(function=func_name):
                self.assertTrue(
                    hasattr(self.s3dlio, func_name),
                    f"Advanced function '{func_name}' not available"
                )
    
    def test_dataset_classes_available(self):
        """Test that dataset classes are available."""
        dataset_classes = [
            'PyVecDataset', 'PyAsyncDataLoader', 'PyAsyncDataLoaderIter'
        ]
        
        for class_name in dataset_classes:
            with self.subTest(class_name=class_name):
                self.assertTrue(
                    hasattr(self.s3dlio, class_name),
                    f"Dataset class '{class_name}' not available"
                )
    
    def test_checkpoint_classes_available(self):
        """Test that checkpoint classes are available."""
        checkpoint_classes = [
            'PyCheckpointStore', 'PyCheckpointWriter', 
            'PyCheckpointReader', 'PyCheckpointInfo'
        ]
        
        for class_name in checkpoint_classes:
            with self.subTest(class_name=class_name):
                self.assertTrue(
                    hasattr(self.s3dlio, class_name),
                    f"Checkpoint class '{class_name}' not available"
                )
    
    def test_pyvecdataset_functionality(self):
        """Test PyVecDataset basic functionality."""
        # Test creation
        test_data = [1, 2, 3, 4, 5]
        dataset = self.s3dlio.PyVecDataset(test_data)
        
        # Test length
        self.assertEqual(len(dataset), 5, "Dataset length incorrect")
        
        # Test indexing
        for i, expected_value in enumerate(test_data):
            with self.subTest(index=i):
                self.assertEqual(
                    dataset[i], expected_value,
                    f"Dataset[{i}] = {dataset[i]}, expected {expected_value}"
                )
        
        # Test out of bounds
        with self.assertRaises(Exception):
            _ = dataset[10]
    
    def test_pyvecdataset_empty(self):
        """Test PyVecDataset with empty list."""
        dataset = self.s3dlio.PyVecDataset([])
        self.assertEqual(len(dataset), 0, "Empty dataset should have length 0")
    
    def test_pyvecdataset_large(self):
        """Test PyVecDataset with larger dataset."""
        test_data = list(range(100))
        dataset = self.s3dlio.PyVecDataset(test_data)
        
        self.assertEqual(len(dataset), 100, "Large dataset length incorrect")
        self.assertEqual(dataset[0], 0, "First element incorrect")
        self.assertEqual(dataset[99], 99, "Last element incorrect")
        self.assertEqual(dataset[50], 50, "Middle element incorrect")
    
    def test_numpy_functions_callable(self):
        """Test that NumPy functions are callable (basic smoke test)."""
        # We can't test actual functionality without S3 setup,
        # but we can verify they're callable functions
        
        functions_to_test = [
            'read_npz'
            # Note: save_numpy_array and load_numpy_array removed in v0.9.0
        ]
        
        for func_name in functions_to_test:
            with self.subTest(function=func_name):
                func = getattr(self.s3dlio, func_name)
                self.assertTrue(callable(func), f"{func_name} is not callable")
    
    def test_checkpoint_functions_callable(self):
        """Test that checkpoint functions are callable."""
        checkpoint_functions = [
            'save_checkpoint', 'load_checkpoint',
            'load_checkpoint_with_validation'
        ]
        
        for func_name in checkpoint_functions:
            with self.subTest(function=func_name):
                func = getattr(self.s3dlio, func_name)
                self.assertTrue(callable(func), f"{func_name} is not callable")
    
    def test_function_count_regression(self):
        """Regression test: ensure we haven't lost functions during refactoring."""
        # Count public functions/classes (non-underscore prefixed)
        public_attrs = [name for name in dir(self.s3dlio) if not name.startswith('_')]
        function_count = len(public_attrs)
        
        # We should have at least 40+ public functions/classes
        # This is a regression test to ensure refactoring doesn't lose functionality
        self.assertGreaterEqual(
            function_count, 40,
            f"Expected at least 40 public functions/classes, got {function_count}. "
            f"Available: {sorted(public_attrs)}"
        )
        
        print(f"✅ Total public functions/classes: {function_count}")
    
    def test_specific_critical_functions(self):
        """Test that specific critical functions haven't been lost."""
        critical_functions = [
            # Core S3 operations
            'get', 'put', 'list', 'delete', 'stat',
            
            # Async operations  
            'get_many_async', 'put_async',
            
            # NumPy integration
            'read_npz',
            # Note: save_numpy_array and load_numpy_array removed in v0.9.0
            
            # Data loaders
            'PyVecDataset', 'PyAsyncDataLoader',
            
            # Checkpoints
            'save_checkpoint', 'load_checkpoint',
            
            # Utility
            'init_logging'
        ]
        
        missing_functions = []
        for func_name in critical_functions:
            if not hasattr(self.s3dlio, func_name):
                missing_functions.append(func_name)
        
        self.assertEqual(
            missing_functions, [],
            f"Critical functions missing: {missing_functions}"
        )
    
    def test_dataloader_creation(self):
        """Test that data loader can be created without errors."""
        # Create a simple dataset
        dataset = self.s3dlio.PyVecDataset([1, 2, 3])
        
        # Create data loader with explicit None options (should not raise)
        loader = self.s3dlio.PyAsyncDataLoader(dataset, None)
        self.assertIsNotNone(loader)
    
    def test_module_version_info(self):
        """Test module version and metadata."""
        # Should have version info
        if hasattr(self.s3dlio, '__version__'):
            version = self.s3dlio.__version__
            self.assertIsInstance(version, str)
            self.assertRegex(version, r'\d+\.\d+\.\d+', "Version should be semver format")
            print(f"✅ s3dlio version: {version}")


class TestAPIConsistency(unittest.TestCase):
    """Test API consistency and signatures."""
    
    def setUp(self):
        """Set up test fixtures."""
        try:
            import s3dlio
            self.s3dlio = s3dlio
        except ImportError as e:
            self.fail(f"Failed to import s3dlio: {e}")
    
    def test_function_signatures_exist(self):
        """Test that functions have proper signatures (don't raise immediately)."""
        # Test functions that should accept basic arguments
        test_cases = [
            ('PyVecDataset', ([1, 2, 3],), {}),
            # Add more as needed
        ]
        
        for func_name, args, kwargs in test_cases:
            with self.subTest(function=func_name):
                func = getattr(self.s3dlio, func_name)
                try:
                    # This should create object without error
                    result = func(*args, **kwargs)
                    self.assertIsNotNone(result)
                except Exception as e:
                    # Expected for some functions without proper setup, 
                    # but shouldn't be import or signature errors
                    self.assertNotIsInstance(e, (TypeError, ImportError))


def run_comprehensive_tests():
    """Run all tests and provide summary."""
    print("🧪 Running comprehensive s3dlio modular API tests...")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestModularAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestAPIConsistency))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, trace in result.failures:
            print(f"  - {test}: {trace.split(chr(10))[-2]}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, trace in result.errors:
            print(f"  - {test}: {trace.split(chr(10))[-2]}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    if success:
        print("\n🎉 ALL TESTS PASSED! Modular API is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please review the issues above.")
    
    return success


if __name__ == "__main__":
    # Run the comprehensive test suite
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
