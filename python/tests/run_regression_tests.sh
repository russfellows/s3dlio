#!/bin/bash
# Test runner for s3dlio regression tests
# This script runs our comprehensive test suite to ensure no regressions

set -e

echo "🧪 Running s3dlio Comprehensive Test Suite"
echo "=========================================="

# Change to repository root
cd "$(dirname "$0")/../.."

echo "📋 Test Environment:"
echo "   Python: $(python --version)"
echo "   Working Directory: $(pwd)"

# Run the modular API regression tests
echo ""
echo "🔧 Running Modular API Regression Tests..."
python python/tests/test_modular_api_regression.py

echo ""
echo "✅ All regression tests completed successfully!"
echo ""
echo "💡 To add this to CI/CD:"
echo "   - Add this script to your GitHub Actions workflow"
echo "   - Run after any API changes or refactoring"
echo "   - Helps catch regressions early"
