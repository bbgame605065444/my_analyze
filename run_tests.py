#!/usr/bin/env python3
"""
Test runner for the Chain-of-Thought implementation.
Runs all component tests and provides a summary report.
"""

import unittest
import sys
import os

def discover_and_run_tests():
    """Discover and run all tests in the current directory"""
    
    # Discover all test files
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(os.path.abspath(__file__))
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        descriptions=True,
        failfast=False
    )
    
    print("=" * 70)
    print("Chain-of-Thought Implementation Test Suite")
    print("=" * 70)
    
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\nFAILED TESTS:")
        for test, traceback in result.failures:
            print(f"  - {test}")
            
    if result.errors:
        print(f"\nERROR TESTS:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    # Return success status
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print(f"\n✅ All tests passed!")
    else:
        print(f"\n❌ Some tests failed. See details above.")
        
    return success

def run_specific_test(test_module):
    """Run a specific test module"""
    try:
        module = __import__(test_module)
        suite = unittest.TestLoader().loadTestsFromModule(module)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        return len(result.failures) == 0 and len(result.errors) == 0
    except ImportError:
        print(f"Could not import test module: {test_module}")
        return False

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Run specific test module
        test_module = sys.argv[1]
        if not test_module.startswith('test_'):
            test_module = f'test_{test_module}'
        if test_module.endswith('.py'):
            test_module = test_module[:-3]
            
        success = run_specific_test(test_module)
    else:
        # Run all tests
        success = discover_and_run_tests()
    
    sys.exit(0 if success else 1)