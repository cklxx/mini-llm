#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Master Test Runner
运行所有测试脚本的统一入口
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))


def run_test_script(script_path: str, description: str) -> bool:
    """运行单个测试脚本"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print(f"{'='*60}")

    try:
        # Run the test script
        result = subprocess.run([
            sys.executable, script_path
        ], capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"❌ {description} - FAILED")
            if result.stderr:
                print("STDERR:", result.stderr)
            if result.stdout:
                print("STDOUT:", result.stdout)
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT (>300s)")
        return False
    except Exception as e:
        print(f"💥 {description} - ERROR: {e}")
        return False


def main():
    """主测试函数"""
    print("🚀 MiniGPT Master Test Runner")
    print("运行所有测试脚本验证系统完整性")

    # Define test scripts in order of execution
    test_scripts = [
        {
            'script': 'scripts/tests/test_code_structure.py',
            'description': 'Code Structure Validation',
            'required': True
        },
        {
            'script': 'scripts/tests/test_architecture.py',
            'description': 'Architecture Components Test',
            'required': False  # May require PyTorch
        },
        {
            'script': 'scripts/tests/test_training_inference.py',
            'description': 'Training & Inference Test',
            'required': False  # May require PyTorch
        },
        {
            'script': 'scripts/tests/test_inference_legacy.py',
            'description': 'Legacy Inference Compatibility Test',
            'required': False  # May require PyTorch
        }
    ]

    # Track results
    results = {}
    total_tests = len(test_scripts)
    passed_tests = 0

    start_time = time.time()

    # Run each test
    for test_info in test_scripts:
        script_path = test_info['script']
        description = test_info['description']

        # Check if script exists
        if not os.path.exists(script_path):
            print(f"⚠️  {description} - Script not found: {script_path}")
            results[description] = False
            continue

        # Run the test
        success = run_test_script(script_path, description)
        results[description] = success

        if success:
            passed_tests += 1
        elif test_info['required']:
            print(f"💀 Required test failed: {description}")
            break

    end_time = time.time()
    total_time = end_time - start_time

    # Print summary
    print(f"\n{'='*60}")
    print("MASTER TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {(passed_tests/total_tests)*100:.1f}%")
    print()

    # Detailed results
    for description, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"  {status} - {description}")

    # Final verdict
    if passed_tests == total_tests:
        print(f"\n🎉 All tests passed! MiniGPT system is ready!")
        return True
    else:
        print(f"\n⚠️  {total_tests - passed_tests} test(s) failed.")
        print("Please check the individual test outputs above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)