#!/usr/bin/env python3
"""
Integration Test for Complete LIDAR Processing Pipeline
=======================================================

This script tests the entire pipeline by running all steps in sequence
using the main_pipeline.py script with test configurations.
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def setup_integration_test_environment():
    """Setup environment for integration testing"""
    test_dirs = [
        "test/outputs/integration",
        "test/outputs/logs",
        "test/outputs/temp",
        "test/outputs/backups",
    ]

    for dir_path in test_dirs:
        os.makedirs(dir_path, exist_ok=True)

    print("🔧 Integration test environment setup completed")


def run_individual_step_tests():
    """Run individual step tests to ensure they work independently"""
    print("🧪 Running individual step tests...")
    print("=" * 60)

    tests = [
        ("Step 1", "test/step1/test_step1_process_lidar.py"),
        ("Step 2", "test/step2/test_step2_estimate_chm.py"),
        ("Step 3", "test/step3/test_step3_chm_clustering.py"),
    ]

    results = {}

    for step_name, test_script in tests:
        print(f"\n🎯 Running {step_name} test...")

        if not os.path.exists(test_script):
            print(f"❌ Test script not found: {test_script}")
            results[step_name] = False
            continue

        try:
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, test_script],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout per test
            )
            duration = time.time() - start_time

            if result.returncode == 0:
                print(f"✅ {step_name} test passed ({duration:.2f}s)")
                results[step_name] = True
            else:
                print(f"❌ {step_name} test failed ({duration:.2f}s)")
                print(f"   Error output: {result.stderr[:200]}...")
                results[step_name] = False

        except subprocess.TimeoutExpired:
            print(f"⏰ {step_name} test timed out")
            results[step_name] = False
        except Exception as e:
            print(f"💥 {step_name} test error: {e}")
            results[step_name] = False

    return results


def test_pipeline_executor():
    """Test the main pipeline executor"""
    print("\n🚀 Testing Main Pipeline Executor...")
    print("=" * 60)

    # Check if main pipeline script exists
    pipeline_script = "main_pipeline.py"
    if not os.path.exists(pipeline_script):
        print(f"❌ Main pipeline script not found: {pipeline_script}")
        return False

    # Test pipeline listing
    print("📋 Testing pipeline step listing...")
    try:
        result = subprocess.run(
            [
                sys.executable,
                pipeline_script,
                "--config",
                "test/config/meta_test_config.json",
                "--list",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print("✅ Pipeline listing successful")
            print("   Available steps:")
            for line in result.stdout.split("\n"):
                if "step" in line.lower():
                    print(f"      {line.strip()}")
        else:
            print(f"❌ Pipeline listing failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Pipeline listing error: {e}")
        return False

    # Test pipeline execution (with shorter timeout for testing)
    print("\n🔄 Testing pipeline execution...")
    try:
        start_time = time.time()
        result = subprocess.run(
            [
                sys.executable,
                pipeline_script,
                "--config",
                "test/config/meta_test_config.json",
            ],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout for full pipeline
        )
        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"✅ Pipeline execution completed successfully ({duration:.2f}s)")

            # Show some output
            output_lines = result.stdout.split("\n")
            print("📤 Pipeline output (last 10 lines):")
            for line in output_lines[-10:]:
                if line.strip():
                    print(f"   {line}")

            return True
        else:
            print(f"❌ Pipeline execution failed ({duration:.2f}s)")
            print("📤 Error output:")
            for line in result.stderr.split("\n")[:10]:
                if line.strip():
                    print(f"   {line}")
            return False

    except subprocess.TimeoutExpired:
        print("⏰ Pipeline execution timed out")
        return False
    except Exception as e:
        print(f"💥 Pipeline execution error: {e}")
        return False


def analyze_test_outputs():
    """Analyze the outputs generated by the tests"""
    print("\n📊 Analyzing test outputs...")
    print("=" * 60)

    output_base = Path("test/outputs")

    if not output_base.exists():
        print("❌ No test outputs directory found")
        return

    # Analyze each step's outputs
    for step_dir in ["step1", "step2", "step3", "integration"]:
        step_path = output_base / step_dir
        if step_path.exists():
            files = list(step_path.rglob("*"))
            if files:
                print(f"📁 {step_dir.title()}: {len(files)} files")

                # Show file types
                extensions = {}
                for file in files:
                    if file.is_file():
                        ext = file.suffix.lower()
                        extensions[ext] = extensions.get(ext, 0) + 1

                for ext, count in sorted(extensions.items()):
                    if ext:
                        print(f"   - {ext[1:].upper()} files: {count}")
                    else:
                        print(f"   - No extension: {count}")
            else:
                print(f"📁 {step_dir.title()}: No files")
        else:
            print(f"📁 {step_dir.title()}: Directory not found")

    # Check for specific key files
    key_files = [
        ("CHM files", "**/*chm*.tif"),
        ("Classification files", "**/*classified*.tif"),
        ("Multichannel files", "**/*multichannel*.tif"),
        ("Log files", "**/*.log"),
        ("NPY files", "**/*.npy"),
    ]

    print("\n🔍 Key file analysis:")
    for file_type, pattern in key_files:
        matching_files = list(output_base.rglob(pattern))
        if matching_files:
            print(f"   ✅ {file_type}: {len(matching_files)} found")
        else:
            print(f"   ⚠️  {file_type}: None found")


def cleanup_test_outputs():
    """Optional cleanup of test outputs"""
    print("\n🧹 Test cleanup options:")
    print("   - Test outputs are preserved in test/outputs/ for inspection")
    print("   - To clean up: rm -rf test/outputs/*")
    print("   - To keep results: Leave files as-is")


def print_integration_summary(step_results, pipeline_result):
    """Print comprehensive test summary"""
    print("\n" + "=" * 80)
    print("🎯 INTEGRATION TEST SUMMARY")
    print("=" * 80)

    total_tests = len(step_results) + 1
    passed_tests = sum(step_results.values()) + (1 if pipeline_result else 0)

    print(f"📊 Overall Results: {passed_tests}/{total_tests} tests passed")

    print("\n📋 Individual Step Tests:")
    for step_name, result in step_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   - {step_name}: {status}")

    print("\n🚀 Pipeline Integration Test:")
    status = "✅ PASS" if pipeline_result else "❌ FAIL"
    print(f"   - Full Pipeline: {status}")

    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ The LIDAR processing pipeline is working correctly")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} TEST(S) FAILED")
        print("❌ Some components need attention")

    print("\n📚 Next Steps:")
    if passed_tests == total_tests:
        print("   - Pipeline is ready for production use")
        print("   - Consider running with real data")
        print("   - Review output quality and parameters")
    else:
        print("   - Check individual test outputs for details")
        print("   - Review error messages and logs")
        print("   - Fix failing components before production use")

    print("=" * 80)


def main():
    """Main integration test function"""
    print("🧪 LIDAR Processing Pipeline - Integration Test Suite")
    print(f"📅 Test date: {os.popen('date').read().strip()}")
    print("=" * 80)

    # Setup test environment
    setup_integration_test_environment()

    # Run individual step tests
    step_results = run_individual_step_tests()

    # Test the main pipeline executor
    pipeline_result = test_pipeline_executor()

    # Analyze outputs
    analyze_test_outputs()

    # Print comprehensive summary
    print_integration_summary(step_results, pipeline_result)

    # Cleanup options
    cleanup_test_outputs()

    # Determine overall success
    all_passed = all(step_results.values()) and pipeline_result
    exit_code = 0 if all_passed else 1

    print(f"\n🏁 Integration test completed with exit code: {exit_code}")
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
