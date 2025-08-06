#!/usr/bin/env python3
"""
Quick Test Runner for LIDAR Processing Pipeline
===============================================

This script provides a simple interface to run different types of tests
for the LIDAR processing pipeline.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_step_test(step_number, verbose=False):
    """Run a specific step test"""
    test_scripts = {
        1: "test/step1/test_step1_process_lidar.py",
        2: "test/step2/test_step2_estimate_chm.py",
        3: "test/step3/test_step3_chm_clustering.py",
    }

    if step_number not in test_scripts:
        print(f"❌ Invalid step number: {step_number}")
        print(f"   Available steps: {list(test_scripts.keys())}")
        return False

    script = test_scripts[step_number]

    if not os.path.exists(script):
        print(f"❌ Test script not found: {script}")
        return False

    print(f"🧪 Running Step {step_number} test...")
    try:
        cmd = [sys.executable, script]
        if verbose:
            result = subprocess.run(cmd, timeout=300)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                print(f"❌ Test failed with error:")
                print(result.stderr[:500])

        success = result.returncode == 0
        print(
            f"{'✅' if success else '❌'} Step {step_number} test {'passed' if success else 'failed'}"
        )
        return success

    except subprocess.TimeoutExpired:
        print(f"⏰ Step {step_number} test timed out")
        return False
    except Exception as e:
        print(f"💥 Step {step_number} test error: {e}")
        return False


def run_integration_test(verbose=False):
    """Run the full integration test"""
    script = "test/integration/test_full_pipeline.py"

    if not os.path.exists(script):
        print(f"❌ Integration test script not found: {script}")
        return False

    print("🚀 Running integration test...")
    try:
        cmd = [sys.executable, script]
        if verbose:
            result = subprocess.run(cmd, timeout=900)
        else:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
            if result.returncode != 0:
                print(f"❌ Integration test failed:")
                print(result.stderr[:1000])
            else:
                # Show last part of output for success summary
                lines = result.stdout.split("\n")
                summary_start = -20
                for i, line in enumerate(lines):
                    if "INTEGRATION TEST SUMMARY" in line:
                        summary_start = i
                        break

                if summary_start > 0:
                    print("\n".join(lines[summary_start:]))
                else:
                    print(result.stdout[-1000:])

        success = result.returncode == 0
        print(
            f"{'✅' if success else '❌'} Integration test {'passed' if success else 'failed'}"
        )
        return success

    except subprocess.TimeoutExpired:
        print("⏰ Integration test timed out")
        return False
    except Exception as e:
        print(f"💥 Integration test error: {e}")
        return False


def run_all_tests(verbose=False):
    """Run all tests in sequence"""
    print("🧪 Running all LIDAR processing tests...")
    print("=" * 50)

    results = {}

    # Run individual step tests
    for step in [1, 2, 3]:
        results[f"Step {step}"] = run_step_test(step, verbose)

    # Run integration test
    results["Integration"] = run_integration_test(verbose)

    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! Pipeline is ready.")
    else:
        print("⚠️  Some tests failed. Check individual results above.")

    return passed == total


def clean_test_outputs():
    """Clean up test output directories"""
    import shutil

    output_dirs = [
        "test/outputs/step1",
        "test/outputs/step2",
        "test/outputs/step3",
        "test/outputs/integration",
        "test/outputs/logs",
    ]

    cleaned = 0
    for dir_path in output_dirs:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                cleaned += 1
                print(f"🧹 Cleaned: {dir_path}")
            except Exception as e:
                print(f"❌ Failed to clean {dir_path}: {e}")

    if cleaned > 0:
        print(f"✅ Cleaned {cleaned} output directories")
    else:
        print("🔍 No test output directories found to clean")


def setup_test_environment():
    """Setup the test environment"""
    print("🔧 Setting up test environment...")

    # Create necessary directories
    dirs_to_create = [
        "test/outputs/step1",
        "test/outputs/step2",
        "test/outputs/step3",
        "test/outputs/integration",
        "test/outputs/logs",
    ]

    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)

    print("✅ Test environment setup complete")


def show_test_info():
    """Show information about available tests"""
    print("📚 LIDAR Processing Pipeline - Test Information")
    print("=" * 60)

    tests = {
        "Step 1": {
            "script": "test/step1/test_step1_process_lidar.py",
            "description": "Tests LIDAR point cloud processing, denoising, and classification",
        },
        "Step 2": {
            "script": "test/step2/test_step2_estimate_chm.py",
            "description": "Tests CHM and FHD estimation from processed LIDAR data",
        },
        "Step 3": {
            "script": "test/step3/test_step3_chm_clustering.py",
            "description": "Tests vegetation clustering and classification",
        },
        "Integration": {
            "script": "test/integration/test_full_pipeline.py",
            "description": "Tests the complete pipeline end-to-end",
        },
    }

    print("Available Tests:")
    for test_name, info in tests.items():
        exists = "✅" if os.path.exists(info["script"]) else "❌"
        print(f"  {exists} {test_name}")
        print(f"     Script: {info['script']}")
        print(f"     Description: {info['description']}")
        print()

    print("Test Output Locations:")
    output_dirs = [
        "test/outputs/step1/ - Step 1 outputs (processed LIDAR data)",
        "test/outputs/step2/ - Step 2 outputs (CHM, FHD, DEM files)",
        "test/outputs/step3/ - Step 3 outputs (classification results)",
        "test/outputs/integration/ - Integration test outputs",
        "test/outputs/logs/ - Log files from all tests",
    ]

    for output_dir in output_dirs:
        print(f"  📁 {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Quick test runner for LIDAR processing pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --all              # Run all tests
  python run_tests.py --step 1           # Run only step 1 test
  python run_tests.py --integration      # Run integration test only
  python run_tests.py --info             # Show test information
  python run_tests.py --clean            # Clean test outputs
  python run_tests.py --setup            # Setup test environment
        """,
    )

    parser.add_argument(
        "--all", action="store_true", help="Run all tests (step tests + integration)"
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3],
        help="Run a specific step test (1, 2, or 3)",
    )
    parser.add_argument(
        "--integration", action="store_true", help="Run the integration test only"
    )
    parser.add_argument(
        "--info", action="store_true", help="Show information about available tests"
    )
    parser.add_argument(
        "--clean", action="store_true", help="Clean up test output directories"
    )
    parser.add_argument("--setup", action="store_true", help="Setup test environment")
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show verbose output from tests"
    )

    args = parser.parse_args()

    # Change to the script directory for relative paths to work
    script_dir = Path(__file__).parent.parent  # Go up from test/ to project root
    if script_dir.exists():
        os.chdir(script_dir)

    success = True

    if args.info:
        show_test_info()
    elif args.clean:
        clean_test_outputs()
    elif args.setup:
        setup_test_environment()
    elif args.all:
        success = run_all_tests(args.verbose)
    elif args.step:
        success = run_step_test(args.step, args.verbose)
    elif args.integration:
        success = run_integration_test(args.verbose)
    else:
        parser.print_help()
        return 0

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
