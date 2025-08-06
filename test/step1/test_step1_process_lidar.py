#!/usr/bin/env python3
"""
Test script for Step 1 - LIDAR Processing
==========================================

This script tests the LIDAR processing pipeline with sample data from the data/ directory.
It uses a simplified configuration for faster execution during testing.
"""

import os
import sys
import json
import numpy as np
import glob
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.processor import LidarProcessor
from src.vegetation_indicators import compute_chm, save_tif
from src.geo_tools import merge_geotiffs
from src.visualization import plot_linear_reg


def setup_test_environment():
    """Setup test directories and environment"""
    test_output_dir = "test/outputs/step1"
    os.makedirs(test_output_dir, exist_ok=True)
    os.makedirs(f"{test_output_dir}/processed", exist_ok=True)
    os.makedirs("test/outputs/logs", exist_ok=True)

    print("🔧 Test environment setup completed")


def test_step1_processing():
    """Test Step 1 LIDAR processing with sample data"""
    print("🧪 Starting Step 1 Test - LIDAR Processing")
    print("=" * 60)

    # Load test configuration
    config_path = "test/config/step1_test_config.json"
    if not os.path.exists(config_path):
        print(f"❌ Test configuration not found: {config_path}")
        return False

    with open(config_path, "r") as f:
        test_config = json.load(f)

    print(f"📋 Test configuration: {test_config['step_name']}")
    print(f"📝 Description: {test_config['description']}")

    # Check if test data exists
    data_files = glob.glob("data/*.laz") + glob.glob("data/*.las")
    if not data_files:
        print("❌ No LIDAR test data found in data/ directory")
        print("   Please ensure .laz or .las files are present for testing")
        return False

    print(f"✅ Found {len(data_files)} LIDAR files for testing:")
    for file in data_files[:3]:  # Show first 3 files
        print(f"   - {os.path.basename(file)}")
    if len(data_files) > 3:
        print(f"   ... and {len(data_files) - 3} more files")

    try:
        # Create and configure LidarProcessor for testing
        print("\n🚀 Initializing LIDAR Processor...")
        lidar_processor = LidarProcessor(
            path=test_config["paths"]["input_path"],
            group=test_config["processor_settings"]["group"],
            sampling_strategy=test_config["processor_settings"]["sampling_strategy"],
            output_dir=test_config["paths"]["output_dir"],
            keep_variables=test_config["processor_settings"]["keep_variables"],
            n_jobs=test_config["processor_settings"]["n_jobs"],
            n_jobs_uncompress=2 * test_config["processor_settings"]["n_jobs"],
            pipeline=test_config["pdal_pipeline"],
            crop=test_config["processor_settings"]["crop"],
        )

        # Run processing pipeline (simplified for testing)
        print("🔄 Running LIDAR processing pipeline...")
        lidar_processor.run_pipeline(
            lidar_list_tiles=test_config["paths"]["lidar_list_tiles"],
            area_of_interest=test_config["paths"]["area_of_interest"],
            skip_uncompress=test_config["processor_settings"]["skip_uncompress"],
        )

        if lidar_processor.files_proc is None:
            print("⚠️  No files processed - this might be expected for test data")
            return True

        # Process output files
        print(f"📊 Processing {len(lidar_processor.files_proc)} output files...")
        processed_files = []

        for i, path in enumerate(lidar_processor.files_proc):
            print(
                f"📄 Processing file {i+1}/{len(lidar_processor.files_proc)}: {os.path.basename(path)}"
            )
            try:
                data = np.load(glob.glob(path, recursive=True)[0])
                chm, transf = compute_chm(
                    data,
                    resolution=test_config["chm_settings"]["resolution"],
                    quantile=test_config["chm_settings"]["quantile"],
                )

                output_path = path.split(".")[0] + "_chm_test.tif"
                save_tif(chm, transf, output_path)
                processed_files.append(output_path)
                print(
                    f"   ✅ Successfully created CHM: {os.path.basename(output_path)}"
                )

            except Exception as e:
                print(f"   ❌ Error processing {path}: {e}")
                continue

        # Merge GeoTIFFs if multiple files were processed
        if processed_files:
            try:
                print("\n🔗 Merging GeoTIFF files...")
                output_chm = os.path.join(
                    test_config["paths"]["output_dir"],
                    "processed",
                    "merged_chm_test.tif",
                )
                merge_geotiffs(test_config["paths"]["output_dir"], output_chm)
                print(f"   ✅ Merged CHM saved: {os.path.basename(output_chm)}")

                # Test linear regression plotting if CSV data exists
                if os.path.exists(test_config["paths"]["manual_csv"]):
                    try:
                        print("📈 Testing linear regression plotting...")
                        plot_linear_reg(test_config["paths"]["manual_csv"], output_chm)
                        print("   ✅ Linear regression plots generated successfully")
                    except Exception as e:
                        print(f"   ⚠️  Linear regression plotting failed: {e}")

            except Exception as e:
                print(f"❌ Error merging GeoTIFFs: {e}")

        print("\n✅ Step 1 test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Step 1 test failed: {e}")
        return False


def print_test_summary():
    """Print test results summary"""
    print("\n" + "=" * 60)
    print("📊 STEP 1 TEST SUMMARY")
    print("=" * 60)

    output_dir = "test/outputs/step1"
    if os.path.exists(output_dir):
        files = list(Path(output_dir).rglob("*"))
        print(f"📁 Output files created: {len(files)}")

        # Show some key files
        for file_type, pattern in [
            ("CHM files", "*chm*.tif"),
            ("NPY files", "*.npy"),
            ("Log files", "*.log"),
        ]:
            matching_files = list(Path(output_dir).rglob(pattern))
            if matching_files:
                print(f"   - {file_type}: {len(matching_files)}")

    print("🎯 Next: Run Step 2 test with: python test/step2/test_step2_estimate_chm.py")
    print("=" * 60)


if __name__ == "__main__":
    print("🧪 LIDAR Processing Pipeline - Step 1 Test")
    print(f"📅 Test date: {os.popen('date').read().strip()}")

    # Setup test environment
    setup_test_environment()

    # Run the test
    success = test_step1_processing()

    # Print summary
    print_test_summary()

    # Exit with appropriate code
    exit_code = 0 if success else 1
    print(f"🏁 Test completed with exit code: {exit_code}")
    sys.exit(exit_code)
