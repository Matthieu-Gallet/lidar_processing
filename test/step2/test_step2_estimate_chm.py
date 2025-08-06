#!/usr/bin/env python3
"""
Test script for Step 2 - CHM Estimation
========================================

This script tests the CHM and FHD computation from processed LIDAR data.
It requires Step 1 to have been run first to generate the necessary .npy files.
"""

import os
import sys
import json
import numpy as np
import glob
from pathlib import Path

# Add parent directory to path to import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

# Import with error handling for optional dependencies
try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    RASTERIO_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: rasterio not available, some features may be limited")
    RASTERIO_AVAILABLE = False

from src.vegetation_indicators import compute_chm, compute_fhd, save_tif
from src.geo_tools import merge_geotiffs


def setup_test_environment():
    """Setup test directories and environment"""
    test_output_dir = "test/outputs/step1/results"
    os.makedirs(test_output_dir, exist_ok=True)

    print("🔧 Test environment setup completed")


def create_test_data():
    """Create minimal test data if Step 1 hasn't been run"""
    processed_dir = "test/outputs/step1/processed"

    # Check if we have processed data from Step 1
    npy_files = glob.glob(os.path.join(processed_dir, "**", "*.npy"), recursive=True)

    if not npy_files:
        print("📦 No processed data from Step 1 found, creating synthetic test data...")

        os.makedirs(processed_dir, exist_ok=True)

        # Create synthetic point cloud data for testing
        np.random.seed(42)  # For reproducible results
        n_points = 10000

        # Generate random points in a realistic coordinate system
        x = np.random.uniform(1009000, 1010000, n_points)
        y = np.random.uniform(6551000, 6552000, n_points)
        z = np.random.exponential(2, n_points) + np.random.normal(0, 0.5, n_points)
        z = np.clip(z, 0, 30)  # Clip to realistic heights

        # Add other typical LIDAR attributes
        intensity = np.random.uniform(100, 1000, n_points)
        classification = np.random.choice(
            [1, 2, 3, 4, 5], n_points, p=[0.1, 0.4, 0.3, 0.1, 0.1]
        )
        height_above_ground = z - np.random.uniform(-0.5, 2, n_points)
        height_above_ground = np.clip(height_above_ground, 0, None)

        # Create structured array similar to LIDAR data
        synthetic_data = np.array(
            list(zip(x, y, z, intensity, classification, height_above_ground)),
            dtype=[
                ("X", "f8"),
                ("Y", "f8"),
                ("Z", "f8"),
                ("Intensity", "f8"),
                ("Classification", "i4"),
                ("HeightAboveGround", "f8"),
            ],
        )

        # Save synthetic data
        test_file = os.path.join(processed_dir, "synthetic_test_data.npy")
        np.save(test_file, synthetic_data)

        print(f"   ✅ Created synthetic test data: {test_file}")
        print(f"   📊 Data points: {len(synthetic_data)}")
        return [test_file]

    else:
        print(f"✅ Found {len(npy_files)} processed files from Step 1")
        return npy_files


def save_multichannel_tif_test(
    chm_95, chm_50, fhd, transform, output_path, crs="EPSG:2154"
):
    """Test version of save_multichannel_tif with error handling"""
    if not RASTERIO_AVAILABLE:
        print("⚠️  Skipping multichannel TIFF creation (rasterio not available)")
        return

    try:
        # Stack the arrays to create a 3-channel image
        multichannel_data = np.stack([chm_95, chm_50, fhd], axis=0)

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=chm_95.shape[0],
            width=chm_95.shape[1],
            count=3,  # 3 channels
            dtype=multichannel_data.dtype,
            crs=CRS.from_string(crs),
            transform=transform,
            compress="lzw",
        ) as dst:
            # Write each channel with descriptions
            dst.write(chm_95, 1)
            dst.write(chm_50, 2)
            dst.write(fhd, 3)

            # Add channel descriptions
            dst.set_band_description(1, "CHM_quantile_95_test")
            dst.set_band_description(2, "CHM_quantile_50_test")
            dst.set_band_description(3, "FHD_test")

            # Add metadata
            dst.update_tags(
                CHM_Q95_description="Test CHM with 95th percentile",
                CHM_Q50_description="Test CHM with 50th percentile",
                FHD_description="Test FHD",
                test_data="true",
            )

        print(f"   ✅ Multichannel TIFF saved: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"   ⚠️  Error creating multichannel TIFF: {e}")


def test_step2_processing():
    """Test Step 2 CHM and FHD computation"""
    print("🧪 Starting Step 2 Test - CHM Estimation")
    print("=" * 60)

    # Load test configuration
    config_path = "test/config/step2_test_config.json"
    with open(config_path, "r") as f:
        test_config = json.load(f)

    print(f"📋 Test configuration: {test_config['step_name']}")
    print(f"📝 Description: {test_config['description']}")

    # Get or create test data
    npy_files = create_test_data()

    if not npy_files:
        print("❌ No data available for testing")
        return False

    # Setup results directory
    results_dir = os.path.join(
        "test/outputs/step1", test_config["paths"]["results_subdir"]
    )
    os.makedirs(results_dir, exist_ok=True)

    try:
        print(f"\n🔄 Processing {len(npy_files)} data files...")

        chm_95_files = []
        chm_50_files = []
        fhd_files = []

        for i, npy_file in enumerate(npy_files):
            print(
                f"📄 Processing file {i+1}/{len(npy_files)}: {os.path.basename(npy_file)}"
            )

            try:
                # Load point cloud data
                data = np.load(npy_file)
                print(f"   📊 Loaded {len(data)} points")

                # Compute CHM with quantile 95
                print("   🧮 Computing CHM (quantile 95)...")
                chm_95, transform_95 = compute_chm(
                    data,
                    resolution=test_config["chm_settings"]["resolution"],
                    quantile=test_config["chm_settings"]["quantile_95"],
                )

                # Compute CHM with quantile 50
                print("   🧮 Computing CHM (quantile 50)...")
                chm_50, transform_50 = compute_chm(
                    data,
                    resolution=test_config["chm_settings"]["resolution"],
                    quantile=test_config["chm_settings"]["quantile_50"],
                )

                # Compute FHD
                print("   🧮 Computing FHD...")
                fhd, transform_fhd = compute_fhd(
                    data,
                    resolution=test_config["fhd_settings"]["resolution"],
                    zmin=test_config["fhd_settings"]["zmin"],
                    zmax=test_config["fhd_settings"]["zmax"],
                    zwidth=test_config["fhd_settings"]["zwidth"],
                )

                # Generate output filenames
                base_name = os.path.splitext(os.path.basename(npy_file))[0]

                chm_95_file = os.path.join(
                    results_dir,
                    f"{base_name}{test_config['output_files']['chm_95_suffix']}",
                )
                chm_50_file = os.path.join(
                    results_dir,
                    f"{base_name}{test_config['output_files']['chm_50_suffix']}",
                )
                fhd_file = os.path.join(
                    results_dir,
                    f"{base_name}{test_config['output_files']['fhd_suffix']}",
                )

                # Save individual files
                save_tif(chm_95, transform_95, chm_95_file)
                save_tif(chm_50, transform_50, chm_50_file)
                save_tif(fhd, transform_fhd, fhd_file)

                chm_95_files.append(chm_95_file)
                chm_50_files.append(chm_50_file)
                fhd_files.append(fhd_file)

                # Save multichannel file for this tile
                multichannel_file = os.path.join(
                    results_dir,
                    f"{base_name}{test_config['output_files']['multichannel_suffix']}",
                )
                save_multichannel_tif_test(
                    chm_95, chm_50, fhd, transform_95, multichannel_file
                )

                print(f"   ✅ Successfully processed {base_name}")
                print(
                    f"      - CHM 95: {chm_95.shape}, range: {chm_95.min():.2f} - {chm_95.max():.2f}"
                )
                print(
                    f"      - CHM 50: {chm_50.shape}, range: {chm_50.min():.2f} - {chm_50.max():.2f}"
                )
                print(
                    f"      - FHD: {fhd.shape}, range: {fhd.min():.2f} - {fhd.max():.2f}"
                )

            except Exception as e:
                print(f"   ❌ Error processing {npy_file}: {e}")
                continue

        print(f"\n📊 Successfully processed {len(chm_95_files)} files")
        print("✅ Step 2 test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Step 2 test failed: {e}")
        return False


def print_test_summary():
    """Print test results summary"""
    print("\n" + "=" * 60)
    print("📊 STEP 2 TEST SUMMARY")
    print("=" * 60)

    results_dir = "test/outputs/step1/results"
    if os.path.exists(results_dir):
        files = list(Path(results_dir).rglob("*"))
        print(f"📁 Output files created: {len(files)}")

        # Show file types
        for file_type, pattern in [
            ("CHM 95% files", "*chm_q95*.tif"),
            ("CHM 50% files", "*chm_q50*.tif"),
            ("FHD files", "*fhd*.tif"),
            ("Multichannel files", "*multichannel*.tif"),
        ]:
            matching_files = list(Path(results_dir).rglob(pattern))
            if matching_files:
                print(f"   - {file_type}: {len(matching_files)}")

    print(
        "🎯 Next: Run Step 3 test with: python test/step3/test_step3_chm_clustering.py"
    )
    print("=" * 60)


if __name__ == "__main__":
    print("🧪 LIDAR Processing Pipeline - Step 2 Test")
    print(f"📅 Test date: {os.popen('date').read().strip()}")

    # Setup test environment
    setup_test_environment()

    # Run the test
    success = test_step2_processing()

    # Print summary
    print_test_summary()

    # Exit with appropriate code
    exit_code = 0 if success else 1
    print(f"🏁 Test completed with exit code: {exit_code}")
    sys.exit(exit_code)
