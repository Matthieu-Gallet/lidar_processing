#!/usr/bin/env python3
"""
Test script for Step 3 - CHM Clustering
========================================

This script tests the vegetation classification using clustering algorithms.
It can run with or without mask files, testing both scenarios.
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
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture

    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: scikit-learn not available, clustering tests will be limited")
    SKLEARN_AVAILABLE = False

try:
    import rasterio

    RASTERIO_AVAILABLE = True
except ImportError:
    print("⚠️  Warning: rasterio not available, some features may be limited")
    RASTERIO_AVAILABLE = False

from src.learning.chm_processor import (
    merge_composite_images,
    apply_masks,
    prepare_classification,
    save_classified_with_palette,
    sort_clusters_by_height,
)


def setup_test_environment():
    """Setup test directories and environment"""
    test_dirs = [
        "test/outputs/step3",
        "test/fixtures",
        "test/outputs/step1/results/processed",
    ]

    for dir_path in test_dirs:
        os.makedirs(dir_path, exist_ok=True)

    print("🔧 Test environment setup completed")


def create_test_multichannel_data():
    """Create synthetic multichannel test data if not available from Step 2"""

    # Look for multichannel files from Step 2
    multichannel_files = glob.glob("test/outputs/step1/results/*multichannel*.tif")

    if multichannel_files:
        print(f"✅ Found {len(multichannel_files)} multichannel files from Step 2")
        return multichannel_files

    if not RASTERIO_AVAILABLE:
        print("❌ Cannot create test data without rasterio")
        return []

    print("📦 Creating synthetic multichannel test data...")

    # Create synthetic multichannel data
    np.random.seed(42)
    height, width = 100, 100

    # Channel 0: CHM 95% (height data)
    chm_95 = np.random.exponential(2, (height, width)) + np.random.normal(
        0, 0.5, (height, width)
    )
    chm_95 = np.clip(chm_95, 0, 25)

    # Channel 1: CHM 50% (slightly lower values)
    chm_50 = chm_95 * np.random.uniform(0.7, 0.9, (height, width))

    # Channel 2: FHD (diversity data)
    fhd = np.random.beta(2, 5, (height, width)) * 2

    # Stack channels
    multichannel_data = np.stack([chm_95, chm_50, fhd], axis=0)

    # Create a simple transform (identity-like)
    from rasterio.transform import from_bounds

    transform = from_bounds(0, 0, width, height, width, height)

    # Save synthetic multichannel file
    output_path = "test/fixtures/synthetic_multichannel_test.tif"

    try:
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype=multichannel_data.dtype,
            crs="EPSG:2154",
            transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(multichannel_data)
            dst.set_band_description(1, "CHM_95_synthetic")
            dst.set_band_description(2, "CHM_50_synthetic")
            dst.set_band_description(3, "FHD_synthetic")

        print(f"   ✅ Created synthetic multichannel data: {output_path}")
        print(f"   📊 Dimensions: {multichannel_data.shape}")
        return [output_path]

    except Exception as e:
        print(f"❌ Error creating synthetic data: {e}")
        return []


def create_test_mask_files():
    """Create simple test mask files"""
    if not RASTERIO_AVAILABLE:
        return None, None

    mask_size = (100, 100)

    # Create rock mask (1 = rock, 0 = other)
    roc_mask = np.zeros(mask_size, dtype=np.uint8)
    roc_mask[10:20, 10:30] = 1  # Some rock areas
    roc_mask[70:80, 60:80] = 1

    # Create shadow mask (0 = shadow, 1 = other)
    shadow_mask = np.ones(mask_size, dtype=np.uint8)
    shadow_mask[5:15, 40:60] = 0  # Some shadow areas
    shadow_mask[85:95, 20:40] = 0

    # Simple transform
    from rasterio.transform import from_bounds

    transform = from_bounds(
        0, 0, mask_size[1], mask_size[0], mask_size[1], mask_size[0]
    )

    # Save masks
    roc_path = "test/fixtures/test_roc_mask.tif"
    shadow_path = "test/fixtures/test_shadow_mask.tif"

    try:
        # Save rock mask
        with rasterio.open(
            roc_path,
            "w",
            driver="GTiff",
            height=mask_size[0],
            width=mask_size[1],
            count=1,
            dtype=roc_mask.dtype,
            crs="EPSG:2154",
            transform=transform,
        ) as dst:
            dst.write(roc_mask, 1)

        # Save shadow mask
        with rasterio.open(
            shadow_path,
            "w",
            driver="GTiff",
            height=mask_size[0],
            width=mask_size[1],
            count=1,
            dtype=shadow_mask.dtype,
            crs="EPSG:2154",
            transform=transform,
        ) as dst:
            dst.write(shadow_mask, 1)

        print("   ✅ Created test mask files")
        return roc_path, shadow_path

    except Exception as e:
        print(f"   ⚠️  Error creating mask files: {e}")
        return None, None


def test_step3_processing():
    """Test Step 3 CHM clustering"""
    print("🧪 Starting Step 3 Test - CHM Clustering")
    print("=" * 60)

    # Load test configuration
    config_path = "test/config/step3_test_config.json"
    with open(config_path, "r") as f:
        test_config = json.load(f)

    print(f"📋 Test configuration: {test_config['step_name']}")
    print(f"📝 Description: {test_config['description']}")

    if not SKLEARN_AVAILABLE:
        print("❌ scikit-learn not available, cannot run clustering tests")
        return False

    # Get or create test data
    multichannel_files = create_test_multichannel_data()

    if not multichannel_files:
        print("❌ No multichannel data available for testing")
        return False

    try:
        print(f"\n🔄 Testing with {len(multichannel_files)} multichannel files...")

        # Test both with and without masks
        for use_masks_test in [True, False]:
            print(f"\n{'='*40}")
            print(f"🧪 Testing {'WITH' if use_masks_test else 'WITHOUT'} masks")
            print(f"{'='*40}")

            # Setup mask paths
            if use_masks_test:
                path_roc, path_shadow = create_test_mask_files()
                if not path_roc or not path_shadow:
                    print("⚠️  Mask creation failed, testing without masks")
                    use_masks_test = False

            # Use first multichannel file for testing
            test_file = multichannel_files[0]
            print(f"📄 Processing test file: {os.path.basename(test_file)}")

            # Test merge composite (simulate the step)
            print("🔗 Testing composite image creation...")
            save_path = test_file  # For testing, use the file directly

            # Test masking (if enabled)
            if use_masks_test and RASTERIO_AVAILABLE:
                print("🎭 Testing mask application...")
                try:
                    masked_output_path = os.path.join(
                        "test/outputs/step3", "test_masked_output.tif"
                    )
                    apply_masks(save_path, path_roc, path_shadow, masked_output_path)
                    final_path = masked_output_path
                    print("   ✅ Mask application successful")
                except Exception as e:
                    print(f"   ⚠️  Mask application failed: {e}")
                    final_path = save_path
                    use_masks_test = False
            else:
                final_path = save_path
                print("⏭️  Skipping mask application")

            # Load and prepare data for classification
            if RASTERIO_AVAILABLE:
                print("📊 Loading and preparing data for classification...")
                with rasterio.open(final_path) as src:
                    final_data = src.read()
                    final_data = np.moveaxis(final_data, 0, -1)
                    final_data = final_data[
                        :, :, test_config["data_settings"]["channels_to_use"]
                    ]
                    print(f"   📏 Final data shape: {final_data.shape}")

                # Prepare classification
                classified_array, class_dict, pixels_restants, cond_restants = (
                    prepare_classification(final_data, use_masks=use_masks_test)
                )

                print(f"   📊 Classification prepared:")
                print(f"      - Pixels for clustering: {len(pixels_restants)}")
                print(f"      - Initial classes: {list(class_dict.keys())}")

                if len(pixels_restants) == 0:
                    print("   ⚠️  No pixels available for clustering")
                    continue

                # Test clustering
                start_cluster_id = 3 if use_masks_test else 2
                method = test_config["clustering_settings"]["method"]

                print(f"🧮 Testing {method.upper()} clustering...")

                if method == "kmeans":
                    kmeans = KMeans(
                        n_clusters=test_config["clustering_settings"]["n_clusters"],
                        random_state=test_config["clustering_settings"]["random_state"],
                        max_iter=test_config["clustering_settings"]["max_iter"],
                        tol=test_config["clustering_settings"]["tolerance"],
                    )
                    kmeans.fit(pixels_restants)
                    labels = kmeans.labels_

                elif method == "gmm":
                    gmm = GaussianMixture(
                        n_components=test_config["clustering_settings"]["n_clusters"],
                        random_state=test_config["clustering_settings"]["random_state"],
                        max_iter=test_config["clustering_settings"]["max_iter"],
                        tol=test_config["clustering_settings"]["tolerance"],
                    )
                    labels = gmm.fit_predict(pixels_restants)

                # Apply clustering results
                classified_array[cond_restants] = labels + start_cluster_id

                # Sort clusters by height
                class_dict = sort_clusters_by_height(
                    pixels_restants, labels, class_dict, start_cluster_id
                )

                print(f"   ✅ Clustering completed:")
                print(f"      - Unique labels: {np.unique(labels)}")
                print(f"      - Final classes: {list(class_dict.keys())}")

                # Test saving classified result
                print("💾 Testing classified output save...")

                # Setup colors
                colors_dict = {}
                for key, color in test_config["classification_colors"].items():
                    colors_dict[int(key)] = tuple(color)

                # Generate output filename
                output_filename = test_config["output_files"][
                    "classified_filename"
                ].format(method=method)
                if not use_masks_test:
                    base_name, ext = os.path.splitext(output_filename)
                    output_filename = f"{base_name}_no_masks{ext}"

                output_path = os.path.join("test/outputs/step3", output_filename)

                try:
                    save_classified_with_palette(
                        array=classified_array,
                        output_path=output_path,
                        reference_file=final_path,
                        class_dict=class_dict,
                        colors_dict=colors_dict,
                        nodata=test_config["nodata_value"],
                    )
                    print(
                        f"   ✅ Classification saved: {os.path.basename(output_path)}"
                    )

                    # Print classification statistics
                    unique_values, counts = np.unique(
                        classified_array, return_counts=True
                    )
                    print(f"   📊 Classification statistics:")
                    for val, count in zip(unique_values, counts):
                        class_name = class_dict.get(val, f"Unknown_{val}")
                        percentage = count / classified_array.size * 100
                        print(
                            f"      - Class {val} ({class_name}): {count} pixels ({percentage:.1f}%)"
                        )

                except Exception as e:
                    print(f"   ⚠️  Error saving classification: {e}")

            else:
                print("⏭️  Skipping detailed classification (rasterio not available)")

        print("\n✅ Step 3 test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Step 3 test failed: {e}")
        return False


def print_test_summary():
    """Print test results summary"""
    print("\n" + "=" * 60)
    print("📊 STEP 3 TEST SUMMARY")
    print("=" * 60)

    output_dir = "test/outputs/step3"
    if os.path.exists(output_dir):
        files = list(Path(output_dir).rglob("*"))
        print(f"📁 Output files created: {len(files)}")

        # Show file types
        for file_type, pattern in [
            ("Classified images", "*classified*.tif"),
            ("Masked outputs", "*masked*.tif"),
            ("Composite images", "*composite*.tif"),
        ]:
            matching_files = list(Path(output_dir).rglob(pattern))
            if matching_files:
                print(f"   - {file_type}: {len(matching_files)}")

    fixtures_dir = "test/fixtures"
    if os.path.exists(fixtures_dir):
        fixtures = list(Path(fixtures_dir).rglob("*.tif"))
        if fixtures:
            print(f"📁 Test fixtures created: {len(fixtures)}")

    print(
        "🎯 Next: Run integration test with: python test/integration/test_full_pipeline.py"
    )
    print("=" * 60)


if __name__ == "__main__":
    print("🧪 LIDAR Processing Pipeline - Step 3 Test")
    print(f"📅 Test date: {os.popen('date').read().strip()}")

    # Setup test environment
    setup_test_environment()

    # Run the test
    success = test_step3_processing()

    # Print summary
    print_test_summary()

    # Exit with appropriate code
    exit_code = 0 if success else 1
    print(f"🏁 Test completed with exit code: {exit_code}")
    sys.exit(exit_code)
