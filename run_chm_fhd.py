import os
import numpy as np
import glob
from pathlib import Path
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

from src.vegetation_indicators import compute_chm, compute_fhd, save_tif
from src.geo_tools import merge_geotiffs


def save_multichannel_tif(chm_95, chm_50, fhd, transform, output_path, crs="EPSG:2154"):
    """
    Save CHM (quantile 95 and 50) and FHD as a 3-channel GeoTIFF

    Args:
        chm_95: CHM with quantile 95
        chm_50: CHM with quantile 50
        fhd: Foliage Height Diversity
        transform: Affine transformation
        output_path: Output file path
        crs: Coordinate reference system
    """
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
        dst.set_band_description(1, "CHM_quantile_95")
        dst.set_band_description(2, "CHM_quantile_50")
        dst.set_band_description(3, "FHD")

        # Add metadata
        dst.update_tags(
            CHM_Q95_description="Canopy Height Model with 95th percentile, 20m resolution",
            CHM_Q50_description="Canopy Height Model with 50th percentile, 20m resolution",
            FHD_description="Foliage Height Diversity, zmin=0, zmax=2, zwidth=0.15",
            creation_date=str(np.datetime64("now")),
        )


def process_experiment_data(experiment_dir):
    """
    Process existing experiment data to compute CHM and FHD with new parameters

    Args:
        experiment_dir: Path to the experiment directory
    """
    print(f"Processing experiment data in: {experiment_dir}")

    # Create results directory
    results_dir = os.path.join(experiment_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Find processed numpy files
    processed_dir = os.path.join(experiment_dir, "processed")
    if not os.path.exists(processed_dir):
        print(f"Processed directory not found: {processed_dir}")
        return

    # Look for .npy files
    npy_files = glob.glob(os.path.join(processed_dir, "**", "*.npy"), recursive=True)

    if not npy_files:
        print(f"No .npy files found in {processed_dir}")
        return

    print(f"Found {len(npy_files)} .npy files to process")

    # Lists to store individual results for merging
    chm_95_files = []
    chm_50_files = []
    fhd_files = []

    # Process each file
    for i, npy_file in enumerate(npy_files):
        print(f"Processing file {i+1}/{len(npy_files)}: {os.path.basename(npy_file)}")

        try:
            # Load the point cloud data
            data = np.load(npy_file)
            print(f"Loaded {len(data)} points from {os.path.basename(npy_file)}")

            # Compute CHM with quantile 95 (resolution 20m)
            print("Computing CHM with quantile 95...")
            chm_95, transform_95 = compute_chm(data, resolution=20, quantile=95)

            # Compute CHM with quantile 50 (resolution 20m)
            print("Computing CHM with quantile 50...")
            chm_50, transform_50 = compute_chm(data, resolution=20, quantile=50)

            # Compute FHD (resolution 20m, zmin=0, zmax=2, zwidth=0.15)
            print("Computing FHD...")
            fhd, transform_fhd = compute_fhd(
                data, resolution=20, zmin=0, zmax=2, zwidth=0.15
            )

            # Verify that all transforms are the same
            if not (
                np.allclose(transform_95, transform_50)
                and np.allclose(transform_95, transform_fhd)
            ):
                print("Warning: Transforms are not identical across CHM and FHD")

            # Generate output filenames
            base_name = os.path.splitext(os.path.basename(npy_file))[0]

            # Save individual files for potential merging
            chm_95_file = os.path.join(results_dir, f"{base_name}_chm_q95.tif")
            chm_50_file = os.path.join(results_dir, f"{base_name}_chm_q50.tif")
            fhd_file = os.path.join(results_dir, f"{base_name}_fhd.tif")

            save_tif(chm_95, transform_95, chm_95_file)
            save_tif(chm_50, transform_50, chm_50_file)
            save_tif(fhd, transform_fhd, fhd_file)

            chm_95_files.append(chm_95_file)
            chm_50_files.append(chm_50_file)
            fhd_files.append(fhd_file)

            # Save multichannel file for this tile
            multichannel_file = os.path.join(
                results_dir, f"{base_name}_multichannel.tif"
            )
            save_multichannel_tif(chm_95, chm_50, fhd, transform_95, multichannel_file)

            print(f"Successfully processed {base_name}")

        except Exception as e:
            print(f"Error processing {npy_file}: {e}")
            continue

    # Merge all individual files into single mosaics
    if chm_95_files and chm_50_files and fhd_files:
        try:
            print("Merging CHM quantile 95 files...")
            merged_chm_95 = os.path.join(results_dir, "merged_chm_q95.tif")
            merge_geotiffs_from_list(chm_95_files, merged_chm_95)

            print("Merging CHM quantile 50 files...")
            merged_chm_50 = os.path.join(results_dir, "merged_chm_q50.tif")
            merge_geotiffs_from_list(chm_50_files, merged_chm_50)

            print("Merging FHD files...")
            merged_fhd = os.path.join(results_dir, "merged_fhd.tif")
            merge_geotiffs_from_list(fhd_files, merged_fhd)

            # Create final multichannel merged file
            print("Creating final multichannel merged file...")
            create_merged_multichannel(
                merged_chm_95, merged_chm_50, merged_fhd, results_dir
            )

        except Exception as e:
            print(f"Error during merging: {e}")

    print(f"Processing completed. Results saved in: {results_dir}")


def merge_geotiffs_from_list(file_list, output_path):
    """
    Merge a list of GeoTIFF files into a single mosaic
    """
    if len(file_list) == 1:
        # If only one file, just copy it
        import shutil

        shutil.copy2(file_list[0], output_path)
    else:
        # Use the existing merge_geotiffs function logic
        # This assumes merge_geotiffs can handle a list of files
        temp_dir = os.path.dirname(file_list[0])
        merge_geotiffs(
            temp_dir,
            output_path,
            pattern="*" + os.path.basename(file_list[0]).split("_")[-1],
        )


def create_merged_multichannel(chm_95_file, chm_50_file, fhd_file, output_dir):
    """
    Create a merged multichannel file from the three merged single-channel files
    """
    output_path = os.path.join(output_dir, "merged_multichannel.tif")

    # Read the three merged files
    with rasterio.open(chm_95_file) as src_95:
        chm_95 = src_95.read(1)
        transform = src_95.transform
        crs = src_95.crs

    with rasterio.open(chm_50_file) as src_50:
        chm_50 = src_50.read(1)

    with rasterio.open(fhd_file) as src_fhd:
        fhd = src_fhd.read(1)

    # Save as multichannel
    save_multichannel_tif(chm_95, chm_50, fhd, transform, output_path, str(crs))
    print(f"Final multichannel file saved: {output_path}")


if __name__ == "__main__":
    # Configuration
    base_output_dir = "/mnt/sentinel4To/LIDAR_RUNEXP_MB"
    experiment_name = "exp_20250528_202329_s0p75_w25_t0p25_sc2p0_c2p85"
    experiment_dir = os.path.join(base_output_dir, experiment_name)

    # Check if experiment directory exists
    if not os.path.exists(experiment_dir):
        print(f"Experiment directory not found: {experiment_dir}")
        exit(1)

    # Process the experiment data
    process_experiment_data(experiment_dir)
