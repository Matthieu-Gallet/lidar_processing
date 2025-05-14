from src.processor import LidarProcessor
import os, json

from pyforestscan.calculate import calculate_chm, calculate_fhd, assign_voxels
from pyforestscan.handlers import create_geotiff
import numpy as np
import glob

if __name__ == "__main__":

    with open("config/config_csf_molosse.json", "r") as f:
        config = json.load(f)

    lidar_processor = LidarProcessor(
        path=config["paths"]["input_path"],
        group=config["processor_settings"]["group"],
        sampling_strategy=config["processor_settings"]["sampling_strategy"],
        output_dir=config["paths"]["output_dir"],
        keep_variables=config["processor_settings"]["keep_variables"],
        n_jobs=config["processor_settings"]["n_jobs"],
        n_jobs_uncompress=2 * config["processor_settings"]["n_jobs"],
        pipeline=config["pdal_pipeline"],
    )

    lidar_processor.run_pipeline(
        lidar_list_tiles=config["paths"]["lidar_list_tiles"],
        area_of_interest=config["paths"]["area_of_interest"],
        skip_uncompress=False,
    )

    if lidar_processor.files_proc is None:
        raise ValueError("No files to process. Check the input path and file format.")
    else:
        path2 = lidar_processor.files_proc[0]
        print(path2)
        data = np.load(glob.glob(path2, recursive=True)[0])
        voxel_resolution = (20, 20, 0.1)
        voxels, extent = assign_voxels(data, voxel_resolution)
        chm = calculate_chm(data, voxel_resolution, interpolation=None)
        name = path2.split(".")[0]
        create_geotiff(chm[0], name + "_chm.tif", "EPSG:2154", extent)
