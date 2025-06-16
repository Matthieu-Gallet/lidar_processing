from src.processor import LidarProcessor
import os, json

from src.vegetation_indicators import compute_chm,save_tif
import numpy as np
import glob
from src.geo_tools import merge_geotiffs


if __name__ == "__main__":

    with open("config/config_csf_molosse.json", "r") as f:
        exp_config = json.load(f)

    # Create and run the LidarProcessor
    lidar_processor = LidarProcessor(
        path=exp_config["paths"]["input_path"],
        group=exp_config["processor_settings"]["group"],
        sampling_strategy=exp_config["processor_settings"]["sampling_strategy"],
        output_dir=exp_config["paths"]["output_dir"],
        keep_variables=exp_config["processor_settings"]["keep_variables"],
        n_jobs=exp_config["processor_settings"]["n_jobs"],
        n_jobs_uncompress=2 * exp_config["processor_settings"]["n_jobs"],
        pipeline=exp_config["pdal_pipeline"],
        crop=exp_config["processor_settings"]["crop"],
    )
    
    lidar_processor.run_pipeline(
        lidar_list_tiles=exp_config["paths"]["lidar_list_tiles"],
        area_of_interest=exp_config["paths"]["area_of_interest"],
        skip_uncompress=False,
    )

    if lidar_processor.files_proc is None:
        print(f"No files processed for experimen")
    else:
        # Process output files
        for path in lidar_processor.files_proc:
            print(f"Processing file: {path}")
            try:
                data = np.load(glob.glob(path, recursive=True)[0])
                chm, transf = compute_chm(data, resolution=5, quantile=95)
                save_tif(chm, transf, path.split(".")[0] + "_chm.tif")
                print(f"Successfully created CHM for {path}")
            except Exception as e:
                print(f"Error processing {path}: {e}")
        try:
            output_chm = os.path.join(exp_config["paths"]["output_dir"], "processed", "merged_chm.tif")
            merge_geotiffs(exp_config["paths"]["output_dir"], output_chm)
        except Exception as e:
            print(f"Error merging GeoTIFFs: {e}")
    # lidar_processor = LidarProcessor(
    #     path=config["paths"]["input_path"],
    #     group=config["processor_settings"]["group"],
    #     sampling_strategy=config["processor_settings"]["sampling_strategy"],
    #     output_dir=config["paths"]["output_dir"],
    #     keep_variables=config["processor_settings"]["keep_variables"],
    #     n_jobs=config["processor_settings"]["n_jobs"],
    #     n_jobs_uncompress=2 * config["processor_settings"]["n_jobs"],
    #     pipeline=config["pdal_pipeline"],
    # )

    # lidar_processor.run_pipeline(
    #     lidar_list_tiles=config["paths"]["lidar_list_tiles"],
    #     area_of_interest=config["paths"]["area_of_interest"],
    #     skip_uncompress=False,
    # )

    # if lidar_processor.files_proc is None:
    #     raise ValueError("No files to process. Check the input path and file format.")
    # else:

    #     path2 = lidar_processor.files_proc[0]
    #     print(path2)
    #     data = np.load(glob.glob(path2, recursive=True)[0])
    #     chm, transf = compute_chm(data, resolution=10, quantile=95)
    #     save_tif(chm,transf, path2.split(".")[0] + "_chm.tif")


    #     # manual_csv = "/home/mgallet/Documents/Lidar/lidar_processing/data/indicators_from_observed_variables.csv"
    #     # plot_linear_reg(manual_csv, output_chm)