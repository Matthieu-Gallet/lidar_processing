import os
import json
import shutil
import itertools
from pathlib import Path
import numpy as np
import glob
from datetime import datetime
from copy import deepcopy
import pandas as pd

from src.processor import LidarProcessor
from src.vegetation_indicators import compute_chm, save_tif
from src.linear_reg import plot_scenario, _SCENARIO, _AX_TITLES, _AX_LABELS, _SUP_TITLES
from src.geo_tools import extract_windows, extract_diagonals, compute_stats, merge_geotiffs


def plot_linear_reg(manual_csv, output_chm):
    try:
        pd_survey = deepcopy(pd.read_csv(manual_csv))
    except FileNotFoundError:
        raise FileNotFoundError(f"File {manual_csv} not found.")
    try:
        windows = extract_windows(output_chm, pd_survey['X'], pd_survey['Y'], (7,7))
        diagonals = extract_diagonals(windows)
        allvalue = np.array(windows).reshape(len(windows), -1)
        stats = compute_stats(diagonals, axis=1)
        for keys, values in stats.items():
            pd_survey[keys] = values
        stats_all = compute_stats(allvalue, axis=1)
        for keys, values in stats_all.items():
            pd_survey[keys+'_all'] = values
    except Exception as e:
        print(f"Error extracting windows: {e}")
    try:
        for K, _SC in enumerate(_SCENARIO):
            sup_title = _SUP_TITLES[K] + f" - 20m"
            plot_scenario(pd_survey, _SC, _AX_TITLES, _AX_LABELS[K], sup_title, save=os.path.join(os.path.dirname(output_chm), 'FIGURES'))
    except Exception as e:
        print(f"Error plotting scenario: {e}")
    del pd_survey


def run_experiment(config, experiment_params, experiment_name):
    """
    Run a single experiment with specific SMRF parameters
    
    Args:
        config: Base configuration dictionary
        experiment_params: Dictionary of SMRF parameters for this experiment
        experiment_name: Name of the experiment (used for output directory)
    """
    # Create a copy of the configuration
    exp_config = config.copy()
    
    # Update the SMRF parameters
    for phase in exp_config["pdal_pipeline"]["phase_1"]:
        if phase["type"] == "filters.smrf":
            for param, value in experiment_params.items():
                phase[param] = value
    
    # Create a unique output directory at root level
    base_output_dir = "/mnt/sentinel4To/LIDAR_RUNEXP_MB"
    exp_output_dir = os.path.join(base_output_dir, experiment_name)
    exp_config["paths"]["output_dir"] = exp_output_dir
    
    # Create output directory if it doesn't exist
    os.makedirs(exp_output_dir, exist_ok=True)
    
    # source_path = "/mnt/sentinel4To/LIDAR_cropMB/uncompress"
    # shutil.move(source_path, exp_output_dir)

    # Save the experiment configuration
    config_path = os.path.join(exp_output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(exp_config, f, indent=2)
    
    print(f"Starting experiment: {experiment_name}")
    print(f"SMRF Parameters: {experiment_params}")
    
    # Create and run the LidarProcessor
    lidar_processor = LidarProcessor(
        path=exp_config["paths"]["input_path"],
        group=exp_config["processor_settings"]["group"],
        sampling_strategy=exp_config["processor_settings"]["sampling_strategy"],
        output_dir=exp_output_dir,
        keep_variables=exp_config["processor_settings"]["keep_variables"],
        n_jobs=exp_config["processor_settings"]["n_jobs"],
        n_jobs_uncompress=2 * exp_config["processor_settings"]["n_jobs"],
        pipeline=exp_config["pdal_pipeline"],
    )
    
    lidar_processor.run_pipeline(
        lidar_list_tiles=exp_config["paths"]["lidar_list_tiles"],
        area_of_interest=exp_config["paths"]["area_of_interest"],
        skip_uncompress=False,
    )

    if lidar_processor.files_proc is None:
        print(f"No files processed for experiment: {experiment_name}")
    else:
        # Process output files
        for path in lidar_processor.files_proc:
            print(f"Processing file: {path}")
            try:
                data = np.load(glob.glob(path, recursive=True)[0])
                chm, transf = compute_chm(data, resolution=2.85, quantile=100)
                save_tif(chm, transf, path.split(".")[0] + "_chm.tif")
                print(f"Successfully created CHM for {path}")
            except Exception as e:
                print(f"Error processing {path}: {e}")
        try:
            output_chm = os.path.join(exp_output_dir, "processed", "merged_chm.tif")
            merge_geotiffs(exp_output_dir, output_chm)
        except Exception as e:
            print(f"Error merging GeoTIFFs: {e}")
        try:
            manual_csv = "/home/mgallet/Documents/Lidar/lidar_processing/data/indicators_from_observed_variables_with_stats.csv"
            plot_linear_reg(manual_csv, output_chm)
        except Exception as e:
            print(f"Error plotting linear regression for {path}: {e}")
    print(f"Experiment {experiment_name} completed")
    del lidar_processor
    del data
    del chm
    del transf
    # uncompress_dir = os.path.join(exp_output_dir, "uncompress")
    # shutil.move(uncompress_dir,os.path.dirname(source_path))
    return exp_output_dir

if __name__ == "__main__":

    with open("config/config_csf_molosse.json", "r") as f:
        config = json.load(f)

    # Combinaisons codées en dur de paramètres à tester
    param_combinations = [

                {
            "slope": 0.3,
            "window": 15,
            "threshold": 0.25,
            "scalar": 2.0,
            "cell": 4.0
        },
        {
            "slope": 0.3,
            "window": 50,
            "threshold": 0.25,
            "scalar": 2.0,
            "cell": 2.85
        },
        {
            "slope": 0.6,
            "window": 15,
            "threshold": 0.25,
            "scalar": 2.0,
            "cell": 2.85
        },
        {
            "slope": 0.3,
            "window": 15,
            "threshold": 0.25,
            "scalar": 2.0,
            "cell": 2.5
        },
        {
            "slope": 0.3,
            "window": 15,
            "threshold": 0.35,
            "scalar": 2.0,
            "cell": 2.5
        },
        {
            "slope": 0.3,
            "window": 15,
            "threshold": 0.45,
            "scalar": 2.0,
            "cell": 2.5
        },
        {
            "slope": 0.3,
            "window": 15,
            "threshold": 0.55,
            "scalar": 2.0,
            "cell": 2.5
        },
        {
            "slope": 0.3,
            "window": 15,
            "threshold": 0.35,
            "scalar": 2.5,
            "cell": 2.5
        },
        {
            "slope": 0.3,
            "window": 15,
            "threshold": 0.45,
            "scalar": 2.5,
            "cell": 2.85
        },
        {
            "slope": 0.45,
            "window": 15,
            "threshold": 0.35,
            "scalar": 1.5,
            "cell": 2.85
        }
           
    ]
    
    # Create timestamp for group of experiments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get the root output directory from config
    root_output_dir = "/mnt/sentinel4To/LIDAR_RUNEXP"
    
    # Summary file to store all experiment parameters and results
    summary_file = os.path.join(root_output_dir, f"experiments_summary_{timestamp}.csv")
    os.makedirs(root_output_dir, exist_ok=True)
    
    # Create the header for the summary file
    with open(summary_file, "w") as f:
        header = "experiment_name,slope,window,threshold,scalar,cell,output_dir\n"
        f.write(header)
    
    print(f"Preparing to run {len(param_combinations)} experiments")
    
    # Run experiments
    for i, params in enumerate(param_combinations):
        # Create experiment name
        experiment_name = f"exp_{timestamp}_s{params['slope']}_w{params['window']}_t{params['threshold']}_sc{params['scalar']}_c{params['cell']}"
        experiment_name = experiment_name.replace(".", "p")  # Replace dots with 'p' for file naming
        
        print(f"\nExperiment {i+1}/{len(param_combinations)}")
        
        # Run the experiment
        output_dir = run_experiment(config, params, experiment_name)
        
        # Append results to summary file
        with open(summary_file, "a") as f:
            result_line = f"{experiment_name},{params['slope']},{params['window']},{params['threshold']},{params['scalar']},{params['cell']},{output_dir}\n"
            f.write(result_line)
    
    print(f"\nAll experiments completed. Summary saved to {summary_file}")