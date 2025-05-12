from src.processor import LidarProcessor
import os, json


if __name__ == "__main__":

    with open("config/default_config.json", "r") as f:
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
