#!/usr/bin/env python
"""
Command line interface for PyForestLidar.
"""

import os
import json
import click
import logging
from pathlib import Path

from src.processor import LidarProcessor


def load_config(config_path):
    """Load configuration from JSON file."""
    if config_path:
        with open(config_path, 'r') as f:
            return json.load(f)
    else:
        # Default configuration
        default_config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "config", 
            "default_config.json"
        )
        if os.path.exists(default_config_path):
            with open(default_config_path, 'r') as f:
                return json.load(f)
        else:
            return {"kwargs": {"srs": "EPSG:2154", "hag": False, "crop_poly": False}}


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output.')
def cli(verbose):
    """PyForestLidar - Process LiDAR data for forestry applications."""
    log_level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')


@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.argument('output_path', type=click.Path())
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to configuration file.')
def process(input_path, output_path, config):
    """Process LiDAR data from INPUT_PATH and save results to OUTPUT_PATH."""
    click.echo(f"Processing LiDAR data from {input_path} to {output_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Load configuration
    config_dict = load_config(config)
    
    # Extract parameters from config
    processing_params = config_dict.get('processing', {})
    group_size = processing_params.get('group_size', 5)
    jobs = processing_params.get('jobs', 1)
    keep_variables = processing_params.get('variables', None)
    lidar_tiles = processing_params.get('lidar_tiles', None)
    aoi = processing_params.get('aoi', None)
    skip_uncompress = processing_params.get('skip_uncompress', False)
    sequential = processing_params.get('sequential', False)
    memory_limit = processing_params.get('memory_limit', None)
    
    click.echo(f"Using configuration: Group size={group_size}, Jobs={jobs}")
    
    # Initialize processor
    processor = LidarProcessor(
        path=input_path,
        group=group_size,
        output_dir=output_path,
        keep_variables=keep_variables,
        n_jobs=jobs,
        memory_limit=memory_limit,
        **config_dict
    )
    
    # Run pipeline
    if sequential:
        click.echo("Using sequential processing mode")
        if not skip_uncompress:
            processor.uncompress_lidar(lidar_tiles, aoi)
        processor.process_lidar_sequential()
    else:
        click.echo("Using parallel processing mode")
        processor.run_pipeline(lidar_tiles, aoi, skip_uncompress)
    
    click.echo("Processing complete!")



if __name__ == '__main__':
    cli()