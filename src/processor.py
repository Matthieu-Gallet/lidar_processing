"""
Main LiDAR processor class for handling LiDAR data processing pipelines.
With improved memory management and resource utilization.
"""

import os, sys
import glob
import json
import time
import gc
import traceback
import numpy as np
import tracemalloc
import psutil
import subprocess
import tempfile
from hashlib import sha256
from joblib import Parallel, delayed
from tqdm import tqdm
import logging
import shutil
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Union
from .uncompress import uncompress_crop_tiles

from .grid import (
    construct_matrix_coordinates,
    construct_grid,
    group_adjacent_tiles_by_n,
    select_group_tiles,
)
from .visualization import plot_grouped_tiles
from .utils import init_logger, generate_hash

from .monitoring import MemoryMonitor
from .status import ProcessingStatus


class LidarProcessor:
    """
    Main class for processing LiDAR data through various pipeline stages.

    Enhanced with:
    - Memory-aware processing
    - Adaptive batch sizes
    - Quality level degradation
    - Progress tracking and recovery
    - Detailed logging and visualization
    """

    def __init__(
        self,
        path,
        group=5,
        output_dir=None,
        keep_variables=None,
        n_jobs=1,
        n_jobs_uncompress=1,
        memory_limit=None,
        **kwargs,
    ):
        """
        Initialize the LiDAR processor.

        Parameters:
        ----------
        path : str
            Path to the directory containing LiDAR files.
        group : int
            Number of tiles to process as a group. Default is 5.
        output_dir : str
            Directory to save processed files. Required.
        keep_variables : list
            List of variables to keep from the LiDAR data. Default is None (keep all).
        n_jobs : int
            Number of parallel jobs to run. Default is 1.
        n_jobs_uncompress : int
            Number of parallel jobs for uncompressing tiles. Default is 1.
        memory_limit : int
            Memory limit per worker in MB. Default is None (no limit).
        **kwargs : dict
            Additional parameters for the LiDAR processing.
        """
        self.path = path
        self.group = group
        self.tiles = glob.glob(os.path.join(self.path, "**/*.laz"), recursive=True)
        self.output_dir = output_dir
        self.options = {"keep_variables": keep_variables}
        self.options["thin_radius"] = kwargs.get("thin_radius", None)
        self.options["quality_levels"] = [
            {"level": "high", "thin_radius": None},
            {"level": "medium", "thin_radius": 0.5},
            {"level": "low", "thin_radius": 1.0},
            {"level": "minimal", "thin_radius": 2.0},
        ]
        self.n_jobs_uncompress = min(n_jobs_uncompress, os.cpu_count() or 1)
        self.n_jobs = min(n_jobs, os.cpu_count() or 1)
        self.memory_limit = memory_limit or self._estimate_memory_limit()
        self.kwargs = kwargs
        self.files_unc = None
        self.tiles_uncomp = None
        self.memory_monitor = None
        self.current_quality_level = 0  # Start with highest quality

        # Create output directory
        if self.output_dir is None:
            raise ValueError("Output directory must be specified.")
        else:
            os.makedirs(self.output_dir, exist_ok=True)

        # Initialize logging
        self.log = init_logger(self.output_dir)

        # Initialize status tracking
        status_file = os.path.join(self.output_dir, "processing_status.json")
        self.status = ProcessingStatus(status_file)

        # Print initialization information
        self.log.info("=" * 50)
        self.log.info("LidarProcessor initialized")
        self.log.info(f"Found {len(self.tiles)} tiles in {self.path}")
        self.log.info(f"Group size: {self.group}")
        self.log.info(f"Using {self.n_jobs} workers")
        self.log.info(f"Memory limit: {self.memory_limit} MB")
        self.log.info("=" * 50)

    def __getstate__(self):
        """
        Prépare l'état de l'objet pour le pickling.
        Exclut les attributs non-picklables comme le logger et le memory monitor.
        """
        state = self.__dict__.copy()  # Copie du dictionnaire des attributs
        # Supprimer les attributs non-picklables
        if "log" in state:
            del state["log"]
        if "memory_monitor" in state:
            del state["memory_monitor"]
        # Ajoutez ici d'autres attributs non-picklables si nécessaire
        return state

    def __setstate__(self, state):
        """
        Restaure l'état de l'objet à partir de l'état picklé (dans le worker).
        Réinitialise les attributs qui n'ont pas été picklés.
        """
        self.__dict__.update(state)
        self.memory_monitor = None
        self.log = None

    def _estimate_memory_limit(self):
        """Estimate memory limit based on system memory."""
        system_memory = psutil.virtual_memory().total / (1024 * 1024)  # MB
        # Use 75% of available memory as a default limit
        return int(system_memory * 0.75 / self.n_jobs)

    def uncompress_lidar(self, lidar_list_tiles, area_of_interest=None):
        """
        Uncompress all LiDAR tiles and optionally crop them to the area of interest.

        Parameters:
        ----------
        lidar_list_tiles : str
            Path to the shapefile containing the list of LiDAR tiles.
        area_of_interest : str, optional
            Path to the shapefile containing the area of interest.
        """
        self.output_dir_uncompress = os.path.join(self.output_dir, "uncompress")
        os.makedirs(self.output_dir_uncompress, exist_ok=True)

        self.log.info(f"Uncompressing tiles to {self.output_dir_uncompress}")
        self.log.info(f"Uncompressing {len(self.tiles)} tiles")

        if area_of_interest is not None:
            self.log.info(f"Shapefile of tiles: {lidar_list_tiles}")
            self.log.info(f"Area of interest: {area_of_interest}")
        else:
            self.log.info("No area of interest provided, processing all tiles")

        # Check for already uncompressed files
        existing_files = glob.glob(
            os.path.join(self.output_dir_uncompress, "**/*.las"), recursive=True
        )
        existing_basenames = {
            os.path.basename(f).split("_")[1].split(".")[0] for f in existing_files
        }

        tiles_to_process = []
        for tile in self.tiles:
            basename = os.path.basename(tile).split(".")[0]
            if basename not in existing_basenames:
                tiles_to_process.append(tile)
                self.log.info(f"Tile {tile} needs to be uncompressed")

        self.log.info(f"Found {len(existing_files)} already uncompressed files")
        self.log.info(f"Need to uncompress {len(tiles_to_process)} more tiles")

        if not tiles_to_process:
            self.log.info("All tiles already uncompressed, skipping this step")
            self.tiles_uncomp = existing_files
            return

        results = Parallel(n_jobs=self.n_jobs_uncompress, verbose=100)(
            delayed(uncompress_crop_tiles)(
                self.path,
                self.output_dir_uncompress,
                input_file,
                lidar_list_tiles,
                area_of_interest,
                self.log,
            )
            for input_file in tiles_to_process
        )
        # Add valid results to list
        valid_results = [r for r in results if r is not None]
        self.tiles_uncomp.extend(valid_results)

        # Force garbage collection
        gc.collect()
        self.log.info(f"Uncompressed {len(valid_results)} tiles successfully")

    def _check_existing_files(self):
        """
        Check for already processed files to avoid duplicate processing.
        """
        group2process = []

        for group in self.group_path:
            basename = generate_hash(group)

            if not self.status.is_processed(basename):
                group2process.append(group)

        self.log.info(
            f"Found {len(self.group_path) - len(group2process)} already processed groups"
        )
        self.log.info(f"Need to process {len(group2process)} more groups")
        self.group_path = group2process

    def process_worker(self, input_files):
        """
        Worker function for processing a group of tiles.

        Parameters:
        ----------
        input_files : list
            List of file paths to process.

        Returns:
        -------
        tuple
            (output path, input files, basename) or (None, None, None) if processing failed.
        """
        # Initialiser un logger spécifique au worker si self.log est None
        current_log = (
            self.log if self.log else logging.getLogger(f"worker_{os.getpid()}")
        )

        start_time = time.time()
        if not input_files:
            current_log.warning("No input files provided for processing.")
            return (None, None, None)

        # Create unique ID for this group
        basename = generate_hash(input_files)

        # Check if already processed
        if self.status.is_processed(basename):
            current_log.info(f"Group {basename} already processed, skipping.")

            # Find the output file
            diff_dir = os.path.relpath(
                os.path.dirname(input_files[0]), self.output_dir_uncompress
            )
            name_out = os.path.join(
                self.output_dir_proc,
                diff_dir,
                "processed_" + basename + ".npy",
            )

            if os.path.exists(name_out):
                return (name_out, input_files, basename)

        # Start memory tracking
        tracemalloc.start()
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)

        # Prepare output path
        diff_dir = os.path.relpath(
            os.path.dirname(input_files[0]), self.output_dir_uncompress
        )
        name_out = os.path.join(
            self.output_dir_proc,
            diff_dir,
            "processed_" + basename + ".npy",
        )
        os.makedirs(os.path.dirname(name_out), exist_ok=True)

        # Try processing with different quality levels
        for quality_idx, quality_level in enumerate(self.options["quality_levels"]):
            if quality_idx < self.current_quality_level:
                continue  # Skip higher quality levels if we've already degraded
                # Apply quality settings
            modified_kwargs = self.kwargs.copy()

            if quality_level["thin_radius"] is not None:
                modified_kwargs["thin_radius"] = quality_level["thin_radius"]

            # Log attempt with quality level
            quality_name = quality_level["level"]
            current_log.info(
                f"Processing {basename} with quality level: {quality_name}"
            )

            # Process data using subprocess
            success_try = True
            try:
                # Create temp directory for subprocess I/O
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Prepare parameters for read_lidar
                    params = {
                        "input_file": input_files,
                        "keep_variables": self.options["keep_variables"],
                        **modified_kwargs,
                    }

                    # Create parameter file
                    params_file = os.path.join(temp_dir, f"params_{basename}.json")
                    with open(params_file, "w") as f:
                        json.dump(params, f)

                    # Create temporary output file for subprocess
                    temp_output = os.path.join(temp_dir, f"output_{basename}.npy")

                    # Path to the read_lidar_subprocess.py script
                    script_path = os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "read_lidar_subprocess.py",
                    )

                    # Make sure the script is executable
                    if not os.access(script_path, os.X_OK):
                        os.chmod(script_path, 0o755)

                    # Execute subprocess
                    process = subprocess.run(
                        [sys.executable, script_path, params_file, temp_output],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if process.stdout:
                        current_log.info(f"Subprocess stdout:\n{process.stdout}")

                    # Check if subprocess was successful
                    if process.returncode != 0:
                        error_msg = f"Subprocess failed with code {process.returncode}"
                        if process.stderr:
                            error_msg += f":\n{process.stderr}"
                        raise Exception(error_msg)

                    # Check if subprocess was successful
                    if process.returncode != 0:
                        raise Exception(f"Subprocess failed: {process.stderr}")

                    # Copy result to final destination
                    if os.path.exists(temp_output):
                        shutil.copy(temp_output, name_out)
                    else:
                        raise Exception(
                            f"Output file {temp_output} not created by subprocess"
                        )

                    # Read the output file

                # Clean up memory
                gc.collect()

            except Exception as e:
                success_try = False
                current_log.warning(
                    f"Failed to process {basename} with quality level {quality_level['level']}: {e}"
                )
                tracemalloc.stop()

                # If we've reached the last quality level, mark as failed
                if quality_idx == len(self.options["quality_levels"]) - 1:
                    current_log.error(f"All quality levels failed for {basename}")
                    self.status.mark_failed(basename)
                    return (None, None, None)

            # Track memory
            if success_try:
                current_log.info(
                    "=" * 25
                    + f" ✅ Group {basename} processed successfully "
                    + "=" * 25
                )
                current_log.info(
                    f"Quality: {quality_name}, "
                    f"Memory: Before={mem_before:.2f}MB, "
                    f"Time: {time.time()-start_time:.2f}s"
                )
                self.remaining_groups.remove(input_files)
                percentage = (
                    (len(self.group_path) - len(self.remaining_groups))
                    * 100
                    / len(self.group_path)
                )
                # Mark as processed in status tracker
                self.corresponding_files[basename] = input_files
                check = self._save_checkpoint(basename)
                current_log.info(f"Checkpoint saved: {True if check else False}")
                self.status.mark_processed(basename)
                current_log.info("=" * 50)

                return (name_out, input_files, basename)

        # This should not be reached, but just in case
        return (None, None, None)

    def process_lidar(self):
        """
        Process the LiDAR tiles in groups with adaptive batch processing.
        Uses memory monitoring to adjust batch sizes dynamically.
        """
        if self.tiles_uncomp is None:
            try:
                self.output_dir_uncompress = os.path.join(self.output_dir, "uncompress")
                self.tiles_uncomp = glob.glob(
                    os.path.join(self.output_dir_uncompress, "**/*.las"), recursive=True
                )
            except Exception as e:
                self.log.error(f"Error finding uncompressed files: {e}")

        self.corresponding_files = {}
        self.output_dir_proc = os.path.join(self.output_dir, "processed")
        os.makedirs(self.output_dir_proc, exist_ok=True)

        plot_file = os.path.join(self.output_dir_proc, "strategy_grouped_tiles.png")
        if self.tiles_uncomp is None:
            self.log.info("No uncompressed tiles found, using original tiles.")
            self.log.info(f"First tile: {self.tiles[0]}")
            original = True
        else:
            self.log.info("Using uncompressed tiles for processing.")
            self.log.info(f"First tile: {self.tiles_uncomp[0]}")
            original = False

        self.group_path = select_group_tiles(
            self.tiles,
            self.output_dir_uncompress,
            tilesingroup=self.group,
            original=original,
            plot_file=plot_file,
            log=self.log,
        )

        self.log.info(f"Processing {len(self.group_path)} groups of tiles")
        self.log.info(f"Keeping variables: {self.options["keep_variables"]}")
        self._check_existing_files()

        if not self.group_path:
            self.log.info("All groups already processed, skipping processing step")
            return

        # Initialize memory monitoring
        self.memory_monitor = MemoryMonitor(
            threshold_mb=self.memory_limit * 0.9,
            critical_threshold_mb=self.memory_limit * 0.975,
        )
        self.memory_monitor.start()

        # Process groups in adaptive batches
        self.files_proc = []

        self.remaining_groups = self.group_path.copy()
        self.process_time = time.time()

        self.files_proc = Parallel(n_jobs=self.n_jobs, verbose=100)(
            delayed(self.process_worker)(input_files) for input_files in self.group_path
        )

        self.files_proc_valid = [r[0] for r in self.files_proc if r[0] is not None]
        self.log.info(
            f"Processed {len(self.files_proc_valid)} groups valid out of {len(self.group_path)}"
        )
        self.files_proc = self.files_proc_valid
        self.log.info(f"Processing time: {time.time() - self.process_time:.2f} seconds")

        # Force garbage collection
        gc.collect()

        # Stop memory monitoring
        if self.memory_monitor:
            self.memory_monitor.stop()
            self.memory_monitor = None

        # Final checkpoint
        self._save_checkpoint()

        self.log.info(
            f"Processing complete: {len(self.files_proc)}/{len(self.group_path)} groups processed successfully"
        )

    def _save_checkpoint(self, basename=None):
        """
        Save processing checkpoint and configuration to disk.
        """
        try:
            # Save configuration and status
            config_file = os.path.join(self.output_dir, "configuration.json")
            with open(config_file, "a") as f:
                json.dump(
                    {
                        "parameters": self.kwargs,
                        "timestamp": time.strftime("%Y%m%d%H%M%S"),
                        "keep_variables": self.options["keep_variables"],
                        "processed_files": len(self.group_path)
                        - len(self.remaining_groups),
                        "total_files": len(self.group_path),
                        "quality_level": self.options["quality_levels"][
                            self.current_quality_level
                        ]["level"],
                        "memory_peak": (
                            self.memory_monitor.peak_memory
                            if self.memory_monitor
                            else None
                        ),
                        "basename": basename if basename else None,
                    },
                    f,
                    indent=4,
                )

            # Save corresponding files mapping separately (could be large)
            corr_file = os.path.join(self.output_dir, "corresponding_files.json")
            with open(corr_file, "w") as f:
                json.dump(self.corresponding_files, f)
            return 1
        except Exception as e:
            self.log.error(f"Failed to save checkpoint: {e}")

    def run_pipeline(
        self, lidar_list_tiles=None, area_of_interest=None, skip_uncompress=False
    ):
        """
        Run the entire processing pipeline with improved resilience and resource management.

        Parameters:
        ----------
        lidar_list_tiles : str
            Path to the shapefile containing the list of LiDAR tiles.
        area_of_interest : str, optional
            Path to the shapefile containing the area of interest.
        skip_uncompress : bool, optional
            If True, skip the uncompression step. Default is False.
        """
        try:
            self.log.info("=" * 80)
            self.log.info(
                "Starting Lidar processing pipeline with enhanced memory management"
            )
            self.log.info(f"Memory limit: {self.memory_limit} MB per worker")
            self.log.info(f"Using {self.n_jobs} workers")
            self.log.info("=" * 80)

            t_start = time.time()

            if not skip_uncompress:
                t_uncompress = time.time()
                self.log.info("Phase 1: Uncompressing and cropping tiles")
                self.uncompress_lidar(lidar_list_tiles, area_of_interest)
                self.log.info("-" * 60)
                self.log.info(
                    f"Phase 1 completed in {time.time() - t_uncompress:.2f} seconds"
                )

            t_process = time.time()
            self.log.info("Phase 2: Processing tiles with adaptive resource management")
            self.process_lidar()
            self.log.info("-" * 60)
            self.log.info(f"Phase 2 completed in {time.time() - t_process:.2f} seconds")

            self.log.info("=" * 80)
            self.log.info(f"Pipeline completed in {time.time() - t_start:.2f} seconds")

            # Print final statistics
            quality_level = self.options["quality_levels"][self.current_quality_level][
                "level"
            ]
            self.log.info(
                f"Processed {len(self.files_proc)}/{len(self.group_path)} groups successfully"
            )
            self.log.info(f"Final quality level used: {quality_level}")
            if self.memory_monitor:
                self.log.info(
                    f"Peak memory usage: {self.memory_monitor.peak_memory:.2f} MB"
                )
            self.log.info("=" * 80)

        except Exception as e:
            self.log.error("!" * 80)
            self.log.error(f"Fatal error in pipeline: {e}")
            self.log.error(traceback.format_exc())
            self.log.error("!" * 80)
            raise
