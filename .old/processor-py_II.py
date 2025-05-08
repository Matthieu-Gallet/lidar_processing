"""
Main LiDAR processor class for handling LiDAR data processing pipelines.
With improved memory management and resource utilization.
"""

import os
import glob
import json
import time
import gc
import numpy as np
import tracemalloc
import psutil
import threading
from hashlib import sha256
from joblib import Parallel, delayed
from tqdm import tqdm
import logging
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any, Union

from pyforestscan.handlers import read_lidar, write_las

from .grid import (
    construct_matrix_coordinates, 
    construct_grid, 
    group_adjacent_tiles_by_n
)
from .visualization import plot_grouped_tiles
from .utils import init_logger, select_and_save_tiles


@dataclass
class ProcessingOptions:
    """Configuration options for LiDAR processing."""
    keep_variables: Optional[List[str]] = None
    thin_radius: Optional[float] = None
    quality_levels: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Set default quality levels if none provided."""
        if self.quality_levels is None:
            self.quality_levels = [
                {"name": "high", "thin_radius": None},
                {"name": "medium", "thin_radius": 0.5},
                {"name": "low", "thin_radius": 1.0},
                {"name": "minimal", "thin_radius": 2.0}
            ]


class MemoryMonitor:
    """
    Class to monitor memory usage during processing.
    Provides real-time feedback and adaptive recommendations.
    """
    
    def __init__(self, threshold_mb=8000, critical_threshold_mb=9500, update_interval=1):
        """
        Initialize memory monitor.
        
        Parameters:
        ----------
        threshold_mb : int
            Warning memory threshold in MB.
        critical_threshold_mb : int
            Critical memory threshold in MB.
        update_interval : float
            Interval in seconds between memory checks.
        """
        self.threshold_mb = threshold_mb
        self.critical_threshold_mb = critical_threshold_mb
        self.update_interval = update_interval
        self.running = False
        self.thread = None
        self.peak_memory = 0
        self.current_memory = 0
        self.history = []
        self.warning_triggered = False
        self.critical_triggered = False
        self.logger = logging.getLogger("MemoryMonitor")
        
    def start(self):
        """Start monitoring memory usage."""
        self.running = True
        self.peak_memory = 0
        self.warning_triggered = False
        self.critical_triggered = False
        self.history = []
        self.thread = threading.Thread(target=self._monitor_memory)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop monitoring memory usage."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
            
    def _monitor_memory(self):
        """Memory monitoring thread function."""
        while self.running:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            self.current_memory = memory_mb
            
            # Track memory history (last 10 data points)
            timestamp = time.time()
            self.history.append((timestamp, memory_mb))
            if len(self.history) > 10:
                self.history.pop(0)
                
            # Update peak memory
            if memory_mb > self.peak_memory:
                self.peak_memory = memory_mb
                
            # Check thresholds
            if memory_mb > self.critical_threshold_mb and not self.critical_triggered:
                self.logger.critical(
                    f"CRITICAL: Memory usage ({memory_mb:.2f} MB) exceeds critical threshold "
                    f"({self.critical_threshold_mb} MB)"
                )
                self.critical_triggered = True
                
            elif memory_mb > self.threshold_mb and not self.warning_triggered:
                self.logger.warning(
                    f"WARNING: Memory usage ({memory_mb:.2f} MB) exceeds threshold "
                    f"({self.threshold_mb} MB)"
                )
                self.warning_triggered = True
                
            # Reset triggers if memory goes back below thresholds
            if memory_mb < self.threshold_mb * 0.9:
                self.warning_triggered = False
                
            if memory_mb < self.critical_threshold_mb * 0.9:
                self.critical_triggered = False
                
            time.sleep(self.update_interval)
    
    def get_trend(self) -> float:
        """
        Calculate memory usage trend (MB/s).
        Positive value means memory is increasing.
        """
        if len(self.history) < 2:
            return 0.0
            
        # Calculate trend from first and last data points
        first_time, first_mem = self.history[0]
        last_time, last_mem = self.history[-1]
        time_diff = last_time - first_time
        
        if time_diff <= 0:
            return 0.0
            
        return (last_mem - first_mem) / time_diff
        
    def should_reduce_batch_size(self) -> bool:
        """Check if batch size should be reduced based on memory usage."""
        # Check if we're approaching the threshold with a rising trend
        trend = self.get_trend()
        approaching_threshold = self.current_memory > self.threshold_mb * 0.8
        
        return (self.warning_triggered or self.critical_triggered or 
                (approaching_threshold and trend > 10))  # Rising by 10MB/s
                
    def should_increase_batch_size(self) -> bool:
        """Check if batch size can be increased based on memory usage."""
        trend = self.get_trend()
        return (self.current_memory < self.threshold_mb * 0.5 and 
                trend < 5 and 
                not self.warning_triggered and 
                not self.critical_triggered)


class ProcessingStatus:
    """Class to track and save processing status for recovery."""
    
    def __init__(self, status_file):
        """
        Initialize processing status tracker.
        
        Parameters:
        ----------
        status_file : str
            Path to the status file.
        """
        self.status_file = status_file
        self.processed_groups = set()
        self.failed_groups = set()
        self.load_status()
        
    def load_status(self):
        """Load processing status from file if it exists."""
        if os.path.exists(self.status_file):
            try:
                with open(self.status_file, 'r') as f:
                    data = json.load(f)
                    self.processed_groups = set(data.get('processed', []))
                    self.failed_groups = set(data.get('failed', []))
            except Exception as e:
                logging.warning(f"Failed to load status file: {e}")
                
    def save_status(self):
        """Save current processing status to file."""
        try:
            with open(self.status_file, 'w') as f:
                json.dump({
                    'processed': list(self.processed_groups),
                    'failed': list(self.failed_groups),
                    'timestamp': time.time()
                }, f)
        except Exception as e:
            logging.error(f"Failed to save status file: {e}")
            
    def mark_processed(self, group_id):
        """Mark a group as processed."""
        self.processed_groups.add(group_id)
        self.save_status()
        
    def mark_failed(self, group_id):
        """Mark a group as failed."""
        self.failed_groups.add(group_id)
        self.save_status()
        
    def is_processed(self, group_id):
        """Check if a group has been processed."""
        return group_id in self.processed_groups
        
    def get_remaining(self, all_groups):
        """Get list of groups that still need processing."""
        all_group_ids = {self._get_group_id(g) for g in all_groups}
        return [g for g in all_groups if self._get_group_id(g) not in self.processed_groups]
        
    def _get_group_id(self, group):
        """Generate a consistent ID for a group."""
        if isinstance(group, list) and len(group) > 0:
            return sha256("".join(group).encode()).hexdigest()[:16]
        return None


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
        self, path, group=5, output_dir=None, keep_variables=None, n_jobs=1, 
        memory_limit=None, chunk_size=None, checkpoint_interval=5,
        **kwargs
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
        memory_limit : int
            Memory limit per worker in MB. Default is None (no limit).
        chunk_size : int
            Initial chunk size for processing. Default is equal to group.
        checkpoint_interval : int
            Number of groups to process before saving checkpoint. Default is 5.
        **kwargs : dict
            Additional parameters for the LiDAR processing.
        """
        self.path = path
        self.group = group
        self.tiles = glob.glob(os.path.join(self.path, "**/*.laz"), recursive=True)
        self.output_dir = output_dir
        self.options = ProcessingOptions(keep_variables=keep_variables)
        self.n_jobs = min(n_jobs, os.cpu_count() or 1)
        self.memory_limit = memory_limit or self._estimate_memory_limit()
        self.chunk_size = chunk_size or group
        self.checkpoint_interval = checkpoint_interval
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
        self.log.info(f"Initial chunk size: {self.chunk_size}")
        self.log.info("=" * 50)
        
    def _estimate_memory_limit(self):
        """Estimate memory limit based on system memory."""
        system_memory = psutil.virtual_memory().total / (1024 * 1024)  # MB
        # Use 75% of available memory as a default limit
        return int(system_memory * 0.75 / self.n_jobs)
        
    def _select_group_tiles(self):
        """
        Select and group tiles based on their coordinates.
        """
        if self.tiles_uncomp is None:
            self.coords = construct_matrix_coordinates(self.tiles)
        else:
            self.coords = construct_matrix_coordinates(self.tiles_uncomp)
            
        self.grid, self.indexes, self.indices = construct_grid(self.coords)
        self.groups = group_adjacent_tiles_by_n(self.grid, self.indexes, n=self.group)
        self._maps_group2tilespath()
    
    def _maps_group2tilespath(self):
        """
        Create a mapping of group IDs to tile paths.
        """
        all_tiles = []
        for group in self.groups:
            group_paths = []
            for tile in group:
                try:
                    matching_files = glob.glob(
                        os.path.join(
                            self.output_dir_uncompress, '**', f"*{tile[0]}*{tile[1]}*.las"
                        ),
                        recursive=True,
                    )
                    if matching_files:
                        group_paths.append(matching_files[0])
                except Exception as e:
                    self.log.error(f"Error mapping tile {tile}: {e}")
                    group_paths.append(None)
            group_paths = [path for path in group_paths if path is not None]
            all_tiles.append(group_paths)
        self.group_path = all_tiles

    def plot_tiles_strategy(self):
        """
        Generate a visualization of the tile grouping strategy.
        """
        output_file = os.path.join(self.output_dir_proc, "strategy_grouped_tiles.png")
        plot_grouped_tiles(
            self.grid, self.groups, self.indices, output_file=output_file
        )

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
        if not input_files:
            return (None, None, None)
            
        # Create unique ID for this group
        basename = sha256("".join(input_files).encode()).hexdigest()[:16]
        
        # Check if already processed
        if self.status.is_processed(basename):
            self.log.info(f"Group {basename} already processed, skipping.")
            
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
        for quality_idx, quality_level in enumerate(self.options.quality_levels):
            if quality_idx < self.current_quality_level:
                continue  # Skip higher quality levels if we've already degraded
                
            try:
                # Apply quality settings
                modified_kwargs = self.kwargs.copy()
                modified_kwargs["kwargs"] = self.kwargs["kwargs"].copy()
                
                if quality_level["thin_radius"] is not None:
                    modified_kwargs["kwargs"]["thin_radius"] = quality_level["thin_radius"]
                
                # Log attempt with quality level
                quality_name = quality_level["name"]
                self.log.info(f"Processing {basename} with quality level: {quality_name}")
                
                # Process data
                data = read_lidar(input_files, **modified_kwargs["kwargs"])
                
                # Filter variables if needed
                if self.options.keep_variables is not None:
                    data = data[0][:][self.options.keep_variables]
                
                # Save data
                np.save(name_out, data)
                
                # Clean up memory
                del data
                gc.collect()
                
                # Track memory
                mem_after = process.memory_info().rss / (1024 * 1024)
                self.log.info(
                    f"Group {basename} processed with {quality_name} quality. "
                    f"Memory: Before={mem_before:.2f}MB, After={mem_after:.2f}MB, "
                    f"Diff={mem_after-mem_before:.2f}MB"
                )
                
                # Mark as processed in status tracker
                self.status.mark_processed(basename)
                
                return (name_out, input_files, basename)
                
            except Exception as e:
                self.log.warning(
                    f"Failed to process {basename} with quality level {quality_level['name']}: {e}"
                )
                tracemalloc.stop()
                
                # If we've reached the last quality level, mark as failed
                if quality_idx == len(self.options.quality_levels) - 1:
                    self.log.error(f"All quality levels failed for {basename}")
                    self.status.mark_failed(basename)
                    return (None, None, None)
        
        # This should not be reached, but just in case
        return (None, None, None)

    def _adapt_quality_level(self, memory_usage):
        """
        Adapt quality level based on memory usage.
        
        Parameters:
        ----------
        memory_usage : float
            Current memory usage in MB.
        """
        max_level = len(self.options.quality_levels) - 1
        
        # Increase quality level (decrease quality) if memory usage is high
        if memory_usage > self.memory_limit * 0.9 and self.current_quality_level < max_level:
            self.current_quality_level += 1
            quality_name = self.options.quality_levels[self.current_quality_level]["name"]
            self.log.warning(
                f"Memory usage high ({memory_usage:.2f} MB). "
                f"Reducing quality to {quality_name}"
            )
            return True
            
        # Decrease quality level (increase quality) if memory usage is low
        elif memory_usage < self.memory_limit * 0.5 and self.current_quality_level > 0:
            self.current_quality_level -= 1
            quality_name = self.options.quality_levels[self.current_quality_level]["name"]
            self.log.info(
                f"Memory usage low ({memory_usage:.2f} MB). "
                f"Increasing quality to {quality_name}"
            )
            return True
            
        return False

    def uncompress_crop_tiles(self, input_file, lidar_list_tiles, area_of_interest):
        """
        Uncompress LiDAR tiles and crop them to the area of interest.

        Parameters:
        ----------
        input_file : str
            Path to the input file (LiDAR tiles).
        lidar_list_tiles : str
            Path to the shapefile containing the list of LiDAR tiles.
        area_of_interest : str
            Path to the shapefile containing the area of interest.
            
        Returns:
        -------
        str or None
            Path to the uncompressed output file or None if processing failed.
        """
        if self.path is None:
            name_out = os.path.join(
                self.output_dir_uncompress,
                "uncompress_" + os.path.basename(input_file).split(".")[0] + ".las",
            )
        else:
            diff_dir = os.path.relpath(
                os.path.dirname(input_file), os.path.dirname(self.path)
            )
            name_out = os.path.join(
                self.output_dir_uncompress,
                diff_dir,
                "uncompress_" + os.path.basename(input_file).split(".")[0] + ".las",
            )
            os.makedirs(os.path.dirname(name_out), exist_ok=True)
            
        if os.path.exists(name_out):
            return name_out
        else:
            try:
                if area_of_interest is not None:
                    name_poly = os.path.join(
                        self.output_dir_uncompress,
                        "temp_" + os.path.basename(input_file).split(".")[0] + "_poly.shp",
                    )
                    select_and_save_tiles(
                        tuiles_path=lidar_list_tiles,
                        parcelle_path=area_of_interest,
                        name_file=input_file,
                        name_out=name_poly,
                    )
                    data = read_lidar(
                        input_file,
                        "EPSG:2154",
                        hag=False,
                        crop_poly=True,
                        poly=name_poly,
                        outlier=None,
                        smrf=False,
                        only_vegetation=False,
                    )
                else:
                    data = read_lidar(
                        input_file,
                        "EPSG:2154",
                        hag=False,
                        crop_poly=False,
                        poly=None,
                        outlier=None,
                        smrf=False,
                        only_vegetation=False,
                    )
                    
                write_las(data, name_out, srs=None, compress=False)
                
                return name_out
                
            except Exception as e:
                self.log.error(f"Error uncompressing {input_file}: {e}")
                return None

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
            self.log.info(f"Cropping to area of interest: {area_of_interest}")
        else:
            self.log.info("No area of interest provided, processing all tiles")
        
        # Check for already uncompressed files
        existing_files = glob.glob(os.path.join(self.output_dir_uncompress, "**/*.las"), recursive=True)
        existing_basenames = {os.path.basename(f).split("_")[1].split(".")[0] for f in existing_files}
        
        tiles_to_process = []
        for tile in self.tiles:
            basename = os.path.basename(tile).split(".")[0]
            if basename not in existing_basenames:
                tiles_to_process.append(tile)
                
        self.log.info(f"Found {len(existing_files)} already uncompressed files")
        self.log.info(f"Need to uncompress {len(tiles_to_process)} more tiles")
        
        if not tiles_to_process:
            self.log.info("All tiles already uncompressed, skipping this step")
            self.tiles_uncomp = existing_files
            return
            
        # Start memory monitor
        if self.memory_limit:
            self.memory_monitor = MemoryMonitor(threshold_mb=self.memory_limit * 0.8)
            self.memory_monitor.start()
        
        # Process in batches to control memory usage
        batch_size = min(100, max(1, int(len(tiles_to_process) / (self.n_jobs * 2))))
        total_batches = (len(tiles_to_process) + batch_size - 1) // batch_size
        
        self.tiles_uncomp = existing_files.copy()
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(tiles_to_process))
            current_batch = tiles_to_process[batch_start:batch_end]
            
            self.log.info(f"Processing batch {batch_idx+1}/{total_batches} ({len(current_batch)} tiles)")
            
            # Process batch with progress bar
            with tqdm(total=len(current_batch), desc=f"Batch {batch_idx+1}/{total_batches}") as pbar:
                # Define callback to update progress
                def update_progress(*args):
                    pbar.update(1)
                    
                results = Parallel(n_jobs=self.n_jobs, backend="threading", verbose=0)(
                    delayed(self.uncompress_crop_tiles)(
                        input_file,
                        lidar_list_tiles,
                        area_of_interest
                    )
                    for input_file in current_batch
                )
                
                # Update progress bar manually since callback may not work with joblib
                pbar.update(len(current_batch))
            
            # Add valid results to list
            valid_results = [r for r in results if r is not None]
            self.tiles_uncomp.extend(valid_results)
            
            # Adjust batch size based on memory usage
            if self.memory_monitor and self.memory_monitor.should_reduce_batch_size():
                batch_size = max(1, int(batch_size * 0.75))
                self.log.warning(f"Reducing batch size to {batch_size} due to memory pressure")
            elif self.memory_monitor and self.memory_monitor.should_increase_batch_size():
                batch_size = min(100, int(batch_size * 1.25))
                self.log.info(f"Increasing batch size to {batch_size}")
            
            # Force garbage collection
            gc.collect()
        
        # Clean up temporary polygon files
        poly_files = glob.glob(
            os.path.join(self.output_dir_uncompress, "temp_*poly*")
        )
        for file in poly_files:
            try:
                os.remove(file)
            except Exception as e:
                self.log.warning(f"Failed to remove temporary file {file}: {e}")
        
        # Stop memory monitor
        if self.memory_monitor:
            self.memory_monitor.stop()
            self.memory_monitor = None

    def _check_existing_files(self):
        """
        Check for already processed files to avoid duplicate processing.
        """
        group2process = []
        
        for group in self.group_path:
            basename = sha256("".join(group).encode()).hexdigest()[:16]
            
            if not self.status.is_processed(basename):
                group2process.append(group)
                
        self.log.info(f"Found {len(self.group_path) - len(group2process)} already processed groups")
        self.log.info(f"Need to process {len(group2process)} more groups")
        self.group_path = group2process

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
        
        self._select_group_tiles()
        self.plot_tiles_strategy()
        self.log.info(f"Processing {len(self.groups)} groups of tiles")
        self.log.info(f"Keeping variables: {self.options.keep_variables}")

        self._check_existing_files()
        
        if not self.group_path:
            self.log.info("All groups already processed, skipping processing step")
            return
        
        # Initialize memory monitoring
        self.memory_monitor = MemoryMonitor(threshold_mb=self.memory_limit * 0.8)
        self.memory_monitor.start()
        
        # Process groups in adaptive batches
        self.files_proc = []
        
        # Start with the configured chunk size
        current_chunk_size = self.chunk_size
        remaining_groups = self.group_path.copy()
        
        with tqdm(total=len(remaining_groups), desc="Processing groups") as pbar:
            while remaining_groups:
                # Take the next chunk of groups
                current_batch = remaining_groups[:current_chunk_size]
                remaining_groups = remaining_groups[current_chunk_size:]
                
                self.log.info(f"Processing batch of {len(current_batch)} groups (chunk size: {current_chunk_size})")
                
                # Process the batch
                batch_start_time = time.time()
                


                batch_results = Parallel(n_jobs=self.n_jobs, backend="threading", verbose=0)(
                    delayed(self.process_worker)(input_files) for input_files in current_batch
                )
                
                # Filter valid results and add to processed files
                valid_results = [r[0] for r in batch_results if r[0] is not None]
                self.files_proc.extend(valid_results)
                
                # Update corresponding files dictionary
                for result in batch_results:
                    if result[0] is not None:
                        self.corresponding_files[result[2]] = result[1]
                
                # Update progress bar
                pbar.update(len(current_batch))
                
                # Calculate batch processing statistics
                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                batch_success_rate = len(valid_results) / len(current_batch) if current_batch else 0
                
                self.log.info(
                    f"Batch processed in {batch_time:.2f}s with {len(valid_results)}/{len(current_batch)} "
                    f"successful ({batch_success_rate*100:.1f}%)"
                )
                
                # Adapt chunk size based on memory usage and processing success
                if self.memory_monitor:
                    # Check if we should adjust batch size
                    if self.memory_monitor.should_reduce_batch_size():
                        # Reduce batch size if memory pressure is high
                        new_chunk_size = max(1, int(current_chunk_size * 0.75))
                        if new_chunk_size != current_chunk_size:
                            self.log.warning(
                                f"Reducing chunk size from {current_chunk_size} to {new_chunk_size} "
                                f"due to memory pressure ({self.memory_monitor.current_memory:.1f} MB)"
                            )
                            current_chunk_size = new_chunk_size
                    elif self.memory_monitor.should_increase_batch_size() and batch_success_rate > 0.9:
                        # Increase batch size if memory usage is low and success rate is high
                        new_chunk_size = min(50, int(current_chunk_size * 1.25))
                        if new_chunk_size != current_chunk_size:
                            self.log.info(
                                f"Increasing chunk size from {current_chunk_size} to {new_chunk_size} "
                                f"(memory: {self.memory_monitor.current_memory:.1f} MB)"
                            )
                            current_chunk_size = new_chunk_size
                            
                # Check if quality level should be adapted based on memory usage
                if self.memory_monitor:
                    self._adapt_quality_level(self.memory_monitor.current_memory)
                    
                # Save checkpoint if needed
                if (len(self.files_proc) % self.checkpoint_interval == 0 or 
                        len(remaining_groups) == 0):
                    self._save_checkpoint()
                    
                # Force garbage collection
                gc.collect()
        
        # Stop memory monitoring
        if self.memory_monitor:
            self.memory_monitor.stop()
            self.memory_monitor = None
            
        # Final checkpoint
        self._save_checkpoint()
        
        self.log.info(f"Processing complete: {len(self.files_proc)}/{len(self.group_path)} groups processed successfully")

    def _save_checkpoint(self):
        """
        Save processing checkpoint and configuration to disk.
        """
        try:
            # Save configuration and status
            config_file = os.path.join(self.output_dir, "configuration.json")
            with open(config_file, "w") as f:
                json.dump({
                    "parameters": self.kwargs,
                    "timestamp": time.strftime("%Y%m%d%H%M%S"),
                    "keep_variables": self.options.keep_variables,
                    "processed_files": len(self.files_proc),
                    "total_files": len(self.group_path),
                    "quality_level": self.options.quality_levels[self.current_quality_level]["name"],
                    "memory_peak": self.memory_monitor.peak_memory if self.memory_monitor else None,
                }, f, indent=2)
                
            # Save corresponding files mapping separately (could be large)
            corr_file = os.path.join(self.output_dir, "corresponding_files.json")
            with open(corr_file, "w") as f:
                json.dump(self.corresponding_files, f)
                
            self.log.info(f"Checkpoint saved: {len(self.files_proc)}/{len(self.group_path)} groups processed")
            
        except Exception as e:
            self.log.error(f"Failed to save checkpoint: {e}")

    def run_pipeline(self, lidar_list_tiles=None, area_of_interest=None, skip_uncompress=False):
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
            self.log.info("Starting Lidar processing pipeline with enhanced memory management")
            self.log.info(f"Memory limit: {self.memory_limit} MB per worker")
            self.log.info(f"Using {self.n_jobs} workers")
            self.log.info(f"Initial chunk size: {self.chunk_size}")
            self.log.info("=" * 80)
            
            t_start = time.time()
            
            if not skip_uncompress:
                t_uncompress = time.time()
                self.log.info("Phase 1: Uncompressing and cropping tiles")
                self.uncompress_lidar(lidar_list_tiles, area_of_interest)
                self.log.info("-" * 60)
                self.log.info(f"Phase 1 completed in {time.time() - t_uncompress:.2f} seconds")
            
            t_process = time.time()
            self.log.info("Phase 2: Processing tiles with adaptive resource management")
            self.process_lidar()
            self.log.info("-" * 60)
            self.log.info(f"Phase 2 completed in {time.time() - t_process:.2f} seconds")
            
            self.log.info("=" * 80)
            self.log.info(f"Pipeline completed in {time.time() - t_start:.2f} seconds")
            
            # Print final statistics
            quality_level = self.options.quality_levels[self.current_quality_level]["name"]
            self.log.info(f"Processed {len(self.files_proc)}/{len(self.group_path)} groups successfully")
            self.log.info(f"Final quality level used: {quality_level}")
            if self.memory_monitor:
                self.log.info(f"Peak memory usage: {self.memory_monitor.peak_memory:.2f} MB")
            self.log.info("=" * 80)
            
        except Exception as e:
            self.log.error("!" * 80)
            self.log.error(f"Fatal error in pipeline: {e}")
            self.log.error(traceback.format_exc())
            self.log.error("!" * 80)
            raise
            