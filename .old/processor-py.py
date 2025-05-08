"""
Main LiDAR processor class for handling LiDAR data processing pipelines.
"""

import os
import glob
import json
import time
import gc
import numpy as np
import tracemalloc
import psutil
from hashlib import sha256
from joblib import Parallel, delayed

from pyforestscan.handlers import read_lidar, write_las

from .grid import (
    construct_matrix_coordinates, 
    construct_grid, 
    group_adjacent_tiles_by_n
)
from .visualization import plot_grouped_tiles
from .utils import TermLoading, init_logger, select_and_save_tiles


class LidarProcessor:
    """
    Main class for processing LiDAR data through various pipeline stages.
    
    This processor handles:
    - Uncompressing LiDAR files
    - Cropping to areas of interest
    - Processing tile groups
    - Extracting indicators (to be implemented)
    """
    
    def __init__(
        self, path, group=5, output_dir=None, keep_variables=None, n_jobs=1, **kwargs
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
        **kwargs : dict
            Additional parameters for the LiDAR processing.
        """
        self.path = path
        self.group = group
        self.tiles = glob.glob(os.path.join(self.path, "**/*.laz"), recursive=True)
        self.output_dir = output_dir
        self.keep_variables = keep_variables
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.files_unc = None
        self.tiles_uncomp = None
        
        if self.output_dir is None:
            raise ValueError("Output directory must be specified.")
        else:
            os.makedirs(self.output_dir, exist_ok=True)
            
        self.loader = TermLoading()
        self.log = init_logger(self.output_dir)
        
        self.log.info("LidarProcessor initialized")
        self.log.info(f"Found {len(self.tiles)} tiles in {self.path}")
        self.log.info(f"Group size: {self.group}")
        self.log.info(f"Tiles: {self.tiles}")

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
                    group_paths.append(
                        glob.glob(
                            os.path.join(
                                self.output_dir_uncompress, '**', f"*{tile[0]}*{tile[1]}*.las"
                            ),
                            recursive=True,
                        )[0]
                    )
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

    def _process_group(self, input_file):
        """
        Process a group of LiDAR tiles.
        
        Parameters:
        ----------
        input_file : list
            List of paths to LiDAR files to process as a group.
            
        Returns:
        -------
        str or None
            Path to the processed output file or None if processing failed.
        """
        # Start memory tracking
        tracemalloc.start()
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / (1024 * 1024)
        
        # Process a group of tiles
        diff_dir = os.path.relpath(
            os.path.dirname(input_file[0]), self.output_dir_uncompress
        )
        basename = sha256("".join(input_file).encode()).hexdigest()[:16]
        self.corresponding_files = {}
        name_out = os.path.join(
            self.output_dir_proc,
            diff_dir,
            "processed_" + basename + ".npy",
        )
        os.makedirs(os.path.dirname(name_out), exist_ok=True)

        if isinstance(input_file, str):
            self.log.info(f"Processing single file {input_file}")
            
        try:
            # First attempt with normal parameters
            data = read_lidar(input_file, **self.kwargs["kwargs"])
            if self.keep_variables is not None:
                data = data[0][:][self.keep_variables]
                
            files_ = [os.path.basename(file) for file in input_file]
            self.log.info(
                f"{self.counter}/{len(self.groups)} Saving {len(files_)} files to {name_out}"
            )
            self.counter += 1
            self.corresponding_files[basename] = input_file
            
            np.save(name_out, data)
            del data
            gc.collect()
            
            mem_after = process.memory_info().rss / (1024 * 1024)
            self.log.info(f"Memory: Before={mem_before:.2f}MB, After={mem_after:.2f}MB, Diff={mem_after-mem_before:.2f}MB")
            
            return name_out
            
        except Exception as e:
            self.log.error(f"Error processing {input_file}: {e}")
            
            try: 
                # Second attempt with sample thinning
                self.log.info(f"Trying to process with filter sample 0.5")
                modified_kwargs = self.kwargs.copy()
                modified_kwargs["kwargs"] = self.kwargs["kwargs"].copy() 
                modified_kwargs["kwargs"]["thin_radius"] = 0.5
                
                data = read_lidar(input_file, **modified_kwargs["kwargs"])
                if self.keep_variables is not None:
                    data = data[0][:][self.keep_variables]
                    
                files_ = [os.path.basename(file) for file in input_file]
                self.log.info(
                    f"{self.counter}/{len(self.groups)} Saving {len(files_)} files to {name_out} with correction"
                )
                self.counter += 1
                self.corresponding_files[basename] = input_file
                
                np.save(name_out, data)
                del data
                gc.collect()
                
                return name_out
                
            except Exception as e:
                self.log.error(f"Failed with filter sample: {e}")
                return None

    def uncompress_crop_tiles(self, input_file, lidar_list_tiles, area_of_interest):
        """
        Uncompress LiDAR tiles (from .copc.laz to .laz) and crop them to the area of interest.

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
            self.log.info(f"File {name_out} already exists, skipping.")
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
                self.log.info(f"Uncompressing {input_file} to {name_out}")
                self.counter += 1
                self.log.info(f" Files saved: {self.counter}/{len(self.tiles)}")
                
                return name_out
                
            except Exception as e:
                self.log.error(f"Error processing {input_file}: {e}")
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
        self.loader.show(
            "Uncompressing tiles",
            finish_message="✅ Finished uncompressing tiles",
            failed_message="❌ Failed uncompressing tiles",
        )
        
        self.output_dir_uncompress = os.path.join(self.output_dir, "uncompress")
        os.makedirs(self.output_dir_uncompress, exist_ok=True)
        
        self.log.info(f"Uncompressing tiles to {self.output_dir_uncompress}")
        self.log.info(f"Uncompressing {len(self.tiles)} tiles")
        
        if area_of_interest is not None:
            self.log.info(f"Shapefile of tiles: {lidar_list_tiles}")
            self.log.info(f"Cropping to area of interest: {area_of_interest}")
        else:
            self.log.info("No area of interest provided, processing all tiles")
            
        self.counter = 1
        self.tiles_uncomp = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(self.uncompress_crop_tiles)(
                input_file,
                lidar_list_tiles,
                area_of_interest
            )
            for input_file in self.tiles
        )
        
        self.tiles_uncomp = [
            file for file in self.tiles_uncomp if file is not None
        ]
        
        # Clean up temporary polygon files
        poly_files = glob.glob(
            os.path.join(self.output_dir_uncompress, "temp_*poly*")
        )
        [os.remove(file) for file in poly_files]
        
        self.loader.finished = True

    def _check_existing_files(self):
        """
        Check for already processed files to avoid duplicate processing.
        """
        group2process = self.group_path.copy()
        self.log.info(f"Checking existing files {len(group2process)} groups")
        
        for group in group2process.copy():  # Create a copy to avoid modification during iteration
            diff_dir = os.path.relpath(
                os.path.dirname(group[0]), self.output_dir_uncompress
            )
            basename = sha256("".join(group).encode()).hexdigest()[:16]
            name_out = os.path.join(
                self.output_dir_proc,
                diff_dir,
                "processed_" + basename + ".npy",
            )
            
            if os.path.exists(name_out):
                try:
                    data = np.load(name_out)
                    if data is not None:
                        self.log.info(
                            f"File {name_out} already exists, skipping group"
                        )
                        group2process.remove(group)
                except Exception as e:
                    self.log.error(f"Error loading {name_out}: {e}")
            else:
                self.log.info(f"File {name_out} does not exist, processing group")
                
        self.log.info(f"Processing {len(group2process)} groups")
        self.group_path = group2process

    def process_lidar(self):
        """
        Process the LiDAR tiles in groups.
        """
        self.loader.show(
            "Processing tiles",
            finish_message="✅ Finished processing tiles",
            failed_message="❌ Failed processing tiles",
        )
        
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
        self.counter = 1
        self.log.info(f"Processing {len(self.groups)} groups of tiles")
        self.log.info(f"Keeping variables: {self.keep_variables}")

        self._check_existing_files()

        self.files_proc = Parallel(n_jobs=self.n_jobs, backend="threading")(
            delayed(self._process_group)(
                input_files,
            )
            for input_files in self.group_path
        )
        
        self.files_proc = [
            file for file in self.files_proc if file is not None
        ]
        
        # Save the configuration
        with open(os.path.join(self.output_dir, "configuration.json"), "w") as f:
            json.dump(self.kwargs, f)
            timestamp = time.strftime("%Y%m%d%H%M%S")
            json.dump({"timestamp": timestamp}, f)
            json.dump({"keep_variables": self.keep_variables}, f)
            json.dump(self.corresponding_files, f)

        self.loader.finished = True

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
        diff_dir = os.path.relpath(
            os.path.dirname(input_files[0]), self.output_dir_uncompress
        )
        basename = sha256("".join(input_files).encode()).hexdigest()[:16]
        name_out = os.path.join(
            self.output_dir_proc,
            diff_dir,
            "processed_" + basename + ".npy",
        )
        os.makedirs(os.path.dirname(name_out), exist_ok=True)
        
        try:
            data = read_lidar(input_files, **self.kwargs["kwargs"])
            if self.keep_variables is not None:
                data = data[0][:][self.keep_variables]
            
            np.save(name_out, data)
            
            # Explicit memory cleanup
            del data
            gc.collect()
            
            return (name_out, input_files, basename)
        except Exception as e:
            self.log.error(f"Error processing {basename}: {e}")
            try:
                # Try again with more aggressive sampling
                modified_kwargs = self.kwargs.copy()
                modified_kwargs["kwargs"] = self.kwargs["kwargs"].copy()
                modified_kwargs["kwargs"]["thin_radius"] = 0.5
                
                data = read_lidar(input_files, **modified_kwargs["kwargs"])
                if self.keep_variables is not None:
                    data = data[0][:][self.keep_variables]
                
                np.save(name_out, data)
                
                # Explicit memory cleanup
                del data
                gc.collect()
                
                return (name_out, input_files, basename)
            except Exception as e:
                self.log.error(f"Failed with filter sample: {e}")
                return (None, None, None)

    def process_lidar_sequential(self):
        """
        Process LiDAR tiles sequentially instead of using parallelization.
        Useful for debugging or when memory constraints are severe.
        """
        self.loader.show(
            "Processing tiles sequentially",
            finish_message="✅ Finished processing tiles",
            failed_message="❌ Failed processing tiles",
        )
        
        # Setup output directories and other initialization
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
        self.counter = 1
        self.log.info(f"Processing {len(self.groups)} groups of tiles")
        self.log.info(f"Keeping variables: {self.keep_variables}")

        self._check_existing_files()
        
        # Process groups sequentially
        self.files_proc = []
        
        for input_files in self.group_path:
            result = self.process_worker(input_files)
            name_out, input_files, basename = result
            if name_out:
                self.files_proc.append(name_out)
                files_ = [os.path.basename(file) for file in input_files]
                self.log.info(f"{self.counter}/{len(self.groups)} Saved {len(files_)} files to {name_out}")
                self.counter += 1
                self.corresponding_files[basename] = input_files
        
        self.files_proc = [file for file in self.files_proc if file is not None]
        
        # Save the configuration
        with open(os.path.join(self.output_dir, "configuration.json"), "w") as f:
            json.dump(self.kwargs, f)
            timestamp = time.strftime("%Y%m%d%H%M%S")
            json.dump({"timestamp": timestamp}, f)
            json.dump({"keep_variables": self.keep_variables}, f)
            json.dump(self.corresponding_files, f)

        self.loader.finished = True

    def run_pipeline(self, lidar_list_tiles=None, area_of_interest=None, skip_uncompress=False):
        """
        Run the entire processing pipeline.
        
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
            self.log.info("Starting Lidar processing pipeline")
            t_start = time.time()
            
            if not skip_uncompress:
                t_uncompress = time.time()
                self.uncompress_lidar(lidar_list_tiles, area_of_interest)
                self.log.info("#" * 50)
                self.log.info(f"Uncompressing and cropping tiles finished in {time.time() - t_uncompress:.2f} seconds")
            
            t_process = time.time()
            self.process_lidar()
            self.log.info("#" * 50)
            self.log.info(f"Processing tiles finished in {time.time() - t_process:.2f} seconds")
            
            self.log.info(f"Global processing time: {time.time() - t_start:.2f} seconds")
            self.log.info("Lidar processing pipeline finished")
            
        except Exception as e:
            self.log.error(f"Fatal error in pipeline: {e}", exc_info=True)
            raise
