"""
Utility functions for Lidar processing.
"""

import os
import geopandas as gpd
import time
import logging
from hashlib import sha256
import pdal, json, gc


class TermLoading:
    """Simple terminal loading indicator class."""

    def __init__(self):
        self.finished = False
        self.message = ""
        self.finish_message = ""
        self.failed_message = ""

    def show(self, message, finish_message="Done", failed_message="Failed"):
        """
        Set up the loading display.

        Parameters:
        ----------
        message : str
            Message to display during loading.
        finish_message : str
            Message to display when finished successfully.
        failed_message : str
            Message to display when failed.
        """
        self.message = message
        self.finish_message = finish_message
        self.failed_message = failed_message
        self.finished = False
        print(f"{self.message}...")


def init_logger(output_dir):
    """
    Initialize a logger for the application.

    Parameters:
    ----------
    output_dir : str
        Directory to save log files.

    Returns:
    -------
    logging.Logger
        Configured logger instance.
    """
    # Initialize the logger
    time_str = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"log_{time_str}.txt")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s:%(levelname)s:%(message)s",
    )
    return logging.getLogger(__name__)


def select_and_save_tiles(
    tuiles_path, parcelle_path, name_file, name_out="temp.shp", crop=False
):
    """
    Select and extract the footprint of a shapefile inside a LiDAR tile,
    from the LiDAR tile acquisition shapefile.

    Parameters:
    ----------
    tuiles_path : str
        Path to the shapefile containing the list of LiDAR tiles (square polygons).
    parcelle_path : str
        Path to the shapefile containing the area of interest (polygon).
    name_file : str
        Path to the input file (LiDAR tiles).
    name_out : str, optional
        Path to the output file (shapefile) where the selected tile will be saved.
        If not provided, a temporary file will be created.
    crop : bool, optional
        If True, the output will be the intersection of the tile and the area of interest.
        If False, the output will be the tile itself. Default is True.

    Returns:
    -------
    int
        Returns 1 if the operation is successful.
    """
    name_file = os.path.basename(name_file).split(".")[0]
    coords = name_file.split("_")[2:4]

    # Load the tiles shapefile
    tuiles = gpd.read_file(tuiles_path)

    # Load the parcel shapefile
    parcelle = gpd.read_file(parcelle_path)

    # Filter tiles by coordinates
    tuiles = tuiles[
        tuiles["nom"].str.contains(coords[0]) & tuiles["nom"].str.contains(coords[1])
    ]

    # Intersect and save
    if len(tuiles) > 0:
        if crop:
            parcelle.intersection(tuiles.geometry.iloc[0]).to_file(
                name_out, driver="ESRI Shapefile"
            )
        return 1
    return 0


def generate_hash(input_list):
    """
    Generate a short hash from a list of strings.

    Parameters:
    ----------
    input_list : list
        List of strings to hash.

    Returns:
    -------
    str
        Short hash string.
    """
    return sha256("".join(input_list).encode()).hexdigest()[:16]


def uncompress_lidar(input_file, output_file):
    """
    Uncompress a LAS file using PDAL.

    :param input_file: str, path to the input LAS file.
    :param output_file: str, path to the output LAS file.
    """
    pipeline = pdal.Pipeline(
        json.dumps(
            {
                "pipeline": [
                    {
                        "type": "readers.las",
                        "filename": input_file,
                        "spatialreference": "EPSG:2154",
                    },
                    {"type": "filters.range", "limits": "Z[1200:2800]"},
                    {
                        "type": "writers.las",
                        "filename": output_file,
                        "compression": "false",
                        "minor_version": "4",
                        "forward": "all",
                    },
                ]
            }
        )
    )
    pipeline.execute()
    del pipeline
    gc.collect()
    return 1
