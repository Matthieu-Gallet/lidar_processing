import pdal, json
import numpy as np
import os
import geopandas as gpd
import gc


def load_polygon_from_file(vector_file_path, index=0):
    """
    Load a polygon geometry and its CRS from a given vector file.

    Parameters:
    ----------
    vector_file_path : str
        Path to the vector file containing the polygon.
    index : int, optional
        Index of the polygon to be loaded (default is 0).
    Returns:
    -------
    tuple
        containing the Well-Known Text (WKT) representation of the polygon
        and the coordinate reference system (CRS) as a string.
    Raises:
    ------
    FileNotFoundError
        If the vector file does not exist.
    ValueError
        If the file cannot be read or is not a valid vector file format.
    """
    if not os.path.isfile(vector_file_path):
        raise FileNotFoundError(f"No such file: '{vector_file_path}'")

    try:
        gdf = gpd.read_file(vector_file_path)
    except Exception as e:
        raise ValueError(
            f"Unable to read file: {vector_file_path}. Ensure it is a valid vector file format."
        ) from e

    polygon = gdf.loc[index, "geometry"]

    return polygon.wkt, gdf.crs.to_string()


def select_and_save_tiles(
    tuiles_path, parcelle_path, name_file, name_out="temp.shp", crop=1
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
            return name_out
        else:
            return 1
    return 0


def uncompress_crop_tiles(
    root_dir,
    output_dir,
    input_file,
    lidar_list_tiles,
    area_of_interest,
    log,
    crop_D=True,
):
    """
    Uncompress LiDAR tiles and crop them to the area of interest.

    Parameters:
    ----------
    root_dir : str
        Root directory for the LiDAR files.
    output_dir : str
        Directory to save the uncompressed files.
    input_file : str
        Path to the input file (LiDAR tiles).
    lidar_list_tiles : str
        Path to the shapefile containing the list of LiDAR tiles.
    area_of_interest : str
        Path to the shapefile containing the area of interest.
    log : logging.Logger
        Logger instance for logging messages.

    Returns:
    -------
    str or None
        Path to the uncompressed output file or None if processing failed.
    """
    if root_dir is None:
        name_out = os.path.join(
            output_dir,
            "uncompress_" + os.path.basename(input_file).split(".")[0] + ".las",
        )
    else:
        diff_dir = os.path.relpath(
            os.path.dirname(input_file), os.path.dirname(root_dir)
        )
        name_out = os.path.join(
            output_dir,
            diff_dir,
            "uncompress_" + os.path.basename(input_file).split(".")[0] + ".las",
        )
        os.makedirs(os.path.dirname(name_out), exist_ok=True)

    if os.path.exists(name_out):
        return name_out
    else:
        try:

            check_tile = select_and_save_tiles(
                tuiles_path=lidar_list_tiles,
                parcelle_path=area_of_interest,
                name_file=input_file,
                name_out=name_out.replace(".las", ".shp"),
                crop=crop_D,
            )
            if check_tile == 1:
                uncompress_lidar(input_file, name_out)
            elif check_tile != 0:
                uncompress_lidar(input_file, name_out, crop=check_tile)

            return name_out

        except Exception as e:
            log.error(f"Error uncompressing {input_file}: {e}")
            return None


def uncompress_lidar(input_file, output_file, crop=None):
    """
    Uncompress a LAS file using PDAL.

    Parameters:
    ----------
    input_file : str
        Path to the input LAS file.
    output_file : str
        Path to the output LAS file.
    crop : bool
        If True, the output will be the intersection of the tile and the area of interest.
        If False, the output will be the tile itself.

    Returns:
    -------
    int
        Returns 1 if the operation is successful.
    """
    pipe = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": input_file,
                "spatialreference": "EPSG:2154",
            }
        ]
    }
    if crop is not None:
        polygon_wkt, _ = load_polygon_from_file(crop)
        pipe["pipeline"].append(
            {
                "type": "filters.crop",
                "polygon": polygon_wkt,
            }
        )
    pipe["pipeline"].append(
        {
            "type": "filters.range",
            "limits": "Z[1300:2900]",
        }
    )
    pipe["pipeline"].append(
        {
            "type": "writers.las",
            "filename": output_file,
            "compression": "false",
            "minor_version": "4",
            "forward": "all",
        }
    )

    pipeline = pdal.Pipeline(json.dumps(pipe))
    pipeline.execute()
    return 1


if __name__ == "__main__":
    input_file = "/home/mgallet/Documents/Herbiland/vegetation/DATA/LIDAR/RAW/peclerey/LHD_FXX_1007_6552_PTS_C_LAMB93_IGN69.copc.laz"
    name_out = (
        "/home/mgallet/Téléchargements/LHD_FXX_0987_6571_PTS_C_LAMB93_IGN69_999999.las"
    )
    import time

    print("Start uncompressing")
    start = time.time()
    uncompress_lidar(input_file, name_out)
    print("Uncompressing took", time.time() - start, "seconds")
