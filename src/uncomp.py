import pdal, json
import numpy as np


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
