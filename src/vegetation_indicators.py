import numpy as np
import rasterio
from rasterio.transform import from_origin
from collections import defaultdict

from scipy.stats import entropy


def process_cell_chm(cell, grid_dict, quantile=100):
    r, c = cell
    values = grid_dict[(r, c)]
    return (r, c, np.quantile(values, quantile / 100))


def process_cell_fhd(cell, grid_dict, zmin=0, zmax=10, zwidth=0.1):
    r, c = cell
    values = grid_dict[(r, c)]
    bins = np.arange(zmin, zmax + zwidth, zwidth)
    hist, _ = np.histogram(values, bins=bins)
    hist = hist / hist.sum()
    percent_max = entropy(hist)
    return (r, c, percent_max)


def construct_voxels(x, y, z, resolution_xy=1.0):
    # Définir l’étendue
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    width = int(np.ceil((xmax - xmin) / resolution_xy))
    height = int(np.ceil((ymax - ymin) / resolution_xy))

    # Convertir coordonnées XY en indices de grille
    col = np.clip(((x - xmin) / resolution_xy).astype(int), 0, width - 1)
    row = np.clip(((ymax - y) / resolution_xy).astype(int), 0, height - 1)

    cells_dict = defaultdict(list)
    for r, c, val in zip(row, col, z):
        cells_dict[(r, c)].append(val)

    transform = from_origin(xmin, ymax, resolution_xy, resolution_xy)
    return cells_dict, transform, (height, width)


def compute_indicators(cells_dict, size, process_cell, **kwargs):
    data_grid = np.full(size, np.nan, dtype=np.float32)
    results = [
        process_cell(cell, cells_dict, **kwargs) for cell in list(cells_dict.keys())
    ]
    for r, c, val in results:
        data_grid[r, c] = val
    return data_grid


def compute_chm(points, resolution=1.0, quantile=100):
    x = points["X"]
    y = points["Y"]
    z = points["HeightAboveGround"]

    cells_dict, transform, size = construct_voxels(x, y, z, resolution_xy=resolution)
    data_grid = compute_indicators(
        cells_dict, size, process_cell_chm, quantile=quantile
    )
    return data_grid, transform


def compute_fhd(points, resolution=1.0, zmin=0, zmax=10, zwidth=0.1):
    x = points["X"]
    y = points["Y"]
    z = points["HeightAboveGround"]

    cells_dict, transform, size = construct_voxels(x, y, z, resolution_xy=resolution)
    data_grid = compute_indicators(
        cells_dict, size, process_cell_fhd, zmin=zmin, zmax=zmax, zwidth=zwidth
    )
    return data_grid, transform


def save_tif(data, transform, output_tif):
    if data.ndim == 2:
        with rasterio.open(
            output_tif,
            "w",
            driver="GTiff",
            height=data.shape[0] - 1,
            width=data.shape[1] - 1,
            count=1,
            dtype=data.dtype,
            crs="EPSG:2154",
            transform=transform,
            nodata=np.nan,
        ) as dst:
            dst.write(data[:-1, :-1], 1)
    elif data.ndim == 3:
        # Assuming the last dimension is the band dimension
        with rasterio.open(
            output_tif,
            "w",
            driver="GTiff",
            height=data.shape[0] - 1,
            width=data.shape[1] - 1,
            count=data.shape[2],
            dtype=data.dtype,
            crs="EPSG:2154",
            transform=transform,
            nodata=np.nan,
        ) as dst:
            for i in range(data.shape[2]):
                dst.write(data[:, :, i], i + 1)
