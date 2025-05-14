import numpy as np
import rasterio
from rasterio.transform import from_origin


def compute_chm(points, resolution=1.0, quantile=100, output_tif=None):
    # Extraire les colonnes
    x = points["X"]
    y = points["Y"]
    z = points["HeightAboveGround"]

    # Définir l’étendue
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    width = int(np.ceil((xmax - xmin) / resolution))
    height = int(np.ceil((ymax - ymin) / resolution))

    # Créer la grille CHM avec les dimensions correctes
    chm = np.full((height, width), np.nan, dtype=np.float32)

    # Convertir coordonnées XY en indices de grille
    col = np.clip(((x - xmin) / resolution).astype(int), 0, width - 1)
    row = np.clip(((ymax - y) / resolution).astype(int), 0, height - 1)

    from collections import defaultdict

    grid_dict = defaultdict(list)
    for r, c, val in zip(row, col, z):
        grid_dict[(r, c)].append(val)

    # Fonction pour calculer le quantile sur une cellule
    def process_cell(cell):
        r, c = cell
        values = grid_dict[(r, c)]
        return (r, c, np.nanquantile(values, quantile / 100.0))

    cells = list(grid_dict.keys())
    results = [process_cell(cell) for cell in cells]
    for r, c, val in results:
        chm[r, c] = val

    if output_tif:
        transform = from_origin(xmin, ymax, resolution, resolution)
        with rasterio.open(
            output_tif,
            "w",
            driver="GTiff",
            height=chm.shape[0] - 1,
            width=chm.shape[1] - 1,
            count=1,
            dtype=chm.dtype,
            crs="EPSG:2154",
            transform=transform,
            nodata=np.nan,
        ) as dst:
            dst.write(chm[:-1, :-1], 1)

    return chm


# Exemple d'utilisation
# chm = compute_chm(mon_array, resolution=1.0, quantile=95, output_tif="chm_95.tif")
