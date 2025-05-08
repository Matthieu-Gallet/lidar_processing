import os, glob, json, time
import numpy as np

from hashlib import sha256
from joblib import Parallel, delayed
from pyforestscan.handlers import read_lidar, write_las
from tqdm import tqdm
import warnings

import geopandas as gpd
from matplotlib import pyplot as plt
from collections import deque

from display import TermLoading
import gc

import logging


def construct_matrix_coordinates(list_tiles):
    """
    Construct a matrix of coordinates from a list of tiles.

    Parameters:
    ----------
    list_tiles : list
        List of tiles to construct the matrix from.

    Returns:
    -------
    list
        List of coordinates of the tiles.
    """
    coords = []
    for tile in list_tiles:
        coord = tile.split("_")[2:4]
        coords.append(np.array(coord, dtype=int))
    coords = np.array(coords)
    return coords


def construct_grid(coords):
    """
    Construct a rectangular grid from a list of coordinates.

    Parameters:
    ----------
    coords : list
        List of coordinates to construct the grid from.
    Returns:
    -------
    grid : numpy.ndarray
        2D numpy array representing the grid.
    indexes : tuple
        Tuple containing two dictionaries for mapping coordinates to indices
    indices : tuple
        Tuple containing two dictionaries for mapping indices to coordinates
    """
    # 1. Normaliser les coordonnées pour les mapper à une grille compacte
    xs, ys = zip(*coords)
    xs = np.array(xs, dtype=int)
    ys = np.array(ys, dtype=int)

    x_indices = {
        i: x for i, x in zip(np.arange(xs.min(), xs.max() + 1), np.arange(0, len(xs)))
    }
    y_indices = {
        i: y for i, y in zip(np.arange(ys.min(), ys.max() + 1), np.arange(0, len(ys)))
    }

    # Inverse mapping pour retrouver les coords plus tard
    index_to_x = {i: x for x, i in x_indices.items()}
    index_to_y = {i: y for y, i in y_indices.items()}

    # 2. Créer la grille logique
    height = len(y_indices)
    width = len(x_indices)
    grid = np.zeros((height, width), dtype=int)
    for x, y in coords:
        i = y_indices[y]
        j = x_indices[x]
        if grid[i, j] == 0:  # Only increment if the cell is not already filled
            grid[i, j] = 1  # 1 = tuile présente
        else:
            warnings.warn(f"Tuile déjà présente à la position ({i}, {j}) : {x}, {y}")
    return grid, (index_to_x, index_to_y), (x_indices, y_indices)


def get_coords(i, j, indexes):
    """
    Get the coordinates corresponding to the indices in the grid.

    Parameters:
    ----------
    i : int
        Row index in the grid.
    j : int
        Column index in the grid.
    indexes : tuple
        Tuple containing two dictionaries for mapping coordinates to indices and vice versa.
    Returns:
    -------
    tuple
        Coordinates corresponding to the indices in the grid.
    """
    index_to_x = indexes[0]
    index_to_y = indexes[1]
    return (index_to_x[j], index_to_y[i])  # attention à l’ordre : numpy = (row, col)


def get_adjacent(i, j, mask):
    for di, dj in [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, 1),
        (1, -1),
    ]:
        ni, nj = i + di, j + dj
        if 0 <= ni < mask.shape[0] and 0 <= nj < mask.shape[1]:
            if mask[ni, nj]:
                yield ni, nj


def group_adjacent_tiles_by_n(grid, indexes, n=-1):
    """
    Group adjacent tiles in the grid into components of size n.

    Parameters
    ----------
    grid : np.ndarray
        2D array representing the grid of tiles.
    n : int
        Size of the groups to form. If -1, all tiles are grouped and if 0, no grouping is done.
        Default is -1.
    Returns
    -------
    list
        List of groups, where each group is a list of coordinates.
    """
    mask = grid == 1
    visited = np.zeros_like(grid, dtype=bool)
    groups = []

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if mask[i, j] and not visited[i, j]:
                queue = deque()
                queue.append((i, j))
                visited[i, j] = True
                component = [(i, j)]

                while queue and len(component) < n:
                    ci, cj = queue.popleft()
                    for ni, nj in get_adjacent(ci, cj, mask):
                        if not visited[ni, nj]:
                            visited[ni, nj] = True
                            queue.append((ni, nj))
                            component.append((ni, nj))

                groups.append([get_coords(i, j, indexes) for i, j in component])
    if n == -1:
        groups = [
            [
                get_coords(i, j, indexes)
                for i in range(grid.shape[0])
                for j in range(grid.shape[1])
                if grid[i, j] == 1
            ]
        ]
    return groups

def plot_grouped_tiles(grid, groups, indices, output_file=None):
    x_indices, y_indices = indices
    group_map = np.ones_like(grid, dtype=int) * -1  # fond blanc (valeur -1)

    for group_id, group in enumerate(groups, start=1):
        for x, y in group:
            j = x_indices[x]  # colonne
            i = y_indices[y]  # ligne
            group_map[i, j] = group_id

    fig, ax = plt.subplots(figsize=(25, 25))
    cmap = plt.get_cmap("gnuplot", len(groups))
    # mélanger l'ordre des couleurs
    cmap = cmap(np.arange(cmap.N))
    np.random.shuffle(cmap)
    cmap = plt.cm.colors.ListedColormap(cmap)
    ax.pcolormesh(
        np.where(grid == 1, group_map, np.nan) - 1,
        cmap=cmap,
        vmin=-1,
        vmax=len(groups),
        alpha=1,
        edgecolors="white",
        linewidth=0.01,
    )
    ax.matshow(
        np.where(grid == 1, np.nan, 0), cmap="gray_r", alpha=1
    )  # tuiles présentes en gris
    # add number of group on the tile
    for group_id, group in enumerate(groups, start=1):
        for x, y in group:
            j = x_indices[x]  # colonne
            i = y_indices[y]  # ligne
            ax.text(
                j + 0.5,
                i + 0.5,
                str(group_id),
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    ax.set_title("Grouped Tiles", fontsize=16)
    ax.set_xticks(np.array(list((x_indices.values()))) + 0.5)
    ax.set_xticklabels(list(x_indices.keys()), fontsize=8, rotation=45)

    ax.set_yticks(np.array(list((y_indices.values()))) + 0.5)
    ax.set_yticklabels(list(y_indices.keys()), fontsize=8, rotation=45)
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.set_aspect("equal")
    ax.grid(which="major", color="black", linestyle="--", linewidth=0.1, alpha=0.5)
    ax.set_xlim(list(x_indices.values())[0] - 2, list(x_indices.values())[-1] + 2)
    ax.set_ylim(list(y_indices.values())[-1] + 2, list(y_indices.values())[0] - 2)
    ax.invert_yaxis()  # Invert the Y-axis to make it decreasing
    ax.tick_params(which="minor", bottom=False, left=False)
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    else:
        plt.show()


def select_and_save_tiles(tuiles_path, parcelle_path, name_file, name_out=None):
    """
    Select and extract the footprint of a shapefile inside a LiDAR tile, from the LiDAR tile acquisition shapefile.

    Parameters:
    ----------
    tuiles_path : str
        Path to the shapefile containing the list of LiDAR tiles (square polygons).
    parcelle_path : str
        Path to the shapefile containing the area of interest (polygon).
    name_file : str
        Path to the input file (LiDAR tiles).
    name_out : str, optional
        Path to the output file (shapefile) where the selected tile will be saved. If not provided, a temporary file will be created.

    Returns:
    -------
    int
        Returns 1 if the operation is successful.

    """
    name_file = os.path.basename(name_file).split(".")[0]
    coords = name_file.split("_")[2:4]
    # Charger le shapefile des tuiles
    tuiles = gpd.read_file(tuiles_path)
    # Charger le shapefile de la parcelle
    parcelle = gpd.read_file(parcelle_path)

    tuiles = tuiles[
        tuiles["nom"].str.contains(coords[0]) & tuiles["nom"].str.contains(coords[1])
    ]
    parcelle.intersection(tuiles.geometry.iloc[0]).to_file(
        name_out, driver="ESRI Shapefile"
    )
    return 1



def construct_matrix_coordinates(list_tiles):
    """
    Construct a matrix of coordinates from a list of tiles.

    Parameters:
    ----------
    list_tiles : list
        List of tiles to construct the matrix from.

    Returns:
    -------
    list
        List of coordinates of the tiles.
    """
    coords = []
    for tile in list_tiles:
        coord = tile.split("_")[3:5]
        coords.append(np.array(coord, dtype=int))
    coords = np.array(coords)
    return coords




class LidarProcessor:
    def __init__(
        self, path, group=5, output_dir=None, keep_variables=None, n_jobs=1, **kwargs
    ):

        self.path = path
        self.group = group
        self.tiles = glob.glob(os.path.join(self.path, "**/*.laz"), recursive=True)
        self.output_dir = output_dir
        self.keep_variables = keep_variables
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        self.files_unc = None

        if self.output_dir is None:
            raise ValueError("Output directory must be specified.")
        else:
            os.makedirs(self.output_dir, exist_ok=True)
        self.tiles_uncomp = None
        self.loader = TermLoading()
        self.log = self._init_logger()
        self.log.info("LidarProcessor initialized")
        self.log.info(f"Found {len(self.tiles)} tiles in {self.path}")
        self.log.info(f"Group size: {self.group}")
        self.log.info(f"Tiles: {self.tiles}")

    def _init_logger(self):
        # Initialize the logger
        time_str = time.strftime("%Y%m%d_%H%M%S")
        logging.basicConfig(filename=os.path.join(self.output_dir, f"log_{time_str}.txt"), level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')
        return logging.getLogger(__name__)

    def _select_group_tiles(self):
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
                                    self.output_dir_uncompress,'**' ,f"*{tile[0]}*{tile[1]}*.las"
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
        output_file = os.path.join(self.output_dir_proc, "strategy_grouped_tiles.png")
        plot_grouped_tiles(
            self.grid, self.groups, self.indices, output_file=output_file
        )

    def _process_group(self, input_file):
        import psutil, os, tracemalloc
        
        # Démarrer le traçage mémoire
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

        if type(input_file) == str:
            self.log.info(f"Processing single file {input_file}")
        try:
            data = read_lidar(input_file, **self.kwargs["kwargs"])
            if self.keep_variables is not None:
                data = data[0][:][self.keep_variables]
            files_ = [os.path.basename(file) for file in input_file]
            self.log.info(
                f"{self.counter}/{len(self.groups)} Saving {len(files_)} files  to {name_out}"
            )
            self.counter += 1
            self.corresponding_files[basename] = input_file
                    # import tempfile
                    # with tempfile.NamedTemporaryFile() as tmp:
                    #     # Créer un array mmap
                    #     fp = np.memmap(tmp.name, dtype=data.dtype, mode='w+', shape=data.shape)
                        
                    #     # Copier les données en petits morceaux
                    #     chunk_size = 100000
                    #     for i in range(0, len(data), chunk_size):
                    #         fp[i:i+chunk_size] = data[i:i+chunk_size]
                    #         del data[i:i+chunk_size]  # Libérer la mémoire du chunk
                    #         gc.collect()
                            
                    #     # Sauvegarder le fichier mmap
                    #     fp.flush()
                    #     np.save(name_out, fp)
                        
                    #     # Fermer et libérer
                    #     del fp
            np.save(name_out, data)
            del data
            gc.collect()
            mem_after = process.memory_info().rss / (1024 * 1024)
            self.log.info(f"Memory: Before={mem_before:.2f}MB, After={mem_after:.2f}MB, Diff={mem_after-mem_before:.2f}MB")
            return name_out
        except Exception as e:
            self.log.error(f"Error processing {input_file}: {e}")
            try: 
                self.log.info(f"Trying to process with filter sample 0.5")
                self.kwargs["kwargs"]["thin_radius"] = 0.5
                data = read_lidar(input_file, **self.kwargs["kwargs"])
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




    def uncompress_crop_tiles(self,
        input_file, lidar_list_tiles, area_of_interest
    ):
        """
        Uncompress LiDAR tiles (from .copc.laz to .laz) and crop them to the area of interest on the tiles of acquisition.

        .. warning:: The lines below of the `handlers.py` file of the pyforestscan package should be commented out:
        # if isinstance(polygon, MultiPolygon):
        #    polygon = list(polygon.geoms)[0]

        Parameters:
        ----------
        input_file : str
            Path to the input file (LiDAR tiles).
        lidar_list_tiles : str
            Path to the shapefile containing the list of LiDAR tiles (square polygons).
        area_of_interest : str
            Path to the shapefile containing the area of interest (polygon).
        output_dir : str
            Path to the output directory where the cropped tiles will be saved.
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
        poly_files = glob.glob(
            os.path.join(self.output_dir_uncompress, "temp_*poly*")
        )
        [os.remove(file) for file in poly_files]
        self.loader.finished = True

    def _check_existing_files(self):
        group2process = self.group_path.copy()
        self.log.info(f"Checking existing files {len(group2process)} groups")
        for group in group2process:
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

    # def extract_indicators(self):
    #     # Extract indicators from the processed tiles
    #     # This function should be implemented based on the specific indicators to be extracted

    #     if group:

    #     pass
    def process_worker(self, input_files):
        import os, gc, numpy as np
        from hashlib import sha256
        from pyforestscan.handlers import read_lidar
        
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
            
            # Libération explicite de la mémoire
            del data
            gc.collect()
            
            return (name_out, input_files, basename)
        except Exception as e:
            print(f"Error processing {basename}: {e}")
            try:
                # Essai avec un échantillonnage plus agressif
                modified_kwargs = self.kwargs.copy()
                modified_kwargs["kwargs"] = self.kwargs["kwargs"].copy()
                modified_kwargs["kwargs"]["thin_radius"] = 0.5
                
                data = read_lidar(input_files, **modified_kwargs["kwargs"])
                if self.keep_variables is not None:
                    data = data[0][:][self.keep_variables]
                
                np.save(name_out, data)
                
                # Libération explicite de la mémoire
                del data
                gc.collect()
                
                return (name_out, input_files, basename)
            except Exception as e:
                print(f"Failed with filter sample: {e}")
                return (None, None, None)
    # Define this at the top level of your module
    def _process_worker_mp(args):
        self, input_files = args
        return self.process_worker(input_files)


    def process_lidar_claude(self):
        from multiprocessing import Pool
        
        self.loader.show(
            "Processing tiles",
            finish_message="✅ Finished processing tiles",
            failed_message="❌ Failed processing tiles",
        )
        
        # Code existant pour l'initialisation
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
        
        # Déterminer le nombre optimal de processus
        n_processes = max(1, min(self.n_jobs, 2))  # Limité à 2 processus au maximum par sécurité
        self.log.info(f"Using {n_processes} processes for multiprocessing")
        
        # Utiliser un pool de processus
        self.files_proc = []
        
        # Traiter les groupes séquentiellement pour éviter les problèmes de multiprocessing
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
        
        # Sauvegarder la configuration
        with open(os.path.join(self.output_dir, "configuration.json"), "w") as f:
            json.dump(self.kwargs, f)
            timestamp = time.strftime("%Y%m%d%H%M%S")
            json.dump({"timestamp": timestamp}, f)
            json.dump({"keep_variables": self.keep_variables}, f)
            json.dump(self.corresponding_files, f)

        self.loader.finished = True

    def run_pipeline(self, lidar_list_tiles=None, area_of_interest=None):
        """
        Run the entire processing pipeline.
        """
        try:
            self.log.info("Starting Lidar processing pipeline")
            t = time.time()
            # self.uncompress_lidar(lidar_list_tiles, area_of_interest)
            self.log.info("#" * 50)
            self.log.info(f"Uncompressing and cropping tiles finished in {time.time() - t:.2f} seconds")
            t2 = time.time()
            self.process_lidar()
            # self.process_lidar_claude()
            self.log.info("#" * 50)
            self.log.info(f"Processing tiles finished in {time.time() - t2:.2f} seconds")
            # self.extract_indicators()
            self.log.info(f"Global processing time: {time.time() - t:.2f} seconds")
            self.log.info("Lidar processing pipeline finished")
        except Exception as e:
            print(f"Erreur fatale: {e}")
            # Enregistrer l'erreur dans le journal
            self.log.error(f"Erreur fatale: {e}", exc_info=True)

if __name__ == "__main__":

    # resort une liste de paquet/liste de 5ou 9 tuiles
    # extract indicators (CHM, FHD, other)

    # merge tiles


    LidarProcessor(
        path="/mnt/crea_camtrap/LIDAR/RAW/",
        # path="/mnt/crea_camtrap/LIDAR/RAW/MontBlanc/loriaz/",
        group=4,
        output_dir="/mnt/sentinel4To/LIDAR/",
        keep_variables=[
            "X",
            "Y",
            "Z",
            "Intensity",
            "Classification",
            "HeightAboveGround",
        ],
        n_jobs=3,
        kwargs={
            "srs": "EPSG:2154",
            "hag": True,
            "thin_radius": None,
            "outlier": [40, 1.5],
            "smrf": False,
            "only_vegetation": False,
        },
    ).run_pipeline(
        lidar_list_tiles="/mnt/crea_camtrap/LIDAR/SHP/TA_programme_lidar_HD_path.shp",
        area_of_interest="/mnt/crea_camtrap/LIDAR/SHP/Emprise_totale_3massifs_EPSG2154_V2.shp",
    )




# ###
