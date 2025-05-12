"""
Grid management functions for Lidar processing.
"""

import numpy as np
import warnings
from collections import deque
import os, glob
from .visualization import plot_grouped_tiles


def construct_matrix_coordinates(list_tiles, original=True):
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
        if original:
            coord = tile.split("_")[2:4]
        else:
            coord = tile.split("_")[3:5]  # Using indices 3:5 from your updated function
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
        Tuple containing two dictionaries for mapping indices to coordinates.
    indices : tuple
        Tuple containing two dictionaries for mapping coordinates to indices.
    """
    # Normalize coordinates to map to a compact grid
    xs, ys = zip(*coords)
    xs = np.array(xs, dtype=int)
    ys = np.array(ys, dtype=int)

    x_x = np.arange(xs.min(), xs.max() + 1)
    y_y = np.arange(ys.min(), ys.max() + 1)

    x_indices = {i: x for i, x in zip(x_x, np.arange(0, len(x_x)))}
    y_indices = {
        # i: y for i, y in zip(np.arange(ys.min(), ys.max() + 1), np.arange(0, len(ys)))
        i: y
        for i, y in zip(y_y, np.arange(0, len(y_y)))
    }

    # Inverse mapping to retrieve coords later
    index_to_x = {i: x for x, i in x_indices.items()}
    index_to_y = {i: y for y, i in y_indices.items()}

    # Create the logical grid
    height = len(y_indices)
    width = len(x_indices)
    grid = np.zeros((height, width), dtype=int)

    for x, y in coords:
        i = y_indices[y]
        j = x_indices[x]
        if grid[i, j] == 0:  # Only increment if the cell is not already filled
            grid[i, j] = 1  # 1 = tile present
        else:
            warnings.warn(f"Tile already present at position ({i}, {j}): {x}, {y}")

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
    return (index_to_x[j], index_to_y[i])  # note the order: numpy = (row, col)


def get_adjacent(i, j, mask):
    """
    Get adjacent cells in a grid that match a mask.

    Parameters:
    ----------
    i : int
        Row index in the grid.
    j : int
        Column index in the grid.
    mask : numpy.ndarray
        Boolean mask indicating valid cells.

    Returns:
    -------
    generator
        Generator yielding (row, col) tuples for adjacent valid cells.
    """
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
    indexes : tuple
        Tuple containing two dictionaries for mapping indices to coordinates.
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

    # Special case: group all tiles
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

    # Group tiles into components of size <= n
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

    return groups


def select_group_tiles(
    tiles2coord,
    output_dir_uncompress,
    tilesingroup=4,
    original=True,
    plot_file=None,
    log=None,
):
    """
    Select and group tiles based on their coordinates.
    """

    coords = construct_matrix_coordinates(tiles2coord, original=original)
    grid, indexes, indices = construct_grid(coords)
    groups = group_adjacent_tiles_by_n(grid, indexes, n=tilesingroup)
    group_path = _maps_group2tilespath(groups, output_dir_uncompress, log)
    if plot_file is not None:
        plot_grouped_tiles(grid, groups, indices, output_file=plot_file)
    return group_path


def _maps_group2tilespath(groups, output_dir_uncompress, log):
    """
    Create a mapping of group IDs to tile paths.
    """
    all_tiles = []
    for group in groups:
        group_paths = []
        for tile in group:
            try:
                matching_files = glob.glob(
                    os.path.join(
                        output_dir_uncompress,
                        "**",
                        f"*{tile[0]}*{tile[1]}*.las",
                    ),
                    recursive=True,
                )
                if matching_files:
                    group_paths.append(matching_files[0])
            except Exception as e:
                log.error(f"Error mapping tile {tile}: {e}")
                group_paths.append(None)
        group_paths = [path for path in group_paths if path is not None]
        all_tiles.append(group_paths)
    return all_tiles
