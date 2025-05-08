"""
Visualization utilities for Lidar data processing.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_grouped_tiles(grid, groups, indices, output_file=None):
    """
    Create a visualization of the grouped tiles.
    
    Parameters:
    ----------
    grid : numpy.ndarray
        2D array representing the grid of tiles.
    groups : list
        List of groups, where each group is a list of coordinates.
    indices : tuple
        Tuple containing dictionaries for mapping coordinates to indices.
    output_file : str, optional
        Path to save the output file. If None, the plot is displayed.
    """
    x_indices, y_indices = indices
    group_map = np.ones_like(grid, dtype=int) * -1  # white background (value -1)

    # Map groups to the grid
    for group_id, group in enumerate(groups, start=1):
        for x, y in group:
            j = x_indices[x]  # column
            i = y_indices[y]  # row
            group_map[i, j] = group_id

    # Create figure
    fig, ax = plt.subplots(figsize=(25, 25))
    
    # Set up colormap with shuffled colors for better differentiation
    cmap = plt.get_cmap("gnuplot", len(groups))
    cmap = cmap(np.arange(cmap.N))
    np.random.shuffle(cmap)
    cmap = plt.cm.colors.ListedColormap(cmap)
    
    # Plot tiles with colors based on their group
    ax.pcolormesh(
        np.where(grid == 1, group_map, np.nan) - 1,
        cmap=cmap,
        vmin=-1,
        vmax=len(groups),
        alpha=1,
        edgecolors="white",
        linewidth=0.01,
    )
    
    # Plot missing tiles in gray
    ax.matshow(
        np.where(grid == 1, np.nan, 0), cmap="gray_r", alpha=1
    )
    
    # Add group numbers to each tile
    for group_id, group in enumerate(groups, start=1):
        for x, y in group:
            j = x_indices[x]  # column
            i = y_indices[y]  # row
            ax.text(
                j + 0.5,
                i + 0.5,
                str(group_id),
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )
    
    # Set up axis labels and appearance
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
    
    # Save or display
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
    else:
        plt.show()
