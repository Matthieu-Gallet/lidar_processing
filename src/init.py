"""
PyForestLidar - A Python package for processing LiDAR data for forestry applications.

This package provides tools for handling large LiDAR datasets, including:
- Reading and writing LiDAR data
- Cropping to areas of interest
- Processing tiles in groups
- Memory-efficient data handling
- Visualization of processing strategy
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .processor import LidarProcessor

__all__ = ["LidarProcessor"]
