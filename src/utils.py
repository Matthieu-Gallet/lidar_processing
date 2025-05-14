"""
Utility functions for Lidar processing.
"""

import os
import geopandas as gpd
import time
import logging
from hashlib import sha256


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
    # Create formatter
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
    
    # Initialize the logger
    time_str = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"log_{time_str}.txt")
    
    # Créer un file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Configurer le root logger pour que tous les loggers en héritent
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Supprimer les handlers existants pour éviter les doublons
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Ajouter notre file handler au root logger
    root_logger.addHandler(file_handler)
    
    # Créer le logger spécifique
    logger = logging.getLogger(__name__)
    
    logger.info("Logger initialized successfully")
    return logger



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
