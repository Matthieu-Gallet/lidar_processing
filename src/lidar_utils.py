"""
PDAL processing utilities for LiDAR data.
"""

import os
import json
import tempfile
import subprocess
import logging

import gc
import numpy as np
import pdal
import os
import json
import logging
import tempfile
import subprocess
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm


def create_pdal_pipeline_phase1(input_file, output_file):
    """
    Crée la première phase du pipeline PDAL pour traiter chaque tuile individuellement.
    Cette phase réduit la taille des données et nettoie les points aberrants.

    Parameters:
    ----------
    input_file : str
        Chemin du fichier LAS/LAZ d'entrée.
    output_file : str
        Chemin du fichier de sortie pour la phase 1.

    Returns:
    -------
    dict
        Configuration du pipeline PDAL phase 1.
    """
    pipeline = {
        "pipeline": [
            {
                "type": "readers.las",
                "filename": input_file,
                "spatialreference": "EPSG:2154",
            },
            {"type": "filters.elm"},
            {
                "type": "filters.outlier",
                "method": "statistical",
                "mean_k": 40,
                "multiplier": 1.5,
            },
            {
                "type": "filters.csf",
                "resolution": 0.75,
                "rigidness": 1,
                "iterations": 800,
                "threshold": 0.2,
            },
            {
                "type": "filters.voxeldownsize",
                "cell": 0.02,  # 2cm en X,Y et Z
            },
            {
                "type": "writers.las",
                "filename": output_file,
                "compression": "false",
                "minor_version": "4",
                "forward": "all",
            },
        ]
    }

    return pipeline


def create_pdal_pipeline_phase2(input_files, output_file):
    """
    Crée la deuxième phase du pipeline PDAL pour fusionner les tuiles prétraitées
    et appliquer les traitements qui nécessitent une vue globale.

    Parameters:
    ----------
    input_files : list
        Liste des fichiers LAS/LAZ prétraités par la phase 1.
    output_file : str
        Chemin du fichier de sortie final.
    pdal_pipeline_template : list, optional
        Modèle pour les étapes du pipeline PDAL (phase 2 uniquement).
    quality_level : dict, optional
        Paramètres de niveau de qualité à appliquer.

    Returns:
    -------
    dict
        Configuration du pipeline PDAL phase 2.
    """
    pipeline = {"pipeline": []}

    # Ajouter la première tuile avec readers.las
    pipeline["pipeline"].append(
        {
            "type": "readers.las",
            "filename": input_files[0],
            "spatialreference": "EPSG:2154",
        }
    )

    # Ajouter les autres tuiles avec readers.las
    for file in input_files[1:]:
        pipeline["pipeline"].append(
            {"type": "readers.las", "filename": file, "spatialreference": "EPSG:2154"}
        )

    # Fusionner les tuiles
    pipeline["pipeline"].append({"type": "filters.merge"})

    # Utiliser hag_nn au lieu de hag_delaunay pour de meilleures performances
    pipeline["pipeline"].append({"type": "filters.hag_nn", "count": 80})

    # Filtrer par hauteur
    pipeline["pipeline"].append(
        {"type": "filters.range", "limits": "HeightAboveGround[0:35]"}
    )

    # Filtrer par classification
    pipeline["pipeline"].append(
        {
            "type": "filters.expression",
            "expression": "((Classification >= 2) && (Classification <= 5))",
        }
    )

    # Ajout de l'étape d'écriture
    pipeline["pipeline"].append(
        {
            "type": "writers.las",
            "filename": output_file,
            "compression": "false",
            "minor_version": "4",
            "forward": "all",
            "extra_dims": "all",
        }
    )

    return pipeline


def process_single_tile(input_file, temp_dir, log=None):
    """
    Traite une seule tuile LiDAR avec la phase 1 du pipeline.

    Parameters:
    ----------
    input_file : str
        Chemin du fichier d'entrée.
    temp_dir : str
        Répertoire temporaire pour stocker le résultat.
    log : logging.Logger, optional
        Instance de logger.

    Returns:
    -------
    str
        Chemin du fichier de sortie si réussi, None sinon.
    """
    current_log = log or logging.getLogger(f"pdal_runner_{os.getpid()}")

    # Créer le nom du fichier de sortie
    basename = os.path.basename(input_file)
    output_file = os.path.join(temp_dir, f"phase1_{basename}")

    # Créer le pipeline phase 1
    pipeline_config = create_pdal_pipeline_phase1(input_file, output_file)

    # Exécuter le pipeline
    success = run_pdal_pipeline(pipeline_config, current_log)

    if success:
        return output_file
    return None


def run_pdal_pipeline(pipeline_config, log=None):
    """
    Exécute un pipeline PDAL.

    Parameters:
    ----------
    pipeline_config : dict
        Configuration du pipeline PDAL.
    log : logging.Logger, optional
        Instance de logger.

    Returns:
    -------
    bool
        True si réussi, False sinon.
    """
    current_log = log or logging.getLogger(f"pdal_runner_{os.getpid()}")

    # Créer un fichier JSON temporaire pour le pipeline
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
        json.dump(pipeline_config, f)
        pipeline_file = f.name

    try:
        # Exécuter le pipeline PDAL
        current_log.info(f"Exécution du pipeline PDAL: {pipeline_file}")

        # Exécuter PDAL en tant que sous-processus
        process = subprocess.run(
            ["pdal", "pipeline", pipeline_file],
            capture_output=True,
            text=True,
            check=False,
        )

        # Vérifier les erreurs
        if process.returncode != 0:
            current_log.error(f"Échec du pipeline PDAL (code {process.returncode}):")
            current_log.error(process.stderr)
            return False

        # Journaliser stdout si verbeux
        if process.stdout and len(process.stdout) > 0:
            current_log.debug(f"Sortie PDAL: {process.stdout}")

        return True

    except Exception as e:
        current_log.error(f"Erreur lors de l'exécution du pipeline PDAL: {e}")
        return False
    finally:
        # Nettoyer le fichier temporaire
        try:
            os.unlink(pipeline_file)
        except:
            pass


def process_tiles_two_phase(
    input_files,
    output_file,
    n_jobs=None,
    log=None,
):
    """
    Traite les tuiles LiDAR en deux phases:
    1) Traitement parallèle de chaque tuile pour réduire la taille
    2) Fusion et traitement global des tuiles prétraitées

    Parameters:
    ----------
    input_files : list
        Liste des fichiers LAS/LAZ d'entrée.
    output_file : str
        Chemin du fichier de sortie final.
    n_jobs : int, optional
        Nombre de processus parallèles (par défaut: nombre de CPU - 2).
    log : logging.Logger, optional
        Instance de logger.

    Returns:
    -------
    bool
        True si réussi, False sinon.
    """
    import multiprocessing

    current_log = log or logging.getLogger("pdal_processor")

    # Définir le nombre de processus parallèles
    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 2)

    # Créer un répertoire temporaire pour les fichiers intermédiaires
    with tempfile.TemporaryDirectory() as phase1_temp_dir:
        current_log.info(
            f"Traitement de {len(input_files)} tuiles en {n_jobs} processus parallèles"
        )

        # Phase 1: Traitement parallèle des tuiles individuelles
        processed_files = []

        # Utiliser ProcessPoolExecutor pour la parallélisation
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Créer une fonction partielle avec les arguments fixes
            process_func = partial(
                process_single_tile, temp_dir=phase1_temp_dir, log=current_log
            )

            # Soumettre les tâches et collecter les résultats
            for result in executor.map(process_func, input_files):
                if result:
                    processed_files.append(result)
                else:
                    current_log.error("Échec du traitement d'une tuile dans la phase 1")
                    return False

        current_log.info(
            f"Phase 1 terminée. {len(processed_files)} tuiles traitées avec succès."
        )

        if not processed_files:
            current_log.error("Aucune tuile traitée avec succès dans la phase 1")
            return False

        # Phase 2: Fusion et traitement global
        current_log.info("Démarrage de la phase 2: fusion et traitement global")

        # Créer le pipeline de phase 2
        pipeline_config = create_pdal_pipeline_phase2(processed_files, output_file)

        # Exécuter le pipeline de phase 2
        if not run_pdal_pipeline(pipeline_config, current_log):
            current_log.error("Échec de la phase 2 du pipeline PDAL")
            return False

        current_log.info(
            f"Traitement en deux phases terminé avec succès. Résultat: {output_file}"
        )
        return True


def create_pdal_pipeline(
    input_files, output_file, pdal_pipeline_template=None, quality_level=None
):
    """
    Create a PDAL pipeline configuration from template.

    Parameters:
    ----------
    input_files : list
        List of input LAS/LAZ files.
    output_file : str
        Output file path.
    pdal_pipeline_template : list, optional
        Template for PDAL pipeline stages.
    quality_level : dict, optional
        Quality level parameters to apply.

    Returns:
    -------
    dict
        PDAL pipeline configuration.
    """
    # Start with base pipeline
    pipeline = {"pipeline": []}

    # Add reader stage for each input file or use a merged reader
    if len(input_files) == 1:
        pipeline["pipeline"].append({"type": "readers.las", "filename": input_files[0]})
    else:
        pipeline["pipeline"] = [
            {
                "type": "readers.las",
                "filename": input_files[0],
                "spatialreference": "EPSG:2154",
            }
        ]

        # Ajouter les autres fichiers avec readers.las
        for file in input_files[1:]:
            pipeline["pipeline"].append(
                {
                    "type": "readers.las",
                    "filename": file,
                    "spatialreference": "EPSG:2154",
                }
            )

        # Ajouter une étape de fusion si nécessaire
        pipeline["pipeline"].append({"type": "filters.merge"})
    # Deep copy the pipeline to avoid modifying the template
    import copy

    pipeline_stages = copy.deepcopy(pdal_pipeline_template or [])

    # Update the thin_radius in the filters.sample stage if quality level is specified
    if quality_level and quality_level["thin_radius"] is not None:
        for stage in pipeline_stages["pipeline"]:
            if stage["type"] == "filters.sample":
                stage["radius"] = quality_level["thin_radius"]
                break

    # Add all pipeline stages
    pipeline["pipeline"].extend(pipeline_stages["pipeline"])

    # Add writer stage
    pipeline["pipeline"].append(
        {
            "type": "writers.las",
            "filename": output_file,
            "compression": "false",
            "minor_version": "4",
            "forward": "all",
        }
    )

    return pipeline


def construct_numpy_dtype(keep_variables, data_type, data_shape):
    """
    Construct a numpy dtype based on the fields of interest.
    """
    # Create a list of tuples for the new dtype
    new_dtype = []
    data_type_dr = np.array(data_type.descr)
    for var in keep_variables:
        if var in data_type_dr:
            # Get the index of the variable
            index = np.where(data_type_dr == var)[0][0]
            # Append the variable and its type to the new dtype
            new_dtype.append(tuple(data_type_dr[index].tolist()))
    # Create the new numpy dtype
    new_data = np.empty(data_shape, dtype=new_dtype)
    return new_data


def convert_las_to_numpy(las_file):
    """
    Convert a LAS file to a numpy array.
    """
    pipeline = pdal.Pipeline(
        json.dumps(
            {
                "pipeline": [
                    {
                        "type": "readers.las",
                        "filename": las_file,
                        "spatialreference": "EPSG:2154",
                    }
                ]
            }
        )
    )
    pipeline.execute()
    return pipeline.arrays[0]


def selected_las2npy(las_file, output_file, keep_variables):
    """
    Extract specific fields from a LAS file and save to a numpy file.
    """
    # Read the LAS file
    data = convert_las_to_numpy(las_file)
    selected_data = construct_numpy_dtype(keep_variables, data.dtype, data.shape[0])
    for var in selected_data.dtype.names:
        selected_data[var] = data[var]
    # Save to a numpy file
    np.save(output_file, selected_data)
    del data
    del selected_data
    gc.collect()
    return 1


# Chemin vers votre fichier LAS
def las2npy_chunk(las_file, output_file, keep_variables, chunk_size=1000000):
    # Taille du chunk (ajustez selon votre mémoire disponible)
    chunk_size = 1000000  # par exemple 1 million de points par chunk

    # Obtenir un échantillon pour connaître la structure des données
    sample_pipeline = pdal.Pipeline(
        json.dumps(
            {
                "pipeline": [
                    {
                        "type": "readers.las",
                        "filename": las_file,
                        "spatialreference": "EPSG:2154",
                        "count": 1,  # Juste un point pour connaître la structure
                    }
                ]
            }
        )
    )
    sample_pipeline.execute()
    point_count = sample_pipeline.metadata["metadata"]["readers.las"]["count"]
    sample_data = sample_pipeline.arrays[0]

    # Initialiser all_data avec notre fonction
    all_data = construct_numpy_dtype(
        keep_variables=keep_variables,
        data_type=sample_data.dtype,
        data_shape=point_count,
    )

    # Traiter par morceaux avec tqdm pour la progression
    processed_points = 0
    for start in tqdm(range(0, point_count, chunk_size), desc="Traitement des chunks"):
        count = min(chunk_size, point_count - start)

        chunk_pipeline = pdal.Pipeline(
            json.dumps(
                {
                    "pipeline": [
                        {
                            "type": "readers.las",
                            "filename": las_file,
                            "spatialreference": "EPSG:2154",
                            "start": start,
                            "count": count,
                        }
                    ]
                }
            )
        )

        chunk_pipeline.execute()
        chunk_data = chunk_pipeline.arrays[0]

        # Copier chaque dimension d'intérêt dans all_data
        for dim in keep_variables:
            if dim in chunk_data.dtype.names:
                all_data[dim][processed_points : processed_points + count] = chunk_data[
                    dim
                ]

        processed_points += count

        # Libérer la mémoire
        del chunk_data
        del chunk_pipeline

    # Sauvegarder en format NPY
    np.save(output_file, all_data)
    print(f"Données sauvegardées, forme finale: {all_data.shape}")
