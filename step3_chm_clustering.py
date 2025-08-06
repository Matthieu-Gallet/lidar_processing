from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from src.learning.chm_processor import (
    merge_composite_images,
    apply_masks,
    prepare_classification,
    save_classified_with_palette,
    sort_clusters_by_height,
)
import rasterio, glob, os, json
import numpy as np


if __name__ == "__main__":
    # Load configuration
    with open("config/step3_config.json", "r") as f:
        config = json.load(f)

    # Extract paths from config
    path = config["paths"]["input_path"]
    files = glob.glob(path)

    path_roc = config["paths"]["roc_mask_path"]
    path_shadow = config["paths"]["shadow_mask_path"]

    save_path = merge_composite_images(files)
    masked_output_path = os.path.join(
        os.path.dirname(save_path), config["output_files"]["masked_filename"]
    )
    apply_masks(
        save_path,
        path_roc,
        path_shadow,
        masked_output_path,
    )

    final_path = os.path.join(
        os.path.dirname(path),
        config["paths"]["output_subdir"],
        config["output_files"]["masked_filename"],
    )
    with rasterio.open(final_path) as src:
        final_data = src.read()
        final_data = np.moveaxis(
            final_data,
            config["data_settings"]["move_axis_from"],
            config["data_settings"]["move_axis_to"],
        )  # Déplacer la première dimension à la fin
        final_data = final_data[:, :, config["data_settings"]["channels_to_use"]]
        profile = src.profile.copy()
        print(f"Dimensions finales des données: {final_data.shape}")

    classified_array, class_dict, pixels_restants, cond_restants = (
        prepare_classification(final_data)
    )
    # Appliquer clustering sur les pixels restants
    method = config["clustering_settings"][
        "method"
    ]  # Utiliser la méthode de clustering du config

    if method == "kmeans":
        kmeans = KMeans(
            n_clusters=config["clustering_settings"]["n_clusters"],
            random_state=config["clustering_settings"]["random_state"],
            max_iter=config["clustering_settings"]["max_iter"],
            tol=config["clustering_settings"]["tolerance"],
        )
        kmeans.fit_transform(pixels_restants)
        # Ajouter les résultats de KMeans à la classification
        labels = kmeans.labels_
        classified_array[cond_restants] = (
            labels + 3
        )  # Décaler les labels pour éviter les conflits avec les classes existantes

    elif method == "gmm":
        gmm = GaussianMixture(
            n_components=config["clustering_settings"]["n_clusters"],
            random_state=config["clustering_settings"]["random_state"],
            max_iter=config["clustering_settings"]["max_iter"],
            tol=config["clustering_settings"]["tolerance"],
            init_params=config["clustering_settings"]["gmm_settings"]["init_params"],
            covariance_type=config["clustering_settings"]["gmm_settings"][
                "covariance_type"
            ],
        )
        labels = gmm.fit_predict(pixels_restants)
        classified_array[cond_restants] = (
            labels + 3
        )  # Décaler les labels pour éviter les conflits avec les classes existantes

    class_dict = sort_clusters_by_height(pixels_restants, labels, class_dict)

    # Définir des couleurs adaptées à la végétation depuis la config
    colors_dict = {}
    for key, color in config["classification_colors"].items():
        colors_dict[int(key)] = tuple(color)

    # Sauvegarder avec palette
    output_filename = config["output_files"]["classified_filename"].format(
        method=method
    )
    save_classified_with_palette(
        array=classified_array,
        output_path=os.path.join(os.path.dirname(final_path), output_filename),
        reference_file=final_path,
        class_dict=class_dict,
        colors_dict=colors_dict,
        nodata=config["nodata_value"],
    )
