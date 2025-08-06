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

    # Vérifier l'existence des masques
    use_masks = True
    if not os.path.exists(path_roc):
        print(f"⚠️  Masque de roche non trouvé: {path_roc}")
        use_masks = False

    if not os.path.exists(path_shadow):
        print(f"⚠️  Masque d'ombre non trouvé: {path_shadow}")
        use_masks = False

    if use_masks:
        print("✅ Utilisation des masques de roche et d'ombre")
    else:
        print("🔄 Traitement sans masques - utilisation de toute la tuile")

    save_path = merge_composite_images(files)

    if use_masks:
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
    else:
        # Utiliser directement le fichier fusionné sans masquage
        final_path = save_path
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
        prepare_classification(final_data, use_masks=use_masks)
    )

    # Déterminer l'ID de départ pour les clusters selon l'utilisation des masques
    start_cluster_id = 3 if use_masks else 2

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
            labels + start_cluster_id
        )  # Décaler les labels selon le mode utilisé

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
            labels + start_cluster_id
        )  # Décaler les labels selon le mode utilisé

    class_dict = sort_clusters_by_height(
        pixels_restants, labels, class_dict, start_cluster_id
    )

    # Définir des couleurs adaptées à la végétation depuis la config
    colors_dict = {}
    for key, color in config["classification_colors"].items():
        colors_dict[int(key)] = tuple(color)

    # Sauvegarder avec palette
    output_filename = config["output_files"]["classified_filename"].format(
        method=method
    )

    # Ajouter un suffixe pour distinguer le mode utilisé
    if not use_masks:
        base_name, ext = os.path.splitext(output_filename)
        output_filename = f"{base_name}_no_masks{ext}"

    print(f"💾 Sauvegarde du fichier de classification: {output_filename}")
    if use_masks:
        print(
            f"📊 Classes utilisées: Roche (0), Ombre (1), Arbustes/Forêts (2), Landes ({start_cluster_id}-{start_cluster_id+config['clustering_settings']['n_clusters']-1}), NoData (128)"
        )
    else:
        print(
            f"📊 Classes utilisées: Arbustes/Forêts (2), Landes ({start_cluster_id}-{start_cluster_id+config['clustering_settings']['n_clusters']-1}), NoData (128)"
        )

    save_classified_with_palette(
        array=classified_array,
        output_path=os.path.join(os.path.dirname(final_path), output_filename),
        reference_file=final_path,
        class_dict=class_dict,
        colors_dict=colors_dict,
        nodata=config["nodata_value"],
    )
