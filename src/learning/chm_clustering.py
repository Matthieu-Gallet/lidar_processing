from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from src.learning.chm_processor import (
    merge_composite_images,
    apply_masks,
    prepare_classification,
    save_classified_with_palette,
    sort_clusters_by_height,
)
import rasterio, glob, os
import numpy as np


if __name__ == "__main__":
    path = "/home/mgallet/Téléchargements/results/*multi*.tif"
    files = glob.glob(path)

    path_roc = "/home/mgallet/Téléchargements/output/roc.tif"
    path_shadow = "/home/mgallet/Téléchargements/output/ombre.tif"

    save_path = merge_composite_images(files)
    apply_masks(
        save_path,
        path_roc,
        path_shadow,
        os.path.join(os.path.dirname(save_path), "masked_output.tif"),
    )

    final_path = os.path.join(os.path.dirname(path), "processed", "masked_output.tif")
    with rasterio.open(final_path) as src:
        final_data = src.read()
        final_data = np.moveaxis(
            final_data, 0, -1
        )  # Déplacer la première dimension à la fin
        final_data = final_data[:, :, [0, 2]]
        profile = src.profile.copy()
        print(f"Dimensions finales des données: {final_data.shape}")

    classified_array, class_dict, pixels_restants, cond_restants = (
        prepare_classification(final_data)
    )
    # Appliquer KMeans sur les pixels restants
    method = "gmm"  # Choisir la méthode de clustering: "kmeans" ou "dbscan"

    if method == "kmeans":
        kmeans = KMeans(n_clusters=4, random_state=42, max_iter=20000, tol=1e-6)
        kmeans.fit_transform(pixels_restants)
        # Ajouter les résultats de KMeans à la classification
        labels = kmeans.labels_
        classified_array[cond_restants] = (
            labels + 3
        )  # Décaler les labels pour éviter les conflits avec les classes existantes

    elif method == "gmm":
        gmm = GaussianMixture(
            n_components=4,
            random_state=42,
            max_iter=20000,
            tol=1e-6,
            init_params="k-means++",
            covariance_type="full",
        )
        labels = gmm.fit_predict(pixels_restants)
        classified_array[cond_restants] = (
            labels + 3
        )  # Décaler les labels pour éviter les conflits avec les classes existantes

    class_dict = sort_clusters_by_height(pixels_restants, labels, class_dict)
    class_dict

    # Définir des couleurs adaptées à la végétation
    colors_dict = {
        0: (139, 137, 137),  # Gris pierre pour roche
        1: (47, 47, 47),  # Gris très foncé pour ombre
        2: (34, 139, 34),  # Vert forêt foncé pour arbustes et forêts
        3: (255, 218, 185),  # Pêche clair pour lande rase
        4: (205, 133, 63),  # Marron clair pour lande dense
        5: (160, 82, 45),  # Marron saddle pour lande moyenne
        6: (72, 30, 0),  # Marron chocolat pour lande arbustive
        7: (0, 100, 0),  # Vert foncé pour lande très dense
        128: (0, 0, 0),  # Noir pour NoData (sera transparent)
    }

    # Sauvegarder avec palette
    save_classified_with_palette(
        array=classified_array,
        output_path=os.path.join(
            os.path.dirname(final_path), f"classified_output_{method}_full.tif"
        ),
        reference_file=final_path,
        class_dict=class_dict,
        colors_dict=colors_dict,
        nodata=128,
    )
