import os


import rasterio
import numpy as np
from rasterio.merge import merge
from rasterio.crs import CRS
from rasterio.enums import ColorInterp
from rasterio.windows import from_bounds


def merge_composite_images(files):

    # Étape 1: Traiter chaque fichier
    processed_files = []
    for file_path in files:
        try:
            with rasterio.open(file_path, "r") as src:
                # Lire toutes les bandes
                data = src.read()
                profile = src.profile.copy()

                # Identifier les pixels où les 3 canaux sont égaux à 0
                if data.shape[0] >= 3:  # Au moins 3 bandes
                    mask = (data[0] == 0) & (data[1] == 0) & (data[2] == 0)
                    # Appliquer -999 à tous les canaux pour ces pixels
                    data[:, mask] = -999

                # Mettre à jour le profil
                profile.update(
                    {"nodata": -999, "crs": CRS.from_epsg(2154), "compress": "lzw"}
                )

                # Sauvegarder le fichier traité
                output_path = os.path.join(
                    os.path.dirname(file_path), "processed", os.path.basename(file_path)
                )
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(data)

                processed_files.append(output_path)
                print(f"Traité: {file_path}")

        except Exception as e:
            print(f"Erreur avec {file_path}: {e}")

    # Étape 2: Fusionner tous les fichiers traités
    if processed_files:
        try:
            # Ouvrir les fichiers pour la fusion
            src_files_to_mosaic = []
            for fp in processed_files:
                src = rasterio.open(fp)
                src_files_to_mosaic.append(src)

            # Créer la mosaïque
            mosaic, out_trans = merge(src_files_to_mosaic, nodata=-999)

            # Obtenir le profil de référence
            out_profile = src_files_to_mosaic[0].profile.copy()

            # Fermer tous les fichiers sources
            for src in src_files_to_mosaic:
                src.close()

            # Mettre à jour le profil de sortie
            out_profile.update(
                {
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans,
                    "nodata": -999,
                    "crs": CRS.from_epsg(2154),
                }
            )
            save_path = os.path.join(
                os.path.dirname(processed_files[0]), "merged_output.tif"
            )
            # Sauvegarder le fichier fusionné
            with rasterio.open(save_path, "w", **out_profile) as dest:
                dest.write(mosaic)

            print(f"Fusion terminée: {save_path}")
        except Exception as e:
            print(f"Erreur lors de la fusion: {e}")

    else:
        print("Aucun fichier traité avec succès")
    return save_path


def apply_masks(save_path, path_roc, path_shadow, output_path="masked_output.tif"):
    """
    Applique les masques de roche et d'ombre au fichier fusionné en croppant sur l'intersection

    Args:
        save_path: chemin du fichier fusionné
        path_roc: chemin du masque de roche (roc = 1)
        path_shadow: chemin du masque d'ombre (ombre = 0)
        output_path: chemin de sortie
    """

    # Ouvrir tous les fichiers pour obtenir leurs bounds
    with rasterio.open(save_path) as main_src, rasterio.open(
        path_roc
    ) as roc_src, rasterio.open(path_shadow) as shadow_src:

        # Calculer l'intersection des bounds
        main_bounds = main_src.bounds
        roc_bounds = roc_src.bounds
        shadow_bounds = shadow_src.bounds

        # Intersection des bounds
        left = max(main_bounds.left, roc_bounds.left, shadow_bounds.left)
        bottom = max(main_bounds.bottom, roc_bounds.bottom, shadow_bounds.bottom)
        right = min(main_bounds.right, roc_bounds.right, shadow_bounds.right)
        top = min(main_bounds.top, roc_bounds.top, shadow_bounds.top)

        intersection_bounds = (left, bottom, right, top)

        # Calculer les fenêtres pour chaque fichier
        main_window = from_bounds(*intersection_bounds, main_src.transform)
        roc_window = from_bounds(*intersection_bounds, roc_src.transform)
        shadow_window = from_bounds(*intersection_bounds, shadow_src.transform)

        # Lire les données croppées
        main_data = main_src.read(window=main_window)
        roc_mask = roc_src.read(1, window=roc_window)
        shadow_mask = shadow_src.read(1, window=shadow_window)

        # Vérifier que toutes les données ont la même taille
        print(f"Taille fichier principal: {main_data.shape}")
        print(f"Taille masque roche: {roc_mask.shape}")
        print(f"Taille masque ombre: {shadow_mask.shape}")
        # ajuste les dimensions des masques pour correspondre à celles du fichier principal
        if (
            main_data.shape[1:] != roc_mask.shape
            or main_data.shape[1:] != shadow_mask.shape
        ):
            roc_mask = roc_mask[
                : min(roc_mask.shape[0], main_data.shape[1]),
                : min(roc_mask.shape[1], main_data.shape[2]),
            ]
            shadow_mask = shadow_mask[
                : min(shadow_mask.shape[0], main_data.shape[1]),
                : min(shadow_mask.shape[1], main_data.shape[2]),
            ]

        print(f"Taille ajustée masque roche: {roc_mask.shape}")
        print(f"Taille ajustée masque ombre: {shadow_mask.shape}")
        # Créer une copie des données principales
        masked_data = main_data.copy()
        # Appliquer le masque de roche (roc = 1, valeur -99)
        roc_pixels = roc_mask == 1
        for band in range(masked_data.shape[0]):
            masked_data[band, roc_pixels] = -99

        # Appliquer le masque d'ombre (ombre = 0, valeur -89)
        shadow_pixels = shadow_mask == 0
        for band in range(masked_data.shape[0]):
            masked_data[band, shadow_pixels] = -89

        # Calculer le nouveau transform pour la zone croppée
        new_transform = main_src.window_transform(main_window)

        # Mettre à jour le profil
        profile = main_src.profile.copy()
        profile.update(
            {
                "height": masked_data.shape[1],
                "width": masked_data.shape[2],
                "transform": new_transform,
                "nodata": -999,
                "compress": "lzw",
            }
        )

        # Sauvegarder le fichier masqué
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(masked_data)

        print(f"Fichier masqué sauvegardé: {output_path}")
        print(f"Pixels de roche masqués: {np.sum(roc_pixels)}")
        print(f"Pixels d'ombre masqués: {np.sum(shadow_pixels)}")
        print(f"Bounds d'intersection: {intersection_bounds}")


def create_aux_xml_file(tiff_path, class_dict):
    """
    Crée un fichier .aux.xml pour forcer QGIS à afficher les noms des classes
    """
    aux_path = tiff_path + ".aux.xml"

    xml_content = """<PAMDataset>
  <PAMRasterBand band="1">
    <CategoryNames>"""

    # Créer la liste des catégories (jusqu'à la valeur max)
    max_value = max(class_dict.keys()) if class_dict else 0
    categories = [""] * (max_value + 1)

    for value, description in class_dict.items():
        if 0 <= value <= max_value:
            categories[value] = description

    # Ajouter chaque catégorie au XML
    for category in categories:
        xml_content += f"""
      <Category>{category}</Category>"""

    xml_content += """
    </CategoryNames>
    <GDALRasterAttributeTable>"""

    # Ajouter la table d'attributs raster
    for value, description in sorted(class_dict.items()):
        xml_content += f"""
      <Row index="{value}">
        <F>VALUE</F>
        <F>{value}</F>
        <F>CLASS</F>
        <F>{description}</F>
      </Row>"""

    xml_content += """
    </GDALRasterAttributeTable>
  </PAMRasterBand>
</PAMDataset>"""

    # Écrire le fichier XML
    with open(aux_path, "w", encoding="utf-8") as f:
        f.write(xml_content)


def save_classified_with_palette(
    array,
    output_path,
    reference_file,
    class_dict,
    colors_dict=None,
    transform=None,
    nodata=None,
):
    """
    Sauvegarde un array classifié avec palette de couleurs et noms des classes
    pour une ouverture automatique optimale dans QGIS

    Args:
        array: array à sauvegarder (2D)
        output_path: chemin de sortie
        reference_file: fichier de référence pour le profil
        class_dict: dictionnaire {valeur_pixel: nom_classe}
        colors_dict: dictionnaire {valeur_pixel: (R,G,B)} optionnel
        transform: transform spécifique (optionnel)
        nodata: valeur nodata (optionnel)
    """

    # Générer des couleurs aléatoirement si non fournies
    if colors_dict is None:
        colors_dict = generate_random_colors(class_dict.keys())

    with rasterio.open(reference_file) as ref:
        profile = ref.profile.copy()

        # Utiliser le transform fourni ou celui du fichier de référence
        if transform is not None:
            profile["transform"] = transform

        # S'assurer que l'array est en 2D
        if len(array.shape) != 2:
            raise ValueError("L'array doit être en 2D pour la classification")

        # Mettre à jour le profil
        profile.update(
            {
                "count": 1,
                "height": array.shape[0],
                "width": array.shape[1],
                "dtype": "uint8",  # Important pour les classes
                "compress": "lzw",
            }
        )

        # Définir nodata si fourni
        if nodata is not None:
            profile["nodata"] = nodata

        # Convertir l'array en uint8 si nécessaire
        if array.dtype != np.uint8:
            array = array.astype(np.uint8)

        with rasterio.open(output_path, "w", **profile) as dst:
            # Écrire les données
            dst.write(array, 1)

            # Créer la table de couleurs (palette)
            colormap = {}
            for value in class_dict.keys():
                if value in colors_dict:
                    r, g, b = colors_dict[value]
                    colormap[value] = (r, g, b, 255)  # RGBA

            # Appliquer la table de couleurs
            dst.write_colormap(1, colormap)

            # Définir l'interprétation des couleurs comme palette
            dst.colorinterp = [ColorInterp.palette]
            # Méthode alternative pour les catégories (compatible toutes versions)
            try:
                # Tenter la méthode moderne
                categories = [""] * 256  # Initialiser avec des chaînes vides
                for value, description in class_dict.items():
                    if 0 <= value < 256:
                        categories[value] = description
                dst.set_category_names(1, categories)
            except AttributeError:
                # Méthode alternative avec les tags RAT (Raster Attribute Table)
                rat_metadata = {}
                for value, description in class_dict.items():
                    rat_metadata[f"CATEGORY_{value}"] = description
                    rat_metadata[f"CLASS_NAME_{value}"] = description
                    # Ajouter des métadonnées spécifiques pour QGIS
                    rat_metadata[f"GDAL_RAT_VALUE_{value}"] = str(value)
                    rat_metadata[f"GDAL_RAT_DESCRIPTION_{value}"] = description
                dst.update_tags(1, **rat_metadata)

                # Créer un fichier .aux.xml pour forcer l'affichage des noms
                create_aux_xml_file(output_path, class_dict)

            # Ajouter des métadonnées supplémentaires
            metadata = {"LAYER_TYPE": "classifications", "STATISTICS_APPROXIMATE": "NO"}

            # Ajouter les descriptions dans les tags aussi (compatibilité)
            for value, description in class_dict.items():
                metadata[f"Class_{value}"] = description

            dst.update_tags(**metadata)


def generate_random_colors(class_values, seed=42):
    """
    Génère des couleurs aléatoirement pour les classes

    Args:
        class_values: liste des valeurs de classes
        seed: graine pour la reproductibilité

    Returns:
        dict: {valeur: (R,G,B)}
    """
    np.random.seed(seed)
    colors = {}

    # Couleurs prédéfinies pour les premières classes
    predefined_colors = [
        (255, 0, 0),  # Rouge
        (0, 255, 0),  # Vert
        (0, 0, 255),  # Bleu
        (255, 255, 0),  # Jaune
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cyan
        (255, 165, 0),  # Orange
        (128, 0, 128),  # Violet
        (255, 192, 203),  # Rose
        (165, 42, 42),  # Marron
    ]

    class_list = sorted(list(class_values))

    for i, value in enumerate(class_list):
        if i < len(predefined_colors):
            colors[value] = predefined_colors[i]
        else:
            # Générer des couleurs aléatoirement pour les classes supplémentaires
            colors[value] = (
                np.random.randint(0, 256),
                np.random.randint(0, 256),
                np.random.randint(0, 256),
            )

    return colors


def prepare_classification(final_data, use_masks=True):
    """
    Prépare les données pour la classification en remplaçant les valeurs spécifiques
    et en créant un tableau de classification.

    Args:
        final_data: tableau numpy 3D des données finales
        use_masks: booléen indiquant si les masques ont été appliqués

    Returns:
        classified_array: tableau 2D de classification
        class_dict: dictionnaire des classes
        pixels_restants: pixels à classifier
        cond_restants: condition pour les pixels restants
    """
    # Assurez-vous que final_data est un tableau numpy
    if not isinstance(final_data, np.ndarray):
        raise ValueError("final_data doit être un tableau numpy")

    if final_data.ndim != 3:
        raise ValueError(
            "final_data doit être un tableau 3D (hauteur, largeur, canaux)"
        )

    classified_array = np.zeros(
        final_data.shape[:2], dtype=np.uint8
    )  # Créer un tableau de classification vide

    if use_masks:
        # Classifier les pixels = -999 ou nan
        classified_array[final_data[:, :, 0] == -999] = 128
        classified_array[np.isnan(final_data[:, :, 0])] = 128

        # Classifier les pixels = -99 (roche)
        classified_array[final_data[:, :, 0] == -99] = 0
        # Classifier les pixels = -89 (ombre)
        classified_array[final_data[:, :, 0] == -89] = 1
        # Classifier les pixels bands 0 supérieurs à 2
        classified_array[final_data[:, :, 0] > 2] = 2

        # récupérer tous les autres pixels
        cond_restants = (
            (final_data[:, :, 0] != -99)
            & (final_data[:, :, 0] != -89)
            & (final_data[:, :, 0] <= 2)
            & (final_data[:, :, 0] != -999)
            & (~np.isnan(final_data[:, :, 0]))
        )

        class_dict = {
            0: "Roche",
            1: "Ombre",
            2: "Arbustes et Forêts",
            128: "NoData",
        }
    else:
        # Mode sans masques - commencer à partir de la classe 2
        # Classifier seulement les pixels = -999 ou nan
        classified_array[final_data[:, :, 0] == -999] = 128
        classified_array[np.isnan(final_data[:, :, 0])] = 128

        # Classifier les pixels bands 0 supérieurs à 2
        classified_array[final_data[:, :, 0] > 2] = 2

        # récupérer tous les autres pixels valides (exclure seulement NoData)
        cond_restants = (
            (final_data[:, :, 0] != -999)
            & (~np.isnan(final_data[:, :, 0]))
            & (final_data[:, :, 0] <= 2)
        )

        class_dict = {
            2: "Arbustes et Forêts",
            128: "NoData",
        }

    pixels_restants = final_data[cond_restants]

    return classified_array, class_dict, pixels_restants, cond_restants


def sort_clusters_by_height(pixels_restants, labels, class_dict, start_cluster_id=3):
    """
    Trie les clusters par hauteur et met à jour le dictionnaire des classes

    Args:
        pixels_restants: pixels utilisés pour le clustering
        labels: résultats du clustering
        class_dict: dictionnaire existant des classes
        start_cluster_id: ID de départ pour les nouveaux clusters

    Returns:
        class_dict: dictionnaire mis à jour
    """
    unique_clusters = np.unique(labels)

    # Calculer hauteur moyenne par cluster et trier
    cluster_heights = [
        (i, np.mean(pixels_restants[labels == i, 0])) for i in unique_clusters
    ]
    cluster_heights.sort(key=lambda x: x[1])  # Trier par hauteur

    # Créer le dictionnaire trié
    for new_id, (original_id, _) in enumerate(cluster_heights):
        mask = labels == original_id
        mean_height = np.mean(pixels_restants[mask, 0])
        mean_distance = (
            np.mean(pixels_restants[mask, -1]) if pixels_restants.shape[1] > 1 else 0
        )
        n_points = np.sum(mask)

        class_dict[new_id + start_cluster_id] = (
            f"Landes cluster {new_id + 1} H={mean_height:.2f}m "
            f"D={mean_distance:.2f} N={n_points}"
        )
    return class_dict
