# Documentation Technique - Package de Traitement LIDAR

## Vue d'ensemble

Le package LIDAR Processing Pipeline est un système de traitement complet pour les données LIDAR permettant de générer des indicateurs de végétation et des classifications de couverture végétale.

## Architecture du Pipeline

Le pipeline est organisé en trois étapes principales qui s'exécutent de manière séquentielle :

### Étape 1 : Traitement LIDAR (step1_process_lidar.py)
### Étape 2 : Estimation CHM et FHD (step2_estimate_chm.py) 
### Étape 3 : Classification par Clustering (step3_chm_clustering.py)

## Structure du Projet

```bash
lidar_processing/
├── main_pipeline.py              # Script principal d'orchestration
├── step1_process_lidar.py        # Traitement des données LIDAR brutes
├── step2_estimate_chm.py         # Calcul CHM et FHD
├── step3_chm_clustering.py       # Classification par clustering
├── config/                       # Fichiers de configuration
│   ├── meta_config.json         # Configuration globale du pipeline
│   ├── step1_config.json        # Configuration étape 1
│   ├── step2_config.json        # Configuration étape 2
│   └── step3_config.json        # Configuration étape 3
├── src/                         # Code source
│   ├── processor.py             # Classe principale de traitement
│   ├── lidar_utils.py          # Utilitaires PDAL
│   ├── vegetation_indicators.py # Calcul des indicateurs
│   ├── geo_tools.py            # Outils géographiques
│   └── learning/               # Algorithmes de classification
└── test/                       # Suite de tests
```

## Étape 1 : Traitement LIDAR

### Objectif
Traiter les données LIDAR brutes pour les nettoyer, les filtrer et calculer les hauteurs au-dessus du sol.

### Fichiers d'entrée
- **Données LIDAR** : Fichiers LAZ compressés contenant les nuages de points
  - Format : LAZ/LAS (standard ASPRS)
  - Contenu : Coordonnées 3D (X, Y, Z), intensité, classification des points
  - Système de coordonnées : EPSG:2154 (RGF93 / Lambert-93)

- **Zone d'intérêt** : Fichier shapefile délimitant la zone d'étude
  - Format : SHP (ESRI Shapefile)
  - Contenu : Polygones définissant les limites géographiques
  - Usage : Découpage spatial des données LIDAR

- **Liste des tuiles** : Fichier shapefile indexant les tuiles LIDAR disponibles
  - Format : SHP (ESRI Shapefile)  
  - Contenu : Référencement spatial des fichiers LIDAR

### Configuration (step1_config.json)

#### Paramètres de traitement (processor_settings)
- **group** (20) : Nombre de tuiles traitées simultanément pour optimiser l'usage mémoire
- **sampling_strategy** ("adjacent") : Stratégie de regroupement des tuiles spatiales
- **n_jobs** (10) : Nombre de processus parallèles pour le traitement
- **crop** (1) : Active le découpage selon la zone d'intérêt
- **skip_uncompress** (false) : Permet d'ignorer la décompression si déjà effectuée

**Justification de la stratégie de regroupement :**
La stratégie "adjacent" groupe 20 tuiles côte à côte pour le traitement parallèle, maximisant la quantité de données en mémoire tout en permettant l'estimation de la hauteur au-dessus du sol (HAG) sur une grande zone. Cette approche assure une meilleure homogénéité lors de l'interpolation (interpolation sur plusieurs tuiles fusionnées plutôt qu'interpolation individuelle puis fusion). L'alternative "random" sélectionne des tuiles au hasard dans tout le jeu de données, permettant de traiter des zones variées mais perdant l'avantage de continuité spatiale.

#### Variables conservées (keep_variables)
- **X, Y, Z** : Coordonnées spatiales 3D des points LIDAR
- **Intensity** : Valeur de réflectance du signal LIDAR
- **Classification** : Classification automatique des points (sol, végétation, bâtiments)
- **HeightAboveGround** : Hauteur relative au sol calculée par le pipeline

#### Pipeline PDAL (pdal_pipeline)
Le traitement utilise PDAL (Point Data Abstraction Library) en deux phases :

**Phase 1 - Nettoyage individuel des tuiles :**
- **filters.outlier** : Suppression des points aberrants (méthode statistique, k=20, multiplicateur=3.0)
- **filters.assign** : Normalisation des attributs de retour (NumberOfReturns=1, ReturnNumber=1)
- **filters.elm** : Filtrage statistique du bruit (Extended Local Minimum) (cellule=2.85m)

**Justification du pipeline de nettoyage :**
Cette séquence de nettoyage a été choisie après tests de 4 configurations différentes face à l'hétérogénéité qualitative des données LIDAR entre tuiles. L'approche initiale utilisant directement la classification IGN intégrée échoue sur certaines zones (classification erronée en roche alors que végétation présente). Les tentatives de re-classification avec Myriad3D (IGN) donnent des résultats similaires, indiquant un problème dans les données initiales. Les filtres statistiques (CSF, PMF, SMRF) nécessitent un réglage d'hyperparamètres variant selon les régions et donnent des résultats moins bons que la classification IGN originale mais permettent un traitement homogène sur toutes les tuiles, y compris celles avec une classification de base défaillante.

**Phase 2 - Traitement global :**
- **filters.hag_delaunay** : Calcul de la hauteur au-dessus du sol par triangulation (count=80)
- **filters.range** : Filtrage des hauteurs valides (0-35m au-dessus du sol)
- **filters.merge** : Fusion des données traitées

#### Paramètres CHM initial (chm_settings)
- **resolution** (5m) : Résolution spatiale du modèle de hauteur de canopée
- **quantile** (95%) : Percentile utilisé pour le calcul de hauteur par pixel

**Justification de la résolution 5m :**
Cette résolution permet d'avoir suffisamment de points LIDAR par pixel tout en gardant une échelle locale d'analyse. Le regroupement ultérieur à 20m correspond aux relevés terrain réalisés sur des fenêtres de 20m×20m. La discrimination à une échelle extrêmement fine (5m) n'est pas forcément pertinente avec seulement deux caractéristiques (CHM et FHD) qui sont loin d'être optimales.

### Processus de traitement

1. **Sélection des tuiles** : Identification des fichiers LIDAR intersectant la zone d'intérêt
2. **Décompression** : Conversion LAZ vers LAS si nécessaire
3. **Découpage spatial** : Application de la zone d'intérêt pour réduire le volume de données
4. **Traitement PDAL** : Application du pipeline de nettoyage et calcul des hauteurs
5. **Conversion** : Sauvegarde des données traitées au format NumPy (.npy)
6. **Génération CHM** : Création d'un modèle de hauteur de canopée initial
7. **Fusion** : Assemblage des tuiles individuelles en une mosaïque globale

### Sorties
- **Données traitées** : Fichiers .npy contenant les points LIDAR nettoyés et enrichis
- **CHM individuel** : Modèles de hauteur de canopée par tuile (.tif)
- **CHM fusionné** : Mosaïque complète de la zone d'étude
- **Logs** : Fichiers de journalisation détaillant le processus et les erreurs

## Étape 2 : Estimation CHM et FHD

### Objectif
Calculer des modèles de hauteur de canopée avec différents quantiles et estimer la diversité de hauteur du feuillage.

### Fichiers d'entrée
- **Données traitées** : Fichiers .npy issus de l'étape 1
  - Contenu : Points LIDAR nettoyés avec attributs calculés
  - Format : Tableaux NumPy structurés

### Configuration (step2_config.json)

#### Paramètres CHM (chm_settings)
- **resolution** (20m) : Résolution spatiale finale des rasters
- **quantile_95** (95%) : Percentile pour la hauteur maximale de canopée
- **quantile_50** (50%) : Percentile médian pour la hauteur de canopée

**Justification des percentiles 95% et 50% :**
Le choix s'éloigne du CHM traditionnel (hauteur maximale) car l'observation montre qu'un CHM construit à 2m puis moyenné à 20m pour la classification donne des résultats plus réalistes. Le percentile 95% filtre les outliers potentiels sur la cellule de résolution tandis que la médiane (50%) montre une bonne corrélation avec les mesures terrain. Le maximum n'est pas pertinent sur des cellules larges car trop sensible aux outliers présents sur une zone étendue. [Ligne réservée pour image des corrélations terrain]

#### Paramètres FHD (fhd_settings)
- **resolution** (20m) : Résolution spatiale pour le calcul FHD
- **zmin** (0m) : Hauteur minimale considérée
- **zmax** (2m) : Hauteur maximale considérée pour la végétation basse
- **zwidth** (0.15m) : Largeur des classes de hauteur

**Justification des paramètres FHD :**
Le range 0-2m est optimisé pour la végétation basse (landes) en milieux montagneux, sujet d'étude principal. Le seuil de 15cm minimum est basé sur la résolution altimétrique absolue (10cm) avec une marge pour différencier les zones de rocaille de la végétation. Ce paramétrage est adapté à l'étude des landes en haute montagne.

#### Fichiers de sortie (output_files)
- **chm_95_suffix** : Suffixe pour les fichiers CHM 95% (_chm_q95.tif)
- **chm_50_suffix** : Suffixe pour les fichiers CHM 50% (_chm_q50.tif)
- **fhd_suffix** : Suffixe pour les fichiers FHD (_fhd.tif)
- **multichannel_suffix** : Fichiers multicanaux combinés (_multichannel.tif)

### Processus de traitement

1. **Lecture des données** : Chargement des fichiers .npy de l'étape 1
2. **Calcul CHM 95%** : Génération du modèle de hauteur avec le 95e percentile
3. **Calcul CHM 50%** : Génération du modèle de hauteur avec le percentile médian
4. **Calcul FHD** : Estimation de la diversité de hauteur du feuillage
5. **Validation spatiale** : Vérification de la cohérence des transformations géospatiales
6. **Sauvegarde individuelle** : Export des rasters par tuile
7. **Création multicanal** : Combinaison CHM95, CHM50 et FHD en un seul fichier
8. **Fusion globale** : Assemblage des tuiles en mosaïques complètes

### Algorithmes utilisés

#### Modèle de Hauteur de Canopée (CHM)
Le CHM est calculé par rasterisation des points LIDAR selon une grille régulière. Pour chaque pixel :
- Sélection des points LIDAR contenus dans le pixel
- Calcul du percentile spécifié des hauteurs au-dessus du sol
- Attribution de cette valeur au pixel du raster

#### Diversité de Hauteur du Feuillage (FHD)
Le FHD quantifie la diversité verticale de la végétation :
- Division de l'espace vertical en classes de hauteur (zwidth)
- Comptage des points par classe dans chaque pixel
- Calcul de l'indice de diversité de Shannon
- FHD = -Σ(pi × ln(pi)) où pi est la proportion de points dans la classe i

### Sorties
- **CHM 95%** : Raster de hauteur maximale de canopée (.tif)
- **CHM 50%** : Raster de hauteur médiane de canopée (.tif)
- **FHD** : Raster de diversité de hauteur du feuillage (.tif)
- **Multicanal** : Raster à 3 bandes combinant les trois indicateurs (.tif)
- **Métadonnées** : Information sur les paramètres et dates de création

## Étape 3 : Classification par Clustering

### Objectif
Appliquer des algorithmes de clustering pour classifier automatiquement les types de végétation basés sur les indicateurs CHM et FHD.

### Fichiers d'entrée
- **Rasters multicanaux** : Fichiers .tif issus de l'étape 2
  - Bande 1 : CHM 95% (hauteur maximale)
  - Bande 2 : CHM 50% (hauteur médiane)  
  - Bande 3 : FHD (diversité de hauteur)

- **Masques optionnels** :
  - **Masque de roche** : Zones rocheuses à exclure (.tif)
  - **Masque d'ombre** : Zones d'ombre à exclure (.tif)

### Configuration (step3_config.json)

#### Paramètres de clustering (clustering_settings)
- **method** ("gmm") : Algorithme de clustering
  - "kmeans" : K-moyennes classique
  - "gmm" : Mélange de gaussiennes (Gaussian Mixture Model)
- **n_clusters** (4) : Nombre de classes de végétation à identifier
- **random_state** (42) : Graine aléatoire pour la reproductibilité
- **max_iter** (20000) : Nombre maximum d'itérations
- **tolerance** (1e-6) : Critère de convergence

**Justification du choix GMM vs K-means :**
Deux algorithmes statistiquement différents sont proposés pour observer les différences potentielles. GMM est privilégié par défaut pour sa flexibilité avec des clusters non sphériques, plus adaptés aux données de végétation hétérogènes.

#### Paramètres spécifiques GMM (gmm_settings)
- **init_params** ("k-means++") : Méthode d'initialisation des centres
- **covariance_type** ("full") : Type de matrice de covariance

#### Configuration des données (data_settings)
- **channels_to_use** ([0, 2]) : Bandes utilisées pour la classification
  - 0 : CHM 95% (hauteur maximale)
  - 2 : FHD (diversité de hauteur)
- **move_axis_from/to** (0, -1) : Réorganisation des dimensions pour le traitement

**Justification de la sélection des canaux :**
Seuls CHM 95% et FHD sont utilisés pour maintenir l'approche de classification basée uniquement sur ces deux indicateurs complémentaires. Le CHM 95% filtre les outliers potentiels tout en conservant l'information de hauteur, tandis que FHD apporte la dimension de diversité structurelle. Le CHM 50% (médiane) n'est pas utilisé pour éviter la redondance avec le CHM 95%.

#### Couleurs de classification (classification_colors)
- **0** : Roche (Gris pierre [139, 137, 137])
- **1** : Ombre (Gris très foncé [47, 47, 47])
- **2** : Arbustes et forêts (Vert forêt [34, 139, 34])
- **3** : Lande rase (Pêche clair [255, 218, 185])
- **4** : Lande dense (Marron clair [205, 133, 63])
- **5** : Lande moyenne (Marron saddle [160, 82, 45])
- **6** : Lande arbustive (Marron chocolat [72, 30, 0])
- **7** : Lande très dense (Vert foncé [0, 100, 0])
- **128** : NoData (Noir transparent [0, 0, 0])

### Processus de traitement

1. **Fusion des images** : Assemblage des rasters multicanaux en une image composite
2. **Application des masques** : Exclusion des zones rocheuses et d'ombre si disponibles
3. **Préparation des données** : 
   - Extraction des canaux sélectionnés
   - Masquage des valeurs NoData
   - Préparation des pixels pour la classification
4. **Classification initiale** : Attribution des classes prédéfinies (roche=0, ombre=1, végétation=2)
5. **Clustering** : Application de l'algorithme sélectionné sur les pixels restants
6. **Tri par hauteur** : Réorganisation des clusters selon la hauteur moyenne croissante
7. **Sauvegarde** : Export du raster classifié avec palette de couleurs

### Algorithmes de clustering

#### K-moyennes (K-means)
Algorithme de partitionnement qui divise les données en k clusters :
- Initialisation aléatoire de k centres
- Attribution de chaque point au centre le plus proche
- Recalcul des centres comme moyenne des points assignés
- Répétition jusqu'à convergence

#### Mélange de Gaussiennes (GMM)
Modèle probabiliste qui considère les données comme un mélange de distributions gaussiennes :
- Chaque cluster est modélisé par une gaussienne multivariée
- Estimation des paramètres par l'algorithme EM (Espérance-Maximisation)
- Attribution probabiliste des points aux clusters
- Plus flexible que K-moyennes pour des clusters non sphériques

### Gestion des masques

Le système supporte deux modes de fonctionnement :

#### Avec masques
- Classes utilisées : Roche (0), Ombre (1), Arbustes/Forêts (2), Landes (3-6), NoData (128)
- Les pixels masqués sont pré-classifiés et exclus du clustering
- Les clusters commencent à l'ID 3

#### Sans masques  
- Classes utilisées : Arbustes/Forêts (2), Landes (3-6), NoData (128)
- Tous les pixels valides participent au clustering
- Les clusters commencent à l'ID 2

### Sorties
- **Raster classifié** : Image avec les classes de végétation identifiées
- **Palette de couleurs** : Attribution de couleurs spécifiques à chaque classe
- **Métadonnées** : Informations sur l'algorithme utilisé et les paramètres
- **Statistiques** : Nombre de pixels par classe et répartition spatiale

## Orchestration du Pipeline

### Script principal (main_pipeline.py)

Le pipeline est orchestré par une classe PipelineExecutor qui :

#### Gestion de configuration
- Charge la configuration globale (meta_config.json)
- Valide les fichiers de configuration des étapes
- Vérifie l'existence des scripts d'exécution

#### Gestion des dépendances
- Étape 1 : Aucune dépendance
- Étape 2 : Dépend de l'étape 1
- Étape 3 : Dépend de l'étape 2

#### Surveillance et logging
- Logs centralisés dans le répertoire logs/
- Suivi des temps d'exécution par étape
- Gestion des erreurs et récupération
- Rapports de progression détaillés

#### Exécution
```bash
# Exécution complète
python main_pipeline.py

# Étapes spécifiques
python main_pipeline.py --steps 1 2

# Configuration personnalisée
python main_pipeline.py --config config/custom_meta_config.json

# Listage des étapes
python main_pipeline.py --list
```

### Configuration globale (meta_config.json)

#### Paramètres du pipeline (pipeline_settings)
- **run_all_steps** (true) : Exécute toutes les étapes activées
- **steps_to_run** ([1,2,3]) : Liste des étapes à exécuter
- **continue_on_error** (false) : Continue l'exécution malgré les erreurs
- **verbose** (true) : Active les logs détaillés
- **save_intermediate_results** (true) : Conserve les résultats intermédiaires

#### Surveillance (monitoring)
- **track_progress** (true) : Active le suivi de progression
- **send_notifications** (false) : Notifications par email (non implémenté)

## Gestion des Ressources

### Optimisation mémoire
- Traitement par groupes de tuiles pour limiter l'usage RAM
- Libération explicite de la mémoire (garbage collection)
- Surveillance continue de l'utilisation mémoire
- Adaptation dynamique de la taille des groupes

### Parallélisation
- Traitement multiprocessus des tuiles LIDAR
- Paramétrage du nombre de workers selon les ressources disponibles
- Optimisation I/O avec traitement en pipeline

### Gestion d'erreurs
- Mécanisme de retry avec paramètres dégradés
- Sauvegarde de points de contrôle (checkpoints)
- Logs détaillés pour le débogage
- Récupération automatique après interruption

## Formats de Données

### Fichiers d'entrée
- **LAZ/LAS** : Nuages de points LIDAR compressés/non compressés
- **SHP** : Shapefiles pour zones d'intérêt et indexation
- **JSON** : Fichiers de configuration

### Fichiers intermédiaires
- **NPY** : Arrays NumPy pour stockage efficace des points traités
- **TIF** : Rasters géoréférencés (CHM, FHD, masques)

### Fichiers de sortie
- **TIF** : Rasters de classification avec géoréférencement
- **LOG** : Fichiers de journalisation
- **JSON** : Métadonnées et statistiques

## Limitations et Perspectives

### Limitations actuelles

#### Qualité hétérogène des données LIDAR
- **Variabilité de la classification IGN** : Certaines tuiles présentent des classifications erronées (zones végétalisées classées en roche)
- **Densité variable** : La densité des points diminue avec l'altitude et varie significativement entre zones, impactant directement l'estimation de végétation
- **Cohérence spatiale** : Les classifications peuvent être inconsistantes aux bordures de tuiles

#### Compromis techniques
- **Re-classification nécessaire** : Bien que donnant des résultats moins précis que la classification IGN originale, la re-classification assure l'homogénéité du traitement
- **Hyperparamètres régionaux** : Les filtres statistiques (CSF, PMF, SMRF) nécessitent un ajustement selon les régions

### Perspectives d'amélioration

#### Classification LIDAR
- **Réentraînement de Myriad3D** : Adapter le classifieur IGN avec des points de validation spécifiques à la végétation de haute montagne
- **Fenêtre glissante** : Améliorer le traitement des bordures de tuiles pour éviter les aberrations de classification

#### Indicateurs de végétation
- **Enrichissement des variables** : Intégrer d'autres indicateurs au-delà de CHM et FHD pour améliorer la discrimination
- **Validation terrain étendue** : Augmenter le jeu de données de référence pour la validation des corrélations

[Ligne réservée pour images des zones de Belledone illustrant les problèmes de qualité des données]

## Système de Projection

Tous les traitements utilisent le système de coordonnées **EPSG:2154 (RGF93 / Lambert-93)** :
- Système officiel pour la France métropolitaine
- Unités en mètres
- Minimise les déformations pour les calculs de surface et distance
- **Avantage technique** : Système de coordonnées des fichiers LIDAR IGN, simplifiant les traitements sans reprojection

## Exigences Système

### Matériel recommandé
- **RAM** : Minimum 16 GB, recommandé 32 GB ou plus
- **Stockage** : SSD recommandé pour améliorer les performances I/O
- **CPU** : Processeur multicœur (8+ cœurs recommandés)

### Logiciels requis
- **Python** 3.8+
- **PDAL** 2.0+ avec support LAZ
- **GDAL** 3.0+ pour les opérations raster
- **scikit-learn** pour les algorithmes de clustering
- **NumPy, rasterio, pandas** pour la manipulation de données

### Estimation des temps de traitement
- **Étape 1** : 1-3 heures pour 100 km² (selon densité LIDAR)
- **Étape 2** : 30-60 minutes pour 100 km²
- **Étape 3** : 15-30 minutes pour 100 km²

Les temps varient selon la densité des points LIDAR, la complexité du terrain, et les ressources matérielles disponibles.
