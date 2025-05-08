lidar_processing/
├── config/
│   └── default_config.json
├── src/
│   ├── __init__.py
│   ├── grid.py         # Fonctions de gestion de la grille
│   ├── visualization.py # Fonctions de visualisation
│   ├── utils.py        # Fonctions utilitaires
│   ├── processor.py    # Classe principale LidarProcessor
│   └── extractors/     # Pour les futurs extracteurs d'indicateurs
│       └── __init__.py
├── cli.py              # Interface CLI avec Click
└── README.md