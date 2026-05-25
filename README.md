# 🔍 Visual Search - Moteur de Recherche Visuelle

Une application web permettant de trouver des images similaires dans un dataset à partir d'une image requête. Utilise ResNet50 pour l'encodage d'images et la recherche par plus proches voisins (KNN).

## 📋 Pipeline du Projet

### 1. **Préparation - Construction de l'Index**

```
build_index.py
├─ Scan le dataset (clothing-dataset/)
├─ Charge ResNet50 (modèle pré-entraîné)
├─ Encode chaque image → vecteur 2048D
├─ Sauvegarde :
│  ├─ embeddings.npy (tous les vecteurs)
│  ├─ image_paths.json (métadonnées)
│  └─ clothing_index_sklearn.pkl (index KNN)
└─ À lancer UNE SEULE FOIS en début
```

**Commande :**
```bash
python build_index.py --data_dir clothing-dataset
```

### 2. **Application Web (Flask)**

```
run.py / app.py
├─ Lance le serveur Flask (http://localhost:5000)
├─ Charge l'encodeur et l'index (au démarrage)
└─ Routes :
   ├─ GET / → formulaire d'upload d'image
   ├─ POST /search → traite l'image requête
   └─ GET /image → sert les images du dataset
```

### 3. **Recherche à l'Exécution**

```
User upload image
    ↓
encode_image() → vecteur 2048D
    ↓
search() → calcul similarité cosine
    ↓
Retourne top-8 images similaires
    ↓
Template HTML affiche résultats + pourcentage similarité
```

## 🛠️ Stack Technique

| Composant | Technologie |
|-----------|-------------|
| **Backend** | Flask (Python) |
| **Vision** | ResNet50 (PyTorch/TorchVision) |
| **Recherche** | sklearn NearestNeighbors (cosine distance) |
| **Frontend** | Jinja2 templates |
| **Traitement Image** | Pillow |
| **Calcul Numérique** | NumPy |

## 📦 Installation

### Prérequis
- Python 3.8+
- pip

### Étapes

1. **Cloner le dépôt et naviguer au dossier :**
```bash
cd visual_search
```

2. **Créer un environnement virtuel (optionnel mais recommandé) :**
```bash
python -m venv venv
source venv/bin/activate  # Unix/Mac
# ou
venv\Scripts\activate  # Windows
```

3. **Installer les dépendances :**
```bash
pip install -r requirements.txt
```

## 🚀 Utilisation

### Étape 1 : Construire l'Index Vectoriel

Avant de lancer l'application, créer l'index à partir du dataset :

```bash
python build_index.py --data_dir clothing-dataset --batch_size 32
```

**Options :**
- `--data_dir` : Chemin du dossier dataset (défaut: `clothing-dataset`)
- `--batch_size` : Taille des batches pour l'encodage (défaut: 32)

**Sortie :**
- `static/embeddings/embeddings.npy` - Vecteurs embeddings
- `static/embeddings/image_paths.json` - Métadonnées (chemins et labels)
- `clothing_index_sklearn.pkl` - Index KNN persistant

### Étape 2 : Lancer l'Application

```bash
python run.py
```

ou directement :

```bash
python app.py
```

L'application démarre sur : **http://localhost:5000**

### Étape 3 : Utiliser l'Interface

1. Accéder à http://localhost:5000
2. Uploader une image (JPG, PNG, WEBP)
3. Voir les 8 images les plus similaires du dataset

## 📁 Structure du Projet

```
visual_search/
├── app.py                          # Application Flask principale
├── run.py                          # Script de démarrage avec vérifications
├── build_index.py                  # Construction de l'index vectoriel
├── encoder.py                      # Chargement et encodage ResNet50
├── search.py                       # Logique de recherche KNN
├── requirements.txt                # Dépendances Python
├── clothing-dataset/               # Dataset d'images
├── static/
│   ├── embeddings/
│   │   ├── embeddings.npy         # Vecteurs (généré)
│   │   └── image_paths.json       # Métadonnées (généré)
│   └── uploads/                    # Images uploadées temporairement
├── templates/
│   ├── index.html                  # Formulaire upload
│   └── results.html                # Page résultats
└── clothing_index_sklearn.pkl      # Index KNN (généré)
```

## 🔧 Configuration

### Paramètres Principaux (app.py)

```python
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # Taille max upload (16 MB)
ALLOWED_EXT = {"jpg", "jpeg", "png", "webp"}  # Formats autorisés
TOP_K = 8  # Nombre de résultats retournés
```

## ⚙️ Modules Clés

### `encoder.py`
- `load_encoder()` - Charge ResNet50 pré-entraîné
- `encode_image()` - Encode une image unique
- `encode_batch()` - Encode un batch d'images

### `search.py`
- `build_index()` - Crée l'index KNN
- `load_index()` - Charge les embeddings
- `load_metadata()` - Charge les chemins et labels
- `search()` - Recherche les K plus proches voisins

### `app.py`
- Route `/` - Page d'accueil
- Route `/search` (POST) - Traitement de la recherche
- Route `/image` - Serveur d'images

## 📊 Performance

- **Encodage** : ~30-50 ms par image (batch)
- **Recherche** : <5 ms pour top-8 résultats
- **Mémoire** : ~2 GB pour 1000 images (ResNet50 + embeddings)

## ⚠️ Notes Importantes

1. **Index obligatoire** : L'application ne fonctionne que si `build_index.py` a été exécuté
2. **Première exécution** : Le chargement de ResNet50 prend ~30 secondes
3. **Dataset** : Assurez-vous que le dossier `clothing-dataset/` existe et contient des images

## 🔒 Sécurité

- Validation des extensions de fichier
- Limite de taille d'upload (16 MB)
- Vérification du path pour éviter les traversées de répertoires
- Gestion des erreurs robuste

## 📝 Logs

Les logs sont configurés en INFO level et affichent :
- Chargement des modèles
- État de l'index
- Erreurs de requête
- Opérations de recherche

## 🤝 Contribution

Améliorations possibles :
- Support GPU pour ResNet50
- Cache des embeddings en base de données
- API REST complète
- Interface avec pagination
- Support de la recherche par texte

## 📄 Licence

À compléter selon vos besoins.
