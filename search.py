import json
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List, Dict, Optional

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 2048
BASE_DIR      = Path(__file__).resolve().parent
EMBED_DIR     = BASE_DIR / "static" / "embeddings"
INDEX_FILE    = EMBED_DIR / "embeddings.npy"
PATHS_FILE    = EMBED_DIR / "image_paths.json"
EMBED_FILE    = INDEX_FILE
SKLEARN_INDEX_FILE = BASE_DIR / "clothing_index_sklearn.pkl"


def build_index(embeddings, save_path=SKLEARN_INDEX_FILE):
    from sklearn.neighbors import NearestNeighbors
    import pickle

    index = NearestNeighbors(
        n_neighbors=20,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1
    )
    index.fit(embeddings)

    # Sauvegarde du .pkl
    with open(save_path, "wb") as f:
        pickle.dump({"index": index, "embeddings": embeddings}, f)

    print(f"Index sauvegardé : {save_path}")
    return index


def load_index(path: str = INDEX_FILE) -> Tuple[np.ndarray, None]:
    """Charge les embeddings depuis le disque."""
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Index introuvable : {path}")
    
    try:
        embeddings = np.load(path)
        logger.info(f"Index chargé : {embeddings.shape}")
        
        if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
            raise ValueError(f"Format d'index invalide: {embeddings.shape}")
        
        return embeddings, None
    except Exception as e:
        logger.error(f"Erreur lors du chargement de l'index : {e}")
        raise


def load_metadata(path: str = PATHS_FILE) -> Tuple[List[str], List[str]]:
    """Charge les chemins et labels des images."""
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Métadonnées introuvables : {path}")
    
    try:
        with open(path, "r") as f:
            data = json.load(f)
        
        if "paths" not in data or "labels" not in data:
            raise ValueError("Format des métadonnées invalide")
        
        if len(data["paths"]) != len(data["labels"]):
            raise ValueError("Mismatch entre chemins et labels")
        
        logger.info(f"Métadonnées chargées : {len(data['paths'])} images")
        return data["paths"], data["labels"]
    
    except json.JSONDecodeError as e:
        logger.error(f"Erreur de décodage JSON : {e}")
        raise


def search(query_vector: np.ndarray,
           index: np.ndarray,
           image_paths: List[str],
           image_labels: List[str],
           top_k: int = 6,
           exclude_path: Optional[str] = None) -> List[Dict]:
    """
    Recherche les top_k images les plus similaires.
    Retourne une liste de dict {path, label, similarity, rank}.
    """
    embeddings = np.asarray(index, dtype=np.float32)
    
    if embeddings.size == 0:
        logger.warning("Index vide")
        return []
    
    if len(embeddings) != len(image_paths) or len(image_paths) != len(image_labels):
        raise ValueError("Mismatch entre embeddings et métadonnées")

    # Normaliser le vecteur de requête
    q = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    q_norm[q_norm == 0] = 1.0
    q = q / q_norm

    # Calculer les similarités par produit scalaire (cosine avec vecteurs normalisés)
    similarities = embeddings @ q[0]
    ranked_indices = np.argsort(-similarities)

    results = []
    exclude_resolved = None
    if exclude_path:
        try:
            exclude_resolved = Path(exclude_path).resolve()
        except Exception:
            pass
    
    for idx in ranked_indices:
        idx = int(idx)
        path = image_paths[idx]
        
        # Exclure l'image requête si spécifié
        if exclude_resolved:
            try:
                if Path(path).resolve() == exclude_resolved:
                    continue
            except Exception:
                pass
        
        similarity = round(float(similarities[idx]), 4)
        results.append({
            "path"      : path,
            "label"     : image_labels[idx],
            "similarity": similarity,
            "rank"      : len(results) + 1
        })
        
        if len(results) >= top_k:
            break

    return results