# build_index.py
"""
Script à lancer UNE SEULE FOIS pour :
1. Scanner toutes les images du Clothing Dataset
2. Les encoder avec ResNet50
3. Sauvegarder les embeddings utilisés par la recherche

Utilisation :
    python build_index.py --data_dir clothing-dataset
"""
import os
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

from encoder import load_encoder, encode_batch
from search  import build_index, EMBED_FILE, PATHS_FILE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SUPPORTED = {".jpg", ".jpeg", ".png", ".webp"}


def resolve_data_dir(data_dir: str) -> Path:
    """Résout le dossier dataset depuis plusieurs emplacements courants."""
    candidate = Path(data_dir)
    if candidate.exists():
        return candidate

    script_dir = Path(__file__).resolve().parent
    for base in (script_dir, script_dir.parent, Path.cwd()):
        fallback = base / data_dir
        if fallback.exists():
            return fallback

    return candidate


def scan_images(root: Path):
    """Scanne récursivement les images et retourne (paths, labels)."""
    paths, labels = [], []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in SUPPORTED:
            paths.append(str(p))
            labels.append(p.parent.name)
    return paths, labels


def main(data_dir: str, batch_size: int = 32):
    try:
        root = resolve_data_dir(data_dir)
        if not root.exists():
            logger.error(f"Dossier introuvable : {data_dir}")
            logger.info("Vérifie le chemin ou passe un chemin absolu vers le dataset.")
            return False

        logger.info(f"Scan des images dans : {root}")
        paths, labels = scan_images(root)
        logger.info(f"{len(paths):,} images trouvées")
        logger.info(f"{len(set(labels))} catégories : {sorted(set(labels))}")

        if len(paths) == 0:
            logger.error("Aucune image trouvée. Vérifie le chemin du dataset.")
            return False

        logger.info("Chargement de ResNet50...")
        model = load_encoder()
        logger.info("ResNet50 chargé avec succès")

        logger.info(f"Extraction des embeddings (batch_size={batch_size})...")
        embeddings = encode_batch(paths, model, batch_size=batch_size)
        logger.info(f"Embeddings extraits : {embeddings.shape}")

        # ── Sauvegarde embeddings .npy ─────────────────────────────
        import numpy as np
        np.save(EMBED_FILE, embeddings)
        logger.info(f"Embeddings sauvegardés : {EMBED_FILE}")

        # ── Sauvegarde métadonnées .json ───────────────────────────
        with open(PATHS_FILE, "w") as f:
            json.dump({"paths": paths, "labels": labels}, f)
        logger.info(f"Métadonnées sauvegardées : {PATHS_FILE}")

        # ── Construction index sklearn .pkl ────────────────────────
        logger.info("Construction de l'index de recherche...")
        build_index(embeddings)   # crée clothing_index_sklearn.pkl
        logger.info("Index sklearn construit !")

        logger.info(f"✅ Index vectoriel prêt ({len(embeddings):,} vecteurs)")
        logger.info("Lance maintenant : python app.py")
        return True

    except Exception as e:
        logger.error(f"Erreur : {e}")
        import traceback
        traceback.print_exc()
        return False




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Construit l'index vectoriel pour la recherche visuelle"
    )
    parser.add_argument("--data_dir",   default="clothing-dataset",
                        help="Chemin vers le dossier du dataset (relatif ou absolu)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Taille des batches pour l'encodage")
    args = parser.parse_args()
    
    success = main(args.data_dir, args.batch_size)
    exit(0 if success else 1)