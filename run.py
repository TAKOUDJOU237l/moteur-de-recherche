#!/usr/bin/env python3
"""
Script de démarrage rapide du moteur de recherche visuel
Vérifie les prérequis et lance l'application
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Vérifie que toutes les dépendances sont installées."""
    deps = ['torch', 'torchvision', 'flask', 'PIL', 'numpy']
    missing = []
    
    for dep in deps:
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    
    if missing:
        logger.error(f"Dépendances manquantes: {', '.join(missing)}")
        logger.info("Installe-les avec: pip install -r requirements.txt")
        return False
    return True

def check_index():
    """Vérifie que l'index a été construit."""
    embed_path = Path("static/embeddings/embeddings.npy")
    paths_path = Path("static/embeddings/image_paths.json")
    
    if not embed_path.exists() or not paths_path.exists():
        logger.error("Index vectoriel non trouvé!")
        logger.info("Lance d'abord: python build_index.py")
        return False
    
    logger.info("✅ Index trouvé")
    return True

def main():
    logger.info("🔍 Vérification du moteur de recherche visuel...")
    
    if not check_dependencies():
        return 1
    
    logger.info("✅ Toutes les dépendances sont installées")
    
    if not check_index():
        return 1
    
    logger.info("\n" + "="*50)
    logger.info("✅ Toutes les vérifications sont passées")
    logger.info("Démarrage du serveur...")
    logger.info("="*50 + "\n")
    
    from app import app
    app.run(debug=True, host="0.0.0.0", port=5000)
    return 0

if __name__ == "__main__":
    sys.exit(main())
