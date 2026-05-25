# app.py
import os
import uuid
import logging
from pathlib import Path
from flask import (Flask, render_template, request,
                   jsonify, send_file, url_for, redirect)

from encoder import load_encoder, encode_image
from search  import load_index, load_metadata, search

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────
app             = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max
UPLOAD_FOLDER   = Path("static/uploads")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
ALLOWED_EXT     = {"jpg", "jpeg", "png", "webp"}
TOP_K           = 8

# ── Chargement du modèle et de l'index (une seule fois au démarrage) ──────
logger.info("Chargement de ResNet50...")
encoder = load_encoder()
logger.info("✅ Encodeur prêt")

logger.info("Chargement de l'index vectoriel...")
try:
    ann_index, _ = load_index()
    image_paths, image_labels = load_metadata()
    logger.info(f"✅ Index prêt ({len(image_paths):,} images)")
except FileNotFoundError as e:
    logger.warning(f"Index introuvable: {e}")
    logger.info("Lance d'abord : python build_index.py")
    ann_index, image_paths, image_labels = None, [], []
except Exception as e:
    logger.error(f"Erreur lors du chargement de l'index : {e}")
    ann_index, image_paths, image_labels = None, [], []


def allowed_file(filename: str) -> bool:
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Page d'accueil avec le formulaire d'upload."""
    return render_template("index.html",
                           n_images=len(image_paths),
                           index_ready=ann_index is not None)


@app.route("/search", methods=["POST"])
def search_image():
    """Reçoit l'image, l'encode et retourne les résultats."""
    if ann_index is None:
        return render_template("results.html", error=(
            "L'index n'est pas encore construit. "
            "Lance d'abord : python build_index.py"
        ))

    if "image" not in request.files:
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return render_template("results.html",
                               error="Format non supporté. Utilise JPG, PNG ou WEBP.")

    # Sauvegarde temporaire de l'image requête
    ext       = file.filename.rsplit(".", 1)[1].lower()
    filename  = f"{uuid.uuid4().hex}.{ext}"
    save_path = UPLOAD_FOLDER / filename
    file.save(save_path)

    try:
        # Encodage de la requête
        query_vec = encode_image(str(save_path), encoder)

        # Recherche des plus proches voisins
        results = search(
            query_vector  = query_vec,
            index         = ann_index,
            image_paths   = image_paths,
            image_labels  = image_labels,
            top_k         = TOP_K,
            exclude_path  = str(save_path)
        )

        # Construire les URLs des images résultats
        for r in results:
            r["url"]     = url_for("serve_image", path=r["path"])
            r["sim_pct"] = round(r["similarity"] * 100, 1)

        query_url = url_for("static",
                            filename=f"uploads/{filename}")

        return render_template("results.html",
                               query_url  = query_url,
                               results    = results,
                               top_k      = TOP_K,
                               error      = None)

    except Exception as e:
        logger.error(f"Erreur lors de la recherche : {e}")
        return render_template("results.html",
                               error=f"Erreur lors de la recherche : {type(e).__name__}: {str(e)}")


@app.route("/image")
def serve_image():
    """Sert les images du dataset."""
    path = request.args.get("path", "")
    
    if not path:
        logger.warning("Tentative de service d'image sans chemin")
        return "Chemin manquant", 400
    
    try:
        p = Path(path).resolve()
        
        # Vérifications de sécurité
        if not p.exists():
            logger.warning(f"Image non trouvée : {path}")
            return "Image introuvable", 404
        
        if p.suffix.lower().lstrip(".") not in ALLOWED_EXT:
            logger.warning(f"Format non autorisé : {p.suffix}")
            return "Format non supporté", 400
        
        # Vérifier que le fichier est dans un dossier autorisé
        base_dir = Path(__file__).resolve().parent
        if not str(p).startswith(str(base_dir)):
            logger.error(f"Tentative d'accès non autorisé : {path}")
            return "Accès non autorisé", 403
        
        return send_file(p)
    
    except Exception as e:
        logger.error(f"Erreur lors du service de l'image : {e}")
        return "Erreur serveur", 500


if __name__ == "__main__":
    logger.info("Démarrage du serveur Flask...")
    logger.info("Accédez à l'application à : http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)