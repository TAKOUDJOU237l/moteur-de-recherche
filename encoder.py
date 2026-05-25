# encoder.py
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224

if torch.cuda.is_available():
    logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
else:
    logger.info("Using CPU for image encoding")

# Transformations identiques à l'entraînement ImageNet
encode_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])


class ImageEncoder(nn.Module):
    """
    Encodeur basé sur ResNet50 pré-entraîné.
    Retourne un vecteur de 2048 dimensions par image.
    """
    def __init__(self):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # On supprime la couche de classification finale
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])

    def forward(self, x):
        feat = self.encoder(x)           # (B, 2048, 1, 1)
        return feat.flatten(start_dim=1) # (B, 2048)


def _l2_normalize(array: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def load_encoder():
    """Charge et retourne l'encodeur en mode évaluation."""
    model = ImageEncoder().to(DEVICE)
    model.eval()
    return model


def encode_image(img_path: str, model: ImageEncoder) -> np.ndarray:
    """
    Encode une image depuis son chemin.
    Retourne un vecteur numpy normalisé (2048,).
    """
    try:
        img = Image.open(img_path).convert("RGB")
    except FileNotFoundError:
        raise ValueError(f"Fichier introuvable : {img_path}")
    except Exception as e:
        raise ValueError(f"Impossible d'ouvrir l'image {img_path}: {type(e).__name__}: {e}")

    try:
        tensor = encode_transform(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            emb = model(tensor).cpu().numpy()
        return _l2_normalize(emb)[0]
    except Exception as e:
        logger.error(f"Erreur lors de l'encodage de {img_path}: {e}")
        raise


def encode_batch(img_paths: list, model: ImageEncoder,
                 batch_size: int = 32) -> np.ndarray:
    """
    Encode une liste d'images par batch.
    Retourne un tableau numpy (N, 2048) normalisé.
    """
    from torch.utils.data import Dataset, DataLoader

    class PathDataset(Dataset):
        def __init__(self, paths):
            self.paths = paths
            self.corrupted = set()
        
        def __len__(self):
            return len(self.paths)
        
        def __getitem__(self, idx):
            path = self.paths[idx]
            try:
                img = Image.open(path).convert("RGB")
                # Valider que l'image a des dimensions correctes
                if img.size[0] < 10 or img.size[1] < 10:
                    raise ValueError(f"Image trop petite: {img.size}")
                return encode_transform(img), path
            except Exception as e:
                logger.warning(f"Image corrompue ignorée: {path} ({e})")
                self.corrupted.add(path)
                # Retourner une image de remplacement
                img = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))
                return encode_transform(img), path

    dataset = PathDataset(img_paths)
    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=0)
    all_embs = []
    model.eval()
    
    with torch.no_grad():
        for imgs, paths in loader:
            imgs = imgs.to(DEVICE)
            embs = model(imgs).cpu().numpy()
            all_embs.append(embs)

    if dataset.corrupted:
        logger.info(f"{len(dataset.corrupted)} images corrompues détectées")
    
    embeddings = np.vstack(all_embs)
    return _l2_normalize(embeddings)