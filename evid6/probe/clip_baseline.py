"""EVID-6 CLIP baseline.

Mandatory sanity check: if a CLIP linear probe matches the VLM internal
probe, the VLM probe is not learning anything beyond what the vision
encoder already exposes — and the paper should rest on the rung 1 vs rung 4
gap, not absolute probe numbers.

Runs in week one on CPU.  CLIP ViT-B/32 is small enough.
"""

import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def clip_features(paths: list, bs: int = 32, model_id: str = "openai/clip-vit-base-patch32"):
    """Extract L2-normalised CLIP image features for a batch of image paths.

    Parameters
    ----------
    paths : list of str
        Paths to images.
    bs : int
        Batch size.
    model_id : str
        CLIP model to use.

    Returns
    -------
    np.ndarray, shape (len(paths), feature_dim)
        L2-normalised image features.
    """
    m = CLIPModel.from_pretrained(model_id).eval()
    p = CLIPProcessor.from_pretrained(model_id)
    feats = []
    with torch.inference_mode():
        for i in range(0, len(paths), bs):
            batch_paths = paths[i:i + bs]
            ims = [Image.open(q).convert("RGB") for q in batch_paths]
            inputs = p(images=ims, return_tensors="pt")
            f = m.get_image_features(**inputs)
            feats.append(torch.nn.functional.normalize(f, dim=-1).numpy())
    return np.concatenate(feats)


def clip_probe(paths, y, folds, C: float = 1.0):
    """Run the same logistic regression probe on CLIP features.

    Uses the identical cross-validation setup as the VLM probe for fair
    comparison.

    Parameters
    ----------
    paths : list of str
        Image paths, aligned with y and folds.
    y : np.ndarray of int
        State labels.
    folds : np.ndarray of int
        Fold assignments.
    C : float
        Regularization strength.

    Returns
    -------
    (mean_accuracy, std_accuracy)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    X = clip_features(paths).astype(np.float32)
    acc = []
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=C),
        )
        clf.fit(X[tr], y[tr])
        acc.append((clf.predict(X[te]) == y[te]).mean())
    return float(np.mean(acc)), float(np.std(acc))
