"""EVID-6 CLIP baseline.

Mandatory sanity check: if a CLIP linear probe matches the VLM internal
probe, the VLM probe is not learning anything beyond what the vision
encoder already exposes - and the paper should rest on the rung 1 vs rung 4
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
            f = _image_embeds(m, inputs)
            feats.append(torch.nn.functional.normalize(f, dim=-1).numpy())
    return np.concatenate(feats)


def _image_embeds(model, inputs):
    """Projected CLIP image embedding, across transformers versions.

    ``get_image_features`` returned a plain Tensor through transformers 4.x,
    but 5.x returns a ``BaseModelOutputWithPooling``.  Passing that straight to
    ``F.normalize`` raises ``AttributeError: 'BaseModelOutputWithPooling'
    object has no attribute 'norm'`` -- which is exactly how a 5.7-hour NB4 run
    died 40 seconds after the probes had finished.

    Returns the 512-d projected embedding either way, so the baseline is the
    same quantity on both versions.
    """
    out = model.get_image_features(**inputs)
    if torch.is_tensor(out):
        return out

    proj = model.visual_projection
    in_dim = getattr(proj, "in_features", None)      # 768 for ViT-B/32
    out_dim = getattr(proj, "out_features", None)    # 512

    def _project_if_needed(t):
        """Apply visual_projection only when the tensor is pre-projection.

        transformers 5.x hands back an output object whose ``pooler_output``
        is ALREADY the 512-d projected embedding, not the 768-d pooled vision
        state. Projecting it again raises
        "mat1 and mat2 shapes cannot be multiplied (32x512 and 768x512)".
        Decide by width rather than assuming either convention.
        """
        if in_dim is not None and t.shape[-1] == in_dim:
            return proj(t)
        if out_dim is not None and t.shape[-1] == out_dim:
            return t                                  # already projected
        return proj(t)                                # unknown: try, and fail loudly

    # 5.x: unwrap, projecting only if we were handed the pre-projection state.
    embeds = getattr(out, "image_embeds", None)
    if torch.is_tensor(embeds):
        return embeds
    pooled = getattr(out, "pooler_output", None)
    if torch.is_tensor(pooled):
        return _project_if_needed(pooled)
    hidden = getattr(out, "last_hidden_state", None)
    if torch.is_tensor(hidden):
        return _project_if_needed(hidden[:, 0])
    raise TypeError(
        f"CLIPModel.get_image_features returned {type(out).__name__} with no "
        "recognisable image embedding; update _image_embeds for this "
        "transformers version."
    )


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
