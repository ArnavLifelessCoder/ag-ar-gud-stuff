"""EVID-6 cross-validation splits with leakage guard.

Every derived item from one COCO image must land in the same fold.
This module asserts that invariant — it does not trust it.
"""

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def make_folds(items, n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Assign each item to a fold using StratifiedGroupKFold.

    Stratifies on ``state`` and groups by ``base_image_id`` so that no
    COCO image appears in both train and test of any fold.

    Parameters
    ----------
    items : list of Item (or list of dicts with 'state' and 'base_image_id')
    n_splits : int
    seed : int

    Returns
    -------
    np.ndarray of int, shape (len(items),)
        Fold assignment for each item (0 .. n_splits-1).
    """
    if hasattr(items[0], "state"):
        y = np.array([it.state for it in items])
        g = np.array([it.base_image_id for it in items])
    else:
        y = np.array([it["state"] for it in items])
        g = np.array([it["base_image_id"] for it in items])

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(items), -1)
    for k, (_, te) in enumerate(sgkf.split(np.zeros(len(y)), y, g)):
        folds[te] = k

    # Assertions: every item assigned, and no image leaks across folds
    assert (folds >= 0).all(), "Some items were not assigned to any fold"
    for k in range(n_splits):
        tr_g = set(g[folds != k])
        te_g = set(g[folds == k])
        overlap = tr_g & te_g
        assert not overlap, (
            f"LEAK: fold {k} shares {len(overlap)} base images between train and test"
        )

    return folds
