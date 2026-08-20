"""EVID-6 probe generalization.

Two questions the reviewer will ask about any probe result:

  1. Does the probe generalize across models?  Train on model A's residual
     stream, test on model B's.  Only meaningful when the two share items,
     and only possible after a linear map between hidden spaces - different
     models have different widths, so we align by fitting the probe in a
     shared low-dimensional space (PCA) rather than pretending the axes match.

  2. Does the probe generalize across content?  Hold out whole COCO
     categories, or whole intervention severities, and see whether the
     decision boundary survives. This is the harder and more honest test:
     a probe that only works within-category has learned the objects, not
     the evidence state.

Both return the same shape of result so they can go in one table.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA


def _clf(C=1.0):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=C))


# ── 1. Cross-model transfer ────────────────────────────────────────────────

def align_and_transfer(HA, IA, HB, IB, y_by_item, layer_a, layer_b,
                       n_components: int = 64, C: float = 1.0, seed: int = 0):
    """Train on model A's activations, test on model B's.

    The two hidden spaces are unrelated bases of different widths, so we
    cannot apply A's weight vector to B directly.  Instead both are projected
    into a ``n_components``-dimensional PCA space fitted on the SHARED items,
    and the probe is trained in A's projection and evaluated in B's.

    This is a weak form of transfer and should be labelled as such: it asks
    whether the same low-dimensional structure exists in both models, not
    whether a literal direction carries over.

    Parameters
    ----------
    HA, HB : np.ndarray, (N, L+1, D) per model
    IA, IB : np.ndarray of item_id strings, aligned with HA / HB
    y_by_item : dict item_id -> int state index
    layer_a, layer_b : int - usually each model's own best layer

    Returns
    -------
    dict with within_a, within_b, a_to_b, b_to_a, n_shared, chance
    """
    shared = [i for i in IA if i in set(IB) and i in y_by_item]
    if len(shared) < 50:
        return {"error": f"only {len(shared)} shared items, need >= 50"}

    pos_a = {iid: k for k, iid in enumerate(IA)}
    pos_b = {iid: k for k, iid in enumerate(IB)}
    XA = HA[[pos_a[i] for i in shared], layer_a, :].astype(np.float32)
    XB = HB[[pos_b[i] for i in shared], layer_b, :].astype(np.float32)
    y = np.array([y_by_item[i] for i in shared])

    k = int(min(n_components, XA.shape[1], XB.shape[1], len(shared) // 3))
    PA = PCA(n_components=k, random_state=seed).fit_transform(XA)
    PB = PCA(n_components=k, random_state=seed).fit_transform(XB)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(shared))
    cut = len(shared) // 2
    tr, te = idx[:cut], idx[cut:]

    out = {"n_shared": len(shared), "k_components": k,
           "chance": 1.0 / len(set(y))}
    for name, Xtr, Xte in [("within_a", PA, PA), ("within_b", PB, PB),
                           ("a_to_b", PA, PB), ("b_to_a", PB, PA)]:
        m = _clf(C).fit(Xtr[tr], y[tr])
        out[name] = float((m.predict(Xte[te]) == y[te]).mean())
    out["transfer_gap"] = out["within_a"] - out["a_to_b"]
    return out


# ── 2. Held-out content generalization ─────────────────────────────────────

def leave_group_out(H, y, groups, layer, C: float = 1.0, min_test: int = 20):
    """Leave-one-group-out probe accuracy.

    ``groups`` is any array of labels you want to hold out wholesale:
    COCO category names (does the probe survive unseen objects?), severity
    levels (does it survive unseen intervention doses?), or question
    templates.

    Returns
    -------
    dict with per_group accuracies, mean, std, and n_groups.
    """
    X = H[:, layer, :].astype(np.float32)
    groups = np.asarray(groups)
    per, skipped = {}, 0
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if te.sum() < min_test or len(np.unique(y[tr])) < len(np.unique(y)):
            skipped += 1
            continue
        m = _clf(C).fit(X[tr], y[tr])
        per[str(g)] = float((m.predict(X[te]) == y[te]).mean())
    if not per:
        return {"error": "no group had enough test items", "skipped": skipped}
    vals = list(per.values())
    return {
        "per_group": per,
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "worst": min(per, key=per.get),
        "worst_acc": min(vals),
        "n_groups": len(per),
        "skipped": skipped,
    }


def severity_extrapolation(H, y, severities, layer, C: float = 1.0):
    """Train on mild degradation, test on severe (and vice versa).

    If the probe trained on severity 1-2 still reads severity 3 correctly,
    the representation is about the *kind* of evidence failure rather than
    the amount of blur. That is the claim the taxonomy needs.
    """
    X = H[:, layer, :].astype(np.float32)
    sev = np.asarray(severities)
    have = sev != None                                    # noqa: E711
    if have.sum() < 60:
        return {"error": "too few items carry a severity label"}

    out = {}
    for name, tr_mask, te_mask in [
        ("mild_to_severe", np.isin(sev, [1, 2]), sev == 3),
        ("severe_to_mild", sev == 3, np.isin(sev, [1, 2])),
    ]:
        tr = tr_mask & have
        te = te_mask & have
        if te.sum() < 20 or len(np.unique(y[tr])) < 2:
            out[name] = None
            continue
        m = _clf(C).fit(X[tr], y[tr])
        out[name] = {"acc": float((m.predict(X[te]) == y[te]).mean()),
                     "n_train": int(tr.sum()), "n_test": int(te.sum())}
    return out
