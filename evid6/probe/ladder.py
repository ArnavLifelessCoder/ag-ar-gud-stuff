"""EVID-6 evaluation ladder - all four rungs on identical items.

Rung 1: Zero-shot GENERATED answer, letter-parsed (deployed behaviour)
Rung 2: Few-shot, 8 in-context examples drawn from training folds only
Rung 3: Option-token argmax from logits (same forward pass as probe)
Rung 4: Linear probe on residual-stream activations (cross-validated)

The gap between rung 3 and rung 4 is the headline number.
"""

import sys
import os
import numpy as np
import json
import glob
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

STATES = ["S0", "S1", "S2", "S3", "S4", "S5"]


# ── Activation loading ──────────────────────────────────────────────────────

def load_acts(tag: str, base: str = "acts"):
    """Load cached hidden states from sharded .npy files.

    Returns
    -------
    H : np.ndarray, shape (N, n_layers+1, hidden_dim)
    I : np.ndarray, shape (N,) of item_id strings
    """
    H = np.concatenate(
        [np.load(p) for p in sorted(glob.glob(f"{base}/{tag}/h_*.npy"))]
    )
    I = np.concatenate(
        [np.load(p) for p in sorted(glob.glob(f"{base}/{tag}/i_*.npy"))]
    )
    return H, I


# ── Rung 4: Linear probe ───────────────────────────────────────────────────

def probe_layer(H, y, folds, layer: int, C: float = 1.0):
    """Cross-validated logistic regression probe at one layer.

    Parameters
    ----------
    H : np.ndarray, shape (N, n_layers+1, hidden_dim)
    y : np.ndarray of int, shape (N,) - state indices
    folds : np.ndarray of int, shape (N,) - fold assignments
    layer : int
    C : float - inverse regularization strength

    Returns
    -------
    (mean_accuracy, std_accuracy)
    """
    X = H[:, layer, :].astype(np.float32)
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


def layer_sweep(H, y, folds):
    """Probe every layer and return (layer, mean_acc, std_acc) triples.

    This produces the layer-sweep heatmap in the paper.
    """
    return [(l,) + probe_layer(H, y, folds, l) for l in range(H.shape[1])]


# ── Rung 3: Option-token argmax ────────────────────────────────────────────

def rung3_from_logits(results: list) -> float:
    """Accuracy from option-token argmax. No generation needed.

    Same forward pass as the probe, making the rung 3 vs rung 4 comparison
    perfectly paired.
    """
    y = np.array([STATES.index(r["state"]) for r in results])
    yh = np.array([int(np.argmax(r["probs"])) for r in results])
    return float((y == yh).mean())


# ── Rung 1 & 2: Generated text ─────────────────────────────────────────────

def rung_from_generation(results: list, count_unparsed_as_wrong: bool = True):
    """Accuracy from GENERATED text (rung 1 zero-shot, rung 2 few-shot).

    Rows come from ``run_inference.run_choice_generation`` and carry ``raw``
    (the emitted string) and ``pred`` (the parsed letter, possibly None).

    A reply that commits to no option is a real deployment failure, so by
    default it counts as incorrect.  The unparsed rate is returned alongside
    the accuracy and belongs in the paper next to it - an accuracy computed
    only over parseable replies is a different, flattering quantity.

    Returns
    -------
    dict with accuracy, n, unparsed, unparsed_rate,
    and accuracy_parsed_only (reported for transparency, not headline use).
    """
    if not results:
        return {"accuracy": 0.0, "n": 0, "unparsed": 0,
                "unparsed_rate": 0.0, "accuracy_parsed_only": None}

    correct = parsed_correct = parsed = 0
    for r in results:
        expected = chr(65 + STATES.index(r["state"]))
        pred = r.get("pred")
        if pred is None:
            continue
        parsed += 1
        if pred == expected:
            correct += 1
            parsed_correct += 1

    n = len(results)
    unparsed = n - parsed
    denom = n if count_unparsed_as_wrong else max(parsed, 1)
    return {
        "accuracy": correct / denom,
        "n": n,
        "unparsed": unparsed,
        "unparsed_rate": unparsed / n,
        "accuracy_parsed_only": (parsed_correct / parsed) if parsed else None,
    }


def rung1_zeroshot(results: list) -> dict:
    """Rung 1: zero-shot generated answer. Deployed behaviour."""
    return rung_from_generation(results)


def rung2_fewshot(results: list) -> dict:
    """Rung 2: same, with an 8-example in-context prefix.

    The prefix must be built from training-fold items only - see
    ``prompts.build_fewshot_prefix``.
    """
    return rung_from_generation(results)


def rung1_from_text(results: list) -> float:
    """Backwards-compatible scalar accessor for rung 1.

    NOTE: earlier versions of this function read the option-token argmax,
    which made rung 1 a duplicate of rung 3.  It now requires generation
    rows and returns the headline accuracy only.
    """
    return rung_from_generation(results)["accuracy"]


# ── Per-state breakdown ────────────────────────────────────────────────────

def per_state_accuracy(results: list) -> dict:
    """Compute accuracy broken down by state.

    Returns dict of {state: (accuracy, n_items)}.
    """
    by_state = {}
    for r in results:
        s = r["state"]
        by_state.setdefault(s, []).append(r)

    out = {}
    for s in STATES:
        if s not in by_state:
            continue
        items = by_state[s]
        expected = chr(65 + STATES.index(s))
        correct = sum(1 for r in items if r.get("pred") == expected)
        out[s] = (correct / len(items), len(items))
    return out


# ── Best layer selection ───────────────────────────────────────────────────

def best_layer(sweep_results: list) -> tuple:
    """Find the layer with highest mean accuracy from a layer sweep.

    Returns (layer_index, mean_acc, std_acc).

    WARNING: the accuracy this returns is selected by ``max`` over every layer,
    using the same folds it was scored on, so it is optimistically biased.  On
    pure noise (N=900, 29 layers, 6 classes) the max over layers reads 19.4%
    against a true 16.7% - +2.4 points of free accuracy.  Use this for the
    layer-sweep FIGURE and for choosing which layer to plot.  Do NOT use it as
    the headline rung-4 number; use ``nested_probe`` for that.
    """
    return max(sweep_results, key=lambda x: x[1])


def nested_probe(H, y, folds, C: float = 1.0):
    """Rung 4, with the layer chosen inside the training folds.

    For each outer fold: sweep every layer using only the remaining folds
    (an inner leave-one-fold-out over the training data), pick the layer that
    wins there, then score that layer once on the held-out outer fold.  The
    test fold never participates in the choice, so the resulting accuracy is
    the honest one - it is the number the R4 − R1 gap should be built from.

    Returns
    -------
    dict with
        accuracy      : mean over outer folds (report this)
        std           : spread over outer folds
        layers_chosen : the layer each outer fold selected - if these disagree
                        wildly, the "the probe reads it at layer k" story is
                        weaker than a single number suggests
        selection_bias: how much max-over-layers would have added, i.e. the
                        overstatement you avoided
    """
    folds = np.asarray(folds)
    y = np.asarray(y)
    outer = np.unique(folds)
    accs, chosen = [], []

    for k in outer:
        tr, te = folds != k, folds == k
        inner_folds = folds[tr]
        H_tr, y_tr = H[tr], y[tr]

        # Inner sweep: training folds only.
        inner_sweep = layer_sweep(H_tr, y_tr, inner_folds)
        layer = int(max(inner_sweep, key=lambda x: x[1])[0])
        chosen.append(layer)

        X = H[:, layer, :].astype(np.float32)
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=C))
        clf.fit(X[tr], y[tr])
        accs.append(float((clf.predict(X[te]) == y[te]).mean()))

    honest = float(np.mean(accs))
    flat = best_layer(layer_sweep(H, y, folds))[1]
    return {
        "accuracy": honest,
        "std": float(np.std(accs)),
        "per_fold": accs,
        "layers_chosen": chosen,
        "flat_max_over_layers": float(flat),
        "selection_bias": float(flat - honest),
    }
