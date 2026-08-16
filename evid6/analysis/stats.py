"""EVID-6 statistical helpers.

Everything in the paper regenerates from ``results/`` through this module.
Nothing gets typed into LaTeX by hand.
"""

import numpy as np
from statsmodels.stats.contingency_tables import mcnemar


def boot_ci(x, n: int = 2000, seed: int = 0):
    """Bootstrap confidence interval for the mean.

    Returns (mean, lower_2.5%, upper_97.5%).
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    bs = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return (
        float(x.mean()),
        float(np.percentile(bs, 2.5)),
        float(np.percentile(bs, 97.5)),
    )


def paired_test(a, b):
    """McNemar test on two boolean correctness arrays over the same items.

    Parameters
    ----------
    a, b : array-like of bool
        Correctness indicators from two conditions on identical items.

    Returns
    -------
    float
        p-value from McNemar's test (chi-square with continuity correction).
    """
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    t = np.array([
        [int((a & b).sum()),  int((a & ~b).sum())],
        [int((~a & b).sum()), int((~a & ~b).sum())],
    ])
    return float(mcnemar(t, exact=False, correction=True).pvalue)


def accuracy_by_state(results, states=None):
    """Compute per-state accuracy from a list of result dicts.

    Each dict must have 'state' and 'pred' keys.
    Returns dict of {state: (accuracy, n)}.
    """
    if states is None:
        states = sorted(set(r["state"] for r in results))
    out = {}
    for s in states:
        subset = [r for r in results if r["state"] == s]
        if not subset:
            continue
        state_idx = int(s[1])  # S0->0, S1->1, ...
        correct = sum(1 for r in subset if r["pred"] == chr(65 + state_idx))
        out[s] = (correct / len(subset), len(subset))
    return out


def pair_main_vs_control(results):
    """Pair each s0ctrl row with the main row it controls for.

    The control item records ``parent_item_id``, so the pairing is exact.
    Pairing on ``base_image_id`` instead is wrong: one COCO image can yield
    several main items across states, so the match becomes arbitrary.

    Returns
    -------
    (a, b, n) : np.ndarray of bool, np.ndarray of bool, int
        ``a`` is main-item correctness, ``b`` is the paired control's, both
        scored against each row's own expected letter.  Feed straight into
        ``paired_test``.
    """
    main = {r["item_id"]: r for r in results if r.get("condition") == "main"}
    a, b = [], []
    for r in results:
        if r.get("condition") != "s0ctrl":
            continue
        parent = r.get("parent_item_id")
        if parent is None or parent not in main:
            continue
        m = main[parent]
        exp_m = chr(65 + int(str(m["state"])[1]))
        exp_c = chr(65 + int(str(r["state"])[1]))
        a.append(m.get("pred") == exp_m)
        b.append(r.get("pred") == exp_c)
    return np.array(a, dtype=bool), np.array(b, dtype=bool), len(a)


def pair_on_field(results_a, results_b, key="item_id", field="consistent"):
    """Align two result lists on a shared key and return paired boolean arrays.

    Use for main-vs-prior-only consistency comparisons, where the two rows
    live in different conditions but share a reference group.
    """
    idx_b = {r[key]: r for r in results_b}
    a, b = [], []
    for r in results_a:
        o = idx_b.get(r[key])
        if o is None:
            continue
        a.append(bool(r[field]))
        b.append(bool(o[field]))
    return np.array(a, dtype=bool), np.array(b, dtype=bool), len(a)
