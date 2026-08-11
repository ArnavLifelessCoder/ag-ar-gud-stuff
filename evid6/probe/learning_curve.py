"""EVID-6 learning curve analysis.

The plot that answers "your gap is just supervision."

If the probe saturates at n=25 training examples while zero-shot behaviour
sits near chance, the "accessible but unused" claim is made — and made fairly.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from schema import STATES


def learning_curve(H, y, folds, layer: int,
                   ns=(10, 25, 50, 100, 250, 500, 1000),
                   seed: int = 0):
    """Compute cross-validated accuracy at varying training-set sizes.

    Parameters
    ----------
    H : np.ndarray, shape (N, n_layers+1, hidden_dim)
    y : np.ndarray of int, shape (N,) — state indices
    folds : np.ndarray of int, shape (N,) — fold assignments
    layer : int — which layer to probe
    ns : tuple of int — training sizes to try
    seed : int

    Returns
    -------
    list of (n_train, mean_accuracy, std_accuracy)
    """
    rng = np.random.default_rng(seed)
    X = H[:, layer, :].astype(np.float32)
    curve = []

    for n in ns:
        accs = []
        for k in np.unique(folds):
            tr = np.where(folds != k)[0]
            te = folds == k
            if n > len(tr):
                continue
            sub = rng.choice(tr, n, replace=False)
            # Need at least one example per state
            if len(np.unique(y[sub])) < len(STATES):
                continue
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000),
            )
            clf.fit(X[sub], y[sub])
            accs.append((clf.predict(X[te]) == y[te]).mean())
        if accs:
            curve.append((n, float(np.mean(accs)), float(np.std(accs))))

    return curve
