"""EVID-6 activation steering (gated on 20 Aug).

Computes a "this-is-unanswerable" direction via difference of means and
injects it at inference time through a forward hook.

The key result: does abstention rise faster than over-abstention?
If both rise together, the direction is a generic hedging knob and not
an evidence signal - worth one honest paragraph.
"""

import numpy as np
import torch


def insufficiency_direction(H, y, layer: int) -> torch.Tensor:
    """Compute the difference-of-means direction: not-S0 minus S0.

    Parameters
    ----------
    H : np.ndarray, shape (N, n_layers+1, hidden_dim)
    y : np.ndarray of int, shape (N,) - state indices (0 = S0)
    layer : int

    Returns
    -------
    torch.Tensor, shape (hidden_dim,), dtype float16, L2-normalised.
    """
    X = H[:, layer, :].astype(np.float32)
    v = X[y != 0].mean(0) - X[y == 0].mean(0)
    return torch.tensor(v / np.linalg.norm(v), dtype=torch.float16)


def attach(model, layer: int, vec: torch.Tensor, alpha: float):
    """Attach a forward hook that steers the residual stream at the last token.

    NOTE: ``print(model)`` first and confirm the layer path.
    It differs per architecture:
      - Qwen2.5-VL: model.model.layers[layer]
      - InternVL3:  model.language_model.model.layers[layer]

    Parameters
    ----------
    model : nn.Module
    layer : int
        Transformer block index.
    vec : torch.Tensor
        Steering direction (unit norm).
    alpha : float
        Steering strength.  Sweep over [-4, -2, -1, 0, 1, 2, 4].

    Returns
    -------
    hook handle (call .remove() to detach)
    """
    # Try common paths for the transformer block
    block = None
    for path in [
        lambda: model.model.layers[layer],                         # Qwen-style
        lambda: model.language_model.model.layers[layer],          # InternVL-style
        lambda: model.model.language_model.model.layers[layer],    # Some wrappers
    ]:
        try:
            block = path()
            break
        except (AttributeError, IndexError):
            continue

    if block is None:
        raise RuntimeError(
            f"Cannot find transformer block at layer {layer}. "
            f"Run print(model) and update the path in steer.py."
        )

    def hook(mod, args, out):
        h = out[0] if isinstance(out, tuple) else out
        h[:, -1, :] = h[:, -1, :] + alpha * vec.to(h.device)
        return (h,) + out[1:] if isinstance(out, tuple) else h

    return block.register_forward_hook(hook)


def sweep_alphas(model, proc, items, layer, vec, opt_ids, prompt_fn,
                 alphas=(-4, -2, -1, 0, 1, 2, 4)):
    """Sweep steering strength and measure AbsAcc vs OverAbs.

    Parameters
    ----------
    items : list of dict
        Subset of test items.
    alphas : tuple of float
        Steering strengths to try.

    Returns
    -------
    list of dict: {alpha, abs_acc, over_abs, n}
    """
    from run_inference import score_one   # evid6/eval is on sys.path

    results = []
    for a in alphas:
        handle = attach(model, layer, vec, a)
        correct_abs, over_abs, total = 0, 0, 0
        for it in items:
            probs, _ = score_one(proc, model, it,
                                 prompt_fn(q=it["question"]), opt_ids,
                                 want_hidden=False)
            pred = int(probs.argmax())
            true_state = int(it["state"][1])  # S0->0, S1->1, ...
            # Abstention = predicting any non-S0 state
            pred_abs = pred != 0
            true_abs = true_state != 0
            if pred_abs == true_abs:
                correct_abs += 1
            if pred_abs and not true_abs:
                over_abs += 1
            total += 1
        handle.remove()
        results.append({
            "alpha": a,
            "abs_acc": correct_abs / max(total, 1),
            "over_abs": over_abs / max(total, 1),
            "n": total,
        })
    return results
