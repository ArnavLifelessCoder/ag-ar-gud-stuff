"""EVID-6 paper figures.

All figures are generated programmatically from the results/ directory.
Nothing gets typed into LaTeX by hand.

Figures produced:
  1. Dose-response curve (consistency vs. occlusion fraction / severity)
  2. Ladder comparison bar chart (4 rungs × 3 models)
  3. Learning curve (probe acc vs. n_train)
  4. Layer-sweep heatmap (acc vs. layer, per model)
  5. Steering sweep (AbsAcc vs. OverAbs vs. alpha)
  6. Per-state confusion matrix
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Kaggle
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ── Style ───────────────────────────────────────────────────────────────────

COLORS = {
    "S0": "#2ecc71", "S1": "#e74c3c", "S2": "#3498db",
    "S3": "#f39c12", "S4": "#9b59b6", "S5": "#1abc9c",
}
MODEL_COLORS = {
    "qwen": "#e74c3c",
    "internvl": "#3498db",
    "smolvlm": "#2ecc71",
}

def _setup_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
    })

_setup_style()


def _ensure_dir(out_path: str) -> None:
    """Create the parent directory of ``out_path`` if it has one."""
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)


# ── Figure 1: Dose-response ────────────────────────────────────────────────

def fig_dose_response(results: list, out_path: str = "figures/dose_response.pdf"):
    """Consistency vs. occlusion fraction (S2) and severity (S3).

    Two panels: left = S2 (continuous occl_frac), right = S3 (discrete severity).
    Dashed line = prior-only floor.
    """
    _ensure_dir(out_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # S2: consistency vs occlusion fraction
    s2 = [r for r in results
          if r.get("state") == "S2" and r.get("condition") == "main"
          and r.get("occl_frac") is not None]
    if s2:
        fracs = [r["occl_frac"] for r in s2]
        consist = [1 if r["pred"] == chr(65 + 2) else 0 for r in s2]
        ax1.scatter(fracs, consist, alpha=0.3, s=10, color=COLORS["S2"])
        ax1.set_xlabel("Occlusion fraction")
        ax1.set_ylabel("Correct (S2)")
        ax1.set_title("S2: Occlusion dose-response")

    # Prior-only floor for S2
    po_s2 = [r for r in results if r.get("state") == "S2" and r.get("condition") == "prioronly"]
    if po_s2:
        floor = np.mean([1 if r["pred"] == chr(65 + 2) else 0 for r in po_s2])
        ax1.axhline(floor, ls="--", color="grey", label=f"Prior-only floor ({floor:.2f})")
        ax1.legend()

    # S3: consistency vs severity level
    s3 = [r for r in results if r.get("state") == "S3" and r.get("condition") == "main"]
    by_sev = {}
    for r in s3:
        sev = r.get("severity")
        if sev is None:
            continue
        by_sev.setdefault(sev, []).append(1 if r["pred"] == chr(65 + 3) else 0)
    if by_sev:
        sevs = sorted(by_sev)
        means = [np.mean(by_sev[s]) for s in sevs]
        stds = [np.std(by_sev[s]) / np.sqrt(len(by_sev[s])) for s in sevs]
        ax2.errorbar(sevs, means, yerr=stds, marker="o", color=COLORS["S3"], capsize=4)
        ax2.set_xlabel("Severity level")
        ax2.set_ylabel("Correct (S3)")
        ax2.set_title("S3: Degradation dose-response")
        ax2.set_xticks([1, 2, 3])

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure 2: Ladder comparison ────────────────────────────────────────────

def fig_ladder(rung_data: dict, out_path: str = "figures/ladder.pdf"):
    """Bar chart: 4 rungs × N models.

    Parameters
    ----------
    rung_data : dict
        {model_name: {rung_name: accuracy, ...}, ...}
        e.g. {"qwen": {"R1_zeroshot": 0.25, "R2_fewshot": 0.30,
                        "R3_logits": 0.42, "R4_probe": 0.71}}
    """
    _ensure_dir(out_path)
    models = list(rung_data.keys())
    rungs = ["R1_zeroshot", "R2_fewshot", "R3_logits", "R4_probe"]
    rung_labels = ["R1: Zero-shot", "R2: Few-shot", "R3: Logit argmax", "R4: Probe"]
    x = np.arange(len(rungs))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(models):
        vals = [rung_data[m].get(r, 0) for r in rungs]
        color = MODEL_COLORS.get(m, f"C{i}")
        ax.bar(x + i * width, vals, width, label=m, color=color, alpha=0.85)

    ax.set_ylabel("Accuracy")
    ax.set_title("Evidence-State Classification: Evaluation Ladder")
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(rung_labels)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.axhline(1 / 6, ls=":", color="grey", alpha=0.5, label="Chance (1/6)")
    ax.legend()
    ax.set_ylim(0, 1)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure 3: Learning curve ──────────────────────────────────────────────

def fig_learning_curve(curves: dict, zero_shot_acc: dict = None,
                       out_path: str = "figures/learning_curve.pdf"):
    """Probe accuracy vs. training-set size.

    Parameters
    ----------
    curves : dict
        {model_name: [(n, mean_acc, std_acc), ...], ...}
    zero_shot_acc : dict or None
        {model_name: float} - horizontal dashed lines for zero-shot baseline.
    """
    _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(7, 5))

    for i, (m, curve) in enumerate(curves.items()):
        ns = [c[0] for c in curve]
        means = [c[1] for c in curve]
        stds = [c[2] for c in curve]
        color = MODEL_COLORS.get(m, f"C{i}")
        ax.errorbar(ns, means, yerr=stds, marker="o", label=f"{m} (probe)",
                    color=color, capsize=3)
        if zero_shot_acc and m in zero_shot_acc:
            ax.axhline(zero_shot_acc[m], ls="--", color=color, alpha=0.5,
                       label=f"{m} (zero-shot)")

    ax.set_xlabel("Number of training examples")
    ax.set_ylabel("Probe accuracy (cross-validated)")
    ax.set_title("Learning Curve: Probe Saturation")
    ax.set_xscale("log")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend()
    ax.axhline(1 / 6, ls=":", color="grey", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure 4: Layer-sweep heatmap ──────────────────────────────────────────

def fig_layer_sweep(sweeps: dict, out_path: str = "figures/layer_sweep.pdf"):
    """Heatmap: probe accuracy vs. layer, per model.

    Parameters
    ----------
    sweeps : dict
        {model_name: [(layer, mean_acc, std_acc), ...], ...}
    """
    _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(10, 4))

    for i, (m, sweep) in enumerate(sweeps.items()):
        layers = [s[0] for s in sweep]
        accs = [s[1] for s in sweep]
        color = MODEL_COLORS.get(m, f"C{i}")
        ax.plot(layers, accs, marker=".", label=m, color=color, alpha=0.8)

    ax.set_xlabel("Layer index")
    ax.set_ylabel("Probe accuracy")
    ax.set_title("Layer Sweep: Evidence-State Probe")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend()
    ax.axhline(1 / 6, ls=":", color="grey", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure 5: Steering sweep ──────────────────────────────────────────────

def fig_steering(sweep_results: list, out_path: str = "figures/steering.pdf"):
    """AbsAcc vs OverAbs vs alpha.

    Parameters
    ----------
    sweep_results : list of dict
        Each has 'alpha', 'abs_acc', 'over_abs'.
    """
    _ensure_dir(out_path)
    alphas = [r["alpha"] for r in sweep_results]
    abs_acc = [r["abs_acc"] for r in sweep_results]
    over_abs = [r["over_abs"] for r in sweep_results]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(alphas, abs_acc, "o-", label="Abstention accuracy", color="#2ecc71")
    ax.plot(alphas, over_abs, "s--", label="Over-abstention rate", color="#e74c3c")
    ax.set_xlabel("Steering strength (α)")
    ax.set_ylabel("Rate")
    ax.set_title("Activation Steering: Insufficiency Direction")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.legend()
    ax.axhline(0.5, ls=":", color="grey", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure 6: Confusion matrix ────────────────────────────────────────────

def fig_confusion(results: list, model_name: str = "",
                  out_path: str = "figures/confusion.pdf"):
    """Per-state confusion matrix from forced-choice results.

    Parameters
    ----------
    results : list of dict
        Each has 'state' and 'pred' (letter).
    """
    _ensure_dir(out_path)
    STATES = ["S0", "S1", "S2", "S3", "S4", "S5"]
    n = len(STATES)
    cm = np.zeros((n, n), dtype=int)
    for r in results:
        true_idx = STATES.index(r["state"])
        pred_idx = ord(r["pred"]) - 65
        if 0 <= pred_idx < n:
            cm[true_idx, pred_idx] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(STATES)
    ax.set_yticklabels(STATES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix{' - ' + model_name if model_name else ''}")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=9)

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure 7: Consistency dose-response (the paper's core plot) ────────────

def fig_consistency(summaries: dict, out_path: str = "figures/consistency.pdf"):
    """Self-consistency against the clean-image answer, per model.

    Left panel: consistency by state (main condition) with the prior-only
    floor and the S0 ceiling drawn in, so the reader can see immediately how
    much of the drop is real signal loss and how much was never evidence-based.
    Right panel: the S3 severity curve, which is where P2 lives.

    Parameters
    ----------
    summaries : dict {model_name: consistency.summarise(...) output}
    """
    _ensure_dir(out_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    states = ["S0", "S1", "S2", "S3", "S4"]
    x = np.arange(len(states))
    width = 0.8 / max(len(summaries), 1)

    for i, (m, summ) in enumerate(summaries.items()):
        color = MODEL_COLORS.get(m, f"C{i}")
        bsc = summ.get("by_state_condition", {})
        vals = [(bsc.get(f"{s}|main") or (0, 0))[0] or 0 for s in states]
        ax1.bar(x + i * width, vals, width, label=m, color=color, alpha=0.85)

        floor = (summ.get("prior_floor_pooled") or (None,))[0]
        if floor is not None:
            ax1.axhline(floor, ls="--", color=color, alpha=0.6,
                        label=f"{m} prior floor")

        curve = summ.get("dose_S3_severity") or []
        if curve:
            ax2.errorbar([c[0] for c in curve], [c[1] for c in curve],
                         marker="o", color=color, label=m, capsize=3)

    ax1.set_xticks(x + width * (len(summaries) - 1) / 2)
    ax1.set_xticklabels(states)
    ax1.set_ylabel("Consistency with clean-image answer")
    ax1.set_title("Does the answer survive evidence removal?")
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.set_ylim(0, 1)
    ax1.legend(fontsize=7)

    ax2.set_xlabel("S3 severity")
    ax2.set_ylabel("Consistency")
    ax2.set_title("P2: does degradation attenuate smoothly?")
    ax2.set_xticks([1, 2, 3])
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure 8: Abstention trade-off ────────────────────────────────────────

def fig_abstain(summaries: dict, out_path: str = "figures/abstain.pdf"):
    """AbsAcc against OverAbs, with the always-abstain baseline marked.

    A point above the baseline line but far right is a model that got its
    accuracy by refusing to answer anything, which is the failure mode this
    plot exists to expose.

    Parameters
    ----------
    summaries : dict {model_name: abstain.summarise(...) output}
    """
    _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(6.5, 5))

    for i, (m, a) in enumerate(summaries.items()):
        if a.get("AbsAcc") is None:
            continue
        color = MODEL_COLORS.get(m, f"C{i}")
        ax.scatter(a.get("OverAbs") or 0, a["AbsAcc"], s=140, color=color,
                   label=m, zorder=3, edgecolor="white", linewidth=1.5)
        base = a.get("always_abstain_baseline")
        if base is not None and i == 0:
            ax.axhline(base, ls="--", color="grey", alpha=0.7,
                       label=f"always-abstain baseline ({base:.0%})")

    ax.set_xlabel("Over-abstention (refuses an answerable item)")
    ax.set_ylabel("Abstention accuracy")
    ax.set_title("Abstention: accuracy is meaningless without the false-alarm rate")
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved: {out_path}")
