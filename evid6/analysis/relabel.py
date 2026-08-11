"""EVID-6 blind self-relabel harness (plan section 6, layer 2).

There is no annotation budget, so the human check is you, relabelling a
sample of your own items from scratch after enough time has passed that you
do not remember them.  This is intra-annotator agreement, which is weaker
than inter-annotator agreement and must be described as such in Limitations.

The harness exists to make the check honest:
  - the sample is drawn with a fixed seed and saved, so you cannot quietly
    redraw it after seeing a bad result;
  - the sheet you fill in carries no state labels and is shuffled;
  - the gold labels are written to a separate file you are not meant to open
    until you have finished;
  - the export records the timestamp, and scoring warns if you relabelled
    sooner than the cooling-off period.

Workflow
--------
    export_sheet(items, out_dir, n=100, seed=0)      # then wait >= 48 h
    # fill in the 'label' column of relabel_sheet.csv
    report = score_sheet(out_dir)
"""

import os
import csv
import json
import time
import random
import numpy as np

COOLING_OFF_HOURS = 48


def export_sheet(items, out_dir: str, n: int = 100, seed: int = 0,
                 conditions=("main",)):
    """Write a blind relabelling sheet plus a sealed answer key.

    Parameters
    ----------
    items : list of Item or dict
    out_dir : str — directory to write into
    n : int — sample size
    conditions : tuple — which conditions are eligible (main by default)

    Returns
    -------
    path to the sheet the annotator fills in.
    """
    os.makedirs(out_dir, exist_ok=True)

    def get(it, k):
        return it[k] if isinstance(it, dict) else getattr(it, k)

    pool = [it for it in items if get(it, "condition") in conditions]
    rng = random.Random(seed)
    sample = rng.sample(pool, min(n, len(pool)))
    rng.shuffle(sample)

    sheet = os.path.join(out_dir, "relabel_sheet.csv")
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "item_id", "image_path", "question", "label"])
        for i, it in enumerate(sample):
            w.writerow([i, get(it, "item_id"), get(it, "image_path"),
                        get(it, "question"), ""])

    key = os.path.join(out_dir, "relabel_key.json")
    with open(key, "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "exported_at": time.time(),
            "exported_at_human": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cooling_off_hours": COOLING_OFF_HOURS,
            "gold": {get(it, "item_id"): get(it, "state") for it in sample},
        }, f, indent=2)

    print(f"Sheet:  {sheet}   ({len(sample)} items)")
    print(f"Key:    {key}   (do not open until the sheet is filled in)")
    print(f"Wait at least {COOLING_OFF_HOURS} h before relabelling.")
    return sheet


def cohens_kappa(a, b, labels=None):
    """Cohen's kappa between two label sequences."""
    a, b = list(a), list(b)
    labels = sorted(set(a) | set(b)) if labels is None else list(labels)
    idx = {l: i for i, l in enumerate(labels)}
    k = len(labels)
    m = np.zeros((k, k))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    n = m.sum()
    if n == 0:
        return 0.0
    po = np.trace(m) / n
    pe = float((m.sum(0) * m.sum(1)).sum()) / (n * n)
    return float((po - pe) / (1 - pe)) if pe < 1 else 1.0


def score_sheet(out_dir: str, per_state_threshold: float = 0.6):
    """Score a filled-in sheet against the sealed key.

    ``per_state_threshold`` implements the plan's 14 Aug kill criterion:
    any state whose agreement falls below it should be dropped from the
    benchmark, and the drop reported.
    """
    sheet = os.path.join(out_dir, "relabel_sheet.csv")
    key_path = os.path.join(out_dir, "relabel_key.json")
    with open(key_path, encoding="utf-8") as f:
        key = json.load(f)
    gold = key["gold"]

    elapsed_h = (time.time() - key["exported_at"]) / 3600.0
    warnings = []
    if elapsed_h < key.get("cooling_off_hours", COOLING_OFF_HOURS):
        warnings.append(
            f"relabelled after only {elapsed_h:.1f} h, below the "
            f"{key.get('cooling_off_hours')} h cooling-off period — "
            f"agreement is inflated by recall, say so in the paper"
        )

    mine, theirs, missing = [], [], 0
    with open(sheet, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lab = (row.get("label") or "").strip().upper()
            if not lab:
                missing += 1
                continue
            iid = row["item_id"]
            if iid not in gold:
                continue
            mine.append(lab)
            theirs.append(gold[iid])

    if not mine:
        return {"error": "no labels filled in", "warnings": warnings}

    overall = float(np.mean([a == b for a, b in zip(mine, theirs)]))
    kappa = cohens_kappa(theirs, mine)

    per_state, drops = {}, []
    for st in sorted(set(theirs)):
        pairs = [(t, m) for t, m in zip(theirs, mine) if t == st]
        acc = float(np.mean([t == m for t, m in pairs]))
        per_state[st] = {"agreement": acc, "n": len(pairs)}
        if acc < per_state_threshold:
            drops.append(st)

    # Where did the disagreements go?
    confusion = {}
    for t, m in zip(theirs, mine):
        if t != m:
            confusion[f"{t}->{m}"] = confusion.get(f"{t}->{m}", 0) + 1

    return {
        "n_scored": len(mine),
        "n_missing": missing,
        "hours_elapsed": round(elapsed_h, 1),
        "overall_agreement": overall,
        "cohens_kappa": kappa,
        "per_state": per_state,
        "states_below_threshold": drops,
        "confusion": dict(sorted(confusion.items(), key=lambda x: -x[1])),
        "warnings": warnings,
        "verdict": ("drop " + ", ".join(drops) + " (14 Aug kill criterion)"
                    if drops else "all states clear the agreement threshold"),
    }
