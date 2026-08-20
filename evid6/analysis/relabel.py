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
    out_dir : str - directory to write into
    n : int - sample size
    conditions : tuple - which conditions are eligible (main by default)

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


def export_html_sheet(out_dir: str, image_roots=(), max_px: int = 420):
    """Render the sheet as one self-contained HTML file you can label offline.

    The CSV records the image paths from the session that built the dataset
    (``/kaggle/working/evid6/images/...``), which do not exist anywhere else -
    so the CSV alone cannot be labelled. This embeds each image as a data URI,
    giving a single file that works on any machine with no image directory
    beside it.

    Carries no state labels and preserves the shuffled order, so it is exactly
    as blind as the CSV. Fill the labels into the CSV; this is only for looking.

    Parameters
    ----------
    image_roots : sequence of str
        Directories to search for each image by basename, e.g. NB1's output.
    """
    import base64
    from io import BytesIO

    sheet = os.path.join(out_dir, "relabel_sheet.csv")
    with open(sheet, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def _find(path):
        if os.path.isfile(path):
            return path
        name = os.path.basename(path)
        for root in image_roots:
            for dirpath, _dirnames, files in os.walk(root):
                if name in files:
                    return os.path.join(dirpath, name)
        return None

    from PIL import Image
    parts, missing = [], 0
    for r in rows:
        found = _find(r["image_path"])
        if found is None:
            missing += 1
            img_tag = "<p style='color:#b00'>image not found</p>"
        else:
            im = Image.open(found).convert("RGB")
            im.thumbnail((max_px, max_px))
            buf = BytesIO()
            im.save(buf, format="JPEG", quality=82)
            b64 = base64.b64encode(buf.getvalue()).decode()
            img_tag = f"<img src='data:image/jpeg;base64,{b64}'>"
        parts.append(
            f"<section><h2>row {r['row']}</h2>"
            f"<p class='q'>{r['question']}</p>{img_tag}</section>"
        )

    html = f"""<!doctype html><meta charset="utf-8">
<title>EVID-6 blind relabel - {len(rows)} items</title>
<style>
 body {{ font: 15px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 900px; }}
 section {{ border-bottom: 1px solid #ddd; padding: 1.2rem 0; }}
 h2 {{ margin: 0 0 .2rem; font-size: 1rem; color: #555; }}
 .q {{ margin: 0 0 .6rem; font-weight: 600; }}
 img {{ max-width: 100%; border: 1px solid #e5e5e5; }}
 ol {{ color: #333; }} code {{ background: #f4f4f4; padding: 0 .2em; }}
</style>
<h1>Blind relabel - {len(rows)} items</h1>
<p>For each row, decide which evidence state the image and question show, and
type it into the <code>label</code> column of <code>relabel_sheet.csv</code>
against the same row number. No state labels appear here.</p>
<ol>
 <li><b>S0</b> answerable - the referent is visible and unambiguous</li>
 <li><b>S1</b> out of frame - it exists in the scene but is not in the crop</li>
 <li><b>S2</b> occluded - inside the frame, blocked by something in front</li>
 <li><b>S3</b> sub-resolution - visible and unblocked, but too small, blurred or dark</li>
 <li><b>S4</b> ambiguous - more than one object matches the description</li>
 <li><b>S5</b> false premise - the named thing is not in this scene at all</li>
</ol>
<p><b>Do not open <code>relabel_key.json</code> until every row is filled in.</b></p>
{''.join(parts)}"""

    out = os.path.join(out_dir, "relabel_sheet.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Relabel sheet: {out}  ({len(rows)} items"
          + (f", {missing} images not found)" if missing else ")"))
    return out


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
            f"{key.get('cooling_off_hours')} h cooling-off period - "
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
