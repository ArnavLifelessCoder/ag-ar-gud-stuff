"""EVID-6 Tier B: VizWiz items.

Tier A (COCO) is synthetic — we control the intervention exactly, which is
what makes the dose-response analysis possible, but every image is a clean
photograph that we then broke on purpose.  Tier B exists to answer the
obvious objection: does any of this hold on images that were genuinely hard
to begin with?

VizWiz photographs are taken by blind and low-vision users, so unanswerable
questions there fail for real reasons — the camera moved, the object is out
of frame, the lighting is gone.  The taxonomy either applies to them or it
does not.

Important honesty constraint: VizWiz ships an ``answerable`` flag but NOT the
*reason* a question is unanswerable, which is exactly the label this paper
needs.  So Tier B is hand-sorted, 200 items, by you.  This module does the
mechanical parts (loading, filtering, stratified sampling, sheet export,
merging your labels back) and deliberately does not guess the state for you.

Workflow
--------
    anns  = load_annotations(VIZWIZ_JSON)
    cands = candidates(anns, IMG_DIR, n=200, seed=0)
    export_labelling_sheet(cands, "/kaggle/working/tierb")
    # hand-sort the 'state' column, then:
    items = load_labelled_sheet("/kaggle/working/tierb", IMG_DIR)
"""

import os
import csv
import json
import random

from schema import Item, STATES

# Typical Kaggle layout; override in the notebook.
VIZWIZ_ROOT = "/kaggle/input/vizwiz-vqa"
VAL_JSON = f"{VIZWIZ_ROOT}/Annotations/val.json"
VAL_IMG_DIR = f"{VIZWIZ_ROOT}/val"


def load_annotations(path: str = VAL_JSON):
    """Load the VizWiz VQA annotation file.

    Handles both the list-of-dicts export and the {'annotations': [...]}
    wrapper, since the Kaggle mirrors differ.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("annotations", data.get("data", []))
    return data


def _answerable(a: dict):
    """VizWiz marks answerability inconsistently across releases."""
    for k in ("answerable", "answer_type"):
        if k in a:
            v = a[k]
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                return v.lower() not in ("unanswerable", "unsuitable")
    return None


def candidates(anns, img_dir: str = VAL_IMG_DIR, n: int = 200, seed: int = 0,
               unanswerable_frac: float = 0.75):
    """Sample candidates for hand-sorting.

    Oversamples unanswerable questions, because those are the ones that carry
    the states we care about (S1-S5); answerable ones supply S0.  The default
    75/25 split gives roughly 150 unanswerable and 50 answerable, matching
    the 150-per-state target of Tier A at a smaller scale.
    """
    rng = random.Random(seed)
    ok, bad = [], []
    for a in anns:
        img = a.get("image") or a.get("image_id") or a.get("file_name")
        q = a.get("question")
        if not img or not q:
            continue
        p = os.path.join(img_dir, img) if not os.path.isabs(img) else img
        rec = {"image": img, "image_path": p, "question": q,
               "answers": [x.get("answer") for x in a.get("answers", [])
                           if isinstance(x, dict)]}
        (bad if _answerable(a) is False else ok).append(rec)

    n_bad = int(round(n * unanswerable_frac))
    n_ok = n - n_bad
    picked = (rng.sample(bad, min(n_bad, len(bad))) +
              rng.sample(ok, min(n_ok, len(ok))))
    rng.shuffle(picked)
    print(f"VizWiz candidates: {len(picked)} "
          f"({min(n_bad, len(bad))} unanswerable, {min(n_ok, len(ok))} answerable) "
          f"from a pool of {len(bad)} / {len(ok)}")
    return picked


def export_labelling_sheet(cands, out_dir: str):
    """Write the sheet you hand-sort.

    Fill the ``state`` column with S0-S5.  Leave a row blank to exclude it;
    ambiguous items should be excluded rather than forced, and the exclusion
    count belongs in the appendix.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tierb_sheet.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "image", "image_path", "question",
                    "vizwiz_answers", "state", "notes"])
        for i, c in enumerate(cands):
            w.writerow([i, c["image"], c["image_path"], c["question"],
                        " | ".join(a for a in c["answers"] if a), "", ""])
    print(f"Hand-sorting sheet: {path}")
    print("Fill the 'state' column with S0-S5. Blank rows are excluded.")
    return path


def load_labelled_sheet(out_dir: str, img_dir: str = VAL_IMG_DIR):
    """Read the hand-sorted sheet back as EVID-6 Items.

    Tier B items get ``condition="main"`` and no intervention metadata —
    nothing was done to these images, the failure was already there. They
    are therefore excluded from dose-response and used only for the
    classification ladder and the probe's cross-domain test.
    """
    path = os.path.join(out_dir, "tierb_sheet.csv")
    items, skipped = [], 0
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            st = (row.get("state") or "").strip().upper()
            if st not in STATES:
                skipped += 1
                continue
            p = row["image_path"]
            if not os.path.isfile(p):
                p = os.path.join(img_dir, row["image"])
            items.append(Item(
                item_id=f"tb_{int(row['row']):04d}",
                base_image_id=-(int(row["row"]) + 1),   # negative: not COCO
                state=st,
                condition="main",
                category="vizwiz",
                question=row["question"],
                image_path=p,
            ))
    print(f"Tier B: {len(items)} labelled, {skipped} excluded")
    dist = {}
    for it in items:
        dist[it.state] = dist.get(it.state, 0) + 1
    print("  state distribution:", dict(sorted(dist.items())))
    return items
