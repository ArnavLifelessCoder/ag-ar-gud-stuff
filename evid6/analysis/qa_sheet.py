"""EVID-6 visual QA contact sheets.

Every geometric check in the generator is a proxy. S2 measures realised
coverage, S4 measures a colour distance, S1 checks a bounding box - none of
them can tell you the occluder landed somewhere absurd, or that the "ambiguous"
pair is obviously distinguishable to a human, or that the referent was already
unrecognisable before we touched it.

So look at the images. This module lays them out in labelled grids and writes
a small HTML index so a few hundred can be scanned in a couple of minutes,
which is the only realistic way anyone actually does it.

Do this on the pilot, before the full build. Finding a broken generator after
1,500 images and a GPU sweep is the expensive way.

Usage
-----
    from qa_sheet import contact_sheets, triptychs
    contact_sheets(items, "/kaggle/working/qa", per_state=48)
    triptychs(items, "/kaggle/working/qa", n=24)
"""

import os
import math
import random
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

STATE_ORDER = ["S0", "S1", "S2", "S3", "S4", "S5"]


def _get(it, k, default=None):
    if isinstance(it, dict):
        return it.get(k, default)
    return getattr(it, k, default)


def _label(it):
    """Short caption carrying the numbers that justify the item's label."""
    st = _get(it, "state")
    bits = [st, _get(it, "category", "?")]
    occ = _get(it, "occl_frac")
    sev = _get(it, "severity")
    de = _get(it, "delta_e")
    nc = _get(it, "n_candidates")
    if occ is not None and st == "S2":
        bits.append(f"cov={occ:.2f}")
    if sev is not None:
        bits.append(f"sev={sev}")
    if de is not None:
        bits.append(f"dE={de:.0f}")
    if nc is not None:
        bits.append(f"n={nc}")
    return " ".join(str(b) for b in bits)


def _grid(items, out_path, title, cols=8, thumb=220):
    rows = math.ceil(len(items) / cols)
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 2.1, rows * 2.35))
    axes = axes.reshape(rows, cols) if rows > 1 else axes.reshape(1, cols)
    missing = 0
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i >= len(items):
            continue
        it = items[i]
        p = _get(it, "image_path")
        try:
            im = Image.open(p).convert("RGB")
            im.thumbnail((thumb, thumb))
            ax.imshow(im)
        except (OSError, ValueError):
            missing += 1
            ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=7)
        ax.set_title(_label(it), fontsize=6.2)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return missing


def contact_sheets(items, out_dir: str, per_state: int = 48, cols: int = 8,
                   seed: int = 0, conditions=("main",)):
    """One contact sheet per state, plus an HTML index.

    ``per_state=48`` over six states is ~290 images, which is the 200-300 the
    review asks you to eyeball before committing to the full build.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)

    by_state = defaultdict(list)
    for it in items:
        if _get(it, "condition") in conditions:
            by_state[_get(it, "state")].append(it)

    made, total, missing = [], 0, 0
    for st in STATE_ORDER:
        pool = by_state.get(st, [])
        if not pool:
            continue
        sample = rng.sample(pool, min(per_state, len(pool)))
        path = os.path.join(out_dir, f"qa_{st}.png")
        missing += _grid(sample, path,
                         f"{st} - {len(sample)} of {len(pool)} items", cols=cols)
        made.append((st, os.path.basename(path), len(sample), len(pool)))
        total += len(sample)
        print(f"  {st}: {len(sample)} shown of {len(pool)} -> {path}")

    _write_index(out_dir, made, total)
    if missing:
        print(f"  WARNING: {missing} image paths could not be opened")
    print(f"\n{total} images laid out. Open {out_dir}/index.html and scan them.")
    return made


def triptychs(items, out_dir: str, n: int = 24, seed: int = 0):
    """Clean reference | intervention | prior-only, side by side.

    A contact sheet shows whether an image looks broken. A triptych shows
    whether the *intervention* did what it claims: same scene, referent
    progressively removed. This is the figure a reviewer asks for when they
    suspect the transformation is doing something other than advertised, so
    it is worth having one panel of these in the appendix anyway.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(seed)

    by_group = defaultdict(dict)
    for it in items:
        g = _get(it, "ref_group")
        if g is None:
            continue
        cond = _get(it, "condition")
        st = _get(it, "state")
        if cond == "clean_ref":
            by_group[g]["ref"] = it
        elif cond == "main" and st in ("S2", "S3"):
            by_group[g].setdefault("main", it)
        elif cond == "prioronly":
            by_group[g].setdefault("po", it)
        elif cond == "s0ctrl":
            by_group[g].setdefault("ctrl", it)

    full = [g for g, d in by_group.items()
            if {"ref", "main", "po"} <= set(d)]
    if not full:
        print("  no complete reference groups found - skipping triptychs")
        return None

    picked = rng.sample(full, min(n, len(full)))
    cols = 4
    fig, axes = plt.subplots(len(picked), cols,
                             figsize=(cols * 2.6, len(picked) * 2.5))
    axes = axes.reshape(len(picked), cols)
    heads = ["clean reference", "intervention", "S0-ctrl (same artifact)",
             "prior-only (floor)"]
    for r, g in enumerate(picked):
        d = by_group[g]
        for c, key in enumerate(["ref", "main", "ctrl", "po"]):
            ax = axes[r, c]
            ax.axis("off")
            it = d.get(key)
            if it is None:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=7)
                continue
            try:
                im = Image.open(_get(it, "image_path")).convert("RGB")
                im.thumbnail((260, 260))
                ax.imshow(im)
            except (OSError, ValueError):
                ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=7)
            if r == 0:
                ax.set_title(heads[c], fontsize=8)
            if c == 0:
                ax.text(-0.05, 0.5, _label(d.get("main", it)),
                        transform=ax.transAxes, rotation=90, fontsize=6,
                        va="center", ha="right")
    path = os.path.join(out_dir, "qa_triptychs.png")
    fig.suptitle("Reference / intervention / control / floor", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"  {len(picked)} triptychs -> {path}")
    return path


def _write_index(out_dir, made, total):
    rows = "\n".join(
        f'<section><h2>{st} <small>{shown} of {pool} shown</small></h2>'
        f'<img src="{fn}"></section>'
        for st, fn, shown, pool in made
    )
    extra = ""
    if os.path.isfile(os.path.join(out_dir, "qa_triptychs.png")):
        extra = ('<section><h2>Triptychs</h2>'
                 '<img src="qa_triptychs.png"></section>')
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>EVID-6 visual QA</title>
<style>
 body {{ font: 15px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 1400px;
        color: #1a1a1a; }}
 h1 {{ margin-bottom: .2rem; }}
 p.lead {{ color: #555; margin-top: 0; }}
 section {{ margin: 2.5rem 0; }}
 h2 {{ border-bottom: 1px solid #ddd; padding-bottom: .3rem; }}
 h2 small {{ font-weight: 400; color: #777; font-size: .7em; }}
 img {{ width: 100%; height: auto; border: 1px solid #e5e5e5; }}
 ul {{ color: #444; }}
</style>
<h1>EVID-6 visual QA - {total} images</h1>
<p class="lead">Captions carry the numbers the generator used to accept each
item. Scan for anything the geometry could not catch.</p>
<ul>
  <li><b>S1</b> - is every instance really outside the crop?</li>
  <li><b>S2</b> - does the occluder sit on the referent, and does it look like
      an object rather than a pasted rectangle? <code>cov</code> should be ≥ 0.90.</li>
  <li><b>S3</b> - is the referent degraded but still clearly <i>there</i>?
      If severity 3 looks like deletion, S3 has collapsed into S2.</li>
  <li><b>S4</b> - would a person actually be unsure which one is meant?
      A high <code>dE</code> with obviously different objects is not ambiguity.</li>
  <li><b>S5</b> - is the named category genuinely absent?</li>
  <li><b>S0</b> - was it answerable in the first place?</li>
</ul>
{rows}
{extra}
"""
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
