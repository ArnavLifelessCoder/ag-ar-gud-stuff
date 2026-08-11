"""EVID-6 dataset generator.

Reads COCO val2017 annotations and images, applies six evidence-state
transforms, and writes an ``items.jsonl`` manifest plus transformed images.

Every generator is deterministic given ``seed`` and records its intervention
parameters so the dose-response analysis can use them as regressors.
"""

import os
import json
import random
import hashlib
import numpy as np
from PIL import Image, ImageFilter
from pycocotools.coco import COCO
from skimage.color import rgb2lab, deltaE_ciede2000

from schema import Item, STATES, save_items

# ── Paths (Kaggle layout) ──────────────────────────────────────────────────

ROOT = "/kaggle/input/coco-2017-dataset/coco2017"
IMG_DIR = f"{ROOT}/val2017"
OUT_DIR = "/kaggle/working/evid6/images"


# ── COCO setup ──────────────────────────────────────────────────────────────

def init_coco(root=ROOT):
    """Initialise the COCO API.  Called once at module level or by the driver."""
    global coco, CATS, ALL_CAT_NAMES
    os.makedirs(OUT_DIR, exist_ok=True)
    coco = COCO(f"{root}/annotations/instances_val2017.json")
    CATS = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    ALL_CAT_NAMES = set(CATS.values())
    return coco


# ── Constants ───────────────────────────────────────────────────────────────

MIN_AREA = 4000       # px — referent must be big enough that S0 is genuinely answerable
MAX_AREA = 120000     # keep the referent from dominating the frame

PERSONISH = {"person", "dog", "cat", "horse", "bird", "sheep", "cow", "elephant"}
TEXTISH   = {"stop sign", "clock", "book", "laptop", "tv", "cell phone", "bottle"}


# ── Question templates ──────────────────────────────────────────────────────

def question_for(cat: str, rng=None) -> str:
    """Pick a question template appropriate for the category.

    ``rng`` must be the build's seeded ``random.Random``.  It used to default
    to the global ``random`` module, which made the question — the actual
    experimental stimulus — irreproducible: two ``build(seed=0)`` calls in one
    process assigned different questions to 38 of 68 items while every
    ``item_id`` matched, because ``uid()`` does not hash the question. The
    determinism test compared ids only and so never saw it.
    """
    if rng is None:
        raise ValueError(
            "question_for() needs the build's seeded rng; passing None would "
            "draw from the global random stream and break reproducibility"
        )
    if cat in PERSONISH:
        pool = ["What colour is the {c}?", "What is the {c} doing?"]
    elif cat in TEXTISH:
        pool = ["What colour is the {c}?", "What is written on the {c}?"]
    else:
        pool = ["What colour is the {c}?", "What is the {c} made of?"]
    return rng.choice(pool).format(c=cat)


# ── Helpers ─────────────────────────────────────────────────────────────────

def uid(*parts) -> str:
    """Deterministic 12-char hex id from arbitrary parts."""
    return hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:12]


def save(img: Image.Image, name: str) -> str:
    """Save an image to the output directory and return its path."""
    p = f"{OUT_DIR}/{name}.jpg"
    img.convert("RGB").save(p, quality=95)
    return p


def filename_for(state: str, condition: str, item_id: str) -> str:
    """Self-describing filename keyed on the item's OWN id.

    ``{state}_{condition}_{item_id}`` — so a stray file in the images
    directory can be traced back to its manifest row without a lookup, and
    sorting the directory groups by state then condition. Earlier revisions
    named auxiliary images after their *parent's* id, which meant 42 of 139
    files could not be mapped to the item they belonged to.
    """
    return f"{state}_{condition}_{item_id}"


def masks_for(img_id: int, cat_name: str):
    """All instance masks of one category in one image."""
    cid = [k for k, v in CATS.items() if v == cat_name][0]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=[cid], iscrowd=False))
    return [(a, coco.annToMask(a).astype(bool)) for a in anns]


# ── Occluder bank ───────────────────────────────────────────────────────────

def build_occluder_bank(n: int = 400, seed: int = 0) -> list:
    """Build a bank of real-object RGBA crops from *other* COCO images.

    Using real objects instead of black boxes prevents the trivial
    reviewer attack that the model just learned to detect black rectangles.
    """
    rng = random.Random(seed)
    ids = rng.sample(coco.getImgIds(), min(1200, len(coco.getImgIds())))
    bank = []
    for iid in ids:
        for a in coco.loadAnns(coco.getAnnIds(imgIds=iid, iscrowd=False)):
            if not (6000 < a["area"] < 90000):
                continue
            m = coco.annToMask(a).astype(bool)
            im = Image.open(
                f"{IMG_DIR}/{coco.loadImgs(iid)[0]['file_name']}"
            ).convert("RGB")
            arr = np.array(im)
            ys, xs = np.where(m)
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            patch = arr[y0:y1, x0:x1]
            alpha = (m[y0:y1, x0:x1] * 255).astype(np.uint8)
            rgba = np.dstack([patch, alpha])
            bank.append((Image.fromarray(rgba, "RGBA"), CATS[a["category_id"]]))
            if len(bank) >= n:
                return bank
    return bank


# ── Six state generators ───────────────────────────────────────────────────

def gen_S0(img, *_args, **_kw):
    """S0 — Answerable: return untouched image."""
    return img, {}


def gen_S1(img, masks, **_kw):
    """S1 — Out of frame: crop so every instance falls outside.

    Picks the largest axis-aligned crop that excludes the union bounding box,
    subject to a minimum-size constraint (35% of each dimension).
    """
    W, H = img.size
    union = np.zeros((H, W), bool)
    for _, m in masks:
        union |= m
    ys, xs = np.where(union)
    # Candidate crops: left, right, top, bottom of the union bbox
    cands = [
        (0, 0, xs.min(), H),
        (xs.max(), 0, W, H),
        (0, 0, W, ys.min()),
        (0, ys.max(), W, H),
    ]
    cands = [c for c in cands if (c[2] - c[0]) > 0.35 * W and (c[3] - c[1]) > 0.35 * H]
    if not cands:
        return None, None
    box = max(cands, key=lambda c: (c[2] - c[0]) * (c[3] - c[1]))
    return img.crop(box), {"occl_frac": 1.0}


def gen_S2(img, masks, bank=None, seed=0, **_kw):
    """S2 — Occluded: composite a real object over >=90% of every instance mask.

    Rejects items where realised coverage falls below 90%.
    """
    rng = random.Random(seed)
    out = img.copy()
    covered_num, covered_den = 0, 0
    for _, m in masks:
        ys, xs = np.where(m)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        bw, bh = int((x1 - x0) * 1.25), int((y1 - y0) * 1.25)
        occ, _ = rng.choice(bank)
        occ = occ.resize((max(bw, 8), max(bh, 8)), Image.LANCZOS)
        px = max(0, x0 - (bw - (x1 - x0)) // 2)
        py = max(0, y0 - (bh - (y1 - y0)) // 2)
        out.paste(occ, (px, py), occ)
        # Measure realised coverage
        occ_m = np.zeros(m.shape, bool)
        a = np.array(occ)[:, :, 3] > 128
        h, w = a.shape
        occ_m[py:py + h, px:px + w] |= a[:min(h, m.shape[0] - py), :min(w, m.shape[1] - px)]
        covered_num += (m & occ_m).sum()
        covered_den += m.sum()
    frac = covered_num / max(covered_den, 1)
    if frac < 0.90:
        return None, None       # reject — do not silently ship under-occluded items
    return out, {"occl_frac": float(frac), "artifact": "occlude"}


# S3 calibration.
#
# Two things were wrong with the original fixed-division-factor design, both
# found by measuring the generator's output rather than reading the code:
#
#   1. A fixed factor means severity is confounded with referent size. At
#      factor 24 a 63x63 referent (MIN_AREA) becomes 2x2 — deletion, not
#      degradation, which collapses S3 into S2 and makes P1/P2 inseparable.
#      A 346x346 referent at the same factor becomes 14x14, a mild blur.
#      We therefore target an absolute effective resolution instead, so
#      "severity 2" means the same thing for every object.
#
#   2. Downsampling averages pixels, so the mean colour inside the mask
#      survives to within CIEDE2000 dE < 0.25 at every severity. Since
#      "What colour is the {category}?" is in every question pool, the
#      intervention was removing almost nothing the question depended on.
#      The state text already promises "too small, blurred *or dark* to make
#      out"; the generator now implements the third term, which attenuates
#      chroma and luminance along with spatial detail.
S3_TARGET_RES = {1: 32, 2: 16, 3: 8}    # effective px on the longer side
S3_BLUR = {1: 1.0, 2: 2.0, 3: 3.5}
S3_CONTRAST = {1: 0.75, 2: 0.55, 3: 0.35}   # multiplicative, toward mid-grey
S3_LUMA = {1: 0.85, 2: 0.70, 3: 0.55}       # overall darkening


def _degrade_region(region: Image.Image, severity: int):
    """Downsample to a target resolution, blur, then reduce contrast and
    luminance. Returns (degraded RGB array, achieved effective resolution).

    Shared by gen_S3 and the matched S0-ctrl, so the control carries exactly
    the same artifact rather than an approximation of it.
    """
    target = S3_TARGET_RES[severity]
    w, h = region.size
    longer = max(w, h)
    scale = min(1.0, target / max(longer, 1))       # never upsample
    # round, not truncate: target/longer is a float, so longer*scale lands on
    # 7.99999... instead of 8.0 for many sizes (a 166px referent at severity 3
    # gave 7). int() then floored it below the 8px line that keeps S3 from
    # collapsing into S2 — the exact failure this severity scheme exists to
    # prevent, and small enough to hide in a "min 8px" printout.
    sw = max(int(round(w * scale)), 2)
    sh = max(int(round(h * scale)), 2)

    small = region.resize((sw, sh), Image.BILINEAR)
    back = small.resize((w, h), Image.NEAREST).filter(
        ImageFilter.GaussianBlur(S3_BLUR[severity])
    )

    x = np.asarray(back).astype(np.float32) / 255.0
    x = (x - 0.5) * S3_CONTRAST[severity] + 0.5     # contrast toward grey
    x = x * S3_LUMA[severity]                       # and darker
    out = np.clip(x * 255.0, 0, 255).astype(np.uint8)
    return out, max(sw, sh)


def gen_S3(img, masks, severity=2, **_kw):
    """S3 — Sub-resolution: degrade only the instance regions.

    Severity now controls an absolute target resolution rather than a
    division factor, so the dose is not confounded with referent size, and
    the achieved resolution is recorded as a continuous regressor.
    """
    arr = np.array(img).copy()
    total_px, eff = 0, []
    for _, m in masks:
        ys, xs = np.where(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        reg = Image.fromarray(arr[y0:y1, x0:x1])
        patch, res = _degrade_region(reg, severity)
        sub = m[y0:y1, x0:x1]
        arr[y0:y1, x0:x1][sub] = patch[sub]
        eff.append(res)
        total_px += int(res * res)
    return Image.fromarray(arr), {
        "severity": severity,
        "inst_pixels": total_px,
        "eff_res": float(np.mean(eff)) if eff else None,
        "artifact": "degrade",
    }


def gen_S4(img, masks, **_kw):
    """S4 — Ambiguous reference: >=2 instances, verified colour-distinct.

    Rejects if CIEDE2000 colour distance between top-2 candidates is below 12,
    meaning they are too similar for the question to be genuinely ambiguous.
    """
    if len(masks) < 2:
        return None, None
    arr = np.array(img).astype(float) / 255.0
    labs = []
    for _, m in masks[:3]:
        mean_rgb = arr[m].mean(0).reshape(1, 1, 3)
        labs.append(rgb2lab(mean_rgb))
    d = float(deltaE_ciede2000(labs[0], labs[1])[0, 0])
    if d < 12.0:
        return None, None       # too similar, not genuinely ambiguous
    return img, {"n_candidates": len(masks), "delta_e": d}


def gen_S5(img, *_args, **_kw):
    """S5 — False premise: image untouched, question names an absent class."""
    return img, {}


# ── Auxiliary conditions ────────────────────────────────────────────────────

def gen_s0ctrl(img, masks, bank, seed=0, artifact="occlude", severity=2):
    """S0-ctrl: the same artifact, at the same area, on a region that touches
    no instance of the queried category.

    ``artifact`` must match the parent state, otherwise the control is not a
    control:
      - S2 parent -> "occlude": paste an occluder patch elsewhere.
      - S3 parent -> "degrade": downsample+blur an equal-area region elsewhere.

    Returns (image, meta) or (None, None) if no clear region was found.
    """
    rng = random.Random(seed + 777)
    W, H = img.size
    union = np.zeros((H, W), bool)
    for _, m in masks:
        union |= m
    area = int(union.sum())
    side = int(np.sqrt(area * 1.25))
    side = max(8, min(side, min(W, H) - 1))

    for _ in range(60):
        x = rng.randint(0, max(W - side, 1))
        y = rng.randint(0, max(H - side, 1))
        if union[y:y + side, x:x + side].sum() != 0:
            continue

        if artifact == "occlude":
            occ, _ = rng.choice(bank)
            out = img.copy()
            occ_resized = occ.resize((side, side), Image.LANCZOS)
            out.paste(occ_resized, (x, y), occ_resized)
            return out, {"occl_frac": 0.0, "artifact": "occlude"}

        elif artifact == "degrade":
            # Identical transform to gen_S3, applied off-target.
            arr = np.array(img).copy()
            reg = Image.fromarray(arr[y:y + side, x:x + side])
            patch, res = _degrade_region(reg, severity)
            arr[y:y + side, x:x + side] = patch
            return Image.fromarray(arr), {
                "occl_frac": 0.0, "severity": severity,
                "eff_res": float(res), "artifact": "degrade",
            }

        else:
            raise ValueError(f"unknown artifact {artifact!r}")

    return None, None


def gen_prioronly(img, masks, **_kw):
    """Prior-only: blank the queried category's regions to grey (128).

    If the model still answers correctly, it was answering from priors.
    """
    arr = np.array(img).copy()
    for _, m in masks:
        arr[m] = 128
    return Image.fromarray(arr), {"occl_frac": 1.0, "artifact": "blank"}


# ── Main driver ─────────────────────────────────────────────────────────────

def build(n_per_state: int = 150, seed: int = 0) -> list:
    """Build the full EVID-6 dataset.

    Returns a list of Item objects.  Rejection counts per state are printed
    at the end — these go in the appendix.

    Target: 150 items per state = 900 main items, plus ~300 S0-ctrl,
    ~300 prior-only and one clean_ref per reference group.

    Every non-S5 item carries ``ref_group``; exactly one item per group has
    ``condition == "clean_ref"`` and points at the untouched image.  That is
    the item whose answer becomes the reference for the whole group.
    """
    rng = random.Random(seed)
    bank = build_occluder_bank(seed=seed)
    items = []
    rejects = {s: 0 for s in STATES}
    img_ids = coco.getImgIds()
    rng.shuffle(img_ids)

    GEN = {"S1": gen_S1, "S2": gen_S2, "S3": gen_S3, "S4": gen_S4}
    need = {s: n_per_state for s in STATES}

    for iid in img_ids:
        if sum(need.values()) == 0:
            break
        info = coco.loadImgs(iid)[0]
        img = Image.open(f"{IMG_DIR}/{info['file_name']}").convert("RGB")
        anns = coco.loadAnns(coco.getAnnIds(imgIds=iid, iscrowd=False))
        present = {CATS[a["category_id"]] for a in anns}

        # Group annotations by category, filtering by area
        by_cat = {}
        for a in anns:
            if MIN_AREA < a["area"] < MAX_AREA:
                by_cat.setdefault(CATS[a["category_id"]], []).append(a)
        if not by_cat:
            continue

        cat = rng.choice(list(by_cat))
        masks = [(a, coco.annToMask(a).astype(bool)) for a in by_cat[cat]]
        q = question_for(cat, rng)

        # One reference group per (image, category, question).  Every derived
        # item below carries this key, and exactly one clean_ref item per
        # group holds the untouched image the reference answer comes from.
        rgid = uid(iid, cat, "refgroup")
        group_items = []

        # Try each state that still needs items
        for st in ["S0", "S1", "S2", "S3", "S4"]:
            if need[st] == 0:
                continue

            sev = rng.choice([1, 2, 3])
            if st == "S0":
                # S0 requires exactly one instance (unambiguous)
                if len(masks) != 1:
                    continue
                out, meta = img, {}
            else:
                out, meta = GEN[st](img, masks, bank=bank, seed=iid,
                                    severity=sev)

            if out is None:
                rejects[st] += 1
                continue

            iid_s = uid(iid, cat, st)
            p_img = save(out, filename_for(st, "main", iid_s))
            group_items.append(Item(iid_s, iid, st, "main", cat, q, p_img,
                                    ref_group=rgid, **meta))
            need[st] -= 1

            # Auxiliary conditions for S2 and S3.  The control must carry the
            # SAME artifact as its parent state, otherwise it is not a control.
            if st in ("S2", "S3"):
                art = "occlude" if st == "S2" else "degrade"
                c, cm = gen_s0ctrl(img, masks, bank, seed=iid,
                                   artifact=art, severity=sev)
                if c is not None:
                    ctrl_id = uid(iid, cat, st, "ctrl")
                    group_items.append(Item(
                        ctrl_id, iid, "S0", "s0ctrl", cat, q,
                        save(c, filename_for("S0", "s0ctrl", ctrl_id)),
                        ref_group=rgid, parent_item_id=iid_s, **cm,
                    ))
                po, pm = gen_prioronly(img, masks)
                po_id = uid(iid, cat, st, "po")
                group_items.append(Item(
                    po_id, iid, st, "prioronly", cat, q,
                    save(po, filename_for(st, "prioronly", po_id)),
                    ref_group=rgid, parent_item_id=iid_s, **pm,
                ))

        # Exactly one clean reference per group, on the untouched image.
        # This is the pseudo-ground-truth every derived item is scored against.
        if group_items:
            ref_id = uid(iid, cat, "cleanref")
            items.append(Item(ref_id, iid, "S0", "clean_ref", cat, q,
                              save(img, filename_for("S0", "clean_ref", ref_id)),
                              ref_group=rgid))
            items.extend(group_items)

        # S5: false premise — name a category that is absent from the image.
        # No clean reference: there is no evidence to remove, so consistency
        # is undefined here (see schema.CONSISTENCY_STATES).
        if need["S5"] > 0:
            absent = list(ALL_CAT_NAMES - present)
            if absent:
                c5 = rng.choice(absent)
                iid_s = uid(iid, c5, "S5")
                items.append(Item(
                    iid_s, iid, "S5", "main", c5, question_for(c5, rng),
                    save(img, filename_for("S5", "main", iid_s)),
                ))
                need["S5"] -= 1

    print("rejection counts:", rejects)
    print(f"total items: {len(items)}")
    print(f"remaining need: {need}")

    # Persist the rejection stats — this table goes in the appendix and is
    # lost forever when the Kaggle session is wiped.
    try:
        stats_path = os.path.join(os.path.dirname(OUT_DIR), "build_stats.json")
        os.makedirs(os.path.dirname(stats_path), exist_ok=True)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump({
                "n_per_state_target": n_per_state,
                "seed": seed,
                "rejections": rejects,
                "unfilled": need,
                "n_items": len(items),
            }, f, indent=2)
        print(f"build stats written to {stats_path}")
    except OSError as e:
        print(f"could not write build stats: {e}")

    return items
