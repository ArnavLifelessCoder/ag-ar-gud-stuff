"""EVID-6 core schema: taxonomy constants and the Item dataclass.

Every module in the pipeline imports from here.  Keep this file free of
heavy dependencies (no numpy, no torch, no PIL).
"""

from dataclasses import dataclass, asdict, field
from typing import Optional
import json
import os

# ── Taxonomy ────────────────────────────────────────────────────────────────

STATES = ["S0", "S1", "S2", "S3", "S4", "S5"]

STATE_TEXT = {
    "S0": "The image contains enough evidence to answer.",
    "S1": "The object is not inside the frame, though it exists in the scene.",
    "S2": "The object is inside the frame but blocked by something in front of it.",
    "S3": "The object is visible and unblocked, but too small, blurred or dark to make out.",
    "S4": "More than one object matches the description, so the question is ambiguous.",
    "S5": "The object referred to is not present in this scene at all.",
}

CONDITIONS = ["main", "s0ctrl", "prioronly", "clean_ref"]

# States for which self-consistency against the clean-image answer is
# meaningful.  S5 names a category that is absent, so there is no evidence
# to remove and no reference answer worth comparing against.
CONSISTENCY_STATES = ["S0", "S1", "S2", "S3", "S4"]

REPAIR = {
    "S0": "answer",
    "S1": "pan the camera",
    "S2": "move to another angle",
    "S3": "zoom in",
    "S4": "ask the questioner",
    "S5": "correct the premise",
}


# ── Item dataclass ──────────────────────────────────────────────────────────

@dataclass
class Item:
    """One benchmark item: an image + question + evidence-state label.

    Fields below ``ref_answer`` are intervention metadata - the regressors
    for the dose-response analysis.
    """
    item_id: str
    base_image_id: int          # COCO image id.  THIS is the split group key.
    state: str
    condition: str              # main | s0ctrl | prioronly | clean_ref
    category: str
    question: str
    image_path: str
    ref_answer: Optional[str] = None
    # Linkage
    ref_group: Optional[str] = None      # all items sharing one clean reference
    parent_item_id: Optional[str] = None # for s0ctrl/prioronly: the main item
    artifact: Optional[str] = None       # occlude | degrade | blank | None
    # Intervention metadata
    occl_frac: Optional[float] = None
    inst_pixels: Optional[int] = None
    severity: Optional[int] = None
    eff_res: Optional[float] = None   # achieved effective resolution, px
    n_candidates: Optional[int] = None
    delta_e: Optional[float] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, line: str) -> "Item":
        return cls(**json.loads(line))


# ── Closed-set reference task ───────────────────────────────────────────────

# The free-form clean-reference task failed its own stability gate on all three
# models: three samples at temperature 0.7 agreed on only 41.2% / 21.3% / 41.0%
# of groups, against a 35% maximum drop rate. Open-ended answers to "What is the
# wine glass made of?" are simply not reproducible.
#
# The plan's remedy is a closed answer set. Colour is the right choice: it is
# answerable for essentially any object, it is what S4's CIEDE2000 gate already
# certifies as distinguishing the ambiguous candidates, and it is what S3's
# contrast/luminance reduction actually destroys - so the question probes the
# intervention rather than sitting beside it.
CLOSED_COLOURS = ["black", "blue", "brown", "green", "grey", "orange",
                  "pink", "purple", "red", "white", "yellow"]


def closed_colour_question(category: str) -> str:
    """The single question used for the closed-set consistency measurement."""
    return f"What colour is the {category}?"


def make_closed_manifest(items_path: str, out_path: str = None) -> str:
    """Rewrite a manifest so every item asks the closed-set colour question.

    Consistency asks whether the model's answer to a *fixed* question survives
    an intervention. The question therefore need not be the one drawn at build
    time - and using one question for every item both maximises usable n and
    removes question type as a confound between states.

    Only ``question`` changes. Item ids, states, conditions, reference groups
    and image paths are untouched, so the rows still align with every other
    pass and with the activations.

    Returns the path written.
    """
    out_path = out_path or "/kaggle/working/items_closed.jsonl"
    rows, changed = [], 0
    with open(items_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            it = json.loads(line)
            q = closed_colour_question(it["category"])
            if it.get("question") != q:
                changed += 1
            it["question"] = q
            rows.append(it)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in rows:
            f.write(json.dumps(it) + "\n")
    print(f"make_closed_manifest: {len(rows)} items, {changed} questions "
          f"rewritten -> {out_path}")
    return out_path


def is_closed_answer(ans: str) -> bool:
    """Did the model comply with the closed answer set?

    Reported as a compliance rate. A constrained prompt that the model ignores
    is not a fix, and the honest way to find out is to measure it rather than
    to snap near-misses onto the set afterwards.
    """
    if not ans:
        return False
    a = "".join(c for c in ans.strip().lower() if c.isalpha() or c.isspace())
    return a.strip() in CLOSED_COLOURS


# ── Locating the manifest across notebooks ──────────────────────────────────

def find_items(search_roots=("/kaggle/input", "/kaggle/working")) -> str:
    """Locate ``items.jsonl`` wherever NB1's output got mounted.

    NB2/NB3/NB4 attach NB1's output as a dataset, and its mount path is the NB1
    notebook's title slugified - which nobody can guarantee matches a hard-coded
    guess. Rather than list three paths and assert, walk the input roots and
    find the file. Same approach as ``find_coco``.

    Returns the path, or raises listing what *is* attached so the fix is obvious.
    """
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        # Shallow, common locations first (fast path), then a full walk.
        direct = os.path.join(root, "items.jsonl")
        if os.path.isfile(direct):
            return direct
        for dirpath, _dirnames, filenames in os.walk(root):
            if "items.jsonl" in filenames:
                return os.path.join(dirpath, "items.jsonl")
    seen = {}
    for root in search_roots:
        if os.path.isdir(root):
            seen[root] = sorted(os.listdir(root))
    raise FileNotFoundError(
        "items.jsonl not found. Attach NB1's Saved output as a Notebook input. "
        f"Currently attached: {seen}"
    )


# ── Manifest rebasing ───────────────────────────────────────────────────────

def rebase_items(items_path: str, search_roots, out_path: str = None) -> str:
    """Rewrite ``items.jsonl`` so every ``image_path`` points at a file that exists.

    NB1 records absolute paths from its own session
    (``/kaggle/working/evid6/images/...``).  In NB2/NB3 those images arrive as an
    attached dataset under a different root, so the recorded path does not
    resolve and the first ``Image.open`` in ``build_inputs`` kills the pass -
    on the first item, after the model has already loaded.  NB4 remaps for the
    CLIP baseline; the inference notebooks had no equivalent.

    Call this once in setup and point ``ITEMS_PATH`` at what it returns.

    Parameters
    ----------
    items_path : str
        The manifest as written by NB1.
    search_roots : sequence of str
        Directories to look under, in priority order.  Both ``<root>/<name>``
        and ``<root>/evid6/images/<name>`` are tried.
    out_path : str, optional
        Where to write the rebased manifest.  Defaults to
        ``/kaggle/working/items_local.jsonl`` - the NB1 manifest usually lives
        under a read-only ``/kaggle/input`` mount, so it cannot be rewritten
        in place.

    Returns
    -------
    str - path to the rebased manifest.  Raises if nothing resolved at all,
    since that means the wrong dataset is attached and every later number
    would be built on an empty run.
    """
    out_path = out_path or "/kaggle/working/items_local.jsonl"
    roots = [r for r in search_roots if r and os.path.isdir(r)]

    rows, n_ok, n_moved, missing = [], 0, 0, []
    with open(items_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            it = json.loads(line)
            p = it.get("image_path", "")
            if p and os.path.isfile(p):
                n_ok += 1
            else:
                name = os.path.basename(p)
                for root in roots:
                    for cand in (os.path.join(root, name),
                                 os.path.join(root, "evid6", "images", name)):
                        if os.path.isfile(cand):
                            it["image_path"] = cand
                            n_moved += 1
                            break
                    else:
                        continue
                    break
                else:
                    missing.append(name)
            rows.append(it)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for it in rows:
            f.write(json.dumps(it) + "\n")

    print(f"rebase_items: {len(rows)} items - {n_ok} already valid, "
          f"{n_moved} remapped, {len(missing)} unresolved")
    if missing:
        print(f"  first unresolved: {missing[0]}")
    if n_ok + n_moved == 0:
        raise FileNotFoundError(
            f"No image in {items_path} resolved under {roots}. The wrong "
            f"dataset is attached - fix this before spending GPU quota."
        )
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(rows)} images unresolved (e.g. "
            f"{missing[0]}). A partial run would silently drop items from "
            f"every downstream count."
        )
    return out_path

def load_items(path: str) -> list:
    """Load items from a JSONL file."""
    with open(path, encoding="utf-8") as f:
        return [Item(**json.loads(l)) for l in f if l.strip()]


def save_items(items: list, path: str) -> None:
    """Save items to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(it.to_json() + "\n")
