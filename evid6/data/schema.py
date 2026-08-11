"""EVID-6 core schema: taxonomy constants and the Item dataclass.

Every module in the pipeline imports from here.  Keep this file free of
heavy dependencies (no numpy, no torch, no PIL).
"""

from dataclasses import dataclass, asdict, field
from typing import Optional
import json

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

    Fields below ``ref_answer`` are intervention metadata — the regressors
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


def load_items(path: str) -> list:
    """Load items from a JSONL file."""
    with open(path, encoding="utf-8") as f:
        return [Item(**json.loads(l)) for l in f if l.strip()]


def save_items(items: list, path: str) -> None:
    """Save items to a JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(it.to_json() + "\n")
