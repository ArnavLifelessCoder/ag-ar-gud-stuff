"""EVID-6 abstention analysis.

Scores the abstain-or-answer prompt: given an image and a question, does the
model answer, or does it say CANNOT ANSWER?

Two numbers matter, and reporting either alone is misleading:

  AbsAcc   - how often the abstain/answer decision matches the ground state.
             S0 (and the S0-ctrl control) should be answered; S1-S5 should be
             abstained on.
  OverAbs  - how often the model abstains on an S0 item it could have
             answered.  A model that abstains on everything scores 5/6 on a
             naive accuracy and is useless.

The pairing with S0-ctrl is what makes this diagnostic rather than
descriptive: the control carries the same visual artifact as its parent S2/S3
item but leaves the referent visible.  A model abstaining on the control is
reacting to the artifact, not to the missing evidence.
"""

import re
import json
import numpy as np

# Reuse one normaliser so "CANNOT ANSWER." and "cannot answer" agree.
from consistency import normalise

ABSTAIN_MARKERS = (
    "cannot answer", "can not answer", "cant answer",
    "unable to answer", "i cannot", "i can not",
)

# States on which abstaining is the correct behaviour.
SHOULD_ABSTAIN = {"S1", "S2", "S3", "S4", "S5"}


def is_abstention(text: str) -> bool:
    """Did the model decline to answer?

    The prompt asks for the exact string CANNOT ANSWER, so we match that
    first and fall back to close paraphrases.  Deliberately narrow: treating
    any hedged answer as an abstention would inflate AbsAcc.
    """
    if not text:
        return True
    a = normalise(text)
    return any(m in a for m in ABSTAIN_MARKERS)


def should_abstain(row: dict) -> bool:
    """Ground truth for the abstain decision.

    The s0ctrl condition is labelled S0 and carries a visible referent, so
    the correct behaviour there is to ANSWER despite the artifact.  The
    prior-only condition has the referent painted out, so it should be
    abstained on regardless of the state label it inherited.
    """
    cond = row.get("condition")
    if cond == "s0ctrl":
        return False
    if cond == "prioronly":
        return True
    return row.get("state") in SHOULD_ABSTAIN


def score(results: list) -> list:
    """Attach abstention decisions and correctness to each row."""
    out = []
    for r in results:
        text = r.get("answer") or (r.get("answers") or [""])[0]
        row = dict(r)
        row["abstained"] = is_abstention(text)
        row["should_abstain"] = should_abstain(r)
        row["abs_correct"] = row["abstained"] == row["should_abstain"]
        out.append(row)
    return out


def _rate(rows, field):
    if not rows:
        return None, 0
    return float(np.mean([r[field] for r in rows])), len(rows)


def summarise(results: list) -> dict:
    """AbsAcc, OverAbs, UnderAbs and the per-state breakdown."""
    scored = score(results)
    if not scored:
        return {}

    answerable = [r for r in scored if not r["should_abstain"]]
    unanswerable = [r for r in scored if r["should_abstain"]]

    abs_acc, n = _rate(scored, "abs_correct")
    over_abs, n_ans = _rate(answerable, "abstained")            # false alarm
    under_abs = (None if not unanswerable else
                 float(np.mean([not r["abstained"] for r in unanswerable])))

    per_state = {}
    for r in scored:
        per_state.setdefault(r["state"], []).append(r)
    per_state = {k: {"abstention_rate": float(np.mean([x["abstained"] for x in v])),
                     "n": len(v)}
                 for k, v in sorted(per_state.items())}

    per_condition = {}
    for r in scored:
        per_condition.setdefault(r.get("condition", "main"), []).append(r)
    per_condition = {k: {"abstention_rate": float(np.mean([x["abstained"] for x in v])),
                         "n": len(v)}
                     for k, v in sorted(per_condition.items())}

    return {
        "AbsAcc": abs_acc,
        "n": n,
        "OverAbs": over_abs,
        "n_answerable": n_ans,
        "UnderAbs": under_abs,
        "n_unanswerable": len(unanswerable),
        "always_abstain_baseline": len(unanswerable) / len(scored),
        "per_state": per_state,
        "per_condition": per_condition,
        "scored": scored,
    }


def artifact_sensitivity(results: list) -> dict:
    """Does the model abstain because evidence is missing, or because it sees
    an artifact?

    Compares abstention on s0ctrl (artifact present, referent visible) with
    abstention on clean S0 main items (no artifact, referent visible).  A
    large gap means the abstention signal is partly artifact detection, which
    is a confound the paper has to name.
    """
    scored = score(results)
    ctrl = [r for r in scored if r.get("condition") == "s0ctrl"]
    clean = [r for r in scored
             if r.get("condition") == "main" and r.get("state") == "S0"]
    c_rate, c_n = _rate(ctrl, "abstained")
    k_rate, k_n = _rate(clean, "abstained")
    gap = None if (c_rate is None or k_rate is None) else c_rate - k_rate
    return {
        "s0ctrl_abstention": c_rate, "n_s0ctrl": c_n,
        "s0_clean_abstention": k_rate, "n_s0_clean": k_n,
        "artifact_gap": gap,
        "interpretation": (
            None if gap is None else
            "abstention tracks the artifact, not the missing evidence - "
            "confound, report it" if gap > 0.15 else
            "abstention is not driven by artifact detection"
        ),
    }


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
