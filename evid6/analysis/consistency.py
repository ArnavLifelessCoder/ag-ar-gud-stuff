"""EVID-6 self-consistency scoring.

This is the metric v3 is built around.  There are no ground-truth answers.
Instead, for every reference group we take the model's own answer on the
untouched image (``condition == "clean_ref"``) and ask whether that answer
survives the evidence intervention.

Reading the numbers:
  - Consistency on S0 is the ceiling.  It is not 100% because greedy decoding
    on the same image can still drift; report it as the noise floor of the
    measure itself.
  - Consistency on the prior-only condition is the *prior floor*: the rate at
    which the model reproduces its answer with the referent painted out, i.e.
    the rate at which it was never using the evidence at all.
  - The interesting quantity is main-condition consistency relative to that
    floor, as a function of intervention dose.

S5 is excluded throughout: the queried category is absent, so there is no
evidence to remove and no reference worth comparing against.
"""

import re
import json
import numpy as np

# ── Answer normalisation ────────────────────────────────────────────────────

_ARTICLES = {"a", "an", "the"}
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")

_NUMBERS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10",
}

_CONTRACTIONS = {
    "dont": "do not", "cant": "can not", "isnt": "is not",
    "theres": "there is", "its": "it is",
}

# Phrases models emit instead of answering.  Counting these as "consistent"
# with a real reference answer would be a scoring bug.
#
# These are matched as substrings of the NORMALISED answer, so every entry has
# to be a phrase that cannot open a real answer.  "no " was not: it made
# "no parking sign", "no hat" and "No, it is a cat" all read as refusals, which
# both deflated consistency and inflated the reference drop rate toward the
# 13 Aug kill criterion.  "stop sign" is in the TEXTISH question pool, so this
# was live, not hypothetical.  The genuine no-answer sense is covered by the
# explicit phrases below.
REFUSAL_MARKERS = (
    "cannot answer", "can not answer", "cant answer", "unable to",
    "not visible", "cannot tell", "cannot determine", "unclear",
    "no idea", "no way to tell", "not sure", "unknown",
    "i don't know", "i do not know",
)

# Answers that are refusals only when they are the WHOLE answer.  "nothing" as
# a complete reply is a non-answer; "nothing on the sign" is a real one.
# Matched against the answer with punctuation stripped but articles KEPT -
# ``normalise`` drops "a", which turns "N/A" into a bare "n".
REFUSAL_EXACT = frozenset({
    "n a", "na", "none", "no answer", "cannot", "can not", "do not know",
    "not applicable", "not enough information", "nothing",
})


def normalise(ans: str) -> str:
    """VQA-style answer normalisation.

    Lowercase, strip punctuation and articles, map number words to digits,
    collapse whitespace.  Deliberately conservative: it should merge
    "A red one." with "red", not merge "red" with "dark red".
    """
    if ans is None:
        return ""
    a = ans.strip().lower()
    a = _PUNCT.sub(" ", a)
    a = _WS.sub(" ", a).strip()
    toks = []
    for t in a.split():
        t = _CONTRACTIONS.get(t, t)
        t = _NUMBERS.get(t, t)
        if t in _ARTICLES:
            continue
        toks.append(t)
    return " ".join(toks)


def _normalise_keep_articles(ans: str) -> str:
    """``normalise`` without article stripping, for whole-answer matching."""
    if ans is None:
        return ""
    a = _WS.sub(" ", _PUNCT.sub(" ", ans.strip().lower())).strip()
    return " ".join(_CONTRACTIONS.get(t, t) for t in a.split())


def is_refusal(ans: str) -> bool:
    """True if the answer is a non-answer (abstention or hedge)."""
    a = normalise(ans)
    if not a:
        return True
    if _normalise_keep_articles(ans) in REFUSAL_EXACT:
        return True
    return any(m in a for m in REFUSAL_MARKERS)


def agree(a: str, b: str, relaxed: bool = False) -> bool:
    """Do two free-form answers agree?

    Exact match after normalisation by default.  The optional relaxed rule
    treats one answer's token set being a subset of the other's as agreement
    ("red" vs "red and white").  It is a sensitivity analysis only: shorter
    answers can otherwise look spuriously consistent after degradation.
    Refusals never agree with anything, including each other.
    """
    if is_refusal(a) or is_refusal(b):
        return False
    na, nb = normalise(a), normalise(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if not relaxed:
        return False
    ta, tb = na.split(), nb.split()
    short, long_ = (ta, tb) if len(ta) <= len(tb) else (tb, ta)
    return set(short).issubset(set(long_))


# ── Reference table ─────────────────────────────────────────────────────────

def build_references(clean_results, require_stable=True):
    """Map ref_group -> reference answer, from the clean_ref generations.

    Parameters
    ----------
    clean_results : list of dict
        Rows written by ``run_generation`` on the clean prompt.  Each needs
        ``ref_group``, ``answers`` (list) and ``stable`` (bool).
    require_stable : {True, "unanimous", "majority", False}
        How much sample agreement a group needs to contribute a reference.

        ``True`` / ``"unanimous"`` - all three samples must match. **This is the
        pre-registered rule** and the one the 13 Aug kill criterion (>35% drop)
        was written against.

        ``"majority"`` - the modal answer wins if at least two of three samples
        agree; the reference becomes that modal answer rather than
        ``answers[0]``. **Post-hoc.** It was added after unanimity failed its
        gate on the closed-set rerun (41.0% drop for Qwen), and anything
        reported under it must say so. On that data it yields 7.0% drop, but it
        also lowers the S0 ceiling from 96.6% to 88.0% - the reference is
        genuinely noisier, which is the price of the yield. Worse, the P1
        verdict *flips* between the two rules (gap +5.0% vs +5.6% against a
        5-point tolerance), so P1 is not robust to this choice and should not
        be claimed under either.

        ``False`` - keep every group.

    Returns
    -------
    refs : dict, ref_group -> reference answer string
    stats : dict with n_groups, n_stable, n_usable, drop_rate, n_refusal, rule
    """
    if require_stable is True:
        require_stable = "unanimous"
    if require_stable not in ("unanimous", "majority", False):
        raise ValueError(f"require_stable must be 'unanimous', 'majority' or "
                         f"False, got {require_stable!r}")

    refs, n_total, n_stable, n_refusal = {}, 0, 0, 0
    for r in clean_results:
        if r.get("condition") != "clean_ref":
            continue
        grp = r.get("ref_group")
        if grp is None:
            continue
        n_total += 1
        answers = r.get("answers") or ([r["answer"]] if r.get("answer") else [])
        if not answers:
            continue
        if is_refusal(answers[0]):
            n_refusal += 1
            continue

        norm = [normalise(a) for a in answers]
        counts = {}
        for a in norm:
            counts[a] = counts.get(a, 0) + 1
        top, top_n = max(counts.items(), key=lambda kv: (kv[1], kv[0]))

        stable = r.get("stable")
        if stable is None:
            stable = len(set(norm)) == 1
        if stable:
            n_stable += 1

        if require_stable == "unanimous":
            if not stable:
                continue
            refs[grp] = answers[0]
        elif require_stable == "majority":
            if top_n < 2:
                continue          # all three samples differ: genuinely undefined
            refs[grp] = top       # the MODAL answer, not answers[0]
        else:
            refs[grp] = answers[0]

    stats = {
        "n_groups": n_total,
        "n_stable": n_stable,
        "n_refusal": n_refusal,
        "n_usable": len(refs),
        "drop_rate": 1 - (len(refs) / n_total) if n_total else 0.0,
        # Which rule produced these references. Recorded so a downstream table
        # cannot silently mix the pre-registered rule with the post-hoc one.
        "rule": require_stable if require_stable else "none",
        "pre_registered": require_stable == "unanimous",
    }
    return refs, stats


# ── Scoring ─────────────────────────────────────────────────────────────────

def score_consistency(treat_results, refs, relaxed: bool = False):
    """Attach a boolean ``consistent`` to every treatment row that has a ref.

    Parameters
    ----------
    treat_results : list of dict
        Rows from the clean prompt on *degraded* images (conditions main,
        s0ctrl, prioronly).  Each needs ``ref_group`` and ``answer``.
    refs : dict from ``build_references``.

    Returns
    -------
    list of dict - the scored subset (rows without a usable reference are
    dropped, and the count is reported by ``summarise``).
    """
    out = []
    for r in treat_results:
        if r.get("condition") == "clean_ref":
            continue
        grp = r.get("ref_group")
        if grp is None or grp not in refs:
            continue
        ans = r.get("answer") or (r.get("answers") or [""])[0]
        row = dict(r)
        row["ref_answer"] = refs[grp]
        row["answer_norm"] = normalise(ans)
        row["refused"] = is_refusal(ans)
        row["consistent"] = agree(ans, refs[grp], relaxed=relaxed)
        out.append(row)
    return out


def _rate(rows):
    if not rows:
        return None, 0
    return float(np.mean([r["consistent"] for r in rows])), len(rows)


def by_state_condition(scored):
    """Consistency broken down by (state, condition).

    Returns dict {(state, condition): (rate, n)}.
    """
    buckets = {}
    for r in scored:
        buckets.setdefault((r["state"], r["condition"]), []).append(r)
    return {k: _rate(v) for k, v in sorted(buckets.items())}


def prior_floor(scored, state=None):
    """Prior-only floor: consistency with the referent painted out.

    Agreement here is the rate at which the model was answering from priors
    rather than from the evidence.  Pass ``state`` to get the floor for one
    state, or leave it None for the pooled floor.
    """
    rows = [r for r in scored if r["condition"] == "prioronly"
            and (state is None or r["state"] == state)]
    return _rate(rows)


def dose_response(scored, state, dose_field, bins=None):
    """Consistency as a function of intervention dose.

    Parameters
    ----------
    state : "S2" or "S3"
    dose_field : "occl_frac" (continuous) or "severity" (ordinal)
    bins : sequence of bin edges for continuous doses, or None for ordinal.

    Returns
    -------
    list of (dose_label, rate, n)
    """
    rows = [r for r in scored
            if r["state"] == state and r["condition"] == "main"
            and r.get(dose_field) is not None]
    if not rows:
        return []

    if bins is None:
        buckets = {}
        for r in rows:
            buckets.setdefault(r[dose_field], []).append(r)
        return [(k,) + _rate(v) for k, v in sorted(buckets.items())]

    edges = np.asarray(bins, dtype=float)
    buckets = {}
    for r in rows:
        i = int(np.clip(np.digitize(r[dose_field], edges) - 1, 0, len(edges) - 2))
        buckets.setdefault(i, []).append(r)
    out = []
    for i in sorted(buckets):
        label = f"{edges[i]:.2f}-{edges[i+1]:.2f}"
        out.append((label,) + _rate(buckets[i]))
    return out


def p1_p2_verdict(scored, flat_tol: float = 0.05):
    """The taxonomy's falsifiable content, scored.

    P1: occlusion (S2) destroys signal, so main consistency should sit at the
        prior-only floor and not vary with dose.
    P2: degradation (S3) attenuates signal, so consistency should decline
        monotonically with severity and stay above the floor until the
        referent is unresolvable.

    Returns a dict with the numbers and a plain-language verdict for each.
    Collapsing P1 and P2 is a real outcome, not a failure: the plan says
    report five states in that case.
    """
    v = {}

    s2_main, n2 = _rate([r for r in scored
                         if r["state"] == "S2" and r["condition"] == "main"])
    s2_floor, n2f = prior_floor(scored, "S2")
    if s2_main is not None and s2_floor is None:
        v["P1"] = {
            "s2_main": s2_main, "n_main": n2,
            "s2_prior_floor": None, "n_floor": n2f,
            "gap": None,
            "verdict": "unknown (missing S2 prior-only floor)",
        }
    elif s2_main is not None:
        gap = s2_main - s2_floor
        v["P1"] = {
            "s2_main": s2_main, "n_main": n2,
            "s2_prior_floor": s2_floor, "n_floor": n2f,
            "gap": gap,
            "verdict": ("supported (S2 sits at the prior floor)"
                        if abs(gap) <= flat_tol else
                        "challenged (S2 retains signal above the prior floor)"),
        }

    curve = dose_response(scored, "S3", "severity")
    s3_floor, n3f = prior_floor(scored, "S3")
    if curve and s3_floor is None:
        v["P2"] = {
            "curve": curve,
            "s3_prior_floor": None, "n_floor": n3f,
            "monotone_decline": all(
                curve[i][1] >= curve[i + 1][1] for i in range(len(curve) - 1)
            ),
            "stays_above_floor": None,
            "verdict": "unknown (missing S3 prior-only floor)",
        }
    elif curve:
        rates = [c[1] for c in curve]
        monotone = all(rates[i] >= rates[i + 1] for i in range(len(rates) - 1))
        above = min(rates) > s3_floor + flat_tol
        v["P2"] = {
            "curve": curve,
            "s3_prior_floor": s3_floor, "n_floor": n3f,
            "monotone_decline": monotone,
            "stays_above_floor": above,
            "verdict": ("supported (monotone decline, above the prior floor)"
                        if monotone and above else
                        "challenged (" + ", ".join(
                            ([] if monotone else ["non-monotone"]) +
                            ([] if above else ["reaches the prior floor"])) + ")"),
        }

    if "P1" in v and "P2" in v and v["P1"]["gap"] is not None \
            and v["P2"]["stays_above_floor"] is not None:
        s3_min = min(c[1] for c in v["P2"]["curve"])
        v["collapse"] = abs(v["P1"]["s2_main"] - s3_min) <= flat_tol
        v["note"] = ("S2 and S3 curves coincide - report five states, per the "
                     "pre-registered plan." if v["collapse"] else
                     "S2 and S3 separate - the six-state taxonomy earns its "
                     "extra state.")
    return v


def summarise_both(treat_results, refs, ref_stats):
    """Score the same data under strict AND relaxed matching.

    Relaxed matching counts "red" as agreeing with "red and white"; strict
    demands exact equality after normalisation.  The choice moves the
    headline numbers, so the paper should report both whenever they differ
    meaningfully.  ``delta`` is the size of that dependence - if it is small,
    say so and move on; if it is large, the relaxed number needs defending.

    Returns
    -------
    dict with keys "relaxed", "strict", "delta" and "sensitive".
    """
    out = {}
    for name, relaxed in [("strict", False), ("relaxed", True)]:
        sc = score_consistency(treat_results, refs, relaxed=relaxed)
        out[name] = summarise(sc, refs, ref_stats)

    def _pull(d, key):
        v = d.get(key)
        return v[0] if isinstance(v, tuple) else v

    deltas = {}
    for k in ["s0_ceiling", "prior_floor_pooled", "s0ctrl"]:
        a, b = _pull(out["relaxed"], k), _pull(out["strict"], k)
        if a is not None and b is not None:
            deltas[k] = a - b
    for key in out["relaxed"]["by_state_condition"]:
        a = out["relaxed"]["by_state_condition"][key][0]
        b = out["strict"]["by_state_condition"].get(key, (None,))[0]
        if a is not None and b is not None:
            deltas[key] = a - b

    biggest = max((abs(v) for v in deltas.values()), default=0.0)
    out["delta"] = deltas
    out["max_abs_delta"] = biggest
    out["sensitive"] = biggest > 0.05
    out["primary"] = "strict"
    # Strict is the headline (see ``agree``); relaxed is the sensitivity arm.
    # This note used to say "justify the relaxed one" / "report relaxed", which
    # contradicted that and is printed verbatim by NB4 - the easiest possible
    # way for the write-up to quietly revert to the weaker rule.
    out["note"] = (
        f"headline is STRICT matching; relaxed differs by up to {biggest:.1%}, "
        "which exceeds the 5-point threshold - report the relaxed column beside "
        "it as a sensitivity analysis and say the conclusion is rule-dependent"
        if biggest > 0.05 else
        f"headline is STRICT matching; relaxed changes nothing material (max "
        f"delta {biggest:.1%}), so the choice of rule does not carry the result"
    )
    return out


def summarise(scored, refs, ref_stats):
    """One dict holding everything the paper needs from this analysis."""
    return {
        "references": ref_stats,
        "n_scored": len(scored),
        "n_refusals": int(sum(r["refused"] for r in scored)),
        "by_state_condition": {f"{k[0]}|{k[1]}": v
                               for k, v in by_state_condition(scored).items()},
        "prior_floor_pooled": prior_floor(scored),
        "s0_ceiling": _rate([r for r in scored
                             if r["state"] == "S0" and r["condition"] == "main"]),
        "s0ctrl": _rate([r for r in scored if r["condition"] == "s0ctrl"]),
        "dose_S2_occl": dose_response(scored, "S2", "occl_frac",
                                      bins=[0.90, 0.93, 0.96, 1.001]),
        "dose_S3_severity": dose_response(scored, "S3", "severity"),
        "dose_S3_eff_res": dose_response(scored, "S3", "eff_res",
                                         bins=[0, 10, 20, 40, 1e9]),
        "verdict": p1_p2_verdict(scored),
    }


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
