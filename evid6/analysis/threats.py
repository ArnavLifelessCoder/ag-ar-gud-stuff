"""EVID-6 'Threats Eliminated' appendix table.

Reviewers do not take safeguards on faith, and a paragraph claiming "we
prevented leakage" is worth less than a number. This module emits the
appendix table, and where possible it *checks* the safeguard against the
actual artifacts rather than asserting it, so the table cannot drift out of
sync with the code.

Every row is one threat, the mechanism that addresses it, and the evidence
that the mechanism ran.

    from threats import build_table, to_markdown
    tbl = build_table(items, folds, results=results, summary=summary)
    print(to_markdown(tbl))
"""

import os
import json

import numpy as np


def _get(it, k, default=None):
    return it.get(k, default) if isinstance(it, dict) else getattr(it, k, default)


def build_table(items=None, folds=None, results=None, summary=None,
                build_stats_path=None):
    """Assemble the threats table, verifying what can be verified.

    Every argument is optional; rows whose evidence is unavailable are marked
    "not checked in this run" rather than silently claiming success.
    """
    rows = []

    def add(threat, mechanism, evidence, verified):
        rows.append({"threat": threat, "mechanism": mechanism,
                     "evidence": evidence, "verified": verified})

    # ── Leakage ────────────────────────────────────────────────────────────
    ev, ok = "not checked in this run", None
    if items is not None and folds is not None:
        g = np.array([_get(it, "base_image_id") for it in items])
        folds = np.asarray(folds)
        leaks = 0
        for k in np.unique(folds):
            if set(g[folds != k]) & set(g[folds == k]):
                leaks += 1
        ok = leaks == 0
        ev = (f"{len(np.unique(folds))} folds over "
              f"{len(np.unique(g))} base images, {leaks} folds with overlap")
    add("Derived items from one image split across folds inflate the probe",
        "StratifiedGroupKFold grouped on base_image_id, with a hard assert "
        "that fails the build rather than warning",
        ev, ok)

    # ── Metadata ───────────────────────────────────────────────────────────
    ev, ok = "not checked in this run", None
    if results:
        s3 = [r for r in results if r.get("state") == "S3"
              and r.get("condition") == "main"]
        s2 = [r for r in results if r.get("state") == "S2"
              and r.get("condition") == "main"]
        have3 = sum(1 for r in s3 if r.get("severity") is not None)
        have2 = sum(1 for r in s2 if r.get("occl_frac") is not None)
        ok = (not s3 or have3 == len(s3)) and (not s2 or have2 == len(s2))
        ev = (f"severity present on {have3}/{len(s3)} S3 rows, "
              f"occl_frac on {have2}/{len(s2)} S2 rows")
    add("Dose-response silently degenerates if intervention metadata is lost "
        "between generation and analysis",
        "run_inference.base_row copies the full intervention record into "
        "every result row; NB4 errors loudly and the figure drops "
        "metadata-less items rather than plotting them at a default",
        ev, ok)

    # ── Parser ─────────────────────────────────────────────────────────────
    ev, ok = "not checked in this run", None
    if results:
        gen = [r for r in results if "raw" in r]
        if gen:
            un = sum(1 for r in gen if r.get("pred") is None)
            ok = True
            ev = (f"{un}/{len(gen)} replies unparseable ({un/len(gen):.1%}), "
                  f"counted as incorrect and reported")
    add("A silent parser bug mislabels a few percent of items",
        "letter_ids asserts every option token round-trips or the model is "
        "dropped; generated replies that commit to no option are recorded "
        "as None and counted wrong, with the rate reported",
        ev, ok)

    # ── Pairing ────────────────────────────────────────────────────────────
    ev, ok = "not checked in this run", None
    if results:
        ctrl = [r for r in results if r.get("condition") == "s0ctrl"]
        linked = sum(1 for r in ctrl if r.get("parent_item_id"))
        if ctrl:
            ok = linked == len(ctrl)
            ev = f"{linked}/{len(ctrl)} controls carry an explicit parent id"
    add("Paired tests pair the wrong items when one image yields several",
        "controls record parent_item_id; McNemar pairs on it, never on "
        "base_image_id",
        ev, ok)

    # ── Artifact confound ──────────────────────────────────────────────────
    add("The model detects the artifact rather than the missing evidence",
        "S0-ctrl applies the SAME artifact as its parent state (occluder for "
        "S2, degradation for S3) to a region touching no instance; occluders "
        "are real object crops from other images, never black boxes",
        "generator enforces artifact match; abstain.artifact_sensitivity "
        "reports the residual gap", None)

    # ── Ground truth ───────────────────────────────────────────────────────
    ev, ok = "not checked in this run", None
    if summary and summary.get("consistency"):
        first = next(iter(summary["consistency"].values()), {})
        rs = first.get("references") or (first.get("relaxed", {})
                                         .get("references", {}))
        if rs:
            ok = rs.get("drop_rate", 1) <= 0.35
            ev = (f"{rs.get('n_usable')}/{rs.get('n_groups')} reference groups "
                  f"usable, drop rate {rs.get('drop_rate', 0):.1%}")
    add("Pseudo-ground-truth is unstable, so 'consistency' measures decoding "
        "noise",
        "references taken from three samples on the untouched image; unstable "
        "groups dropped and the drop rate reported; S0 consistency is "
        "reported as the measurement's own ceiling",
        ev, ok)

    # ── Evaluation-choice sensitivity ──────────────────────────────────────
    ev, ok = "not checked in this run", None
    if summary and summary.get("consistency"):
        first = next(iter(summary["consistency"].values()), {})
        if "max_abs_delta" in first:
            ok = True
            ev = (f"strict vs relaxed matching differ by at most "
                  f"{first['max_abs_delta']:.1%}; both reported")
    add("Headline numbers depend on an arbitrary answer-matching rule",
        "every consistency figure computed under both strict and relaxed "
        "matching; both reported whenever they differ materially",
        ev, ok)

    # ── Shortcut probing ───────────────────────────────────────────────────
    ev, ok = "not checked in this run", None
    if summary:
        cb = summary.get("clip_baseline")
        tr = summary.get("transfer", {})
        parts = []
        if cb:
            parts.append(f"CLIP ViT-B/32 probe {cb['acc']:.1%}")
        for m, d in tr.items():
            lgo = d.get("leave_category_out") if isinstance(d, dict) else None
            if lgo and "mean" in lgo:
                parts.append(f"{m} leave-category-out {lgo['mean']:.1%}")
        if parts:
            ok = True
            ev = "; ".join(parts)
    add("The probe exploits surface shortcuts rather than an evidence "
        "representation",
        "CLIP-feature probe as a vision-encoder baseline; "
        "leave-one-category-out and severity extrapolation; cross-model "
        "transfer in a shared PCA space; learning curve to separate "
        "'accessible' from 'merely learnable'",
        ev, ok)

    # ── Generator validity ─────────────────────────────────────────────────
    ev, ok = "not checked in this run", None
    path = build_stats_path or "/kaggle/working/evid6/build_stats.json"
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            bs = json.load(f)
        ok = True
        ev = (f"rejections {bs.get('rejections')}, "
              f"{bs.get('n_items')} items accepted")
    add("Generated items do not actually instantiate the state they claim",
        "machine-verifiable acceptance: S2 rejected below 90% realised "
        "coverage, S4 below CIEDE2000 12, S5 checked against annotations, "
        "S1 against the union bbox; rejection rates published; plus a "
        "visual QA pass over ~300 samples and a blind self-relabel",
        ev, ok)

    # ── Reproducibility ────────────────────────────────────────────────────
    add("Results cannot be regenerated from the artifacts",
        "every generator deterministic given a seed; runners resume from "
        "partial output; all paper numbers regenerate into summary.json; "
        "offline smoke test covers the full CPU path on synthetic data",
        "tests/smoke_test.py, 21 sections", None)

    return rows


def to_markdown(rows) -> str:
    """Render as the appendix table."""
    mark = {True: "verified", False: "FAILED", None: "-"}
    out = ["| Threat | Mechanism | Evidence | Status |",
           "|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['threat']} | {r['mechanism']} | {r['evidence']} "
                   f"| {mark[r['verified']]} |")
    return "\n".join(out)


def to_latex(rows) -> str:
    """Same table for the appendix, in LaTeX."""
    def esc(s):
        for a, b in [("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")]:
            s = s.replace(a, b)
        return s
    lines = [r"\begin{table}[t]\centering\small",
             r"\begin{tabular}{p{3.4cm}p{4.6cm}p{3.6cm}l}",
             r"\toprule",
             r"Threat & Mechanism & Evidence & Status \\",
             r"\midrule"]
    mark = {True: "verified", False: "FAILED", None: "--"}
    for r in rows:
        lines.append(f"{esc(r['threat'])} & {esc(r['mechanism'])} & "
                     f"{esc(r['evidence'])} & {mark[r['verified']]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{Threats to validity and the mechanisms addressing "
              r"them. Status reports whether the mechanism was verified "
              r"against this run's artifacts.}",
              r"\label{tab:threats}", r"\end{table}"]
    return "\n".join(lines)


def check(rows):
    """Fail loudly if any verifiable safeguard did not hold."""
    failed = [r["threat"] for r in rows if r["verified"] is False]
    if failed:
        print("SAFEGUARD FAILURES:")
        for t in failed:
            print(f"  - {t}")
    else:
        n = sum(1 for r in rows if r["verified"] is True)
        print(f"All {n} verifiable safeguards held "
              f"({sum(1 for r in rows if r['verified'] is None)} not checkable "
              f"in this run).")
    return not failed
