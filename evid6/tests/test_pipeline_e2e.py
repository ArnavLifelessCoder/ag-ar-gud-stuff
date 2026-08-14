"""End-to-end integration test: generator driver through to the P1/P2 verdict.

`smoke_test.py` tests functions in isolation. This runs the actual
``generate.build()`` driver — the reference-group bookkeeping, the auxiliary
conditions, the rejection accounting — against a synthetic COCO-format
fixture, then pushes the real items through the whole analysis path with
simulated model answers.

What it proves: the driver produces a structurally valid dataset and every
downstream stage consumes it without a schema mismatch. What it cannot
prove: that the images look right to a human. For that, run NB1 on real COCO
and open the QA sheets.

    python tests/test_pipeline_e2e.py
"""

import os
import sys
import json
import random
import shutil
import tempfile
from collections import Counter, defaultdict

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("data", "eval", "probe", "analysis"):
    sys.path.insert(0, os.path.join(BASE, _sub))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FAILS = []


def chk(name, cond, detail=""):
    print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


def main(n_per_state=12, keep=False):
    work = tempfile.mkdtemp(prefix="evid6_e2e_")
    fixture = os.path.join(work, "fakecoco")

    print("[1] building COCO-format fixture")
    from make_fake_coco import build_fake_coco
    build_fake_coco(fixture, n_images=160, seed=0)

    print("\n[2] running the real generate.build() driver")
    import generate as g
    g.ROOT = fixture
    g.IMG_DIR = os.path.join(fixture, "val2017")
    g.OUT_DIR = os.path.join(work, "evid6", "images")
    g.init_coco(root=fixture)
    items = g.build(n_per_state=n_per_state, seed=0)

    from schema import save_items, load_items, STATES
    items_path = os.path.join(work, "items.jsonl")
    save_items(items, items_path)
    items = load_items(items_path)

    print("\n[2b] COCO layout autodetection")
    # Kaggle hosts several COCO 2017 datasets with different directory shapes.
    # init_coco(root=...) used to set only where annotations were read from
    # while IMG_DIR kept its hard-coded default, so a non-default layout loaded
    # the annotations and then failed on every image, far from the cause.
    alt = os.path.join(work, "alt_layout", "coco")
    shutil.copytree(fixture, alt)
    os.makedirs(os.path.join(alt, "images"), exist_ok=True)
    shutil.move(os.path.join(alt, "val2017"),
                os.path.join(alt, "images", "val2017"))
    a_root, a_img = g.find_coco(os.path.join(work, "alt_layout"))
    chk("find_coco locates annotations and images in a non-default layout",
        os.path.isfile(os.path.join(a_root, "annotations",
                                    "instances_val2017.json"))
        and a_img.endswith(os.path.join("images", "val2017")),
        os.path.relpath(a_img, work))

    saved = (g.ROOT, g.IMG_DIR, g.OUT_DIR)
    g.OUT_DIR = os.path.join(work, "alt_out")
    g.init_coco(search_root=os.path.join(work, "alt_layout"))
    chk("init_coco updates IMG_DIR, not just the annotation path",
        g.IMG_DIR == a_img, "IMG_DIR used to keep its hard-coded default")
    alt_items = g.build(n_per_state=4, seed=0)
    chk("build() works end to end on the awkward layout",
        alt_items and all(os.path.isfile(i.image_path) for i in alt_items),
        f"{len(alt_items)} items")
    try:
        g.find_coco(os.path.join(work, "definitely_not_here"))
        chk("find_coco raises when no COCO is attached", False, "it returned")
    except FileNotFoundError:
        chk("find_coco raises when no COCO is attached", True,
            "and names the inputs it did see")

    g.ROOT, g.IMG_DIR, g.OUT_DIR = saved
    g.init_coco(root=fixture)

    print("\n[3] dataset invariants")
    chk("every image file exists",
        all(os.path.isfile(i.image_path) for i in items), f"{len(items)} files")
    chk("all six states present",
        len({i.state for i in items if i.condition == "main"}) == 6,
        str(dict(sorted(Counter(i.state for i in items
                                if i.condition == "main").items()))))

    grp = defaultdict(list)
    for i in items:
        if i.ref_group:
            grp[i.ref_group].append(i)
    chk("exactly one clean_ref per reference group",
        {sum(1 for x in v if x.condition == "clean_ref")
         for v in grp.values()} == {1}, f"{len(grp)} groups")
    chk("no non-S5 item is orphaned from a group",
        not [i for i in items if i.state != "S5" and not i.ref_group])

    ids = {i.item_id: i for i in items}
    aux = [i for i in items if i.condition in ("s0ctrl", "prioronly")]
    chk("auxiliary items link to a real main S2/S3 parent",
        all(i.parent_item_id in ids
            and ids[i.parent_item_id].condition == "main"
            and ids[i.parent_item_id].state in ("S2", "S3") for i in aux),
        f"{len(aux)} aux items")

    want = {"S2": "occlude", "S3": "degrade"}
    chk("control artifact matches its parent state (B5)",
        all(i.artifact == want[ids[i.parent_item_id].state]
            for i in items if i.condition == "s0ctrl"),
        str(dict(Counter(f"{ids[i.parent_item_id].state}->{i.artifact}"
                         for i in items if i.condition == "s0ctrl"))))

    print("\n[3a] cross-session image paths (the NB1 -> NB2 break)")
    # NB1 writes absolute paths from its own session. NB2/NB3 see those images
    # under a different root, so every path in the manifest is stale and the
    # first Image.open in build_inputs would kill the pass after the model had
    # already loaded. Simulate that move and check rebase_items repairs it.
    # Imported from schema, NOT run_inference: this suite must stay runnable on
    # a CPU box with no torch and no transformers. run_inference pulls both in
    # at module level, and the transformers auto-class for VLMs was renamed
    # between versions, so importing it here made a CPU-only test fail on an
    # unrelated GPU dependency.
    from schema import rebase_items
    moved_root = os.path.join(work, "as_attached_dataset")
    shutil.copytree(g.OUT_DIR, os.path.join(moved_root, "evid6", "images"))
    stale = os.path.join(work, "stale_items.jsonl")
    with open(items_path, encoding="utf-8") as fsrc, \
         open(stale, "w", encoding="utf-8") as fdst:
        for line in fsrc:
            if line.strip():
                row = json.loads(line)
                row["image_path"] = ("/kaggle/working/evid6/images/"
                                     + os.path.basename(row["image_path"]))
                fdst.write(json.dumps(row) + "\n")

    stale_rows = [json.loads(l) for l in open(stale, encoding="utf-8") if l.strip()]
    chk("stale manifest really is broken (guards the guard)",
        not any(os.path.isfile(r["image_path"]) for r in stale_rows),
        f"{len(stale_rows)} unreadable paths")

    rebased = rebase_items(stale, [moved_root],
                              out_path=os.path.join(work, "items_local.jsonl"))
    rebased_rows = [json.loads(l) for l in open(rebased, encoding="utf-8") if l.strip()]
    chk("rebase_items resolves every image under the new root",
        all(os.path.isfile(r["image_path"]) for r in rebased_rows),
        f"{len(rebased_rows)} items")
    chk("rebase_items preserves item identity and order",
        [r["item_id"] for r in rebased_rows] == [r["item_id"] for r in stale_rows])
    try:
        rebase_items(stale, [os.path.join(work, "nonexistent")],
                        out_path=os.path.join(work, "items_bad.jsonl"))
        chk("rebase_items raises when nothing resolves", False, "it returned")
    except FileNotFoundError:
        chk("rebase_items raises when nothing resolves",
            True, "wrong dataset attached fails loudly, not silently")

    print("\n[3b] image filenames")
    files = sorted(os.listdir(g.OUT_DIR))
    chk("no filename collisions", len(files) == len(set(files)), f"{len(files)} files")
    chk("filenames follow {state}_{condition}_{item_id}.jpg",
        all(os.path.basename(i.image_path)
            == f"{i.state}_{i.condition}_{i.item_id}.jpg" for i in items))
    chk("every file traces back to its OWN item_id",
        all(i.item_id in os.path.basename(i.image_path) for i in items),
        "auxiliary images used to be named after their parent")
    chk("manifest and directory agree both ways",
        set(files) == {os.path.basename(i.image_path) for i in items},
        f"{len(files)} files, {len(items)} rows")

    print("\n[4] intervention metadata")
    s2 = [i for i in items if i.state == "S2" and i.condition == "main"]
    s3 = [i for i in items if i.state == "S3" and i.condition == "main"]
    s4 = [i for i in items if i.state == "S4" and i.condition == "main"]
    chk("S2 coverage >= 0.90", all(i.occl_frac >= 0.90 for i in s2),
        f"range {min(i.occl_frac for i in s2):.3f}-{max(i.occl_frac for i in s2):.3f}")
    chk("S4 delta_e >= 12", all(i.delta_e >= 12 for i in s4),
        f"min {min(i.delta_e for i in s4):.1f}")

    by_sev = defaultdict(list)
    for i in s3:
        by_sev[i.severity].append(i.eff_res)
    chk("S3 effective resolution is constant per severity",
        all(max(v) - min(v) <= 1 for v in by_sev.values()),
        ", ".join(f"sev{k}->{np.mean(v):.0f}px" for k, v in sorted(by_sev.items())))
    chk("no S3 referent collapses below 8px (S3 would become S2)",
        all(i.eff_res >= 8 for i in s3),
        f"min {min(i.eff_res for i in s3):.0f}px")

    print("\n[5] determinism and splits")
    # Compare the FULL row, not just item_id. uid() does not hash the
    # question, so an id-only check passed while question_for() drew from the
    # global random stream and reassigned questions on 38 of 68 items between
    # two identically seeded builds. The question is the stimulus; if it moves,
    # the dataset is not reproducible whatever the ids say.
    rebuilt = g.build(n_per_state=n_per_state, seed=0)
    def _sig(i):
        return (i.item_id, i.state, i.condition, i.category, i.question,
                os.path.basename(i.image_path))
    chk("build() is deterministic given a seed",
        [_sig(i) for i in rebuilt] == [_sig(i) for i in items])
    chk("question assignment is reproducible across builds",
        {i.item_id: i.question for i in rebuilt}
        == {i.item_id: i.question for i in items},
        "question_for() must draw from the build's seeded rng")

    # Same seed, DIFFERENT PROCESS. The check above cannot see hash-ordering
    # bugs: it builds twice in one interpreter, where PYTHONHASHSEED is fixed.
    # `list(ALL_CAT_NAMES - present)` iterated a set of strings, so rng.choice
    # picked a different absent category in every new session and S5 — a sixth
    # of the benchmark — was entirely different run to run.
    import subprocess
    probe = os.path.join(work, "probe_determinism.py")
    with open(probe, "w", encoding="utf-8") as f:
        f.write(
            "import sys, os, json\n"
            f"for s in {[os.path.join(BASE, s) for s in ('data','eval','probe','analysis')]!r}:\n"
            "    sys.path.insert(0, s)\n"
            "import generate as g\n"
            f"g.OUT_DIR = {os.path.join(work, 'det_images')!r}\n"
            f"g.init_coco(root={fixture!r})\n"
            f"items = g.build(n_per_state={n_per_state}, seed=0)\n"
            "print('SIG', json.dumps([[i.item_id, i.state, i.category, i.question]\n"
            "                          for i in items]))\n")
    sigs = []
    for hashseed in ("0", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hashseed)
        out = subprocess.run([sys.executable, probe], capture_output=True,
                             text=True, env=env)
        line = next((l for l in out.stdout.splitlines() if l.startswith("SIG")), None)
        sigs.append(line)
    chk("build() is deterministic ACROSS processes (hash-seed independent)",
        sigs[0] is not None and sigs[0] == sigs[1],
        "sets of strings must be sorted() before an rng draws from them")
    from splits import make_folds
    main_items = [i for i in items if i.condition == "main"]
    folds = make_folds(main_items, n_splits=5, seed=0)
    chk("StratifiedGroupKFold asserts no base-image leakage", True,
        f"{len(main_items)} main items across 5 folds")

    print("\n[6] analysis path on the real items (simulated answers)")
    rng = random.Random(0)
    colours = ["red", "blue", "green", "yellow", "black"]
    truth, clean, treat = {}, [], []
    for it in items:
        if it.condition == "clean_ref":
            truth[it.ref_group] = rng.choice(colours)
            clean.append({"item_id": it.item_id, "condition": "clean_ref",
                          "state": it.state, "ref_group": it.ref_group,
                          "answers": [truth[it.ref_group]] * 3, "stable": True})
    for it in items:
        if it.condition == "clean_ref" or it.state == "S5" or not it.ref_group:
            continue
        p = ({"prioronly": 0.20, "s0ctrl": 0.90}.get(it.condition)
             or {"S0": 0.95, "S1": 0.20, "S2": 0.22, "S4": 0.55}.get(it.state)
             or {32: 0.75, 16: 0.50, 8: 0.28}.get(int(it.eff_res or 16), 0.5))
        treat.append({
            "item_id": it.item_id, "condition": it.condition, "state": it.state,
            "ref_group": it.ref_group, "parent_item_id": it.parent_item_id,
            "severity": it.severity, "eff_res": it.eff_res,
            "occl_frac": it.occl_frac, "category": it.category,
            "answer": truth[it.ref_group] if rng.random() < p
            else rng.choice(colours),
        })

    from consistency import (build_references, score_consistency,
                             summarise_both, by_state_condition)
    refs, rstats = build_references(clean)
    chk("references built from clean_ref rows", rstats["n_usable"] > 0,
        f"{rstats['n_usable']}/{rstats['n_groups']} usable")
    scored = score_consistency(treat, refs)
    chk("every treatment row resolved to a reference",
        len(scored) == len(treat), f"{len(scored)}/{len(treat)}")

    both = summarise_both(treat, refs, rstats)
    summ = both["relaxed"]
    chk("S0 ceiling above the prior floor",
        summ["s0_ceiling"][0] > summ["prior_floor_pooled"][0],
        f"ceiling {summ['s0_ceiling'][0]:.0%} vs floor "
        f"{summ['prior_floor_pooled'][0]:.0%}")
    chk("S3 dose-response populated on the eff_res regressor",
        len(summ["dose_S3_eff_res"]) >= 2,
        ", ".join(f"{l}:{r:.0%}" for l, r, _ in summ["dose_S3_eff_res"]))
    chk("P1/P2 verdict produced",
        "P1" in summ["verdict"] and "P2" in summ["verdict"])

    from stats import pair_on_field, paired_test
    a, b, n = pair_on_field(
        [r for r in scored if r["condition"] == "main"],
        [dict(r, item_id=r["parent_item_id"]) for r in scored
         if r["condition"] == "prioronly" and r.get("parent_item_id")],
        field="consistent")
    chk("main vs prior-only pairs on parent_item_id", n > 0,
        f"n={n}, {a.mean():.0%} vs {b.mean():.0%}, p={paired_test(a, b):.3f}")

    print("\n[7] visual QA artifacts")
    from qa_sheet import contact_sheets, triptychs
    qa = os.path.join(work, "qa")
    made = contact_sheets(items, qa, per_state=12, cols=6, seed=0)
    tri = triptychs(items, qa, n=6, seed=0)
    chk("contact sheets for every state", len(made) == 6, f"{len(made)} sheets")
    chk("triptychs rendered", tri is not None)
    chk("index.html written", os.path.isfile(os.path.join(qa, "index.html")))

    print("\n[8] threats table against the real artifacts")
    from threats import build_table, check
    tt = build_table(items=main_items, folds=folds,
                     results=[dict(t, pred="A") for t in treat],
                     summary={"consistency": {"m": dict(
                         summ, max_abs_delta=both["max_abs_delta"])}},
                     build_stats_path=os.path.join(work, "evid6",
                                                   "build_stats.json"))
    chk("no safeguard reported as failed", check(tt))

    if keep:
        print(f"\nartifacts kept in {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)

    print()
    if FAILS:
        print(f"FAILURES: {FAILS}")
        return 1
    print("END-TO-END PIPELINE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main(keep="--keep" in sys.argv))
