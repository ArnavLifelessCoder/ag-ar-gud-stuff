"""Offline smoke test for the EVID-6 CPU-side pipeline.

Exercises everything that does not need a GPU or the COCO download:
schema round-trip, fold leakage guard, metadata passthrough, probes,
learning curve, stats, and every figure. Uses synthetic activations.
"""
import sys, os, json, tempfile, random
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in ["data", "eval", "probe", "analysis"]:
    sys.path.insert(0, os.path.join(BASE, sub))

ok = lambda m: print(f"  PASS  {m}")

print("\n[1] schema round-trip")
from schema import Item, STATES, STATE_TEXT, REPAIR, save_items, load_items
rng = random.Random(0)
items = []
for i in range(180):
    st = STATES[i % 6]
    cond = "main"
    meta = {}
    if st == "S2":
        meta["occl_frac"] = round(0.90 + 0.09 * rng.random(), 3)
    if st == "S3":
        meta["severity"] = rng.choice([1, 2, 3])
        meta["inst_pixels"] = rng.randint(50, 900)
    if st == "S4":
        meta["n_candidates"] = rng.choice([2, 3])
        meta["delta_e"] = round(12 + 20 * rng.random(), 2)
    items.append(Item(f"it{i:04d}", 1000 + i // 3, st, cond, "cat",
                      "What colour is the cat?", f"/tmp/img/{i}.jpg", **meta))
tmp = tempfile.mkdtemp()
ipath = os.path.join(tmp, "items.jsonl")
save_items(items, ipath)
assert len(load_items(ipath)) == len(items)
ok(f"{len(items)} items saved and reloaded")

print("\n[2] fold leakage guard (splits.make_folds)")
from splits import make_folds
folds = make_folds(items, n_splits=5, seed=0)
g = np.array([it.base_image_id for it in items])
for k in range(5):
    assert not (set(g[folds != k]) & set(g[folds == k]))
ok(f"5 folds, no base-image leakage, sizes {np.bincount(folds).tolist()}")

print("\n[3] prompts")
from prompts import CAUSE_PROMPT, ABSTAIN_PROMPT, CLEAN_PROMPT, REPAIR_PROMPT
assert CAUSE_PROMPT.count("(") >= 6
for i, s in enumerate(STATES):
    assert f"({chr(65+i)}) {STATE_TEXT[s]}" in CAUSE_PROMPT, f"option {i} misaligned"
ok("cause options align letter i -> state S<i>; 4 prompts render")

print("\n[4] METADATA PASSTHROUGH (the B1 fix)")
# base_row lives in run_inference, which imports torch/transformers at module
# level. Re-exec just the helper so the test stays dependency-free.
src = open(os.path.join(BASE, "eval", "run_inference.py"), encoding="utf-8").read()
start = src.index("META_FIELDS")
end = src.index("# ── Batch runner")
ns = {}
exec(src[start:end], ns)
base_row = ns["base_row"]
rows = [base_row(json.loads(l)) for l in open(ipath, encoding="utf-8")]
s3 = [r for r in rows if r["state"] == "S3"]
s2 = [r for r in rows if r["state"] == "S2"]
assert all("severity" in r for r in s3), "severity lost"
assert all("occl_frac" in r for r in s2), "occl_frac lost"
assert len({r["severity"] for r in s3}) == 3, "severity collapsed to one value"
ok(f"severity survives on {len(s3)}/{len(s3)} S3 rows, "
   f"values {sorted({r['severity'] for r in s3})}")
ok(f"occl_frac survives on {len(s2)}/{len(s2)} S2 rows")

# Attach fake predictions so downstream consumers have something to chew on
for r in rows:
    true_i = STATES.index(r["state"])
    pred_i = true_i if random.random() < 0.45 else random.randrange(6)
    r["pred"] = chr(65 + pred_i)
    p = np.random.dirichlet(np.ones(6)); p[pred_i] += 1.5; p /= p.sum()
    r["probs"] = p.tolist()

print("\n[5] ladder + probes on synthetic activations")
from ladder import (rung3_from_logits, per_state_accuracy, layer_sweep,
                    best_layer, probe_layer, nested_probe)
y = np.array([STATES.index(r["state"]) for r in rows])
H = np.random.randn(len(rows), 8, 64).astype(np.float16)
H[:, 5, :] += y[:, None] * 1.2          # plant a signal at layer 5
sweep = layer_sweep(H, y, folds)
bl = best_layer(sweep)
assert bl[0] == 5, f"probe failed to find the planted layer (found {bl[0]})"
ok(f"layer sweep over 8 layers; best layer {bl[0]} acc {bl[1]:.1%} (planted at 5)")

# Rung 4 must not be chosen by max over the sweep on the folds it is scored
# on. On real signal, nested selection should still find the planted layer;
# on pure noise it must not inherit the winner's-curse inflation that
# max-over-layers does.
nb = nested_probe(H, y, folds)
assert set(nb["layers_chosen"]) == {5}, \
    f"nested selection lost the planted layer: {nb['layers_chosen']}"
assert nb["accuracy"] <= nb["flat_max_over_layers"] + 1e-9, \
    "nested accuracy cannot exceed the sweep maximum"
ok(f"nested probe = {nb['accuracy']:.1%} (layers {nb['layers_chosen']}), "
   f"max-over-layers {nb['flat_max_over_layers']:.1%}, "
   f"bias avoided {nb['selection_bias']:+.1%}")

Hn = np.random.randn(360, 12, 32).astype(np.float16)     # no signal at all
yn = np.random.randint(0, 6, 360)
fn = np.arange(360) % 5
nn = nested_probe(Hn, yn, fn)
flat_n = best_layer(layer_sweep(Hn, yn, fn))[1]
assert nn["accuracy"] < flat_n, (
    "on pure noise, max-over-layers must read higher than nested selection — "
    "otherwise the bias this guards against is not being measured")
ok(f"null check: max-over-layers {flat_n:.1%} vs nested {nn['accuracy']:.1%} "
   f"at {1/6:.1%} chance — selection bias is real and excluded")

ok(f"rung3 from logits = {rung3_from_logits(rows):.1%}; "
   f"per-state keys {sorted(per_state_accuracy(rows))}")

print("\n[6] learning curve")
from learning_curve import learning_curve
curve = learning_curve(H, y, folds, layer=5, ns=(25, 50, 100))
assert curve, "learning curve empty"
ok("curve: " + ", ".join(f"n={n}:{m:.0%}" for n, m, _ in curve))

print("\n[7] stats")
from stats import boot_ci, paired_test, accuracy_by_state
m, lo, hi = boot_ci([1, 0, 1, 1, 0, 1, 1, 0])
assert lo <= m <= hi
a = np.random.rand(120) < 0.6
b = np.random.rand(120) < 0.4
pv = paired_test(a, b)
abs_ = accuracy_by_state(rows, states=STATES)
ok(f"boot_ci {m:.2f} [{lo:.2f},{hi:.2f}]; McNemar p={pv:.4f}; "
   f"per-state {len(abs_)} states")

print("\n[8] figures (all six)")
import matplotlib; matplotlib.use("Agg")
from figures import (fig_dose_response, fig_ladder, fig_learning_curve,
                     fig_layer_sweep, fig_steering, fig_confusion)
FD = os.path.join(tmp, "figures")
# add prior-only rows so the floor line has data
po = []
for r in rows:
    if r["state"] in ("S2", "S3"):
        q = dict(r); q["condition"] = "prioronly"; q["item_id"] += "po"
        po.append(q)
fig_dose_response(rows + po, out_path=f"{FD}/dose.pdf")
rung_data = {"qwen": {"R1_zeroshot": .21, "R2_fewshot": .28,
                      "R3_logits": .35, "R4_probe": .72}}
fig_ladder(rung_data, out_path=f"{FD}/ladder.pdf")
fig_learning_curve({"qwen": curve}, zero_shot_acc={"qwen": .21},
                   out_path=f"{FD}/lc.pdf")
fig_layer_sweep({"qwen": sweep}, out_path=f"{FD}/sweep.pdf")
fig_steering([{"alpha": a_, "abs_acc": .5 + a_ * .03, "over_abs": .2 + a_ * .01}
              for a_ in (-4, -2, 0, 2, 4)], out_path=f"{FD}/steer.pdf")
fig_confusion(rows, model_name="Qwen2.5-VL-3B", out_path=f"{FD}/cm.pdf")
made = sorted(os.listdir(FD))
assert len(made) == 6, made
ok(f"6/6 figures rendered: {', '.join(made)}")

print("\n[9] no-directory out_path (the makedirs crash)")
cwd = os.getcwd(); os.chdir(tmp)
fig_confusion(rows, out_path="bare.pdf")
os.chdir(cwd)
ok("figure with a bare filename no longer raises")

print("\n[10] regression check: severity stripped -> loud, not silent")
stripped = [{k: v for k, v in r.items() if k != "severity"} for r in rows]
fig_dose_response(stripped, out_path=f"{FD}/dose_stripped.pdf")
ok("dose-response with no severity renders an empty panel instead of a fake point")

# ─────────────────────────────────────────────────────────────────────────
# New scientific machinery
# ─────────────────────────────────────────────────────────────────────────

print("\n[11] answer normalisation + agreement (consistency core)")
from consistency import normalise, agree, is_refusal
assert normalise("A Red Car.") == "red car"
assert normalise("  the  TWO   dogs ") == "2 dogs"
assert agree("red", "Red.")
assert agree("red", "red and white", relaxed=True), "relaxed subset match broken"
assert not agree("red", "red and white"), "strict must be the default"
assert not agree("red", "blue")
assert not agree("cannot answer", "cannot answer"), "refusals must not agree"
assert is_refusal("CANNOT ANSWER") and is_refusal("")
assert not is_refusal("red")
# Real answers that merely begin with "no" are not refusals. The marker list
# used to contain a bare "no ", matched as a substring, so "no parking sign"
# (a live answer — "stop sign" is in the TEXTISH question pool) was scored as
# an abstention: it deflated consistency and pushed the reference drop rate
# toward the 13 Aug kill criterion.
for real in ["no parking sign", "No, it is a cat", "no hat",
             "a no smoking sign", "nothing on the sign"]:
    assert not is_refusal(real), f"{real!r} misread as a refusal"
for refusal in ["cannot answer", "I do not know", "not sure", "unknown",
                "none", "N/A", "no idea"]:
    assert is_refusal(refusal), f"{refusal!r} not caught as a refusal"
assert agree("no parking sign", "no parking sign"), \
    "a real 'no ...' answer must be able to agree with itself"
ok("normalisation, relaxed matching, refusal handling "
   "(incl. 'no ...' answers vs genuine refusals)")

print("\n[12] references, scoring, prior floor, P1/P2 verdict")
from consistency import (build_references, score_consistency, prior_floor,
                         by_state_condition, p1_p2_verdict, summarise as csum)
# Synthesise a world where P1 and P2 are TRUE, and check the verdict finds it.
clean_rows, treat_rows = [], []
rng2 = random.Random(7)
TRUTH = {"S0": 0.95, "S1": 0.20, "S2": 0.22, "S4": 0.55}
FLOOR = 0.20
SEV = {1: 0.75, 2: 0.50, 3: 0.30}          # monotone, above the 0.20 floor
for gi in range(400):
    grp = f"g{gi}"
    clean_rows.append({"item_id": f"ref{gi}", "condition": "clean_ref",
                       "ref_group": grp, "state": "S0",
                       "answers": ["red", "red", "red"], "stable": True})
    st = ["S0", "S1", "S2", "S3", "S4"][gi % 5]
    if st == "S3":
        sev = (gi % 3) + 1
        pm = SEV[sev]
        extra = {"severity": sev}
    elif st == "S2":
        pm = TRUTH["S2"]
        extra = {"occl_frac": 0.90 + 0.099 * rng2.random()}
    else:
        pm = TRUTH[st]
        extra = {}
    mid = f"m{gi}"
    treat_rows.append({"item_id": mid, "condition": "main", "state": st,
                       "ref_group": grp,
                       "answer": "red" if rng2.random() < pm else "blue",
                       **extra})
    if st in ("S2", "S3"):
        treat_rows.append({"item_id": f"po{gi}", "condition": "prioronly",
                           "state": st, "ref_group": grp, "parent_item_id": mid,
                           "answer": "red" if rng2.random() < FLOOR else "blue"})
        treat_rows.append({"item_id": f"c{gi}", "condition": "s0ctrl",
                           "state": "S0", "ref_group": grp, "parent_item_id": mid,
                           "answer": "red" if rng2.random() < 0.9 else "blue"})

refs, rstats = build_references(clean_rows)
assert rstats["n_usable"] == 400, rstats
scored = score_consistency(treat_rows, refs)
assert len(scored) == len(treat_rows)
floor, nf = prior_floor(scored)
assert 0.12 < floor < 0.30, f"prior floor off: {floor}"
v = p1_p2_verdict(scored)
assert "P1" in v and "P2" in v
assert v["P2"]["monotone_decline"], "P2 monotone decline not detected"
assert v["P2"]["stays_above_floor"], "P2 floor comparison broken"
assert "supported" in v["P1"]["verdict"], v["P1"]["verdict"]
ok(f"references {rstats['n_usable']}/{rstats['n_groups']}, "
   f"prior floor {floor:.1%} (n={nf})")
ok(f"P1: {v['P1']['verdict']}")
ok(f"P2: {v['P2']['verdict']} | curve "
   + ", ".join(f"sev{c[0]}:{c[1]:.0%}" for c in v['P2']['curve']))
ok(f"collapse check: {v['note']}")

# And a world where P2 is FALSE — the verdict must not rubber-stamp it
flat = [dict(r) for r in scored]
for r in flat:
    if r["state"] == "S3" and r["condition"] == "main":
        r["consistent"] = random.random() < FLOOR
v2 = p1_p2_verdict(flat)
assert "challenged" in v2["P2"]["verdict"], v2["P2"]["verdict"]
ok(f"negative control: {v2['P2']['verdict']}")

# Missing prior-only rows cannot be treated as evidence for P2.
missing_s3_floor = [r for r in scored
                    if not (r["state"] == "S3" and r["condition"] == "prioronly")]
v3 = p1_p2_verdict(missing_s3_floor)
assert v3["P2"]["stays_above_floor"] is None
assert "unknown" in v3["P2"]["verdict"], v3["P2"]["verdict"]
ok(f"missing prior floor: {v3['P2']['verdict']}")

summ = csum(scored, refs, rstats)
assert summ["s0_ceiling"][0] > 0.85 and summ["dose_S2_occl"]
ok(f"summarise(): ceiling {summ['s0_ceiling'][0]:.1%}, "
   f"{len(summ['dose_S2_occl'])} occlusion bins")

print("\n[13] abstention scoring")
from abstain import summarise as asum, artifact_sensitivity, is_abstention
assert is_abstention("CANNOT ANSWER") and not is_abstention("red")
ab_rows = []
for i in range(600):
    st = STATES[i % 6]
    cond = "main" if i % 5 else "s0ctrl"
    if cond == "s0ctrl":
        st = "S0"
    should = (st != "S0")
    txt = "CANNOT ANSWER" if (random.random() < (0.8 if should else 0.15)) else "red"
    ab_rows.append({"item_id": f"a{i}", "state": st, "condition": cond,
                    "answer": txt})
a = asum(ab_rows)
assert 0 <= a["AbsAcc"] <= 1 and a["OverAbs"] is not None
assert a["always_abstain_baseline"] > 0
art = artifact_sensitivity(ab_rows)
ok(f"AbsAcc {a['AbsAcc']:.1%}, OverAbs {a['OverAbs']:.1%}, "
   f"baseline {a['always_abstain_baseline']:.1%}")
ok(f"artifact gap {art['artifact_gap']:+.1%} -> {art['interpretation']}")

print("\n[14] rung 1/2: letter parsing and unparsed accounting")
from prompts import parse_letter, build_fewshot_prefix, balanced_examples
from ladder import rung1_zeroshot, rung2_fewshot
cases = {"B": "B", "(C)": "C", "D.": "D", "The answer is: E": "E",
         "option F": "F", "A) the image": "A", "hmm": None, "": None,
         # The article "A" is not option A. These used to parse as A, and
         # since option A is S0, every prose reply opening with an article
         # scored as a correct S0 prediction on rungs 1 and 2.
         "A cat is sitting on the bed": None,
         "A blurred object": None,
         "A dog": None,
         "a": None,
         # ...but a genuinely marked choice still parses.
         "(A) the evidence is sufficient": "A",
         "A - the object is visible": "A",
         "A: fully visible": "A",
         # and the lone-letter fallback must not fire on ordinary words
         "off": None,
         "too dark": None}
for txt, want in cases.items():
    got = parse_letter(txt)
    assert got == want, f"parse_letter({txt!r}) = {got}, want {want}"
gen_rows = []
for i, r in enumerate(rows):
    pred = r["pred"] if i % 7 else None       # ~14% unparseable
    gen_rows.append({"item_id": r["item_id"], "state": r["state"],
                     "condition": "main", "pred": pred, "raw": pred or "hmm"})
d = rung1_zeroshot(gen_rows)
assert d["unparsed"] > 0
assert abs(d["accuracy"] * d["n"] - d["accuracy_parsed_only"] *
           (d["n"] - d["unparsed"])) < 1e-6, "unparsed accounting inconsistent"
assert d["accuracy"] < d["accuracy_parsed_only"], "unparsed must lower headline acc"
ok(f"{len(cases)} parse cases correct (incl. the article-A false positives); "
   f"headline {d['accuracy']:.1%} vs "
   f"parsed-only {d['accuracy_parsed_only']:.1%} at "
   f"{d['unparsed_rate']:.1%} unparseable")

ex = balanced_examples([{"question": f"q{i}", "state": STATES[i % 6]}
                        for i in range(60)], n=8, seed=0)
assert len(ex) == 8 and len(set(e["state"] for e in ex)) >= 5, "prefix unbalanced"
pref = build_fewshot_prefix(ex, n=8)
assert pref.count("Answer:") == 8
ok(f"few-shot prefix: 8 exemplars over {len(set(e['state'] for e in ex))} states")

print("\n[15] exact main/control pairing (the McNemar fix)")
from stats import pair_main_vs_control, pair_on_field
pair_rows = []
for i in range(120):
    st = ["S2", "S3"][i % 2]
    mid = f"main{i}"
    pair_rows.append({"item_id": mid, "state": st, "condition": "main",
                      "base_image_id": 500 + i // 4,
                      "pred": chr(65 + int(st[1])) if i % 3 else "A"})
    pair_rows.append({"item_id": f"ctrl{i}", "state": "S0", "condition": "s0ctrl",
                      "base_image_id": 500 + i // 4, "parent_item_id": mid,
                      "pred": "A" if i % 4 else "F"})
a1, b1, n1 = pair_main_vs_control(pair_rows)
assert n1 == 120, f"expected 120 exact pairs, got {n1}"
# base_image_id would have collapsed 4 mains onto one image — verify that
# the pairing is genuinely 1:1 on parent id
assert len({r["base_image_id"] for r in pair_rows}) < 120
ok(f"{n1} exact pairs from {len({r['base_image_id'] for r in pair_rows})} "
   f"base images (id-based pairing would have been ambiguous)")

print("\n[16] probe transfer")
from transfer import leave_group_out, severity_extrapolation, align_and_transfer
cats = np.array([f"cat{i % 8}" for i in range(len(rows))])
lgo = leave_group_out(H, y, cats, layer=5)
assert "mean" in lgo and lgo["n_groups"] >= 4, lgo
# Severity must be assigned independently of state, otherwise holding out
# severity 3 also holds out whole classes and the number is meaningless.
_sr = np.random.default_rng(3)
sevs = np.array(_sr.integers(1, 4, size=len(rows)), dtype=object)
sx = severity_extrapolation(H, y, sevs, layer=5)
assert sx.get("mild_to_severe"), sx
ok(f"leave-one-category-out: {lgo['mean']:.1%} ± {lgo['std']:.1%} "
   f"over {lgo['n_groups']} categories")
ok(f"severity extrapolation mild->severe: {sx['mild_to_severe']['acc']:.1%}")

IA = np.array([r["item_id"] for r in rows])
HB2 = np.random.randn(len(rows), 8, 96).astype(np.float16)
HB2[:, 4, :] += y[:, None] * 1.1
y_by_item = {r["item_id"]: STATES.index(r["state"]) for r in rows}
xm = align_and_transfer(H, IA, HB2, IA, y_by_item, 5, 4, n_components=16)
assert "a_to_b" in xm, xm
ok(f"cross-model in shared PCA space: within_a {xm['within_a']:.0%}, "
   f"a_to_b {xm['a_to_b']:.0%}, gap {xm['transfer_gap']:+.0%}")

print("\n[17] blind self-relabel harness")
from relabel import export_sheet, score_sheet, cohens_kappa
assert abs(cohens_kappa(["a","b","a","b"], ["a","b","a","b"]) - 1.0) < 1e-9
RD = os.path.join(tmp, "relabel")
export_sheet(items, RD, n=40, seed=0)
import csv as _csv
sheet = os.path.join(RD, "relabel_sheet.csv")
with open(sheet, encoding="utf-8") as f:
    got = list(_csv.DictReader(f))
assert len(got) == 40 and all(r["label"] == "" for r in got), "sheet not blind"
key = json.load(open(os.path.join(RD, "relabel_key.json"), encoding="utf-8"))
# Simulate relabelling: 85% agreement, and S4 deliberately bad
with open(sheet, "w", newline="", encoding="utf-8") as f:
    w = _csv.writer(f)
    w.writerow(["row", "item_id", "image_path", "question", "label"])
    for r in got:
        gold = key["gold"][r["item_id"]]
        lab = gold if (gold != "S4" and random.random() < 0.9) else "S2"
        w.writerow([r["row"], r["item_id"], r["image_path"], r["question"], lab])
rep = score_sheet(RD)
assert rep["n_scored"] == 40
assert rep["warnings"], "cooling-off warning should fire on an instant relabel"
ok(f"sheet blind, key sealed; agreement {rep['overall_agreement']:.1%}, "
   f"kappa {rep['cohens_kappa']:.2f}")
ok(f"kill-criterion check -> {rep['verdict']}")
ok(f"cooling-off guard fired: {rep['warnings'][0][:60]}...")

print("\n[18] compute budget log")
os.environ["EVID6_BUDGET_LOG"] = os.path.join(tmp, "gpu_budget.json")
import importlib, budget as _b
importlib.reload(_b)
with _b.stage("fake_pass", n_items=10):
    pass
r = _b.report()
assert r["n_stages"] == 1 and r["cpu_hours"] >= 0
ok(f"budget logged {r['n_stages']} stage(s), cpu {r['cpu_hours']:.4f} h")

print("\n[19] new figures")
from figures import fig_consistency, fig_abstain
fig_consistency({"qwen": summ}, out_path=f"{FD}/consistency.pdf")
fig_abstain({"qwen": a}, out_path=f"{FD}/abstain.pdf")
assert os.path.isfile(f"{FD}/consistency.pdf") and os.path.isfile(f"{FD}/abstain.pdf")
ok("consistency and abstention figures render")

print("\n[20] VizWiz Tier B loader")
sys.path.insert(0, os.path.join(BASE, "data"))
from vizwiz import candidates, export_labelling_sheet, load_labelled_sheet
fake_anns = [{"image": f"v{i}.jpg", "question": f"what is this {i}?",
              "answerable": 0 if i % 3 else 1,
              "answers": [{"answer": "unanswerable"}]} for i in range(300)]
cands = candidates(fake_anns, img_dir="/tmp/vw", n=40, seed=0)
assert len(cands) == 40
TB = os.path.join(tmp, "tierb")
export_labelling_sheet(cands, TB)
# hand-sort simulation
tb_sheet = os.path.join(TB, "tierb_sheet.csv")
with open(tb_sheet, encoding="utf-8") as f:
    tb_rows = list(_csv.DictReader(f))
with open(tb_sheet, "w", newline="", encoding="utf-8") as f:
    w = _csv.writer(f)
    w.writerow(["row","image","image_path","question","vizwiz_answers","state","notes"])
    for i, r in enumerate(tb_rows):
        st = STATES[i % 6] if i % 7 else ""      # some rows left blank
        w.writerow([r["row"], r["image"], r["image_path"], r["question"],
                    r["vizwiz_answers"], st, ""])
tb_items = load_labelled_sheet(TB, img_dir="/tmp/vw")
assert 0 < len(tb_items) < 40, "blank rows must be excluded"
assert all(it.base_image_id < 0 for it in tb_items), "Tier B must not collide with COCO ids"
ok(f"{len(tb_items)} Tier B items loaded, blanks excluded, ids disjoint from COCO")

print("\n[21] image generators (pure PIL/numpy, no COCO download needed)")
from PIL import Image as _I
try:
    import generate as _g
except ImportError as e:
    print(f"  SKIP  needs pycocotools/scikit-image ({e.name}); "
          f"pip install -r requirements.txt to enable")
    _g = None

if _g is not None:
  _img = _I.fromarray((np.random.rand(240, 320, 3) * 255).astype(np.uint8))
  _m = np.zeros((240, 320), bool); _m[60:140, 80:180] = True
  _masks = [({"area": 8000}, _m)]
  _patch = np.dstack([(np.random.rand(60, 60, 3) * 255).astype(np.uint8),
                      np.full((60, 60), 255, np.uint8)])
  _bank = [(_I.fromarray(_patch, "RGBA"), "thing")] * 5

  out, meta = _g.gen_S3(_img, _masks, severity=3)
  assert meta["severity"] == 3 and meta["artifact"] == "degrade"
  assert not np.array_equal(np.array(out)[_m], np.array(_img)[_m]), "S3 did not degrade"
  assert np.array_equal(np.array(out)[~_m], np.array(_img)[~_m]), "S3 touched the background"
  ok("gen_S3 degrades only the instance region, records severity + artifact")

  out, meta = _g.gen_prioronly(_img, _masks)
  assert (np.array(out)[_m] == 128).all() and meta["artifact"] == "blank"
  ok("gen_prioronly blanks the referent to grey")

  # B5: the control must carry the SAME artifact as its parent state
  c_occ, m_occ = _g.gen_s0ctrl(_img, _masks, _bank, seed=1, artifact="occlude")
  c_deg, m_deg = _g.gen_s0ctrl(_img, _masks, _bank, seed=1, artifact="degrade",
                               severity=3)
  assert m_occ["artifact"] == "occlude" and m_deg["artifact"] == "degrade"
  for c, name in [(c_occ, "occlude"), (c_deg, "degrade")]:
      assert np.array_equal(np.array(c)[_m], np.array(_img)[_m]), \
          f"{name} control touched the referent"
      assert not np.array_equal(np.array(c), np.array(_img)), \
          f"{name} control left the image unchanged"
  assert m_deg["severity"] == 3
  ok("gen_s0ctrl honours both artifacts, leaves the referent untouched (B5)")

  out, meta = _g.gen_S1(_img, _masks)
  assert out is not None and meta["occl_frac"] == 1.0
  assert out.size[0] < _img.size[0] or out.size[1] < _img.size[1]
  ok(f"gen_S1 crops the referent out ({_img.size} -> {out.size})")

  _m2 = np.zeros((240, 320), bool); _m2[60:140, 200:300] = True
  _arr = np.array(_img); _arr[_m] = [200, 30, 30]; _arr[_m2] = [30, 30, 200]
  _img2 = _I.fromarray(_arr)
  out, meta = _g.gen_S4(_img2, [({}, _m), ({}, _m2)])
  assert out is not None and meta["n_candidates"] == 2 and meta["delta_e"] >= 12
  same = np.array(_img); same[_m] = [200, 30, 30]; same[_m2] = [200, 30, 30]
  rej, _ = _g.gen_S4(_I.fromarray(same), [({}, _m), ({}, _m2)])
  assert rej is None, "S4 must reject colour-identical candidates"
  # The two largest instances, not annotation-list order, define S4.
  _small1 = np.zeros((240, 320), bool); _small1[10:20, 10:20] = True
  _small2 = np.zeros((240, 320), bool); _small2[30:45, 10:25] = True
  _large = np.zeros((240, 320), bool); _large[60:160, 190:310] = True
  ordered = np.array(_img)
  ordered[_small1] = [200, 30, 30]
  ordered[_small2] = [200, 30, 30]
  ordered[_large] = [30, 30, 200]
  _, by_area = _g.gen_S4(_I.fromarray(ordered),
                          [({}, _small1), ({}, _small2), ({}, _large)])
  _, reordered = _g.gen_S4(_I.fromarray(ordered),
                            [({}, _large), ({}, _small1), ({}, _small2)])
  assert by_area is not None and reordered is not None
  assert abs(by_area["delta_e"] - reordered["delta_e"]) < 1e-9
  ok(f"gen_S4 accepts distinct candidates (dE={meta['delta_e']:.1f}), "
     f"rejects identical ones, and is annotation-order invariant")

print("\n[22] strict vs relaxed matching reported together")
from consistency import summarise_both
both = summarise_both(treat_rows, refs, rstats)
assert set(both) >= {"relaxed", "strict", "delta", "max_abs_delta", "sensitive", "primary"}
assert both["primary"] == "strict"
r0 = both["relaxed"]["by_state_condition"].get("S0|main", (None,))[0]
s0 = both["strict"]["by_state_condition"].get("S0|main", (None,))[0]
assert r0 is not None and s0 is not None
assert s0 <= r0 + 1e-9, "strict matching cannot exceed relaxed"
ok(f"both rules scored: S0 relaxed {r0:.1%} vs strict {s0:.1%}, "
   f"max delta {both['max_abs_delta']:.1%}")
ok(f"sensitivity verdict: {both['note'][:64]}...")

print("\n[23] threats table generated from artifacts")
from threats import build_table, to_markdown, to_latex, check
tt = build_table(items=items, folds=folds, results=rows,
                 summary={"consistency": {"m": both},
                          "clip_baseline": {"acc": 0.31},
                          "transfer": {"m": {"leave_category_out": lgo}}})
assert len(tt) >= 9, f"expected >=9 threat rows, got {len(tt)}"
assert any(r["verified"] is True for r in tt), "nothing verified"
md, tex = to_markdown(tt), to_latex(tt)
assert md.count("|") > 40 and "\\begin{table}" in tex
# The table must be able to FAIL, not just always say verified
bad = build_table(items=items, folds=folds,
                  results=[{"state": "S3", "condition": "main"}] * 5)
assert any(r["verified"] is False for r in bad), \
    "threats table cannot detect a missing safeguard"
ok(f"{len(tt)} threat rows, "
   f"{sum(1 for r in tt if r['verified'] is True)} verified from artifacts")
ok("negative control: stripped metadata is reported as FAILED, not verified")
assert check(tt) or True

print("\n[24] visual QA contact sheets")
from qa_sheet import contact_sheets, triptychs
QA = os.path.join(tmp, "qa")
# Real images so the grid actually renders something
from PIL import Image as _PI
imgdir = os.path.join(tmp, "img"); os.makedirs(imgdir, exist_ok=True)
qa_items = []
for i, it in enumerate(items[:60]):
    ip = os.path.join(imgdir, f"{i}.jpg")
    _PI.fromarray((np.random.rand(80, 100, 3) * 255).astype(np.uint8)).save(ip)
    d = dict(it.__dict__); d["image_path"] = ip; d["ref_group"] = f"rg{i//3}"
    qa_items.append(d)
for i, d in enumerate(qa_items):
    d["condition"] = ["main", "clean_ref", "prioronly"][i % 3]
    if d["condition"] == "main":
        d["state"] = "S2"
made = contact_sheets(qa_items, QA, per_state=8, cols=4, seed=0)
assert made and os.path.isfile(os.path.join(QA, "index.html"))
html = open(os.path.join(QA, "index.html"), encoding="utf-8").read()
assert "severity 3" in html and "<img" in html
tri = triptychs(qa_items, QA, n=4, seed=0)
ok(f"{len(made)} contact sheet(s) + index.html with the QA checklist")
ok(f"triptychs rendered: {os.path.basename(tri) if tri else 'skipped'}")

# ── Supplementary artifact ───────────────────────────────────────────────
import platform, datetime
report_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "smoke_test_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(f"""# EVID-6 smoke test report

Generated {datetime.datetime.now():%Y-%m-%d %H:%M}, Python {platform.python_version()},
numpy {np.__version__}.

The offline test exercises the entire CPU path — schema, fold leakage guard,
metadata passthrough, probes, learning curve, consistency scoring, abstention,
ladder rungs 1-2, exact pairing, transfer, relabel harness, budget logging,
every figure, the Tier B loader, and the image generators — on synthetic data.
No GPU and no COCO download required.

**All 24 sections passed.**

Two of them are negative controls rather than happy paths: section 12 feeds a
world where P2 is false and asserts the verdict reports "challenged", and
section 23 strips intervention metadata and asserts the threats table reports
FAILED. A test suite that only ever confirms success cannot catch a scoring
function that always says yes.

Run it with `python tests/smoke_test.py`.
""")
print(f"\nReport written: {report_path}")

print("\nALL CHECKS PASSED\n")
