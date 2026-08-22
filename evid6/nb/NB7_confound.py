# %% [markdown]
# # NB7: is the probe reading the evidence state, or the edit?
# **Accelerator: CPU only** - about 45 minutes. Consumes no GPU quota.
#
# ## Why
# A blind annotator reproduces the six construction labels at 56%, while the
# probe reads them at 73 to 79%. The probe beats human agreement with the
# ground truth, which invites one obvious explanation: three of the six states
# edit the image (S1 crops, S2 pastes an occluder, S3 blurs) and three do not
# (S0, S4, S5), so a probe could score well by detecting the *edit* rather than
# the evidence state.
#
# One state makes this concrete. S1 is the only state that changes image
# dimensions, and it is perfectly separable on that alone: median aspect ratio
# 0.814 against 1.33 to 1.50 elsewhere, fifteen distinct ratios across fifteen
# sampled items, and not one of them on a COCO-native shape.
#
# This notebook measures the size of that problem instead of caveating it. It
# adds no inference: every activation it needs was already saved by the cause
# passes, which covered all 1,838 items rather than the 900 main ones.
#
# ## Three analyses
# 1. **The artifact control that already exists.** `gen_s0ctrl` built, for every
#    S2 and S3 item, a twin carrying the *same artifact at the same area* on a
#    region touching none of the queried object, labelled S0. It is used in the
#    abstention and consistency analyses but has never been shown to the probe -
#    NB4 feeds the probe `condition == "main"` only. If the probe separates S2
#    from its own control, it is not merely detecting that an occluder is
#    present. If it cannot, that is a finding about rung 4.
# 2. **A low-level image baseline.** Geometry and simple statistics only, no
#    network at all. This is the artifact floor as a number, and it is the
#    baseline CLIP's 44.3% should be read against.
# 3. **The probe without S1.** How much of rung 4 survives when the one
#    geometry-confounded state is removed, plus per-state recall to show whether
#    S1 is sitting at ceiling.
#
# ## Prerequisites
# Attach NB1 (items + images), NB2 and NB3 (results + qwen/internvl acts), and
# `nb6 vlm` (smolvlm acts). Do **not** attach COCO - 164k files make the input
# search crawl.

# %% [markdown]
# ## Setup

# %%
import sys, os, json, glob
import numpy as np
from collections import Counter, defaultdict

SEARCH_ROOTS = ["/kaggle/input", "/kaggle/working"]


def find_dir(name, roots=None):
    """First directory called ``name`` under any input root, else None."""
    for root in (roots or SEARCH_ROOTS):
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, _files in os.walk(root):
            if name in dirnames:
                return os.path.join(dirpath, name)
    return None


def find_file(name, roots=None):
    for root in (roots or SEARCH_ROOTS):
        if not os.path.isdir(root):
            continue
        for dirpath, _dirnames, files in os.walk(root):
            if name in files:
                return os.path.join(dirpath, name)
    return None


# NB4 is a linear script: importing it would run the whole analysis. The two
# helpers above are copied deliberately rather than imported.
from schema import find_items, STATES
from splits import make_folds
from ladder import load_acts, layer_sweep, probe_layer, nested_probe

MODELS = {"qwen": "Qwen2.5-VL-3B", "internvl": "InternVL3-2B",
          "smolvlm": "SmolVLM2-2.2B"}

items_path = find_items()
items = [json.loads(l) for l in open(items_path, encoding="utf-8")]
print(f"manifest: {items_path}  ({len(items)} items)")
print("conditions:", dict(Counter(it["condition"] for it in items)))

by_id = {it["item_id"]: it for it in items}
main_items = [it for it in items if it["condition"] == "main"]
ctrl_items = [it for it in items if it["condition"] == "s0ctrl"]
print(f"main {len(main_items)}, s0ctrl {len(ctrl_items)}")


def acts_for(tag):
    """Locate and load one model's cached activations, or (None, None)."""
    cand = find_dir(tag)
    if not cand or not glob.glob(f"{cand}/h_*.npy"):
        print(f"  no activations found for {tag}")
        return None, None
    H, I = load_acts(tag, base=os.path.dirname(cand))
    return H, I


def align(H, I, keep_items):
    """Restrict activations to ``keep_items``, drop non-finite rows.

    Mirrors NB4's alignment exactly, including the NaN guard: a forward pass
    can emit NaN in fp16 and sklearn raises partway through the sweep rather
    than at load. Report the exclusion, never impute.
    """
    idx_of = {it["item_id"]: i for i, it in enumerate(keep_items)}
    mask = np.array([iid in idx_of for iid in I])
    H_a, I_a = H[mask], I[mask]
    order = np.array([idx_of[iid] for iid in I_a])
    finite = np.isfinite(H_a.astype(np.float32)).all(axis=(1, 2))
    if not finite.all():
        print(f"    dropped {int((~finite).sum())} non-finite rows: "
              f"{sorted(I_a[~finite].tolist())}")
    return H_a[finite], I_a[finite], order[finite]


# %% [markdown]
# ## 1. The artifact control: S2 vs its own S0-ctrl twin
#
# Same occluder, same size, same location. The only difference is whether it
# covers the queried object. A probe that separates these is reading more than
# "an artifact is present"; one that cannot is reading the artifact.
#
# Chance here is the majority class, not 1/6 - these are binary problems with
# unequal n, so the majority rate is printed alongside every accuracy.

# %%
def artifact_control(parent_state):
    """Build the (main vs s0ctrl) contrast for one parent state."""
    parents = [it for it in main_items if it["state"] == parent_state]
    parent_ids = {it["item_id"] for it in parents}
    ctrls = [it for it in ctrl_items
             if it.get("parent_item_id") in parent_ids]
    pool = parents + ctrls
    y = np.array([0] * len(parents) + [1] * len(ctrls))
    return pool, y


control_results = {}
for parent_state in ["S2", "S3"]:
    pool, y_pool = artifact_control(parent_state)
    if len(set(y_pool)) < 2 or len(pool) < 50:
        print(f"\n{parent_state}: only {len(pool)} items with both classes - skipped")
        continue

    folds_pool = make_folds(pool, n_splits=5, seed=0)
    majority = max(np.bincount(y_pool)) / len(y_pool)
    print(f"\n=== {parent_state} (n={int((y_pool==0).sum())}) vs its S0-ctrl "
          f"twin (n={int((y_pool==1).sum())}) ===")
    print(f"    majority-class baseline: {majority:.1%}")

    for tag_prefix in MODELS:
        H, I = acts_for(f"{tag_prefix}_cause")
        if H is None:
            continue
        H_a, _I_a, order = align(H, I, pool)
        if len(H_a) < 50:
            print(f"  {tag_prefix}: only {len(H_a)} aligned rows - skipped")
            continue
        res = nested_probe(H_a, y_pool[order], folds_pool[order])
        control_results[(parent_state, tag_prefix)] = {
            "accuracy": res["accuracy"], "std": res["std"],
            "majority": float(majority), "n": int(len(H_a)),
            "above_majority": float(res["accuracy"] - majority),
        }
        print(f"  {tag_prefix:9} {res['accuracy']:.1%} +/-{res['std']:.1%}"
              f"   ({res['accuracy'] - majority:+.1%} over majority)")

# %% [markdown]
# ## 2. The low-level image baseline
#
# No network. Image geometry plus three cheap statistics that track exactly the
# edits the generator makes: a Laplacian-variance blur measure for S3, mean and
# standard deviation of luminance for the pasted occluders of S2. If this scores
# well, the six-way task is partly solvable without looking at content, and that
# is the number CLIP's 44.3% has to be read against.

# %%
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

img_root = find_dir("images")
print(f"images: {img_root}")


def low_level_features(path):
    """Geometry and three cheap statistics. Never opens the label."""
    im = Image.open(path).convert("L")
    w, h = im.size
    a = np.asarray(im, dtype=np.float32) / 255.0
    # Laplacian variance: the standard cheap focus measure, low when blurred.
    lap = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
           - 4 * a[1:-1, 1:-1])
    return [w, h, w / h, w * h, float(lap.var()), float(a.mean()), float(a.std())]


X_ll, y_ll, kept = [], [], []
for it in main_items:
    p = it["image_path"]
    if not os.path.isfile(p) and img_root:
        p = os.path.join(img_root, os.path.basename(it["image_path"]))
    if not os.path.isfile(p):
        continue
    try:
        X_ll.append(low_level_features(p))
    except Exception:
        continue
    y_ll.append(STATES.index(it["state"]))
    kept.append(it)

X_ll = np.array(X_ll, dtype=np.float32)
y_ll = np.array(y_ll)
print(f"featurised {len(X_ll)} of {len(main_items)} main items")

if len(X_ll) > 100:
    folds_ll = make_folds(kept, n_splits=5, seed=0)
    accs, per_state = [], defaultdict(lambda: [0, 0])
    for k in np.unique(folds_ll):
        tr, te = folds_ll != k, folds_ll == k
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000))
        clf.fit(X_ll[tr], y_ll[tr])
        pred = clf.predict(X_ll[te])
        accs.append((pred == y_ll[te]).mean())
        for p_, t_ in zip(pred, y_ll[te]):
            per_state[STATES[t_]][1] += 1
            per_state[STATES[t_]][0] += int(p_ == t_)

    ll_acc, ll_std = float(np.mean(accs)), float(np.std(accs))
    print(f"\nlow-level 6-way: {ll_acc:.1%} +/-{ll_std:.1%}   (chance 16.7%)")
    print("per state:")
    for s in STATES:
        hit, n = per_state[s]
        if n:
            print(f"  {s}  {hit:>3}/{n:<4} {hit/n:>6.1%}")
    # Geometry alone, to isolate the S1 shortcut from the blur/paste signatures.
    accs_geo = []
    for k in np.unique(folds_ll):
        tr, te = folds_ll != k, folds_ll == k
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X_ll[tr][:, :4], y_ll[tr])
        accs_geo.append((clf.predict(X_ll[te][:, :4]) == y_ll[te]).mean())
    print(f"\ngeometry only (w, h, aspect, area): {np.mean(accs_geo):.1%}")
else:
    ll_acc = ll_std = None
    print("not enough images found - low-level baseline skipped")

# %% [markdown]
# ## 3. Rung 4 without S1, and per-state recall
#
# If the probe is exploiting the crop, dropping S1 should cost more than the
# one-sixth a uniformly-difficult state would, and S1's recall in the six-way
# probe should sit near ceiling.

# %%
five_way, per_state_r4 = {}, {}
y_main_all = np.array([STATES.index(it["state"]) for it in main_items])
folds_main = make_folds(main_items, n_splits=5, seed=0)

for tag_prefix in MODELS:
    H, I = acts_for(f"{tag_prefix}_cause")
    if H is None:
        continue
    H_a, _I_a, order = align(H, I, main_items)
    y_a, f_a = y_main_all[order], folds_main[order]
    print(f"\n{tag_prefix}: {len(H_a)} aligned rows")

    # Per-state recall at the single best layer (biased upward by selection;
    # used only to compare states against each other, never quoted as R4).
    sweep = layer_sweep(H_a, y_a, f_a)
    layer = int(max(sweep, key=lambda x: x[1])[0])
    ps = defaultdict(lambda: [0, 0])
    X = H_a[:, layer, :].astype(np.float32)
    for k in np.unique(f_a):
        tr, te = f_a != k, f_a == k
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], y_a[tr])
        for p_, t_ in zip(clf.predict(X[te]), y_a[te]):
            ps[STATES[t_]][1] += 1
            ps[STATES[t_]][0] += int(p_ == t_)
    per_state_r4[tag_prefix] = {s: (ps[s][0], ps[s][1]) for s in STATES if ps[s][1]}
    print(f"  per-state recall at layer {layer} (selection-biased, comparative only):")
    for s in STATES:
        if ps[s][1]:
            print(f"    {s}  {ps[s][0]:>3}/{ps[s][1]:<4} {ps[s][0]/ps[s][1]:>6.1%}")

    # Five-way: drop S1 entirely and refit, nested as in the headline number.
    keep = y_a != STATES.index("S1")
    y5 = y_a[keep]
    remap = {v: i for i, v in enumerate(sorted(set(y5.tolist())))}
    res5 = nested_probe(H_a[keep], np.array([remap[v] for v in y5.tolist()]),
                        f_a[keep])
    five_way[tag_prefix] = {"accuracy": res5["accuracy"], "std": res5["std"],
                            "chance": 1.0 / len(remap), "n": int(keep.sum())}
    print(f"  5-way without S1: {res5['accuracy']:.1%} +/-{res5['std']:.1%} "
          f"(chance {1.0/len(remap):.1%})")

# %% [markdown]
# ## Summary

# %%
out = {
    "artifact_control": {f"{s}_{m}": v for (s, m), v in control_results.items()},
    "low_level_baseline": ({"acc": ll_acc, "std": ll_std} if ll_acc else None),
    "five_way_no_S1": five_way,
    "per_state_r4_biased": per_state_r4,
}
os.makedirs("/kaggle/working/figures", exist_ok=True)
with open("/kaggle/working/figures/confound.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
print("\nwrote /kaggle/working/figures/confound.json")

print("""
HOW TO READ THIS

Analysis 1 is the one that matters. If the probe separates S2 from its own
S0-ctrl twin well above the majority baseline, then it is distinguishing "the
occluder covers the thing you asked about" from "an occluder is present", which
is the evidence state and not the artifact. If it sits near majority, rung 4 is
substantially artifact detection and the paper's framing has to change.

Analysis 2 sets the floor. Quote the probe against this and against CLIP, not
against 16.7% chance.

Analysis 3 bounds the S1 problem. A five-way number close to the six-way one
means the crop shortcut was not carrying the result.
""")
