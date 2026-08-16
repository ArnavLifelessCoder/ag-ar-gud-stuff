# %% [markdown]
# # NB4: Analysis — Probes, Ladder, CLIP Baseline, Stats, Figures
# **Accelerator: CPU only** — does not consume GPU quota.
#
# ## What this notebook does
# 1. Loads all results from NB1, NB2, NB3
# 2. Runs the 4-rung evaluation ladder
# 3. Layer sweep + best-layer probe
# 4. Learning curve analysis
# 5. CLIP baseline comparison
# 6. Dose-response analysis (P1/P2 verdict)
# 7. Statistical tests
# 8. Generates all paper figures
#
# ## Prerequisites
# Attach NB1, NB2, and NB3 outputs as dataset inputs.

# %% [markdown]
# ## Setup

# %%
import sys, os, json, glob
import numpy as np
from collections import Counter

# Attached notebook outputs mount at a path derived from each notebook's TITLE,
# so hard-coding them breaks the moment a notebook is named anything else --
# which is exactly what happened ("vlm neurips nb1" -> /kaggle/input/
# vlm-neurips-nb1, not evid6-nb1-output). Everything below is discovered by
# searching the input roots instead, the same way NB2/NB3 locate the manifest.
SEARCH_ROOTS = ["/kaggle/input", "/kaggle/working"]

# Copy source if needed
import shutil
EVID6_SRC = "/kaggle/input/evid6-code/evid6"
EVID6_DST = "/kaggle/working/evid6"
if os.path.isdir(EVID6_SRC) and not os.path.isdir(EVID6_DST):
    shutil.copytree(EVID6_SRC, EVID6_DST)

sys.path.insert(0, "/kaggle/working/evid6/data")
sys.path.insert(0, "/kaggle/working/evid6/eval")
sys.path.insert(0, "/kaggle/working/evid6/probe")
sys.path.insert(0, "/kaggle/working/evid6/analysis")

os.makedirs("/kaggle/working/figures", exist_ok=True)

# %%
# Locate data files by searching, not by guessing a mount path.
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
    """First file called ``name`` under any input root, else None."""
    for root in (roots or SEARCH_ROOTS):
        if not os.path.isdir(root):
            continue
        direct = os.path.join(root, name)
        if os.path.isfile(direct):
            return direct
        for dirpath, _dirnames, files in os.walk(root):
            if name in files:
                return os.path.join(dirpath, name)
    return None


from schema import find_items

ITEMS_PATH = find_items(SEARCH_ROOTS)     # raises listing what IS attached
NB1_PATH = os.path.dirname(ITEMS_PATH)
print(f"Items: {ITEMS_PATH}")
print(f"NB1 root: {NB1_PATH}")

# Load items
from schema import load_items, STATES
items = load_items(ITEMS_PATH)
main_items = [it for it in items if it.condition == "main"]
print(f"Total items: {len(items)}, main: {len(main_items)}")

# %% [markdown]
# ## Load all inference results

# %%
def load_results(tag, search_paths):
    for p in search_paths:
        for root, dirs, files in os.walk(p):
            if f"{tag}.jsonl" in files:
                path = os.path.join(root, f"{tag}.jsonl")
                with open(path, encoding="utf-8") as f:
                    return [json.loads(l) for l in f if l.strip()]
    return []

search = SEARCH_ROOTS

MODELS = {
    "qwen": "Qwen2.5-VL-3B",
    "internvl": "InternVL3-2B",
    "smolvlm": "SmolVLM2-2.2B",
}

results = {}
for tag_prefix in MODELS:
    for suffix in ["cause", "abstain", "repair", "clean", "treat",
                   "rung1", "rung2"]:
        tag = f"{tag_prefix}_{suffix}"
        r = load_results(tag, search)
        if r:
            results[tag] = r
            print(f"  {tag}: {len(r)} items")
        else:
            print(f"  {tag}: NOT FOUND")

# %% [markdown]
# ## 1. Evaluation Ladder — Rung 3 (logit argmax)

# %%
from ladder import rung3_from_logits, per_state_accuracy, rung1_from_text

print("=" * 60)
print("Rung 3: Option-token argmax accuracy")
print("=" * 60)

rung3_scores = {}
for tag_prefix, model_name in MODELS.items():
    tag = f"{tag_prefix}_cause"
    if tag not in results:
        continue
    r = [r for r in results[tag] if r["condition"] == "main"]
    acc = rung3_from_logits(r)
    rung3_scores[tag_prefix] = acc
    print(f"\n{model_name}: {acc:.1%}")
    psa = per_state_accuracy(r)
    for s, (a, n) in psa.items():
        print(f"  {s}: {a:.1%} ({n} items)")

# %% [markdown]
# ## 2. Cross-validated splits

# %%
from splits import make_folds

folds = make_folds(main_items, n_splits=5, seed=0)
print(f"Fold distribution: {dict(Counter(folds))}")

# Build aligned arrays for probing
item_id_to_idx = {it.item_id: i for i, it in enumerate(main_items)}
y_main = np.array([STATES.index(it.state) for it in main_items])

# %% [markdown]
# ## 3. Rung 4: Linear probe (layer sweep)

# %%
from ladder import (load_acts, layer_sweep, probe_layer, best_layer,
                    nested_probe)

sweeps = {}
best_layers = {}     # for the sweep figure and the learning curve only
probe_r4 = {}        # the honest rung-4 number: layer chosen inside training folds

for tag_prefix in MODELS:
    tag = f"{tag_prefix}_cause"
    # Try to find activation files
    acts_base = None
    _cand = find_dir(tag)          # searches for a directory named <tag>
    if _cand and glob.glob(f"{_cand}/h_*.npy"):
        acts_base = os.path.dirname(_cand)

    if acts_base is None:
        print(f"No activations found for {tag}")
        continue

    print(f"\nLoading activations for {tag}...")
    H, I = load_acts(tag, base=acts_base)
    print(f"  Shape: {H.shape}")

    # Align with main items
    mask = np.array([iid in item_id_to_idx for iid in I])
    H_aligned = H[mask]
    I_aligned = I[mask]
    idx = np.array([item_id_to_idx[iid] for iid in I_aligned])
    y_aligned = y_main[idx]
    folds_aligned = folds[idx]

    print(f"  Aligned: {len(I_aligned)} items")

    # Drop rows whose activations are not finite. A forward pass can emit NaN
    # in fp16 (5 of Qwen's 900 aligned rows did), and sklearn raises
    # "Input X contains NaN" partway through the sweep rather than at load.
    # Those rows also carry NaN option probabilities, so their R3 label is an
    # argmax-over-NaN artifact, not a prediction -- they are unusable, not
    # merely inconvenient. Report the count; never impute.
    finite = np.isfinite(H_aligned.astype(np.float32)).all(axis=(1, 2))
    if not finite.all():
        print(f"  DROPPED {int((~finite).sum())} of {len(finite)} rows with "
              f"non-finite activations: {sorted(I_aligned[~finite].tolist())}")
        print("   -> report this exclusion; rerun those items to recover them")
        H_aligned = H_aligned[finite]
        I_aligned = I_aligned[finite]
        y_aligned = y_aligned[finite]
        folds_aligned = folds_aligned[finite]

    # Layer sweep
    print(f"  Running layer sweep ({H.shape[1]} layers)...")
    sweep = layer_sweep(H_aligned, y_aligned, folds_aligned)
    sweeps[tag_prefix] = sweep

    bl = best_layer(sweep)
    best_layers[tag_prefix] = bl
    print(f"  Best layer: {bl[0]} (acc={bl[1]:.1%} ± {bl[2]:.1%})")

    # The line above picks the layer by max over the sweep, on the same folds
    # it scores on. That is worth +2-3 points on pure noise, and the headline
    # claim is the R4 - R1 gap, so R4 gets chosen nested instead: each outer
    # fold picks its layer using only the training folds.
    print("  Nested layer selection (this is the reported R4)...")
    nb = nested_probe(H_aligned, y_aligned, folds_aligned)
    probe_r4[tag_prefix] = nb
    print(f"  R4 = {nb['accuracy']:.1%} ± {nb['std']:.1%}  "
          f"layers chosen per fold: {nb['layers_chosen']}")
    print(f"  (max-over-layers would have read {nb['flat_max_over_layers']:.1%}"
          f" — {nb['selection_bias']:+.1%} of selection bias, not reported)")
    if len(set(nb["layers_chosen"])) > 1:
        print("  NOTE: folds disagree on the best layer. Say so in the paper "
              "rather than quoting a single layer index.")

# %% [markdown]
# ## 4. Learning curves

# %%
from learning_curve import learning_curve

curves = {}
for tag_prefix in MODELS:
    tag = f"{tag_prefix}_cause"
    if tag_prefix not in best_layers:
        continue

    # Reload activations
    acts_base = None
    _cand = find_dir(tag)          # searches for a directory named <tag>
    if _cand and glob.glob(f"{_cand}/h_*.npy"):
        acts_base = os.path.dirname(_cand)
    if acts_base is None:
        continue

    H, I = load_acts(tag, base=acts_base)
    mask = np.array([iid in item_id_to_idx for iid in I])
    H_a, I_a = H[mask], I[mask]
    idx = np.array([item_id_to_idx[iid] for iid in I_a])
    y_a, f_a = y_main[idx], folds[idx]

    # Same non-finite filter as the sweep, so the curve is fitted on exactly
    # the rows the probe was.
    _fin = np.isfinite(H_a.astype(np.float32)).all(axis=(1, 2))
    if not _fin.all():
        H_a, I_a, y_a, f_a = H_a[_fin], I_a[_fin], y_a[_fin], f_a[_fin]

    bl = best_layers[tag_prefix][0]
    print(f"\nLearning curve for {tag_prefix} (layer {bl})...")
    curve = learning_curve(H_a, y_a, f_a, layer=bl)
    curves[tag_prefix] = curve
    for n, mean, std in curve:
        print(f"  n={n:4d}: {mean:.1%} ± {std:.1%}")

# %% [markdown]
# ## 5. CLIP baseline

# %%
from clip_baseline import clip_probe

# Get image paths for main items
clip_paths = [it.image_path for it in main_items]

# Fix paths if they reference NB1 output location
fixed_paths = []
for p in clip_paths:
    if os.path.isfile(p):
        fixed_paths.append(p)
    else:
        # Try NB1 dataset path
        basename = os.path.basename(p)
        for search_dir in [NB1_PATH]:
            candidate = os.path.join(search_dir, "evid6", "images", basename)
            if os.path.isfile(candidate):
                fixed_paths.append(candidate)
                break
        else:
            fixed_paths.append(p)  # will fail downstream

clip_acc, clip_std = None, None
missing = [p for p in fixed_paths if not os.path.isfile(p)]
if missing:
    print(f"WARNING: {len(missing)} image paths unresolved, e.g. {missing[0]}")

print("Running CLIP baseline probe...")
clip_acc, clip_std = clip_probe(fixed_paths, y_main, folds)
print(f"CLIP baseline: {clip_acc:.1%} ± {clip_std:.1%}")

# Kill criterion: if CLIP >= the VLM probe, drop absolute numbers from abstract.
# Compare against the nested R4, not the max-over-layers number — comparing a
# selection-inflated probe against an honest CLIP baseline would let the kill
# criterion pass on bias alone.
for tag_prefix, nb in probe_r4.items():
    acc = nb["accuracy"]
    if clip_acc is not None and clip_acc >= acc:
        print(f"⚠ CLIP >= {tag_prefix} probe! Rebuild paper on rung 1 vs rung 4 gap.")

# %% [markdown]
# ## 6. Build the full ladder
# Four rungs on identical items. R1 and R2 are GENERATED answers (what the
# model does); R3 is the option-token argmax from the same forward pass that
# produced the activations; R4 is the probe on those activations.
# The R1-to-R4 gap is the headline.

# %%
from ladder import rung1_zeroshot, rung2_fewshot

rung_data, rung_detail = {}, {}
for tag_prefix, model_name in MODELS.items():
    tag = f"{tag_prefix}_cause"
    if tag not in results:
        continue

    r_cause = [r for r in results[tag] if r["condition"] == "main"]
    entry = {"R3_logits": rung3_from_logits(r_cause)}
    detail = {}

    r1_rows = [r for r in results.get(f"{tag_prefix}_rung1", [])
               if r.get("condition") == "main"]
    if r1_rows:
        d = rung1_zeroshot(r1_rows)
        entry["R1_zeroshot"] = d["accuracy"]
        detail["R1"] = d
    else:
        print(f"  {tag_prefix}: no rung-1 generations found — R1 omitted "
              f"(do NOT substitute R3 for it)")

    r2_rows = results.get(f"{tag_prefix}_rung2", [])
    if r2_rows:
        d = rung2_fewshot(r2_rows)
        entry["R2_fewshot"] = d["accuracy"]
        detail["R2"] = d
        # Rung 2 runs on the held-out fold only, so compare it against rung 1
        # restricted to the same items rather than against the full R1.
        r2_ids = {r["item_id"] for r in r2_rows}
        r1_same = [r for r in r1_rows if r["item_id"] in r2_ids]
        if r1_same:
            detail["R1_on_R2_items"] = rung1_zeroshot(r1_same)

    if tag_prefix in probe_r4:
        # Nested selection, not max-over-layers — see section 3.
        entry["R4_probe"] = probe_r4[tag_prefix]["accuracy"]
        detail["R4"] = probe_r4[tag_prefix]
    elif tag_prefix in best_layers:
        print(f"  {tag_prefix}: nested probe missing — R4 omitted rather than "
              f"falling back to the biased max-over-layers number")

    rung_data[tag_prefix] = entry
    rung_detail[tag_prefix] = detail

    print(f"\n{model_name}:")
    for k in ["R1_zeroshot", "R2_fewshot", "R3_logits", "R4_probe"]:
        if k in entry:
            print(f"  {k}: {entry[k]:.1%}")
    if "R1" in detail:
        print(f"  (R1 unparseable replies: {detail['R1']['unparsed_rate']:.1%}"
              f" — reported alongside the accuracy, not hidden)")
    if "R4" in detail:
        print(f"  (R4 layer chosen inside training folds; max-over-layers "
              f"would read {detail['R4']['flat_max_over_layers']:.1%}, "
              f"{detail['R4']['selection_bias']:+.1%})")
    if "R1_zeroshot" in entry and "R4_probe" in entry:
        print(f"  ** R1 -> R4 gap: "
              f"{entry['R4_probe'] - entry['R1_zeroshot']:+.1%} **")

# %% [markdown]
# ## 7. Self-consistency and the P1/P2 verdict
# This is the paper's core measurement. There is no ground truth: the
# reference is the model's own answer on the untouched image, and every
# degraded condition is scored by whether that answer survives.

# %%
from consistency import (build_references, score_consistency, summarise,
                         summarise_both, by_state_condition, prior_floor,
                         p1_p2_verdict)

consistency_summary = {}
scored_by_model = {}

for tag_prefix, model_name in MODELS.items():
    clean = results.get(f"{tag_prefix}_clean", [])
    treat = results.get(f"{tag_prefix}_treat", [])
    if not clean or not treat:
        print(f"{model_name}: missing clean/treat generations, skipping")
        continue

    refs, ref_stats = build_references(clean, require_stable=True)
    scored = score_consistency(treat, refs)              # strict (headline)
    scored_by_model[tag_prefix] = scored

    # Score under BOTH matching rules. The choice moves the headline numbers,
    # so the paper reports both whenever they differ materially.
    both = summarise_both(treat, refs, ref_stats)
    # Strict matching is the headline. Relaxed matching is reported only as a
    # sensitivity analysis because degraded images can induce shorter answers.
    summary = both["strict"]
    summary["max_abs_delta"] = both["max_abs_delta"]
    summary["relaxed"] = both["relaxed"]
    summary["matching_note"] = both["note"]
    consistency_summary[tag_prefix] = summary

    print("\n" + "=" * 60)
    print(f"{model_name}")
    print("=" * 60)
    print(f"  references: {ref_stats['n_usable']}/{ref_stats['n_groups']} usable "
          f"(drop rate {ref_stats['drop_rate']:.1%})")
    if ref_stats["drop_rate"] > 0.35:
        print("  KILL CRITERION (13 Aug): reference too noisy, switch to "
              "closed-set colour questions and report it")

    ceil = summary["s0_ceiling"]
    floor = summary["prior_floor_pooled"]
    if ceil[0] is not None:
        print(f"  S0 ceiling (measurement noise floor): {ceil[0]:.1%} (n={ceil[1]})")
    if floor[0] is not None:
        print(f"  prior-only floor (answering from priors): {floor[0]:.1%} (n={floor[1]})")

    print("\n  consistency by state and condition:")
    for (st, cond), (rate, n) in by_state_condition(scored).items():
        if rate is not None:
            print(f"    {st:<3} {cond:<10} {rate:6.1%}  (n={n})")

    print(f"\n  matching rule: {both['note']}")
    if both["sensitive"]:
        print("    relaxed-matching sensitivity numbers:")
        for (st, cond), (rate, n) in \
                by_state_condition(score_consistency(treat, refs,
                                                     relaxed=True)).items():
            if rate is not None and cond == "main":
                print(f"      {st:<3} {rate:6.1%}  (n={n})")

    v = summary["verdict"]
    print("\n  P1/P2:")
    if "P1" in v:
        print(f"    P1 (occlusion destroys signal): {v['P1']['verdict']}")
        if v["P1"]["gap"] is not None:
            print(f"       S2 main {v['P1']['s2_main']:.1%} vs floor "
                  f"{v['P1']['s2_prior_floor']:.1%}  (gap {v['P1']['gap']:+.1%})")
    if "P2" in v:
        print(f"    P2 (degradation attenuates):     {v['P2']['verdict']}")
        for sev, rate, n in v["P2"]["curve"]:
            print(f"       severity {sev}: {rate:.1%} (n={n})")
    if "note" in v:
        print(f"    -> {v['note']}")

# %% [markdown]
# ## 7b. Abstention: AbsAcc vs OverAbs
# A model that abstains on everything scores well on naive accuracy and is
# useless, so these two numbers only mean anything together.

# %%
from abstain import summarise as abstain_summarise, artifact_sensitivity

abstain_summary = {}
for tag_prefix, model_name in MODELS.items():
    rows = results.get(f"{tag_prefix}_abstain", [])
    if not rows:
        continue
    a = abstain_summarise(rows)
    art = artifact_sensitivity(rows)
    a.pop("scored", None)
    a["artifact_sensitivity"] = art
    abstain_summary[tag_prefix] = a

    print(f"\n{model_name}:")
    print(f"  AbsAcc  {a['AbsAcc']:.1%}   (always-abstain baseline "
          f"{a['always_abstain_baseline']:.1%})")
    if a["OverAbs"] is not None:
        print(f"  OverAbs {a['OverAbs']:.1%}  (abstains on answerable items)")
    if a["UnderAbs"] is not None:
        print(f"  UnderAbs {a['UnderAbs']:.1%} (answers unanswerable items)")
    if art["artifact_gap"] is not None:
        print(f"  artifact gap {art['artifact_gap']:+.1%} — {art['interpretation']}")

# %% [markdown]
# ## 8. Statistical tests
# Pairing is on ``parent_item_id``, not ``base_image_id``: one COCO image can
# yield several main items, so matching on the image makes the pair arbitrary.

# %%
print("\n" + "=" * 60)
print("Paired tests (McNemar)")
print("=" * 60)

from stats import boot_ci, paired_test, pair_main_vs_control, pair_on_field

stat_results = {}
for tag_prefix, model_name in MODELS.items():
    tag = f"{tag_prefix}_cause"
    if tag not in results:
        continue
    entry = {}

    # (a) cause-prompt: does the artifact alone move the prediction?
    a, b, n = pair_main_vs_control(results[tag])
    if n >= 10:
        pv = paired_test(a, b)
        entry["main_vs_ctrl_cause"] = {
            "main_acc": float(a.mean()), "ctrl_acc": float(b.mean()),
            "n_pairs": n, "p": pv,
        }
        print(f"\n{model_name}: main vs s0-ctrl (cause prompt, {n} exact pairs)")
        print(f"  main {a.mean():.1%} | ctrl {b.mean():.1%} | "
              f"McNemar p={pv:.4f} {'*' if pv < 0.05 else 'ns'}")
    else:
        print(f"\n{model_name}: only {n} exact main/ctrl pairs, skipping")

    # (b) consistency: main vs its own prior-only floor, paired by parent
    scored = scored_by_model.get(tag_prefix)
    if scored:
        main_rows = [r for r in scored if r["condition"] == "main"]
        po_rows = [dict(r, item_id=r["parent_item_id"]) for r in scored
                   if r["condition"] == "prioronly" and r.get("parent_item_id")]
        a2, b2, n2 = pair_on_field(main_rows, po_rows, field="consistent")
        if n2 >= 10:
            pv2 = paired_test(a2, b2)
            entry["main_vs_prioronly_consistency"] = {
                "main": float(a2.mean()), "prior_only": float(b2.mean()),
                "n_pairs": n2, "p": pv2,
            }
            m, lo, hi = boot_ci(a2.astype(float))
            print(f"  consistency: main {m:.1%} [{lo:.1%},{hi:.1%}] vs "
                  f"prior-only {b2.mean():.1%} | p={pv2:.4f} "
                  f"({n2} pairs)")
            if pv2 >= 0.05:
                print("    -> main consistency is NOT distinguishable from the "
                      "prior floor: the answer was never using the evidence")

    stat_results[tag_prefix] = entry

# %% [markdown]
# ## 8b. Probe generalization
# Does the probe survive unseen object categories and unseen severities? A
# probe that only works within-category has learned the objects, not the
# evidence state.

# %%
from transfer import leave_group_out, severity_extrapolation, align_and_transfer

transfer_results = {}
cat_by_item = {it.item_id: it.category for it in main_items}
sev_by_item = {it.item_id: it.severity for it in main_items}

acts_cache = {}
for tag_prefix in MODELS:
    tag = f"{tag_prefix}_cause"
    acts_base = None
    cand = find_dir(tag)
    if cand and glob.glob(f"{cand}/h_*.npy"):
        acts_base = os.path.dirname(cand)
    if acts_base is None or tag_prefix not in best_layers:
        continue

    H, I = load_acts(tag, base=acts_base)
    mask = np.array([iid in item_id_to_idx for iid in I])
    H_a, I_a = H[mask], I[mask]
    idx = np.array([item_id_to_idx[iid] for iid in I_a])
    y_a = y_main[idx]

    # Non-finite rows would poison leave-category-out and the shared PCA space.
    _fin = np.isfinite(H_a.astype(np.float32)).all(axis=(1, 2))
    if not _fin.all():
        H_a, I_a, y_a = H_a[_fin], I_a[_fin], y_a[_fin]

    acts_cache[tag_prefix] = (H_a, I_a, y_a)

    bl = best_layers[tag_prefix][0]
    cats = np.array([cat_by_item.get(i, "?") for i in I_a])
    sevs = np.array([sev_by_item.get(i) for i in I_a], dtype=object)

    lgo = leave_group_out(H_a, y_a, cats, layer=bl)
    sx = severity_extrapolation(H_a, y_a, sevs, layer=bl)
    transfer_results[tag_prefix] = {"leave_category_out": lgo,
                                    "severity_extrapolation": sx}

    print(f"\n{MODELS[tag_prefix]} (layer {bl}):")
    if "error" not in lgo:
        print(f"  leave-one-category-out: {lgo['mean']:.1%} ± {lgo['std']:.1%} "
              f"over {lgo['n_groups']} categories "
              f"(worst: {lgo['worst']} at {lgo['worst_acc']:.1%})")
        within = best_layers[tag_prefix][1]
        print(f"    vs within-distribution {within:.1%} "
              f"-> drop {within - lgo['mean']:+.1%}")
    for k, v in (sx or {}).items():
        if isinstance(v, dict):
            print(f"  {k}: {v['acc']:.1%} (train {v['n_train']}, test {v['n_test']})")

# %%
# Cross-model transfer, if two models cached activations
if len(acts_cache) >= 2:
    keys = list(acts_cache)[:2]
    (HA, IA, _), (HB, IB, _) = acts_cache[keys[0]], acts_cache[keys[1]]
    y_by_item = {it.item_id: STATES.index(it.state) for it in main_items}
    xm = align_and_transfer(HA, IA, HB, IB, y_by_item,
                            best_layers[keys[0]][0], best_layers[keys[1]][0])
    transfer_results["cross_model"] = {"pair": keys, **xm}
    print(f"\nCross-model ({keys[0]} <-> {keys[1]}), shared PCA space:")
    for k in ["within_a", "within_b", "a_to_b", "b_to_a"]:
        if k in xm:
            print(f"  {k}: {xm[k]:.1%}")
    if "transfer_gap" in xm:
        print(f"  transfer gap: {xm['transfer_gap']:+.1%} "
              f"(chance {xm['chance']:.1%})")
else:
    print("\nCross-model transfer needs activations from two models; skipping")

# %% [markdown]
# ## 9. Generate all figures

# %%
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from figures import (
    fig_ladder, fig_learning_curve, fig_layer_sweep,
    fig_dose_response, fig_confusion, fig_consistency, fig_abstain,
)

FIG_DIR = "/kaggle/working/figures"

# %% [markdown]
# ### Figure 2: Ladder comparison

# %%
if rung_data:
    fig_ladder(rung_data, out_path=f"{FIG_DIR}/ladder.pdf")

# %% [markdown]
# ### Figure 3: Learning curves

# %%
if curves:
    zs_acc = {k: rung_data[k].get("R1_zeroshot", 0) for k in curves if k in rung_data}
    fig_learning_curve(curves, zero_shot_acc=zs_acc,
                       out_path=f"{FIG_DIR}/learning_curve.pdf")

# %% [markdown]
# ### Figure 4: Layer sweep

# %%
if sweeps:
    fig_layer_sweep(sweeps, out_path=f"{FIG_DIR}/layer_sweep.pdf")

# %% [markdown]
# ### Figure 1: Dose-response

# %%
for tag_prefix, model_name in MODELS.items():
    tag = f"{tag_prefix}_cause"
    if tag in results:
        fig_dose_response(results[tag],
                         out_path=f"{FIG_DIR}/dose_response_{tag_prefix}.pdf")

# %% [markdown]
# ### Figure 6: Confusion matrices

# %%
for tag_prefix, model_name in MODELS.items():
    tag = f"{tag_prefix}_cause"
    if tag in results:
        main_r = [r for r in results[tag] if r["condition"] == "main"]
        fig_confusion(main_r, model_name=model_name,
                     out_path=f"{FIG_DIR}/confusion_{tag_prefix}.pdf")

# %% [markdown]
# ### Figure 7: Self-consistency (the core plot)

# %%
if consistency_summary:
    fig_consistency(consistency_summary, out_path=f"{FIG_DIR}/consistency.pdf")

# %% [markdown]
# ### Figure 8: Abstention trade-off

# %%
if abstain_summary:
    fig_abstain(abstain_summary, out_path=f"{FIG_DIR}/abstain.pdf")

# %% [markdown]
# ## 10. Summary table for paper

# %%
print("\n" + "=" * 78)
print("TABLE 1: Evidence-State Classification (main condition)")
print("=" * 78)

header = (f"{'Model':<20} {'R1 gen':>9} {'R2 ICL':>9} {'R3 logit':>9} "
          f"{'R4 probe':>9} {'CLIP':>9}")
print(header)
print("-" * len(header))

def _f(d, k):
    return f"{d[k]:.1%}" if k in d and d[k] is not None else "—"

for tag_prefix, model_name in MODELS.items():
    if tag_prefix in rung_data:
        rd = rung_data[tag_prefix]
        cl = f"{clip_acc:.1%}" if clip_acc is not None else "—"
        print(f"{model_name:<20} {_f(rd,'R1_zeroshot'):>9} "
              f"{_f(rd,'R2_fewshot'):>9} {_f(rd,'R3_logits'):>9} "
              f"{_f(rd,'R4_probe'):>9} {cl:>9}")

print(f"\nChance level: {1/6:.1%}")
if clip_acc is not None:
    print(f"CLIP ViT-B/32 probe: {clip_acc:.1%} ± {clip_std:.1%}")

# %%
print("\n" + "=" * 78)
print("TABLE 2: Self-consistency (does the answer survive evidence removal?)")
print("=" * 78)
h2 = f"{'Model':<20} {'S0 ceil':>9} {'S1':>7} {'S2':>7} {'S3':>7} {'S4':>7} {'floor':>8}"
print(h2)
print("-" * len(h2))
for tag_prefix, model_name in MODELS.items():
    summ = consistency_summary.get(tag_prefix)
    if not summ:
        continue
    bsc = summ["by_state_condition"]
    def g(st):
        v = bsc.get(f"{st}|main")
        return f"{v[0]:.1%}" if v and v[0] is not None else "—"
    fl = summ.get("prior_floor_pooled") or (None, 0)
    print(f"{model_name:<20} {g('S0'):>9} {g('S1'):>7} {g('S2'):>7} "
          f"{g('S3'):>7} {g('S4'):>7} "
          f"{(f'{fl[0]:.1%}' if fl[0] is not None else '—'):>8}")

# %% [markdown]
# ## 11. Blind self-relabel sheet
# Exports a shuffled, unlabelled sample plus a sealed key. Fill it in after
# at least 48 hours; ``relabel.score_sheet`` reports agreement and warns if
# you scored it too early.

# %%
from relabel import export_sheet

RELABEL_DIR = "/kaggle/working/relabel"
if not os.path.isfile(os.path.join(RELABEL_DIR, "relabel_sheet.csv")):
    export_sheet(main_items, RELABEL_DIR, n=100, seed=0)
else:
    print(f"Sheet already exists at {RELABEL_DIR} — not regenerating "
          f"(redrawing after seeing results would invalidate it)")

# To score it once filled in:
#     from relabel import score_sheet
#     import json; print(json.dumps(score_sheet(RELABEL_DIR), indent=2))

# %% [markdown]
# ## 12. Compute budget

# %%
try:
    from budget import report as budget_report, print_report
    _bud = find_file("gpu_budget.json")
    budget = print_report(_bud) if _bud else budget_report()
except Exception as e:
    print(f"budget log unavailable: {e}")
    budget = None

# %% [markdown]
# ## 12b. Threats-eliminated appendix table
# Generated from the run's own artifacts, so it cannot drift out of sync with
# what the code actually did. Rows whose evidence is unavailable say so
# instead of claiming success.

# %%
from threats import build_table, to_markdown, to_latex, check

any_results = next((results[f"{k}_cause"] for k in MODELS
                    if f"{k}_cause" in results), [])
threat_rows = build_table(
    items=main_items, folds=folds, results=any_results,
    summary={"consistency": consistency_summary,
             "clip_baseline": ({"acc": clip_acc} if clip_acc is not None else None),
             "transfer": transfer_results},
    build_stats_path=find_file("build_stats.json"),
)
check(threat_rows)
print()
print(to_markdown(threat_rows))

with open(f"{FIG_DIR}/threats_table.md", "w", encoding="utf-8") as f:
    f.write(to_markdown(threat_rows))
with open(f"{FIG_DIR}/threats_table.tex", "w", encoding="utf-8") as f:
    f.write(to_latex(threat_rows))
print(f"\nWritten to {FIG_DIR}/threats_table.{{md,tex}}")

# %% [markdown]
# ## 13. Export everything for the paper

# %%
def _clean(o):
    """Make numpy types JSON-serialisable."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o

summary = _clean({
    "rung_data": rung_data,
    "rung_detail": rung_detail,
    # best_layers is the sweep maximum — kept for the figure and for the
    # record, explicitly NOT the reported R4. probe_r4 is what the paper quotes.
    "best_layers_sweep_max_biased": {
        k: {"layer": int(v[0]), "acc": float(v[1]), "std": float(v[2])}
        for k, v in best_layers.items()},
    "probe_r4_nested": probe_r4,
    "clip_baseline": ({"acc": float(clip_acc), "std": float(clip_std)}
                      if clip_acc is not None else None),
    "curves": {k: [(int(n), float(m), float(sd)) for n, m, sd in v]
               for k, v in curves.items()},
    "consistency": consistency_summary,
    "abstention": abstain_summary,
    "statistics": stat_results,
    "transfer": transfer_results,
    "budget": budget,
    "threats": threat_rows,
    "chance": 1 / 6,
})

with open(f"{FIG_DIR}/summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved to {FIG_DIR}/summary.json")
print(f"  top-level keys: {sorted(summary)}")

# %%
# List all output files
print("\nAll output files:")
for root, dirs, files in os.walk("/kaggle/working"):
    for f in sorted(files):
        fp = os.path.join(root, f)
        print(f"  {fp} ({os.path.getsize(fp) / 1e6:.2f} MB)")

# %% [markdown]
# ## Done
# Figures are in `/kaggle/working/figures/`, every number the paper needs is
# in `summary.json`. Nothing should be typed into LaTeX by hand.
