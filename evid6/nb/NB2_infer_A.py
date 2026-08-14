# %% [markdown]
# # NB2: Inference — Qwen2.5-VL-3B (Model A)
# **Accelerator: T4 GPU** — ~1.5 hours of GPU quota.
#
# ## What this notebook does
# 1. Loads Qwen2.5-VL-3B in fp16 with SDPA attention
# 2. Verifies letter-token round-trip (kill criterion: 8 Aug)
# 3. Generates clean reference answers (3 samples, stability check)
# 4. Runs cause-prompt inference with hidden-state caching
# 5. Runs abstain and repair prompts (no hidden states)
#
# ## Prerequisites
# Attach NB1's output as a dataset input.

# %% [markdown]
# ## Setup

# %%
import sys, os, json, gc, shutil
import torch
import numpy as np

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %%
# Copy source code if needed
EVID6_SRC = "/kaggle/input/evid6-code/evid6"
EVID6_DST = "/kaggle/working/evid6"
if os.path.isdir(EVID6_SRC) and not os.path.isdir(EVID6_DST):
    shutil.copytree(EVID6_SRC, EVID6_DST)

sys.path.insert(0, "/kaggle/working/evid6/data")
sys.path.insert(0, "/kaggle/working/evid6/eval")
sys.path.insert(0, "/kaggle/working/evid6/probe")
sys.path.insert(0, "/kaggle/working/evid6/analysis")

# %%
# Verify NB1 output is attached
NB1_DATA = "/kaggle/input/evid6-nb1-output"  # adjust path after attaching
ITEMS_PATH = f"{NB1_DATA}/items.jsonl"

# Try common paths
for candidate in [
    "/kaggle/input/evid6-nb1-output/items.jsonl",
    "/kaggle/input/evid6-dataset/items.jsonl",
    "/kaggle/working/items.jsonl",
]:
    if os.path.isfile(candidate):
        ITEMS_PATH = candidate
        break

assert os.path.isfile(ITEMS_PATH), f"items.jsonl not found. Attach NB1 output."
print(f"Using items from: {ITEMS_PATH}")

# NB1 recorded absolute paths from its own session (/kaggle/working/evid6/...).
# Here the images arrive as an attached dataset under a different root, so
# those paths do not resolve and the first Image.open would kill the pass —
# after the model has already loaded. Rebase before scoring anything.
from run_inference import rebase_items

ITEMS_PATH = rebase_items(ITEMS_PATH, [NB1_DATA,
                                       "/kaggle/input/evid6-nb1-output",
                                       "/kaggle/input/evid6-dataset",
                                       "/kaggle/working"])
print(f"Rebased items: {ITEMS_PATH}")

with open(ITEMS_PATH, encoding="utf-8") as f:
    n_items = sum(1 for l in f if l.strip())
print(f"Total items: {n_items}")

# %% [markdown]
# ## Model setup

# %%
MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MODEL_TAG = "qwen"

# %% [markdown]
# ## Step 1: Load model and verify letter tokens

# %%
from run_inference import load, letter_ids

print(f"Loading {MODEL_ID}...")
proc, model = load(MODEL_ID)
print("Model loaded successfully")

# %%
# Kill criterion (8 Aug): letter tokens must round-trip
opt_ids = letter_ids(proc, n=6)
print("Letter token IDs:", opt_ids)
for i, tid in enumerate(opt_ids):
    L = chr(65 + i)
    decoded = proc.tokenizer.decode([tid]).strip()
    print(f"  {L} -> token {tid} -> '{decoded}' {'✓' if decoded == L else '✗ FAIL'}")

# %% [markdown]
# ## Step 2: Smoke test — one item

# %%
from prompts import CAUSE_PROMPT, CLEAN_PROMPT, ABSTAIN_PROMPT, REPAIR_PROMPT
from run_inference import score_one, generate_one

with open(ITEMS_PATH, encoding="utf-8") as f:
    test_item = json.loads(f.readline())

print(f"Test item: {test_item['item_id']} (state={test_item['state']})")
print(f"Question: {test_item['question']}")

# Forced-choice scoring
probs, hs = score_one(proc, model, test_item,
                       CAUSE_PROMPT.format(q=test_item["question"]),
                       opt_ids, want_hidden=True)
print(f"\nLogit probabilities: {dict(zip('ABCDEF', [f'{p:.3f}' for p in probs]))}")
print(f"Prediction: {chr(65 + int(probs.argmax()))}")
print(f"Hidden states shape: {hs.shape}")

# Generation
answer = generate_one(proc, model, test_item,
                       CLEAN_PROMPT.format(q=test_item["question"]))
print(f"\nGenerated answer: '{answer}'")

# %%
# Check for NaN in hidden states (critical on T4)
assert not np.isnan(hs).any(), "NaN in hidden states! Check dtype (must be fp16, not bf16)"
assert not np.isinf(hs).any(), "Inf in hidden states!"
print("Hidden state sanity check passed ✓")

# %% [markdown]
# ## Step 3: Clean reference answers
# Three samples at temperature=0.7, keep only stable items.

# %%
from run_inference import run_generation

from budget import stage

# 3a. The REFERENCE answers: untouched images only (condition == clean_ref),
#     three samples each so we can drop unstable ones.
print("Generating clean reference answers (3 samples each)...")
with stage(f"{MODEL_TAG}_clean_ref"):
    run_generation(
        MODEL_ID, ITEMS_PATH,
        tag=f"{MODEL_TAG}_clean",
        prompt_fn=lambda q: CLEAN_PROMPT.format(q=q),
        n_samples=3,
        temperature=0.7,
        proc=proc, model=model, keep_loaded=True,
        item_filter=lambda it: it["condition"] == "clean_ref",
    )

# %%
# 3b. The TREATMENT answers: the same question on the degraded images.
#     Greedy, one sample. S5 is skipped — no evidence was removed there, so
#     consistency is undefined (schema.CONSISTENCY_STATES).
print("Generating treatment answers on degraded images...")
with stage(f"{MODEL_TAG}_treat"):
    run_generation(
        MODEL_ID, ITEMS_PATH,
        tag=f"{MODEL_TAG}_treat",
        prompt_fn=lambda q: CLEAN_PROMPT.format(q=q),
        n_samples=1,
        temperature=0.0,
        proc=proc, model=model, keep_loaded=True,
        item_filter=lambda it: (it["condition"] != "clean_ref"
                                and it["state"] != "S5"),
    )

# %%
# Check stability
clean_path = f"/kaggle/working/results/{MODEL_TAG}_clean.jsonl"
with open(clean_path, encoding="utf-8") as f:
    clean_results = [json.loads(l) for l in f if l.strip()]

from consistency import build_references

refs, ref_stats = build_references(clean_results, require_stable=True)
for k, v in ref_stats.items():
    print(f"  {k}: {v}")

# Kill criterion (13 Aug): drop rate > 35% means the reference is too noisy.
if ref_stats["drop_rate"] > 0.35:
    print("\nWARNING: drop rate exceeds 35%. Per the plan, switch to a "
          "closed-set colour question only, and say so in the paper.")
else:
    print(f"\nReference set usable: {ref_stats['n_usable']} groups.")

# %% [markdown]
# ## Step 4: Cause-prompt inference (with hidden states)
# This is the main sweep — forced-choice logit scoring + residual stream caching.

# %%
# NOTE: the model stays loaded from Step 1 and is handed to every runner
# below.  Do not call load() again here — two copies of a 3B model in fp16
# will not fit alongside activations on a 15 GB T4.
from run_inference import run

gc.collect(); torch.cuda.empty_cache()
print(f"GPU allocated: {torch.cuda.memory_allocated()/1e9:.1f} GB")

print("Running cause-prompt inference with hidden-state caching...")
with stage(f"{MODEL_TAG}_cause", n_items=n_items):
    run(
        MODEL_ID, ITEMS_PATH,
        tag=f"{MODEL_TAG}_cause",
        prompt_fn=lambda q: CAUSE_PROMPT.format(q=q),
        cache_hidden=True,
        shard=100,
        proc=proc, model=model, keep_loaded=True,
    )

# %% [markdown]
# ## Step 4b: Rung 1 and rung 2 — the SAME task, answered by generation
# Rung 3 (above) reads option-token logits. Rung 1 is what the model actually
# emits when asked. Rung 2 adds an 8-example text-only in-context prefix built
# from TRAINING-FOLD items only, so it cannot leak the test fold.

# %%
from run_inference import run_choice_generation
from prompts import build_fewshot_prefix, balanced_examples
from schema import load_items
from splits import make_folds

all_items = load_items(ITEMS_PATH)
main_items = [it for it in all_items if it.condition == "main"]
folds = make_folds(main_items, n_splits=5, seed=0)

# Fold 0 is the evaluation fold for rung 2; exemplars come from folds 1-4.
train_pool = [{"question": it.question, "state": it.state}
              for it, f in zip(main_items, folds) if f != 0]
eval_ids = {it.item_id for it, f in zip(main_items, folds) if f == 0}
prefix = build_fewshot_prefix(balanced_examples(train_pool, n=8, seed=0), n=8)
print(f"Few-shot prefix built from {len(train_pool)} training-fold items")
print(prefix[:400])

# %%
# Rung 1: zero-shot generation, all main items
with stage(f"{MODEL_TAG}_rung1"):
    run_choice_generation(
        MODEL_ID, ITEMS_PATH,
        tag=f"{MODEL_TAG}_rung1",
        prompt_fn=lambda q: CAUSE_PROMPT.format(q=q),
        proc=proc, model=model, keep_loaded=True,
        item_filter=lambda it: it["condition"] == "main",
    )

# %%
# Rung 2: same, with the in-context prefix, on the held-out fold only
with stage(f"{MODEL_TAG}_rung2"):
    run_choice_generation(
        MODEL_ID, ITEMS_PATH,
        tag=f"{MODEL_TAG}_rung2",
        prompt_fn=lambda q: CAUSE_PROMPT.format(q=q),
        fewshot_prefix=prefix,
        proc=proc, model=model, keep_loaded=True,
        item_filter=lambda it: it["item_id"] in eval_ids,
    )

# %% [markdown]
# ## Step 5: Abstain prompt (no hidden states)

# %%
gc.collect(); torch.cuda.empty_cache()

print("Running abstain-prompt inference...")
with stage(f"{MODEL_TAG}_abstain"):
    run_generation(
        MODEL_ID, ITEMS_PATH,
        tag=f"{MODEL_TAG}_abstain",
        prompt_fn=lambda q: ABSTAIN_PROMPT.format(q=q),
        n_samples=1,
        temperature=0.0,
        proc=proc, model=model, keep_loaded=True,
    )

# %% [markdown]
# ## Step 6: Repair prompt (no hidden states)

# %%
gc.collect(); torch.cuda.empty_cache()

print("Running repair-prompt inference...")
with stage(f"{MODEL_TAG}_repair"):
    run(
        MODEL_ID, ITEMS_PATH,
        tag=f"{MODEL_TAG}_repair",
        prompt_fn=lambda q: REPAIR_PROMPT.format(q=q),
        cache_hidden=False,
        proc=proc, model=model, keep_loaded=True,
    )

# %% [markdown]
# ## Compute budget

# %%
from budget import print_report
print_report()

# %% [markdown]
# ## Quick results check

# %%
cause_path = f"/kaggle/working/results/{MODEL_TAG}_cause.jsonl"
with open(cause_path, encoding="utf-8") as f:
    cause_results = [json.loads(l) for l in f if l.strip()]

from collections import Counter
pred_dist = Counter(r["pred"] for r in cause_results)
state_dist = Counter(r["state"] for r in cause_results)
print(f"Total scored: {len(cause_results)}")
print(f"Prediction distribution: {dict(pred_dist)}")
print(f"State distribution: {dict(state_dist)}")

# Per-state accuracy
for s in ["S0", "S1", "S2", "S3", "S4", "S5"]:
    subset = [r for r in cause_results if r["state"] == s]
    if subset:
        expected = chr(65 + int(s[1]))
        correct = sum(1 for r in subset if r["pred"] == expected)
        print(f"  {s}: {correct}/{len(subset)} = {correct/len(subset):.1%}")

# %% [markdown]
# ## Cleanup and save

# %%
del model, proc
gc.collect()
torch.cuda.empty_cache()

# List outputs
print("\nOutput files:")
for d in ["results", "acts"]:
    path = f"/kaggle/working/{d}"
    if os.path.isdir(path):
        for root, dirs, files in os.walk(path):
            for f in files:
                fp = os.path.join(root, f)
                print(f"  {fp} ({os.path.getsize(fp) / 1e6:.1f} MB)")

# %% [markdown]
# ## Done
# **Save Version** now.  The output becomes a dataset input for NB4.
# Hidden states in `acts/qwen_cause/` are needed for probing.
