# %% [markdown]
# # NB3: Inference - InternVL3-2B (Model B) + SmolVLM2-2.2B (Model C)
# **Accelerator: T4 GPU** - ~2 hours of GPU quota.
#
# ## What this notebook does
# 1. Runs InternVL3-2B: cause prompt with hidden states, abstain, repair
# 2. Runs SmolVLM2-2.2B: cause prompt (behavioural only if quota tight), abstain, repair
#
# ## Prerequisites
# Attach NB1's output as a dataset input.

# %% [markdown]
# ## Setup

# %%
import sys, os, json, gc, shutil
import torch
import numpy as np

# SmolVLM2 imports this package while its processor is constructed. Check it
# before spending hours on InternVL, so a minimal Kaggle image fails fast with
# the exact installation command instead of failing halfway through NB3.
try:
    import num2words  # noqa: F401
except ImportError as e:
    raise ImportError(
        "NB3 requires num2words for SmolVLM2. Run: "
        "!pip install -q num2words"
    ) from e

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %%
EVID6_SRC = "/kaggle/input/evid6-code/evid6"
EVID6_DST = "/kaggle/working/evid6"
if os.path.isdir(EVID6_SRC) and not os.path.isdir(EVID6_DST):
    shutil.copytree(EVID6_SRC, EVID6_DST)

sys.path.insert(0, "/kaggle/working/evid6/data")
sys.path.insert(0, "/kaggle/working/evid6/eval")
sys.path.insert(0, "/kaggle/working/evid6/probe")
sys.path.insert(0, "/kaggle/working/evid6/analysis")

# %%
# Locate NB1's output wherever it got mounted (its path is NB1's title
# slugified), then rebase image paths to this session. Raises listing what IS
# attached if nothing is found.
from schema import find_items
from run_inference import rebase_items

ITEMS_PATH = find_items()
print(f"Items: {ITEMS_PATH}")
ITEMS_PATH = rebase_items(ITEMS_PATH, [os.path.dirname(ITEMS_PATH),
                                       "/kaggle/input", "/kaggle/working"])
print(f"Rebased items: {ITEMS_PATH}")

# %% [markdown]
# ---
# # Model B: InternVL3-2B

# %%
# The "-hf" checkpoint, not the plain repo. OpenGVLab/InternVL3-2B ships a
# CUSTOM config (InternVLChatConfig, loaded via trust_remote_code) that the
# transformers Auto-classes cannot map. transformers 5.x has NATIVE InternVL
# support (InternVLConfig / InternVLForConditionalGeneration), and the "-hf"
# repo is the converted checkpoint that uses it - so AutoModelForImageTextToText
# maps it and no custom loader is needed. Same model, HF-native packaging.
MODEL_B_ID = "OpenGVLab/InternVL3-2B-hf"
MODEL_B_TAG = "internvl"

# %% [markdown]
# ## B.1: Load and verify

# %%
from run_inference import (load, letter_ids, run, run_generation,
                           run_choice_generation, score_one)
from prompts import (CAUSE_PROMPT, ABSTAIN_PROMPT, REPAIR_PROMPT, CLEAN_PROMPT,
                     build_fewshot_prefix, balanced_examples)
from budget import stage, print_report
from schema import load_items
from splits import make_folds

# Shared few-shot material: exemplars from folds 1-4, evaluate on fold 0.
_all = load_items(ITEMS_PATH)
_main = [it for it in _all if it.condition == "main"]
_folds = make_folds(_main, n_splits=5, seed=0)
TRAIN_POOL = [{"question": it.question, "state": it.state}
              for it, f in zip(_main, _folds) if f != 0]
EVAL_IDS = {it.item_id for it, f in zip(_main, _folds) if f == 0}
FEWSHOT_PREFIX = build_fewshot_prefix(
    balanced_examples(TRAIN_POOL, n=8, seed=0), n=8)
print(f"Few-shot prefix from {len(TRAIN_POOL)} train-fold items; "
      f"{len(EVAL_IDS)} eval-fold items")


def full_passes(model_id, tag, proc, model, cache_hidden=True, fewshot=True):
    """Every pass one model needs, sharing a single loaded copy.

    Order matters only in that the cause pass caches hidden states and is the
    most expensive; everything after it is cheap by comparison.
    """
    with stage(f"{tag}_clean_ref"):
        run_generation(model_id, ITEMS_PATH, tag=f"{tag}_clean",
                       prompt_fn=lambda q: CLEAN_PROMPT.format(q=q),
                       n_samples=3, temperature=0.7,
                       proc=proc, model=model, keep_loaded=True,
                       item_filter=lambda it: it["condition"] == "clean_ref")

    with stage(f"{tag}_treat"):
        run_generation(model_id, ITEMS_PATH, tag=f"{tag}_treat",
                       prompt_fn=lambda q: CLEAN_PROMPT.format(q=q),
                       n_samples=1, temperature=0.0,
                       proc=proc, model=model, keep_loaded=True,
                       item_filter=lambda it: (it["condition"] != "clean_ref"
                                               and it["state"] != "S5"))

    with stage(f"{tag}_cause"):
        run(model_id, ITEMS_PATH, tag=f"{tag}_cause",
            prompt_fn=lambda q: CAUSE_PROMPT.format(q=q),
            cache_hidden=cache_hidden, shard=100,
            proc=proc, model=model, keep_loaded=True)

    with stage(f"{tag}_rung1"):
        run_choice_generation(model_id, ITEMS_PATH, tag=f"{tag}_rung1",
                              prompt_fn=lambda q: CAUSE_PROMPT.format(q=q),
                              proc=proc, model=model, keep_loaded=True,
                              item_filter=lambda it: it["condition"] == "main")

    if fewshot:
        with stage(f"{tag}_rung2"):
            run_choice_generation(model_id, ITEMS_PATH, tag=f"{tag}_rung2",
                                  prompt_fn=lambda q: CAUSE_PROMPT.format(q=q),
                                  fewshot_prefix=FEWSHOT_PREFIX,
                                  proc=proc, model=model, keep_loaded=True,
                                  item_filter=lambda it: it["item_id"] in EVAL_IDS)

    with stage(f"{tag}_abstain"):
        run_generation(model_id, ITEMS_PATH, tag=f"{tag}_abstain",
                       prompt_fn=lambda q: ABSTAIN_PROMPT.format(q=q),
                       n_samples=1, temperature=0.0,
                       proc=proc, model=model, keep_loaded=True)

    with stage(f"{tag}_repair"):
        run(model_id, ITEMS_PATH, tag=f"{tag}_repair",
            prompt_fn=lambda q: REPAIR_PROMPT.format(q=q),
            cache_hidden=False,
            proc=proc, model=model, keep_loaded=True)

print(f"Loading {MODEL_B_ID}...")
proc, model = load(MODEL_B_ID)
opt_ids = letter_ids(proc, n=6)
print("Letter tokens verified ✓")

# %% [markdown]
# ## B.2: Smoke test

# %%
with open(ITEMS_PATH, encoding="utf-8") as f:
    test_item = json.loads(f.readline())

probs, hs = score_one(proc, model, test_item,
                       CAUSE_PROMPT.format(q=test_item["question"]),
                       opt_ids, want_hidden=True)
print(f"Probs: {dict(zip('ABCDEF', [f'{p:.3f}' for p in probs]))}")
print(f"Hidden shape: {hs.shape}")
assert not np.isnan(hs).any(), "NaN in hidden states!"
print("Smoke test passed ✓")

# %% [markdown]
# ## B.3: Cause prompt (with hidden states)

# %%
# The model loaded in B.1 stays live and is handed to every pass.
gc.collect(); torch.cuda.empty_cache()
print(f"GPU allocated: {torch.cuda.memory_allocated()/1e9:.1f} GB")

full_passes(MODEL_B_ID, MODEL_B_TAG, proc, model,
            cache_hidden=True, fewshot=True)

# %%
del model, proc; gc.collect(); torch.cuda.empty_cache()
print(f"Done with {MODEL_B_TAG}")

# %% [markdown]
# ---
# # Model C: SmolVLM2-2.2B

# %%
MODEL_C_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
MODEL_C_TAG = "smolvlm"

# %% [markdown]
# ## C.1: Load and verify

# %%
print(f"Loading {MODEL_C_ID}...")
proc, model = load(MODEL_C_ID)
opt_ids = letter_ids(proc, n=6)
print("Letter tokens verified ✓")

# %% [markdown]
# ## C.2: Smoke test

# %%
with open(ITEMS_PATH, encoding="utf-8") as f:
    test_item = json.loads(f.readline())

probs, hs = score_one(proc, model, test_item,
                       CAUSE_PROMPT.format(q=test_item["question"]),
                       opt_ids, want_hidden=False)  # behavioural only
print(f"Probs: {dict(zip('ABCDEF', [f'{p:.3f}' for p in probs]))}")
print("Smoke test passed ✓")

# %% [markdown]
# ## C.3: Cause prompt (behavioural only - no hidden states to save quota)
# If GPU quota allows, set `cache_hidden=True` for SmolVLM too.

# %%
CACHE_SMOL_HIDDEN = False  # Set True if you have quota headroom

gc.collect(); torch.cuda.empty_cache()
print(f"GPU allocated: {torch.cuda.memory_allocated()/1e9:.1f} GB")

# SmolVLM is behavioural-only by default to protect the quota. Flip
# CACHE_SMOL_HIDDEN above if you have headroom and want a third probe.
full_passes(MODEL_C_ID, MODEL_C_TAG, proc, model,
            cache_hidden=CACHE_SMOL_HIDDEN, fewshot=True)

# %%
del model, proc; gc.collect(); torch.cuda.empty_cache()
print(f"Done with {MODEL_C_TAG}")

# %% [markdown]
# ## Results summary

# %%
print("\n" + "=" * 60)
print("NB3 Results Summary")
print("=" * 60)

from collections import Counter

for tag in [f"{MODEL_B_TAG}_cause", f"{MODEL_C_TAG}_cause"]:
    path = f"/kaggle/working/results/{tag}.jsonl"
    if not os.path.isfile(path):
        print(f"  {tag}: not found")
        continue
    with open(path, encoding="utf-8") as f:
        results = [json.loads(l) for l in f if l.strip()]
    print(f"\n{tag}: {len(results)} items")
    for s in ["S0", "S1", "S2", "S3", "S4", "S5"]:
        subset = [r for r in results if r["state"] == s]
        if subset:
            expected = chr(65 + int(s[1]))
            correct = sum(1 for r in subset if r["pred"] == expected)
            print(f"  {s}: {correct}/{len(subset)} = {correct/len(subset):.1%}")

# %% [markdown]
# ## Output listing

# %%
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
# **Save Version** now.  Output is needed by NB4.

# %% [markdown]
# ## Compute budget

# %%
print_report()
