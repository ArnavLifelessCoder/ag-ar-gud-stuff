# %% [markdown]
# # NB5: SmolVLM2 cause pass WITH hidden states
# **Accelerator: GPU (T4 or P100)** — about 0.7 GPU-hours.
#
# ## Why this run exists
# NB3 ran SmolVLM with `CACHE_SMOL_HIDDEN = False` to protect quota, so it has
# behavioural results but no activations and therefore no rung-4 probe.
#
# That gap sits exactly on the paper's most interesting question. SmolVLM is at
# **chance behaviourally** — R3 19.9% against 16.7%, and it answers option D on
# 753 of 900 main items. Qwen and InternVL both *report* poorly (41.0%, 38.6%)
# while their activations carry the state (73.0%, 79.2%). If SmolVLM turns out
# to encode the states too, the claim strengthens from "models under-report what
# they represent" to "even a model that cannot perform the task at all still
# represents it". If it does not, that is an equally clean result: the
# representation tracks capability, and the gap is not universal.
#
# Either answer is worth 0.7 GPU-hours.
#
# ## What it does NOT do
# Only the cause pass, only SmolVLM. Re-running all of NB3 would redo
# InternVL's 4.85 GPU-hours for output that already exists. Every other
# SmolVLM pass (clean, treat, rung1, rung2, abstain, repair) is already saved
# in the NB3 output and is untouched here.
#
# ## Prerequisites
# Attach NB1's output. Attaching NB3's output as well is harmless.

# %% [markdown]
# ## Setup

# %%
import sys, os, json, gc
import torch
import numpy as np

# SmolVLM2's processor imports this at construction time.
try:
    import num2words  # noqa: F401
except ImportError as e:
    raise ImportError("NB5 requires num2words for SmolVLM2. "
                      "Run: !pip install -q num2words") from e

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %%
from schema import find_items
from run_inference import rebase_items, env_report

env_report()

ITEMS_PATH = find_items()
print(f"Items: {ITEMS_PATH}")
ITEMS_PATH = rebase_items(ITEMS_PATH, [os.path.dirname(ITEMS_PATH),
                                       "/kaggle/input", "/kaggle/working"])
print(f"Rebased items: {ITEMS_PATH}")

# %% [markdown]
# ## Load SmolVLM2 and verify the option tokens
# Same tag as NB3 (`smolvlm_cause`) so NB4 picks this up as a drop-in. The
# behavioural rows will match NB3's — the cause pass is a deterministic forward
# pass — and the difference is that this run also writes `acts/smolvlm_cause/`.

# %%
from run_inference import load, letter_ids, run, score_one
from prompts import CAUSE_PROMPT
from budget import stage, print_report

MODEL_C_ID = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
MODEL_C_TAG = "smolvlm"

print(f"Loading {MODEL_C_ID}...")
proc, model = load(MODEL_C_ID)
opt_ids = letter_ids(proc, n=6)
print("Letter tokens verified ✓", opt_ids)

# %% [markdown]
# ## Smoke test — one item, WITH hidden states
# NB3's smoke test asked for `want_hidden=False`, so nothing checked the
# hidden-state path for this model. Qwen's run produced eight rows of NaN
# activations that were only discovered in NB4, long after the quota was spent.
# Check the shape and finiteness here, on one item, before the sweep.

# %%
with open(ITEMS_PATH, encoding="utf-8") as f:
    test_item = json.loads(f.readline())

probs, hs = score_one(proc, model, test_item,
                      CAUSE_PROMPT.format(q=test_item["question"]),
                      opt_ids, want_hidden=True)
print(f"Probs: {dict(zip('ABCDEF', [f'{p:.3f}' for p in probs]))}")
print(f"Hidden states: {hs.shape}  dtype={hs.dtype}")
assert np.isfinite(hs).all(), "non-finite hidden states on the very first item"
assert np.isfinite(probs).all(), "non-finite option probabilities"
print("Hidden-state path verified ✓")

est_gb = hs.nbytes * 1838 / 1e9
print(f"Projected activation size for 1,838 rows: {est_gb:.2f} GB")

# %% [markdown]
# ## The cause pass, with activations
# ~0.7 GPU-hours. Resumable: `run()` skips items already in
# `/kaggle/working/results/smolvlm_cause.jsonl`, so a session timeout costs the
# current item rather than the pass.

# %%
gc.collect(); torch.cuda.empty_cache()
print(f"GPU allocated before: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

with stage(f"{MODEL_C_TAG}_cause", n_items=1838):
    run(MODEL_C_ID, ITEMS_PATH, tag=f"{MODEL_C_TAG}_cause",
        prompt_fn=lambda q: CAUSE_PROMPT.format(q=q),
        cache_hidden=True, shard=100,
        proc=proc, model=model, keep_loaded=True)

print(f"GPU allocated after: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

# %% [markdown]
# ## Verify what was written
# Confirm the activations are complete and finite before the session ends —
# this is the check whose absence cost Qwen eight unusable rows.

# %%
import glob

ACTS = f"/kaggle/working/acts/{MODEL_C_TAG}_cause"
h_files = sorted(glob.glob(f"{ACTS}/h_*.npy"))
i_files = sorted(glob.glob(f"{ACTS}/i_*.npy"))
print(f"shards: {len(h_files)} h_*.npy, {len(i_files)} i_*.npy")
assert len(h_files) == len(i_files) and h_files, "shard mismatch"

H = np.concatenate([np.load(p) for p in h_files])
I = np.concatenate([np.load(p) for p in i_files])
print(f"activations: {H.shape}  ids: {I.shape}")

bad = ~np.isfinite(H.astype(np.float32)).all(axis=(1, 2))
print(f"non-finite rows: {int(bad.sum())} / {len(H)}")
if bad.any():
    print("  ids:", sorted(I[bad].tolist()))
    print("  NB4 will drop and report these; rerun them to recover.")

res = f"/kaggle/working/results/{MODEL_C_TAG}_cause.jsonl"
rows = [json.loads(l) for l in open(res, encoding="utf-8") if l.strip()]
print(f"result rows: {len(rows)}  (expect 1838)")
assert len(rows) == len(H), "results and activations disagree in length"

# Sanity: does this reproduce NB3's behavioural numbers?
STATES = ["S0", "S1", "S2", "S3", "S4", "S5"]
main = [r for r in rows if r["condition"] == "main"]
acc = np.mean([r["pred"] == chr(65 + STATES.index(r["state"])) for r in main])
print(f"\nR3 on {len(main)} main rows: {acc:.1%}  (NB3 recorded 19.9%)")
if abs(acc - 0.199) > 0.02:
    print("  NOTE: differs from NB3 by more than 2 points — investigate before "
          "treating the activations as matched to the existing behavioural rows.")

# %% [markdown]
# ## Budget

# %%
print_report()

# %% [markdown]
# ## Done
# **Save Version.** Then attach this output to NB4 alongside NB1/NB2/NB3 and
# the previous NB4 output — the cached Qwen and InternVL probes are reused, and
# only SmolVLM's probe is computed. NB4 will report a third R4 and a third row
# in the ladder table.
