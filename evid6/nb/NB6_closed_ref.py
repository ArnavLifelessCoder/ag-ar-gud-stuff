# %% [markdown]
# # NB6: closed-set clean + treatment passes
# **Accelerator: GPU (T4 or P100)** — about 2.6 GPU-hours for all three models.
#
# ## Why
# The clean-reference task failed its own pre-registered stability gate on every
# model. Three samples at temperature 0.7 agreed on 176/427, 91/427 and 175/427
# reference groups — drop rates of 58.8%, 78.7% and 59.0% against a 35% ceiling.
# Open-ended answers to "What is the wine glass made of?" are not reproducible,
# so every consistency and P1/P2 number computed from them is provisional.
#
# This is not fixable by reanalysis: the reference *answers* are unstable, and
# rescoring the survivors differently does not change that. The only remedy is
# to change the reference task, which the plan anticipated — a closed answer
# set, specifically colour.
#
# ## What changes, and what deliberately does not
# Only the clean and treatment passes rerun, under **new tags** (`_clean_v2`,
# `_treat_v2`). The ladder, abstention and repair passes never touch the
# reference and are untouched here — rerunning them would burn ~6.5 GPU-hours
# to reproduce numbers that already exist.
#
# The question is replaced for *every* item with "What colour is the
# {category}?" via `make_closed_manifest`. Consistency asks whether an answer to
# a fixed question survives an intervention, so the question need not be the one
# drawn at build time; using one question everywhere maximises usable n and
# removes question type as a confound between states. Item ids, states,
# conditions, reference groups and image paths are untouched, so rows still
# align with every other pass.
#
# ## Order matters
# Qwen runs first because it is the cheapest (~0.33 h). Its drop rate is the
# gate: if the closed set does not fix stability there, it will not fix it for
# the other two, and the run stops rather than spending another 2.2 hours.
#
# ## Prerequisites
# Attach NB1's output.

# %% [markdown]
# ## Setup

# %%
import sys, os, json, gc
import torch
import numpy as np

try:
    import num2words  # noqa: F401  (SmolVLM2's processor needs it)
except ImportError as e:
    raise ImportError("NB6 requires num2words for SmolVLM2. "
                      "Run: !pip install -q num2words") from e

print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %%
from schema import find_items, make_closed_manifest, is_closed_answer, CLOSED_COLOURS
from run_inference import rebase_items, env_report, load, run_generation
from prompts import CLEAN_CLOSED_PROMPT
from consistency import build_references
from budget import stage, print_report

env_report()
print("\nclosed answer set:", ", ".join(CLOSED_COLOURS))

ITEMS_PATH = find_items()
ITEMS_PATH = rebase_items(ITEMS_PATH, [os.path.dirname(ITEMS_PATH),
                                       "/kaggle/input", "/kaggle/working"])
ITEMS_PATH = make_closed_manifest(ITEMS_PATH)
print(f"Closed-set manifest: {ITEMS_PATH}")

with open(ITEMS_PATH, encoding="utf-8") as f:
    _first = json.loads(f.readline())
print(f"Example question: {_first['question']!r}")
print("\nPrompt sent to the model:\n")
print(CLEAN_CLOSED_PROMPT.format(q=_first["question"]))

# %% [markdown]
# ## The two passes, per model


# %%
def closed_passes(model_id, tag):
    """Clean references (3 samples, T=0.7) and treatment answers (greedy).

    Returns the reference stats so the caller can gate on the drop rate.
    """
    proc, model = load(model_id)
    try:
        with stage(f"{tag}_clean_v2"):
            run_generation(model_id, ITEMS_PATH, tag=f"{tag}_clean_v2",
                           prompt_fn=lambda q: CLEAN_CLOSED_PROMPT.format(q=q),
                           n_samples=3, temperature=0.7,
                           proc=proc, model=model, keep_loaded=True,
                           item_filter=lambda it: it["condition"] == "clean_ref")

        with stage(f"{tag}_treat_v2"):
            run_generation(model_id, ITEMS_PATH, tag=f"{tag}_treat_v2",
                           prompt_fn=lambda q: CLEAN_CLOSED_PROMPT.format(q=q),
                           n_samples=1, temperature=0.0,
                           proc=proc, model=model, keep_loaded=True,
                           item_filter=lambda it: (it["condition"] != "clean_ref"
                                                   and it["state"] != "S5"))
    finally:
        del model, proc
        gc.collect(); torch.cuda.empty_cache()

    clean = [json.loads(l) for l in
             open(f"/kaggle/working/results/{tag}_clean_v2.jsonl", encoding="utf-8")
             if l.strip()]

    # Compliance: did the constraint actually bind? A prompt the model ignores
    # is not a fix, and this is the number that says which happened.
    answers = [a for r in clean for a in (r.get("answers") or [])]
    compliant = sum(1 for a in answers if is_closed_answer(a))
    print(f"\n  closed-set compliance: {compliant}/{len(answers)} "
          f"({compliant / max(len(answers), 1):.1%}) of sampled answers are "
          f"exactly one listed colour")
    off = [a for a in answers if not is_closed_answer(a)][:5]
    if off:
        print(f"  examples outside the set: {off}")

    refs, st = build_references(clean, require_stable=True)
    print(f"  references: {st['n_usable']}/{st['n_groups']} usable, "
          f"drop rate {st['drop_rate']:.1%} "
          f"({st['n_refusal']} refusals)")
    return st


# %% [markdown]
# ## Model A — Qwen2.5-VL-3B (the gate, ~0.33 h)

# %%
QWEN_STATS = closed_passes("Qwen/Qwen2.5-VL-3B-Instruct", "qwen")

GATE = 0.35
if QWEN_STATS["drop_rate"] > GATE:
    print("\n" + "=" * 70)
    print(f"GATE FAILED: drop rate {QWEN_STATS['drop_rate']:.1%} still exceeds "
          f"{GATE:.0%}.")
    print("The closed answer set did NOT fix reference stability.")
    print("Do not spend the remaining ~2.2 GPU-hours on InternVL and SmolVLM.")
    print("Report the free-form AND closed-set drop rates together: two")
    print("independent reference designs failing is a finding about the")
    print("measure, not a gap in the paper.")
    print("=" * 70)
else:
    print("\n" + "=" * 70)
    print(f"GATE PASSED: drop rate {QWEN_STATS['drop_rate']:.1%} <= {GATE:.0%}. "
          f"Was 58.8% free-form.")
    print("Continue to InternVL and SmolVLM.")
    print("=" * 70)

# %% [markdown]
# ## Models B and C — only if the gate passed
# Set `FORCE = True` to run them anyway, e.g. to document that the closed set
# fails for every model rather than only for Qwen.

# %%
FORCE = False

if QWEN_STATS["drop_rate"] <= GATE or FORCE:
    INTERNVL_STATS = closed_passes("OpenGVLab/InternVL3-2B-hf", "internvl")
    SMOLVLM_STATS = closed_passes("HuggingFaceTB/SmolVLM2-2.2B-Instruct", "smolvlm")
else:
    INTERNVL_STATS = SMOLVLM_STATS = None
    print("Skipped: the Qwen gate failed.")

# %% [markdown]
# ## Summary

# %%
print("=" * 70)
print("REFERENCE STABILITY: free-form vs closed set")
print("=" * 70)
print(f"{'model':12}{'free-form drop':>16}{'closed-set drop':>18}{'verdict':>12}")
FREE = {"qwen": 0.588, "internvl": 0.787, "smolvlm": 0.590}
for tag, st in [("qwen", QWEN_STATS), ("internvl", INTERNVL_STATS),
                ("smolvlm", SMOLVLM_STATS)]:
    if st is None:
        print(f"{tag:12}{FREE[tag]:>15.1%}{'not run':>18}{'—':>12}")
        continue
    verdict = "PASS" if st["drop_rate"] <= GATE else "FAIL"
    print(f"{tag:12}{FREE[tag]:>15.1%}{st['drop_rate']:>17.1%}{verdict:>12}")
print(f"\nGate: drop rate must be <= {GATE:.0%} (plan, 13 Aug).")

print()
print_report()

# %% [markdown]
# ## Done
# **Save Version.** Then rerun NB4 with this output attached alongside NB1, NB2,
# NB3, the SmolVLM activations run, and the previous NB4 output for the probe
# cache. NB4 prefers the `_v2` tags when it finds them and says which it used,
# so the ladder and probe numbers are untouched while consistency and the P1/P2
# verdict are recomputed on a reference that meets its own criterion.
