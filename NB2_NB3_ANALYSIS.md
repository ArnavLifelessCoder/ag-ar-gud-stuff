# EVID-6 results - Qwen2.5-VL-3B, InternVL3-2B, SmolVLM2-2.2B

Analysed 16 Aug 2026 from `vlm nb2 results/` (NB2), `results-nb3/` (NB3) and
`vlm nb5 output/` (the completed NB4 analysis). The inference numbers were
recomputed directly from the saved JSONL rather than copied from a notebook
display; the probe, CLIP, transfer and figure numbers come from NB4's own
`figures/summary.json`.

**Status: the pipeline has run end to end.** NB1 built 1,838 items from real
COCO, NB2/NB3 scored them with three models (9.18 GPU-h), NB4 produced every
figure and `summary.json`. One pre-registered criterion failed - reference
stability - and that failure is reported in §3 rather than worked around.

## What ran

The manifest is 1,838 rows: 900 main (150 each for S0–S5), 427 clean
references, 211 S0 artifact controls, 300 prior-only (150 each for S2 and S3).
All three models ran all seven passes over it.

| Model | Checkpoint revision | GPU | Activations |
|---|---|---:|---|
| Qwen2.5-VL-3B-Instruct | `66285546d2b821cf421d4f5eb2576359d3770cd3` | 1.19 h | 1838 × 37 × 2048 |
| InternVL3-2B-**hf** | `cb57a075cb75a2e6d1b668b128d48bb00ae321d2` | 4.85 h | 1838 × 29 × 1536 |
| SmolVLM2-2.2B-Instruct | `482adb537c021c86670beed01cd58990d01e72e4` | 3.13 h + 0.64 h | 1838 × 25 × 2048 |

Total 9.18 GPU-hours. Row counts are identical across all three models and all
seven passes (clean 427, treat 1,261, cause 1,838, rung1 900, rung2 179,
abstain 1,838, repair 1,838), so the three are directly comparable.

NB3 ran SmolVLM with `CACHE_SMOL_HIDDEN = False` to protect quota, so its
activations were captured afterwards by `NB5_smolvlm_acts.py` (0.64 GPU-h, zero
non-finite rows, R3 reproduced at 19.9% exactly). **All three models therefore
have a probe.**

Total consumed is **9.82 GPU-h** across every run; the *reproduction* cost is
~9.2 h, since `smolvlm_cause` genuinely ran twice - once behaviourally and once
with activations. Report one and footnote the other.

---

## 1. The headline: behaviour is weak, activations are not

NB4 completed 16 Aug. R4 is the nested probe - each outer fold picks its layer
using only its training folds, so the number carries no selection bias.

| Model | R1 zero-shot | R2 few-shot | R3 logit | **R4 probe** | **R1 → R4** |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-3B | 41.0% | 39.1% | 41.0% | **73.0% ±2.7** | **+32.0** |
| InternVL3-2B | 38.6% | 35.8% | 39.1% | **79.2% ±2.2** | **+40.7** |
| SmolVLM2-2.2B | 23.9% | 21.8% | 19.9% | **72.9% ±5.5** | **+49.0** |
| *CLIP ViT-B/32 probe* | | | | *44.3% ±3.3* | |
| *chance* | | | | *16.7%* | |

### SmolVLM is the result: representation and reportability dissociate

SmolVLM2-2.2B **cannot do this task**. R3 is 19.9% against 16.7% chance, it
never emits B, E or F, and 84% of its 900 main-condition predictions are a
single option (D). By every behavioural measure it is at floor.

A linear probe on its layer-17 activations reads the six states at **72.9%** -
statistically indistinguishable from Qwen's 73.0%, and 28.6 points above the
CLIP control. Its R1→R4 gap of **+49.0** is the largest of the three.

This is a stronger claim than "models under-report what they represent". A
model can **encode why it cannot see while being wholly unable to say so**:
representation and reportability are dissociable, and the gap does not track
capability. Had SmolVLM's probe also come in near chance, the honest reading
would have been the opposite - that the representation simply tracks how good
the model is. It does not.

Two caveats to state rather than smooth over: SmolVLM's fold variance is
notably wider (**±5.5** against ±2.7 and ±2.2), and its per-fold layer choices
scatter more (22/23/17/21/21). Quote the spread, not a layer index.

Selection bias was small and is reported rather than absorbed: max-over-layers
would have read 74.6% for Qwen (+1.6) and 79.6% for InternVL (+0.4). Layers
chosen per fold were 24/21/27/33/23 (Qwen) and 21/21/21/19/20 (InternVL) - say
that InternVL localises tightly and Qwen does not, rather than quoting one
layer index for both.

**The CLIP baseline is the control that makes this claim survive.** A linear
probe on frozen CLIP ViT-B/32 image features reaches 44.3%. So:

- The VLM probe beats it by **+28.8** (Qwen) and **+35.0** (InternVL). The
  representation is not merely what a generic vision encoder already exposes,
  which is the obvious reviewer objection. The pre-registered kill criterion
  (CLIP ≥ VLM probe → drop absolute numbers) does **not** fire.
- CLIP at 44.3% is well above chance, so some evidence state is visible in
  generic image statistics - occlusion, blur and cropping are, after all,
  visually detectable. Say this; it is not a weakness.
- **CLIP also beats what Qwen actually reports** (44.3% vs R1 41.0%). A frozen
  encoder with a linear head outperforms the VLM's own answer. That is the
  sharpest one-line statement of the gap available.

Few-shot prompting does not help any model. On the 179 held-out rows R2 is
below R1 restricted to the same rows in all three cases (Qwen 39.1 vs 41.3;
InternVL 35.8 vs 38.0; SmolVLM 21.8 vs 24.0).

Parsing is not the bottleneck. Unparseable replies: Qwen 5/900 (0.6%),
InternVL 0, SmolVLM 0.

### The probe is not reading object identity

Leave-one-category-out barely costs anything: Qwen **74.2% ±12.0** against
74.6% within-distribution, InternVL **79.9% ±14.9** against 79.6%, SmolVLM
**69.5% ±12.0** against 74.7%. Held-out categories are classified nearly as
well as seen ones, so the probe is not a category detector. SmolVLM's drop
(+5.3) is the largest of the three but still leaves it 25 points above CLIP.
Worst groups were `tv` (58.3%) and `chair` (53.8%, 50.0%).

### The representation does not transfer across models

In a shared PCA space, within-model accuracy is 65.4% (Qwen) and 71.9%
(InternVL), but cross-model is **17.6%** and **20.8%** against 16.7% chance -
a 47.8-point gap. Whatever encodes evidence state is real and linearly
decodable inside each model, and is *not* a shared direction between them.
Report this as a negative result, not an omission.

### Learning curves: accessible, not merely learnable

| n | 10 | 25 | 50 | 100 | 250 | 500 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen (layer 22) | 29.9% | 44.3% | 52.2% | 56.9% | 66.2% | 72.5% |
| InternVL (layer 19) | 52.4% | 54.2% | 60.2% | 67.8% | 72.8% | 77.5% |
| SmolVLM (layer 17) | 39.2% | 40.7% | 50.5% | 60.7% | 64.4% | 70.8% |

InternVL is already above chance-by-a-wide-margin at n=10 and above CLIP's
44.3% by n=10, which argues the structure is genuinely accessible rather than
fitted. Neither curve has saturated at n=500, so these R4 figures are lower
bounds.

---

## 2. State classification collapses, differently for each model

Main-condition R3 accuracy per state, and the prediction distribution:

| | S0 | S1 | S2 | S3 | S4 | S5 | prediction counts |
|---|---:|---:|---:|---:|---:|---:|---|
| Qwen | 80.0% | 1.3% | 8.7% | 60.0% | **0.0%** | 96.0% | A=327 B=15 C=20 D=220 **E=0** F=318 |
| InternVL | 81.3% | **0.0%** | 24.0% | 52.7% | **0.0%** | 76.7% | A=326 **B=0** C=88 D=250 **E=0** F=236 |
| SmolVLM | 18.0% | **0.0%** | 14.7% | 86.7% | **0.0%** | **0.0%** | A=92 **B=0** C=55 **D=753** |

Three findings:

**No model ever predicts E / S4.** Not once, across 2,700 main-condition
predictions. Ambiguous reference is not represented in the output distribution
of any of these models under this prompt. Report the zero column; do not
average it away.

**S1 (out of frame) is equally dead** - 1.3%, 0.0%, 0.0%. Qwen emits B fifteen
times; the other two never do.

**SmolVLM's 86.7% on S3 is an artifact, not a result.** It answers D for 753 of
900 items (84%). A constant-D predictor scores 150/900 = 16.7% overall and
100% on S3. SmolVLM's 19.9% overall is barely above that degenerate baseline,
and its S3 column should not be read as evidence it detects degradation. Its
S5 score is 0.0% where Qwen reaches 96.0%.

---

## 3. Self-consistency fails its own pre-registered gate, in all three models

`build_references` keeps a group only if three samples at temperature 0.7 agree
after normalisation. The plan's 13 Aug kill criterion fires above a 35% drop
rate.

| Model | Usable references | Drop rate | Verdict |
|---|---:|---:|---|
| Qwen | 176 / 427 | **58.8%** | fails |
| InternVL | 91 / 427 | **78.7%** | fails badly |
| SmolVLM | 175 / 427 | **59.0%** | fails |

**This is not fixable by reanalysis.** The reference answers themselves are
unstable; rescoring the survivors differently does not change that. Everything
in this section is therefore provisional and must be reported as a criterion
failure, not as a P1/P2 result.

Strict matching (the headline rule) on the surviving subsets:

| Quantity | Qwen | InternVL | SmolVLM |
|---|---:|---:|---:|
| S0 main ceiling | 98.4% (n=63) | 100% (n=30) | 98.4% (n=61) |
| S0 artifact control | 96.7% (n=90) | 96.0% (n=50) | 97.6% (n=83) |
| Pooled prior-only floor | 47.7% (n=132) | 41.9% (n=74) | 45.5% (n=121) |
| S2 main / its floor | 49.2 / 44.6 (n=65) | 36.8 / 34.2 (n=38) | 42.2 / 45.3 (n=64) |
| S3 main / its floor | 55.2 / 50.7 (n=67) | 63.9 / 50.0 (n=36) | 52.6 / 45.6 (n=57) |
| S3 curve, sev 1→2→3 | 81.0 → 52.2 → 34.8 | 91.7 → 45.5 → **53.8** | 70.0 → 55.6 → 31.6 |
| Max strict-vs-relaxed delta | 7.5% | 8.3% | 6.2% |

**P1 is "supported" for all three** - S2 sits at its prior-only floor, which is
what the taxonomy predicts if occlusion destroys the signal. But note this is
*supported by a null*: S2 main and its floor are both low and close, which is
also what you would see if the reference were simply noisy. With drop rates
this high, that reading cannot be excluded.

**P2 is "challenged" for all three.** Every model's severity-3 consistency
reaches or falls below its S3 prior floor. InternVL is additionally
non-monotone (91.7 → 45.5 → 53.8), on n = 12 / 11 / 13 per severity - too few
items to interpret as anything but noise.

The strict-vs-relaxed delta exceeds the 5-point threshold for all three, so the
matching rule is load-bearing wherever these numbers are used. Strict is the
headline; relaxed is the sensitivity arm.

---

## 4. Abstention: two opposite failure modes

| Model | AbsAcc | OverAbs | UnderAbs | Always-abstain baseline |
|---|---:|---:|---:|---:|
| Qwen | 64.1% | 8.5% | 56.5% | 57.1% |
| InternVL | **71.4%** | **45.7%** | 15.7% | 57.1% |
| SmolVLM | 62.9% | 8.9% | 58.2% | 57.1% |

Per-state abstention rate:

| | S0 | S1 | S2 | S3 | S4 | S5 |
|---|---:|---:|---:|---:|---:|---:|
| Qwen | 8.5% | 54.0% | 41.3% | 36.0% | 10.7% | 85.3% |
| InternVL | 45.7% | 94.7% | 88.7% | 87.0% | 44.7% | 99.3% |
| SmolVLM | 8.9% | 58.7% | 36.0% | 36.0% | 8.0% | 82.0% |

**InternVL's higher AbsAcc is bought by over-abstention, not by calibration.**
It refuses 45.7% of genuinely answerable S0 items and 44.7% of S4. Qwen and
SmolVLM sit at the opposite extreme, answering ~57% of items where the evidence
is absent. Neither profile is calibrated uncertainty; they are opposite priors.
Reporting AbsAcc alone would rank InternVL best and would be misleading - which
is exactly why `abstain.py` returns OverAbs and UnderAbs beside it.

---

## 5. Repair selection tracks the cause collapse

Accuracy on the 900 main rows:

| Model | Overall | S0 | S1 | S2 | S3 | S4 | S5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen | 26.4% | 0.7% | 0.0% | 0.0% | 88.0% | 0.0% | 70.0% |
| InternVL | 19.9% | 0.7% | 8.7% | 11.3% | 73.3% | 25.3% | 0.0% |
| SmolVLM | 17.6% | 95.3% | 10.0% | 0.0% | 0.0% | 0.0% | 0.0% |

Each model concentrates on one or two repair actions and ignores the rest -
Qwen on "zoom in" and "correct the premise", SmolVLM almost entirely on
"answer". Diagnostic, not a repair-policy result.

*(An earlier draft reported Qwen repair as 20.0%. That was computed over all
1,838 rows while the per-state figures beside it used the 900 main rows. The
1,838-row population is also wrong for this metric: `s0ctrl`, `prioronly` and
`clean_ref` rows inherit a state label whose correct repair is undefined - a
prior-only row labelled S2 has the referent painted out, so "move to another
angle" cannot fix it.)*

---

## 6. Eight Qwen rows are numerically invalid

The Qwen activation export has 19 paired shards totalling 1,838 rows × 37
layers × 2,048 dims. **Eight rows (0.44%) contain NaN in every layer above
layer 0, and their forced-choice probabilities are NaN too.** All eight were
saved with prediction `A`, because `argmax` over NaN returns index 0 - so those
R3 labels are artifacts, not predictions.

| State / condition | Rows |
|---|---:|
| S2 main | 3 |
| S2 prior-only | 2 |
| S3 main | 2 |
| S3 prior-only | 1 |

Item IDs: `6224ea5e7870`, `a271d8393673`, `1bb4bc7fc30f`, `be3af9bacf15`,
`b709f755f6f4`, `830a61c8dbe1`, `648e2d7bcf2d`, `63ac488caa17`.

Five of the eight are main-condition rows, so they survive NB4's alignment step
and reach the probe. Before the fix this raised
`ValueError: Input X contains NaN` partway through the layer sweep. NB4 now
filters non-finite rows, prints their IDs, and continues - the exclusion must
be reported.

**InternVL and SmolVLM have zero NaN rows.** This is Qwen-specific.

`score_one` still has no per-item finite-value guard, so a rerun would
reproduce the problem. That guard is the one code change still owed.

---

## 7. Decision: preserve, fix, then rerun selectively

**Do not present these exports as the paper's final consistency evidence.** They
are a complete and useful pilot: the pipeline runs end to end on three models,
the R1→R4 gap replicates, and the failure modes are now precisely located. But
the clean-reference stability criterion fails on all three, and eight Qwen rows
are invalid.

Keep both downloaded folders unchanged for auditability.

**What does *not* need rerunning:** the ladder (`cause`, `rung1`, `rung2`),
abstention, and repair never touch the reference. The R1→R4 gap is unaffected
by the reference failure.

**What does:** `clean_ref` + `treat` only - 28% of each sweep, **2.55 GPU-hours
across all three models**, not the full 9.18. Use a constrained clean-answer
task (the plan suggests closed-set colour questions) so the drop rate can meet
35%.

> **Trap:** every runner resumes by tag. It reads the existing `{tag}.jsonl`,
> collects `item_id`s and skips them. Re-running `qwen_clean` with a new prompt
> under the same tag processes **zero items** and reports success. Use new tags
> (`qwen_clean_v2`, `qwen_treat_v2`) or delete the old files first.

### The run worth doing

Rerun `smolvlm_cause` with `cache_hidden=True` (~0.62 GPU-h). SmolVLM is at
**chance behaviourally** - 19.9%, three dead option letters, 84% of predictions
on a single option. If a linear probe on its activations still separates the
six states, that is the strongest possible form of this paper's thesis: a model
that encodes *why it cannot see* while being completely unable to report it.
Right now that cannot be tested, because no activations were saved.

---

## 8. Remaining work, in order

1. ~~Run NB4~~ **done 16 Aug.** Took 6.5 h on Kaggle CPU, not the 1.5 h
   estimated - the nested probe alone was 3.7 h for Qwen. Results above.
   `figures/probe_cache.json` now lets a rerun skip it in ~10 min.
2. **Add a finite-value guard to `score_one`** - log the item ID and retry once
   rather than writing an `A` prediction over NaN logits.
3. **Rerun `clean` + `treat`** under new tags with a constrained answer task,
   then recompute consistency. 2.55 GPU-h.
4. **Optionally rerun `smolvlm_cause` with activations** (0.62 GPU-h) for the
   three-model probe comparison described above.
5. **Human checks:** the blind 100-item relabel after its 48-hour cooldown, and
   Tier-B hand sorting. The real-COCO QA sheets have been reviewed and passed
   (`qa_real_coco_pilot/`).
6. **Commit and push.** Kaggle's setup cell clones `origin/main`, so unpushed
   analysis fixes never reach the notebooks.

## What to claim, and what not to

**Claim:** a linear probe on mid-layer activations recovers evidence state far
above what the same model reports behaviourally (+26 and +35 points on two
architectures), and few-shot prompting does not close that gap.

**Do not claim:** a P1/P2 verdict, calibrated abstention, or a repair policy.
Report the reference-stability failure as a finding about free-form reference
tasks - it is an honest negative, not a hole.
