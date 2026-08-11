# Typed Evidence States in Vision-Language Models
### Plan v4: zero budget, Kaggle free tier, matched to the implemented code

**Working title:** *Do VLMs represent why they cannot see?*
**Benchmark:** `EVID-6`
**Total cost: 0.** No API calls, no paid compute, no paid annotation, no paid storage.

This supersedes v3. The scientific claim, positioning and paper structure carry
over unchanged. What changed is that the code got written, tested and run, and
several things v3 specified turned out to be wrong when measured.

**The repository is authoritative.** v3 embedded full source listings that have
since diverged from what actually runs. This version keeps only the excerpts that
carry a design decision and points at the module for the rest. If a snippet here
disagrees with the repo, the repo is right.

---

## Changelog from v3

| Change | Reason |
|---|---|
| **S3 severity targets an absolute resolution, not a division factor** | Fixed factors made the dose depend on referent size. At factor 24 a 63×63 object became 2×2 — deletion, so S3 collapsed into S2 at the small end while large objects got a mild blur |
| **S3 also reduces contrast and luminance** | Downsampling preserves mean colour to within ΔE 0.25, and "What colour is the X?" is in every question pool. The intervention was removing nothing the question depended on. The state text already promised "or dark" |
| **`eff_res` added as a continuous dose regressor** | Severity is now an ordinal label over a measurable quantity, so dose-response can be fitted rather than binned |
| **Reference groups made explicit** | v3 described self-consistency but had no mechanism. `ref_group` links every derived item to exactly one `clean_ref` item on the untouched image |
| **S0-ctrl carries its parent's artifact** | v3 always pasted an occluder, so an S3 control was "blur on target, paste elsewhere" — two different manipulations being compared |
| **Rungs 1 and 2 answer by generation** | v3's `rung1_from_text` read the option-token argmax, making R1 a duplicate of R3. The R1↔R4 gap is the paper's headline and was not being measured |
| **Paired tests pair on `parent_item_id`** | Pairing on `base_image_id` was arbitrary, since one image yields several main items |
| **Every result row carries its intervention metadata** | Without it the dose-response analysis silently produced a flat line that looked plausible |
| **Consistency scored under strict *and* relaxed matching** | The matching rule is an evaluation choice that moves the headline numbers |
| **Visual QA and an e2e integration test added** | Both S3 problems above were found by running the generator and measuring its output, not by reading the code |
| **Image filenames are `{state}_{condition}_{item_id}.jpg`** | Auxiliary images were named after their parent, so 42 of 139 files could not be traced to their manifest row |

---

## 0a. Design decisions inherited from v3, and why

These were settled in v3 and have not changed. Keeping the reasoning here so it
does not have to be reconstructed from a deleted document.

- **No API frontier model.** Costs money, and the paper never needed one.
- **COCO val2017 only, GQA dropped.** GQA images are ~20 GB; COCO val2017 is
  1 GB and already carries the instance masks, which is all the generator needs.
- **Ground truth replaced by self-consistency** against the clean-image answer.
  Removes the entire annotation cost, and is a stronger design: it measures
  whether the answer was ever grounded in the removed evidence.
- **Forced choice scored from option-token logits in a single forward pass**,
  not generation. Roughly 5× cheaper than generating, and it yields rung 3 and
  rung 4 from the same pass, which is what makes their comparison exactly
  paired. (v4 adds a *separate* generation pass for rungs 1 and 2 — that is
  deployed behaviour and genuinely does need generating.)
- **S4 verified by CIEDE2000 colour distance** between candidate masks, making
  ambiguity machine-verifiable rather than assumed.
- **Model set: Qwen2.5-VL-3B, InternVL3-2B, SmolVLM2-2.2B.** All fit fp16 on one
  T4, and three separate model families are kept so a result is not an artifact
  of one lineage.
- **Pipeline split across four notebooks, two of them CPU-only**, because CPU
  notebooks do not consume the 30 h/week GPU quota.

---

## 0. Kaggle free tier: the constraints that actually bite

Unchanged from v3, and all still true:

- **~30 GPU hours per week.** The pipeline now needs about 6–7, not 4.
- **Sessions get killed.** Every long loop checkpoints and resumes; the runners
  skip items already present in `results/{tag}.jsonl`.
- **T4 is Turing.** No bf16, no FlashAttention-2. `run_inference.load()` forces
  `torch_dtype=torch.float16` and `attn_implementation="sdpa"`. Models that
  default to bf16 produce NaNs rather than an error. Do not "fix" this.
- **Internet must be enabled** in notebook settings, which needs phone
  verification on the account.
- **`/kaggle/working` is wiped between sessions unless you Save Version.** Save
  outputs, attach that notebook's output as a dataset input to the next.
- **CPU notebooks do not touch the GPU quota.**

**Revised budget.** NB2 now runs seven passes, not four — clean references,
treatment answers, cause prompt with hidden states, rung 1, rung 2, abstain,
repair.

| Notebook | Accelerator | Est. runtime | GPU quota |
|---|---|---|---|
| NB1 build dataset + visual QA | CPU | ~45 min | 0 |
| NB2 inference, model A | T4 | ~3 h | 3 h |
| NB3 inference, models B and C | T4 | ~3.5 h | 3.5 h |
| NB4 analysis, probes, figures | CPU | ~45 min | 0 |

~6.5 of 30 hours. Still room for a full re-run. `eval/budget.py` logs the real
number per stage; `print_report()` gives the line for the abstract.

If quota tightens, `full_passes(..., fewshot=False)` drops rung 2 for SmolVLM
cleanly.

---

## 1. The measurement, and why it is defensible

There are no ground-truth answers. For every reference group the model answers
the question on the **untouched** image, three samples at temperature 0.7;
groups whose samples disagree are dropped and the drop rate reported. That
answer becomes the reference. Every degraded condition is then scored by whether
the model reproduces it.

Three quantities, and none of them means anything alone:

- **S0 consistency is the ceiling.** It is not 100%, because decoding drifts on
  identical inputs. Report it as the noise floor of the measure itself.
- **Prior-only consistency is the floor.** The referent is painted to grey;
  agreement here is the rate at which the model was answering from priors and
  never using the evidence.
- **The result is main-condition consistency relative to that floor, as a
  function of dose.**

S5 is excluded throughout: the queried category is absent, so no evidence was
removed and there is nothing to be consistent with.

Two further consequences worth stating, both from v3:

- The **prior-only floor becomes directly interpretable**. Agreement at the
  floor is the rate at which the model was answering from priors all along —
  a quantity ground-truth scoring cannot give you.
- There is **no annotator disagreement to report**, because there are no
  annotators. That removes a whole class of reviewer question.

The cost is that "accuracy" is self-consistency with model-specific
pseudo-ground-truth. This belongs in Section 4 and again in Limitations, with
the S0 ceiling quoted as evidence that the reference is not noise.

**And one thing the reference still owes the reader.** Self-consistency says
nothing about whether the clean-image answers were *correct*. Take a 100-item
blind sample of the clean references, check them by hand, and report the rate
in one sentence — "on a 100-item blind check the clean-image answers were
correct at rate X" — so the pseudo-ground-truth is shown not to be garbage.
This is cheap, it is not automated anywhere in the repo, and without it the
whole measurement rests on an unexamined assumption.

---

## 2. Taxonomy

| ID | State | Repair | Generator |
|---|---|---|---|
| S0 | Answerable | answer | untouched |
| S1 | Out of frame | pan | crop so all instances fall outside |
| S2 | Occluded | move | composite a real object over ≥90% of every instance mask |
| S3 | Sub-resolution | zoom | reduce the referent to a target resolution, blur, and reduce contrast and luminance |
| S4 | Ambiguous reference | ask | ≥2 instances, verified colour-distinct |
| S5 | False premise | correct | category verified absent from the annotations |

Plus two auxiliary conditions on S2/S3 items:

- **S0-ctrl** — the *same artifact*, at the same area, on a region overlapping no
  instance of the queried category. Occluder for an S2 parent, degradation for an
  S3 parent. This is the control for artifact detection.
- **prior-only** — the queried category's regions blanked to grey. This is the
  floor.

**P1 / P2 remain the taxonomy's falsifiable content.** Occlusion destroys signal,
so S2 consistency should sit at the prior-only floor and not vary with dose.
Degradation attenuates signal, so S3 should decline monotonically with severity
and stay above the floor until the referent is unresolvable. If the curves
coincide, report five states. `consistency.p1_p2_verdict()` scores both.

Note S3 is now a **compound** intervention — resolution, blur, contrast and
luminance together. That is what the state text always described ("too small,
blurred or dark to make out"), but it means S3 is not a purely spatial
manipulation and the paper should say so.

---

## 3. Data

**Tier A — COCO `val2017`** (~1 GB, 5,000 images) plus
`annotations_trainval2017` for instance masks. Both free Kaggle datasets.

Questions are attribute questions about a named category, so no answer
annotation is needed:

```
"What colour is the {category}?"
"What is the {category} made of?"
"What is the {category} doing?"        # person, animal categories only
"What is written on the {category}?"   # text-bearing categories only
```

Target 150 items per state, 900 main, plus ~300 S0-ctrl, ~300 prior-only, and
one `clean_ref` per reference group.

One question is drawn per (image, category) and reused across every state from
that image. Good for pairing; it does mean question type is confounded with base
image, which is worth a sentence in the appendix.

**Tier B — VizWiz**, 200 hand-sorted items. VizWiz ships an `answerable` flag
but not the *reason*, which is the label this paper needs, so this cannot be
automated. `data/vizwiz.py` handles loading, stratified sampling (75%
unanswerable, since that is where S1–S5 live) and sheet export; you fill in the
state column. Tier B items carry negative `base_image_id` so they cannot
collide with COCO groups, and no intervention metadata — nothing was done to
these images, the failure was already there. They are used for the ladder and
the probe's cross-domain test, not for dose-response.

---

## 4. Code

```
evid6/
  data/      schema.py      taxonomy constants, Item dataclass
             generate.py    six state generators, controls, driver
             splits.py      StratifiedGroupKFold with a hard leakage assert
             vizwiz.py      Tier B
  eval/      prompts.py     four prompts, letter parsing, few-shot builder
             run_inference.py  scoring, generation, resume, metadata passthrough
             budget.py      GPU-hour accounting
  probe/     ladder.py      the four rungs
             learning_curve.py
             clip_baseline.py
             transfer.py    cross-model and held-out-content generalization
             steer.py       activation steering (gated)
  analysis/  consistency.py self-consistency, prior floor, P1/P2 verdict
             abstain.py     AbsAcc / OverAbs / artifact sensitivity
             relabel.py     blind self-relabel harness
             qa_sheet.py    visual QA contact sheets and triptychs
             threats.py     threats-eliminated appendix table
             stats.py       bootstrap CIs, McNemar, correct pairing
             figures.py     every figure in the paper
  nb/        NB1..NB4       .py is the source of truth, .ipynb is what Kaggle runs
  tests/     smoke_test.py       24 sections, unit level
             test_pipeline_e2e.py  driver through to the P1/P2 verdict
             make_fake_coco.py     synthetic COCO fixture
```

### 4.1 `data/schema.py`

The `Item` dataclass gained three linkage fields and one regressor since v3:

```python
item_id: str
base_image_id: int          # COCO image id. THIS is the split group key.
state: str
condition: str              # main | s0ctrl | prioronly | clean_ref
category: str
question: str
image_path: str
ref_answer: Optional[str] = None
# Linkage
ref_group: Optional[str] = None       # all items sharing one clean reference
parent_item_id: Optional[str] = None  # for s0ctrl/prioronly: the main item
artifact: Optional[str] = None        # occlude | degrade | blank | None
# Intervention metadata — the dose-response regressors
occl_frac: Optional[float] = None
inst_pixels: Optional[int] = None
severity: Optional[int] = None
eff_res: Optional[float] = None       # achieved effective resolution, px
n_candidates: Optional[int] = None
delta_e: Optional[float] = None
```

`ref_group` is what makes self-consistency possible: exactly one item per group
has `condition == "clean_ref"` and points at the untouched image.
`parent_item_id` is what makes paired tests exact.

### 4.2 `data/generate.py`

Unchanged in design from v3: real-object occluders from other images (never
black boxes — that is the reviewer's favourite attack), machine-verifiable
acceptance thresholds, every rejection counted.

**What changed is S3.** v3 used fixed division factors `{1: 6, 2: 12, 3: 24}`.
Two things were wrong, both found by measuring output rather than reading code:

*A fixed factor makes the dose depend on referent size.*

| referent | factor 6 | factor 12 | factor 24 |
|---|---|---|---|
| 63×63 (MIN_AREA) | 10×10 | 5×5 | **2×2 — deletion** |
| 346×346 (MAX_AREA) | 57×57 | 28×28 | 14×14 — mild blur |

Severity 3 was deletion at the small end, which collapses S3 into S2 and makes
P1 and P2 inseparable — the exact failure the taxonomy is supposed to be able to
detect.

*Downsampling preserves colour.* Mean colour inside the mask, before vs after:

| texture | sev 1 | sev 2 | sev 3 |
|---|---|---|---|
| fine detail | ΔE 0.08 | 0.04 | 0.04 |
| text-like | ΔE 0.15 | 0.11 | 0.21 |

ΔE below 1 is imperceptible. Since "What colour is the X?" is in every question
pool, S3 consistency would have sat near the S0 ceiling and read as "P2
challenged" when the real cause was asking the one question blurring preserves.

The calibration that replaced it:

```python
S3_TARGET_RES = {1: 32, 2: 16, 3: 8}    # effective px on the longer side
S3_BLUR       = {1: 1.0, 2: 2.0, 3: 3.5}
S3_CONTRAST   = {1: 0.75, 2: 0.55, 3: 0.35}   # multiplicative, toward mid-grey
S3_LUMA       = {1: 0.85, 2: 0.70, 3: 0.55}   # overall darkening
```

Severity now means the same thing for every object, and `_degrade_region()`
returns the achieved resolution so `eff_res` can be used as a continuous
regressor. It **rounds** rather than truncating: `target/longer` is a float, so
`longer * scale` lands on 7.99999… for many sizes and `int()` floored a 166 px
referent at severity 3 to 7 px — under the 8 px line, which is exactly the S3→S2
collapse this scheme exists to prevent (6 Aug). After the fix:

| texture | sev 1 | sev 2 | sev 3 |
|---|---|---|---|
| fine detail | ΔE 7.8 | 14.9 | 21.2 |
| text-like | ΔE 8.5 | 16.3 | 23.1 |

High-frequency energy now declines monotonically (2.7% → 1.2% → 0.2%) instead of
saturating at severity 1.

`gen_s0ctrl()` takes an `artifact` argument (`"occlude"` or `"degrade"`) and
shares `_degrade_region()` with `gen_S3`, so the control carries the identical
transform rather than an approximation of it.

**Filenames** are `{state}_{condition}_{item_id}.jpg`, keyed on the item's own
id, so any image traces back to its manifest row and the directory sorts into
state/condition groups.

`build()` writes `build_stats.json` with the per-state rejection counts. That
table goes in the appendix and used to die with the Kaggle session.

`question_for(cat, rng)` takes the build's seeded rng and raises without it. It
used to draw from the global `random` module, which made the **question** — the
experimental stimulus itself — irreproducible: two `build(seed=0)` calls
assigned different questions to 38 of 68 items while every `item_id` matched,
because `uid()` does not hash the question (6 Aug). The e2e determinism check
now compares the full row, not just ids.

### 4.3 `data/splits.py`

Unchanged. `StratifiedGroupKFold` stratified on state, grouped on
`base_image_id`, with an assert that fails the build rather than warning.

### 4.4 `eval/prompts.py`

The four prompts are unchanged. Added:

- `parse_letter(text)` — handles `B`, `(C)`, `D.`, `The answer is: E`,
  `option F`. Returns `None` when the model commits to nothing, which is a real
  deployment failure and must not be coerced into a guess. A bare letter must be
  followed by punctuation, a bracket, or end of string: the earlier pattern
  matched the English article, so `"A cat is sitting on the bed"` parsed as
  option A — and A is S0, so prose replies were scored as correct S0
  predictions on the two rungs the headline gap is measured from (6 Aug).
- `build_fewshot_prefix(examples, n=8)` and `balanced_examples()` for rung 2.
  The exemplars are **text-only** — question plus gold letter, no exemplar
  images. Eight extra images per call does not fit on a T4 and most 2–3B VLMs
  are not trained for multi-image ICL. So rung 2 measures label-space and format
  priming, not multimodal in-context learning. Say this plainly; it is a weaker
  claim than full ICL and must not be reported as one.

### 4.5 `eval/run_inference.py`

Three runners, all sharing one contract: they accept an already-loaded
`(proc, model)` pair, they resume from partial output, they accept an
`item_filter`, and they emit intervention metadata.

```python
run(...)                    # forward pass: option-token logits + hidden states
run_generation(...)         # free-form answers (clean refs, treatments, abstain)
run_choice_generation(...)  # forced choice answered by generation — rungs 1 and 2
```

`base_row(item)` copies `category`, `occl_frac`, `inst_pixels`, `severity`,
`eff_res`, `n_candidates`, `delta_e`, `ref_group`, `parent_item_id` and
`artifact` into every result row. **Without this the dose-response analysis
silently degenerates** — every item falls back to a default and the plot looks
plausible. NB4 errors loudly if S3 rows arrive without severity, and
`fig_dose_response` drops metadata-less items rather than plotting them at zero.

`proc`/`model`/`keep_loaded` exist because a runner that loads its own model
while one is already resident puts two fp16 copies of a 3B model on a 15 GB T4.
Load once per model, thread it through every pass.

### 4.6 `probe/ladder.py`

Four rungs on identical items:

- **Rung 1** — `rung1_zeroshot()`, generated text, letter-parsed. Deployed
  behaviour. Unparseable replies count as **incorrect**; the unparsed rate is
  returned alongside and belongs next to the accuracy in the paper. An accuracy
  computed over parseable replies only is a different, flattering number, and is
  reported separately as `accuracy_parsed_only`.
- **Rung 2** — `rung2_fewshot()`, same with the 8-exemplar prefix, on the
  held-out fold only, exemplars drawn from training folds.
- **Rung 3** — `rung3_from_logits()`, option-token argmax from the same forward
  pass that produced the activations. This pairing is what makes R3↔R4 valid.
- **Rung 4** — `nested_probe()`, cross-validated logistic regression on the
  residual stream with **the layer chosen inside the training folds**. Each
  outer fold sweeps the layers on the remaining folds only, picks the winner
  there, and scores it once on the held-out fold. `probe_layer()` /
  `layer_sweep()` / `best_layer()` remain, but only for the sweep figure:
  `max` over ~29 layers on the same folds it scores on reads 19.4% against a
  true 16.7% on pure noise, and R4 feeds the headline (6 Aug).

The **R1↔R4 gap** is the headline. In v3 it was accidentally R3↔R4.
Because R4 is now nested, it is a lower number than a pre-6 Aug NB4 run
reported — regenerate those figures rather than adjusting them.

### 4.7 `analysis/consistency.py`

```python
build_references(clean_results, require_stable=True)  # -> refs, stats
score_consistency(treat_results, refs, relaxed=True)  # -> rows with 'consistent'
prior_floor(scored, state=None)
dose_response(scored, state, dose_field, bins=None)
p1_p2_verdict(scored, flat_tol=0.05)
summarise_both(treat_results, refs, ref_stats)        # strict AND relaxed
```

`agree()` defaults to relaxed matching — "red" agrees with "red and white",
since the model is naming the same attribute. Refusals never agree with
anything, including each other. Because the rule moves the headline numbers,
`summarise_both()` scores everything both ways and reports the max delta; NB4
prints the strict breakdown automatically when the gap exceeds 5 points.

`is_refusal()` matches `REFUSAL_MARKERS` as substrings, so every entry has to be
a phrase that cannot open a real answer. A bare `"no "` was in that list and
made `"no parking sign"`, `"no hat"` and `"No, it is a cat"` read as
abstentions — deflating consistency and inflating the reference drop rate toward
the 13 Aug kill criterion, with `"stop sign"` live in the `TEXTISH` question
pool (6 Aug). Whole-answer non-answers (`"none"`, `"N/A"`, `"nothing"`) are
matched exactly via `REFUSAL_EXACT` instead.

### 4.8 `analysis/abstain.py`

`AbsAcc`, `OverAbs`, `UnderAbs`, and the always-abstain baseline. A model that
abstains on everything scores 5/6 on naive accuracy and is useless, so these
only mean anything together. `artifact_sensitivity()` compares abstention on
S0-ctrl (artifact present, referent visible) against clean S0 — a large gap
means abstention is partly artifact detection, which is a confound the paper has
to name.

### 4.9 `probe/transfer.py`, `learning_curve.py`, `clip_baseline.py`

The shortcut defences, all mandatory:

- **CLIP ViT-B/32 probe** on the same folds. If it matches the VLM probe,
  absolute probe numbers leave the abstract and the paper rests on the R1↔R4 gap.
- **Learning curve.** If the probe saturates by n=25 while zero-shot behaviour
  sits near chance, "accessible but unused" is made fairly.
- **Leave-one-category-out** and **severity extrapolation.** A probe that only
  works within-category has learned the objects, not the evidence state.
- **Cross-model transfer** in a shared PCA space. Different models have different
  widths so a literal direction cannot carry over; this tests for common
  low-dimensional structure and should be labelled as the weaker claim it is.

### 4.9b `probe/steer.py` — gated on 20 Aug

Difference-of-means direction (not-S0 minus S0) at one layer, injected at the
last token through a forward hook. `attach()` tries the common block paths;
print the model once and confirm, because it differs per architecture.

Sweep alpha over roughly `[-4, -2, -1, 0, 1, 2, 4]` and plot `AbsAcc` against
`OverAbs`. **The result that matters is whether abstention rises faster than
over-abstention.** If both rise together, the direction is a generic hedging
knob rather than an evidence signal — which is itself worth one honest
paragraph, not a discarded experiment.

Nothing in NB4 calls this yet; it is gated on E4 landing cleanly by 20 Aug.

### 4.10 `analysis/stats.py`

`boot_ci`, `paired_test` (McNemar), and two pairing helpers.
`pair_main_vs_control()` pairs on `parent_item_id` — pairing on `base_image_id`
is wrong because one image yields several main items, so the match becomes
arbitrary.

Everything in the paper regenerates from `results/` into `summary.json`. Nothing
gets typed into LaTeX by hand.

---

## 5. Notebook orchestration

**NB1 (CPU).** Attach COCO. Build, write `items.jsonl` + images, verify splits,
**render the visual QA sheets**, print summary statistics. Save Version; the
output becomes a dataset.

**NB2 (T4).** Attach NB1's output. Load Qwen2.5-VL-3B **once** and hand it to
every pass: clean references (3 samples, `clean_ref` items only), treatment
answers (greedy, everything except `clean_ref` and S5), cause prompt with hidden
states, rung 1, rung 2, abstain, repair. Every stage wrapped in
`budget.stage()`. Save Version.

**NB3 (T4).** Same via `full_passes()` for InternVL3-2B and SmolVLM2-2.2B.
Hidden states for InternVL; SmolVLM behavioural-only unless quota allows.

**NB4 (CPU).** Attach NB1–NB3. Ladder, layer sweep, learning curve, CLIP
baseline, consistency and the P1/P2 verdict, abstention, probe transfer, paired
statistics, all figures, the threats table, the relabel sheet export, and
`summary.json`.

Two habits that will save the project:

1. **Never let a GPU notebook do anything a CPU notebook could do.** Quota is the
   scarce resource, not time.
2. **Save Version after every successful GPU run**, even a partial one. The
   runners resume from a partial `results.jsonl`.

---

## 6. Validation without paid annotators

Four layers now, in descending strength.

**1. Automatic verifiability, built into the generator.** S2 rejects below 90%
realised coverage. S4 rejects below CIEDE2000 ΔE 12. S5 checks the category is absent from
the annotations. S1 checks the crop excludes the union bbox. S3 asserts no
referent falls below 8 px effective resolution. Every rejection is counted and
the per-state rate goes in the appendix. Stronger than annotation for the states
where the property is geometric.

**2. Visual QA — new in v4, and it earned its place immediately.** Every check
above is a proxy. Coverage fractions and colour distances cannot tell you the
occluder landed absurdly, that an "ambiguous" pair is obviously distinguishable
to a person, or that severity 3 reads as deletion. `qa_sheet.contact_sheets()`
lays out ~300 images with the acceptance numbers in the captions;
`triptychs()` shows reference / intervention / control / floor side by side.
Run this on the pilot, before the full build.

Both S3 problems in §4.2 were found this way. Neither was visible in the code.

**3. Blind self-relabel.** `analysis/relabel.py` samples 100 items with a fixed
seed, exports a shuffled unlabelled sheet and a sealed key, and refuses to let
you pretend: it records the export timestamp and warns if you score it inside
the 48-hour cooling-off period. Reports Cohen's kappa and per-state agreement.
Weaker than inter-annotator agreement, but honest, and it catches a
systematically broken generator.

**4. One free peer, if you can get one.** Two hours from a labmate buys real
inter-annotator κ on 60 items. Ask. It costs a favour and upgrades the paper.

**Automated tests.** `tests/smoke_test.py` covers the CPU path on synthetic data
(24 sections). `tests/test_pipeline_e2e.py` builds a COCO-format fixture, runs
the real `build()` driver against it and pushes the output through the whole
analysis path (22 checks). Two of these are negative controls: a world where P2
is false, which the verdict must report as "challenged", and stripped metadata,
which the threats table must report as FAILED. A suite that only confirms
success cannot catch a scoring function that always says yes. Both run in
seconds without a GPU or the COCO download — run them before spending quota.

**Threats-eliminated table.** `analysis/threats.py` generates the appendix table
from the run's own artifacts, verifying what can be verified (folds with
overlap, rows carrying severity, controls with a parent id) and marking the rest
"not checked in this run" rather than claiming success.

Limitations must say: no paid annotation, validation is primarily automatic and
geometric plus a visual pass and one blind self-relabel, inter-annotator
agreement is on a small subset or absent. Reviewers accept this from a workshop
paper when it is stated. They do not accept it when it is hidden.

---

## 7. Timeline (to 29 Aug)

| Dates | Work | Gate |
|---|---|---|
| 4–5 Aug | Codebase written, reviewed, fixed. Both test suites green. Generator driver validated end to end on a synthetic fixture; S3 recalibrated | **Done** |
| 6 Aug | InternVL3 load test. Pilot build at `n_per_state=10`, scan the QA sheets on real COCO | Model loads; images look right |
| 7–8 Aug | Full generation, rejection stats, clean references, stability filter | ~1,500 items |
| 9–10 Aug | **E0 go/no-go: pilot probe + CLIP baseline** | **Decision 10 Aug** |
| 11–16 Aug | NB2 and NB3 main sweeps, all three models. Table 1, Figure 2 | Minimum paper exists |
| 17–18 Aug | E2 dose-response on `eff_res`, P1/P2 verdict. Tier B hand-sorting | Taxonomy verdict |
| 19–20 Aug | NB4: ladder, learning curve, transfer. Figures 3, 4 | Headline number |
| 21–22 Aug | Steering if gated in. Blind self-relabel. **Freeze 22 Aug** | Freeze |
| 23–25 Aug | Write-up, figures, appendix | Complete draft |
| 26–27 Aug | Two readers, revise, anonymity and format check | Reviewed draft |
| 28 Aug | Polish, checklist | Ready |
| **29 Aug** | **Submit** | Done |

The 48-hour relabel cooling-off has to start by 20 Aug to be scored before
freeze.

---

## 8. Kill criteria

Carried from v3, plus what the implementation added:

- **10 Aug, probe at chance across all layers** → pivot to a
  benchmark-and-evaluation paper, report the null as a section.
- **10 Aug, CLIP baseline ≥ probe** → drop absolute probe numbers from the
  abstract, rebuild on the R1↔R4 gap.
- **Letter tokens do not round-trip on a model** → drop that model rather than
  patching the parser. `letter_ids()` asserts this at load time.
- **Rung 1 unparsed rate > 10%** → report it prominently; do not quietly coerce
  those replies into guesses.
- **Clean-answer stability drop rate > 35%** → the reference is too noisy.
  Switch to a closed-set colour question only, and say so.
- **Self-relabel agreement below 0.6 on a state** → drop that state.
- **17 Aug, all models at chance on `CauseAcc`** → that is the paper, pivot to
  the negative framing.
- **20 Aug, E4 not clean** → no steering.
- **25 Aug, under seven pages of substance** → submit as an extended abstract.
- **New: S3 severity 3 reads as deletion on real photographs** → S3 has collapsed
  into S2 and P1/P2 cannot separate. Raise `S3_TARGET_RES[3]` above 8 px and
  rebuild. Check this in the QA sheets on the pilot, not after the sweep.
- **New: strict and relaxed matching disagree by more than ~10 points** → the
  relaxed number needs defending in the text, not just reporting.

---

## 9. Zero-cost checklist

- [ ] Kaggle account phone-verified, internet enabled in notebook settings
- [ ] COCO val2017 and annotations attached as free Kaggle datasets
- [ ] `python tests/smoke_test.py` and `python tests/test_pipeline_e2e.py` pass
- [ ] InternVL3 loads through `AutoModelForVision2Seq` — **test this first**, it
      is the highest technical risk in the project
- [ ] All models ≤7B, fp16, `attn_implementation="sdpa"`, no bf16 anywhere
- [ ] Pilot build inspected in the QA sheets before the full build
- [ ] Every GPU notebook resumable from partial `results.jsonl`
- [ ] One model load per notebook, threaded through every pass
- [ ] Save Version after every successful GPU run
- [ ] NB1 and NB4 on CPU accelerator, confirmed not billing GPU quota
- [ ] No API keys anywhere in the repo
- [ ] Total GPU hours logged via `budget.print_report()` and quoted in the paper

That last point still matters. A paper that gets a real result on a handful of
hours of free compute is more interesting to a workshop audience than one that
spent 400. Put the number in the abstract.

---

## 10. What the paper must state plainly

Not limitations to bury — design facts a reviewer will find anyway:

1. **"Accuracy" is self-consistency** with model-specific pseudo-ground-truth,
   not correctness. The S0 ceiling is the measure's own noise floor.
2. **S3 is a compound intervention** — resolution, blur, contrast and luminance.
   Purely spatial degradation preserves mean colour, which the questions depend
   on, so the compound was necessary. But it is compound.
3. **Rung 2 uses text-only exemplars.** Label-space and format priming, not
   multimodal ICL.
4. **Rung 1 counts unparseable replies as incorrect** and reports that rate.
5. **Cross-model transfer is in a shared PCA space** — common low-dimensional
   structure, not a shared direction.
6. **Question type is confounded with base image** — one question per image,
   reused across states.
7. **Validation is automatic, visual and intra-annotator.** No paid annotation,
   and inter-annotator agreement is on a small subset or absent.
8. **The taxonomy is the smallest actionable set**, not an exhaustive ontology.
   Each state implies a different repair action; that is the criterion, not
   completeness.
