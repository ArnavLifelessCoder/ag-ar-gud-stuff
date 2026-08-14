# EVID-6 status

Last updated 6 Aug 2026. Covers the review, the fixes, the validation run, and
what is still open. §1.13 is the newest work: a full read-through of the
repository on 6 Aug that found five defects both test suites were passing
through.

**The plan is now `evid6_plan_v4.md`.** v3 has been moved to `_to_delete/` — it
embedded full source listings that had diverged from the code, and keeping two
disagreeing versions of `gen_S3` around during the write-up was a liability.
v4 keeps only the excerpts that carry a design decision and states that the
repository is authoritative.

Everything substantive in v3 was carried across and verified line by line
against the original: the v2→v3 design rationale (why no API model, why COCO
val2017 instead of GQA, why option-token logits over generation, the model set),
the prior-only floor argument, the 100-item blind correctness check the
reference still owes the reader, CIEDE2000 for S4, the steering alpha sweep and
its hedging-knob interpretation, and all eight original kill criteria. 34 of 34
checks pass.

~6,400 lines across 30 Python files. Everything compiles. Two test suites pass:
`smoke_test.py` (24 sections, unit level) and `test_pipeline_e2e.py` (integration,
generator driver through to the P1/P2 verdict). No module is stubbed. Both now
run on Windows as well as Kaggle — see §1.13.

---

## 0. Documents

| File | What it is |
|---|---|
| `evid6_plan_v4.md` | **the plan.** Design, budget, timeline, kill criteria |
| `EVID6_STATUS.md` | this file — what is done, left, and at risk |
| `EVID6_readiness_review.md` | the original code review, now mostly history |
| `KAGGLE_RUNBOOK.md` | **copy-paste Kaggle cells**, in order, with the pre-quota checks |
| `RUNBOOK.md` | the same run, described rather than pasted; local setup + what is left |
| `evid6/README.md` | how to run it |
| `_to_delete/evid6_plan_v3_kaggle.md` | superseded; delete when you are ready |

---

## 1. Done

### 1.1 Crash fixes

| Was | File | Fix |
|---|---|---|
| `plt` used one line before its import | `nb/NB1_build.py` | import moved above first use |
| `.total_mem` (attribute is `total_memory`) | `nb/NB2`, `nb/NB3` | corrected; failed in the first cell |
| `f"{clip_acc:.1%:>8}"` — invalid chained format spec | `nb/NB4` | formatted to a variable, then padded |
| `from eval.run_inference import ...` — wrong path | `probe/steer.py` | imports `run_inference` directly |
| `multi_class="multinomial"` — removed in sklearn 1.7 | `ladder.py`, `clip_baseline.py` | argument dropped; it is the default now |

All five were reproduced as real failures before patching, not assumed.

### 1.2 Kaggle environment

**Two models resident on one T4.** NB2 held a live model while `run()` loaded its
own second copy — two Qwen-3B fp16 copies plus activations on a 15 GB card. All
runners now accept `proc`/`model`/`keep_loaded`; NB2/NB3 load each model exactly
once and thread it through every pass, printing `memory_allocated()` so drift is
visible.

**Wasted generation.** Clean-reference generation ran over all 1,500 items at
3 samples each. Runners now take an `item_filter`; references are generated only
for `clean_ref` items.

### 1.3 The metadata bug

`run()` wrote no `severity` or `occl_frac`, but NB4 and `figures.py` read them
with `.get(..., default)`. Every item silently reported severity 2 and occlusion
0, so the entire dose-response analysis plotted as a flat line and looked
plausible.

Fixed via `run_inference.base_row()`. The absence is now loud: NB4 errors
explicitly if S3 rows arrive without severity, and `fig_dose_response` drops
metadata-less items rather than plotting them at zero.

### 1.4 Self-consistency — the metric v3 is built on

Previously existed only as a docstring. `build()` emitted no `clean_ref` items,
`ref_answer` was never populated, and `consistency_rate()` was never called and
expected a field name the runner did not write.

Now: `build()` emits one `clean_ref` item per (image, category) group on the
untouched image, every derived item carries `ref_group`, and
`analysis/consistency.py` does normalisation, refusal handling, reference
construction with the stability filter, prior-only floor, S0 ceiling,
dose-response, and `p1_p2_verdict()`.

Verified against a synthetic world where P1 and P2 hold **and** one where P2 is
false. It reports "challenged" in the second, so it is not rubber-stamping.

### 1.5 Ladder rungs 1 and 2

`rung1_from_text` read the logit argmax, making R1 an exact duplicate of R3. R2
was `entry["R2_fewshot"] = entry["R1_zeroshot"]  # placeholder`. Two of four bars
were copies, and the "R1 vs R4 gap" was really R3 vs R4.

Now `run_choice_generation()` makes the model emit a letter; `parse_letter()`
handles `B`, `(C)`, `D.`, `The answer is: E`, `option F`. Replies committing to
nothing are recorded `pred=None`, counted **wrong**, and the unparsed rate is
reported beside the accuracy. Rung 2 builds an 8-exemplar balanced prefix from
folds 1–4, evaluated on fold 0 only.

### 1.6 Two scientific fixes with teeth

**S0-ctrl artifact mismatch.** `gen_s0ctrl` always pasted an occluder, so an S3
item's control was "blur on target, paste elsewhere" — two different
manipulations compared. It now takes an `artifact` argument and shares the exact
transform function with `gen_S3`.

**McNemar pairing.** Pairing matched a main item to the first control sharing a
`base_image_id`, but one image yields several main items, so the match was
arbitrary. `pair_main_vs_control()` pairs on `parent_item_id`.

### 1.7 Modules built

| Module | What it does |
|---|---|
| `analysis/consistency.py` | self-consistency, prior floor, P1/P2 verdict, strict+relaxed |
| `analysis/abstain.py` | AbsAcc, OverAbs, UnderAbs, artifact-sensitivity check |
| `analysis/relabel.py` | blind sheet, sealed key, cooling-off guard, Cohen's kappa |
| `analysis/qa_sheet.py` | visual QA contact sheets, triptychs, HTML index |
| `analysis/threats.py` | threats-eliminated appendix table, verified from artifacts |
| `probe/transfer.py` | leave-one-category-out, severity extrapolation, cross-model PCA transfer |
| `data/vizwiz.py` | Tier B loader, stratified sampling, hand-sorting sheet |
| `eval/budget.py` | per-stage GPU-hour logging, survives Save Version |
| `figures.py` | +`fig_consistency`, +`fig_abstain` |
| `tests/` | `smoke_test.py`, `test_pipeline_e2e.py`, `make_fake_coco.py` |
| `README.md`, `requirements.txt` | — |

### 1.8 From the independent review

Its risk list matched this document item for item, so treat it as a restatement
rather than corroboration. Four concrete asks were actionable and are done:

- **Strict + relaxed matching, both by default.** `summarise_both()` scores every
  consistency figure under both rules, reports the max delta, and NB4 prints the
  strict breakdown automatically when the gap exceeds 5 points.
- **Visual QA before the build.** `qa_sheet.py` lays out ~300 images with
  acceptance numbers in the captions, plus triptychs (reference / intervention /
  control / floor) and an HTML index with a checklist. Wired into NB1.
- **Threats-eliminated table.** `threats.py` generates it from the run's own
  artifacts; what it cannot verify it marks "not checked in this run" rather than
  claiming success. Emits markdown and LaTeX. A negative-control test feeds it
  stripped metadata and asserts it reports FAILED.
- **Smoke-test report** for supplementary material.

Still writing tasks, not code: the taxonomy-framing sentence, the reviewer
questions, and dialling back emphasis on codebase size.

### 1.9 Validation run — 5 Aug

COCO and HuggingFace are blocked in the sandbox (PyPI only), so instead: a
COCO-format fixture, and the **real `build()` driver** run against it. 434 lines
that had never executed. 139 items, all six states filled, every invariant held —
one `clean_ref` per group, controls matching their parent artifact, S2 coverage
0.948–1.000, S4 ΔE ≥ 19.8, S5 categories absent, no fold leakage, deterministic
across re-runs. Then the full analysis path on those real items: references →
consistency → dose-response → verdict → paired stats → QA sheets → threats table.

**This is what found the S3 problems below.** The unit smoke test could not,
because it tested `gen_S3` in isolation and never asked whether the intervention
removed the information the questions depend on.

### 1.10 S3 recalibration (found by the validation run)

Recorded in full in `evid6_plan_v4.md` §4.2, including the before/after
measurement tables.

**Problem 1 — severity was confounded with referent size.** Fixed division
factors {6, 12, 24} meant the dose depended on how big the object was:

| referent | sev 1 | sev 2 | sev 3 |
|---|---|---|---|
| 63×63 (MIN_AREA) | 10×10 | 5×5 | **2×2 — deletion** |
| 346×346 (MAX_AREA) | 57×57 | 28×28 | 14×14 — mild blur |

Severity 3 was deletion at the small end, collapsing S3 into S2 — precisely the
failure that makes P1 and P2 inseparable.

**Problem 2 — S3 did not remove colour, and colour is what the questions ask.**
Mean-colour drift inside the mask, before vs after:

| texture | sev 1 | sev 2 | sev 3 |
|---|---|---|---|
| fine detail | ΔE 0.08 | 0.04 | 0.04 |
| text-like | ΔE 0.15 | 0.11 | 0.21 |

ΔE below 1 is imperceptible. Downsampling averages pixels, so mean colour
survives by construction, and "What colour is the {category}?" is in every
question pool. S3 consistency would have sat near the S0 ceiling and read as
"P2 challenged" when the real cause was asking the one question blurring
preserves.

**Fixes applied.** Severity now targets an absolute effective resolution
(32/16/8 px on the longer side), so it means the same thing for every object,
and `eff_res` is recorded as a continuous regressor alongside the ordinal label.
S3 also reduces contrast and luminance — which the state text always promised
("too small, blurred **or dark**") and the generator never implemented.

Re-measured after the fix:

| texture | sev 1 | sev 2 | sev 3 |
|---|---|---|---|
| fine detail | ΔE 7.8 | 14.9 | 21.2 |
| text-like | ΔE 8.5 | 16.3 | 23.1 |

High-frequency energy now declines monotonically (2.7% → 1.2% → 0.2%) instead of
saturating at severity 1. Verified on real generated items: sev1→32px,
sev2→16px, sev3→8px regardless of object size, nothing below 8px.

### 1.11 Image naming

Auxiliary images were named after their **parent's** id (`ctrl_{parent}.jpg`,
`po_{parent}.jpg`), so 42 of 139 files could not be traced back to the item they
belonged to. Now every file is `{state}_{condition}_{item_id}.jpg`:

```
S0_main_00719bbb7bb7.jpg
S0_clean_ref_b267ef241fbd.jpg
S0_s0ctrl_05f4718df157.jpg
S2_prioronly_2218002265cd.jpg
```

Self-describing, keyed on the item's own id, and the directory sorts into
state/condition groups. Verified: 139 files, 139 unique, manifest and directory
agree in both directions, 0 untraceable. The e2e test enforces it going forward.

### 1.12 Housekeeping

Rejection counts persisted to `build_stats.json`. Dead branch removed from
`accuracy_by_state`. Chance line appears in the ladder legend. Figures no longer
crash on a bare filename. All four notebooks converted to `.ipynb`.

### 1.13 Code review — 6 Aug

A read-through of every module, with both suites re-run locally. Five defects,
all of which the tests were **passing through** rather than catching. Each fix
ships with the assertion that would have caught it.

**Questions were not reproducible.** `question_for` drew from the global
`random` module instead of the build's seeded `rng`. Two `build(seed=0)` calls
in one process assigned different questions to **38 of 68 items**. The
determinism test missed it because it compared `item_id` only, and `uid()` does
not hash the question — so every id matched while the stimulus moved
underneath. `question_for(cat, rng)` now requires the rng and raises if it is
omitted; the e2e check compares the full row and asserts the question map is
stable.

**`parse_letter` read the article "A" as option A.** The pattern
`^\s*\(?([A-F])\)?\b` matched `"A cat is sitting on the bed"` and returned `A`.
Option A is S0, so every prose reply opening with an article scored as a correct
S0 prediction on rungs 1 and 2 — the two rungs the R1→R4 gap is measured from,
concentrated in exactly the state most likely to draw a prose answer. A bare
letter must now be followed by punctuation, a bracket, or end of string, and the
short-reply fallback matches standalone letters only and refuses to return A at
all (a reply that is *only* "A" is already caught upstream). Seventeen parse
cases in the smoke test, including `"A dog"`, `"off"` and `"too dark"` → None.

**`is_refusal` fired on any answer starting with "no ".** `"no "` sat in
`REFUSAL_MARKERS` and was matched as a substring, so `"no parking sign"`,
`"no hat"` and `"No, it is a cat"` all read as abstentions. Refusals never agree
with anything and are dropped from the reference table, so this both deflated
consistency and inflated the drop rate toward the 13 Aug kill criterion.
`"stop sign"` is in the `TEXTISH` pool, so it was live. The marker list now
holds only phrases that cannot open a real answer; whole-answer non-answers
(`"none"`, `"N/A"`, `"nothing"`) are matched exactly instead.

**R4 was selected by `max` over the layer sweep, on the folds it was scored
on.** On pure noise (N=900, 29 layers, 6 classes) that reads **19.4% against a
true 16.7%** — +2.4 points for free, on the paper's headline gap. `ladder.py`
gains `nested_probe`: each outer fold picks its layer using only the training
folds, then scores that layer once on the held-out fold. NB4 reports it as R4,
prints the max-over-layers number beside it as the bias avoided, flags folds
that disagree on the layer, and omits R4 rather than falling back to the biased
figure. `best_layer` is kept for the sweep figure with a warning in its
docstring, and the CLIP kill criterion now compares against the nested number —
comparing an inflated probe to an honest baseline would let that criterion pass
on bias alone. `summary.json` records both under explicit names.

**S3 could land at 7px effective resolution.** `_degrade_region` truncated with
`int()`, and `target/longer` is a float, so `longer * scale` lands on 7.99999…
for many sizes — a 166px referent at severity 3 came out at 7. Below the 8px
line is where S3 collapses into S2, which is the one failure the absolute-
resolution scheme exists to prevent. Rounds now. This was latent before 6 Aug;
the rng fix above shifted the draw and surfaced it, and the "min 8px" printout
had been rounding it out of sight.

**Windows portability.** ~40 `open()` calls had no `encoding=`, so on a cp1252
locale `smoke_test.py` mis-decoded `run_inference.py` and died at section 4.
All text opens are explicit UTF-8 now; both suites run clean on Windows without
`PYTHONUTF8=1`. Local runs also need `pip install pycocotools scikit-image`,
which were missing — on this machine that upgraded numpy to 2.4.6 and pillow to
12.2.0, and everything passes under those versions.

All four `.ipynb` files were regenerated from their `.py` twins and verified
identical on code cells.

### 1.14 The NB1→NB2 image-path break — 6 Aug

Found while verifying the Kaggle runbook, not by either suite.

NB1 records **absolute** image paths from its own session
(`/kaggle/working/evid6/images/…`). In NB2 and NB3 those images arrive as an
attached dataset under a different root, and `run_inference.build_inputs` opens
`item["image_path"]` verbatim. Every forward pass would have died on the first
item — *after* the model had loaded, so it would have cost a session start and
looked like a model problem. NB4 already remapped paths for the CLIP baseline;
the two inference notebooks had no equivalent.

`run_inference.rebase_items(items_path, search_roots)` now rewrites the manifest
against wherever the images actually are, and NB2/NB3 call it in setup and
reassign `ITEMS_PATH`. It **raises** rather than running partially: a manifest
where some images resolve and some do not would drop items from every
downstream count without saying so.

The e2e test now copies the generated images to a second root, rewrites the
manifest to the stale Kaggle paths, asserts that manifest is genuinely broken
first (so the guard cannot pass vacuously), then asserts `rebase_items` repairs
every path, preserves item order, and raises when nothing resolves.

`KAGGLE_RUNBOOK.md` in the project root has the copy-paste cells, verified by
cloning the pushed repo and replicating Kaggle's `sys.path` layout.

### 1.15 transformers 5.x — the auto-class rename

Hit on the first real Kaggle run, in NB1's test cell of all places.

Current Kaggle images ship **transformers 5.x**, where `AutoModelForVision2Seq`
no longer exists — it was renamed `AutoModelForImageTextToText` around 4.45 and
the old alias is gone in 5. `run_inference` imported the old name at module
level, so the import failed before any model was touched.

Two independent problems, both fixed:

**The loader was pinned to one spelling.** `load()` now resolves whichever auto
class the installed transformers has, preferring the new name, and handles the
matching `torch_dtype` → `dtype` keyword rename with a `TypeError` fallback.
Verified both ways: against the real 4.41 here (resolves `Vision2Seq`) and
against a stub presenting only the 5.x surface (resolves `ImageTextToText`).
`env_report()` prints what actually resolved — call it once before a sweep.

**A CPU-only test had a GPU dependency.** The §1.14 e2e check imported
`rebase_items` from `run_inference`, which drags in torch and transformers at
module level. That is what turned a transformers rename into a failure of the
*dataset* test. `rebase_items` now lives in `data/schema.py` — which its own
docstring already promised was dependency-free — and `run_inference` re-exports
it so NB2/NB3 are unchanged.

Both suites now pass with transformers **entirely absent**, verified by running
them against a stub package that raises on import. That is the property the CPU
suites were supposed to have all along.

**Do not pin transformers on Kaggle.** The code adapts; pinning would just move
the breakage.

### 1.17 S5 was irreproducible across sessions — 6 Aug

Found by asking a much smaller question: does the pilot build leave anything
behind? It does, and chasing the leftover files exposed the cause.

`build()` picked the false-premise category with
`rng.choice(list(ALL_CAT_NAMES - present))`. That iterates a **set of strings**,
whose order depends on `PYTHONHASHSEED` — randomised per process. So
`build(seed=0)` chose different absent categories in every new session, giving
different S5 item ids, different questions, different images. **A sixth of the
benchmark was not reproducible**, and re-running NB1 in a fresh Kaggle session
would not have reproduced the dataset any result was computed on.

Neither suite could catch it. The determinism check builds twice **in one
process**, where the hash seed is fixed by definition. Measured directly:
`PYTHONHASHSEED=0/1/2` gave three disjoint S5 sets; after `sorted()`, four seeds
give byte-identical output.

The e2e test now re-runs `build()` in **subprocesses under two different hash
seeds** and compares the full signature. Verified to fail when the `sorted()` is
reverted — it is a real guard, not a decorative one.

Same class as the `question_for` global-RNG bug in §1.13: an ordering dependency
that a same-process test structurally cannot see.

**Related, and how it was found.** A pilot at `n_per_state=10` followed by the
full build at 150 does not overwrite everything: once a state's quota fills, the
`continue` fires before the severity draw, the rng stream shifts, and later
images pick different categories. Leftover JPEGs then ship inside the NB1 output
dataset and break the "manifest and directory agree both ways" invariant.
`build()` now warns and takes `prune_orphans=True`; the runbook clears the
directory between pilot and full build.

### 1.16 COCO layout autodetection — 6 Aug

`ROOT` was hard-coded to one Kaggle COCO dataset's layout
(`<ds>/coco2017/annotations/…`). Kaggle hosts several and they disagree: some
put `annotations/` at the dataset root, some put images under `images/val2017`.
A mismatch surfaced as a `FileNotFoundError` raised inside `pycocotools`, naming
the path the code wanted rather than the path that exists.

Worse, `init_coco(root=X)` set only where annotations were read from —
**`IMG_DIR` kept its module-level default**. So pointing it at a non-default
layout loaded the annotations successfully and then failed on every image, far
from the cause. The e2e test never caught it because it set `g.IMG_DIR` by hand.

`find_coco(search_root)` now locates `instances_val2017.json` and the matching
image directory by searching, and `init_coco()` with no arguments uses it and
sets both globals together. Verified against three layouts, including one with
images under `images/val2017`, plus the missing-dataset case, which now raises
naming the inputs it did see. The e2e test builds a full dataset from the
awkward layout, so the two globals cannot drift apart again.

---

## 2. Left

### 2.1 Needs you, not code

**Tier B hand-sorting — 200 items.** `vizwiz.py` handles loading, sampling and
sheet export. VizWiz ships an `answerable` flag but not the *reason*, which is
the label this paper needs. No code can produce it.

**The blind self-relabel — 100 items.** The sheet exports from NB4 already. The
48-hour cooling-off is real; `score_sheet` warns if you score it early.

**Look at real images.** The fixture is coloured blobs. It proves the pipeline is
structurally sound and that the interventions behave physically, but only real
COCO will tell you whether an occluder landed absurdly or an S4 pair is genuinely
ambiguous to a person. QA sheets are wired into NB1 for exactly this.

### 2.2 Gated by your own plan

**Steering.** `steer.py` is fixed and importable but nothing calls it — gated on
E4 landing by 20 Aug. About an hour to wire into NB4. The layer path differs per
architecture and needs one `print(model)` against a real checkpoint.

### 2.3 The paper

Freeze 22 Aug, submit 29 Aug.

---

## 3. Open issues and risks

### 3.1 Unverified — check before spending quota

**InternVL3-2B through `AutoProcessor` + `AutoModelForVision2Seq` +
`apply_chat_template`.** Cannot be tested without a GPU, and HuggingFace is
blocked here so it could not be tested at all. InternVL historically requires its
own `model.chat()` path with explicit `pixel_values` and dynamic tiling. **The
most likely reason NB3 dies.** Ten-minute test, two hours of quota at stake.

**No model has ever run.** The inference path (`load`, `letter_ids`,
`build_inputs`, `score_one`, hidden-state capture) is statically checked and
contract-tested but has never touched a real checkpoint.

**Option-letter token ids are resolved from a bare `"A"`.** `letter_ids` encodes
the letter with no preceding context and asserts it round-trips. After a chat
template the assistant turn starts at a line boundary, so the bare token is
probably the right one for Qwen and SmolVLM — but rung 3 is *entirely* the
softmax over those ids, so a mismatch would silently score the wrong tokens
rather than crash. On the pilot, check that the unconstrained full-vocab argmax
at the answer position falls inside `opt_ids`. Two lines, and it validates the
whole of rung 3.

### 3.2 Budget

NB2 is seven passes now, not four. The plan's 1.5 h will not hold — assume ~3 h.
Total likely 6–7 GPU-hours rather than 4, still inside the 30 h week. Check
`print_report()` after the pilot. `full_passes(..., fewshot=False)` drops rung 2
for SmolVLM cleanly if quota tightens.

### 3.3 Judgment calls that move the numbers

**Relaxed answer matching.** Now computed both ways automatically, so this is
handled — but read the delta before writing the abstract.

**S3 contrast/luminance reduction** attenuates colour by ΔE 8–23. That is a
design choice, defensible because the state text specifies "or dark", but it does
mean S3 is now a compound intervention (resolution + contrast + luminance) rather
than purely spatial. Say so in the paper.

**Rung 2 exemplars are text-only.** Eight extra images per call will not fit on a
T4. Rung 2 measures label-space and format priming, not multimodal ICL. Must be
stated plainly, not reported as full ICL.

**Cross-model transfer runs in a shared PCA space.** Tests for common
low-dimensional structure, not a literal shared direction.

**Question type is confounded with base image.** `build()` draws one question per
image and reuses it across every state from that image. Good for pairing; worth a
sentence in the appendix.

**R4 is now the nested number, and it will be lower.** Any figure quoted from a
pre-6 Aug run of NB4 carries roughly 2–3 points of layer-selection inflation and
must be regenerated, not adjusted. The R1→R4 gap is the headline, so this moves
the headline. `summary.json` keeps the old max-over-layers value under
`best_layers_sweep_max_biased` for the record — do not quote it.

**If the folds disagree on the best layer, say so.** NB4 prints
`layers_chosen`. A single "the probe reads it at layer k" sentence is only
honest when the five outer folds actually agreed.

### 3.4 Known and accepted

- `clip_baseline.py` imports torch, so NB4 needs it despite being a CPU notebook.
  Kaggle CPU images ship torch; only matters if you run NB4 locally.
- The S0 ceiling may come in lower than expected — greedy decoding drifts on
  identical inputs. That is a property of the measure, and reporting it *is* the
  honest move.
- `MIN_AREA = 4000` was what made the old severity 3 lethal. It is fine now that
  severity is size-independent, but that is why the old ladder failed.

Not fixed on 6 Aug, deliberately — small, but they are real and none of them is
caught by a test:

- **`p1_p2_verdict` scores missing data as support.** `above = s3_floor is None
  or ...`, so absent prior-only rows make P2 "supported". A falsifiability claim
  should report *unknown* when the floor is missing, not pass by default.
- **`gen_S4` compares `labs[0]` against `labs[1]`** — annotation list order, not
  the two largest instances, despite the docstring saying "top-2". The ΔE ≥ 12
  gate is therefore applied to an arbitrary pair when an image has three.
- **`stats.consistency_rate` duplicates `consistency.agree`** with weaker
  matching (exact string, no normalisation). Only the smoke test calls it. Worth
  deleting before someone wires it into NB4 by mistake.
- **Relaxed matching favours short answers.** `set(short).issubset(set(long_))`
  means any one-token answer agreeing on a single word counts as consistent, and
  degraded images plausibly elicit shorter answers — so the matching rule and
  the dose are not fully independent. `summarise_both` reports the strict number
  beside it, which is the mitigation, but read the delta before writing.

---

## 4. Suggested order

1. Smoke-test InternVL3 loading, and check the letter-token argmax at the same
   time (§3.1). (10 min, prevents a 2 h loss)
2. `python tests/smoke_test.py` and `python tests/test_pipeline_e2e.py`.
   Locally these need `pip install pycocotools scikit-image`.
3. NB1 at `n_per_state=10`, open `qa/index.html`, scan every sheet — especially
   whether S3 severity 3 still reads as *present but unresolvable* on real
   photographs rather than as deletion.
4. Full NB1 build, check `build_stats.json` rejection rates.
5. NB2 on the pilot, read `print_report()`, project the real budget.
6. NB2/NB3 full sweeps.
7. NB4 — consistency, ladder, P1/P2 verdict. Export the relabel sheet, start the
   48 h clock.
8. Tier B hand-sorting while the clock runs.
9. Steering, if E4 landed by 20 Aug.
10. Score the relabel sheet. Write.

---

## 5. Plan-to-code correspondence

`evid6_plan_v4.md` was checked mechanically against the repository rather than
by eye: every function it names exists, the four S3 constant dicts match
exactly, all 17 `Item` fields match the dataclass, the MIN/MAX_AREA figures it
quotes are right, and every module in its layout block is real. Re-run that
check if you edit either side.

---

## 6. Artifacts from the validation run

`qa_validation_run/` in the project folder holds the six contact sheets, the
triptych panel and `index.html` from the 5 Aug run. They are synthetic-fixture
output, kept as a reference for what NB1 will produce on real COCO — useful for
checking the layout is readable before you rely on it.
