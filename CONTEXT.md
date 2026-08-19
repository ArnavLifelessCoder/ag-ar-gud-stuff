# EVID-6 — session handoff

Written 18 Aug 2026. Read this first in a new session. Every number here was
read out of a saved artifact or a run log; the provenance column says which.
Nothing is from memory.

---

## 1. What the project is

**"Do VLMs represent why they cannot see?"** Six evidence states describing why
a visual question cannot be answered, generated from COCO val2017 with each
intervention measured rather than assumed, and a four-rung ladder separating
what a model **does** (rungs 1–3) from what its activations **contain** (rung 4).

| State | Meaning |
|---|---|
| S0 | answerable — referent visible and unambiguous |
| S1 | out of frame |
| S2 | occluded — in frame, blocked |
| S3 | sub-resolution — visible but too small / blurred / dark |
| S4 | ambiguous — >1 object matches |
| S5 | false premise — named thing absent |

Repo: `https://github.com/ArnavLifelessCoder/ag-ar-gud-stuff`
Local: `C:\Users\Arnav Gawade(pro)\Downloads\VLM neurips`
Deadlines: freeze 22 Aug, submit 29 Aug.

---

## 2. Status: the pipeline has run end to end

NB1 → NB2/NB3 → NB4 all complete on real COCO with three models.
**HEAD = `8e91d45`, nothing unpushed, working tree clean.**

### The headline result

| Model | R1 says | R2 | R3 | **R4 encodes** | **gap** |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-VL-3B | 41.0% | 39.1% | 41.0% | **73.0% ±2.7** | **+32.0** |
| InternVL3-2B | 38.6% | 35.8% | 39.1% | **79.2% ±2.2** | **+40.7** |
| SmolVLM2-2.2B | 23.9% | 21.8% | 19.9% | **72.9% ±5.5** | **+49.0** |
| *CLIP ViT-B/32 probe* | | | | *44.3% ±3.3* | |
| *chance* | | | | *16.7%* | |

**The claim: representation and reportability are dissociable.** SmolVLM answers
at chance (19.9%, never emits B/E/F, 84% of predictions are one option) yet its
layer-17 activations separate the six states at 72.9% — indistinguishable from
Qwen's 73.0%. The gap does not track capability.

**Provenance — read this before quoting SmolVLM.**

- Qwen / InternVL / CLIP rows, cross-model transfer, Qwen+InternVL learning
  curves and LOCO: verified against `vlm nb5 output/figures/summary.json`,
  re-checked 18 Aug. Solid.
- **SmolVLM's entire row** (R4 72.9% ±5.5, layers `[22,23,17,21,21]`, bias
  +1.9, sweep max 74.7 @ layer 17, LOCO 69.5% / drop +5.3, curve to 70.8%) and
  the **9.82 h budget**: verified against `run_logs/nb5_run2_smolvlm_probe.log`,
  which is committed to this repo. Fifteen separate strings were checked
  against that file on 18 Aug, including the TABLE 1 row and the activation
  shape `(1838, 25, 2048)`.
- ⚠️ `vlm nb5 output/` is the **earlier** nb5 run: its `summary.json` has
  `smolvlm R4 = None` and budget 9.18 h, and it has no `probe_cache.json`. Do
  not read SmolVLM numbers out of it — use the log, or the nb5 v2 output.

**→ Done 18 Aug 21:14 — the nb5 v2 output is now local at `nb5 v2 result/`.**
Its `figures/summary.json` carries all three models and `figures/probe_cache.json`
has all three keys, so SmolVLM is no longer log-only. Cross-checked against the
log the same evening and it agrees exactly: `smolvlm` nested R4
`0.72897 ±0.05456`, sweep max `0.74747 @ layer 17`, CLIP `0.44270`, budget
`9.819 h`. Quote either source.

### Supporting results (all verified)

- **Selection bias reported, not absorbed** — max-over-layers would read 74.6 /
  79.6 / 74.7; nested selection gives 73.0 / 79.2 / 72.9.
- **Leave-one-category-out ≈ free** — 74.2 / 79.9 / 69.5 vs within-distribution
  74.6 / 79.6 / 74.7. Not an object detector.
- **Cross-model transfer collapses to chance** — within 65.4 / 71.9, cross
  17.6 / 20.8 vs 16.7 chance. Real inside each model, not a shared direction.
- **Learning curves unsaturated at n=500** (Qwen 72.5, InternVL 77.5, SmolVLM
  70.8), so every R4 is a lower bound.
- **No model ever predicts S4** — zero across 2,700 main predictions. S1 nearly
  as dead (1.3 / 0.0 / 0.0).
- **SmolVLM's 86.7% on S3 is an artifact** — it answers D on 753/900 items; a
  constant-D predictor scores 100% on S3 and 16.7% overall.

### The pre-registered failure, now diagnosed (NB7, 19 Aug)

**Clean-reference stability fails, and a second reference design shows why.**

| Reference design | Qwen drop | InternVL | SmolVLM | Gate (≤35%) |
|---|---:|---:|---:|---|
| Free-form, 3 samples @ T=0.7 | **58.8%** | 78.7% | 59.0% | FAIL |
| **Closed-set colour** (NB7) | **41.0%** | not run | not run | **FAIL** |

The closed set **worked as an instruction** — compliance **99.1%**
(1270/1281 sampled answers are exactly one listed colour) — and removed
**17.8 points** of variance. What remains is genuine colour disagreement:
dropped groups split on *brown/white*, *black/brown*, *blue/brown*, and
**zero** groups are dropped over a `gray`/`grey` spelling split (checked
explicitly, so this is not a normalisation artifact). Many look like genuinely
multi-coloured objects.

So the instability was never paraphrase. It is sampling uncertainty about the
answer itself. That diagnosis is a **methods result**, not a gap.

**P1 is not defensible under any rule.** `build_references` now also offers
`require_stable="majority"` (modal answer wins with 2 of 3), which would give
7.0% drop and pass the gate — but it is **post-hoc**, flagged as such in the
docstring and in `stats["pre_registered"]`, and it flips the verdict:

| Rule | S2 gap vs floor | P1 verdict |
|---|---:|---|
| unanimous (pre-registered) | **+5.0%** | "supported" |
| majority (post-hoc) | **+5.6%** | "challenged" |

A 0.6-point difference straddling the 5-point tolerance decides it. Majority
also lowers the S0 ceiling 96.6% → 88.0%, the honest cost of the yield.
**Do not report a P1 verdict.**

**P2 is "challenged" under both rules and under both reference designs.** That
is the one consistency conclusion that survives, and it now has independent
support.

McNemar also finds main consistency indistinguishable from the prior floor for
all three (p = 0.36 / 0.18 / 0.86).

Provenance: `run_logs/nb7_closed_set_gate.log`; raw rows in `nb7 results/`.

---

## 3. Compute

**10.15 GPU-hours consumed** (9.82 + NB7's 0.33); **~9.2 h to reproduce** — `smolvlm_cause` ran twice
(0.616 h behavioural in NB3, 0.635 h with activations in NB5). Pick one for the
paper and footnote the other. NB4 merges every attached budget log and
deduplicates on `(name, seconds)`, which is why both survive.

---

## 4. Kaggle notebooks — names are offset from script names

| Kaggle notebook | Runs | Accel | Output holds |
|---|---|---|---|
| `vlm neurips nb1` | NB1_build | CPU | `items.jsonl`, `evid6/images/`, `build_stats.json`, `qa/` |
| `vlm neurips nb2` | NB2_infer_A | GPU | Qwen 7 passes + `acts/qwen_cause/` |
| `vlm neurips nb3` | NB3_infer_BC | GPU | InternVL + SmolVLM 7 passes each, `acts/internvl_cause/` |
| `vlm neurips nb4` | NB4_analyse | CPU | **`figures/probe_cache.json`** (qwen+internvl) |
| `vlm nb5` | NB4_analyse | CPU | v1: figures, CLIP. **v2: all three models + cache** |
| `nb6 vlm` | **NB5**_smolvlm_acts | GPU | `acts/smolvlm_cause/` ← do not overwrite |

**Traps, all real, all hit at least once:**

- **A notebook cannot attach its own output.** The probe cache lives in
  `vlm neurips nb4`, so the NB4 rerun must happen in a *different* notebook.
- **Re-saving a notebook replaces what attaching it gives you.** Never re-run
  `nb6 vlm` with a different script — its output is the only copy of the
  SmolVLM activations.
- **`/kaggle/working` is wiped when a session ends.** Use **Save & Run All**,
  never interactive, for anything whose output you need. Only a committed run
  produces an attachable output.
- **Runners resume by tag** from `/kaggle/working/results/{tag}.jsonl`. A fresh
  batch session starts empty so it genuinely recomputes — but re-running a tag
  *within* a session processes zero items and reports success.
- **Notebook titles no longer matter** — everything searches the input roots
  (`schema.find_items`, `find_dir`/`find_file` in NB4). NB1 actually mounts at
  `/kaggle/input/notebooks/aaryanadutta/vlm-neurips-nb1/`, two levels deeper
  than any hard-coded guess.
- **Don't attach COCO to NB4** — 164k files make the input search crawl.

---

## 5. What is pending

### 5a. NB6 closed-set rerun — RUN 19 Aug, gate FAILED, stopped early

Ran in Kaggle notebook `nb7 vlm` (0.33 GPU-h of the 2.6 budgeted). Qwen's gate
failed at 41.0%, so InternVL and SmolVLM were **not** run — by design.

**Do not spend the remaining 2.2 GPU-h.** It would buy P2-challenged on two
more models under a reference that still fails its pre-registered gate, and P1
stays unusable either way. The diagnosis (§2) is the deliverable.

If you ever do want it: `FORCE = True` in `NB6_closed_ref.py` runs the other
two regardless, to document that the closed set fails for all three.

### 5b. NB4 rerun — probably NOT needed, and one hazard if you do

The nb5 v2 output already has all three models, CLIP, and every figure. There
is nothing new to compute unless you want the closed-set consistency numbers in
`summary.json`.

⚠️ **If you do rerun it, do not attach the `nb7 vlm` output.** NB4 prefers
`_v2` tags per model, and only **Qwen** has them — so you would get Qwen scored
on the closed-set reference and InternVL/SmolVLM on the free-form one, in the
same table. NB4 prints `reference task per model` and records it in
`summary.json` as `reference_task`, so it is visible, but a mixed-reference
comparison across models is not interpretable. Either attach nb7 and read only
Qwen's consistency block, or leave it off.

If you rerun for any other reason: attach nb1, nb2, nb3, `nb6 vlm` (SmolVLM
acts) and nb5 v2 (probe cache). All three probes load from cache, ~15 min.

### 5c. Blind self-relabel — human, 100 items

**Clock: exported 17 Aug 13:40, 48 h cooling-off → ready 19 Aug ~13:40.**
(As of 18 Aug ~15:20: 25.7 h elapsed, 22.3 h remaining.)

- Sheet: `vlm nb5 output/relabel/relabel_sheet.csv` (100 rows, `label` column blank)
- Key: `relabel_key.json` — **do not open**; excluded from the zip and gitignored
- The CSV's image paths are dead outside Kaggle. NB4 now also writes
  `relabel_sheet.html`, a self-contained file with every image embedded —
  verified blind (no state string, item id or filename in the markup).
- Score with `relabel.score_sheet(dir)` → agreement, Cohen's κ, and drops any
  state below 60% (14 Aug criterion).
- NB4 used to **restart this clock on every run** (the guard checked
  `/kaggle/working`, which is always empty in batch). Fixed in `3fb9228`: it
  now carries an existing sheet+key forward from the inputs.

### 5d. Housekeeping — DONE 18 Aug

The nb5 v2 output is downloaded and extracted to `nb5 v2 result/`; see the
§2 provenance note. Two things to know about that folder:

- It ships `relabel/relabel_key.json`. `.gitignore`'s `relabel/` rule catches
  it, and the folder itself is now ignored by name as well — but **do not
  open that file**, the 48 h blind relabel is still pending (§5c).
- It also duplicates the whole `evid6/` source tree, which is why the folder
  is ignored wholesale: editing the wrong copy is the hazard.

### 5e. Still open, lower priority

- **Tier B hand-sorting**, 200 VizWiz items. `data/vizwiz.py` does loading and
  sheet export; the *reason* label cannot be derived from VizWiz's flag.
- **Steering** — `probe/steer.py` importable, nothing calls it. Gated on E4.
- **`score_one` has no finite-value guard.** Qwen produced 8 NaN activation rows
  (5 survive alignment); NB4 drops and reports them. A rerun would reproduce it.
- The 8 NaN item ids: `1bb4bc7fc30f`, `6224ea5e7870`, `648e2d7bcf2d`,
  `b709f755f6f4`, `be3af9bacf15` (main) plus 3 prior-only.

---

## 6. Local file map

| Path | What |
|---|---|
| `evid6/` | the source; NB1–NB6 scripts in `evid6/nb/` |
| `NB2_NB3_ANALYSIS.md` | **the results document** — every figure with provenance |
| `EVID6_STATUS.md` | history of fixes, §1.13–1.18, open issues in §3.4 |
| `KAGGLE_RUNBOOK.md` | copy-paste cells per notebook, traps, troubleshooting |
| `RUNBOOK.md` | the same run described rather than pasted |
| `evid6_plan_v4.md` | the plan — design, kill criteria, timeline |
| `vlm nb2 results/` | NB2 export (Qwen) — gitignored |
| `results-nb3/` | NB3 export (InternVL + SmolVLM) — gitignored |
| `vlm nb5 output/` | NB4 run **1** — stale, see §2 provenance — gitignored |
| `evid6_nb4_output/` | holds the qwen+internvl `probe_cache.json` — gitignored |
| `nb5 v2 result/` | NB4 run **2** — all three models, the good `summary.json` — gitignored |
| `qa_real_coco_pilot/` | real-COCO QA sheets, reviewed and passed — committed |
| `run_logs/nb5_run2_smolvlm_probe.log` | **provenance for every SmolVLM number** — committed |
| `run_logs/nb7_closed_set_gate.log` | **provenance for the closed-set gate failure** — committed |
| `nb7 results/` | NB7 raw `_clean_v2` / `_treat_v2` rows — gitignored |
| `nb5 v2 result/` | nb5 v2 output: three-model `summary.json` + cache — gitignored |

**Never `git add -A` in this repo.** It sweeps the result folders in; it once
committed `relabel_key.json` to a public repo. Stage explicit paths.
`.gitignore` now blocks the output folders, `relabel/` and `relabel_key.json`.

---

## 7. Environment

- Kaggle: **transformers 5.0.0**, torch 2.10, Python 3.12. Do not pin
  transformers down — 5.x renamed `AutoModelForVision2Seq` →
  `AutoModelForImageTextToText` and `torch_dtype` → `dtype`; `load()` resolves
  both. `CLIPModel.get_image_features` returns an output object in 5.x whose
  `pooler_output` is already projected — `_image_embeds` decides by width.
- InternVL must be the **`-hf`** checkpoint; the plain repo ships a custom
  config the auto-classes cannot map.
- SmolVLM2 needs **`num2words`** at processor construction.
- Local (Windows): both suites pass. `pip install pycocotools scikit-image`.
  Run `python tests/smoke_test.py` (25 sections) and
  `python tests/test_pipeline_e2e.py`.

---

## 8. What to claim, and what not to

**Claim:** a linear probe on mid-layer activations recovers evidence state far
above what the model reports (+32.0 / +40.7 / +49.0 on three architectures),
including on a model at chance behaviourally — so representation and
reportability dissociate. Few-shot does not close the gap. Not explained by a
generic vision encoder (CLIP 44.3%), not object-identity detection (LOCO ≈ free),
and not shared between models (transfer at chance).

**Also claim, as a methods contribution:** two independent reference designs
fail the same stability gate (58.8% → 41.0% at 99.1% instruction compliance),
which localises the cause to sampling uncertainty about the answer rather than
to paraphrase; and the P1 verdict is sensitive to the reference-aggregation
rule (+5.0% vs +5.6% against a 5-point tolerance), which is worth a sentence in
Limitations.

**Do not claim:** a **P1** verdict under any rule — it flips. Nor calibrated
abstention, nor a repair policy. **P2 = challenged** is reportable: it holds
under both aggregation rules and both reference designs.

**Also state plainly:** S4 consistency (98.2 / 96.4 / 91.7%) is high because
`gen_S4` leaves the image untouched, so it measures decoding noise like S0 — a
sanity check passing, not a result. Rung 2 exemplars are text-only, so it
measures label-space priming, not multimodal ICL. Intra-annotator agreement is
weaker than inter-annotator and must be described as such.
