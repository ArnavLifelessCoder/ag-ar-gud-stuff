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
**HEAD = `3fb9228`, nothing unpushed, working tree clean.**

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

**→ Worth doing when convenient: download the Kaggle `vlm nb5` version 2
output** (`kaggle kernels output aaryanadutta/vlm-nb5 -p ./nb5-v2`). It carries
the machine-readable `summary.json` with all three models and the three-model
`probe_cache.json`. The log covers the numbers; the JSON is what you would
regenerate a table from.

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

### The one pre-registered failure

**Clean-reference stability.** Three samples at T=0.7 agreed on 176/427, 91/427,
175/427 groups — drop rates **58.8 / 78.7 / 59.0%** against a 35% gate. Every
consistency and P1/P2 number is therefore provisional and reported as a
criterion failure. The threats table carries one **FAILED** row. Not fixable by
reanalysis — the reference answers themselves are unstable.

McNemar also finds main consistency indistinguishable from the prior floor for
all three (p = 0.36 / 0.18 / 0.86).

---

## 3. Compute

**9.82 GPU-hours consumed**; **~9.2 h to reproduce** — `smolvlm_cause` ran twice
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

### 5a. NB6 closed-set reference rerun — GPU, ~2.6 h — NOT YET RUN

The only route to a defensible P1/P2. **Make a new notebook** (`nb7 vlm`);
do not reuse `nb6 vlm`.

Attach **`vlm neurips nb1`** only. **GPU.** Internet on. Three cells: the setup
cell → `!pip install -q num2words` → `%run /kaggle/working/evid6/nb/NB6_closed_ref.py`

It rewrites every question to "What colour is the {category}?"
(`make_closed_manifest` — only the question changes; ids, states, conditions,
ref groups and image paths are untouched, verified on the real 1,838-row
manifest: 857 rewritten, rest identical), then reruns **only** `clean` and
`treat` under `_clean_v2` / `_treat_v2`. Ladder/abstain/repair never touch the
reference and are not rerun.

**Qwen runs first as a gate** (~0.33 h). If its drop rate is still >35% the
notebook stops rather than spending the remaining 2.2 h; `FORCE = True`
overrides. Two independent reference designs failing is a finding about the
measure, not a hole.

Compliance is **measured**, not assumed — `is_closed_answer` rejects "dark red"
and "the cat is black" rather than snapping them onto the set.

### 5b. NB4 rerun after NB6 — CPU, ~15 min

Attach: nb1, nb2, nb3, `nb6 vlm` (SmolVLM acts), and a notebook holding
`probe_cache.json` with all three models (nb5 v2). All three probes load from
cache. NB4 prefers `_v2` tags automatically, prints which reference task it
used per model, and records it in `summary.json` as `reference_task`.

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

### 5d. Housekeeping — cheap, not urgent

**Download the nb5 v2 output** for the machine-readable `summary.json` and the
three-model `probe_cache.json`. The numbers themselves are already preserved in
`run_logs/nb5_run2_smolvlm_probe.log`.

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
| `qa_real_coco_pilot/` | real-COCO QA sheets, reviewed and passed — committed |
| `run_logs/nb5_run2_smolvlm_probe.log` | **provenance for every SmolVLM number** — committed |

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

**Do not claim:** a P1/P2 verdict, calibrated abstention, or a repair policy.
Report the reference-stability failure as a finding about free-form reference
tasks — an honest negative, not a gap.

**Also state plainly:** S4 consistency (98.2 / 96.4 / 91.7%) is high because
`gen_S4` leaves the image untouched, so it measures decoding noise like S0 — a
sanity check passing, not a result. Rung 2 exemplars are text-only, so it
measures label-space priming, not multimodal ICL. Intra-annotator agreement is
weaker than inter-annotator and must be described as such.
