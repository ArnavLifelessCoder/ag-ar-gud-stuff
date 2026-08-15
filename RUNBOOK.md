# EVID-6 runbook

Every command needed to take this from a clean machine to the figures, in
order. Written 6 Aug 2026, against the code in this repository.

## Status in one line

**The CPU pipeline is built and verified. The GPU path has never executed.**

| Layer | State |
|---|---|
| Dataset generator, all six states + controls | verified against a COCO-format fixture, 139 items, every invariant held |
| Analysis path — consistency, P1/P2 verdict, ladder, stats, figures, QA sheets, threats table | verified end to end on those items |
| Both test suites | pass, Windows and Linux, no flags |
| Inference path (`load`, `letter_ids`, `build_inputs`, `score_one`, hidden-state capture) | **statically checked and contract-tested, never run against a real checkpoint** |
| Steering (`probe/steer.py`) | importable, fixed, nothing calls it — gated on E4 landing by 20 Aug |

So: ready to *start*, not proven. Step 2 below is the ten-minute check that
decides whether NB3 will run at all, and it is worth doing before anything
else because it costs no quota and can save two hours of it.

---

## 0. Local setup

```bash
git clone https://github.com/ArnavLifelessCoder/ag-ar-gud-stuff.git && cd ag-ar-gud-stuff
```

```bash
pip install -r evid6/requirements.txt
```

Kaggle images already carry almost all of this. On a local machine the two that
are usually missing are `pycocotools` and `scikit-image`; without them the
smoke test silently skips its image-generator section.

```bash
pip install pycocotools scikit-image
```

---

## 1. Run both test suites

Neither needs a GPU or the COCO download. Under a minute together.

```bash
cd evid6 && python tests/smoke_test.py
```

Expect `ALL CHECKS PASSED` and 24 sections. Two are negative controls — section
12 feeds a world where P2 is false and asserts the verdict says "challenged",
section 23 strips metadata and asserts the threats table says FAILED.

```bash
cd evid6 && python tests/test_pipeline_e2e.py
```

Expect `END-TO-END PIPELINE OK`. This builds a synthetic COCO fixture, runs the
real `build()` driver against it, and pushes the result through the whole
analysis path to the P1/P2 verdict.

**If either fails, stop. Do not spend quota.**

---

## 2. Before any quota: two ten-minute checks

### 2a. Does InternVL3-2B load through this code path?

The single most likely reason NB3 dies. The plain `OpenGVLab/InternVL3-2B`
repository exposes a custom `InternVLChatConfig`, which current Transformers
auto-classes cannot map. NB3 uses the HF-native `-hf` checkpoint through the
project loader; test that exact path before spending the sweep quota.

On a T4 notebook:

```python
from run_inference import load
import gc, torch
mid = "OpenGVLab/InternVL3-2B-hf"
proc, model = load(mid)
msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "hi"}]}]
print(proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))
del model, proc; gc.collect(); torch.cuda.empty_cache()
```

If this raises, do not run Model B's sweep. The error is a real loader
incompatibility, not a transient warning.

### 2b. Do the option-letter token ids match what the model emits?

Rung 3 is *entirely* the softmax over `letter_ids()`. Those ids come from
encoding a bare `"A"` with no preceding context. After a chat template the
assistant turn starts at a line boundary, so the bare token is probably right
for Qwen and SmolVLM — but a mismatch would silently score the wrong tokens
rather than crash, which is the worst failure mode available.

With a model loaded and one real item:

```python
from run_inference import letter_ids, build_inputs
from prompts import CAUSE_PROMPT
ids = letter_ids(proc)
inp = build_inputs(proc, "<path to any generated image>", CAUSE_PROMPT.format(q="What colour is the cat?"))
out = model(**inp, use_cache=False)
top = int(out.logits[0, -1].argmax())
print("argmax token:", repr(proc.tokenizer.decode([top])), "in opt_ids:", top in ids)
```

If the unconstrained argmax is not one of the six letter ids, rung 3 is
measuring the wrong thing. Fix `letter_ids` before the sweep, not after.

---

## 3. Kaggle: build the dataset (NB1, CPU)

Attach as inputs:

- `coco-2017-dataset` (the COCO val2017 images + annotations)
- this repository, as a dataset named so that `evid6/` lands at
  `/kaggle/input/evid6-code/evid6` — or edit `EVID6_SRC` in NB1's copy cell

Run NB1 at a pilot size first. Set `n_per_state=10` in the `build()` cell:

```python
items = build(n_per_state=10, seed=0)
```

Then **open `qa/index.html` and actually look at every contact sheet.** This is
the step no code can do for you. The fixture the tests use is coloured blobs;
it proves the pipeline is structurally sound and that the interventions behave
physically, but only real photographs will tell you whether an occluder landed
absurdly, whether an S4 pair is genuinely ambiguous to a person, or — the one
that matters most — whether **S3 at severity 3 reads as *present but
unresolvable* rather than as deletion**. If it reads as deletion, S3 has
collapsed into S2 and P1/P2 cannot separate. Raise `S3_TARGET_RES[3]` above 8px
and rebuild.

When the sheets look right, restore `n_per_state=150`, run the full build, and
check the rejection rates in `build_stats.json`. Then **Save Version** and
attach NB1's output as a dataset input to NB2 and NB3.

---

## 4. Kaggle: inference (NB2, NB3 — T4)

Run NB2 on the pilot first, read `print_report()` from `eval/budget.py`, and
project the real budget before committing to the full sweep.

Budget reality: NB2 is seven passes, not four. The plan's 1.5 h will not hold —
assume ~3 h, and 6–7 GPU-hours total rather than 4. Still inside the 30 h week.
If quota tightens, `full_passes(..., fewshot=False)` drops rung 2 for SmolVLM
cleanly.

Both notebooks load each model exactly once and thread it through every pass —
they print `memory_allocated()` so drift is visible. Do not add a `load()` call
inside a runner; two fp16 copies of a 3B model will not fit on a 15 GB T4.

Every runner is resumable. Re-running skips items already in the results file,
so a session timeout costs you the current item, not the pass.

- NB2 — Qwen2.5-VL-3B: clean references, cause prompt + hidden states, rungs 1–2, abstain, repair
- NB3 — InternVL3-2B and SmolVLM2-2.2B, same passes

---

## 5. Kaggle: analysis (NB4, CPU)

Produces the ladder, the layer sweep, the learning curve, the CLIP baseline,
consistency and the P1/P2 verdict, abstention, transfer, stats, every figure,
`summary.json`, and the threats-eliminated table.

Three things to read rather than skim:

**R4 is the nested number.** Each outer fold picks its probe layer using only
its training folds. NB4 also prints what max-over-layers *would* have said, as
the bias avoided. Quote the nested one. Any R4 figure from a notebook run
before 6 Aug carries ~2–3 points of selection inflation and must be
regenerated, not adjusted.

**Check `layers_chosen`.** If the five outer folds disagree on the best layer,
"the probe reads it at layer k" is not an honest sentence.

**Read the strict-vs-relaxed delta before writing the abstract.** NB4 prints
the strict breakdown automatically when the gap exceeds 5 points.

Then export the blind relabel sheet and **start the 48-hour clock**.

---

## 6. What is left that no code can do

**Tier B hand-sorting — 200 items.** `data/vizwiz.py` handles loading,
stratified sampling and sheet export. VizWiz ships an `answerable` flag but not
the *reason*, which is the label this paper needs. Do this while the relabel
clock runs.

**The blind self-relabel — 100 items.** The sheet exports from NB4. The 48-hour
cooling-off is enforced: `score_sheet` warns if you score it early.

**Look at the real QA sheets.** See step 3.

**Steering.** `probe/steer.py` is fixed and importable but nothing calls it —
gated on E4 landing by 20 Aug. About an hour to wire into NB4. The layer path
differs per architecture and needs one `print(model)` against a real checkpoint.

---

## 7. Known issues still open

Four small ones, deliberately not fixed, none caught by a test. Full detail in
`EVID6_STATUS.md` §3.4.

- `p1_p2_verdict` scores *missing* prior-only data as support for P2, rather
  than as unknown
- `gen_S4` compares annotation-list order, not the two largest instances,
  despite the docstring saying "top-2"
- `stats.consistency_rate` duplicates `consistency.agree` with weaker matching;
  only the smoke test calls it
- relaxed matching favours short answers, and degraded images plausibly elicit
  shorter answers — `summarise_both` reporting both rules is the mitigation, not
  a fix

---

## 8. Order of operations

1. Local: both test suites (step 1)
2. InternVL3 load check + letter-token check (step 2) — no quota
3. NB1 at `n_per_state=10`, scan every QA sheet (step 3)
4. Full NB1 build, check `build_stats.json`
5. NB2 on the pilot, read `print_report()`, project the budget
6. NB2 / NB3 full sweeps
7. NB4 — consistency, ladder, P1/P2 verdict. Export the relabel sheet, start the 48 h clock
8. Tier B hand-sorting while the clock runs
9. Steering, if E4 landed by 20 Aug
10. Score the relabel sheet. Write.

Paper freeze 22 Aug, submit 29 Aug.
