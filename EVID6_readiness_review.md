# EVID-6: readiness review

Reviewed on 5 Aug against `evid6_plan_v3_kaggle.md`. All 21 source files read; syntax-checked the whole tree.

**Verdict: not ready to run.** The skeleton is complete and the design is faithfully translated, but there are four crash-on-first-run bugs, a GPU memory problem that will likely OOM on a T4, and - more seriously - three of the paper's core measurements are not actually wired up. Roughly a day of work to get to green.

---

## A. Will crash on first run

| # | File | Line | Problem |
|---|---|---|---|
| A1 | `nb/NB1_build.py` | 84-85 | `plt.subplots(...)` is called on line 84, `import matplotlib.pyplot as plt` is on line 85. NameError on the occluder-bank cell. |
| A2 | `nb/NB2_infer_A.py` | 27 | `torch.cuda.get_device_properties(0).total_mem` - the attribute is `total_memory`. AttributeError in the very first cell. |
| A3 | `nb/NB3_infer_BC.py` | 24 | Same `total_mem` typo. |
| A4 | `nb/NB4_analyse.py` | 433 | `f"...{clip_acc:.1%:>8}"` is an invalid format spec (you can't chain `.1%` and `>8`). Compiles fine, raises `ValueError` at runtime, so it kills the Table 1 cell after all the expensive work is done. Use `f"{clip_acc:.1%}"` into a variable first, then `{v:>8}`. |
| A5 | `probe/steer.py` | 128 | `from eval.run_inference import score_one` inside `sweep_alphas`. The notebooks put `evid6/eval` on the path directly, so the importable name is `run_inference`, not `eval.run_inference`. Also `eval` shadows the builtin. Will ImportError the moment you use steering. |

## B. Will run but produce wrong or degenerate results

**B1. Intervention metadata never reaches the results file - the dose-response analysis is silently empty.**
`run_inference.run()` writes only `item_id, state, condition, base_image_id, probs, pred`. But `NB4` section 7 does `x.get("severity", 2)` and `figures.py:fig_dose_response` does `r.get("occl_frac", 0)`. Every item therefore reports severity 2 and occlusion fraction 0. The S3 severity curve collapses to a single point and the S2 scatter plots as a vertical line at x=0. This is E2, the P1/P2 verdict, the thing the taxonomy's falsifiable content rests on. Fix: carry `occl_frac / severity / inst_pixels / n_candidates / delta_e` through from the item dict into each result row.

**B2. Rung 1 and rung 2 of the ladder do not exist.**
`ladder.rung1_from_text()` reads `r["pred"]`, which is the option-token argmax - identical to rung 3. `rung2_fewshot()` just calls `rung1_from_text`. `NB4` line 272 makes it explicit: `entry["R2_fewshot"] = entry["R1_zeroshot"]  # placeholder`. So the ladder figure has four bars of which two are copies. The paper's headline is the rung 1 versus rung 4 gap, and rung 1 is currently rung 3 wearing a hat. You need (a) a real generated-text pass on the cause prompt with a letter parser, and (b) the 8-example ICL prompt builder drawing only from training folds. Neither is written.

**B3. The self-consistency metric - the change that v3 is built around - is not implemented end to end.**
- `build()` never emits any `clean_ref` condition items, even though `schema.Item` documents it. There is no untouched-image row for S1–S4 items, so there is nothing to compare against.
- `run_generation` is instead pointed at *all* items with `CLEAN_PROMPT`, which answers the question on the *degraded* image. Useful, but it is the treatment, not the reference.
- `stats.consistency_rate()` exists, is never called from any notebook, and expects `r["answer"]` (singular string) while `run_generation` writes `answers` (a list).
- `ref_answer` on `Item` is never populated, and the stability filter never actually drops unstable items from downstream analysis - it just prints a percentage.

So "self-consistency with the clean-image answer" currently exists as a docstring. The prior-only floor is likewise computed from cause-prompt argmax rather than from answer agreement.

**B4. The abstain pass is generated and then never analysed.** `NB4` loads `*_abstain` into `results` and does nothing with it. There is no AbsAcc / OverAbs computation anywhere except inside `steer.sweep_alphas`, which derives abstention from the cause prompt instead of the abstain prompt.

**B5. `gen_s0ctrl` applies the wrong artifact for S3 items.** It always pastes an occluder patch, so the S3 control is "blur on target, paste elsewhere" rather than "blur on target, blur elsewhere". The plan asks for the same artifact at the same area. A reviewer comparing S3 to its control is comparing two different manipulations.

**B6. The McNemar pairing in NB4 (lines 337-347) is not a valid pairing.** It matches a main item to the first ctrl item sharing a `base_image_id`, but one image can produce several main items across states, so the match is arbitrary. It also compares "main correct at its own state" against "ctrl correct at S0", which are different tasks. Pair on the derived item id instead (`uid(iid, cat, st, "ctrl")` is deterministic from the main item).

## C. Will probably fail on Kaggle for environmental reasons

**C1. Two models resident on the T4 at once.** In NB2, `load()` is called at line 75 and the model stays alive while `run_generation` (line 128) and `run` (line 166) each call `load()` again internally. Line 157-159 does `del model` and then immediately reloads a model that `run()` is about to duplicate. Qwen2.5-VL-3B in fp16 is roughly 6-7 GB, so two copies plus activations on a 15 GB T4 is an OOM waiting to happen. NB3 repeats the pattern for both models. Either make `run`/`run_generation` accept an already-loaded `(proc, model)`, or drop the notebook-level loads after the smoke test.

**C2. InternVL3-2B through `AutoProcessor` + `AutoModelForVision2Seq` + `apply_chat_template` is the highest-risk assumption in the repo.** InternVL historically requires its own `model.chat()` path with explicit `pixel_values` and dynamic tiling; the generic Vision2Seq route works only on recent transformers versions and even then the chat template shape differs. Smoke-test this on day one - it is the single most likely reason NB3 dies.

**C3. `multi_class="multinomial"`** in `ladder.probe_layer` and `clip_baseline.clip_probe` is deprecated in scikit-learn 1.5 and removed in 1.7. Depending on the Kaggle image this is either a wall of FutureWarnings or a TypeError. It is also the default behaviour now, so just delete the argument.

**C4. Runtime budget looks optimistic.** NB2 alone does 1,500 items × 3 samples of clean generation, then 1,500 forward passes with hidden states, then 1,500 abstain generations, then 1,500 repair passes - with four model loads. The plan budgets 1.5 h. Expect closer to 3. Also, clean-reference generation is currently run over every item including S0-ctrl and prior-only, which is pure waste.

**C5. Layer sweep cost.** NB4 probes every layer (37 for a 3B model) × 5 folds × 2048-dim logistic regression on CPU, twice (sweep, then learning curve reloads and re-probes). The 30 min estimate is optimistic; consider sweeping every other layer first.

## D. Planned but not written at all

- **Rung 2 few-shot / ICL prompt construction** (plan §4.6).
- **VizWiz Tier B, 200 items** (plan §3, timeline 17-18 Aug). No loader, no items, no analysis.
- **Blind self-relabel harness** (plan §6.2) - sampling 100 items, shuffling, stripping labels, scoring agreement.
- **Cross-model generalization / A→B probe transfer** (NB4 header claims it, timeline 19-20 Aug). Not in `ladder.py` or NB4.
- **Rejection-rate export.** `build()` prints rejection counts but writes them nowhere, so the appendix number is lost when the session ends.
- **GPU-hour logging** (plan §9 checklist, and the abstract line about 4 GPU-hours).
- **`.ipynb` files.** The plan calls for `NB1_build.ipynb` etc.; what exists is percent-format `.py`. Convert with jupytext before uploading, or Kaggle can't run them.
- **No `requirements.txt`, no README, no tests.**

## E. Smaller things

- `figures.py:fig_ladder` calls `ax.legend()` before adding the chance-level `axhline`, so the chance line never appears in the legend.
- `os.makedirs(os.path.dirname(out_path))` in each figure function raises if `out_path` has no directory component.
- `stats.accuracy_by_state` computes `correct` twice; the first computation is dead code with a wrong comparison.
- `generate.py` uses `from schema import ...` (flat path) while `data/__init__.py` makes it a package - importing as `evid6.data.generate` will fail. Fine given the notebooks' `sys.path` hacks, but fragile.
- `gen_S1` right/bottom crops start at `xs.max()` / `ys.max()` rather than `+1`, leaving a one-pixel sliver of the referent.
- `build()` runs `question_for` once per image and reuses it across all states for that image, which is good for pairing but means question-type is perfectly confounded with base image. Worth a sentence in the appendix.

---

## Suggested order of work

1. Fix A1-A5. Twenty minutes, unblocks everything.
2. Fix C1 (model reloading) and C3 (`multi_class`). These are the difference between NB2 finishing and NB2 dying at item 400.
3. Fix B1 (metadata passthrough). Small change, and without it E2 produces a figure that is quietly meaningless.
4. Smoke-test InternVL3 loading (C2) before committing GPU quota to NB3.
5. Then B3 (clean-reference pipeline) and B2 (real rung 1 and 2). These are the actual paper and they are the largest remaining chunk.
6. Convert the notebooks to `.ipynb` and do one end-to-end pilot at `n_per_state=10` before the full build.

Items in D beyond rung 2 are scope you can drop and declare in Limitations if the calendar gets tight. Items in B cannot be dropped without changing what the paper claims.
