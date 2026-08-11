# EVID-6

Typed evidence states in vision-language models. *Do VLMs represent why they
cannot see?*

Six states describing why a visual question cannot be answered, a generator
that produces each one from COCO with the intervention measured rather than
assumed, and a four-rung ladder that separates what a model **does** from what
its activations **contain**.

Runs end to end on Kaggle's free tier in roughly 4 GPU-hours.

## Layout

```
evid6/
  data/      schema.py      taxonomy constants + the Item dataclass
             generate.py    the six state generators, controls, driver
             splits.py      StratifiedGroupKFold with a hard leakage assert
             vizwiz.py      Tier B: hand-sorted real-world items
  eval/      prompts.py     four prompts, letter parsing, few-shot builder
             run_inference.py  forward-pass scoring, generation, resume logic
             budget.py      GPU-hour accounting
  probe/     ladder.py      the four rungs
             learning_curve.py
             clip_baseline.py
             transfer.py    cross-model and held-out-content generalization
             steer.py       activation steering (gated)
  analysis/  consistency.py self-consistency scoring, P1/P2 verdict
             abstain.py     AbsAcc / OverAbs
             relabel.py     blind self-relabel harness
             qa_sheet.py    visual QA contact sheets + triptychs
             threats.py     threats-eliminated appendix table
             stats.py       bootstrap CIs, McNemar, correct pairing
             figures.py     every figure in the paper
  nb/        NB1..NB4       .py (source of truth) and .ipynb (what Kaggle runs)
  tests/     smoke_test.py  full CPU pipeline on synthetic data, no GPU needed
```

## Running it

Run both test suites first. Neither needs a GPU or the COCO download, and
together they take under a minute. If either fails, do not spend quota.

```bash
pip install pycocotools scikit-image && python tests/smoke_test.py && python tests/test_pipeline_e2e.py
```

`smoke_test.py` exercises the entire CPU path on synthetic activations —
24 sections, two of which are negative controls. `test_pipeline_e2e.py` builds a
synthetic COCO fixture, runs the real `build()` driver against it, and pushes
the result through the whole analysis path to the P1/P2 verdict.

Kaggle images already carry `pycocotools` and `scikit-image`; a local machine
usually does not, and the smoke test skips its generator section without them.
Both suites run on Windows as well as Linux.

| Notebook | Accelerator | What it does |
|---|---|---|
| NB1 | CPU | Builds the dataset. Save Version, attach the output to NB2/NB3. |
| NB2 | T4 | Qwen2.5-VL-3B: clean references, cause prompt + hidden states, rungs 1-2, abstain, repair. |
| NB3 | T4 | InternVL3-2B and SmolVLM2-2.2B, same passes. |
| NB4 | CPU | Ladder, probes, learning curve, CLIP baseline, consistency, abstention, transfer, stats, figures. |

CPU notebooks do not consume the weekly GPU quota. Save Version after every
successful GPU run, even a partial one — the runners resume from a partial
`results.jsonl`.

## The two things most likely to break

**bf16 on a T4.** Turing has no bf16. Models that default to it produce NaNs
rather than an error. `run_inference.load()` forces `float16` + `sdpa`; do not
"fix" it to bf16 or FlashAttention-2.

**S3 calibration.** Severity targets an absolute effective resolution
(32/16/8 px on the longer side), not a division factor. A fixed factor made
the dose depend on referent size: at factor 24 a 63x63 object became 2x2,
which is deletion, so S3 collapsed into S2 at the small end while a 346x346
object got a mild blur. S3 also reduces contrast and luminance, because
downsampling alone preserves mean colour to within CIEDE2000 dE < 0.25 and
"What colour is the X?" is in every question pool — the intervention was
removing nothing the question depended on. `eff_res` is recorded as a
continuous regressor alongside the ordinal severity.

**Metadata passthrough.** Every result row must carry `severity` / `occl_frac`
(see `run_inference.base_row`). Without them the dose-response analysis
silently collapses to a single point instead of failing. NB4 checks for this
and shouts.

## What the numbers mean

There are no ground-truth answers. The reference is the model's own answer on
the untouched image, verified stable across three samples. Everything else is
scored as agreement with that reference:

Every consistency figure is computed under **both** strict and relaxed answer
matching, and both are reported whenever they differ by more than 5 points.
The matching rule is an evaluation choice, not a fact about the model.

- **S0 consistency** is the ceiling — the noise floor of the measure itself.
- **prior-only consistency** is the floor — how often the model reproduces its
  answer with the referent painted out, i.e. how often it was never using the
  evidence.
- The gap between them, as a function of dose, is the result.

**P1** says occlusion destroys signal, so S2 should sit at the floor and stay
flat. **P2** says degradation attenuates it, so S3 should decline smoothly and
stay above the floor. `consistency.p1_p2_verdict()` scores both. If the curves
coincide, that is a finding: report five states.

## Before the full build

Run NB1 at `n_per_state=10` and open `/kaggle/working/qa/index.html`. It lays
out ~300 generated images with the acceptance numbers in the captions, plus
triptychs showing reference / intervention / control / floor side by side.

Every acceptance check in the generator is a proxy. Coverage fractions and
colour distances cannot tell you the occluder landed somewhere absurd, or that
S3 severity 3 reads as deletion rather than degradation — which would collapse
S3 into S2 and make P1 and P2 impossible to separate. Two minutes of looking
beats finding it after a GPU sweep.

## Honest limitations, built in

- "Accuracy" is self-consistency with model-specific pseudo-ground-truth, not
  correctness.
- Validation is automatic and geometric (coverage thresholds, CIEDE2000
  distance, annotation-verified absence) plus one blind self-relabel. There is
  no paid inter-annotator agreement.
- Rung 2 uses text-only exemplars. It measures label-space and format priming,
  not multimodal in-context learning.
- Rung 1 counts unparseable replies as incorrect and reports that rate
  separately. An accuracy over parseable replies only is a different number.
- Cross-model transfer runs in a shared PCA space, so it tests for common
  low-dimensional structure, not a literal shared direction.
