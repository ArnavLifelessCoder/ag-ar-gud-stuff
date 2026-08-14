# EVID-6 — Kaggle runbook

Copy-paste cells, in order. Every path and import in here was verified against
the code by cloning the repo and replicating Kaggle's `sys.path` layout.

---

## Before you start: notebook titles matter

The notebooks find each other's outputs by **hard-coded Kaggle paths**, and a
notebook's input path is its title lowercased with hyphens. Get these wrong and
NB4 silently reports `NOT FOUND` for every result file.

Create four notebooks titled **exactly**:

| Notebook | Title it exactly | Accelerator | Internet |
|---|---|---|---|
| NB1 | `evid6 nb1 output` | CPU | **On** |
| NB2 | `evid6 nb2 output` | GPU T4 | **On** |
| NB3 | `evid6 nb3 output` | GPU T4 | **On** |
| NB4 | `evid6 nb4 analyse` | CPU | **On** |

Those titles produce `/kaggle/input/evid6-nb1-output`, `-nb2-`, `-nb3-`, which
is exactly what NB2/NB3/NB4 search for. Internet must be **On** in all four —
NB1 clones the repo, NB2/NB3 download models, NB4 downloads CLIP.

---

## Cell A — the setup cell (paste as the FIRST cell of all four notebooks)

Idempotent: safe to re-run, will not wipe generated images.

```python
# --- EVID-6 setup: run FIRST, before anything else ---
import os, sys, shutil, subprocess

if not os.path.isdir("/kaggle/working/evid6"):
    subprocess.run(
        ["git", "clone", "-q",
         "https://github.com/ArnavLifelessCoder/ag-ar-gud-stuff.git",
         "/tmp/evid6repo"], check=True)
    shutil.copytree("/tmp/evid6repo/evid6", "/kaggle/working/evid6")
    print("evid6 source installed")
else:
    print("evid6 already present")

for sub in ["data", "eval", "probe", "analysis"]:
    sys.path.insert(0, f"/kaggle/working/evid6/{sub}")

print(sorted(os.listdir("/kaggle/working/evid6")))
```

Expected output ends with:
`['README.md', '__init__.py', 'analysis', 'data', 'eval', 'nb', 'probe', 'requirements.txt', 'tests']`

---

## NB1 — build the dataset (CPU, no quota)

**Add Input:** search datasets for `coco-2017-dataset` (the one that mounts at
`/kaggle/input/coco-2017-dataset/coco2017`). Then paste the notebook body from
`evid6/nb/NB1_build.ipynb`, with **Cell A above it**.

### Cell B — dependency check

```python
import importlib
for m in ["pycocotools", "skimage", "PIL", "sklearn", "matplotlib"]:
    try:
        importlib.import_module(m); print(f"  {m:14} ok")
    except ImportError:
        print(f"  {m:14} MISSING")
```

If anything says MISSING:

```python
!pip install -q pycocotools scikit-image
```

### Cell C — run the tests before spending anything

```python
!cd /kaggle/working/evid6 && python tests/smoke_test.py 2>&1 | tail -5
```

```python
!cd /kaggle/working/evid6 && python tests/test_pipeline_e2e.py 2>&1 | tail -5
```

Expect `ALL CHECKS PASSED` and `END-TO-END PIPELINE OK`. **If either fails,
stop here.**

### Then: pilot first

In NB1's build cell, change `150` to `10`:

```python
items = build(n_per_state=10, seed=0)
```

Run to the end. Then **open `/kaggle/working/qa/index.html`** (Output tab → the
`qa` folder) and look at every contact sheet. This is the step no code can do.

Check specifically:

- **S3 severity 3** — degraded but still *there*? If it reads as deletion, S3
  has collapsed into S2 and P1/P2 cannot separate. Raise `S3_TARGET_RES[3]`
  above 8 and rebuild.
- **S4** — would a person genuinely be unsure which object is meant?
- **S2** — does the occluder read as an object, not a grey box?

When the sheets look right, set `n_per_state=150`, **Run All**, then check:

```python
import json
print(json.load(open("/kaggle/working/evid6/build_stats.json")))
```

Then **Save Version** (Quick Save is fine). This publishes `items.jsonl`,
`evid6/images/`, `evid6/build_stats.json` and `qa/`.

---

## NB2 — Qwen2.5-VL-3B (T4, ~3 h)

**Add Input:** Notebook Output → `evid6 nb1 output`.

### Cell A0 — transformers version (paste ABOVE Cell A, run it, then restart)

Kaggle's image often ships transformers older than Qwen2.5-VL needs. This will
fail at `load()` with an unrecognised-model error otherwise.

```python
!pip install -q -U "transformers>=4.49.0" accelerate
```

Now **Run → Restart & Clear Cell Outputs**, then Run All from Cell A. The
restart matters — upgrading a package that is already imported does nothing.

### Cell A1 — verify the upgrade took

```python
import transformers, torch
print("transformers", transformers.__version__)   # must be >= 4.49
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
```

### Cell D — the letter-token check (do this before the sweep)

Rung 3 is *entirely* the softmax over six letter-token ids, resolved by
encoding a bare `"A"` with no context. A mismatch scores the wrong tokens
**silently** instead of crashing. Run this right after the model loads, with
`proc`, `model` and `opt_ids` already defined by NB2's own cells:

```python
from run_inference import build_inputs
from prompts import CAUSE_PROMPT
import json, torch

# ITEMS_PATH here is the REBASED manifest produced in setup — the raw NB1
# manifest holds paths from NB1's session that do not exist in this one.
with open(ITEMS_PATH, encoding="utf-8") as f:
    it = json.loads(f.readline())
assert os.path.isfile(it["image_path"]), "run the rebase_items cell in setup first"

inp = build_inputs(proc, it["image_path"], CAUSE_PROMPT.format(q=it["question"]))
with torch.inference_mode():
    out = model(**inp, use_cache=False)
top = int(out.logits[0, -1].argmax())

print("unconstrained argmax token:", repr(proc.tokenizer.decode([top])))
print("is it one of the six option ids?", top in opt_ids)
print("option ids:", opt_ids)
```

If `False`, the model prefers a token that is not in `opt_ids` (commonly a
space-prefixed `" A"`). Fix `letter_ids` in `evid6/eval/run_inference.py` to
encode the letter *in context* before running the sweep — the numbers are
meaningless otherwise.

### Cell D0 — confirm the image rebase happened

NB2's setup calls `rebase_items()` and reassigns `ITEMS_PATH`. Check it before
the sweep — if it were skipped, every forward pass dies on the first image,
*after* the model has loaded.

```python
print("ITEMS_PATH:", ITEMS_PATH)          # should end in items_local.jsonl
import json
rows = [json.loads(l) for l in open(ITEMS_PATH, encoding="utf-8") if l.strip()]
missing = [r for r in rows if not os.path.isfile(r["image_path"])]
print(f"{len(rows)} items, {len(missing)} unreadable images")
assert not missing, "wrong NB1 dataset attached"
```

`rebase_items` raises on any unresolved image rather than running partially —
a half-run would silently drop items from every downstream count.

### Then run the notebook

NB2 loads the model once and threads it through all seven passes. **Do not add
a `load()` call inside a runner** — two fp16 copies of a 3B model will not fit
on a 15 GB T4.

Every runner is resumable: re-running skips items already in the results file,
so a session timeout costs you the current item, not the pass.

Watch the budget as you go:

```python
from budget import print_report
print_report()
```

Budget reality: NB2 is seven passes, not four. Assume **~3 h**, not the plan's
1.5. If quota tightens, `full_passes(..., fewshot=False)` drops rung 2 cleanly.

**Save Version** when done.

---

## NB3 — InternVL3-2B + SmolVLM2-2.2B (T4, ~2 h)

**Add Input:** Notebook Output → `evid6 nb1 output`.
Same Cell A0 (transformers upgrade + restart) and Cell A as NB2.

### Cell E — the InternVL load check (RUN THIS FIRST, alone)

This is the single most likely reason NB3 dies, and it costs ten minutes
against two hours of quota. InternVL historically needs its own `model.chat()`
path with explicit `pixel_values` and dynamic tiling, not the
`AutoModelForVision2Seq` + `apply_chat_template` path this code uses.

```python
from transformers import AutoProcessor, AutoModelForVision2Seq
import torch

mid = "OpenGVLab/InternVL3-2B"
proc = AutoProcessor.from_pretrained(mid, trust_remote_code=True)
print("1/3 processor ok")

model = AutoModelForVision2Seq.from_pretrained(
    mid, torch_dtype=torch.float16, attn_implementation="sdpa",
    device_map="cuda", trust_remote_code=True).eval()
print("2/3 model ok")

msgs = [{"role": "user", "content": [{"type": "image"},
                                     {"type": "text", "text": "hi"}]}]
print(proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))
print("3/3 chat template ok — NB3 will run")
```

If any of the three raises, InternVL needs its own loader. Free the GPU and
skip to Model C rather than burning the session:

```python
del model, proc
import gc; gc.collect(); torch.cuda.empty_cache()
```

Then run only NB3's SmolVLM (Model C) section. Do the same letter-token check
(Cell D) for each model that does load.

**Save Version** when done.

---

## NB4 — analysis (CPU, no quota)

**Add Inputs — all three:** Notebook Output → `evid6 nb1 output`,
`evid6 nb2 output`, `evid6 nb3 output`.

Cell A, then the NB4 body. The load cell prints one line per result file —
**read it**. Anything reported `NOT FOUND` means an input is missing or a
notebook title didn't match; fix that before interpreting any number.

```python
# after NB4's load cell, confirm what actually arrived
for k, v in results.items():
    print(f"{k:20} {len(v):5} rows")
```

### Three things to read, not skim

**R4 is the nested number.** Each outer fold picks its probe layer using only
its training folds. NB4 also prints what max-over-layers *would* have said, as
the bias avoided — on pure noise that gap is +2.4 points. Quote the nested one.

**Check `layers_chosen`.** If the five outer folds disagree on the best layer,
"the probe reads it at layer k" is not an honest sentence.

**Read the strict-vs-relaxed delta** before writing the abstract. NB4 prints
the strict breakdown automatically when the gap exceeds 5 points.

### Export the relabel sheet and start the 48-hour clock

The sheet exports from NB4. The cooling-off is enforced — `score_sheet` warns
if you score it early. Download `relabel_sheet.csv` from the Output tab and
**do not open `relabel_key.json`.**

Then **Save Version**. `figures/` holds every figure, `summary.json`, and the
threats table in markdown and LaTeX.

---

## Order of operations

1. NB1 Cell C — both test suites. If red, stop.
2. NB3 Cell E — InternVL load check. No quota, saves 2 h.
3. NB1 at `n_per_state=10` → scan every QA sheet.
4. NB1 at `n_per_state=150` → check `build_stats.json` → Save Version.
5. NB2 Cell D — letter-token check.
6. NB2 pilot → `print_report()` → project the real budget.
7. NB2 / NB3 full sweeps → Save Version each.
8. NB4 → ladder, consistency, P1/P2 verdict. Export relabel sheet, start 48 h clock.
9. Tier B hand-sorting (200 items) while the clock runs.
10. Steering, if E4 landed by 20 Aug.
11. Score the relabel sheet. Write.

Paper freeze 22 Aug, submit 29 Aug.

---

## If something breaks

| Symptom | Cause | Fix |
|---|---|---|
| `NOT FOUND` for every result in NB4 | notebook titles don't match the slugs | retitle to `evid6 nb1 output` etc., re-save |
| `KeyError`/unrecognised model at `load()` | transformers < 4.49 | Cell A0, then **restart the kernel** |
| `items.jsonl not found` | NB1 output not attached | Add Input → Notebook Output → `evid6 nb1 output` |
| CUDA OOM on the second pass | a `load()` call inside a runner | pass `proc=proc, model=model` instead |
| `ModuleNotFoundError: schema` | Cell A not run first, or run after imports | re-run Cell A, then Run All |
| `FileNotFoundError` on an image, first item of NB2/NB3 | `rebase_items` skipped | run the setup cell that reassigns `ITEMS_PATH` |
| Image paths unresolved in the CLIP cell | NB1 output attached under a different name | check `NB1_PATH` at the top of NB4 |
| Session died mid-sweep | Kaggle timeout | just re-run the pass — every runner is resumable |

## Known issues still open

Four small ones, deliberately unfixed, none caught by a test. Detail in
`EVID6_STATUS.md` §3.4.

- `p1_p2_verdict` scores *missing* prior-only data as support for P2 rather
  than as unknown
- `gen_S4` compares annotation-list order, not the two largest instances
- `stats.consistency_rate` duplicates `consistency.agree` with weaker matching
- relaxed matching favours short answers, and degraded images plausibly elicit
  shorter answers — `summarise_both` reporting both rules is the mitigation
