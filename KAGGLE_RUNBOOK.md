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

**Always refreshes the code**, and never touches generated images. An earlier
version of this cell skipped the clone when `/kaggle/working/evid6` already
existed, which meant a pushed fix never arrived and you kept running the old
copy — with a traceback pointing at line numbers that no longer exist.

```python
# --- EVID-6 setup: run FIRST, before anything else. Safe to re-run. ---
import os, sys, shutil, subprocess

REPO = "https://github.com/ArnavLifelessCoder/ag-ar-gud-stuff.git"
EV   = "/kaggle/working/evid6"

shutil.rmtree("/tmp/evid6repo", ignore_errors=True)
subprocess.run(["git", "clone", "-q", REPO, "/tmp/evid6repo"], check=True)

# Refresh code only. images/ is generated output and must survive.
for sub in ["data", "eval", "probe", "analysis", "tests", "nb"]:
    shutil.rmtree(f"{EV}/{sub}", ignore_errors=True)
    shutil.copytree(f"/tmp/evid6repo/evid6/{sub}", f"{EV}/{sub}")
for f in ["__init__.py", "README.md", "requirements.txt"]:
    shutil.copy(f"/tmp/evid6repo/evid6/{f}", f"{EV}/{f}")

for sub in ["data", "eval", "probe", "analysis"]:
    if f"{EV}/{sub}" not in sys.path:
        sys.path.insert(0, f"{EV}/{sub}")

print("commit:", subprocess.run(
    ["git", "-C", "/tmp/evid6repo", "log", "-1", "--format=%h %s"],
    capture_output=True, text=True).stdout.strip())
print(sorted(os.listdir(EV)))
```

Expected output ends with:
`['README.md', '__init__.py', 'analysis', 'data', 'eval', 'nb', 'probe', 'requirements.txt', 'tests']`
(plus `images` once NB1 has built).

> **After re-running Cell A to pick up a fix, restart the kernel.** Python
> caches imported modules, so refreshing the files on disk does not change what
> an already-running kernel has in memory. **Run → Restart & Run All.**

---

## NB1 — build the dataset (CPU, no quota)

**Add Input:** search datasets for a COCO 2017 set that includes
`annotations/instances_val2017.json` and the `val2017` images. The exact
dataset does not matter — `init_coco()` finds whatever you attached. Then paste
the notebook body from `evid6/nb/NB1_build.ipynb`, with **Cell A above it**.

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
!cd /kaggle/working/evid6 && python tests/smoke_test.py 2>&1 | tail -20
```

```python
!cd /kaggle/working/evid6 && python tests/test_pipeline_e2e.py 2>&1 | tail -20
```

`tail -20`, not `tail -5` — a traceback is longer than five lines, so `-5` hides
the failing file and line and shows you only the last frame.

Expect `ALL CHECKS PASSED` and `END-TO-END PIPELINE OK`. **If either fails,
stop here.**

### Then: pilot first

NB1's own build cell says `build(n_per_state=150, seed=0)`. **Edit that number
to 10** — do not paste a new cell, or you get `NameError: name 'build' is not
defined`, because `build` is imported by an earlier cell in the notebook.

If you would rather run it standalone, this cell is self-contained and does the
imports itself:

```python
from generate import init_coco, build
from schema import save_items
from qa_sheet import contact_sheets, triptychs

init_coco()                 # no arguments: finds COCO wherever it is mounted
items = build(n_per_state=10, seed=0)
save_items(items, "/kaggle/working/items.jsonl")

# The QA sheets are the whole point of the pilot — build() alone does not
# make them, it only writes images.
contact_sheets(items, "/kaggle/working/qa", per_state=48, seed=0)
triptychs(items, "/kaggle/working/qa", n=24, seed=0)
print(f"{len(items)} items — now open /kaggle/working/qa/index.html")
```

**What this writes** (all under `/kaggle/working`, all of it session-local until
you Save Version):

| Path | What |
|---|---|
| `items.jsonl` | the manifest |
| `evid6/images/*.jpg` | one JPEG per item |
| `evid6/build_stats.json` | rejection counts, for the appendix |
| `qa/` | contact sheets, triptychs, `index.html` |

Nothing leaves the session until **Save Version**. Closing the tab or letting
the session expire loses all of it — which is fine for a pilot, and is why the
full build ends with a Save.

`init_coco()` with no arguments searches `/kaggle/input` for
`instances_val2017.json` and the matching image directory, and prints both.
Kaggle hosts several COCO 2017 datasets with different layouts —
`<ds>/coco2017/annotations/…`, `<ds>/annotations/…`, images sometimes under
`images/val2017` — and a hard-coded path gives you a `FileNotFoundError` from
inside `pycocotools` that names the path it wanted, not the one you have.

To see what it found before building anything:

```python
from generate import find_coco
root, img_dir = find_coco("/kaggle/input")
print("annotations:", root)
print("images:     ", img_dir)
```

Run to the end. Then **open `/kaggle/working/qa/index.html`** (Output tab → the
`qa` folder) and look at every contact sheet. This is the step no code can do.

Check specifically:

- **S3 severity 3** — degraded but still *there*? If it reads as deletion, S3
  has collapsed into S2 and P1/P2 cannot separate. Raise `S3_TARGET_RES[3]`
  above 8 and rebuild.
- **S4** — would a person genuinely be unsure which object is meant?
- **S2** — does the occluder read as an object, not a grey box?

### Then the real build — clear the pilot first

The pilot's images do **not** all get overwritten by the full build. Once a
state's quota fills, the generator skips it for later images, which shifts the
random stream, so the full run picks different categories from that point on
and writes different filenames. The pilot's leftovers would ship inside your
NB1 output dataset. `build()` warns if it finds any; clear them instead:

```python
import shutil, os
shutil.rmtree("/kaggle/working/evid6/images", ignore_errors=True)
shutil.rmtree("/kaggle/working/qa", ignore_errors=True)
for f in ["/kaggle/working/items.jsonl",
          "/kaggle/working/evid6/build_stats.json"]:
    if os.path.isfile(f):
        os.remove(f)
print("cleared — ready for the full build")
```

Then set `n_per_state=150`, **Run All**, and check:

```python
import json
print(json.load(open("/kaggle/working/evid6/build_stats.json")))
```

`build()` prints a `WARNING: … belong to no manifest row` line if any stale
images survived. If you see it, run the clear cell and rebuild — or pass
`build(..., prune_orphans=True)`.

Then **Save Version** (Quick Save is fine). This publishes `items.jsonl`,
`evid6/images/`, `evid6/build_stats.json` and `qa/`.

---

## NB2 — Qwen2.5-VL-3B (T4, ~3 h)

**Add Input:** Notebook Output → `evid6 nb1 output`.

### Cell A1 — environment report

**Do not pin or downgrade transformers.** Current Kaggle images ship
transformers 5.x, where `AutoModelForVision2Seq` no longer exists — it was
renamed `AutoModelForImageTextToText`. `run_inference.load()` now resolves
whichever name is present and handles the matching `torch_dtype` → `dtype`
rename, so it works on both. Verified against a real 4.41 and a simulated 5.0.

```python
from run_inference import env_report
env_report()
```

Expect something like:

```
  transformers   5.x.y
  torch          2.x
  auto_class     AutoModelForImageTextToText
  cuda           True
  gpu            Tesla T4
```

If `auto_class` prints `AutoModelForVision2Seq` that is fine too — it means an
older image, and the fallback took.

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
Same Cell A and Cell A1 (env report) as NB2.

### Cell E — the InternVL load check (RUN THIS FIRST, alone)

This is the single most likely reason NB3 dies, and it costs ten minutes
against two hours of quota. InternVL historically needs its own `model.chat()`
path with explicit `pixel_values` and dynamic tiling, not the auto-class +
`apply_chat_template` path this code uses.

This goes through `run_inference.load()` deliberately — it exercises the exact
loader NB3 will use, including the auto-class and dtype fallbacks, rather than
a hand-written approximation that might succeed where the real one fails.

```python
from run_inference import load
import torch

mid = "OpenGVLab/InternVL3-2B"
proc, model = load(mid)          # raises here if the auto-class cannot map it
print("1/2 processor + model ok")

msgs = [{"role": "user", "content": [{"type": "image"},
                                     {"type": "text", "text": "hi"}]}]
print(proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))
print("2/2 chat template ok — NB3 will run")
```

If either step raises, InternVL needs its own loader. Free the GPU and skip to
Model C rather than burning the session:

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
| `ImportError: cannot import name 'AutoModelForVision2Seq'` | transformers 5.x renamed it | fixed in code — re-run Cell A, then restart |
| `FileNotFoundError: .../instances_val2017.json` | the attached COCO has a different layout | use `init_coco()` with no arguments; run `find_coco("/kaggle/input")` to see what it found |
| `items.jsonl not found` | NB1 output not attached | Add Input → Notebook Output → `evid6 nb1 output` |
| CUDA OOM on the second pass | a `load()` call inside a runner | pass `proc=proc, model=model` instead |
| `ModuleNotFoundError: schema` | Cell A not run first, or run after imports | re-run Cell A, then Run All |
| `NameError: name 'build' is not defined` | pasted the build line as a new cell | edit NB1's existing build cell, or use the self-contained pilot cell |
| A fix you just pushed has no effect | old code still on disk, or module cached in the kernel | re-run Cell A (it now always refreshes), then **Restart & Run All** |
| Traceback points at a line that does not match the file | same as above — stale copy | re-run Cell A, then restart |
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
