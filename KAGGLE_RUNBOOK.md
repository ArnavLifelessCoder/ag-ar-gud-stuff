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

## The setup cell (first cell of NB2, NB3, NB4 — identical)

Every notebook below starts with this one cell. It clones the repo fresh, so it
always pulls the latest code. It prints the commit it installed.

```python
# --- EVID-6 setup: run FIRST. Safe to re-run. ---
import os, sys, shutil, subprocess
REPO = "https://github.com/ArnavLifelessCoder/ag-ar-gud-stuff.git"
EV = "/kaggle/working/evid6"
shutil.rmtree("/tmp/evid6repo", ignore_errors=True)
subprocess.run(["git","clone","-q",REPO,"/tmp/evid6repo"], check=True)
for sub in ["data","eval","probe","analysis","tests","nb"]:
    shutil.rmtree(f"{EV}/{sub}", ignore_errors=True)
    shutil.copytree(f"/tmp/evid6repo/evid6/{sub}", f"{EV}/{sub}")
for f in ["__init__.py","README.md","requirements.txt"]:
    shutil.copy(f"/tmp/evid6repo/evid6/{f}", f"{EV}/{f}")
for sub in ["data","eval","probe","analysis"]:
    if f"{EV}/{sub}" not in sys.path: sys.path.insert(0, f"{EV}/{sub}")
print(subprocess.run(["git","-C","/tmp/evid6repo","log","-1","--format=%h %s"],
                     capture_output=True, text=True).stdout.strip())
```

Each notebook is a complete, ordered script already in the repo, so after the
setup cell you run the whole thing with one `%run` line. Every notebook
verifies its letter tokens and smoke-tests one item **before** the expensive
sweep, so it fails fast if anything is wrong — you do not need separate check
cells.

---

## NB2 — Qwen2.5-VL-3B (T4, ~3 h)

**Add Input:** Notebook Output -> `evid6 nb1 output`. Accelerator **GPU T4**.

**Cell 1:** the setup cell above.

**Cell 2 (optional, no model load, 2 s):** confirm the environment before you
commit an hour to it.

```python
from run_inference import env_report
env_report()
```

Expect `transformers 5.x`, `auto_class AutoModelForImageTextToText` (or the
older `AutoModelForVision2Seq` — both work), `cuda True`, `gpu Tesla T4`.

**Cell 3:** run the whole notebook.

```python
%run /kaggle/working/evid6/nb/NB2_infer_A.py
```

It loads Qwen once, verifies the six option-letter tokens round-trip
(`letter_ids`), smoke-tests one item with a NaN check, then runs seven passes:
clean references, treatment answers, the cause prompt with hidden-state caching,
rung 1, rung 2, abstain, repair. It prints `[budget]` after each pass and a
`print_report()` total at the end. The manifest is rebased to this session's
image paths automatically in setup.

Budget: seven passes, **~3 h**, not the plan's 1.5. If it errors at `load()` or
the token round-trip, stop — that is the fail-fast working, and no sweep quota
was spent. **Save Version** when it finishes.

---

## NB3 — InternVL3-2B + SmolVLM2-2.2B (T4, ~2 h)

**Add Input:** Notebook Output -> `evid6 nb1 output`. Accelerator **GPU T4**.

**Cell 1:** the setup cell.

**Cell 2 (recommended pre-flight):** InternVL is the single most likely thing to
fail, and you want to know before the sweep. This runs the exact loader NB3
uses.

```python
from run_inference import load
proc, model = load("OpenGVLab/InternVL3-2B")   # raises here if it cannot map
msgs = [{"role":"user","content":[{"type":"image"},{"type":"text","text":"hi"}]}]
print(proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False))
print("InternVL loads — NB3 will run")
import gc, torch; del model, proc; gc.collect(); torch.cuda.empty_cache()
```

If that raises, InternVL needs its own loader — tell me and I will give you a
one-line edit to skip Model B and run only SmolVLM. If it prints the template,
continue.

**Cell 3:** run the whole notebook.

```python
%run /kaggle/working/evid6/nb/NB3_infer_BC.py
```

Same passes as NB2, for InternVL3 then SmolVLM2, one model resident at a time.
**Save Version** when it finishes.

---

## NB4 — analysis (CPU, no quota)

**Add Inputs — all three:** Notebook Outputs -> `evid6 nb1 output`,
`evid6 nb2 output`, `evid6 nb3 output`. Accelerator **CPU**.

**Cell 1:** the setup cell.

**Cell 2:** run the whole notebook.

```python
%run /kaggle/working/evid6/nb/NB4_analyse.py
```

It loads every result file (printing one line each — **watch for `NOT FOUND`**,
which means an input is missing or a notebook title did not match), runs the
four-rung ladder, the nested probe, the CLIP baseline, consistency and the
P1/P2 verdict, abstention, transfer, every figure, `summary.json`, and the
threats table. No quota.

### Three things to read in its output, not skim

**R4 is the nested number.** Each outer fold picks its probe layer using only
its training folds. NB4 also prints what max-over-layers *would* have said, as
the bias avoided — on pure noise that gap is +2.4 points. Quote the nested one.

**Check `layers_chosen`.** If the five outer folds disagree on the best layer,
"the probe reads it at layer k" is not an honest sentence.

**Read the strict-vs-relaxed delta** before writing the abstract. NB4 prints
the strict breakdown automatically when the gap exceeds 5 points.

### After it finishes

Export `relabel_sheet.csv` from the Output tab and **start the 48-hour clock** —
`score_sheet` warns if you score it early, and **do not open
`relabel_key.json`**. Then **Save Version**; `figures/` holds every figure,
`summary.json`, and the threats table in markdown and LaTeX.


---

## Order of operations

1. NB1: setup cell → tests (`python tests/…`) → pilot at 10 → scan QA sheets.
2. NB1: full build at 150 → check `build_stats.json` → **Save Version**.
3. NB3 pre-flight: the InternVL load cell. No sweep quota, saves 2 h if it fails.
4. NB2: setup + `%run NB2_infer_A.py` → **Save Version**.
5. NB3: setup + `%run NB3_infer_BC.py` → **Save Version**.
6. NB4: attach all three outputs → setup + `%run NB4_analyse.py`.
   Export the relabel sheet, start the 48 h clock → **Save Version**.
7. Tier B hand-sorting (200 items) while the clock runs.
8. Steering, if E4 landed by 20 Aug.
9. Score the relabel sheet. Write.

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
