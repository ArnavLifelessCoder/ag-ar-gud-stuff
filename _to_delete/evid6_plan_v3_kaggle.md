# Typed Evidence States in Vision-Language Models
### Plan v3: zero budget, Kaggle free tier, with working code

**Working title:** *Do VLMs represent why they cannot see?*
**Benchmark:** `EVID-6`
**Total cost: 0.** No API calls, no paid compute, no paid annotation, no paid storage.

This supersedes v2. Sections 1, 2, 9, 10, 13, 17 of v2 (claim, positioning, threats, pre-registered outcomes, paper structure, extension path) carry over unchanged and are summarized here only where the zero-cost constraint changes them.

---

## Changelog from v2

| Change | Reason |
|---|---|
| API frontier model **removed** | Costs money. The paper never needed it |
| **GQA dropped, COCO val2017 only** | GQA images are ~20 GB. COCO val2017 is 1 GB and already carries instance masks, which is all the generator needs |
| Ground-truth answers replaced by **self-consistency against the clean-image answer** | Removes the entire annotation cost. Also a stronger design: it measures whether the model's own answer survives evidence removal |
| Human validation replaced by **automatic verifiability checks + a 100-item blind self-relabel** | Zero cost. Weaker than two paid annotators, so it is stated plainly in Limitations |
| Ambiguity (S4) verified by **CIEDE2000 colour distance** between candidate masks | Makes S4 machine-verifiable instead of assumed |
| Forced choice scored by **option-token logits in a single forward pass**, not generation | Cuts inference roughly 5x and hands you ladder rung 3 and rung 4 from the same pass, for free |
| Model set: Qwen2.5-VL-3B, InternVL3-2B, SmolVLM2-2.2B | All fit fp16 on one T4. Three families kept |
| Pipeline split into **4 notebooks, 2 of them CPU-only** | CPU notebooks do not consume the 30 h/week GPU quota |

---

## 0. Kaggle free tier: the constraints that actually bite

Verify current quotas on the Kaggle docs before you start, but plan against these:

- **~30 GPU hours per week.** Resets weekly. Our full pipeline needs about 4.
- **Session length caps out well short of a day, and idle sessions get killed.** Every long loop must checkpoint and resume.
- **T4 is Turing.** No bf16, no FlashAttention-2. Load `torch_dtype=torch.float16` and `attn_implementation="sdpa"`. Models that default to bf16 will silently produce garbage or NaNs if you let them. This is the single most common way a Kaggle VLM notebook fails.
- **Internet must be enabled** in notebook settings to pull from HuggingFace. Requires phone verification on the account. Do this today, not on 20 Aug.
- **`/kaggle/working` is wiped between sessions unless you Save Version.** The persistence trick: save outputs, then attach that notebook's output as a *dataset input* to the next notebook. This is how activations survive from NB2 to NB4 at zero cost.
- **CPU-only notebooks do not touch the GPU quota.** Generation, probing, and analysis all run on CPU. Only inference needs the T4.

**Budget:**

| Notebook | Accelerator | Est. runtime | GPU quota used |
|---|---|---|---|
| NB1 build dataset | CPU | ~40 min | 0 |
| NB2 inference, model A | T4 | ~1.5 h | 1.5 h |
| NB3 inference, models B and C | T4 | ~2 h | 2 h |
| NB4 probes, ladder, CLIP, stats, figures | CPU | ~30 min | 0 |

Roughly 4 GPU-hours out of 30. That leaves room for two full re-runs, which you will need.

---

## 1. What changed scientifically, and why it is an improvement

v2 needed ground-truth answers for the dose-response experiment. Getting those free meant either templated questions with degenerate answers or unpaid annotation. Both are bad.

The fix: **the reference answer is the model's own answer on the clean image.** For every item we first run the untouched image and record the answer, verified stable across three samples. Every degraded condition is then scored by whether the model reproduces that answer.

This is better, not just cheaper:

- It measures exactly the thing the paper cares about, which is whether the answer was ever grounded in the removed evidence.
- The **prior-only floor** (referent region blanked entirely) becomes directly interpretable: agreement at the floor is the rate at which the model was answering from priors all along.
- No annotator disagreement to report.

The cost is that "accuracy" becomes "self-consistency with the clean-image answer," which is model-specific pseudo-ground-truth. Say this in Section 4 of the paper and again in Limitations. Add one sentence reporting that on a 100-item blind check the clean-image answers were correct at rate X, so the reference is not garbage.

---

## 2. Taxonomy (unchanged from v2, restated for the code)

| ID | State | Repair | Generator |
|---|---|---|---|
| S0 | Answerable | answer | untouched |
| S1 | Out of frame | pan | crop so all instances of the category fall outside |
| S2 | Occluded | move | composite an occluder over ≥90% of every instance mask |
| S3 | Sub-resolution | zoom | downsample only instance regions, 3 severity levels |
| S4 | Ambiguous reference | ask | ≥2 instances, verified colour-distinct |
| S5 | False premise | correct | category verified absent from the image |

Plus two auxiliary conditions on S2/S3 items:
- **S0-ctrl**: same artifact, same area, applied to a region overlapping no instance of the queried category.
- **prior-only**: queried category's regions blanked to grey.

**P1 / P2 remain the taxonomy's falsifiable content.** Occlusion destroys signal so consistency should hit the prior-only floor and flatten. Degradation attenuates signal so consistency should decline continuously and stay above the floor until unresolvable. If the curves coincide, report five states.

---

## 3. Data

**Sources, both free Kaggle datasets:**
- COCO `val2017` images (~1 GB, 5,000 images) and `annotations_trainval2017` (instance masks).
- VizWiz-VQA validation subset for Tier B (200 hand-sorted items).

Questions are attribute questions about a named category, so no answer annotation is needed:

```
"What colour is the {category}?"
"What is the {category} made of?"
"What is the {category} doing?"        # person, animal categories only
"What is written on the {category}?"   # text-bearing categories only
```

Target: 150 items per state, 900 total, plus 300 S0-ctrl and 300 prior-only. 1,500 forward passes per model.

---

## 4. Code

Repo layout:

```
evid6/
  data/schema.py  generate.py  splits.py
  eval/run_inference.py  prompts.py
  probe/ladder.py  learning_curve.py  clip_baseline.py  steer.py
  analysis/stats.py  figures.py
  nb/  NB1_build.ipynb  NB2_infer_A.ipynb  NB3_infer_BC.ipynb  NB4_analyse.ipynb
```

### 4.1 `data/schema.py`

```python
from dataclasses import dataclass, asdict, field
from typing import Optional
import json

STATES = ["S0", "S1", "S2", "S3", "S4", "S5"]

STATE_TEXT = {
    "S0": "The image contains enough evidence to answer.",
    "S1": "The object is not inside the frame, though it exists in the scene.",
    "S2": "The object is inside the frame but blocked by something in front of it.",
    "S3": "The object is visible and unblocked, but too small, blurred or dark to make out.",
    "S4": "More than one object matches the description, so the question is ambiguous.",
    "S5": "The object referred to is not present in this scene at all.",
}

REPAIR = {"S0": "answer", "S1": "pan the camera", "S2": "move to another angle",
          "S3": "zoom in", "S4": "ask the questioner", "S5": "correct the premise"}

@dataclass
class Item:
    item_id: str
    base_image_id: int          # COCO image id. THIS is the split group key.
    state: str
    condition: str              # main | s0ctrl | prioronly | clean_ref
    category: str
    question: str
    image_path: str
    ref_answer: Optional[str] = None
    # intervention metadata, the regressors for dose-response
    occl_frac: Optional[float] = None
    inst_pixels: Optional[int] = None
    severity: Optional[int] = None
    n_candidates: Optional[int] = None
    delta_e: Optional[float] = None

    def to_json(self):
        return json.dumps(asdict(self))

def load_items(path):
    with open(path) as f:
        return [Item(**json.loads(l)) for l in f if l.strip()]
```

### 4.2 `data/generate.py`

The core generator. Every function is deterministic given `seed`, and every one records its intervention parameters.

```python
import os, json, random, hashlib
import numpy as np
from PIL import Image, ImageFilter
from pycocotools.coco import COCO
from skimage.color import rgb2lab, deltaE_ciede2000
from schema import Item

ROOT = "/kaggle/input/coco-2017-dataset/coco2017"
IMG_DIR = f"{ROOT}/val2017"
OUT_DIR = "/kaggle/working/evid6/images"
os.makedirs(OUT_DIR, exist_ok=True)

coco = COCO(f"{ROOT}/annotations/instances_val2017.json")
CATS = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
ALL_CAT_NAMES = set(CATS.values())

MIN_AREA = 4000       # px, referent must be big enough that S0 is genuinely answerable
MAX_AREA = 120000     # keep the referent from dominating the frame

PERSONISH = {"person", "dog", "cat", "horse", "bird", "sheep", "cow", "elephant"}
TEXTISH   = {"stop sign", "clock", "book", "laptop", "tv", "cell phone", "bottle"}

def question_for(cat):
    if cat in PERSONISH: pool = ["What colour is the {c}?", "What is the {c} doing?"]
    elif cat in TEXTISH: pool = ["What colour is the {c}?", "What is written on the {c}?"]
    else:                pool = ["What colour is the {c}?", "What is the {c} made of?"]
    return random.choice(pool).format(c=cat)

def uid(*parts):
    return hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:12]

def save(img, name):
    p = f"{OUT_DIR}/{name}.jpg"
    img.convert("RGB").save(p, quality=95)
    return p

def masks_for(img_id, cat_name):
    """All instance masks of one category in one image."""
    cid = [k for k, v in CATS.items() if v == cat_name][0]
    anns = coco.loadAnns(coco.getAnnIds(imgIds=img_id, catIds=[cid], iscrowd=False))
    return [(a, coco.annToMask(a).astype(bool)) for a in anns]
```

**Occluder bank.** Built once, from instances in *other* images, so the occluder is a real object rather than a black box. Black boxes are the reviewer's favourite thing to attack.

```python
def build_occluder_bank(n=400, seed=0):
    rng = random.Random(seed)
    ids = rng.sample(coco.getImgIds(), 1200)
    bank = []
    for iid in ids:
        for a in coco.loadAnns(coco.getAnnIds(imgIds=iid, iscrowd=False)):
            if not (6000 < a["area"] < 90000):
                continue
            m = coco.annToMask(a).astype(bool)
            im = Image.open(f"{IMG_DIR}/{coco.loadImgs(iid)[0]['file_name']}").convert("RGB")
            arr = np.array(im)
            ys, xs = np.where(m)
            y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
            patch = arr[y0:y1, x0:x1]
            alpha = (m[y0:y1, x0:x1] * 255).astype(np.uint8)
            rgba = np.dstack([patch, alpha])
            bank.append((Image.fromarray(rgba, "RGBA"), CATS[a["category_id"]]))
            if len(bank) >= n:
                return bank
    return bank
```

**The six generators.**

```python
def gen_S0(img, *_):
    return img, {}

def gen_S1(img, masks, **kw):
    """Crop so every instance of the category falls outside the frame."""
    W, H = img.size
    union = np.zeros((H, W), bool)
    for _, m in masks: union |= m
    ys, xs = np.where(union)
    # pick the largest axis-aligned crop excluding the union bbox
    cands = [(0, 0, xs.min(), H), (xs.max(), 0, W, H),
             (0, 0, W, ys.min()), (0, ys.max(), W, H)]
    cands = [c for c in cands if (c[2]-c[0]) > 0.35*W and (c[3]-c[1]) > 0.35*H]
    if not cands: return None, None
    box = max(cands, key=lambda c: (c[2]-c[0])*(c[3]-c[1]))
    return img.crop(box), {"occl_frac": 1.0}

def gen_S2(img, masks, bank=None, seed=0, **kw):
    """Composite a real object over >=90% of every instance mask."""
    rng = random.Random(seed)
    out = img.copy()
    covered_num, covered_den = 0, 0
    for _, m in masks:
        ys, xs = np.where(m)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        bw, bh = int((x1-x0)*1.25), int((y1-y0)*1.25)
        occ, _ = rng.choice(bank)
        occ = occ.resize((max(bw,8), max(bh,8)), Image.LANCZOS)
        px = max(0, x0 - (bw-(x1-x0))//2); py = max(0, y0 - (bh-(y1-y0))//2)
        out.paste(occ, (px, py), occ)
        # measure realised coverage
        occ_m = np.zeros(m.shape, bool)
        a = np.array(occ)[:, :, 3] > 128
        h, w = a.shape
        occ_m[py:py+h, px:px+w] |= a[:min(h, m.shape[0]-py), :min(w, m.shape[1]-px)]
        covered_num += (m & occ_m).sum(); covered_den += m.sum()
    frac = covered_num / max(covered_den, 1)
    if frac < 0.90: return None, None          # reject, do not silently ship
    return out, {"occl_frac": float(frac)}

def gen_S3(img, masks, severity=2, **kw):
    """Degrade only the instance regions. severity 1..3."""
    factor = {1: 6, 2: 12, 3: 24}[severity]
    blur   = {1: 1.0, 2: 2.0, 3: 3.5}[severity]
    arr = np.array(img).copy()
    total_px = 0
    for _, m in masks:
        ys, xs = np.where(m)
        y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
        reg = Image.fromarray(arr[y0:y1, x0:x1])
        w, h = reg.size
        small = reg.resize((max(w//factor, 2), max(h//factor, 2)), Image.BILINEAR)
        back = small.resize((w, h), Image.NEAREST).filter(ImageFilter.GaussianBlur(blur))
        patch = np.array(back)
        sub = m[y0:y1, x0:x1]
        arr[y0:y1, x0:x1][sub] = patch[sub]
        total_px += int(m.sum() / (factor ** 2))
    return Image.fromarray(arr), {"severity": severity, "inst_pixels": total_px}

def gen_S4(img, masks, **kw):
    """Ambiguity is only real if the candidates actually differ. Verify in Lab space."""
    if len(masks) < 2: return None, None
    arr = np.array(img).astype(float) / 255.
    labs = []
    for _, m in masks[:3]:
        mean_rgb = arr[m].mean(0).reshape(1, 1, 3)
        labs.append(rgb2lab(mean_rgb))
    d = float(deltaE_ciede2000(labs[0], labs[1])[0, 0])
    if d < 12.0: return None, None              # too similar, not genuinely ambiguous
    return img, {"n_candidates": len(masks), "delta_e": d}

def gen_S5(img, *_):
    return img, {}                               # image untouched, question names an absent class
```

**S0-ctrl and prior-only.**

```python
def gen_s0ctrl(img, masks, bank, seed=0):
    """Same artifact, same area, on a region touching none of the instances."""
    rng = random.Random(seed + 777)
    W, H = img.size
    union = np.zeros((H, W), bool)
    for _, m in masks: union |= m
    area = int(union.sum())
    side = int(np.sqrt(area * 1.25))
    for _ in range(60):
        x = rng.randint(0, max(W - side, 1)); y = rng.randint(0, max(H - side, 1))
        if union[y:y+side, x:x+side].sum() == 0:
            occ, _ = rng.choice(bank)
            out = img.copy()
            out.paste(occ.resize((side, side), Image.LANCZOS), (x, y),
                      occ.resize((side, side), Image.LANCZOS))
            return out, {"occl_frac": 0.0}
    return None, None

def gen_prioronly(img, masks, **kw):
    arr = np.array(img).copy()
    for _, m in masks: arr[m] = 128
    return Image.fromarray(arr), {"occl_frac": 1.0}
```

**Driver.** Note the rejection accounting: every rejected candidate is logged, and the rejection rate per state goes in the appendix. Reviewers read that number.

```python
def build(n_per_state=150, seed=0):
    rng = random.Random(seed)
    bank = build_occluder_bank(seed=seed)
    items, rejects = [], {s: 0 for s in STATES}
    img_ids = coco.getImgIds(); rng.shuffle(img_ids)

    GEN = {"S1": gen_S1, "S2": gen_S2, "S3": gen_S3, "S4": gen_S4}
    need = {s: n_per_state for s in STATES}

    for iid in img_ids:
        if sum(need.values()) == 0: break
        info = coco.loadImgs(iid)[0]
        img = Image.open(f"{IMG_DIR}/{info['file_name']}").convert("RGB")
        anns = coco.loadAnns(coco.getAnnIds(imgIds=iid, iscrowd=False))
        present = {CATS[a["category_id"]] for a in anns}
        by_cat = {}
        for a in anns:
            if MIN_AREA < a["area"] < MAX_AREA:
                by_cat.setdefault(CATS[a["category_id"]], []).append(a)
        if not by_cat: continue
        cat = rng.choice(list(by_cat))
        masks = [(a, coco.annToMask(a).astype(bool)) for a in by_cat[cat]]
        q = question_for(cat)

        for st in ["S0", "S1", "S2", "S3", "S4"]:
            if need[st] == 0: continue
            if st == "S0":
                if len(masks) != 1: continue
                out, meta = img, {}
            else:
                out, meta = GEN[st](img, masks, bank=bank, seed=iid,
                                    severity=rng.choice([1, 2, 3]))
            if out is None:
                rejects[st] += 1; continue
            iid_s = uid(iid, cat, st)
            p = save(out, f"{st}_{iid_s}")
            items.append(Item(iid_s, iid, st, "main", cat, q, p, **meta))
            need[st] -= 1
            if st in ("S2", "S3"):
                c, cm = gen_s0ctrl(img, masks, bank, seed=iid)
                if c is not None:
                    items.append(Item(uid(iid, cat, st, "ctrl"), iid, "S0", "s0ctrl",
                                      cat, q, save(c, f"ctrl_{iid_s}"), **cm))
                po, pm = gen_prioronly(img, masks)
                items.append(Item(uid(iid, cat, st, "po"), iid, st, "prioronly",
                                  cat, q, save(po, f"po_{iid_s}"), **pm))

        if need["S5"] > 0:
            absent = list(ALL_CAT_NAMES - present)
            if absent:
                c5 = rng.choice(absent)
                iid_s = uid(iid, c5, "S5")
                items.append(Item(iid_s, iid, "S5", "main", c5, question_for(c5),
                                  save(img, f"S5_{iid_s}")))
                need["S5"] -= 1

    print("rejection counts:", rejects)
    return items
```

### 4.3 `data/splits.py`

The leakage guard. Every derived item from one COCO image must land in the same fold. Assert it, do not trust it.

```python
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

def make_folds(items, n_splits=5, seed=0):
    y = np.array([it.state for it in items])
    g = np.array([it.base_image_id for it in items])
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(items), -1)
    for k, (_, te) in enumerate(sgkf.split(np.zeros(len(y)), y, g)):
        folds[te] = k
    assert (folds >= 0).all()
    for k in range(n_splits):
        tr_g, te_g = set(g[folds != k]), set(g[folds == k])
        assert not (tr_g & te_g), f"LEAK: fold {k} shares {len(tr_g & te_g)} base images"
    return folds
```

### 4.4 `eval/prompts.py`

```python
from schema import STATE_TEXT, REPAIR

OPTS = "\n".join(f"({chr(65+i)}) {STATE_TEXT[s]}" for i, s in enumerate(STATE_TEXT))

CAUSE_PROMPT = (
    "Look at the image and the question below.\n\n"
    "Question: {q}\n\n"
    "Which single statement best describes the visual evidence available "
    "for answering this question?\n\n" + OPTS +
    "\n\nReply with the letter only."
)

ABSTAIN_PROMPT = (
    "Question about the image: {q}\n\n"
    "If the image gives you enough evidence, answer the question. "
    "If it does not, reply exactly: CANNOT ANSWER."
)

CLEAN_PROMPT = "Question about the image: {q}\nAnswer in at most four words."

REPAIR_OPTS = "\n".join(
    f"({chr(65+i)}) {v}" for i, v in enumerate(dict.fromkeys(REPAIR.values())))
REPAIR_PROMPT = (
    "Question about the image: {q}\n\n"
    "What single action would best let you answer this question reliably?\n\n"
    + REPAIR_OPTS + "\n\nReply with the letter only."
)
```

### 4.5 `eval/run_inference.py`

The important part. One forward pass gives the forced-choice distribution (ladder rung 3), the argmax answer (rung 1), and the cached residual stream (rung 4). Do not run these as separate passes; the pairing is what makes the comparison valid.

```python
import os, json, gc, torch, numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

WORK = "/kaggle/working"
DEV = "cuda"

def load(model_id):
    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_id,
        torch_dtype=torch.float16,        # T4 is Turing: fp16, never bf16
        attn_implementation="sdpa",       # FlashAttention-2 needs Ampere+
        device_map=DEV,
        trust_remote_code=True,
    ).eval()
    return proc, model

def letter_ids(proc, n=6):
    """Resolve option-letter token ids, and verify they round-trip."""
    tok = proc.tokenizer
    ids = []
    for i in range(n):
        L = chr(65 + i)
        cand = tok.encode(L, add_special_tokens=False)
        assert len(cand) >= 1, L
        tid = cand[0]
        assert tok.decode([tid]).strip() == L, f"token mismatch for {L}: {tok.decode([tid])!r}"
        ids.append(tid)
    return ids

def build_inputs(proc, image_path, prompt):
    msgs = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((640, 640))              # cap visual tokens, keeps T4 memory sane
    return proc(text=[text], images=[img], return_tensors="pt").to(DEV)

@torch.inference_mode()
def score_one(proc, model, item, prompt, opt_ids, want_hidden=True):
    inp = build_inputs(proc, item["image_path"], prompt)
    out = model(**inp, output_hidden_states=want_hidden, use_cache=False)
    last = out.logits[0, -1].float()
    probs = torch.softmax(last[opt_ids], -1).cpu().numpy()
    hs = None
    if want_hidden:
        hs = torch.stack([h[0, -1] for h in out.hidden_states]).half().cpu().numpy()
    del out, inp
    return probs, hs

def run(model_id, items_path, tag, prompt_fn, cache_hidden=True, shard=100):
    """Resumable. Re-running after a session timeout skips finished items."""
    proc, model = load(model_id)
    opt_ids = letter_ids(proc)
    res_path = f"{WORK}/results/{tag}.jsonl"
    os.makedirs(f"{WORK}/results", exist_ok=True)
    os.makedirs(f"{WORK}/acts/{tag}", exist_ok=True)

    done = set()
    if os.path.exists(res_path):
        done = {json.loads(l)["item_id"] for l in open(res_path)}
    items = [json.loads(l) for l in open(items_path)]
    todo = [it for it in items if it["item_id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go")

    buf_h, buf_id = [], []
    with open(res_path, "a") as f:
        for n, it in enumerate(todo):
            probs, hs = score_one(proc, model, it,
                                  prompt_fn(q=it["question"]), opt_ids, cache_hidden)
            f.write(json.dumps({
                "item_id": it["item_id"], "state": it["state"],
                "condition": it["condition"], "base_image_id": it["base_image_id"],
                "probs": probs.tolist(), "pred": chr(65 + int(probs.argmax())),
            }) + "\n")
            if n % 20 == 0: f.flush()
            if hs is not None:
                buf_h.append(hs); buf_id.append(it["item_id"])
                if len(buf_h) >= shard:
                    k = len(os.listdir(f"{WORK}/acts/{tag}")) // 2
                    np.save(f"{WORK}/acts/{tag}/h_{k:04d}.npy", np.stack(buf_h))
                    np.save(f"{WORK}/acts/{tag}/i_{k:04d}.npy", np.array(buf_id))
                    buf_h, buf_id = [], []
            if n % 50 == 0:
                gc.collect(); torch.cuda.empty_cache()
    if buf_h:
        k = len(os.listdir(f"{WORK}/acts/{tag}")) // 2
        np.save(f"{WORK}/acts/{tag}/h_{k:04d}.npy", np.stack(buf_h))
        np.save(f"{WORK}/acts/{tag}/i_{k:04d}.npy", np.array(buf_id))
    del model; gc.collect(); torch.cuda.empty_cache()
```

The generation-based conditions (clean reference answers, abstain-or-answer) use `model.generate(max_new_tokens=12, do_sample=False)` and are a separate, smaller pass. Only the clean reference needs three samples, at `temperature=0.7`, to check answer stability; items whose three samples disagree get dropped and the drop rate is reported.

### 4.6 `probe/ladder.py`

All four rungs, on identical items.

```python
import numpy as np, json, glob
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from splits import make_folds

STATES = ["S0","S1","S2","S3","S4","S5"]

def load_acts(tag):
    H = np.concatenate([np.load(p) for p in sorted(glob.glob(f"acts/{tag}/h_*.npy"))])
    I = np.concatenate([np.load(p) for p in sorted(glob.glob(f"acts/{tag}/i_*.npy"))])
    return H, I                                   # H: [N, L+1, D]

def probe_layer(H, y, folds, layer, C=1.0):
    X = H[:, layer, :].astype(np.float32)
    acc = []
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=C, multi_class="multinomial"))
        clf.fit(X[tr], y[tr])
        acc.append((clf.predict(X[te]) == y[te]).mean())
    return float(np.mean(acc)), float(np.std(acc))

def layer_sweep(H, y, folds):
    return [(l,) + probe_layer(H, y, folds, l) for l in range(H.shape[1])]

def rung3_from_logits(results):
    """Option-token argmax, no generation. Same forward pass as the probe."""
    y  = np.array([STATES.index(r["state"]) for r in results])
    yh = np.array([int(np.argmax(r["probs"])) for r in results])
    return float((y == yh).mean())
```

Rung 1 is `rung3_from_logits` on the zero-shot prompt, read from generated text instead of logits when you want the strictly deployed behaviour. Rung 2 is the same with eight in-context examples prepended, drawn from the training folds only.

### 4.7 `probe/learning_curve.py`

The plot that answers "your gap is just supervision".

```python
def learning_curve(H, y, folds, layer, ns=(10,25,50,100,250,500,1000), seed=0):
    rng = np.random.default_rng(seed)
    X = H[:, layer, :].astype(np.float32)
    curve = []
    for n in ns:
        accs = []
        for k in np.unique(folds):
            tr = np.where(folds != k)[0]; te = folds == k
            if n > len(tr): continue
            sub = rng.choice(tr, n, replace=False)
            if len(np.unique(y[sub])) < len(STATES): continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            clf.fit(X[sub], y[sub])
            accs.append((clf.predict(X[te]) == y[te]).mean())
        if accs: curve.append((n, float(np.mean(accs)), float(np.std(accs))))
    return curve
```

If this saturates by n = 25 while zero-shot behaviour sits near chance, the "accessible but unused" claim is made, and made fairly.

### 4.8 `probe/clip_baseline.py`

Mandatory, and it runs in week one.

```python
import torch, numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

def clip_features(paths, bs=32):
    m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval()
    p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    feats = []
    with torch.inference_mode():
        for i in range(0, len(paths), bs):
            ims = [Image.open(q).convert("RGB") for q in paths[i:i+bs]]
            f = m.get_image_features(**p(images=ims, return_tensors="pt"))
            feats.append(torch.nn.functional.normalize(f, dim=-1).numpy())
    return np.concatenate(feats)
```

Then the same `probe_layer` machinery on CLIP features. If this matches the VLM probe, absolute probe numbers leave the abstract and the paper rests on the rung 1 versus rung 4 gap.

### 4.9 `probe/steer.py`

Gated on 20 Aug. Half a day if E4 landed.

```python
import torch

def insufficiency_direction(H, y, layer):
    """Difference of means: not-S0 minus S0, at one layer."""
    X = H[:, layer, :].astype(np.float32)
    v = X[y != 0].mean(0) - X[y == 0].mean(0)
    return torch.tensor(v / np.linalg.norm(v), dtype=torch.float16)

def attach(model, layer, vec, alpha):
    # print(model) once and confirm the path. It differs per architecture.
    block = model.model.language_model.layers[layer]
    def hook(mod, args, out):
        h = out[0] if isinstance(out, tuple) else out
        h[:, -1, :] = h[:, -1, :] + alpha * vec.to(h.device)
        return (h,) + out[1:] if isinstance(out, tuple) else h
    return block.register_forward_hook(hook)
```

Sweep alpha over roughly `[-4, -2, -1, 0, 1, 2, 4]` and plot `AbsAcc` against `OverAbs`. The result that matters is whether abstention rises faster than over-abstention. If both rise together, the direction is a generic hedging knob and not an evidence signal, which is itself worth one honest paragraph.

### 4.10 `analysis/stats.py`

```python
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

def boot_ci(x, n=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x)
    bs = [rng.choice(x, len(x), replace=True).mean() for _ in range(n)]
    return float(x.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

def paired_test(a, b):
    """a, b: boolean correctness arrays on the same items."""
    t = np.array([[int((a & b).sum()),  int((a & ~b).sum())],
                  [int((~a & b).sum()), int((~a & ~b).sum())]])
    return float(mcnemar(t, exact=False, correction=True).pvalue)
```

Everything in the paper regenerates from `results/` through this module. Nothing gets typed into LaTeX by hand.

---

## 5. Notebook orchestration

**NB1 (CPU, no quota).** Attach COCO as input. Run `build()`, write `items.jsonl` and images to `/kaggle/working`. Save Version. This notebook's output becomes a dataset.

**NB2 (T4).** Attach NB1's output as a dataset input. Run clean-reference generation, then `run()` with the cause prompt for Qwen2.5-VL-3B, caching hidden states. Then abstain and repair prompts without hidden states. Save Version.

**NB3 (T4).** Same for InternVL3-2B and SmolVLM2-2.2B. Hidden states for InternVL only; SmolVLM behavioural only if quota is tight.

**NB4 (CPU, no quota).** Attach NB1, NB2, NB3 outputs. Run ladder, learning curve, CLIP baseline, generalization, stats, figures.

Two habits that will save the project:

1. **Never let a GPU notebook do anything a CPU notebook could do.** Quota is the scarce resource, not time.
2. **Save Version after every successful GPU run**, even a partial one. A half-finished `results.jsonl` saved as a dataset means the next session resumes instead of restarting.

---

## 6. Validation without paid annotators

Three free layers, in descending strength:

1. **Automatic verifiability, built into the generator.** S2 rejects below 90% realised coverage. S4 rejects below ΔE 12. S5 checks the category is absent from the annotations. S1 checks the crop excludes the union bbox. Every rejection is counted and the rejection rate per state goes in the appendix. This is stronger than annotation for the states where the property is geometric.
2. **Blind self-relabel.** Sample 100 items, shuffle, strip state labels, relabel from scratch after at least 48 hours. Report intra-annotator agreement. Weaker than inter-annotator but honest, and it catches a systematically broken generator.
3. **One free peer, if you can get one.** Two hours from any labmate buys real inter-annotator κ on 60 items. Ask. It costs nothing but a favour and it upgrades the paper.

Limitations must say: no paid annotation, validation is primarily automatic and geometric, inter-annotator agreement is on a small subset or absent. Reviewers accept this from a workshop paper when it is stated. They do not accept it when it is hidden.

---

## 7. Timeline (4 Aug to 29 Aug)

| Days | Dates | Work | Gate |
|---|---|---|---|
| 1 | 4 Aug | Kaggle phone verification, internet enabled. Attach COCO. Confirm fp16 + sdpa loads Qwen2.5-VL-3B on T4 and produces one sane output | Model loads, no NaN |
| 2-3 | 5-6 Aug | `schema.py`, `generate.py`, occluder bank. 200-item, 3-state pilot set | Pilot images look real |
| 4-5 | 7-8 Aug | NB2 pilot: option-token scoring, hidden-state caching, resume logic. Verify letter ids round-trip | End-to-end works |
| 6-7 | 9-10 Aug | **E0 go/no-go: pilot probe + CLIP baseline** | **Decision 10 Aug** |
| 8-9 | 11-12 Aug | Full generation, 6 states + s0ctrl + prioronly. Rejection stats | 1,500 items |
| 10 | 13 Aug | Clean reference answers, stability filter, drop-rate logged | References fixed |
| 11-13 | 14-16 Aug | NB2 and NB3 main sweeps, all 3 models. Table 1, Figure 2 | Minimum paper exists |
| 14-15 | 17-18 Aug | E2 dose-response, P1/P2 verdict. Tier B (200 VizWiz items) | Taxonomy verdict |
| 16-17 | 19-20 Aug | NB4: ladder, learning curve, generalization, A→B transfer. Figures 3, 4 | Headline number |
| 18-19 | 21-22 Aug | Steering if gated in. Blind self-relabel. **Freeze 22 Aug** | Freeze |
| 20-22 | 23-25 Aug | Write-up, figures, appendix | Complete draft |
| 23-24 | 26-27 Aug | Two readers, revise, anonymity and format check | Reviewed draft |
| 25 | 28 Aug | Polish, checklist | Ready |
| 26 | **29 Aug** | **Submit** | Done |

---

## 8. Kill criteria (unchanged from v2, plus two)

- **10 Aug, probe at chance across all layers** → pivot to benchmark-and-evaluation paper, report the null as a section.
- **10 Aug, CLIP baseline ≥ probe** → drop absolute probe numbers from the abstract, rebuild on the rung 1 vs rung 4 gap.
- **New: 8 Aug, letter tokens do not round-trip on a model** → drop that model rather than patching the parser. A parser bug that silently mislabels 3% of items is worse than two models.
- **New: 13 Aug, clean-answer stability drop rate > 35%** → the reference is too noisy. Switch to a closed-set colour question only, which stabilizes answers at the cost of task diversity, and say so.
- 14 Aug, self-relabel agreement below 0.6 on a state → drop that state.
- 17 Aug, all models at chance on `CauseAcc` → that is the paper, pivot to the negative framing.
- 20 Aug, E4 not clean → no steering.
- 25 Aug, under seven pages of substance → submit as an extended abstract.

---

## 9. Zero-cost checklist

- [ ] Kaggle account phone-verified, internet enabled in notebook settings
- [ ] COCO val2017 and annotations attached as free Kaggle datasets
- [ ] All models ≤7B, fp16, `attn_implementation="sdpa"`, no bf16 anywhere
- [ ] Every GPU notebook resumable from partial `results.jsonl`
- [ ] Save Version after every successful GPU run
- [ ] NB1 and NB4 on CPU accelerator, confirmed not billing GPU quota
- [ ] No API keys anywhere in the repo
- [ ] Total GPU hours logged and reported in the appendix, because "4 GPU-hours on free-tier T4s" is a genuinely good line in a workshop paper

That last point is worth taking seriously. A paper that gets a real result on 4 hours of free compute is more interesting to a workshop audience than one that spent 400. Put the number in the abstract.
