"""EVID-6 inference runner.

One forward pass gives:
  - forced-choice logit distribution (ladder rung 3)
  - argmax answer (rung 1)
  - cached residual stream for probing (rung 4)

Do NOT run these as separate passes - the pairing is what makes the
comparison valid.

T4-safe: fp16 + SDPA only.  No bf16 (Turing does not have it), no
FlashAttention-2 (needs Ampere+).
"""

import os
import json
import gc
import torch
import numpy as np
from PIL import Image
import transformers
from transformers import AutoProcessor

# Manifest rebasing lives in schema.py so the CPU tests can use it without
# dragging torch and transformers in behind it.
from schema import rebase_items      # noqa: F401  (re-exported for NB2/NB3)

# The auto-class for vision-language models was renamed. transformers >= 4.45
# has AutoModelForImageTextToText; AutoModelForVision2Seq was the old name and
# is gone in transformers 5.x, which is what current Kaggle images ship. Prefer
# the new name and fall back, so this file loads on both.
try:
    from transformers import AutoModelForImageTextToText as AutoVLM
    _AUTO_CLASS = "AutoModelForImageTextToText"
except ImportError:                                  # transformers < 4.45
    from transformers import AutoModelForVision2Seq as AutoVLM
    _AUTO_CLASS = "AutoModelForVision2Seq"

WORK = "/kaggle/working"
DEV = "cuda"


# ── Model loading ───────────────────────────────────────────────────────────

def load(model_id: str):
    """Load a VLM in fp16 with SDPA attention (T4-safe).

    Returns (processor, model).
    """
    # The legacy InternVL repository exposes InternVLChatConfig through remote
    # code.  Current Transformers auto-classes cannot instantiate that config,
    # while the official ``-hf`` conversion uses the native InternVL config.
    # Redirect here as well as in NB3 so stale Kaggle pre-flight cells cannot
    # spend a session failing before the actual notebook starts.
    if model_id.rstrip("/") == "OpenGVLab/InternVL3-2B":
        model_id = "OpenGVLab/InternVL3-2B-hf"
        print("Using HF-native InternVL checkpoint: " + model_id)

    proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # `torch_dtype` was renamed to `dtype` and removed in transformers 5.x.
    # Try the current spelling, fall back to the old one, so the same file
    # works on a 4.x Kaggle image and a 5.x one.
    common = dict(
        attn_implementation="sdpa",       # FlashAttention-2 needs Ampere+
        device_map=DEV,
        trust_remote_code=True,
    )
    def _load(dtype_kw):
        try:
            return AutoVLM.from_pretrained(model_id, **{dtype_kw: torch.float16}, **common)
        except ValueError as e:
            # The model ships a custom config the Auto-class can't map (e.g.
            # OpenGVLab/InternVL3-2B's InternVLChatConfig via remote code).
            # transformers 5.x has native InternVL support, so the "-hf"
            # checkpoint loads; point there instead of the raw ValueError.
            if "Unrecognized configuration class" in str(e):
                hint = ""
                if "internvl" in model_id.lower() and not model_id.lower().endswith("-hf"):
                    hint = f" Try the HF-native checkpoint: '{model_id}-hf'."
                raise ValueError(
                    f"{model_id} uses a custom config that "
                    f"{_AUTO_CLASS} cannot map.{hint} "
                    "If no -hf variant exists, this model needs its own "
                    "loader (model.chat() with explicit pixel_values)."
                ) from e
            raise

    try:
        model = _load("dtype")            # transformers 5.x spelling
    except TypeError:
        model = _load("torch_dtype")      # transformers 4.x spelling
    return proc, model.eval()


def env_report() -> dict:
    """Print the versions and class names this run actually resolved.

    Worth calling once before a sweep: the auto-class rename and the
    torch_dtype/dtype rename both fail at import or first call, after the
    session has already started.
    """
    info = {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "auto_class": _AUTO_CLASS,
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    for k, v in info.items():
        print(f"  {k:14} {v}")
    return info


# ── Token utilities ─────────────────────────────────────────────────────────

def letter_ids(proc, n: int = 6) -> list:
    """Resolve option-letter token ids and verify they round-trip.

    Kill criterion (8 Aug): if any letter does not round-trip, drop that model
    rather than patching the parser.
    """
    tok = proc.tokenizer
    ids = []
    for i in range(n):
        L = chr(65 + i)
        cand = tok.encode(L, add_special_tokens=False)
        assert len(cand) >= 1, f"Letter {L} encodes to empty sequence"
        tid = cand[0]
        decoded = tok.decode([tid]).strip()
        assert decoded == L, (
            f"Token mismatch for {L}: decode({tid}) = {decoded!r}. "
            f"Drop this model (kill criterion 8 Aug)."
        )
        ids.append(tid)
    return ids


# ── Input construction ──────────────────────────────────────────────────────

def build_inputs(proc, image_path: str, prompt: str):
    """Build model inputs from image path and prompt text.

    Caps image to 640x640 to keep visual token count manageable on T4.
    """
    msgs = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt},
    ]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    img = Image.open(image_path).convert("RGB")
    img.thumbnail((640, 640))               # cap visual tokens
    return proc(text=[text], images=[img], return_tensors="pt").to(DEV)


# ── Scoring ─────────────────────────────────────────────────────────────────

@torch.inference_mode()
def score_one(proc, model, item: dict, prompt: str, opt_ids: list,
              want_hidden: bool = True):
    """Score a single item: logit distribution + optional hidden states.

    Parameters
    ----------
    item : dict
        Must have 'image_path' key.
    prompt : str
        Formatted prompt string.
    opt_ids : list of int
        Token ids for the option letters (A..F).
    want_hidden : bool
        If True, cache the full residual stream (all layers, last token).

    Returns
    -------
    probs : np.ndarray, shape (n_options,)
    hs : np.ndarray or None, shape (n_layers+1, hidden_dim)
    """
    inp = build_inputs(proc, item["image_path"], prompt)
    out = model(**inp, output_hidden_states=want_hidden, use_cache=False)

    # Logit distribution over option tokens
    last = out.logits[0, -1].float()
    probs = torch.softmax(last[opt_ids], -1).cpu().numpy()

    # Hidden states: all layers, last token position
    hs = None
    if want_hidden:
        hs = torch.stack([h[0, -1] for h in out.hidden_states]).half().cpu().numpy()

    del out, inp
    return probs, hs


# ── Generation-based inference ──────────────────────────────────────────────

@torch.inference_mode()
def generate_one(proc, model, item: dict, prompt: str, max_new: int = 12,
                 temperature: float = 0.0):
    """Generate a free-form answer (for clean references and abstain).

    Parameters
    ----------
    temperature : float
        0.0 for greedy, > 0 for sampling (clean ref stability check uses 0.7).
    """
    inp = build_inputs(proc, item["image_path"], prompt)
    gen_kwargs = dict(max_new_tokens=max_new, do_sample=(temperature > 0))
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
    output_ids = model.generate(**inp, **gen_kwargs)
    # Decode only the new tokens
    new_ids = output_ids[0, inp["input_ids"].shape[1]:]
    answer = proc.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    del output_ids, inp
    return answer


# ── Result-row construction ─────────────────────────────────────────────────

# Intervention metadata carried from items.jsonl into every result row.
# Without these the dose-response analysis (E2, the P1/P2 verdict) silently
# degenerates: every item falls back to a default severity / occlusion.
META_FIELDS = ["category", "occl_frac", "inst_pixels", "severity",
               "eff_res", "n_candidates", "delta_e", "ref_answer",
               "image_path", "ref_group", "parent_item_id", "artifact"]


def base_row(item: dict) -> dict:
    """Identity + intervention metadata for one result row."""
    row = {
        "item_id": item["item_id"],
        "state": item["state"],
        "condition": item["condition"],
        "base_image_id": item["base_image_id"],
    }
    for k in META_FIELDS:
        if item.get(k) is not None:
            row[k] = item[k]
    return row


# ── Batch runner ────────────────────────────────────────────────────────────

def run(model_id: str, items_path: str, tag: str, prompt_fn,
        cache_hidden: bool = True, shard: int = 100,
        proc=None, model=None, keep_loaded: bool = False,
        item_filter=None):
    """Run inference on all items. Resumable: re-running skips finished items.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.
    items_path : str
        Path to ``items.jsonl``.
    tag : str
        Tag for output files (e.g., 'qwen_cause').
    prompt_fn : callable
        Takes ``q=question`` keyword and returns the formatted prompt.
    cache_hidden : bool
        Whether to save hidden states (needed for probing, costs disk).
    shard : int
        Number of items per hidden-state shard file.
    proc, model : optional
        Pass an already-loaded pair to avoid holding two copies of the model
        on the GPU at once.  If omitted, this function loads its own.
    keep_loaded : bool
        If True, do not free the model on exit (caller owns it).
    item_filter : callable or None
        ``fn(item_dict) -> bool``.  Only items returning True are scored.
    """
    owns_model = model is None
    if owns_model:
        proc, model = load(model_id)
    opt_ids = letter_ids(proc)
    res_path = f"{WORK}/results/{tag}.jsonl"
    os.makedirs(f"{WORK}/results", exist_ok=True)
    os.makedirs(f"{WORK}/acts/{tag}", exist_ok=True)

    # Resume logic: skip already-scored items
    done = set()
    if os.path.exists(res_path):
        with open(res_path, encoding="utf-8") as f:
            done = {json.loads(l)["item_id"] for l in f if l.strip()}
    with open(items_path, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    todo = [it for it in items if it["item_id"] not in done]
    if item_filter is not None:
        todo = [it for it in todo if item_filter(it)]
    print(f"{len(done)} done, {len(todo)} to go")

    buf_h, buf_id = [], []
    with open(res_path, "a", encoding="utf-8") as f:
        for n, it in enumerate(todo):
            probs, hs = score_one(
                proc, model, it,
                prompt_fn(q=it["question"]), opt_ids, cache_hidden,
            )
            row = base_row(it)
            row["probs"] = probs.tolist()
            row["pred"] = chr(65 + int(probs.argmax()))
            f.write(json.dumps(row) + "\n")
            if n % 20 == 0:
                f.flush()

            # Hidden-state caching
            if hs is not None:
                buf_h.append(hs)
                buf_id.append(it["item_id"])
                if len(buf_h) >= shard:
                    k = len(os.listdir(f"{WORK}/acts/{tag}")) // 2
                    np.save(f"{WORK}/acts/{tag}/h_{k:04d}.npy", np.stack(buf_h))
                    np.save(f"{WORK}/acts/{tag}/i_{k:04d}.npy", np.array(buf_id))
                    buf_h, buf_id = [], []

            if n % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    # Flush remaining hidden states
    if buf_h:
        k = len(os.listdir(f"{WORK}/acts/{tag}")) // 2
        np.save(f"{WORK}/acts/{tag}/h_{k:04d}.npy", np.stack(buf_h))
        np.save(f"{WORK}/acts/{tag}/i_{k:04d}.npy", np.array(buf_id))

    if owns_model and not keep_loaded:
        del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Finished {tag}: {len(todo)} new items scored")


def run_generation(model_id: str, items_path: str, tag: str, prompt_fn,
                   n_samples: int = 1, temperature: float = 0.0,
                   proc=None, model=None, keep_loaded: bool = False,
                   item_filter=None):
    """Run generation-based inference (clean refs, abstain).

    For clean references, use n_samples=3 and temperature=0.7 to check
    answer stability.

    ``proc``/``model`` let the caller pass an already-loaded pair so the GPU
    never holds two copies at once.  ``item_filter`` restricts which items
    are generated for (e.g. clean references only need untouched images).
    """
    owns_model = model is None
    if owns_model:
        proc, model = load(model_id)
    res_path = f"{WORK}/results/{tag}.jsonl"
    os.makedirs(f"{WORK}/results", exist_ok=True)

    done = set()
    if os.path.exists(res_path):
        with open(res_path, encoding="utf-8") as f:
            done = {json.loads(l)["item_id"] for l in f if l.strip()}
    with open(items_path, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    todo = [it for it in items if it["item_id"] not in done]
    if item_filter is not None:
        todo = [it for it in todo if item_filter(it)]
    print(f"{len(done)} done, {len(todo)} to go")

    with open(res_path, "a", encoding="utf-8") as f:
        for n, it in enumerate(todo):
            answers = []
            for s in range(n_samples):
                t = temperature if n_samples > 1 else 0.0
                ans = generate_one(proc, model, it, prompt_fn(q=it["question"]),
                                   temperature=t)
                answers.append(ans)
            row = base_row(it)
            row["answers"] = answers
            row["answer"] = answers[0]          # consumed by stats.consistency_rate
            row["stable"] = len(set(a.lower().strip() for a in answers)) == 1
            f.write(json.dumps(row) + "\n")
            if n % 20 == 0:
                f.flush()
            if n % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    if owns_model and not keep_loaded:
        del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Finished {tag}: {len(todo)} new items generated")


def run_choice_generation(model_id: str, items_path: str, tag: str, prompt_fn,
                          n_options: int = 6, max_new: int = 8,
                          fewshot_prefix: str = "",
                          proc=None, model=None, keep_loaded: bool = False,
                          item_filter=None):
    """Rung 1 / rung 2: the forced-choice task answered by GENERATION.

    This is the deployed behaviour the paper compares against the probe.  It
    is deliberately not the option-token argmax (that is rung 3) - the model
    has to actually emit a letter, and replies that commit to nothing are
    recorded as ``pred=None`` and counted as incorrect.

    ``fewshot_prefix`` turns this into rung 2.  Build it with
    ``prompts.build_fewshot_prefix`` from TRAINING-FOLD items only.
    """
    from prompts import parse_letter

    owns_model = model is None
    if owns_model:
        proc, model = load(model_id)
    res_path = f"{WORK}/results/{tag}.jsonl"
    os.makedirs(f"{WORK}/results", exist_ok=True)

    done = set()
    if os.path.exists(res_path):
        with open(res_path, encoding="utf-8") as f:
            done = {json.loads(l)["item_id"] for l in f if l.strip()}
    with open(items_path, encoding="utf-8") as f:
        items = [json.loads(l) for l in f if l.strip()]
    todo = [it for it in items if it["item_id"] not in done]
    if item_filter is not None:
        todo = [it for it in todo if item_filter(it)]
    print(f"{len(done)} done, {len(todo)} to go")

    n_unparsed = 0
    with open(res_path, "a", encoding="utf-8") as f:
        for n, it in enumerate(todo):
            prompt = fewshot_prefix + prompt_fn(q=it["question"])
            raw = generate_one(proc, model, it, prompt, max_new=max_new,
                               temperature=0.0)
            letter = parse_letter(raw, n_options=n_options)
            if letter is None:
                n_unparsed += 1
            row = base_row(it)
            row["raw"] = raw
            row["pred"] = letter          # may be None - that is a real result
            f.write(json.dumps(row) + "\n")
            if n % 20 == 0:
                f.flush()
            if n % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

    if owns_model and not keep_loaded:
        del model
    gc.collect()
    torch.cuda.empty_cache()
    rate = n_unparsed / max(len(todo), 1)
    print(f"Finished {tag}: {len(todo)} generated, "
          f"{n_unparsed} unparseable ({rate:.1%})")
    if rate > 0.10:
        print("  WARNING: >10% unparseable. Report this rate in the paper; "
              "do not quietly coerce these into guesses.")
