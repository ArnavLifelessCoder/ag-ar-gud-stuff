"""EVID-6 prompt templates.

Four prompt types, each used at a different stage of inference:
  CAUSE_PROMPT   – forced-choice: which evidence state best applies?
  ABSTAIN_PROMPT – answer or say CANNOT ANSWER
  CLEAN_PROMPT   – short-form answer on the untouched image (reference)
  REPAIR_PROMPT  – forced-choice: what action would help?
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))

from schema import STATE_TEXT, REPAIR

# ── Option strings ──────────────────────────────────────────────────────────

OPTS = "\n".join(
    f"({chr(65 + i)}) {STATE_TEXT[s]}" for i, s in enumerate(STATE_TEXT)
)

REPAIR_OPTS = "\n".join(
    f"({chr(65 + i)}) {v}" for i, v in enumerate(dict.fromkeys(REPAIR.values()))
)

# ── Prompt templates ────────────────────────────────────────────────────────

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

# Closed-set variant of the clean prompt, for the reference-stability rerun.
# CLEAN_PROMPT's "at most four words" leaves the answer space open, and three
# samples at temperature 0.7 then agree on well under half of reference groups.
# Naming the permitted answers collapses paraphrase without changing what is
# being asked. Compliance is measured (schema.is_closed_answer) rather than
# assumed: answers outside the set are counted, not silently snapped onto it.
from schema import CLOSED_COLOURS  # noqa: E402  (schema is already on the path)

CLEAN_CLOSED_PROMPT = (
    "Question about the image: {q}\n"
    "Answer with exactly one word from this list, and nothing else:\n"
    + ", ".join(CLOSED_COLOURS) + "."
)

REPAIR_PROMPT = (
    "Question about the image: {q}\n\n"
    "What single action would best let you answer this question reliably?\n\n"
    + REPAIR_OPTS + "\n\nReply with the letter only."
)


# ── Letter parsing (rung 1: what the model actually emits) ─────────────────

import re as _re

# The leading-letter patterns must not fire on the English article "A".
# `^\s*\(?([A-F])\)?\b` matched "A cat is sitting on the bed" and returned A -
# and A is S0, so every prose reply beginning with an article was scored as a
# correct S0 prediction. A bare letter therefore has to be followed by
# punctuation, a bracket, or end of string; a letter followed by a lowercase
# word is prose, not a choice.
_LETTER_PATTERNS = [
    _re.compile(r"^\s*\(?([A-F])\)?\s*[.):\-]?\s*$"),   # "B", "(B)", "B."
    _re.compile(r"^\s*\(([A-F])\)"),                    # "(B) the object is..."
    _re.compile(r"^\s*([A-F])\s*[).:\-–-]\s"),          # "B) ...", "B: ...", "B - ..."
    _re.compile(r"\banswer\s*(?:is)?\s*:?\s*\(?([A-F])\)?\b", _re.I),
    _re.compile(r"\boption\s*\(?([A-F])\)?\b", _re.I),
]


def parse_letter(text: str, n_options: int = 6):
    """Extract the chosen option letter from generated text.

    Returns the letter, or None if the model did not commit to one.  Rung 1
    is meant to measure deployed behaviour, so an unparseable reply is a
    genuine failure and must NOT be silently coerced into a guess - count it
    as incorrect and report the unparsed rate alongside the accuracy.
    """
    if not text:
        return None
    t = text.strip()
    valid = {chr(65 + i) for i in range(n_options)}
    for pat in _LETTER_PATTERNS:
        m = pat.search(t)
        if m and m.group(1).upper() in valid:
            return m.group(1).upper()
    # Last resort: a short reply containing exactly one standalone valid
    # letter.  Standalone matters - scanning raw characters made "off" parse
    # as F and "a dog" as A.
    if len(t) <= 40:
        found = {w.upper() for w in _re.findall(r"\b([A-Fa-f])\b", t)
                 if w.upper() in valid}
        # "A" is excluded here on purpose. A reply that is *only* the letter A
        # already matched pattern 1 above, so a standalone A reaching this
        # point is inside a longer reply, where it is the English article far
        # more often than a choice - and A is S0, the state this would
        # otherwise hand free credit to.
        found.discard("A")
        if len(found) == 1:
            return found.pop()
    return None


# ── Few-shot / in-context prompting (rung 2) ───────────────────────────────

FEWSHOT_HEADER = (
    "Here are some solved examples of the same task.\n"
    "Each example gives a question and the correct evidence-state letter.\n\n"
)


def build_fewshot_prefix(examples, n: int = 8):
    """Build a text-only in-context prefix from labelled training examples.

    Parameters
    ----------
    examples : list of dict
        Items drawn ONLY from training folds.  Each needs 'question' and
        'state'.  Caller is responsible for the fold discipline - passing
        test-fold items here silently inflates rung 2.
    n : int
        Number of exemplars.

    Notes
    -----
    The exemplars are text-only: question plus gold letter, with no exemplar
    images.  Interleaving eight extra images per call is not affordable on a
    T4 and most 2-3B VLMs are not trained for multi-image ICL anyway.  So
    rung 2 measures whether *label-space and format priming* closes the gap,
    not whether visual demonstrations do.  State this explicitly in the paper
    - it is a weaker claim than full multimodal ICL and should not be
    reported as one.
    """
    lines = [FEWSHOT_HEADER]
    for ex in examples[:n]:
        letter = chr(65 + int(str(ex["state"])[1]))
        lines.append(f"Question: {ex['question']}\nAnswer: ({letter})\n")
    lines.append("Now the actual image and question.\n\n")
    return "\n".join(lines)


def balanced_examples(train_items, n: int = 8, seed: int = 0):
    """Pick ``n`` exemplars spread as evenly as possible across states.

    A prefix that is 6/8 S0 teaches the model to say A. Balance matters more
    here than sample size.
    """
    import random as _random
    rng = _random.Random(seed)
    by_state = {}
    for it in train_items:
        st = it["state"] if isinstance(it, dict) else it.state
        by_state.setdefault(st, []).append(it)
    states = sorted(by_state)
    picked, i = [], 0
    while len(picked) < n and states:
        st = states[i % len(states)]
        pool = by_state[st]
        if pool:
            picked.append(pool.pop(rng.randrange(len(pool))))
        else:
            states.remove(st)
            continue
        i += 1
    rng.shuffle(picked)
    return [(p if isinstance(p, dict) else
             {"question": p.question, "state": p.state}) for p in picked]
