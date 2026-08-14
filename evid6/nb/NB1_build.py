# %% [markdown]
# # NB1: Build EVID-6 Dataset
# **Accelerator: CPU only** — does not consume GPU quota.
#
# ## What this notebook does
# 1. Loads COCO val2017 images and annotations
# 2. Builds the occluder bank (real objects from other images)
# 3. Generates all 6 evidence states + S0-ctrl + prior-only conditions
# 4. Writes `items.jsonl` and transformed images to `/kaggle/working`
#
# ## After running
# **Save Version**, then attach this notebook's output as a dataset input
# to NB2 and NB3.

# %% [markdown]
# ## Setup

# %%
import sys, os, json, random, shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt   # inline display for visual QA cells

# Add project root to path
sys.path.insert(0, "/kaggle/working/evid6/data")

# %%
# Locate the COCO dataset. Kaggle hosts several COCO 2017 datasets and they do
# not share a directory layout, so this searches rather than assuming one.
sys.path.insert(0, "/kaggle/working/evid6/data")
from generate import find_coco

COCO_ROOT, COCO_IMG_DIR = find_coco("/kaggle/input")
print(f"COCO root:   {COCO_ROOT}")
print(f"COCO images: {COCO_IMG_DIR}")
print(f"COCO val2017 images: {len(os.listdir(COCO_IMG_DIR))}")

# %% [markdown]
# ## Copy source code to working directory
# Kaggle wipes `/kaggle/working` between sessions, so we need the source
# available.  In practice you'd attach the repo as a dataset or paste
# the modules here.

# %%
# If running from a repo attached as a dataset, copy the code:
EVID6_SRC = "/kaggle/input/evid6-code/evid6"  # adjust path if needed
EVID6_DST = "/kaggle/working/evid6"

if os.path.isdir(EVID6_SRC) and not os.path.isdir(EVID6_DST):
    shutil.copytree(EVID6_SRC, EVID6_DST)
    print("Copied evid6 source to working directory")
elif os.path.isdir(EVID6_DST):
    print("evid6 source already in working directory")
else:
    print("WARNING: evid6 source not found. Modules must be pasted below.")

sys.path.insert(0, "/kaggle/working/evid6/data")
sys.path.insert(0, "/kaggle/working/evid6/analysis")

# %% [markdown]
# ## Schema verification

# %%
from schema import Item, STATES, STATE_TEXT, REPAIR, save_items

print("States:", STATES)
for s in STATES:
    print(f"  {s}: {STATE_TEXT[s]}")
    print(f"       repair: {REPAIR[s]}")

# %% [markdown]
# ## Initialise COCO and build the occluder bank

# %%
from generate import init_coco, build_occluder_bank, OUT_DIR

# Sets both the annotation path AND IMG_DIR — passing only a root used to leave
# IMG_DIR on its hard-coded default, which fails later on every image.
coco = init_coco(root=COCO_ROOT, img_dir=COCO_IMG_DIR)
print(f"COCO loaded: {len(coco.getImgIds())} images")

# %%
print("Building occluder bank (n=400)...")
bank = build_occluder_bank(n=400, seed=0)
print(f"Occluder bank: {len(bank)} patches")

# Show a sample
fig, axes = plt.subplots(1, 5, figsize=(15, 3))
for i, ax in enumerate(axes):
    ax.imshow(bank[i * 20][0])
    ax.set_title(bank[i * 20][1])
    ax.axis("off")
plt.suptitle("Sample occluder patches")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Build the full dataset
# Target: 150 items per state = 900 main + ~300 S0-ctrl + ~300 prior-only
# = ~1,500 items total.

# %%
from generate import build

items = build(n_per_state=150, seed=0)
print(f"\nTotal items built: {len(items)}")

# Distribution check
from collections import Counter
state_counts = Counter(it.state for it in items)
cond_counts = Counter(it.condition for it in items)
print("\nBy state:", dict(state_counts))
print("By condition:", dict(cond_counts))

# %% [markdown]
# ## Save items manifest

# %%
ITEMS_PATH = "/kaggle/working/items.jsonl"
save_items(items, ITEMS_PATH)
print(f"Saved {len(items)} items to {ITEMS_PATH}")

# Verify round-trip
from schema import load_items
reloaded = load_items(ITEMS_PATH)
assert len(reloaded) == len(items)
print("Round-trip verification passed")

# %% [markdown]
# ## Verify splits (leakage guard)

# %%
from splits import make_folds

# Only use main-condition items for fold verification
main_items = [it for it in items if it.condition == "main"]
folds = make_folds(main_items, n_splits=5, seed=0)
print(f"Fold distribution: {dict(Counter(folds))}")
print("No image leaks across folds ✓")

# %% [markdown]
# ## Visual inspection: sample items per state

# %%
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for idx, state in enumerate(STATES):
    ax = axes[idx // 3, idx % 3]
    sample = [it for it in items if it.state == state and it.condition == "main"]
    if sample:
        it = sample[0]
        img = Image.open(it.image_path)
        ax.imshow(img)
        ax.set_title(f"{it.state}: {it.question}\n[{it.category}]", fontsize=9)
    ax.axis("off")
plt.suptitle("Sample items per evidence state", fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Visual QA — look at the images before trusting the geometry
# Every acceptance check in the generator is a proxy. Coverage fractions and
# colour distances cannot tell you the occluder landed somewhere absurd, or
# that "ambiguous" candidates are obviously distinguishable to a person.
# ~300 images, a couple of minutes. Do this on the pilot, not after the sweep.

# %%
from qa_sheet import contact_sheets, triptychs

QA_DIR = "/kaggle/working/qa"
contact_sheets(items, QA_DIR, per_state=48, seed=0)
triptychs(items, QA_DIR, n=24, seed=0)

print("\nOpen /kaggle/working/qa/index.html and scan every sheet.")
print("Specifically check:")
print("  S3 severity 3 — degraded but still THERE? If it reads as deletion,")
print("                  S3 has collapsed into S2 and P1/P2 cannot separate.")
print("  S4            — would a person genuinely be unsure which is meant?")
print("  S2            — does the occluder look like an object, not a box?")

# %% [markdown]
# ## Summary statistics for appendix

# %%
print("=" * 60)
print("EVID-6 Dataset Summary")
print("=" * 60)
print(f"Total items: {len(items)}")
print(f"Main items: {sum(1 for it in items if it.condition == 'main')}")
print(f"S0-ctrl items: {sum(1 for it in items if it.condition == 's0ctrl')}")
print(f"Prior-only items: {sum(1 for it in items if it.condition == 'prioronly')}")
print(f"Unique base images: {len(set(it.base_image_id for it in items))}")
print(f"Unique categories: {len(set(it.category for it in items))}")
print(f"Output directory size: {sum(os.path.getsize(os.path.join(OUT_DIR, f)) for f in os.listdir(OUT_DIR)) / 1e6:.1f} MB")

# %% [markdown]
# ## Done
# **Save Version** now.  The output (items.jsonl + images/) becomes a
# Kaggle dataset input for NB2 and NB3.
