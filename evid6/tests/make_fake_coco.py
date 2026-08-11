"""Build a small COCO-format dataset so the generator driver can be run
end to end without downloading 1 GB of val2017.

This is NOT a substitute for looking at real images. It exists to exercise
``generate.build()`` — the driver, the reference-group bookkeeping, the
rejection accounting, the auxiliary conditions — which the unit-level smoke
test never touches because it only calls the individual state generators.

The images are crude: coloured blobs on textured backgrounds. That is enough
for the geometry, which is all the driver cares about.
"""

import os
import json
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

CATEGORIES = [
    "person", "dog", "cat", "car", "chair", "bottle", "laptop", "book",
    "clock", "cup", "bicycle", "bird", "tv", "stop sign", "bench",
]


def _blob(draw, cx, cy, rx, ry, colour, rng, n=14):
    """An irregular closed polygon, so masks are not all ellipses."""
    pts = []
    for i in range(n):
        t = 2 * np.pi * i / n
        jitter = 0.75 + 0.5 * rng.random()
        pts.append((cx + rx * jitter * np.cos(t),
                    cy + ry * jitter * np.sin(t)))
    draw.polygon(pts, fill=colour)
    return pts


def build_fake_coco(out_dir: str, n_images: int = 160, seed: int = 0):
    """Write images + an instances_val2017.json the COCO API can load."""
    rng = random.Random(seed)
    nprng = np.random.default_rng(seed)
    img_dir = os.path.join(out_dir, "val2017")
    ann_dir = os.path.join(out_dir, "annotations")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(ann_dir, exist_ok=True)

    cats = [{"id": i + 1, "name": n, "supercategory": "thing"}
            for i, n in enumerate(CATEGORIES)]
    images, annotations = [], []
    ann_id = 1

    for iid in range(1, n_images + 1):
        W, H = 640, 480
        # Textured background so downsampling and blurring have something
        # to act on — a flat fill would make S3 a no-op.
        bg = nprng.integers(60, 200, size=(H // 16, W // 16, 3), dtype=np.uint8)
        img = Image.fromarray(bg).resize((W, H), Image.BICUBIC)
        img = img.filter(ImageFilter.GaussianBlur(2))
        draw = ImageDraw.Draw(img)

        fname = f"{iid:012d}.jpg"
        images.append({"id": iid, "file_name": fname, "width": W, "height": H})

        n_obj = rng.choice([1, 1, 2, 2, 3, 4])
        chosen = rng.sample(CATEGORIES, min(rng.choice([1, 1, 2]), len(CATEGORIES)))
        for _ in range(n_obj):
            cat = rng.choice(chosen)
            cid = CATEGORIES.index(cat) + 1
            rx = rng.randint(40, 95)
            ry = rng.randint(40, 95)
            cx = rng.randint(rx + 10, W - rx - 10)
            cy = rng.randint(ry + 10, H - ry - 10)
            colour = tuple(int(c) for c in nprng.integers(0, 255, 3))
            pts = _blob(draw, cx, cy, rx, ry, colour, rng)

            flat = [float(v) for p in pts for v in p]
            xs = flat[0::2]
            ys = flat[1::2]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            # Shoelace area, close enough for the MIN/MAX_AREA filter
            area = 0.0
            for i in range(len(pts)):
                x_i, y_i = pts[i]
                x_j, y_j = pts[(i + 1) % len(pts)]
                area += x_i * y_j - x_j * y_i
            area = abs(area) / 2.0

            annotations.append({
                "id": ann_id, "image_id": iid, "category_id": cid,
                "segmentation": [flat], "iscrowd": 0, "area": float(area),
                "bbox": [x0, y0, x1 - x0, y1 - y0],
            })
            ann_id += 1

        img.save(os.path.join(img_dir, fname), quality=92)

    payload = {
        "info": {"description": "synthetic COCO-format fixture for EVID-6"},
        "licenses": [], "images": images,
        "annotations": annotations, "categories": cats,
    }
    with open(os.path.join(ann_dir, "instances_val2017.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"fixture: {len(images)} images, {len(annotations)} annotations, "
          f"{len(cats)} categories -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    import sys
    build_fake_coco(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fakecoco")
