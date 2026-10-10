"""Qualitative UMD figure: Image | GT | method 1 | method 2 ... (no torch).

Every method is given as a directory of label maps ``<frame id>.png``
(uint8, 0 = background, 1..7 = UMD affordance ids), e.g. OOAL's
``eval_*/pred``.  All columns of a row use the same crop, taken from the GT
object box, and the same palette.

    DRAWAFFORD_ROOT=/path/to/ConceptAttention \
    python -m ooal_umd.viz_umd --split seen \
        --method OOAL=runs/ooal_umd/seen/<time>/eval_020000/pred \
        --method Ours=/path/to/ours/seen/pred \
        --per-category 1 --seed 0 --out figs/umd_seen

Frames are either listed with ``--ids`` or drawn with ``--per-category K``:
K random frames per object category with a fixed seed, so the selection does
not depend on which method looks better.  ``--categories`` limits the rows to
some categories; ``--list-ids`` only writes the chosen ids (so other methods
can dump label maps for exactly these frames first).
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np
from PIL import Image

from ooal_umd import umd_protocol as P

NAMES = ("grasp", "cut", "scoop", "contain", "pound", "support", "wrap-grasp")
# Okabe-Ito colours (colour-blind safe), one per UMD id 1..7.
PALETTE = np.array([
    (0, 114, 178),    # grasp       blue
    (213, 94, 0),     # cut         vermillion
    (0, 158, 115),    # scoop       green
    (240, 228, 66),   # contain     yellow
    (204, 121, 167),  # pound       pink
    (86, 180, 233),   # support     sky blue
    (230, 159, 0),    # wrap-grasp  orange
], dtype=np.uint8)


def overlay(rgb: np.ndarray, label: np.ndarray, alpha: float) -> np.ndarray:
    """Blend affordance colours onto the image; background pixels are unchanged."""
    if rgb.shape[:2] != label.shape:
        raise ValueError(f"image {rgb.shape[:2]} vs label {label.shape}")
    out = rgb.astype(np.float32).copy()
    fg = label > 0
    out[fg] = (1.0 - alpha) * out[fg] + alpha * PALETTE[label[fg] - 1].astype(np.float32)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def crop_box(label: np.ndarray, margin: float) -> tuple[int, int, int, int]:
    """Square box (y0, y1, x0, x1) around the GT foreground, padded by ``margin``.

    Falls back to the whole frame when the GT is empty.  The box is shifted
    (not shrunk) to stay inside the image whenever it fits.
    """
    H, W = label.shape
    ys, xs = np.nonzero(label > 0)
    if ys.size == 0:
        return 0, H, 0, W
    cy, cx = (ys.min() + ys.max() + 1) / 2.0, (xs.min() + xs.max() + 1) / 2.0
    side = max(ys.max() - ys.min() + 1, xs.max() - xs.min() + 1) * (1.0 + 2.0 * margin)
    side = int(round(min(side, H, W)))
    y0 = int(round(cy - side / 2.0))
    x0 = int(round(cx - side / 2.0))
    y0 = min(max(y0, 0), H - side)
    x0 = min(max(x0, 0), W - side)
    return y0, y0 + side, x0, x0 + side


def pick_ids(targets, per_category: int, seed: int, categories=None) -> list[str]:
    """``per_category`` random frames for every object category (sorted), fixed seed.

    ``categories`` restricts which categories get rows; the frames inside a
    category are still drawn at random, and each category has its own draw,
    so adding or removing a category does not change the others' frames.
    """
    by_family = defaultdict(list)
    for t in targets:
        by_family[t.object_name.split("_", 1)[0]].append(t.id)
    if categories:
        unknown = sorted(set(categories) - set(by_family))
        if unknown:
            raise SystemExit(f"no test frames for categories {unknown}; available: {sorted(by_family)}")
    ids = []
    for family in sorted(by_family):
        if categories and family not in categories:
            continue
        rng = np.random.default_rng([seed, *family.encode()])
        pool = sorted(by_family[family])
        take = min(per_category, len(pool))
        ids.extend(pool[i] for i in sorted(rng.choice(len(pool), size=take, replace=False)))
    return ids


def parse_methods(specs: list[str]) -> list[tuple[str, str]]:
    methods = []
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--method expects NAME=DIR, got {spec!r}")
        name, path = spec.split("=", 1)
        if not os.path.isdir(path):
            raise SystemExit(f"{name}: {path} is not a directory")
        methods.append((name, path))
    return methods


def panel(img: np.ndarray, box, size: int) -> np.ndarray:
    y0, y1, x0, x1 = box
    return np.asarray(Image.fromarray(img[y0:y1, x0:x1]).resize((size, size), Image.BILINEAR))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("seen", "unseen"), required=True)
    p.add_argument("--method", action="append", default=[], help="NAME=DIR of <id>.png label maps")
    p.add_argument("--ids", nargs="*", default=None)
    p.add_argument("--per-category", type=int, default=1)
    p.add_argument("--categories", nargs="*", default=None, help="only these object categories")
    p.add_argument("--list-ids", action="store_true",
                   help="only write the chosen frame ids to <out>_ids.txt (to dump other methods' maps)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--crop-margin", type=float, default=0.25)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--panel-size", type=int, default=256)
    p.add_argument("--save-panels", action="store_true", help="also write each panel as a PNG")
    p.add_argument("--out", required=True, help="output path without extension")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    methods = parse_methods(args.method)
    _refs, targets = P.inventory(args.split)
    by_id = {t.id: t for t in targets}
    ids = args.ids if args.ids else pick_ids(targets, args.per_category, args.seed, args.categories)
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(f"not test frames of {args.split}: {missing[:5]}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out + "_ids.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(ids) + "\n")
    if args.list_ids:
        print(f"{len(ids)} frame ids -> {args.out}_ids.txt")
        return

    columns = ["Image", "GT"] + [name for name, _ in methods]
    rows = []
    for frame_id in ids:
        item = by_id[frame_id]
        rgb = np.asarray(Image.open(item.image).convert("RGB"))
        gt = P.load_label(item)
        box = crop_box(gt, args.crop_margin)
        row = [panel(rgb, box, args.panel_size), panel(overlay(rgb, gt, args.alpha), box, args.panel_size)]
        for name, path in methods:
            pred_path = os.path.join(path, f"{frame_id}.png")
            if not os.path.isfile(pred_path):
                raise SystemExit(f"{name}: missing {pred_path}")
            pred = np.asarray(Image.open(pred_path))
            row.append(panel(overlay(rgb, pred, args.alpha), box, args.panel_size))
        rows.append(row)

    if args.save_panels:
        panel_dir = args.out + "_panels"
        os.makedirs(panel_dir, exist_ok=True)
        for frame_id, row in zip(ids, rows):
            for col, img in zip(columns, row):
                Image.fromarray(img).save(os.path.join(panel_dir, f"{frame_id}__{col}.png"))

    n_r, n_c = len(rows), len(columns)
    fig, axes = plt.subplots(n_r, n_c, figsize=(1.6 * n_c, 1.6 * n_r + 0.4), squeeze=False)
    for r, row in enumerate(rows):
        for c, img in enumerate(row):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0:
                ax.set_title(columns[c], fontsize=9)
        axes[r, 0].set_ylabel(by_id[ids[r]].object_name.split("_", 1)[0], fontsize=8)
    handles = [Patch(color=PALETTE[i] / 255.0, label=n) for i, n in enumerate(NAMES)]
    fig.legend(handles=handles, loc="lower center", ncol=len(NAMES), fontsize=7, frameon=False)
    fig.subplots_adjust(wspace=0.03, hspace=0.03, bottom=0.4 / (1.6 * n_r + 0.4))
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"{len(ids)} frames -> {args.out}.png / .pdf (ids in {args.out}_ids.txt)")


if __name__ == "__main__":
    main()
