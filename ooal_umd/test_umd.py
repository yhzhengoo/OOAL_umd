"""Evaluate an OOAL UMD checkpoint with DrawAfford's UMD protocol.

    DRAWAFFORD_ROOT=/path/to/ConceptAttention \
    python -m ooal_umd.test_umd --split seen \
        --model_file runs/ooal_umd/seen/<time>/iter_020000.pth \
        --out_dir runs/ooal_umd/seen/<time>/eval_020000

Writes
    summary.json    metrics from ``umd_eval.scores`` (mIoU over 7 affordances)
    per_frame.npz   ids, objects, cm (N, 8, 8) -- same layout as DrawAfford's
    pred/<id>.png   uint8 label maps (0 = background, 1..7 = UMD ids) for figures
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from ooal_umd import umd_protocol as P
from ooal_umd.data_umd import RESIZE_HW, UMDTest
from ooal_umd.model_umd import OOALUMD, check_forward_all
from utils.util import set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("seen", "unseen"), required=True)
    p.add_argument("--model_file", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--resize_hw", type=int, nargs=2, default=list(RESIZE_HW))
    p.add_argument("--no_save_pred", action="store_true", help="skip pred/<id>.png")
    p.add_argument("--limit", type=int, default=0, help="first N frames only (smoke test)")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--gpu", type=str, default="0")
    return p.parse_args()


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    torch.cuda.set_device("cuda:" + args.gpu)
    set_seed(seed=321)
    ev = P.umd_eval()

    _refs, targets = P.inventory(args.split)
    if args.limit:
        targets = targets[: args.limit]
    args.class_names = list(P.OOAL_CLASS_NAMES)

    model = OOALUMD(args, 768, 512).cuda()
    state = torch.load(args.model_file, map_location="cpu")["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    pred_dir = os.path.join(args.out_dir, "pred")
    if not args.no_save_pred:
        os.makedirs(pred_dir, exist_ok=True)

    loader = torch.utils.data.DataLoader(
        UMDTest(targets, args.resize_hw), batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    cm_total = np.zeros((ev.N_CLASSES + 1,) * 2, dtype=np.int64)
    per_frame = np.zeros((len(targets), ev.N_CLASSES + 1, ev.N_CLASSES + 1), dtype=np.int64)
    self_check = None
    t0 = time.time()
    for img, label, index in tqdm(loader):
        img = img.cuda(non_blocking=True)
        if self_check is None:
            self_check = check_forward_all(model, img)
            print(f"self-check forward_all vs Net.forward: max diff {self_check:.2e}")
        probs = model.forward_all(img)[0].float().cpu().numpy()      # (7, h, w)
        gt = label[0].numpy().astype(np.int64)
        pred = P.probs_to_label(probs, gt.shape)                     # (H, W) 0..7
        cm = ev.confusion(gt, pred)
        i = int(index)
        per_frame[i] = cm
        cm_total += cm
        if not args.no_save_pred:
            Image.fromarray(pred.astype(np.uint8)).save(os.path.join(pred_dir, f"{targets[i].id}.png"))

    metrics = ev.scores(cm_total)
    np.savez_compressed(
        os.path.join(args.out_dir, "per_frame.npz"),
        ids=np.array([t.id for t in targets]),
        objects=np.array([t.object_name for t in targets]),
        cm=per_frame,
    )
    summary = {
        "method": "OOAL (re-implemented on UMD)",
        "split": args.split,
        "frames": len(targets),
        "limit": args.limit,
        "model_file": os.path.abspath(args.model_file),
        "model_sha256": sha256(args.model_file),
        "resize_hw": args.resize_hw,
        "threshold": float(ev.THRESHOLD),
        "self_check_max_diff": self_check,
        "confusion": cm_total.tolist(),
        "metrics": metrics,
        "seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({"split": args.split, "frames": len(targets),
                      "mIoU_7fg": metrics["mIoU_7fg"], "iou": metrics["iou"]}, indent=2))


if __name__ == "__main__":
    main()
