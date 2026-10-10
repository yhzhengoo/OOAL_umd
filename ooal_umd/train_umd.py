"""Train OOAL on the UMD one-shot references (run from the OOAL repo root).

    DRAWAFFORD_ROOT=/path/to/ConceptAttention \
    python -m ooal_umd.train_umd --split seen --save_root runs/ooal_umd

Recipe = the released ``train.py`` defaults (SGD lr 0.01, momentum 0.9,
Nesterov, wd 5e-4, cosine schedule, 20k iterations, batch size 1, seed 321),
with UMD data.  Unlike ``train.py`` there is no evaluation on the test set
during training, so no checkpoint is selected on test frames; checkpoints
are saved every ``--save_step`` iterations and the last one is evaluated.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import torch

from ooal_umd import umd_protocol as P
from ooal_umd.data_umd import CROP_HW, RESIZE_HW, UMDTrain
from ooal_umd.model_umd import OOALUMD
from ooal_umd.local_backbones import add_arguments
from utils.util import get_optimizer, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("seen", "unseen"), required=True)
    p.add_argument("--save_root", type=str, default="runs/ooal_umd")
    p.add_argument("--resize_hw", type=int, nargs=2, default=list(RESIZE_HW))
    p.add_argument("--crop_hw", type=int, nargs=2, default=list(CROP_HW))
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--show_step", type=int, default=100)
    p.add_argument("--save_step", type=int, default=5000)
    p.add_argument("--seed", type=int, default=321)
    p.add_argument("--gpu", type=str, default="0")
    add_arguments(p)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.cuda.set_device("cuda:" + args.gpu)
    set_seed(seed=args.seed)

    refs, _targets = P.inventory(args.split)
    args.class_names = list(P.OOAL_CLASS_NAMES)

    run = os.path.join(args.save_root, args.split, time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run, exist_ok=True)
    logging.basicConfig(filename=os.path.join(run, "run.log"), level=logging.INFO, format="%(message)s")
    log = logging.getLogger()
    log.addHandler(logging.StreamHandler(sys.stdout))
    config = {**vars(args), "references": [r.id for r in refs], "protocol": P.metadata(),
              "torch_version": torch.__version__}
    with open(os.path.join(run, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    log.info(f"{args.split}: {len(refs)} one-shot references -> {run}")

    loader = torch.utils.data.DataLoader(
        UMDTrain(refs, args.resize_hw, args.crop_hw),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = OOALUMD(args, 768, 512).cuda()
    model.train()
    optimizer, scheduler = get_optimizer(model, args)

    def save(step: int) -> None:
        path = os.path.join(run, f"iter_{step:06d}.pth")
        torch.save({"iter": step, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()}, path)
        log.info(f"saved {path}")

    step = 0
    while step < args.iters:
        for img, ann in loader:
            img, ann = img.cuda(), ann.cuda().float()
            _pred, loss_dict = model(img, label=ann)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1

            if step % args.show_step == 0:
                losses = " | ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
                log.info(f"iters: {step}/{args.iters} | {losses} | lr {scheduler.get_last_lr()[0]:.6f}")
            if step % args.save_step == 0 or step == args.iters:
                save(step)
            if step >= args.iters:
                break


if __name__ == "__main__":
    main()
