"""Seen / unseen / hIoU of the re-implementation next to OOAL's Table 2.

    python -m ooal_umd.report_umd --seen <seen eval dir> --unseen <unseen eval dir> \
        [--out report.md]
"""

from __future__ import annotations

import argparse
import json
import os

from ooal_umd import umd_protocol as P

NAMES = ("grasp", "cut", "scoop", "contain", "pound", "support", "wrap-grasp")


def load(path: str) -> dict:
    with open(os.path.join(path, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    if summary.get("limit"):
        raise SystemExit(f"{path}: evaluated with --limit {summary['limit']}, not the full split")
    return summary


def build(seen: dict, unseen: dict) -> str:
    s = 100 * seen["metrics"]["mIoU_7fg"]
    u = 100 * unseen["metrics"]["mIoU_7fg"]
    h = 100 * P.hiou(s / 100, u / 100)
    paper = P.OOAL_PAPER
    lines = [
        "# OOAL on UMD: re-implementation vs. paper",
        "",
        "| | Seen | Unseen | hIoU |",
        "|---|---:|---:|---:|",
        f"| OOAL paper (Table 2) | {paper['seen']:.1f} | {paper['unseen']:.1f} | {paper['hIoU']:.1f} |",
        f"| Re-implementation | {s:.1f} | {u:.1f} | {h:.1f} |",
        f"| Difference | {s - paper['seen']:+.1f} | {u - paper['unseen']:+.1f} | {h - paper['hIoU']:+.1f} |",
        "",
        f"Frames: seen {seen['frames']}, unseen {unseen['frames']}.",
        "",
        "Per-affordance IoU (%):",
        "",
        "| | " + " | ".join(NAMES) + " |",
        "|---|" + "---:|" * len(NAMES),
    ]
    for name, summary in (("seen", seen), ("unseen", unseen)):
        iou = summary["metrics"]["iou"]
        lines.append(f"| {name} | " + " | ".join(f"{100 * iou[n]:.1f}" for n in NAMES) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seen", required=True)
    p.add_argument("--unseen", required=True)
    p.add_argument("--out", default="")
    args = p.parse_args()
    text = build(load(args.seen), load(args.unseen))
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)


if __name__ == "__main__":
    main()
