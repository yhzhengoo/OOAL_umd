"""Read-only dataset / cached comparison check before running on a new server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ooal_umd import umd_protocol as P


def check_cached(path: Path, targets, key: str) -> dict:
    with np.load(path, allow_pickle=False) as data:
        ids, objects, cm = data["ids"].astype(str), data["objects"].astype(str), data[key]
        expected = {t.id: t.object_name for t in targets}
        if len(ids) != len(expected) or len(set(ids)) != len(ids) or set(ids) != set(expected):
            raise ValueError(f"{path}: frame IDs do not match the full UMD split")
        if objects.shape != ids.shape or any(expected[i] != o for i, o in zip(ids, objects)):
            raise ValueError(f"{path}: object names do not match frame IDs")
        if (cm.shape != (len(ids), 8, 8) or not np.issubdtype(cm.dtype, np.integer)
                or np.any(cm < 0) or np.any(cm.sum((1, 2)) != 480 * 640)):
            raise ValueError(f"{path}: expected nonnegative integer confusion matrices on 480x640 frames")
        return {"path": str(path.resolve()), "key": key, "frames": len(ids),
                "metrics": P.umd_eval().scores(cm.sum(0))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours_root", type=Path,
                        help="reframe_eval directory containing seen/unseen/per_frame.npz")
    parser.add_argument("--ours_key", default="cm__reframe_bg")
    args = parser.parse_args()
    result = {"protocol": P.metadata(), "splits": {}}
    for split in ("seen", "unseen"):
        refs, targets = P.inventory(split)
        labels = sorted({int(c) for item in refs for c in np.unique(P.load_label(item))})
        info = {"references": len(refs), "targets": len(targets), "reference_labels": labels}
        if args.ours_root:
            info["ours"] = check_cached(args.ours_root / split / "per_frame.npz", targets, args.ours_key)
        result["splits"][split] = info
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
