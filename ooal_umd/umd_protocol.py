"""UMD protocol for the OOAL re-implementation (no torch here).

Which frames are used and how they are scored is *imported* from DrawAfford's
``affordance_research/umd_eval/umd_eval.py``, so OOAL and DrawAfford are
evaluated by the same code:

* references  : ``verify_inventory(split)[0]`` -> 54 (seen) / 76 (unseen)
                one-shot images from ``train_base_one_pick`` / ``train_novel_one_pick``
* test frames : ``verify_inventory(split)[1]`` -> 14,020 / 7,969 frames
* labels      : ``load_label`` (UMD ids 0..7, ladle contain -> scoop)
* prediction  : ``predict`` (bilinear resize to the label size, background
                where the best channel <= 0.5, else argmax + 1)
* metric      : ``confusion`` + ``scores`` (8x8 confusion matrix over all
                pixels, mIoU over the 7 affordances, background excluded)

Set ``DRAWAFFORD_ROOT`` to the ConceptAttention or standalone DrawAfford repo.
Data paths can be overridden with UMD_DATASET_ROOT / UMD_BASE_REF_ROOT /
UMD_NOVEL_REF_ROOT without editing the shared evaluator.
"""

from __future__ import annotations

import os
import sys
import hashlib
import importlib
from functools import lru_cache
from pathlib import Path

import numpy as np

# OOAL class names in UMD label order (id 1..7).  OOAL's PromptLearner turns
# "_" into a space, so the text prompt for id 7 is "wrap grasp".
OOAL_CLASS_NAMES = (
    "grasp", "cut", "scoop", "contain", "pound", "support", "wrap_grasp",
)

# Mean IoU reported in OOAL's Table 2 (CVPR 2024), in percent.
OOAL_PAPER = {"seen": 74.6, "unseen": 59.7, "hIoU": 66.4}


@lru_cache(maxsize=1)
def umd_eval():
    """Import DrawAfford's ``umd_eval`` module (numpy / scipy / cv2 only)."""
    root = os.environ.get("DRAWAFFORD_ROOT")
    if not root:
        raise RuntimeError(
            "Set DRAWAFFORD_ROOT to the ConceptAttention or standalone DrawAfford repo root."
        )
    root_path = Path(root).expanduser().resolve()
    candidates = ("affordance_research.umd_eval.umd_eval", "umd_eval.umd_eval")
    module_name = next((name for name in candidates
                        if (root_path / Path(*name.split("."))).with_suffix(".py").is_file()), None)
    if module_name is None:
        raise FileNotFoundError(
            f"{root_path}: neither affordance_research/umd_eval/umd_eval.py nor umd_eval/umd_eval.py found"
        )
    if str(root_path) not in sys.path:
        sys.path.insert(0, str(root_path))
    ev = importlib.import_module(module_name)
    expected_file = (root_path / Path(*module_name.split("."))).with_suffix(".py")
    if Path(ev.__file__).resolve() != expected_file.resolve():
        raise RuntimeError(f"imported {ev.__file__}, expected {expected_file}; restart with the correct DRAWAFFORD_ROOT")
    for env_name, attr in (("UMD_DATASET_ROOT", "DATASET"),
                           ("UMD_BASE_REF_ROOT", "BASE_REF"),
                           ("UMD_NOVEL_REF_ROOT", "NOVEL_REF")):
        if os.environ.get(env_name):
            setattr(ev, attr, Path(os.environ[env_name]).expanduser().resolve())

    names = tuple(n.replace("-", "_") for n in ev.AFFORDANCE_NAMES)
    if names != OOAL_CLASS_NAMES:
        raise RuntimeError(f"class order differs from umd_eval: {names}")
    if ev.N_CLASSES != 7 or ev.THRESHOLD != 0.5:
        raise RuntimeError("the fixed UMD protocol requires 7 foreground classes and threshold 0.5")
    return ev


def metadata() -> dict:
    """Record the actual shared evaluator and data roots in experiment outputs."""
    ev = umd_eval()
    source = Path(ev.__file__).resolve()
    return {"evaluator": str(source),
            "evaluator_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "dataset_root": str(ev.DATASET), "base_ref_root": str(ev.BASE_REF),
            "novel_ref_root": str(ev.NOVEL_REF), "threshold": float(ev.THRESHOLD),
            "affordance_names": list(ev.AFFORDANCE_NAMES)}


def inventory(split: str):
    """(references, test frames) exactly as DrawAfford's evaluator uses them."""
    return umd_eval().verify_inventory(split)


def load_label(item) -> np.ndarray:
    return umd_eval().load_label(item.label, item.object_name)


def one_hot(label: np.ndarray) -> np.ndarray:
    """UMD label map (H, W) with ids 0..7 -> (7, H, W) float32 in {0, 255}.

    OOAL's training branch computes ``BCELoss(pred, label / 255)``; feeding
    0/255 one-hot maps keeps ``models/ooal.py`` unchanged.  Background pixels
    are 0 in every channel.
    """
    label = np.asarray(label)
    n = len(OOAL_CLASS_NAMES)
    if label.ndim != 2 or label.min(initial=0) < 0 or label.max(initial=0) > n:
        raise ValueError(f"expected a 2-D label map with ids 0..{n}")
    ids = np.arange(1, n + 1, dtype=label.dtype).reshape(n, 1, 1)
    return (label[None] == ids).astype(np.float32) * 255.0


def probs_to_label(probs_chw: np.ndarray, shape_hw: tuple[int, int]) -> np.ndarray:
    """OOAL sigmoid maps (7, h, w) -> UMD label map (H, W) via ``umd_eval.predict``.

    ``predict`` resizes (bilinear) to the label size and applies the same
    background rule as DrawAfford (best channel > 0.5 -> argmax + 1, else 0).
    The 0.5 threshold is also the decision boundary of OOAL's BCE training.
    """
    ev = umd_eval()
    c, h, w = probs_chw.shape
    if c != ev.N_CLASSES:
        raise ValueError(f"expected {ev.N_CLASSES} channels, got {c}")
    tokens = np.ascontiguousarray(probs_chw.transpose(1, 2, 0)).reshape(h * w, c)
    return ev.predict(tokens.astype(np.float32), (h, w), tuple(shape_hw))


def hiou(seen: float, unseen: float) -> float:
    return 0.0 if seen + unseen <= 0 else 2.0 * seen * unseen / (seen + unseen)
