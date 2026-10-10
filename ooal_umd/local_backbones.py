"""Load official local weights while leaving models/ooal.py unchanged.

The patches exist only during Net construction. DINO uses its native hub
implementation and strict state loading; CLIP uses OOAL's own build_model.
No Hugging Face adapter or alternative forward implementation is involved.
"""

from __future__ import annotations

import os
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import torch


def add_arguments(parser) -> None:
    group = parser.add_argument_group("official local backbones (optional)")
    group.add_argument("--dino_repo", default=os.environ.get("OOAL_DINO_REPO"),
                       help="native DINOv2 repo with hubconf.py")
    group.add_argument("--dino_weights", default=os.environ.get("OOAL_DINO_WEIGHTS"),
                       help="official dinov2_vitb14_pretrain.pth (not Hugging Face format)")
    group.add_argument("--clip_weights", default=os.environ.get("OOAL_CLIP_WEIGHTS"),
                       help="official OpenAI ViT-B-16.pt")


def _file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


@contextmanager
def local_backbones(args):
    from models import ooal

    repo = getattr(args, "dino_repo", None)
    weights = getattr(args, "dino_weights", None)
    clip_weights = getattr(args, "clip_weights", None)
    if bool(repo) != bool(weights):
        raise ValueError("provide both --dino_repo and --dino_weights (or both OOAL_DINO_* variables)")

    original_load = torch.hub.load
    with ExitStack() as stack:
        if repo:
            repo_path = _file(str(Path(repo).expanduser() / "hubconf.py")).parent
            weights_path = _file(weights)

            def load_dino(github, model_name, *a, **kw):
                if github != "facebookresearch/dinov2":
                    return original_load(github, model_name, *a, **kw)
                if model_name != "dinov2_vitb14":
                    raise ValueError(f"UMD reproduction expects dinov2_vitb14, got {model_name}")
                model = original_load(str(repo_path), model_name, source="local", pretrained=False)
                state = torch.load(weights_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state, strict=True)
                print(f"Loaded native DINOv2 ViT-B/14: {repo_path} / {weights_path}", flush=True)
                return model

            stack.enter_context(patch.object(torch.hub, "load", side_effect=load_dino))
        if clip_weights:
            clip_path = _file(clip_weights)

            def load_clip(name):
                if name != "ViT-B/16":
                    raise ValueError(f"UMD reproduction expects ViT-B/16, got {name}")
                try:
                    state = torch.jit.load(str(clip_path), map_location="cpu").eval().state_dict()
                except RuntimeError:
                    state = torch.load(clip_path, map_location="cpu", weights_only=True)
                model = ooal.clip.build_model(state)
                print(f"Loaded OpenAI CLIP ViT-B/16: {clip_path}", flush=True)
                return model

            stack.enter_context(patch.object(ooal, "load_clip_to_cpu", side_effect=load_clip))
        yield
