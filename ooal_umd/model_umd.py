"""OOAL model for UMD: ``models/ooal.py`` unchanged, plus one read-out method.

At test time ``Net.forward(img, gt_aff=k)`` returns only channel ``k`` (the
AGD20K protocol queries one affordance per image).  UMD needs all seven
channels to build a label map, so ``forward_all`` repeats the eval-time
computation of ``Net.forward`` and stops before the channel pick.
``check_forward_all`` verifies on a real frame that both paths agree.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.ooal import Net
from ooal_umd.local_backbones import local_backbones


class OOALUMD(Net):

    def __init__(self, args, input_dim, out_dim, dino_pretrained="dinov2_vitb14"):
        with local_backbones(args):
            super().__init__(args, input_dim, out_dim, dino_pretrained)

    @torch.no_grad()
    def forward_all(self, img: torch.Tensor) -> torch.Tensor:
        """img (B, 3, H, W) -> sigmoid maps (B, num_aff, H, W)."""
        b, _, h, w = img.shape

        # --- identical to Net.forward up to ``pred`` ---
        dino_out = self.dino_model.get_intermediate_layers(img, n=3, return_class_token=True)
        merge_weight = torch.softmax(self.merge_weight, dim=0)

        dino_dense = 0
        for i, feat in enumerate(dino_out):
            feat_ = self.lln_linear[i](feat[0])
            feat_ = self.lln_norm[i](feat_)
            dino_dense += feat_ * merge_weight[i]
        dino_dense = self.lln_norm_1(self.embedder(dino_dense))

        prompts = self.prompt_learner()
        text_features = self.lln_norm_2(self.aff_text_encoder(prompts, self.tokenized_prompts))

        dino_cls = self.linear_cls(dino_out[-1][1])
        text_features = text_features.unsqueeze(0).expand(b, -1, -1)
        text_features, _, _ = self.seg_decoder(text_features, dino_dense, extra_token=dino_cls)

        attn = (text_features[-1] @ dino_dense.transpose(-2, -1)) * (512 ** -0.5)
        attn_out = torch.sigmoid(attn).reshape(b, -1, h // 14, w // 14)
        return F.interpolate(attn_out, size=img.shape[-2:], mode="bilinear", align_corners=False)


@torch.no_grad()
def check_forward_all(model: OOALUMD, img: torch.Tensor, atol: float = 1e-5) -> float:
    """Max |forward_all - Net.forward(gt_aff=k)| over all channels; raises if > atol."""
    if model.training:
        raise RuntimeError("call model.eval() first")
    all_maps = model.forward_all(img)
    worst = 0.0
    for k in range(all_maps.shape[1]):
        gt_aff = torch.full((img.shape[0],), k, dtype=torch.long)
        ref = Net.forward(model, img, gt_aff=gt_aff)
        worst = max(worst, float((ref.to(all_maps.device) - all_maps[:, k]).abs().max()))
    if worst > atol:
        raise RuntimeError(f"forward_all differs from Net.forward by {worst:.3g} (> {atol})")
    return worst
