"""UMD datasets for OOAL.

Augmentation is OOAL's own ``data/agd20k_ego.TrainData.transform`` (resize ->
random crop -> horizontal flip -> ImageNet normalisation).  Only the sizes
change: AGD20K uses 256 -> 224 squares; UMD frames are 640x480 with small
objects, so we keep the native resolution, rounded to multiples of DINOv2's
14-pixel patch (476x644), and crop with the same ratio (224/256 = 0.875 ->
420x560).  Test frames are resized to 476x644, the training scale.
"""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms
import torchvision.transforms.functional as TF

from ooal_umd import umd_protocol as P

RESIZE_HW = (476, 644)   # (H, W), multiples of 14, ~native 480x640
CROP_HW = (420, 560)     # 0.875 of RESIZE_HW, multiples of 14
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def check_hw(hw: Sequence[int], name: str) -> tuple[int, int]:
    h, w = int(hw[0]), int(hw[1])
    if h % 14 or w % 14:
        raise ValueError(f"{name}={h}x{w}: OOAL reshapes DINOv2 tokens to (H/14, W/14); use multiples of 14")
    return h, w


class UMDTrain(data.Dataset):
    """The one-shot references (54 seen / 76 unseen) with one-hot 0/255 masks."""

    def __init__(self, items: Sequence, resize_hw=RESIZE_HW, crop_hw=CROP_HW):
        self.items = list(items)
        self.resize_hw = check_hw(resize_hw, "resize_hw")
        self.crop_hw = check_hw(crop_hw, "crop_hw")
        if self.crop_hw[0] > self.resize_hw[0] or self.crop_hw[1] > self.resize_hw[1]:
            raise ValueError("crop must fit inside the resized image")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        img = Image.open(item.image).convert("RGB")
        ann = torch.from_numpy(P.one_hot(P.load_label(item)))
        return self.transform(img, ann)

    def transform(self, img: Image.Image, mask: torch.Tensor):
        # Same steps as OOAL's TrainData.transform.  antialias: PIL images are
        # always antialiased; the mask tensor keeps the old torchvision
        # behaviour of ``antialias=None`` (no antialias for tensors).
        img = TF.resize(img, list(self.resize_hw))
        mask = TF.resize(mask, list(self.resize_hw), antialias=False)

        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=self.crop_hw)
        img = TF.crop(img, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)

        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=MEAN, std=STD)
        return img, mask


class UMDTest(data.Dataset):
    """All test frames of a split; returns (image, uint8 label, index)."""

    def __init__(self, items: Sequence, resize_hw=RESIZE_HW):
        self.items = list(items)
        self.resize_hw = check_hw(resize_hw, "resize_hw")
        self.transform = transforms.Compose([
            transforms.Resize(self.resize_hw),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        item = self.items[index]
        img = self.transform(Image.open(item.image).convert("RGB"))
        label = P.load_label(item).astype(np.uint8)
        return img, torch.from_numpy(label), index
