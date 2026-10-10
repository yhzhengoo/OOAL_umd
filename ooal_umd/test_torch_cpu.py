"""CPU checks that need torch (run on the server, from the OOAL repo root):

    DRAWAFFORD_ROOT=... python -m unittest ooal_umd.test_torch_cpu -v

No DINOv2 / CLIP weights are downloaded: the backbone, prompt learner and
text encoder are replaced by small random modules, while the parts OOAL
trains (lln_*, embedder, SegDecoder, linear_cls) are the real classes.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from models.ooal import Mlp
from models.seg_decoder import SegDecoder
from ooal_umd import umd_protocol as P
from ooal_umd.model_umd import OOALUMD, check_forward_all
from ooal_umd.local_backbones import local_backbones


class StubDino(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, 14, 14)
        self.blocks = nn.ModuleList([nn.Linear(dim, dim) for _ in range(3)])

    def get_intermediate_layers(self, img, n=3, return_class_token=True):
        x = self.patch(img).flatten(2).transpose(1, 2)
        outs = []
        for blk in self.blocks:
            x = torch.tanh(blk(x))
            outs.append((x, x.mean(1)))
        return outs


class StubPrompt(nn.Module):
    def __init__(self, n_cls, length, dim):
        super().__init__()
        self.p = nn.Parameter(torch.randn(n_cls, length, dim))

    def forward(self):
        return self.p


class StubText(nn.Module):
    def __init__(self, dim, out):
        super().__init__()
        self.proj = nn.Linear(dim, out)

    def forward(self, prompts, tokenized_prompts):
        return self.proj(prompts.mean(1))


def make_stub(n_cls=7, input_dim=48, out_dim=32) -> OOALUMD:
    torch.manual_seed(0)
    m = OOALUMD.__new__(OOALUMD)
    nn.Module.__init__(m)
    m.dino_model = StubDino(input_dim)
    m.lln_linear = nn.ModuleList([nn.Linear(input_dim, input_dim) for _ in range(3)])
    m.lln_norm = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(3)])
    m.merge_weight = nn.Parameter(torch.randn(3))
    m.embedder = Mlp(in_features=input_dim, hidden_features=out_dim, out_features=out_dim)
    m.lln_norm_1 = nn.LayerNorm(out_dim)
    m.lln_norm_2 = nn.LayerNorm(out_dim)
    m.prompt_learner = StubPrompt(n_cls, 5, 16)
    m.tokenized_prompts = torch.zeros(n_cls, 5, dtype=torch.long)
    m.aff_text_encoder = StubText(16, out_dim)
    m.linear_cls = nn.Linear(input_dim, out_dim)
    m.seg_decoder = SegDecoder(embed_dims=out_dim, num_layers=2)
    return m


class CudaAsIdentity:
    """Net.forward allocates its eval output with ``.cuda()``; keep it on CPU."""

    def __enter__(self):
        self.saved = torch.Tensor.cuda
        torch.Tensor.cuda = lambda self, *a, **k: self

    def __exit__(self, *exc):
        torch.Tensor.cuda = self.saved


class ModelTest(unittest.TestCase):

    def test_forward_all_matches_net_forward(self):
        model = make_stub().eval()
        img = torch.randn(2, 3, 476, 644)
        with CudaAsIdentity():
            diff = check_forward_all(model, img)
        self.assertLess(diff, 1e-5)
        self.assertEqual(tuple(model.forward_all(img).shape), (2, 7, 476, 644))

    def test_training_loss_is_bce_on_one_hot(self):
        model = make_stub().train()
        label = np.zeros((420, 560), np.int64)
        label[100:200, 100:300] = 1
        label[200:260, 300:400] = 7
        ann = torch.from_numpy(P.one_hot(label))[None]
        pred, loss = model(torch.randn(1, 3, 420, 560), label=ann)
        self.assertEqual(tuple(pred.shape), (1, 7, 420, 560))
        ref = nn.BCELoss()(pred, ann / 255.0)
        self.assertAlmostEqual(float(loss["bce"]), float(ref), places=6)
        loss["bce"].backward()
        self.assertIsNotNone(model.seg_decoder.decoder.layers[0].multihead_attn.q.weight.grad)


class LocalBackboneTest(unittest.TestCase):

    def test_local_dino_is_offline_and_strict_and_restores_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "hubconf.py").touch()
            source = nn.Linear(3, 4)
            checkpoint = root / "weights.pth"
            torch.save(source.state_dict(), checkpoint)
            args = SimpleNamespace(dino_repo=str(root), dino_weights=str(checkpoint), clip_weights=None)
            original = torch.hub.load
            with patch("torch.hub.load", return_value=nn.Linear(3, 4)) as native_loader:
                with local_backbones(args):
                    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
                self.assertIs(torch.hub.load, native_loader)
                native_loader.assert_called_once_with(str(root), "dinov2_vitb14", source="local", pretrained=False)
                self.assertTrue(torch.equal(model.weight, source.weight))
            self.assertIs(torch.hub.load, original)
            torch.save({"wrong_key": torch.zeros(1)}, checkpoint)
            with patch("torch.hub.load", return_value=nn.Linear(3, 4)):
                with local_backbones(args), self.assertRaises(RuntimeError):
                    torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
            self.assertIs(torch.hub.load, original)

    def test_local_dino_requires_both_paths(self):
        args = SimpleNamespace(dino_repo="/unused", dino_weights=None, clip_weights=None)
        with self.assertRaises(ValueError):
            with local_backbones(args):
                pass


@unittest.skipUnless(os.environ.get("DRAWAFFORD_ROOT"), "set DRAWAFFORD_ROOT")
class DataTest(unittest.TestCase):

    def setUp(self):
        from ooal_umd.data_umd import UMDTest, UMDTrain
        self.UMDTrain, self.UMDTest = UMDTrain, UMDTest
        ev = P.umd_eval()
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        rgb = (np.random.default_rng(0).random((480, 640, 3)) * 255).astype(np.uint8)
        label = np.zeros((480, 640), np.uint8)
        label[200:300, 250:330] = 1
        label[200:300, 330:400] = 4
        Image.fromarray(rgb).save(root / "ladle_01_00000001_rgb.jpg")
        Image.fromarray(label).save(root / "ladle_01_00000001_label.png")
        self.item = ev.Item("ladle_01_00000001_rgb", "ladle_01_00000001",
                            root / "ladle_01_00000001_rgb.jpg",
                            root / "ladle_01_00000001_label.png", "exemplar")

    def tearDown(self):
        self.tmp.cleanup()

    def test_train_sample(self):
        img, ann = self.UMDTrain([self.item])[0]
        self.assertEqual(tuple(img.shape), (3, 420, 560))
        self.assertEqual(tuple(ann.shape), (7, 420, 560))
        self.assertGreaterEqual(float(ann.min()), 0.0)
        self.assertLessEqual(float(ann.max()), 255.0)
        # ladle: contain (4) is remapped to scoop (3) by umd_eval.load_label
        self.assertEqual(float(ann[3].max()), 0.0)
        self.assertGreater(float(ann[2].max()), 0.0)

    def test_test_sample(self):
        img, label, index = self.UMDTest([self.item])[0]
        self.assertEqual(tuple(img.shape), (3, 476, 644))
        self.assertEqual(tuple(label.shape), (480, 640))
        self.assertEqual(sorted(np.unique(label.numpy()).tolist()), [0, 1, 3])
        self.assertEqual(index, 0)

    def test_sizes_must_be_multiples_of_14(self):
        with self.assertRaises(ValueError):
            self.UMDTrain([self.item], resize_hw=(480, 640))


if __name__ == "__main__":
    unittest.main()
