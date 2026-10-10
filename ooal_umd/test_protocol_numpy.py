"""CPU checks without torch:  DRAWAFFORD_ROOT=... python -m unittest ooal_umd.test_protocol_numpy -v"""

from __future__ import annotations

import os
import unittest

import numpy as np

from ooal_umd import umd_protocol as P
from ooal_umd import viz_umd as V


def needs_drawafford(test):
    return unittest.skipUnless(os.environ.get("DRAWAFFORD_ROOT"), "set DRAWAFFORD_ROOT")(test)


class ProtocolTest(unittest.TestCase):

    @needs_drawafford
    def test_class_order_matches_umd_eval(self):
        ev = P.umd_eval()   # raises if the order differs
        self.assertEqual(ev.N_CLASSES, len(P.OOAL_CLASS_NAMES))
        self.assertEqual(ev.THRESHOLD, 0.5)

    def test_one_hot_round_trip(self):
        rng = np.random.default_rng(0)
        label = rng.integers(0, 8, size=(31, 47))
        oh = P.one_hot(label)
        self.assertEqual(oh.shape, (7, 31, 47))
        self.assertEqual(set(np.unique(oh)), {0.0, 255.0})
        self.assertTrue((oh.sum(0) <= 255).all())          # one affordance per pixel
        back = np.where(oh.max(0) > 0, oh.argmax(0) + 1, 0)
        np.testing.assert_array_equal(back, label)

    def test_one_hot_rejects_bad_ids(self):
        with self.assertRaises(ValueError):
            P.one_hot(np.array([[0, 8]]))

    @needs_drawafford
    def test_probs_to_label_same_size_is_exact(self):
        rng = np.random.default_rng(1)
        label = rng.integers(0, 8, size=(28, 42))
        probs = P.one_hot(label) / 255.0 * 0.9 + 0.05       # fg 0.95, rest 0.05
        np.testing.assert_array_equal(P.probs_to_label(probs, label.shape), label)

    @needs_drawafford
    def test_threshold_is_strict(self):
        probs = np.zeros((7, 14, 14), np.float32)
        probs[2] = 0.5                                       # exactly 0.5 -> background
        probs[2, :7] = 0.51
        out = P.probs_to_label(probs, (14, 14))
        self.assertTrue((out[:7] == 3).all())
        self.assertTrue((out[7:] == 0).all())

    @needs_drawafford
    def test_resize_to_native_frame(self):
        probs = np.zeros((7, 476, 644), np.float32)
        probs[0, :, :300] = 0.9                              # grasp on the left
        probs[3, :, 344:] = 0.9                              # contain on the right
        out = P.probs_to_label(probs, (480, 640))
        self.assertEqual(out.shape, (480, 640))
        self.assertTrue((out[:, :290] == 1).all())
        self.assertTrue((out[:, 350:] == 4).all())
        self.assertTrue((out[:, 305:335] == 0).all())

    def test_hiou(self):
        self.assertAlmostEqual(P.hiou(0.6, 0.6), 0.6)
        self.assertEqual(P.hiou(0.0, 0.7), 0.0)
        # OOAL Table 2 rounds from unrounded seen/unseen values: 66.3 vs printed 66.4.
        self.assertAlmostEqual(100 * P.hiou(0.746, 0.597), 66.32, places=1)


class CompareTest(unittest.TestCase):

    @staticmethod
    def cm(gt, pred):
        out = np.zeros((8, 8), np.int64)
        np.add.at(out, (np.asarray(gt).ravel(), np.asarray(pred).ravel()), 1)
        return out

    def test_frame_metrics(self):
        from ooal_umd import compare_umd as C
        gt = np.array([0, 0, 1, 1, 1, 1, 2, 2])
        pred = np.array([0, 1, 1, 1, 0, 2, 2, 2])   # 1 spill, 1 miss, 1 grasp->cut
        cm = self.cm(gt, pred)[None]
        # grasp: tp 2, gt 4, pred 3 -> 2/5; cut: tp 2, gt 2, pred 3 -> 2/3
        self.assertAlmostEqual(float(C.frame_miou(cm)[0]), (2 / 5 + 2 / 3) / 2)
        np.testing.assert_allclose(C.error_fractions(cm)[0], [1 / 6, 1 / 6, 1 / 6])
        self.assertEqual(C.top_wrong_pair(cm), ["grasp→cut"])

    def test_spurious_class_counts_as_zero(self):
        from ooal_umd import compare_umd as C
        cm = self.cm([0, 1, 1], [7, 1, 1])[None]          # wrap-grasp not in GT
        self.assertAlmostEqual(float(C.frame_miou(cm)[0]), 0.5)
        self.assertTrue(np.isnan(C.frame_miou(self.cm([0, 0], [0, 0])[None])[0]))

    @needs_drawafford
    def test_split_miou_matches_umd_eval(self):
        from ooal_umd import compare_umd as C
        rng = np.random.default_rng(3)
        cm = rng.integers(0, 50, size=(8, 8))
        self.assertAlmostEqual(C.split_miou(cm), P.umd_eval().scores(cm)["mIoU_7fg"])

    def test_align_and_one_per_instance(self):
        from ooal_umd import compare_umd as C
        ids = np.array(["a_1", "a_2", "b_1"])
        obj = np.array(["a", "a", "b"])
        cms = np.arange(3)[:, None, None] * np.ones((3, 8, 8), np.int64)
        _, _, cm_a, cm_b = C.align((ids, obj, cms), (ids[::-1], obj[::-1], cms[::-1]))
        np.testing.assert_array_equal(cm_a, cm_b)
        with self.assertRaises(SystemExit):
            C.align((ids, obj, cms), (ids[:2], obj[:2], cms[:2]))
        self.assertEqual(C.one_per_instance([0, 1, 2], obj, 5), [0, 2])


class VizTest(unittest.TestCase):

    def test_overlay_keeps_background(self):
        rgb = np.full((10, 12, 3), 100, np.uint8)
        label = np.zeros((10, 12), np.int64)
        label[2:4, 3:5] = 2
        out = V.overlay(rgb, label, 0.6)
        np.testing.assert_array_equal(out[label == 0], rgb[label == 0])
        expected = np.round(0.4 * 100 + 0.6 * V.PALETTE[1].astype(float)).astype(np.uint8)
        np.testing.assert_array_equal(out[2, 3], expected)

    def test_crop_box_square_inside_and_covers_object(self):
        label = np.zeros((480, 640), np.int64)
        label[400:470, 600:635] = 1                          # near the corner
        y0, y1, x0, x1 = V.crop_box(label, 0.25)
        self.assertEqual(y1 - y0, x1 - x0)
        self.assertTrue(0 <= y0 < y1 <= 480 and 0 <= x0 < x1 <= 640)
        self.assertTrue(y0 <= 400 and y1 >= 470 and x0 <= 600 and x1 >= 635)
        self.assertEqual(V.crop_box(np.zeros((480, 640)), 0.25), (0, 480, 0, 640))

    def test_pick_ids_fixed_seed_per_category(self):
        class T:
            def __init__(self, i, o):
                self.id, self.object_name = i, o
        targets = [T(f"{o}_{k:03d}", o) for o in ("knife_01", "knife_02", "cup_03") for k in range(20)]
        a = V.pick_ids(targets, 2, seed=0)
        self.assertEqual(a, V.pick_ids(list(reversed(targets)), 2, seed=0))
        self.assertEqual(len(a), 4)                           # 2 families x 2
        self.assertEqual([x.split("_")[0] for x in a], ["cup", "cup", "knife", "knife"])
        # a category's frames do not depend on which other categories are shown
        self.assertEqual(V.pick_ids(targets, 2, seed=0, categories=["knife"]), a[2:])
        self.assertNotEqual(V.pick_ids(targets, 2, seed=1), a)
        with self.assertRaises(SystemExit):
            V.pick_ids(targets, 1, seed=0, categories=["saw"])


if __name__ == "__main__":
    unittest.main()
