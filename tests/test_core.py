"""Run with:  python -m unittest discover -s tests -v
Torch tests are skipped automatically when torch/transformers are not installed."""
import unittest

import numpy as np
from PIL import Image

from artistid import rejection as rej
from artistid.config import apply_overrides
from artistid.features import HOG_DIM, HSV_LBP_DIM, handcrafted
from artistid.images import augment, letterbox, multi_views
from artistid.metrics import classification_metrics, mcnemar_exact

try:
    import torch
    from transformers import Dinov2Config, Dinov2Model
    HAVE_TORCH = True
except Exception:  # pragma: no cover
    HAVE_TORCH = False


def fake_painting(w=300, h=200, seed=0):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))


class TestImages(unittest.TestCase):
    def test_letterbox_square(self):
        self.assertEqual(letterbox(fake_painting(), 224).size, (224, 224))

    def test_multiview_shape(self):
        v = multi_views(fake_painting(), 224, 448, 2)
        self.assertEqual(v.shape, (5, 3, 224, 224))
        self.assertEqual(multi_views(fake_painting(), 224, 448, 0).shape, (1, 3, 224, 224))

    def test_augment_keeps_valid_image(self):
        out = augment(fake_painting(), np.random.default_rng(1))
        self.assertGreater(out.size[0], 0)

    def test_handcrafted_dims(self):
        hsv, hog = handcrafted(fake_painting(500, 380))
        self.assertEqual(hsv.shape, (HSV_LBP_DIM,))
        self.assertEqual(hog.shape, (HOG_DIM,))
        self.assertAlmostEqual(float(hsv[:16].sum()), 1.0, places=4)


class TestRejection(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.C, self.D = 4, 16
        centers = rng.normal(size=(self.C, self.D)) * 3
        self.feats = np.concatenate([centers[c] + rng.normal(size=(60, self.D)) for c in range(self.C)])
        self.y = np.repeat(np.arange(self.C), 60)
        self.logits = np.eye(self.C)[self.y] * 4 + rng.normal(size=(len(self.y), self.C))
        self.unk_feats = rng.normal(size=(200, self.D)) * 3
        self.unk_logits = rng.normal(size=(200, self.C))

    def test_prototypes_and_gate(self):
        protos = rej.fit_prototypes(self.feats, self.y, self.C, k=3)
        self.assertEqual(protos.shape, (self.C, 3, self.D))
        g_known = rej.gate_scores(self.logits, self.feats, protos, 1.0, 0.5)
        g_unk = rej.gate_scores(self.unk_logits, self.unk_feats, protos, 1.0, 0.5)
        self.assertTrue(((g_known >= 0) & (g_known <= 1)).all())
        self.assertGreater(g_known.mean(), g_unk.mean())

    def test_threshold_respects_far(self):
        s = np.random.default_rng(2).uniform(size=1000)
        thr = rej.threshold_for_far(s, 0.05)
        self.assertLessEqual(float((s > thr).mean()), 0.05)
        self.assertGreater(float((s > thr).mean()), 0.03)

    def test_select_mix_and_metrics(self):
        protos = rej.fit_prototypes(self.feats, self.y, self.C, k=3)
        sel = rej.select_mix(self.logits, self.feats, self.unk_logits, self.unk_feats, protos, 1.0,
                             [0.3, 0.5, 0.7], 0.05)
        self.assertIn(sel["mix"], [0.3, 0.5, 0.7])
        ks = rej.gate_scores(self.logits, self.feats, protos, 1.0, sel["mix"])
        us = rej.gate_scores(self.unk_logits, self.unk_feats, protos, 1.0, sel["mix"])
        m = rej.rejection_metrics(ks, self.logits.argmax(1) == self.y, us, sel["threshold_dev"])
        self.assertLessEqual(m["unknown_false_acceptance"], 0.05)
        self.assertGreater(m["auroc"], 0.5)

    def test_temperature_grid(self):
        self.assertIn(rej.fit_temperature(self.logits, self.y, [0.5, 1.0, 2.0]), [0.5, 1.0, 2.0])


class TestMetrics(unittest.TestCase):
    def test_classification(self):
        y = np.array([0, 1, 2, 1])
        logits = np.eye(3)[[0, 1, 2, 0]] * 5.0
        m = classification_metrics(logits, y)
        self.assertAlmostEqual(m["accuracy"], 0.75)
        self.assertAlmostEqual(m["top5"], 1.0)

    def test_mcnemar(self):
        a = np.array([True] * 30 + [False] * 5)
        b = np.array([False] * 30 + [True] * 5)
        r = mcnemar_exact(a, b)
        self.assertEqual((r["a_only"], r["b_only"]), (30, 5))
        self.assertLess(r["p_value"], 0.001)

    def test_overrides(self):
        cfg = {"views": {"local_grid": 2}}
        apply_overrides(cfg, ["views.local_grid=0"])
        self.assertEqual(cfg["views"]["local_grid"], 0)
        with self.assertRaises(KeyError):
            apply_overrides(cfg, ["views.nope=1"])


@unittest.skipUnless(HAVE_TORCH, "torch/transformers not installed")
class TestModel(unittest.TestCase):
    def test_forward_and_trainable_state(self):
        from artistid.model import ArtistNet, trainable_state
        enc = Dinov2Model(Dinov2Config(hidden_size=32, num_hidden_layers=3, num_attention_heads=2,
                                       intermediate_size=64, image_size=224, patch_size=14))
        net = ArtistNet(5, trainable_blocks=2, hand_weight=0.25, dropout=0.1, encoder=enc)
        views = torch.randn(2, 5, 3, 224, 224)
        logits, fused = net(views, torch.rand(2, HSV_LBP_DIM), torch.rand(2, HOG_DIM), return_features=True)
        self.assertEqual(tuple(logits.shape), (2, 5))
        self.assertEqual(fused.shape[1], 32 + HSV_LBP_DIM + HOG_DIM)
        logits.sum().backward()
        state = trainable_state(net)
        self.assertTrue(any(k.startswith("head") for k in state))
        self.assertFalse(any("encoder.encoder.layer.0." in k for k in state))  # block 0 stays frozen
        self.assertTrue(any("encoder.encoder.layer.2." in k for k in state))  # last block trainable


if __name__ == "__main__":
    unittest.main()
