"""Unit tests for NULL-0 / NULL-1 null model."""

from __future__ import annotations

import math
import unittest

import numpy as np

from null_model import E_GAMMA, NullModel, NullModelCalibrated


class NullModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = NullModel()

    def test_ineligible_offsets_are_zero(self) -> None:
        # N=6: p=3 -> m=3 (eligible); p=5 -> m=1 (ineligible)
        probs = self.model.predict_proba(6)
        self.assertAlmostEqual(probs[0], 1.0)  # m=3 is base prime r_1
        self.assertEqual(probs[1], 0.0)  # m=1 < 3

    def test_small_prime_offset_is_certain(self) -> None:
        # N=10, p=7 -> m=3
        probs = self.model.predict_proba(10)
        prime_index = 2  # 3,5,7 -> index 2 is 7
        self.assertAlmostEqual(probs[prime_index], 1.0)

    def test_composite_hit_gets_eps(self) -> None:
        # N=16, p=3 -> m=13 (survivor for M=32)
        # N=16, p=5 -> m=11 (survivor)
        # N=16, p=7 -> m=9, divisible by 3 -> eps
        probs = self.model.predict_proba(16)
        self.assertAlmostEqual(probs[2], self.model.eps)  # m=9

    def test_survivor_rate_matches_formula(self) -> None:
        n_value = 100
        probs = self.model.predict_proba(n_value)
        offsets = n_value - self.model.cands
        for index, offset in enumerate(offsets):
            if offset < 3:
                self.assertEqual(probs[index], 0.0)
                continue
            if int(offset) in self.model.mod_set:
                self.assertAlmostEqual(probs[index], 1.0)
                continue
            if any(int(offset) % int(modulus) == 0 for modulus in self.model.mods):
                self.assertAlmostEqual(probs[index], self.model.eps)
                continue
            expected = min(0.999, E_GAMMA * self.model.log_zM / math.log(float(offset)))
            self.assertAlmostEqual(probs[index], expected, places=12)

    def test_batch_matches_single(self) -> None:
        values = np.asarray([6, 8, 10, 100], dtype=np.int64)
        batch = self.model.predict_proba_batch(values)
        for row, value in enumerate(values):
            single = self.model.predict_proba(int(value))
            np.testing.assert_allclose(batch[row], single, rtol=0, atol=0)

    def test_logits_are_logit_transform_of_probs(self) -> None:
        probs = self.model.predict_proba(42)
        logits = self.model.predict_logits(42)
        clipped = np.clip(probs, 1e-9, 1.0 - 1e-9)
        expected = np.log(clipped / (1.0 - clipped))
        np.testing.assert_allclose(logits, expected, rtol=1e-12, atol=1e-12)

    def test_null1_ranking_inherits_null0(self) -> None:
        train_n = np.asarray([16, 42, 100, 256, 512, 1024], dtype=np.int64)
        train_probs = self.model.predict_proba_batch(train_n).reshape(-1)
        labels = (train_probs > 0.05).astype(np.float64)

        calibrated = NullModelCalibrated(self.model)
        calibrated.fit(train_n, labels.reshape(train_n.shape[0], -1))

        eval_n = np.asarray([18, 42, 200, 999], dtype=np.int64)
        base_logits = self.model.predict_logits_batch(eval_n)
        ranking_logits = calibrated.ranking_logits_batch(eval_n)
        np.testing.assert_allclose(ranking_logits, base_logits, rtol=0, atol=0)

    def test_null1_is_monotone_recalibration(self) -> None:
        train_n = np.asarray([16, 42, 100, 256, 512, 1024], dtype=np.int64)
        train_probs = self.model.predict_proba_batch(train_n).reshape(-1)
        labels = (train_probs > 0.05).astype(np.float64)

        calibrated = NullModelCalibrated(self.model)
        calibrated.fit(train_n, labels.reshape(train_n.shape[0], -1))

        eval_probs = self.model.predict_proba_batch(train_n).reshape(-1)
        null1_probs = calibrated.predict_proba_batch(train_n).reshape(-1)
        order = np.argsort(eval_probs)
        self.assertTrue(np.all(np.diff(null1_probs[order]) >= -1e-12))


if __name__ == "__main__":
    unittest.main()
