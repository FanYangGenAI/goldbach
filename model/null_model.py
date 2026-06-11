"""Zero-parameter Hardy–Littlewood / Mertens null model (NULL-0).

NULL-0 predicts per-candidate Bernoulli rates for survivors after small-prime
sieving. The singular series S(N) does not appear explicitly: it only affects
how many candidates survive, which is already encoded in the sieve indicators.

See docs/igot_experiment_plan_v2.1.md §3.2.1 for the mathematical definition.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_GENERATOR_PYTHON = Path(__file__).resolve().parents[1] / "generator" / "python"
if str(_GENERATOR_PYTHON) not in sys.path:
    sys.path.insert(0, str(_GENERATOR_PYTHON))

from constants import CANDIDATE_PRIME_COUNT, MODULUS_BASE_COUNT  # noqa: E402
from primes import first_k_odd_primes, first_m_modulus_bases  # noqa: E402

E_GAMMA = 1.7810724179901979  # exp(Euler–Mascheroni constant)


@dataclass(frozen=True)
class NullModelConfig:
    """Hyper-parameters for NULL-0 / NULL-1."""

    candidate_count: int = CANDIDATE_PRIME_COUNT
    modulus_count: int = MODULUS_BASE_COUNT
    eps: float = 1e-4
    min_offset: int = 3  # eligibility: m = N - p_i must be >= 3


class NullModel:
    """Analytic null predictor: NULL-0 probabilities and logits."""

    def __init__(
        self,
        candidate_primes: np.ndarray | None = None,
        modulus_bases: np.ndarray | None = None,
        *,
        config: NullModelConfig | None = None,
    ) -> None:
        cfg = config or NullModelConfig()
        self.eps = cfg.eps
        self.min_offset = cfg.min_offset

        if candidate_primes is None:
            candidate_primes = np.asarray(
                first_k_odd_primes(cfg.candidate_count), dtype=np.int64
            )
        if modulus_bases is None:
            modulus_bases = np.asarray(
                first_m_modulus_bases(cfg.modulus_count), dtype=np.int64
            )

        self.cands = np.asarray(candidate_primes, dtype=np.int64)
        self.mods = np.asarray(modulus_bases, dtype=np.int64)
        if self.mods.size == 0:
            raise ValueError("modulus_bases must be non-empty")

        self.mod_set = frozenset(int(r) for r in self.mods.tolist())
        self.log_zM = math.log(float(self.mods[-1]))

    @classmethod
    def with_modulus_count(cls, modulus_count: int) -> NullModel:
        """Build NULL-0 with the first `modulus_count` odd-prime moduli."""
        return cls(
            modulus_bases=np.asarray(
                first_m_modulus_bases(modulus_count), dtype=np.int64
            )
        )

    def predict_proba(self, even_number: int) -> np.ndarray:
        """Return [K] probabilities for one even N."""
        return self.predict_proba_batch(np.asarray([even_number], dtype=np.int64))[0]

    def predict_proba_batch(self, even_numbers: np.ndarray) -> np.ndarray:
        """Return [B, K] probabilities for a batch of even N values."""
        even_numbers = np.asarray(even_numbers, dtype=np.int64)
        if even_numbers.ndim != 1:
            raise ValueError("even_numbers must be a 1-D array")

        offsets = even_numbers[:, np.newaxis] - self.cands[np.newaxis, :]
        probs = np.zeros(offsets.shape, dtype=np.float64)

        ineligible = offsets < self.min_offset
        probs[ineligible] = 0.0

        active = ~ineligible
        small_prime = np.isin(offsets, list(self.mod_set)) & active
        probs[small_prime] = 1.0

        divisible = np.zeros(offsets.shape, dtype=bool)
        for modulus in self.mods:
            divisible |= offsets % int(modulus) == 0
        sieved = divisible & active & ~small_prime
        probs[sieved] = self.eps

        survivor = active & ~small_prime & ~divisible
        if np.any(survivor):
            survivor_offsets = offsets[survivor].astype(np.float64)
            probs[survivor] = np.minimum(
                0.999, E_GAMMA * self.log_zM / np.log(survivor_offsets)
            )

        return probs

    def predict_logits(self, even_number: int) -> np.ndarray:
        """Return [K] logits for one even N."""
        return self.predict_logits_batch(np.asarray([even_number], dtype=np.int64))[0]

    def predict_logits_batch(self, even_numbers: np.ndarray) -> np.ndarray:
        """Return [K] logits for each row in `even_numbers`."""
        probs = np.clip(self.predict_proba_batch(even_numbers), 1e-9, 1.0 - 1e-9)
        return np.log(probs / (1.0 - probs))


class NullModelCalibrated:
    """NULL-1: isotonic recalibration of NULL-0 probabilities (train-only fit)."""

    def __init__(self, base: NullModel) -> None:
        self.base = base
        self._iso = None

    def fit(self, even_numbers: np.ndarray, labels: np.ndarray) -> None:
        """Fit isotonic regression on flattened (p_hat, y) pairs from train split."""
        from sklearn.isotonic import IsotonicRegression

        even_numbers = np.asarray(even_numbers, dtype=np.int64)
        labels = np.asarray(labels, dtype=np.float64)
        probs = self.base.predict_proba_batch(even_numbers).reshape(-1)
        targets = labels.reshape(-1)

        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(probs, targets)
        self._iso = iso

    @property
    def is_fitted(self) -> bool:
        return self._iso is not None

    def predict_proba_batch(self, even_numbers: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("NULL-1 is not fitted; call fit() on train split first")
        base_probs = self.base.predict_proba_batch(even_numbers)
        flat = self._iso.predict(base_probs.reshape(-1))
        return flat.reshape(base_probs.shape)

    def predict_calibrated_logits_batch(self, even_numbers: np.ndarray) -> np.ndarray:
        """Logits from isotonic-calibrated probabilities (for BCE / calibration only)."""
        probs = np.clip(self.predict_proba_batch(even_numbers), 1e-9, 1.0 - 1e-9)
        return np.log(probs / (1.0 - probs))

    def ranking_logits_batch(self, even_numbers: np.ndarray) -> np.ndarray:
        """Ranking logits: inherit NULL-0 order (calibration must not change sort)."""
        return self.base.predict_logits_batch(even_numbers)

    def predict_logits_batch(self, even_numbers: np.ndarray) -> np.ndarray:
        """Alias for calibrated logits; prefer explicit BCE vs ranking helpers."""
        return self.predict_calibrated_logits_batch(even_numbers)
