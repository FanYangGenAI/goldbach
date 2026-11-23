"""Shared utilities for the exploratory Goldbach experiments.

Conventions
-----------
r(N)  : ordered Goldbach count  #{(p, q) primes : p + q = N}
HL(N) : Hardy-Littlewood prediction for r(N)
        2 * C2 * prod_{p | N, p > 2} (p-1)/(p-2) * Li2(N),
        Li2(N) = int_2^{N-2} dt / (log t * log(N - t))
d(h)  : ordered difference count #{p <= X : p + h prime}
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import integrate

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
OUT_DIR = os.path.join(HERE, "out")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Twin prime constant C2 = prod_{p>2} (1 - 1/(p-1)^2)
C2 = 0.6601618158468695739278121100145


def prime_sieve(n: int) -> np.ndarray:
    """Boolean array is_prime[0..n]."""
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    s[4::2] = False
    for i in range(3, int(n**0.5) + 1, 2):
        if s[i]:
            s[i * i :: 2 * i] = False
    return s


def smallest_prime_factor(n: int) -> np.ndarray:
    spf = np.zeros(n + 1, dtype=np.int32)
    spf[2::2] = 2
    for i in range(3, n + 1, 2):
        if spf[i] == 0:
            spf[i] = i
            if i * i <= n:
                block = spf[i * i :: 2 * i]
                block[block == 0] = i
                spf[i * i :: 2 * i] = block
    return spf


def odd_part_singular_factor(n_max: int, primes: np.ndarray) -> np.ndarray:
    """f[N] = prod_{p | N, p > 2} (p-1)/(p-2) for N <= n_max (vectorised by prime)."""
    f = np.ones(n_max + 1, dtype=np.float64)
    for p in primes:
        if p == 2:
            continue
        f[p::p] *= (p - 1) / (p - 2)
    return f


def li2_goldbach(N: float) -> float:
    """int_2^{N-2} dt / (log t log(N-t)), computed as 2 * int_2^{N/2}."""
    g = lambda s: np.exp(s) / (s * np.log(N - np.exp(s)))  # t = e^s
    val, _ = integrate.quad(g, np.log(2.0), np.log(N / 2.0), limit=200)
    return 2.0 * val


def li2_goldbach_interp(Ns: np.ndarray, grid_points: int = 4000) -> np.ndarray:
    """Li2(N) for many N via accurate values on a log grid + interpolation of
    Li2(N) * log(N)^2 / N, which is a slowly varying function."""
    Ns = np.asarray(Ns, dtype=np.float64)
    lo, hi = max(Ns.min(), 8.0), Ns.max()
    grid = np.exp(np.linspace(np.log(lo), np.log(hi) + 1e-12, grid_points))
    vals = np.array([li2_goldbach(x) for x in grid])
    h = vals * np.log(grid) ** 2 / grid
    hs = np.interp(np.log(np.maximum(Ns, lo)), np.log(grid), h)
    return hs * Ns / np.log(np.maximum(Ns, lo)) ** 2


def li2_twin(X: float, h: float) -> float:
    """int_2^X dt / (log t log(t + h))."""
    val, _ = integrate.quad(lambda t: 1.0 / (np.log(t) * np.log(t + h)), 2.0, X, limit=400)
    return val


def autoconv_counts(is_prime: np.ndarray) -> np.ndarray:
    """r[N] = sum_k a[k] a[N-k] via real FFT (exact after rounding)."""
    a = is_prime.astype(np.float64)
    n = len(a)
    L = 1 << int(np.ceil(np.log2(2 * n)))
    A = np.fft.rfft(a, L)
    r = np.fft.irfft(A * A, L)[: 2 * n - 1]
    return np.rint(r).astype(np.int64)


def autocorr_counts(is_prime: np.ndarray, H: int) -> np.ndarray:
    """d[h] = sum_k a[k] a[k+h] for 0 <= h <= H."""
    a = is_prime.astype(np.float64)
    n = len(a)
    L = 1 << int(np.ceil(np.log2(2 * n)))
    A = np.fft.rfft(a, L)
    c = np.fft.irfft(A * np.conj(A), L)[: H + 1]
    return np.rint(c).astype(np.int64)


def save_json(name: str, obj) -> str:
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, default=float)
    return path
