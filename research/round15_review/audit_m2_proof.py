"""Check the new M2 proof reduction independently on complete small fields.

No-root cubic primality and nonsquare quadratic discriminants define the
arrays. Numerical checks support the written proof; they do not replace it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess

import numpy as np
from scipy import fft
import sympy as sp

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("audit_results.json")
SOURCE = "0a25e29"


def arrays(q):
    a, b = np.indices((q, q), dtype=np.int64)
    roots = np.zeros((q, q, q), np.int8)
    for t in range(q):
        c = (-t**3-a*t*t-b*t) % q
        roots[a, b, c] += 1
    P = roots == 0
    eta = -np.ones(q, np.int8); eta[0] = 0
    eta[np.arange(1, q)**2 % q] = 1
    L = np.zeros_like(P)
    disc = np.zeros_like(roots)
    for lead in range(1, q):
        d = (np.arange(q)[:, None]**2-4*lead*np.arange(q)[None, :]) % q
        disc[lead] = eta[d]
        L[lead] = eta[d] == -1
    L[0, 1:, :] = True
    return L, P, disc, eta, roots


def symbolic_check():
    q = sp.symbols("q")
    pi = (q**3-q)/3
    ell = q*(q*q-1)/2
    B = (q**3+q*q-5*q-1)/4
    ans = {}
    for eps in (1, -1):
        S = q*(q-1)*(q*q-q-4-2*eps)/9
        T = q*(q-1)*(q**3-3*q+2*eps)/9
        R = (q*(q-1)**2*(q**3+q*q-q-4)/18 if eps == 1
             else q*(q-2)*(q-1)*(q+1)*(q*q+q+1)/18)
        m = ell*pi+ell/2*S+B*T+(B+1)*(pi*pi-pi-S-T)
        m -= (q*q-2*q-1)/2*(R-T)+(q*q-1)/4*T
        target = (q*(q-1)**2*(q**6+2*q**5-3*q**3+9*q-3)/36 if eps == 1
                  else q*(q-1)*(q+1)*(q**6-2*q**4-q**3+4*q*q+3*q+5)/36)
        difference = sp.factor(m-target)
        assert difference == 0
        # Norm/trace pair count -> the aggregate linear-shift count.
        npairs = q**4+2*q**3-3*q*q+2*eps*q-2*eps
        assert sp.factor(q*(npairs/9-pi)-T) == 0
        ans[str(eps)] = {"M2": str(sp.factor(m)), "difference": str(difference)}
    return ans


def finite_check(q):
    L, P, disc, eta, roots = arrays(q)
    Q = q**3; eps = 1 if q % 3 == 1 else -1
    pi, ell = int(P.sum()), int(L.sum())
    assert pi == (q**3-q)//3 and ell == q*(q*q-1)//2
    ph = fft.rfftn(P.astype(float)); lh = fft.rfftn(L.astype(float))
    AP = np.rint(fft.irfftn(abs(ph)**2, s=(q, q, q))).astype(np.int64)
    AL = np.rint(fft.irfftn(abs(lh)**2, s=(q, q, q))).astype(np.int64)
    r = np.rint(fft.irfftn(ph*lh, s=(q, q, q))).astype(np.int64)
    B = (q**3+q*q-5*q-1)//4
    weights = np.empty_like(AL)
    weights[0, 0, :] = ell//2
    weights[0, 0, 0] = ell
    weights[0, 1:, :] = B
    weights[1:, :, :] = B+1
    quad_irred = np.zeros_like(P); quad_double = np.zeros_like(P)
    quad_irred[1:] = disc[1:] == -1
    quad_double[1:] = disc[1:] == 0
    weights[quad_irred] -= (q*q-2*q-1)//2
    weights[quad_double] -= (q*q-1)//4
    assert np.array_equal(AL, weights)
    S = int(AP[0, 0, 1:].sum())
    T = int(AP[0, 1:, :].sum())
    I = int(AP[quad_irred].sum())
    D = int(AP[quad_double].sum())
    R = int(r[P].sum())
    expected_R = (q*(q-1)**2*(q**3+q*q-q-4)//18 if eps == 1
                  else q*(q-2)*(q-1)*(q+1)*(q*q+q+1)//18)
    assert R == expected_R
    assert int(AP.sum()) == pi*pi
    assert S == q*(q-1)*(q*q-q-4-2*eps)//9
    assert T == q*(q-1)*(q**3-3*q+2*eps)//9
    assert I+T == R
    assert D == T
    family = P.sum(axis=1).astype(np.int64)  # fixed quadratic coefficient and constant term
    root_pair_count = 9*int(np.sum(family*family))
    assert root_pair_count == q**4+2*q**3-3*q*q+2*eps*q-2*eps
    assert int(family[3 % q, 1]) == (q-eps)//3
    # Missing values of t^3+p*t prove the constant-shift aggregate.
    images = np.zeros((q, q), bool)
    for p in range(q):
        images[p, (np.arange(q)**3+p*np.arange(q)) % q] = True
    missing = q-images.sum(axis=1)
    assert np.all(missing[1:] == (q-eps)//3)
    assert int(missing[0]) == (2*(q-1)//3 if eps == 1 else 0)
    assert S == q*int(np.sum(missing*(missing-1)))
    actual = int(np.sum(r*r))
    inner = int(np.sum(AL*AP))
    reduced = ell*pi+(ell//2)*S+B*T+(B+1)*(pi*pi-pi-S-T)
    reduced -= (q*q-2*q-1)//2*(R-T)+(q*q-1)//4*T
    assert actual == inner == reduced
    # Verify the reciprocal bijection used to identify D=T, at h=a(t-b).
    checked = 0
    inv = lambda z: pow(int(z) % q, q-2, q)
    for a in range(1, q):
        for c2, c1, c0 in np.argwhere(P):
            if not P[c2, (c1+a) % q, c0]:
                continue
            # h=a*t has root 0, and H(0)=c0 is nonzero.
            f0 = int(c0)
            ic = inv(f0)
            K = (int(c1)*ic % q, int(c2)*ic % q, ic)  # (u^2,u,constant)
            assert P[K]
            assert P[(K[0]+a*ic) % q, K[1], K[2]]
            checked += 1
            if checked >= 12:
                break
        if checked >= 12:
            break
    sign_counterexample = int(sum(int(eta[(x-1) % q]) for x in range(2, q)))
    assert sign_counterexample == -int(eta[-1])
    return {"q": q, "epsilon": eps, "pi": pi, "L": ell, "S_const": S, "T_linear": T,
            "quadratic_irreducible_sum": I, "quadratic_double_sum": D, "R_I3": R,
            "norm_trace_root_pair_count": root_pair_count, "M2_direct": actual, "M2_reduced": reduced,
            "max_AL_error": int(abs(AL-weights).max()), "reciprocal_samples": checked,
            "sum_eta_x_minus_1_on_S": sign_counterexample}


def main():
    out = {"source_commit": subprocess.check_output(["git", "rev-parse", SOURCE], cwd=ROOT).decode().strip(),
           "symbolic": symbolic_check(), "fields": []}
    for q in (5, 7, 11, 13, 17, 19, 23, 29, 31, 41):
        row = finite_check(q)
        out["fields"].append(row)
        print(q, row["M2_direct"], row["T_linear"], flush=True)
    out["scope"] = "Finite checks and symbolic substitutions accompany the separate mathematical proof; no M2 candidate is assumed in finite_check."
    OUT.write_text(json.dumps(out, indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
