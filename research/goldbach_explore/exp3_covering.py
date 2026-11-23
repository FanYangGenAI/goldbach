"""Experiment 3: the 'covering' view of a hypothetical counterexample.

A counterexample N needs every N - p (p prime, p <= N/2) to be composite, i.e. each
prime p must fall in the residue class  p == N (mod l)  for some prime l <= sqrt(N).

(A) minimal Goldbach prime p_min(N) for all even N <= NMAX; record holders.
(B) sieve-survivor curves S_z(N) = #{p <= N/2 : N-p has no prime factor <= z}
    for actual N, compared with the independence (Mertens-product) prediction.
(C) adversarial greedy covering: if one could pick the class a_l freely for every
    prime l (CRT), how many moduli are needed to cover all primes p <= Y?

Run:  python research/goldbach_explore/exp3_covering.py [log10 NMAX]
"""
from __future__ import annotations

import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numba as nb
import numpy as np

from common import C2, FIG_DIR, prime_sieve, save_json

NMAX = int(10 ** float(sys.argv[1])) if len(sys.argv) > 1 else 10**9
NMAX -= NMAX % 2
t0 = time.time()


@nb.njit(cache=True)
def odd_sieve(n):
    """s[i] = 1 iff 2i+1 is prime, for 2i+1 <= n."""
    m = n // 2 + 1
    s = np.ones(m, dtype=np.uint8)
    s[0] = 0
    i = 1
    while (2 * i + 1) * (2 * i + 1) <= n:
        if s[i]:
            p = 2 * i + 1
            j = (p * p) // 2
            while j < m:
                s[j] = 0
                j += p
        i += 1
    return s


@nb.njit(parallel=True, cache=True)
def min_goldbach_prime(s, small_primes, nmax):
    """pm[k] = smallest odd prime p with N - p prime, N = 2k (k >= 3)."""
    K = nmax // 2 + 1
    pm = np.zeros(K, dtype=np.uint16)
    for k in nb.prange(3, K):
        N = 2 * k
        for p in small_primes:
            if p > N // 2:
                break
            q = N - p
            if s[q // 2]:
                pm[k] = p
                break
    return pm


s = odd_sieve(NMAX)
small = np.nonzero(prime_sieve(100000))[0][1:].astype(np.int64)  # odd primes
pm = min_goldbach_prime(s, small, NMAX)
print(f"p_min for N <= {NMAX:,}: {time.time()-t0:.1f}s, unresolved={(pm[3:]==0).sum()}")
assert (pm[3:] == 0).sum() == 0, "Goldbach failure?! check code"

# records
cm = np.maximum.accumulate(pm[3:])
chg = np.nonzero(np.diff(cm.astype(np.int32)) > 0)[0] + 1
records = [(6, int(pm[3]))] + [(2 * (int(i) + 3), int(cm[i])) for i in chg]
del cm
rec_N = np.array([x[0] for x in records], float)
rec_p = np.array([x[1] for x in records], float)
# ratio to log^2 N  (a common heuristic scale for the max of p_min)
ratio_log2 = rec_p / np.log(rec_N) ** 2
mean_pmin = {f"1e{e}": float(pm[max(3, int(10 ** (e - 1)) // 2): int(10 ** e) // 2 + 1].mean()) for e in range(3, int(np.log10(NMAX)) + 1)}
hist = np.bincount(pm[3:].astype(np.int64))
tail = {int(p): float(hist[p:].sum() / hist.sum()) for p in [3, 5, 7, 11, 13, 31, 101, 211, 503, 1009] if p < len(hist)}

# ---------- (B) survivor curves for selected N ----------
@nb.njit(cache=True)
def lpf_profile(N, s, primes_upto_sqrt):
    """For each prime p <= N/2 (p odd), the smallest prime factor of N-p (capped).
    Returns array of lpf values (N-p itself if prime)."""
    out = []
    for i in range(1, N // 4 + 1):
        p = 2 * i + 1
        if p > N // 2:
            break
        if not s[i]:
            continue
        q = N - p
        f = q
        for l in primes_upto_sqrt:
            if l * l > q:
                break
            if q % l == 0:
                f = l
                break
        out.append(f)
    return np.array(out, dtype=np.int64)


NB = min(NMAX, 10**8)
sB = odd_sieve(NB)
pr_sqrt = np.nonzero(prime_sieve(int(NB**0.5) + 10))[0][1:].astype(np.int64)
lpf_cases = {}
rng = np.random.default_rng(7)
case_N = [int(x) for x in rec_N[rec_N <= NB][-2:]] + [int(2 * rng.integers(NB // 4, NB // 2)) for _ in range(2)]
case_N += [9699690 * 10]  # very composite N = 2*3*5*...*19 * 10
case_N = [n for n in case_N if n <= NB]
zgrid = pr_sqrt[pr_sqrt <= int(NB**0.5)]
surv_curves = {}
for Nc in case_N:
    lp = lpf_profile(Nc, sB, pr_sqrt)
    zs = zgrid[zgrid <= int(Nc**0.5)]
    surv = np.array([(lp > z).sum() for z in zs])
    # independence prediction: pi(N/2) * prod_{3<=l<=z, l !| N} (1 - 1/(l-1))
    fac = np.where(Nc % zs == 0, 1.0, 1 - 1 / (zs - 1.0))
    pred = len(lp) * np.cumprod(fac)
    surv_curves[Nc] = dict(z=zs, surv=surv, pred=pred, n_primes=len(lp))
    print(f"N={Nc}: primes<=N/2 {len(lp)}, survivors at sqrt(N) {surv[-1]}, naive pred {pred[-1]:.0f}, ratio {surv[-1]/pred[-1]:.3f}")
mertens_ratio = {int(k): float(v["surv"][-1] / v["pred"][-1]) for k, v in surv_curves.items()}

# ---------- (C) adversarial greedy covering ----------
LS_ALL = np.nonzero(prime_sieve(4_000_000))[0][1:]


def greedy_cover(Y):
    P = np.nonzero(prime_sieve(Y))[0]
    P = P[P > 3]
    surv = P.copy()
    g_hist = []
    for l in LS_ALL:
        if len(surv) == 0:
            break
        if 4 * len(surv) < l:
            vals, cnts = np.unique(surv % l, return_counts=True)
            j = int(np.argmax(cnts)); a, killed = int(vals[j]), int(cnts[j])
        else:
            c = np.bincount(surv % l, minlength=l)
            a = int(np.argmax(c)); killed = int(c[a])
        surv = surv[surv % l != a]
        g_hist.append((int(l), len(surv), killed))
    return P, surv, g_hist


scaling = []
for Yi in [10**4, 3 * 10**4, 10**5, 3 * 10**5, 10**6, 3 * 10**6, 10**7, 3 * 10**7]:
    Pi, si, gi = greedy_cover(Yi)
    Lc = gi[-1][0] if len(si) == 0 else None
    scaling.append(dict(Y=Yi, n_primes=int(len(Pi)), covered_at_l=Lc, n_moduli=len(gi),
                        sqrt_2Y=float((2 * Yi) ** 0.5), ratio_to_sqrt2Y=(Lc / (2 * Yi) ** 0.5) if Lc else None))
    print(scaling[-1], f"{time.time()-t0:.1f}s")

Y = 5_000_000
P, surv, g_hist = greedy_cover(Y)
g_l = np.array([x[0] for x in g_hist]); g_s = np.array([x[1] for x in g_hist])
covered_at = int(g_l[-1]) if len(surv) == 0 else None
# compare: a real N ~ 2Y may only use moduli l <= sqrt(2Y)
sqrt2Y = int((2 * Y) ** 0.5)
idx = np.searchsorted(g_l, sqrt2Y)
greedy_surv_at_sqrt = int(g_s[min(idx, len(g_s) - 1)])
fac = 1 - 1 / (g_l - 1.0)
naive_pred = len(P) * np.cumprod(fac)

summary = dict(
    NMAX=NMAX, records=[dict(N=int(a), p=int(b)) for a, b in records],
    record_p_over_log2N=[float(x) for x in ratio_log2], mean_pmin_by_decade=mean_pmin,
    tail_P_pmin_ge=tail, mertens_ratio_at_sqrtN=mertens_ratio,
    greedy=dict(Y=Y, n_primes=int(len(P)), covered_at_l=covered_at, n_moduli_used=len(g_hist),
                survivors_at_sqrt_2Y=greedy_surv_at_sqrt, sqrt_2Y=sqrt2Y,
                naive_pred_at_sqrt_2Y=float(naive_pred[min(idx, len(g_s) - 1)])),
    greedy_scaling=scaling,
    elapsed_s=time.time() - t0,
)
print(save_json("exp3_summary.json", summary))

# ---------- figures ----------
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
axs[0].loglog(rec_N, rec_p, "o-", ms=3, label="record p_min(N)")
nn = np.logspace(1, np.log10(NMAX), 200)
axs[0].loglog(nn, np.log(nn) ** 2, "--", label=r"$\log^2 N$")
axs[0].loglog(nn, np.log(nn) ** 2 * np.log(np.log(nn)), ":", label=r"$\log^2 N\,\log\log N$")
axs[0].set_xlabel("N"); axs[0].legend(); axs[0].set_title("record minimal Goldbach primes")
axs[1].semilogy(np.arange(len(hist)), np.cumsum(hist[::-1])[::-1] / hist.sum(), lw=1)
axs[1].set_xlabel("y"); axs[1].set_ylabel("P(p_min(N) >= y)"); axs[1].set_xlim(0, min(len(hist), 800))
axs[1].set_title(f"tail of p_min over even N <= {NMAX:.0e}")
fig.tight_layout(); fig.savefig(f"{FIG_DIR}/exp3_pmin.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 5))
for Nc, v in surv_curves.items():
    ax.plot(v["z"], v["surv"] / v["n_primes"], lw=1.2, label=f"N={Nc} actual")
    ax.plot(v["z"], v["pred"] / v["n_primes"], "k:", lw=0.6)
ax.plot(g_l[g_l <= sqrt2Y], g_s[g_l <= sqrt2Y] / len(P), "r-", lw=2, label=f"greedy adversarial (primes ≤ {Y:.0e})")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("sieve level z (primes l <= z used)"); ax.set_ylabel("fraction of primes p surviving")
ax.set_title("Covering view: survivors stay a ~1/log z fraction; dotted = independence prediction")
ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{FIG_DIR}/exp3_survivors.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.loglog(g_l, g_s, label="greedy survivors")
ax.loglog(g_l, naive_pred, "k:", label="forced-class (independence) prediction")
ax.axvline(sqrt2Y, color="r", ls="--", lw=0.8, label=r"$\sqrt{2Y}$: moduli a real N may use")
ax.set_xlabel("largest modulus l used"); ax.set_ylabel("uncovered primes")
ax.set_title(f"Greedy covering of all primes ≤ {Y:.0e}: done at l = {covered_at}")
ax.legend(); fig.tight_layout(); fig.savefig(f"{FIG_DIR}/exp3_greedy.png", dpi=140); plt.close(fig)
print(f"done {time.time()-t0:.1f}s")
