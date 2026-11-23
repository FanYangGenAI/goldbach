"""Q7: parameter-free one-system cross-term prediction c_2, c_3 vs observed residual coefficients e_2, e_3.
Background functions are taken verbatim from Q6 (commit on gb/Q6-residual-statistics). See DESIGN.md."""
import json, math, os, subprocess, time
import numba as nb
import numpy as np
from scipy import integrate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
t0 = time.time()
src = subprocess.check_output(["git", "show", "gb/Q6-residual-statistics:research/Q6_residual/q6_residual_stats.py"], cwd=HERE).decode("utf-8")
src = src.split("\ndef load(")[0]
g = {"__name__": "q7", "__file__": os.path.join(HERE, "q6_copy.py")}
exec(compile(src, "q6_residual_stats.py", "exec"), g)
Y = g["Y"]; cweights = g["cweights"]; background = g["background"]; b2_full = g["b2_full"]

# ---- a2(t) = [z^2] delta_z(t) on a log grid ----
ZR = 0.25 * np.exp(2j * np.pi * np.arange(32) / 32)
C2 = sum(cweights(complex(np.round(z.real, 12), np.round(z.imag, 12))) * z ** (-2) for z in ZR) / 32
LG = np.linspace(math.log(2.0), math.log(1.2e9), 16385)
A2TAB = np.real(np.exp(-np.outer(LG, Y)) @ C2)          # a2(e^L)
print("a2 table ready", f"{time.time()-t0:.0f}s", flush=True)
L0 = LG[0]; DL = LG[1] - LG[0]


@nb.njit(cache=True)
def a2(x, tab, L0, DL):
    L = math.log(x)
    u = (L - L0) / DL
    i = int(u)
    if i >= len(tab) - 1:
        i = len(tab) - 2
    f = u - i
    return tab[i] * (1 - f) + tab[i + 1] * f


@nb.njit(cache=True)
def omega_upto(n):
    spf = np.zeros(n + 1, dtype=np.int32)
    for i in range(2, n + 1):
        if spf[i] == 0:
            spf[i] = i
            j = np.int64(i) * i
            while j <= n:
                if spf[j] == 0:
                    spf[j] = i
                j += i
    om = np.zeros(n + 1, dtype=np.int8)
    for m in range(2, n + 1):
        om[m] = om[m // spf[m]] + 1
    return om


@nb.njit(parallel=True, cache=True)
def discrete_parts(Ns, om, tab, L0, DL):
    """S1 = sum_p a1(N-p);  S12 = sum_{f2(n)} a1(N-n);  S21 = sum_p a2(N-p)  (2 <= n <= N-2)."""
    out = np.zeros((len(Ns), 3))
    for k in nb.prange(len(Ns)):
        N = Ns[k]; s1 = 0.0; s12 = 0.0; s21 = 0.0
        for n in range(2, N - 1):
            o = om[n]
            if o == 1:
                m = N - n
                s1 += 1.0 / math.log(m)
                s21 += a2(m, tab, L0, DL)
            elif o == 2:
                s12 += 1.0 / math.log(N - n)
        out[k, 0] = s1; out[k, 1] = s12; out[k, 2] = s21
    return out


def smooth_conv(N, f, gfun):
    I1, _ = integrate.quad(lambda t: f(t) * gfun(N - t), 2, N / 2, limit=400)
    I2, _ = integrate.quad(lambda t: f(t) * gfun(N - t), N / 2, N - 2, limit=400)
    return I1 + I2


a1f = lambda t: 1 / math.log(t)
a2f = lambda t: float(np.interp(math.log(t), LG, A2TAB))


def bcoeffs(N):
    zs = 0.3 * np.exp(2j * np.pi * np.arange(64) / 64)
    B = np.array([background(N, complex(np.round(z.real, 12), np.round(z.imag, 12)))[0] for z in zs])
    c = np.fft.fft(B) / 64
    return (c[2] / 0.3**2).real, (c[3] / 0.3**3).real


def load(commit, path):
    return json.loads(subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=HERE))

q4b = load("gb/Q1-fugacity-sectors", "research/Q1_fugacity/out/q4b_summary.json")
q5d = load("gb/Q5-background-contract", "research/Q5_background/out/q5d_summary.json")
samples = [dict(src="Q4b", X=int(sc), N=r["N"], h=r["h"]) for sc in ["1000000", "10000000", "100000000"] for r in q4b[sc]["rows"]]
samples += [dict(src="Q5d", X=r["scale"], N=r["N"], h=r["h"]) for r in q5d["rows"]]

om = omega_upto(int(1.1e9) + 10)
print("Omega ready", f"{time.time()-t0:.0f}s", flush=True)
Ns = np.array([s_["N"] for s_ in samples], dtype=np.int64)
DP = discrete_parts(Ns, om, A2TAB, L0, DL)
print("discrete convolutions", f"{time.time()-t0:.0f}s", flush=True)
del om

rows = []
for k, s_ in enumerate(samples):
    N = s_["N"]; h = s_["h"]
    b2, b3 = bcoeffs(N)
    b2f = b2_full(N)
    a11 = smooth_conv(N, a1f, a1f); a12 = smooth_conv(N, a1f, a2f)
    c2 = 2 * (DP[k, 0] - a11)
    c3 = 2 * (DP[k, 1] + DP[k, 2] - 2 * a12)
    rows.append(dict(src=s_["src"], X=s_["X"], N=N, mod3=N % 3, nu2=(N & -N).bit_length() - 1,
                     e2=(h[2] - b2) / N, e3=(h[3] - b3) / N, c2=c2 / N, c3=c3 / N, b2_fft=b2, b2_full=b2f))
    if k % 50 == 0:
        print(k, f"{time.time()-t0:.0f}s", flush=True)


def metrics(e, c):
    e = np.asarray(e); c = np.asarray(c)
    ec = e - e.mean(); cc = c - c.mean()
    rho = float(np.corrcoef(ec, cc)[0, 1]) if ec.std() > 0 and cc.std() > 0 else float("nan")
    return dict(rho=rho, sd_ratio=float(cc.std(ddof=1) / ec.std(ddof=1)), R2_pred=float(1 - np.sum((ec - cc) ** 2) / np.sum(ec**2)),
                mean_e=float(e.mean()), mean_c=float(c.mean()))


def within(rs, key_e, key_c):
    """mod-3 within-group centring, then pooled metrics."""
    E = []; C = []
    for g_ in (True, False):
        ss = [r for r in rs if (r["mod3"] == 0) == g_]
        if len(ss) < 3: continue
        e = np.array([r[key_e] for r in ss]); c = np.array([r[key_c] for r in ss])
        E += list(e - e.mean()); C += list(c - c.mean())
    return metrics(E, C)


summ = {}
for src_, X in sorted({(r["src"], r["X"]) for r in rows}):
    rs = [r for r in rows if r["src"] == src_ and r["X"] == X]
    summ[f"{src_} {X}"] = {f"m{m} pooled": metrics([r[f"e{m}"] for r in rs], [r[f"c{m}"] for r in rs]) for m in (2, 3)}
    summ[f"{src_} {X}"].update({f"m{m} within_mod3": within(rs, f"e{m}", f"c{m}") for m in (2, 3)})
    summ[f"{src_} {X}"]["b2_fft_vs_full_maxrel"] = float(max(abs(r["b2_fft"] / r["b2_full"] - 1) for r in rs))
    w = summ[f"{src_} {X}"]["m3 within_mod3"]
    print(src_, X, "m3 within: rho=%.3f sd_ratio=%.3f R2=%.3f | m2 within: rho=%.3f sd_ratio=%.3f" % (
        w["rho"], w["sd_ratio"], w["R2_pred"], summ[f"{src_} {X}"]["m2 within_mod3"]["rho"], summ[f"{src_} {X}"]["m2 within_mod3"]["sd_ratio"]), flush=True)
key = summ["Q5d 1000000000"]["m3 within_mod3"]
verdict = "support" if (key["rho"] > 0.5 and 0.5 <= key["sd_ratio"] <= 2 and key["R2_pred"] > 0.25) else ("reject" if (key["rho"] < 0.2 or key["sd_ratio"] < 0.1) else "partial")
print("verdict:", verdict)
json.dump(dict(summary=summ, verdict=verdict, rows=rows), open(os.path.join(OUT, "q7_summary.json"), "w"), indent=1, allow_nan=False)
print(f"done {time.time()-t0:.0f}s")
