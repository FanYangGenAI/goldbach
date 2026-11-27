"""M2-E4 附：期望邻接核 f(u+w) = 1_D(u+w)/(u+w) 的最大特征值（Hilbert 型核）。
全窗口 [0, Z) 的特征值，以及"顶部区域"[q/4, Z) 的特征值（该区域内的局部增长率）。
无窗口限制时连续核 1/(u+w) 的算子范数是 π（Hilbert 不等式）；光滑密度为 ρ 时约为 ρπ。"""
import sys, json, math
import numpy as np
from m2_e4_random_model import D_of
def top_eig(f, lo, hi, iters=300):
    n = hi - lo; m = 1 << (2 * n).bit_length()
    g = np.zeros(m); idx = np.arange(2 * lo, 2 * hi - 1); g[:len(idx)] = f[idx]
    G = np.fft.rfft(g)
    x = np.ones(n) / math.sqrt(n); lam = 0
    for _ in range(iters):
        # y_i = sum_j f(lo+i+lo+j) x_j = (g ⋆ x)[i+j] → correlation: y_i = sum_j g[i+j] x_j
        X = np.fft.rfft(x[::-1], m)
        conv = np.fft.irfft(G * X, m)[n - 1:2 * n - 1]
        lam_new = float(np.dot(conv, x)); nrm = np.linalg.norm(conv)
        x = conv / nrm
        if abs(lam_new - lam) < 1e-9: break
        lam = lam_new
    return lam_new
out = []
for model in ("plain", "qa", "smooth", "smooth3"):
    for A in ((6,) if model in ("plain", "qa") else (6, 24, 48)):
        for q in (101, 1009, 3001):
            D, x = D_of(model, q, A)
            Z = A * q if model != "plain" else 2 * A * q
            f = np.zeros(2 * Z + 2)
            Ds = set(d for d in D if d <= 2 * Z + 1) | {1}
            for d in Ds: f[d] = 1.0 / d
            # 层级编码：u ≥ 0 与 w = l+1 ≥ 1，边模数 d = u + w；用 i = u、j = w − 1 ⇒ d = i + j + 1
            f1 = np.zeros(2 * Z + 2); f1[:-1] = f[1:]
            r = dict(model=model, A=A, q=q, lam_full=round(top_eig(f1, 0, Z), 4),
                     lam_top=round(top_eig(f1, q // 4, Z), 4))
            print(json.dumps(r), flush=True); out.append(r)
json.dump(out, open("m2_e4_hilbert.json", "w"), indent=1)
