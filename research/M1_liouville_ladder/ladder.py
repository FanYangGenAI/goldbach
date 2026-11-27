"""M1-E1：Liouville–哥德巴赫阶梯的数值形态（见 DESIGN.md 第 3 节）。"""
import json, sys, time
import numpy as np

NMAX = 1 << 24
t0 = time.time()
# 最小素因子 spf（spf[1] = 大数，表示 1 对任意 z 都"粗糙"）
spf = np.zeros(NMAX + 1, dtype=np.int64)
is_comp = np.zeros(NMAX + 1, dtype=bool)
r = int(NMAX ** 0.5) + 1
for p in range(2, r + 1):
    if not is_comp[p]:
        idx = np.arange(p * p, NMAX + 1, p)
        sel = spf[idx] == 0
        spf[idx[sel]] = p
        is_comp[p * p::p] = True
spf[spf == 0] = np.arange(NMAX + 1)[spf == 0]
spf[1] = 1 << 62
primes = np.nonzero(~is_comp)[0]; primes = primes[primes >= 2]
par = np.zeros(NMAX + 1, dtype=np.uint8)
for p in primes:
    pk = int(p)
    while pk <= NMAX:
        par[pk::pk] ^= 1
        pk *= int(p)
lam = (1 - 2 * par.astype(np.int8))
print(f"[setup] {time.time()-t0:.1f}s", flush=True)

rng = np.random.default_rng(20261126)
scales = [17, 20, 23]
out = []
for k in scales:
    X = 1 << k
    Ns = list(2 * rng.integers(X // 4, X // 2, size=30))
    Ns += [X, 3 * (X // 4), 30030 * (X // 30030), X - 2]   # 高度合数与对照
    Ns = [int(n) for n in Ns if n % 2 == 0 and n <= NMAX]
    for N in Ns:
        a = np.arange(1, N)
        m = np.minimum(spf[a], spf[N - a])          # 配对的"粗糙度"：两侧最小素因子的较小者
        la, lb = lam[a], lam[N - a]
        L = int((la.astype(np.int64) * lb).sum())
        row = dict(scale=k, N=N, Lambda=L, Lambda_over_sqrt2N=L / np.sqrt(2 * N), Lambda_over_N=L / N)
        zs = {"1": 1, "2": 2, "3": 3, "5": 5, "7": 7, "11": 11, "31": 31, "101": 101,
              "N^0.1": N ** 0.1, "N^0.2": N ** 0.2, "N^0.25": N ** 0.25, "N^(1/3)": N ** (1 / 3)}
        cells = {}
        for name, z in zs.items():
            rough = m > z
            T = int(rough.sum())
            mm = int((rough & (la == -1) & (lb == -1)).sum())
            # 单侧：[1,N) 中 z-粗糙数里 λ=-1 的比例
            single = spf[a] > z
            q = float((single & (la == -1)).sum() / max(single.sum(), 1))
            pred = T * q * q
            cells[name] = dict(T=T, C_mm=mm, pred=pred, ratio=mm / pred if pred > 0 else None,
                               poisson_sd=1 / np.sqrt(pred) if pred > 0 else None)
        row["cells"] = cells
        out.append(row)
    print(f"[scale 2^{k}] {len(Ns)} N done ({time.time()-t0:.1f}s)", flush=True)

json.dump(out, open(sys.argv[1] if len(sys.argv) > 1 else "m1_e1_raw.json", "w"), indent=1)

# 汇总与判据
summary = {}
names = list(out[0]["cells"].keys())
for k in scales:
    rows = [r_ for r_ in out if r_["scale"] == k]
    s = {}
    for name in names:
        ratios = np.array([r_["cells"][name]["ratio"] for r_ in rows], dtype=float)
        psd = np.array([r_["cells"][name]["poisson_sd"] for r_ in rows], dtype=float)
        s[name] = dict(min_ratio=float(np.nanmin(ratios)), mean_ratio=float(np.nanmean(ratios)),
                       sd_ratio=float(np.nanstd(ratios)), mean_poisson_sd=float(np.nanmean(psd)),
                       min_C_mm=int(min(r_["cells"][name]["C_mm"] for r_ in rows)))
    lam_sc = np.array([abs(r_["Lambda_over_sqrt2N"]) for r_ in rows])
    summary[f"2^{k}"] = dict(cells=s, median_abs_Lambda_over_sqrt2N=float(np.median(lam_sc)),
                             max_abs_Lambda_over_N=float(max(abs(r_["Lambda_over_N"]) for r_ in rows)))
H1 = all(summary[sc]["cells"][nm]["min_ratio"] >= 0.8 for sc in summary for nm in names)
lowz = ["1", "2", "3", "5", "7", "11", "31", "101", "N^0.1", "N^0.2", "N^0.25"]
H2_detail = {sc: {nm: summary[sc]["cells"][nm]["sd_ratio"] / summary[sc]["cells"][nm]["mean_poisson_sd"] for nm in lowz} for sc in summary}
H2 = all(v <= 3 for d in H2_detail.values() for v in d.values())
meds = [summary[sc]["median_abs_Lambda_over_sqrt2N"] for sc in summary]
maxs = [summary[sc]["max_abs_Lambda_over_N"] for sc in summary]
H3 = all(0.3 <= m_ <= 1.2 for m_ in meds) and all(maxs[i] > maxs[i + 1] for i in range(len(maxs) - 1))
verdict = dict(H1=H1, H2=H2, H2_sd_over_poisson=H2_detail, H3=H3, H3_medians=meds, H3_max_over_N=maxs)
json.dump(dict(summary=summary, verdict=verdict), open("m1_e1_summary.json", "w"), indent=1)
print(json.dumps(verdict, indent=1))
for sc in summary:
    print(sc, {nm: (round(v["min_ratio"], 3), round(v["mean_ratio"], 3), v["min_C_mm"]) for nm, v in summary[sc]["cells"].items()})
print(f"[done] {time.time()-t0:.1f}s")
