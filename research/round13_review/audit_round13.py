"""Independent round-13 audit; original experiment files are read through Git.

Saved-summary recomputation, interval overlap, exact algebraic diagnostics,
small function-field reproductions, and an exploratory integer dilation check.
This does not rerun the large CUDA experiments or O25.
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import subprocess
import sys

import numba as nb
import numpy as np
from scipy import fft, stats

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("audit_results.json")
REFS = {
    1: "8948b7a", 2: "cfa6111", 3: "6c774e1", 4: "8042a0e",
    5: "610a25e", 6: "299b205", 7: "b73ccb4", 8: "f0bd7ae",
    9: "17a5864", 10: "335878f", 11: "b5aac2a", 12: "4b31656",
    13: "8790da9", 14: "c05abb3", 15: "e4d8631", 16: "6e1d2a7",
    17: "44d374e", 18: "2021436", 19: "4930a20", 20: "9a54f67",
    21: "4eba674", 22: "df10efd", 23: "2f941aa", 24: "73940de",
}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode("utf-8")


def files(k):
    return [p for p in git("ls-tree", "-r", "--name-only", REFS[k], "research").splitlines()
            if p.startswith(f"research/O{k}_")]


def source(k, suffix):
    matches = [p for p in files(k) if p.endswith(suffix)]
    assert len(matches) == 1, matches
    return git("show", f"{REFS[k]}:{matches[0]}")


def data(k, suffix=None):
    return json.loads(source(k, suffix or f"/out/o{k}_summary.json"))


def manifest():
    ans = {}
    for k, ref in REFS.items():
        relevant = [p for p in files(k) if p.endswith("/DESIGN.md") or p.endswith(".py")]
        ans[str(k)] = {"commit": git("rev-parse", ref).strip(), "files": relevant,
                      "design_history": git("log", "--format=%h %s", ref, "--",
                                            next(p for p in relevant if p.endswith("/DESIGN.md"))).splitlines()}
    return ans


def pair_stats(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    return {"corr": float(np.corrcoef(obs, pred)[0, 1]),
            "slope": float(np.polyfit(pred, obs, 1)[0]),
            "mean_abs_obs": float(np.abs(obs).mean()),
            "mean_abs_resid": float(np.abs(obs-pred).mean())}


def recompute_predictions():
    out = {}
    o5 = data(5)
    out["O5"] = pair_stats([r["mean"]*r["sqrt_x"] for r in o5["rows"]],
                           [r["pred"]*r["sqrt_x"] for r in o5["rows"]])
    o9 = data(9)
    out["O5_plus_O9"] = pair_stats(
        [r["mean"]*r["sqrt_x"] for r in o5["rows"]+o9["rows"]],
        [r["pred"]*r["sqrt_x"] for r in o5["rows"]+o9["rows"]])
    out["O5_plus_O9"]["n_windows"] = len(o5["rows"])+len(o9["rows"])
    for k in (6, 11):
        x = data(k)
        ans = {}
        for scale in sorted({r["X"] for r in x["rows"]}):
            rows = [r for r in x["rows"] if r["X"] == scale]
            obs = np.array([r["chi_obs"] for r in rows])
            pred = np.array([r["chi_ref"] for r in rows])
            ns = np.array([r["N"] for r in rows])
            for group in (ns % 3 == 0, ns % 3 != 0):
                obs[group] -= obs[group].mean()
                pred[group] -= pred[group].mean()
            ans[str(scale)] = {"rho": float(np.corrcoef(obs, pred)[0, 1]),
                               "sd_ratio": float(pred.std()/obs.std()),
                               "R2_pred": float(1-np.sum((obs-pred)**2)/np.sum(obs**2)),
                               "n_prediction_samples": len(rows)}
            assert abs(ans[str(scale)]["R2_pred"]-x["summary"][str(scale)]["R2_pred"]) < 1e-12
        out[f"O{k}"] = ans
    o17 = data(17)
    out["O17"] = pair_stats([r["d_obs"]*r["sqrt_x"] for r in o17["rows"]],
                            [r["d_pred"]*r["sqrt_x"] for r in o17["rows"]])
    for k in (18, 21):
        x = data(k)
        ans = {}
        for q in (4, 5, 7, 12):
            ob, pr = [], []
            for row in x["rows"]:
                rr = row["mods"][str(q)]
                a = np.array([r["obs"] for r in rr]); b = np.array([r["pred"] for r in rr])
                ob.extend((a-a.mean())*row["sqrt_x"])
                pr.extend((b-b.mean())*row["sqrt_x"])
            ans[str(q)] = pair_stats(ob, pr) if q != 4 else {"mean_abs_obs": float(np.abs(ob).mean())}
        out[f"O{k}"] = ans
    return out


def audit_kernel():
    out = {"O16_tie_corrected_spearman": [], "O22": []}
    seen = {1, 2, 3, 4, 5, 6, 8, 10, 15, 21, 30, 50, 105, 210, 1000, 10000}
    for r in data(16)["rows"]:
        mask = np.array([i not in seen for i in range(1, 401)])
        corr = stats.spearmanr(np.array(r["rho"])[mask], np.array(r["s"])[mask]).statistic
        out["O16_tie_corrected_spearman"].append({"x": r["x"], "source": r["spearman_fresh"], "correct": float(corr)})
    for r in data(22)["rows"]:
        y, s, om = map(np.array, (r["rho"], r["s"], r["omega"]))
        X1 = np.column_stack([np.ones(len(y)), s])
        X2 = np.column_stack([np.ones(len(y)), s, om])
        fresh = np.arange(len(y)) >= 400
        b1 = np.linalg.lstsq(X1[fresh], y[fresh], rcond=None)[0]
        b2 = np.linalg.lstsq(X2[fresh], y[fresh], rcond=None)[0]
        r1, r2 = y[fresh]-X1[fresh]@b1, y[fresh]-X2[fresh]@b2
        gain = float((r1.var()-r2.var())/y[fresh].var())
        assert np.max(np.abs(b2-r["coef_omega"])) < 1e-12
        # Further exploratory prediction test: fit only L<=400, freeze, evaluate L>400.
        old1 = np.linalg.lstsq(X1[~fresh], y[~fresh], rcond=None)[0]
        old2 = np.linalg.lstsq(X2[~fresh], y[~fresh], rcond=None)[0]
        out["O22"].append({"x": r["x"], "coef_fresh": b2.tolist(), "gain_R2": gain,
                          "W1": bool(-.008 <= b2[2] <= -.002), "W2": bool(gain >= .01),
                          "W3": bool(np.corrcoef(r1, om[fresh])[0, 1] < -.15),
                          "old_lag_fit_to_new_lag_MSE_ratio": float(np.mean((y[fresh]-X2[fresh]@old2)**2)/np.mean((y[fresh]-X1[fresh]@old1)**2))})
    return out


FORMS = {
    "a*L^b": lambda L: .1354*L**.5147,
    "A(1-C/sqrtL)": lambda L: 1.2484*(1-2.199/math.sqrt(L)),
    "A-C/L": lambda L: .9276-5.8465/L,
    "1-C/L": lambda L: 1-7.2562/L,
    "A-C/L^2": lambda L: .7608-49.8224/L**2,
}


def curve_scores(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r["j"], []).append(r)
    pts = [(np.mean([r["logx"] for r in rs]), np.mean([r["c2"] for r in rs])) for rs in groups.values()]
    return {name: float(np.mean([(c-f(L))**2 for L, c in pts])) for name, f in FORMS.items()}


def overlap24():
    old = data(8, "/out/o8_int_summary.json")["rows"] + data(8, "/out/o8_int_smallj.json")["rows"]
    smoke = data(24, "/out/o24_smoke.json")["rows"]
    new = data(24)["rows"]
    flagged, clean, clean_smoke = [], [], []
    for r in new:
        overlaps = [(v["x"], min(r["x"]+r["H"], v["x"]+v["H"])-max(r["x"], v["x"]))
                    for v in old if max(r["x"], v["x"]) < min(r["x"]+r["H"], v["x"]+v["H"])]
        pilot = any(max(r["x"], v["x"]) < min(r["x"]+r["H"], v["x"]+v["H"]) for v in smoke)
        if overlaps:
            flagged.append({"j": r["j"], "x": r["x"], "H": r["H"], "overlap_length": sum(n for _, n in overlaps)})
        else:
            clean.append(r)
            if not pilot:
                clean_smoke.append(r)
    scores = curve_scores(new)
    saved = data(24)["checks"]
    assert max(abs(scores[k]-saved[k]["mse"]) for k in FORMS) < 1e-15
    return {"n_total": len(new), "training_overlap_windows": flagged, "n_without_training_overlap": len(clean),
            "n_without_training_or_smoke_overlap": len(clean_smoke),
            "source_MSE": scores, "clean_training_MSE_exploratory": curve_scores(clean),
            "clean_training_and_smoke_MSE_exploratory": curve_scores(clean_smoke),
            "high_scale_j28plus_MSE_exploratory": curve_scores([r for r in clean_smoke if r["j"] >= 28])}


def symmetry_null_integer():
    """Compare the published ordered-iid label reference with an unordered-pair iid reference."""
    def var_influence(prob, c, b):
        c = np.asarray(c); b = np.asarray(b)
        mean_b = prob@b
        u = c-2*mean_b[:, None]*b
        return np.sum(prob*u*u, axis=1)-np.sum(prob*u, axis=1)**2
    out = []
    for k in (6, 11):
        x = data(k)
        for scale in sorted({r["X"] for r in x["rows"]}):
            rr = [r for r in x["rows"] if r["X"] == scale]
            counts = np.array([r["counts"] for r in rr], float)
            assert np.array_equal(counts[:, 1], counts[:, 2])
            A = counts.sum(axis=1); prob = counts/A[:, None]
            ordered = var_influence(prob, [1, -1, -1, 1], [-1, 1, -1, 1])
            collapsed = np.column_stack([prob[:, 0], prob[:, 1]+prob[:, 2], prob[:, 3]])
            unordered = 2*var_influence(collapsed, [1, -1, 1], [-1, 0, 1])
            obs = np.array([r["chi_obs"] for r in rr]); pred = np.array([r["chi_ref"] for r in rr])
            residual = obs-pred; residual -= residual.mean()
            out.append({"experiment": f"O{k}", "scale": scale,
                        "D_old_per_row_reference": float(np.std(residual/np.sqrt(ordered/A))),
                        "D_unordered_pair_reference": float(np.std(residual/np.sqrt(unordered/A))),
                        "median_null_variance_ratio": float(np.median(unordered/ordered)),
                        "mean_source_D_if_saved": x["summary"][str(scale)].get("D_sample_after")})
    return out


def diagonal_check():
    s = sieve(160)
    rows = []
    for n in range(6, 60, 2):
        first = [(a, n-a) for a in range(2, n-1) if s[a] and s[n-a]]
        for h in range(0, 12, 2):
            second = [(a, n+h-a) for a in range(2, n+h-1) if s[a] and s[n+h-a]]
            actual = sum(bool(set(a)&set(b)) for a in first for b in second)
            if h == 0:
                raw = 2*len(first); corrected = raw-int(s[n//2])
            else:
                raw = 4*sum(s[a] and s[b] and s[b+h] for a, b in first)
                corrected = raw-2*int(s[n//2] and s[n//2+h])-2*int(s[(n+h)//2] and s[(n-h)//2])
            assert actual == corrected
            if (n, h) in [(10, 0), (10, 2)]:
                rows.append({"N": n, "h": h, "actual_shared_prime_pairs": int(actual), "source_formula": int(raw), "corrected": int(corrected)})
    return {"exhaustive_cases": 27*6, "examples": rows}


def sieve(n):
    s = np.ones(n+1, bool); s[:2] = False
    for p in range(2, math.isqrt(n)+1):
        if s[p]:
            s[p*p::p] = False
    return s


def ff_functions():
    tree = ast.parse(source(12, "/o12_ff_spectrum.py"))
    nodes = [v for v in tree.body if isinstance(v, ast.FunctionDef)]
    for node in nodes:
        for deco in node.decorator_list:
            if isinstance(deco, ast.Call):
                for kw in deco.keywords:
                    if kw.arg == "cache":
                        kw.value = ast.Constant(False)
    namespace = {"np": np, "nb": nb, "math": math}
    exec(compile(ast.fix_missing_locations(ast.Module(nodes, type_ignores=[])), str(__file__), "exec"), namespace)
    return namespace


def spectral_ff():
    fn = ff_functions()
    rows = []
    for q, n in [(3, 4), (3, 5), (3, 8), (5, 4), (13, 4)]:
        om, mfd, irr = fn["tables"](q, n)
        low, min_low = fn["lower_tables"](q, n, om, mfd)
        a, b = (om[n] == 1).astype(float), (low == 1).astype(float)
        Q = q**n; shape = (q,)*n
        ah, bh = np.fft.fftn(a.reshape(shape)), np.fft.fftn(b.reshape(shape))
        r = np.rint(np.fft.ifftn(ah*bh).real).ravel()
        logS = np.full(Q, sum(len(irr[k])*math.log(1-1/(q**k-1)**2) for k in irr))
        for k in irr:
            p = q**k
            fn["mark_batch"](logS, q, n, fn["coeff_matrix"](irr[k], q, k), k, math.log(p/(p-1))-math.log(1-1/(p-1)**2), 0)
        bg = len(irr[n])*b.sum()/Q*np.exp(logS)
        e = r-bg; scale = r.mean()
        p2, l2 = abs(ah.ravel())**2, abs(bh.ravel())**2
        idx = np.arange(Q); lev = np.full(Q, n-1)
        for i in range(n-1, -1, -1):
            lev[(idx//q**i) % q != 0] = i-1
        indep = 0.
        for h in range(-1, n-1):
            m = lev == h
            indep += m.sum()*p2[m].mean()*l2[m].mean()/Q**2/scale
        energy = float((p2[1:]*l2[1:]).sum()/Q**2/scale)
        row = {"q": q, "n": n, "c2_raw": float(r.var()/scale), "c2_residual": float(e.var()/scale),
               "c2_background": float(bg.var()/scale), "twice_cov_r_background": float(2*np.mean((r-r.mean())*(bg-bg.mean()))/scale),
               "raw_independent_spectral_baseline": float(indep), "raw_to_independent_ratio": float(r.var()/scale/indep),
               "raw_Parseval_error": abs(energy-float(r.var()/scale))}
        # The two sides have different degree ranges, so their parity marginals need not agree.
        d0 = (n-1)//3
        pl = ((low == 1) & (min_low > d0)).astype(float)
        sl = ((low == 2) & (min_low > d0)).astype(float)
        pt = ((om[n] == 1) & (mfd[n] > d0)).astype(float)
        st = ((om[n] == 2) & (mfd[n] > d0)).astype(float)
        conv = lambda u, v: fn["conv_group"](u, v, q, n)
        pp, ps, sp, ss = conv(pl, pt), conv(pl, st), conv(sl, pt), conv(sl, st)
        A = pp+ps+sp+ss; ok = A > 0
        ml = (-pp-ps+sp+ss)[ok]/A[ok]
        mt = (-pp+ps-sp+ss)[ok]/A[ok]
        C = (pp-ps-sp+ss)[ok]/A[ok]
        old_chi, correct_chi = C-mt**2, C-ml*mt
        row["asymmetric_chi"] = {"max_abs_PS_minus_SP": float(abs(ps-sp).max()),
            "source_chi_mean": float(old_chi.mean()), "connected_chi_mean": float(correct_chi.mean()),
            "source_chi_sd": float(old_chi.std()), "connected_chi_sd": float(correct_chi.std()),
            "mean_mu_low": float(ml.mean()), "mean_mu_top": float(mt.mean()),
            "max_identity_error": float(abs(old_chi-correct_chi-mt*(ml-mt)).max())}
        assert row["raw_Parseval_error"] < 1e-9
        if n <= 4:
            # Exact direct group convolution at three fixed fibres, independent of FFT multiplication.
            powers = q**np.arange(n)
            inds = np.arange(Q)
            for f in (0, Q//2, Q-1):
                idx2 = sum(((f//powers[i]-inds//powers[i]) % q)*powers[i] for i in range(n))
                assert int(np.sum(b*a[idx2])) == r[f]
        rows.append(row)
    return rows


def liouville_dilation():
    m = 1_600_000
    s = sieve(m)
    lam = np.ones(m+1, np.int8); lam[0] = 0
    for pp in np.flatnonzero(s):
        p = int(pp); power = p
        while power <= m:
            lam[power::power] *= -1
            power *= p
    length = fft.next_fast_len(2*m+1)
    conv = np.rint(fft.irfft(fft.rfft(lam, n=length)**2, n=length))[:m+1]
    for n in (10, 100, 1000, 400_000):
        j = np.arange(1, n)
        assert conv[n] == np.sum(lam[j].astype(np.int64)*lam[n-j])
    ns = np.arange(400_000, 800_000, 2)
    y = conv[ns]/np.sqrt(ns); z = conv[2*ns]/np.sqrt(2*ns)
    identity_errors = []
    for n in (400_000, 500_000, 799_998):
        j = np.arange(1, 2*n, 2)
        identity_errors.append(int(conv[2*n]-conv[n]-np.sum(lam[j].astype(np.int64)*lam[2*n-j])))
    assert identity_errors == [0, 0, 0]
    # Gcd decomposition: C(N)=sum_{d|N} C_primitive(d), including C_primitive(2)=1.
    gcd_errors = []
    for n in (30, 60, 120, 210):
        total = 0
        for d in range(2, n+1):
            if n % d == 0:
                j = np.arange(1, d); coprime = np.gcd(j, d) == 1
                total += int(np.sum(lam[j[coprime]].astype(np.int64)*lam[d-j[coprime]]))
        gcd_errors.append(int(conv[n]-total))
    assert gcd_errors == [0]*4
    return {"exploratory": True, "N_range": [400_000, 799_998], "n_even_N": len(ns),
            "corr_yN_y2N": float(np.corrcoef(y, z)[0, 1]), "sd_yN": float(y.std()), "sd_y2N": float(z.std()),
            "adjacent_even_corr": float(np.corrcoef(y[:-1], y[1:])[0, 1]), "dilation_identity_errors": identity_errors,
            "gcd_decomposition_errors": gcd_errors}


def check23():
    rows = []
    for r in data(23)["rows"]:
        Q = r["q"]**r["n"]
        tail = r["max_abs_l"]*math.sqrt(Q)/math.sqrt(2*math.log(Q))
        rows.append({"q": r["q"], "n": r["n"], "sd_l_scaled": r["sd_l_scaled"], "R2_type": r["R2_type"],
                     "tail_ratio": tail, "H1": .7 <= r["sd_l_scaled"] <= 1.3, "H2": r["R2_type"] < .1, "H3": tail < 1.5})
    return rows


def main():
    out = {"hub_commit": git("rev-parse", "133d82c").strip(), "source_manifest": manifest(),
           "saved_prediction_recomputations": recompute_predictions(), "kernel": audit_kernel(),
           "O24_overlap_and_scores": overlap24(), "O20_diagonal": diagonal_check(),
           "O23_checks": check23(), "integer_chi_symmetry_reference": symmetry_null_integer()}
    print("saved summaries and exact diagonal checks complete", flush=True)
    out["function_field_spectral_diagnostic"] = spectral_ff()
    print("function field diagnostics complete", flush=True)
    out["integer_Liouville_dilation_exploratory"] = liouville_dilation()
    out["scope"] = "Large CUDA runs and O25 not rerun. New diagnostics are exploratory; original rules and verdicts preserved."
    OUT.write_text(json.dumps(out, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "O24": out["O24_overlap_and_scores"],
                      "Liouville": out["integer_Liouville_dilation_exploratory"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
