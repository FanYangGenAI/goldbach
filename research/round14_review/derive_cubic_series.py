"""Algebraic expansion CONDITIONAL ON the unproved integer-moment candidates.

This script proves an algebraic implication between explicit expressions. It
does not prove that the candidate polynomials count the arithmetic objects.
Requires SymPy. Run independently of the numerical audit.
"""
import json
from pathlib import Path

import sympy as s

q, u = s.symbols("q u", positive=True)
Q = q**3
K = (q*q-1)**2/(6*q)
pi = [q, q*(q-1)/2, (q**3-q)/3]
log_c = sum(pi[d-1].subs(q, 1/u)*s.series(s.log(1-u**(2*d)/(1-u**d)**2), u, 0, 12).removeO()
            for d in (1, 2, 3))
c_series = s.series(s.exp(s.series(log_c, u, 0, 9).removeO()), u, 0, 9).removeO()
types = [(q**3-q)/3, q*q*(q-1)/2, q*(q-1)*(q-2)/6, q*(q-1), q]
a1, a2, a3 = (q-1)/(q-2), (q*q-1)/(q*q-2), (q**3-1)/(q**3-2)
weights = [a3, a1*a2, a1**3, a1**2, a1]
mean_w = sum(n*w for n, w in zip(types, weights))/Q
var_w = s.factor(sum(n*w*w for n, w in zip(types, weights))/Q-mean_w**2)
out = {"status": "conditional on the unproved reconstructed integer moments", "series": {}}
for residue in (1, 2):
    if residue == 1:
        m2 = q*(q-1)**2*(q**6+2*q**5-3*q**3+9*q-3)/36
        moments = [q*(q-1)**2*(q**3+q*q-q-4)/18, q*q*(q-1)**3*(q+2)/12,
                   q*(q-2)*(q-1)**2*(q+1)*(q+2)/36, q*q*(q-1)**2*(q+2)/6,
                   q*(q-1)**2*(q+2)/6]
    else:
        m2 = q*(q-1)*(q+1)*(q**6-2*q**4-q**3+4*q*q+3*q+5)/36
        moments = [q*(q-2)*(q-1)*(q+1)*(q*q+q+1)/18, q*(q-1)*(q+1)*(q**3-q*q-2*q+4)/12,
                   q*(q-2)*(q-1)**2*(q+1)*(q+2)/36, q*(q-1)*(q+1)*(q*q-2)/6,
                   q*q*(q-1)*(q+1)/6]
    assert s.factor(sum(moments)-Q*K) == 0
    var_r = s.factor(m2/Q-K*K)
    cov_r_w = s.factor(sum(w*r for w, r in zip(weights, moments))/Q-K*mean_w)
    var_e = 0
    for i, term in enumerate([var_r, K*K*var_w, -2*K*cov_r_w]):
        expanded = s.series(term.subs(q, 1/u), u, 0, 2).removeO()
        if i:
            expanded *= c_series**(3-i)
        var_e += s.series(expanded, u, 0, 2).removeO()
    answer = s.series(var_e*(q/K).subs(q, 1/u), u, 0, 4)
    out["series"][str(residue)] = str(answer)
    print(residue, answer, flush=True)
out["C3_series"] = str(c_series)
target = Path(__file__).with_name("conditional_series.json")
target.write_text(json.dumps(out, indent=2)+"\n", encoding="utf-8")
