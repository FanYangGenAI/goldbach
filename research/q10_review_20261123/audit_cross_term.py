"""Round-10 audit: Q7 numerics, residual compensation, local Hessian predictor.

Run from repository root. Reuses the independently derived Q9 smooth model;
reads fixed git objects and the committed Q9 audit. Never sieves to 1e9.
All new Hessian-predictor comparisons are exploratory, not preregistered.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import warnings

import numpy as np
from scipy import integrate
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
REF = "efea332ac5e6e28b148d53f179f17548a58313bc"
OUT = Path(__file__).with_name("audit_results.json")
spec = importlib.util.spec_from_file_location("q9_model", ROOT / "research/q9_review_20261123/audit_residuals.py")
q9 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q9)


def git_json(path):
    raw = subprocess.check_output(["git", "show", f"{REF}:{path}"], cwd=ROOT)
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def metrics(e, c):
    e = np.asarray(e, float); c = np.asarray(c, float)
    ec = e-e.mean(); cc = c-c.mean()
    return {"rho": float(np.corrcoef(ec, cc)[0, 1]),
            "sd_ratio": float(np.linalg.norm(cc)/np.linalg.norm(ec)),
            "R2_pred": float(1-np.linalg.norm(ec-cc)**2/np.linalg.norm(ec)**2),
            "mean_e": float(e.mean()), "mean_c": float(c.mean())}


def center_mod3(a, rows):
    a = np.asarray(a, float).copy()
    for div in (True, False):
        ii = [i for i, r in enumerate(rows) if (r["mod3"] == 0) == div]
        a[ii] -= a[ii].mean()
    return a


def a1(x): return 1/math.log(x)


def a1cut(x): return (1-x**(-.45))/math.log(x)


def da1cut(x):
    t=math.log(x); v=x**(-.45)
    return .45*v/t-(1-v)/t**2


def d2a1cut(x):
    t=math.log(x); v=x**(-.45)
    return -.45**2*v/t-2*.45*v/t**2+2*(1-v)/t**3


def smooth_parts(model, n):
    def ff(u):
        x=n*u; y=n-x
        return np.array([2*a1(x)*a1(y),
                         a1(x)*float(model.a2(y))+a1(y)*float(model.a2(x)),
                         da1cut(x)*a1cut(y)+da1cut(y)*a1cut(x),
                         d2a1cut(x)*a1cut(y)+d2a1cut(y)*a1cut(x)])
    points=sorted({2/n,1e-5,.001,.05,.5})
    vals=np.zeros(4); error=0.0
    for lo,hi in zip(points,points[1:]):
        v,err=integrate.quad_vec(ff,lo,hi,epsabs=1e-13,epsrel=1e-11)
        vals+=n*v;error+=n*err
    return vals,error


def local_parts(model, n):
    odd=model.odd
    beta=1/(odd*(odd-2))
    sums=np.array([beta.sum(),np.sum(beta*np.log(odd)),np.sum(beta*np.log(odd)**2)])
    lr=np.log(1-1/(odd-1)**2).sum()
    for p in q9.factors(n):
        logs=np.array([1.,math.log(p),math.log(p)**2])
        if 2<p<=1_000_000:
            sums-=logs/(p*(p-2))
            lr-=math.log(1-1/(p-1)**2)
        sums-=logs/p
        lr+=math.log(p/(p-1))
    return math.exp(lr),sums


def direct_identity_trial(model, rows, low_order_rows):
    selected=[r for r in rows if r['src']=='Q4b' and r['X']==10**6][:3]
    limit=max(r['N'] for r in selected)
    om=np.zeros(limit+1,np.uint8)
    for prime in q9.primes(limit):
        power=int(prime)
        while power<=limit:
            om[power::power]+=1
            power*=int(prime)
    grid=np.linspace(math.log(2),math.log(limit),16385)
    a2tab=model.a2(np.exp(grid))
    nvals=np.arange(2,limit+1)
    d1=np.zeros(limit+1);d1[2:]=1/np.log(nvals)
    d2=np.zeros(limit+1);d2[2:]=np.interp(np.log(nvals),grid,a2tab)
    f1=(om==1).astype(float);f2=(om==2).astype(float)
    q1=f1-d1;q2=f2-d2
    result=[]
    for r in selected:
        n=r['N'];left=slice(2,n-1);right=slice(n-2,1,-1)
        h3=2*np.dot(f1[left],f2[right])
        b3=low_order_rows[n]
        e=h3-b3
        c=2*(np.dot(d1[left],q2[right])+np.dot(d2[left],q1[right]))
        u=2*np.dot(q1[left],q2[right])
        a=2*np.dot(d1[left],d2[right])-b3
        result.append({'N':n,'h3_recount':int(h3),'e3':float(e),'linear_c3':float(c),
                       'quadratic_pair_u3_direct':float(u),'local_compensation_a3_direct':float(a),
                       'identity_E_minus_C_U_A':float(e-c-u-a)})
    return result


def main():
    data,sha=git_json("research/Q7_crossterm/out/q7_summary.json")
    q4,_=git_json("research/Q1_fugacity/out/q4b_summary.json")
    q5,_=git_json("research/Q5_background/out/q5d_summary.json")
    hs={}
    for scale,group in q4.items():
        if isinstance(group,dict) and 'rows' in group:
            for r in group['rows']:hs['Q4b',r['N']]=r['h']
    for r in q5['rows']:hs['Q5d',r['N']]=r['h']
    rows=data['rows'];model=q9.Background()
    out={'source_commit':REF,'source_sha256':sha,'actual_n_rows':len(rows),
         'source_verdict':data['verdict'],'groups':{},'per_N':[]}
    # Independent coefficient extraction: differentiate the density at zero.
    zr=.25*np.exp(2j*np.pi*np.arange(32)/32)
    rounded=np.round(zr.real,12)+1j*np.round(zr.imag,12)
    # Reproduce Q6's prime-log arithmetic for the 32-node extraction; compare
    # with a direct derivative rather than relying only on series agreement.
    pp=model.pr[model.pr<=200_000]
    ps=pp[None,:]**(-(1-model.y)[:,None])
    log1mp=np.log1p(-ps).sum(1)
    c2=np.zeros(len(model.y),complex)
    for zz,zz0 in zip(rounded,zr):
        logb=zz*log1mp-np.log(1-zz*ps).sum(1)+(zz*zz-zz)/2*model.tail
        cw=np.sin(np.pi*zz)/np.pi*model.w*np.exp(zz*model.logzeta+logb)
        c2+=cw/zz0**2/32
    del ps
    exact_weights=model.w*model.ell
    lg=np.linspace(math.log(2),math.log(1.2e9),16385)
    powers=np.exp(-lg[:,None]*model.y)
    tab=(powers@c2).real
    direct=powers@exact_weights
    dl=lg[1]-lg[0]
    bound_second=float(np.sum(abs(exact_weights)*model.y**2*2**(-model.y)))
    interp_bound=dl**2/8*bound_second
    fft_diff=float(np.max(abs(tab-direct)))
    fft_uniform=float(np.sum(abs(c2-exact_weights)*2**(-model.y)))
    out['a2_audit']={'max_32point_vs_direct_coefficient_difference':fft_diff,
                     'uniform_32point_vs_direct_difference_bound_finite_model':fft_uniform,
                     'log_grid_spacing':float(dl),'linear_interpolation_abs_bound_finite_model':float(interp_bound),
                     'second_derivative_abs_bound':bound_second}
    def tab_a2(x):return float(np.interp(math.log(x),lg,tab))
    representatives=set()
    for r in rows:
        representatives.add((r['src'],r['X']))
    done=set();quadratures=[];b3_values={}
    for i,r in enumerate(rows):
        n=r['N'];h=hs[r['src'],n]
        vals,quaderr=smooth_parts(model,n)
        i11,i12,m11,m20=vals
        s,(u0,u1,u2)=local_parts(model,n)
        b2=s*i11
        b3=s*(2*i12+2*u0*i11-2*u1*m11)
        source_b3=h[3]-n*r['e3']
        b3_values[n]=b3
        local_hessian=s*u2*m20
        acomp=2*i12-source_b3
        v=exact_weights
        sup_a2=float(np.sum(abs(v)*2**(-model.y)))
        tv_a2=float(np.sum(abs(v)*(2**(-model.y)-(n-2)**(-model.y))))
        endpoints=abs(a1(2)*float(model.a2(n-2)))+abs(float(model.a2(2))*a1(n-2))
        em12_bound=.5*(endpoints+a1(2)*tv_a2+sup_a2*(a1(2)-a1(n-2)))
        # Worst-case over both the prime sum and smooth integral. Bounds are
        # for the finite model, not a proof of the infinite prime-tail error.
        a2_error=interp_bound+fft_uniform
        c3_interp_bound=(2*n+4*n*a1(2))*a2_error
        # A generic forward-error bound for positive/mixed finite sums, using
        # at most n additions and integral total-variation upper bounds.
        eps=np.finfo(float).eps
        gamma=n*eps/(1-n*eps)
        total_a1_bound=integrate.quad(lambda x: 1/math.log(x),2,n,epsrel=1e-10)[0]+a1(2)
        # Integral of the absolute coefficient majorant, plus endpoint bound.
        total_a2_bound=float(np.sum(abs(v)*(n**(1-model.y)-2**(1-model.y))/(1-model.y)))+sup_a2
        c3_sum_roundoff_bound=2*gamma*(total_a1_bound+total_a2_bound)
        out['per_N'].append({'src':r['src'],'X':r['X'],'N':n,'mod3':r['mod3'],
            'e3_over_N':r['e3'],'linear_c3_over_N':r['c3'],
            'local_compensation_approx_over_N':float(acomp/n),
            'quadratic_pair_reconstructed_approx_over_N':float(r['e3']-r['c3']-acomp/n),
            'hessian_prediction_over_N':float(local_hessian/n),
            'b2_source_minus_direct_over_N':float((r['b2_fft']-b2)/n),
            'b3_source_minus_direct_over_N':float((source_b3-b3)/n),
            'em_c3_discrete_vs_integral_abs_bound':float(4*em12_bound),
            'a2_lookup_c3_abs_bound':float(c3_interp_bound),
            'generic_sum_roundoff_c3_abs_bound':float(c3_sum_roundoff_bound),
            'adaptive_integral_estimated_abs_error':float(quaderr)})
        group=(r['src'],r['X'])
        if group not in done:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                low,el=integrate.quad(lambda x:a1(x)*tab_a2(n-x),2,n/2,limit=400)
                high,eh=integrate.quad(lambda x:a1(x)*tab_a2(n-x),n/2,n-2,limit=400)
            quadratures.append({'N':n,'source_style_minus_direct_I12':float(low+high-i12),
                                 'source_style_estimated_error':float(el+eh),
                                 'messages':[str(w.message) for w in caught]})
            done.add(group)
        if i%50==0:print('audited',i,flush=True)
    out['quadrature_checks']=quadratures
    pp=out['per_N']
    for src,x in sorted(representatives):
        ids=[i for i,r in enumerate(rows) if r['src']==src and r['X']==x]
        rr=[rows[i] for i in ids];pr=[pp[i] for i in ids]
        ee=np.array([r['e3'] for r in rr]);cc=np.array([r['c3'] for r in rr])
        aa=np.array([p['local_compensation_approx_over_N'] for p in pr])
        hh=np.array([p['hessian_prediction_over_N'] for p in pr])
        ec=center_mod3(ee,rr);lc=center_mod3(cc,rr);ac=center_mod3(aa,rr);hc=center_mod3(hh,rr)
        qc=ec-lc;uc=qc-ac
        max_error=np.array([(p['em_c3_discrete_vs_integral_abs_bound']+p['a2_lookup_c3_abs_bound']
                             +p['generic_sum_roundoff_c3_abs_bound'])/p['N'] for p in pr])
        rho=float(np.corrcoef(ec,lc)[0,1]);sratio=float(np.linalg.norm(lc)/np.linalg.norm(ec))
        out['groups'][f'{src} {x}']={'n':len(rr),'linear_within_mod3':metrics(ec,lc),
            'hessian_pooled_exploratory':metrics(ee,hh),'hessian_within_mod3_exploratory':metrics(ec,hc),
            'hessian_vs_Q_within_mod3_exploratory':metrics(qc,hc),
            'linear_plus_hessian_vs_E_within_mod3_exploratory':metrics(ec,lc+hc),
            'local_compensation_sd_over_residual_sd':float(np.linalg.norm(ac)/np.linalg.norm(ec)),
            'pair_reconstructed_vs_compensation_correlation':float(np.corrcoef(uc,ac)[0,1]),
            'Q_sd_over_E_sd':float(np.linalg.norm(qc)/np.linalg.norm(ec)),
            'Q_E_correlation':float(np.corrcoef(qc,ec)[0,1]),
            'identity_check_Q_sd_ratio':float(math.sqrt(1+sratio**2-2*rho*sratio)),
            'finite_model_lookup_lattice_roundoff_bound_over_residual_norm':float(np.linalg.norm(max_error)/np.linalg.norm(ec)),
            'hessian_remaining_sd_ratio':float(np.linalg.norm(ec-hc)/np.linalg.norm(ec))}
    prior=json.loads((ROOT/'research/q9_review_20261123/audit_results.json').read_text(encoding='utf-8'))
    prior={r['N']:r for r in prior['one_system_cross_trial']['per_N']}
    out['low_scale_discrete_cross_comparison']=[{'N':r['N'],
             'c3_source_minus_prior_exact_discrete':float((r['c3']-prior[r['N']]['cross3_over_N'])*r['N'])}
             for r in rows if r['src']=='Q4b' and r['X']==10**6]
    out['direct_discrete_identity_trials']=direct_identity_trial(model,rows,b3_values)
    out['limitations']=['No independent recount of the 1e9 one-system sums was run.',
        'Finite-model error estimates do not certify the infinite prime product or prime-tail approximation.',
        'The local Hessian prediction is derived from a formal local Mellin model, not a proven asymptotic theorem.',
        'All Hessian tests use previously explored samples and are exploratory.',
        'The quadratic pair term is reconstructed algebraically, not independently measured.']
    OUT.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'out':str(OUT),'n':len(rows),'large_scale':out['groups']['Q5d 1000000000']},indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
