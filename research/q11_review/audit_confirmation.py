"""Round-11 audit of Q8 and exploratory cubic local-Mellin prediction.

Fixed git inputs; no billion-scale sieve or new fixed-sum observations.
The Q8 quadratic result remains a confirmation. Every cubic comparison below
is a new exploratory diagnostic, not part of its preregistered verdict.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from scipy import integrate, special
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
RESULT="59a93e175ca4fc417c0e840a3b539e72ba6c62e7"
PREREG="f1d9b8ffca6d69229e36f299b4215060e2d0cb7e"
OUT=Path(__file__).with_name("audit_results.json")
spec=importlib.util.spec_from_file_location("q10_audit",ROOT/"research/q10_review_20261123/audit_cross_term.py")
q10=importlib.util.module_from_spec(spec);spec.loader.exec_module(q10)


def raw(path,ref=RESULT):
    return subprocess.check_output(['git','show',f'{ref}:{path}'],cwd=ROOT)


def load(path,ref=RESULT):return json.loads(raw(path,ref))


def center(v,rows):return q10.center_mod3(v,rows)


def derivative_a1cut(x,r):
    logx=math.log(x)
    return (-1)**r*math.factorial(r)*special.gammainc(r+1,.45*logx)/logx**(r+1)


def shift_remainder(t):
    # exp(t)-1-t, with a stable small-t branch.
    return np.where(abs(t)<1e-3,t*t/2*(1+t/3+t*t/12+t**3/60),np.expm1(t)-t)


def shift_base(model,limit):
    pp=model.odd[model.odd<=limit]
    ans=np.zeros(len(model.y))
    for j in range(0,len(pp),2048):
        primes=pp[j:j+2048]
        t=model.y[:,None]*np.log(primes)
        ans+=(shift_remainder(t)/(primes*(primes-2))).sum(1)
    return ans


def resummed_local_candidate(model,n,s,base,limit):
    # Exactly sums the selected local model's Mellin orders r>=2 at m=3.
    # This is not a fit or a search over truncation orders.
    f=base.copy()
    for prime in q10.q9.factors(n):
        term=shift_remainder(model.y*math.log(prime))
        if 2<prime<=limit:f-=term/(prime*(prime-2))
        f-=term/prime
    a,b=model.y[:,None],model.y[None,:]
    ib=n**(1-a-b)*model.beta*(1-special.betainc(1-a,1-b,2/n)-special.betainc(1-b,1-a,2/n))
    return float(2*s*(model.w*f)@ib@model.w)


def cubic_moment(n):
    def f(u):
        x=n*u;y=n-x
        return derivative_a1cut(x,3)*q10.a1cut(y)+derivative_a1cut(y,3)*q10.a1cut(x)
    points=sorted({2/n,1e-5,.001,.05,.5})
    val=0.;error=0.
    for lo,hi in zip(points,points[1:]):
        v,e=integrate.quad(f,lo,hi,epsabs=1e-14,epsrel=1e-11,limit=300)
        val+=n*v;error+=n*e
    return val,error


def metrics(y,p):return q10.metrics(y,p)


def weighted_metrics(y,p,weights):
    w=np.asarray(weights);w=w/w.sum()
    yy=y-np.sum(w*y);pp=p-np.sum(w*p)
    sy=np.sum(w*yy*yy);sp=np.sum(w*pp*pp);cov=np.sum(w*yy*pp)
    return {'rho':float(cov/math.sqrt(sy*sp)),'sd_ratio':float(math.sqrt(sp/sy)),
            'R2_pred':float(1-np.sum(w*(yy-pp)**2)/sy)}


def main():
    data=load('research/Q8_hessian/out/q8_summary.json');rows=data['rows']
    old=load('research/Q7_crossterm/out/q7_summary.json')['rows']
    used={r['N'] for r in load('research/Q5_background/out/q5d_summary.json')['rows']}
    rng=np.random.default_rng(2026112401)
    s3=[];s0=[]
    while len(s3)<40 or len(s0)<40:
        n=int(2*rng.integers(10**9//2,11*10**8//2))
        if n in used or n in s3 or n in s0:continue
        if n%3==0 and len(s3)<40:s3.append(n)
        elif n%3!=0 and len(s0)<40:s0.append(n)
    prereg_files=subprocess.check_output(['git','ls-tree','-r','--name-only',PREREG,'research/Q8_hessian'],cwd=ROOT).decode().splitlines()
    old_design=raw('research/Q8_hessian/DESIGN.md',PREREG)
    result_design=raw('research/Q8_hessian/DESIGN.md')
    out={'source_commit':RESULT,'prereg_commit':PREREG,
         'source_sha256':hashlib.sha256(raw('research/Q8_hessian/out/q8_summary.json')).hexdigest(),
         'sample_checks':{'n':len(rows),'n_unique':len({r['N'] for r in rows}),
             'exact_seed_sequence_match':[r['N'] for r in rows]==s3+s0,
             'all_in_window_even':all(10**9<=r['N']<1.1e9 and r['N']%2==0 for r in rows),
             'strata_sizes':[sum(r['mod3']==0 for r in rows),sum(r['mod3']!=0 for r in rows)],
             'overlap_with_all_old_samples':len({r['N'] for r in rows}&{r['N'] for r in old}),
             'all_h3_even':all(r['h3']%2==0 for r in rows)},
         'prereg_record':{'files':prereg_files,'runner_present':any(p.endswith('q8_confirm.py') for p in prereg_files),
             'design_prefix_unchanged':result_design.startswith(old_design)},
         'per_N':[]}
    y=np.array([r['target'] for r in rows]);p=np.array([r['pred'] for r in rows])
    yc=center(y,rows);pc=center(p,rows)
    out['recomputed_primary_metrics']=metrics(yc,pc)
    m=out['recomputed_primary_metrics'];out['recomputed_verdict']='support' if m['rho']>.5 and .5<=m['sd_ratio']<=2 and m['R2_pred']>.25 else 'not_support'
    out['R2_decomposition']={'unexplained_fraction':float(1-m['R2_pred']),
          'orthogonal_fraction':float(1-m['rho']**2),
          'amplitude_calibration_fraction':float((m['sd_ratio']-m['rho'])**2),
          'remaining_sd_ratio':float(math.sqrt(1-m['R2_pred']))}
    out['means']={}
    for div in (True,False):
        ii=[i for i,r in enumerate(rows) if (r['mod3']==0)==div]
        resid=y[ii]-p[ii]
        out['means'][str(div)]={'n':len(ii),'target_mean':float(y[ii].mean()),'prediction_mean':float(p[ii].mean()),
            'residual_mean':float(resid.mean()),'residual_sem':float(resid.std(ddof=1)/math.sqrt(len(ii)))}
    weights=np.array([1/3/40 if r['mod3']==0 else 2/3/40 for r in rows])
    out['natural_strata_reweighting_descriptive']=weighted_metrics(yc,pc,weights)
    model=q10.q9.Background()
    shift200=shift_base(model,200_000);shift1m=shift_base(model,1_000_000)
    odd=model.odd
    mask=odd<=200_000
    base2=float(np.sum(np.log(odd[mask])**2/(odd[mask]*(odd[mask]-2))))
    base3=float(np.sum(np.log(odd[mask])**3/(odd[mask]*(odd[mask]-2))))
    def local(n):
        s,(u0,u1,u2full)=q10.local_parts(model,n)
        l2=base2;l3=base3
        for prime in q10.q9.factors(n):
            lp=math.log(prime)
            if 2<prime<=200_000:
                l2-=lp**2/(prime*(prime-2));l3-=lp**3/(prime*(prime-2))
            l2-=lp**2/prime;l3-=lp**3/prime
        return s,l2,l3,u2full
    for i,r in enumerate(rows):
        n=r['N'];vals,err=q10.smooth_parts(model,n)
        i11,i12,m11,m20=vals
        s,(u0,u1,_)=q10.local_parts(model,n)
        b2=s*i11;b3=s*(2*i12+2*u0*i11-2*u1*m11)
        s,l2,l3,l2full=local(n)
        ell2=s*l2*m20
        m30,m30err=cubic_moment(n)
        ell3=-s*l3*m30/3
        resum200=resummed_local_candidate(model,n,s,shift200,200_000)
        resum1m=resummed_local_candidate(model,n,s,shift1m,1_000_000)
        c3_from_parts=2*(r['S12']+r['S21']-2*r['a12'])
        target_recompute=(r['h3']-r['b3']-c3_from_parts)/n
        c3_independent_smooth=2*(r['S12']+r['S21']-2*i12)
        c2_independent_smooth=2*(r['S1']-i11)
        out['per_N'].append({'N':n,'mod3':r['mod3'],'target':r['target'],'pred':r['pred'],
             'cubic_pred_exploratory':float(ell3/n),
             'resummed_local_pred_exploratory':float(resum200/n),
             'resummed_prime_cut_200k_to_1m_change_over_N':float((resum1m-resum200)/n),
             'target_independent_smooth':float((r['h3']-b3-c3_independent_smooth)/n),
             'pred_independent_smooth':float(ell2/n),
             'm2_residual_over_N_exploratory':float((r['h2']-b2)/n),
             'm2_linear_over_N_exploratory':float(c2_independent_smooth/n),
             'target_reconstruction_error':float(target_recompute-r['target']),
             'ell2_recompute_error_over_N':float((ell2-r['ell3'])/n),
             'ell2_prime_cut_200k_to_1m_change_over_N':float(s*(l2full-l2)*m20/n),
             'b2_recompute_error_over_N':float((b2-r['b2'])/n),
             'b3_recompute_error_over_N':float((b3-r['b3'])/n),
             'a12_source_minus_direct':float(r['a12']-i12),
             'quad_recompute_estimate':float(err),
             'cubic_moment_error_estimate':float(s*abs(l3)*m30err/3)})
        if i%20==0:print('fresh audit',i,flush=True)
    c=np.array([r['cubic_pred_exploratory'] for r in out['per_N']]);cc=center(c,rows)
    out['independent_smooth_recomputed_metrics']=metrics(
        center([r['target_independent_smooth'] for r in out['per_N']],rows),
        center([r['pred_independent_smooth'] for r in out['per_N']],rows))
    out['m2_linear_diagnostic_exploratory']=metrics(
        center([r['m2_residual_over_N_exploratory'] for r in out['per_N']],rows),
        center([r['m2_linear_over_N_exploratory'] for r in out['per_N']],rows))
    out['cubic_exploration_fresh_previously_used_for_quadratic_confirmation']={
          'cubic_vs_remaining':metrics(yc-pc,cc),'quadratic_plus_cubic_vs_target':metrics(yc,pc+cc),
          'group_cubic_means':{str(div):float(c[[i for i,r in enumerate(rows) if (r['mod3']==0)==div]].mean()) for div in (True,False)}}
    full=np.array([r['resummed_local_pred_exploratory'] for r in out['per_N']]);fullc=center(full,rows)
    out['resummed_local_exploration_fresh']={'total_vs_target':metrics(yc,fullc),
        'higher_orders_vs_remaining':metrics(yc-pc,fullc-pc),
        'group_resummed_means':{str(div):float(full[[i for i,r in enumerate(rows) if (r['mod3']==0)==div]].mean()) for div in (True,False)}}
    prior=json.loads((ROOT/'research/q10_review_20261123/audit_results.json').read_text(encoding='utf-8'))['per_N']
    old_results=[]
    for i,r in enumerate(prior):
        n=r['N'];s,l2,l3,_=local(n);m30,_=cubic_moment(n)
        old_results.append({'src':r['src'],'X':r['X'],'N':n,'mod3':r['mod3'],
              'target':r['e3_over_N']-r['linear_c3_over_N'],'quadratic':r['hessian_prediction_over_N'],
              'cubic':float(-s*l3*m30/3/n)})
    out['cubic_exploration_old_cohorts']={}
    for src,x in sorted({(r['src'],r['X']) for r in old_results}):
        rr=[r for r in old_results if r['src']==src and r['X']==x]
        yy=center([r['target'] for r in rr],rr);pp=center([r['quadratic'] for r in rr],rr);cc0=center([r['cubic'] for r in rr],rr)
        out['cubic_exploration_old_cohorts'][f'{src} {x}']={'cubic_vs_remaining':metrics(yy-pp,cc0),
                   'quadratic_plus_cubic_vs_target':metrics(yy,pp+cc0)}
    grid=np.geomspace(2,1.1e9,100)
    out['derivative_checks']={'max_D2_gamma_vs_closed_form':float(max(abs(derivative_a1cut(x,2)-q10.d2a1cut(x)) for x in grid)),
        'max_D3_gamma_vs_laplace_quadrature':float(max(abs(derivative_a1cut(x,3)+np.sum(model.w*model.y**3*x**(-model.y))) for x in grid))}
    out['limitations']=['The confirmed quadratic result is kept separate from the new exploratory cubic tests.',
        'The resummed local candidate is also exploratory; it sums a fixed analytic local kernel, not an optimized truncation.',
        'No independent recount of the 1e9 h2/h3 or one-system discrete sums.',
        'Numerical checks do not prove the formal local Mellin ansatz for true Goldbach counts.',
        'Git records and deterministic regeneration do not certify unrecorded private execution history.']
    OUT.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'out':str(OUT),'sample_checks':out['sample_checks'],
          'primary':out['recomputed_primary_metrics'],
          'cubic':out['cubic_exploration_fresh_previously_used_for_quadratic_confirmation']},indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
