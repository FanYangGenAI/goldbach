"""Read-only E1 audit and exact reflection/Fourier examples for direction B.

Recomputes statistics from the saved 90 tables. The old Q1b model is evaluated
with convergent prime series instead of importing its expensive top level.
No new large-scale number-theory experiment is run.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from scipy.special import rgamma
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
REF='2ce07cf'
OUT=Path(__file__).with_name('audit_results.json')


def entropy_amplitudes(a):
    a=np.asarray(a,complex);a=a/np.linalg.norm(a)
    lam=np.linalg.svd(a,compute_uv=False)**2
    lam=lam[lam>1e-300]
    return float(-np.sum(lam*np.log(lam)))


def ent(t):
    t=np.maximum(np.asarray(t,float),0);p=t/t.sum()
    def h(a):
        a=a[a>0];return float(-np.sum(a*np.log(a)))
    return entropy_amplitudes(np.sqrt(p)),h(p.sum(0))+h(p.sum(1))-h(p.ravel())


def primes(n):
    s=np.ones(n+1,bool);s[:2]=False
    for p in range(2,math.isqrt(n)+1):
        if s[p]:s[p*p::p]=False
    return np.flatnonzero(s)


def factors(n):
    out={};p=2
    while p*p<=n:
        while n%p==0:out[p]=out.get(p,0)+1;n//=p
        p+=1
    if n>1:out[n]=out.get(n,0)+1
    return out


class FrozenLeadingModel:
    """Reproduce the finite-prime Q1b kernel, including its big-prime rule."""
    def __init__(self):
        self.z=np.exp(2j*np.pi*np.arange(128)/128)
        self.Z=self.z[:,None];self.W=self.z[None,:]
        self.q=(self.Z-1)*(self.W-1)
        pp=primes(1_000_000).astype(float)
        v=1/pp
        linear=np.sum(np.log1p(-v)+v)
        logb=self.z*linear
        power=v*v
        for k in range(2,81):
            logb+=self.z**k*np.sum(power)/k
            power*=v
        bz=np.exp(logb)
        self.marginal=(bz*rgamma(self.z))[:,None]*(bz*rgamma(self.z))[None,:]
        pp=pp[pp>=5];v=1/(pp-1)**2;power=v.copy()
        logtail=np.zeros((128,128),complex)
        for k in range(1,33):
            logtail-=self.q**k*np.sum(power)/k
            power*=v
        self.rtail=np.exp(logtail)

    def j(self,p,e):
        z,w=self.Z,self.W
        zp=(p-1)/(p-z);wp=(p-1)/(p-w)
        value=1-2/p+(z*zp+w*wp)/p
        for _ in range(e):value=1-1/p+z*w*value/p
        return value

    def local(self,p,e):
        return self.j(p,e)/(((p-1)/(p-self.Z))*((p-1)/(p-self.W)))

    def coefficients(self,n):
        fac=factors(n)
        r=self.rtail.copy()*self.local(2,fac[2])
        r*=self.local(3,fac[3]) if 3 in fac else 1-self.q/4
        for p,e in fac.items():
            if p<5:continue
            if p<=1_000_000:r*=self.local(p,e)/(1-self.q/(p-1)**2)
            else:r*=self.j(p,e)  # matches the frozen implementation exactly
        g=n*np.exp((self.Z+self.W-2)*math.log(math.log(n)))*r*self.marginal
        return (np.fft.fft2(g)/128**2).real


def main():
    path='research/E1_entanglement/out/e1_summary.json'
    data_raw=subprocess.check_output(['git','show',f'{REF}:{path}'],cwd=ROOT)
    data=json.loads(data_raw);rows=data['rows'];model=FrozenLeadingModel()
    out={'source_commit':subprocess.check_output(['git','rev-parse',REF],cwd=ROOT).decode().strip(),
         'source_sha256':hashlib.sha256(data_raw).hexdigest(),'n_rows':len(rows),'source_H_local':data['H_local'],
         'per_N':[],'groups':{}}
    for r in rows:
        t=np.array(r['T'],float);s,mi=ent(t)
        p=t/t.sum();a=p.sum(1);b=p.sum(0)
        replacement=t.copy();replacement[1,1]=t.sum(1)[1]*t.sum(0)[1]/t.sum()
        modified_p=replacement/replacement.sum()
        zero=t.copy();zero[1,1]=0
        binary=np.array([[p[1,1],a[1]-p[1,1]],
                         [b[1]-p[1,1],1-a[1]-b[1]+p[1,1]]])
        c=model.coefficients(r['N']);sm,_=ent(c[:21,:21])
        out['per_N'].append({'N':r['N'],'X':r['X'],'mod3':r['mod3'],
            'entropy_recompute_error':s-r['S'],'MI_recompute_error':mi-r['I'],
            'dS11_recompute_error':s-ent(replacement)[0]-r['dS11'],
            'model_entropy_recompute_error':sm-r['S_model'],
            'model_negative_mass_fraction_21':float(np.maximum(-c[:21,:21],0).sum()/r['N']),
            'model_negative_mass_fraction_full':float(np.maximum(-c,0).sum()/r['N']),
            'missing_real_mass_fraction':float(1-t.sum()/(r['N']-3)),
            'P11':float(p[1,1]),'prime_pair_lift':float(p[1,1]/(a[1]*b[1])),
            'binary_prime_MI':ent(binary)[1],
            'entropy_after_zeroing11':ent(zero)[0],
            'replace11_max_row_marginal_change':float(np.max(abs(modified_p.sum(1)-a)))})
    for x in (10**6,10**7,10**8):
        rr=[r for r in rows if r['X']==x];ss=np.array([r['S'] for r in rr]);mm=np.array([r['S_model'] for r in rr])
        sc=ss.copy();mc=mm.copy();strata={}
        for div in (True,False):
            ii=[i for i,r in enumerate(rr) if (r['mod3']==0)==div]
            sc[ii]-=ss[ii].mean();mc[ii]-=mm[ii].mean()
            strata[str(div)]={'n':len(ii),'corr':float(np.corrcoef(ss[ii],mm[ii])[0,1])}
        pp=[r for r in out['per_N'] if r['X']==x]
        out['groups'][str(x)]={'corr':float(np.corrcoef(ss,mm)[0,1]),
            'within_mod3_corr':float(np.corrcoef(sc,mc)[0,1]),
            'between_mod3_variance_fraction':float(1-np.sum(sc*sc)/np.sum((ss-ss.mean())**2)),
            'strata':strata,'median_dS11_over_S':float(np.median([r['dS11']/r['S'] for r in rr])),
            'median_prime_pair_lift':float(np.median([p['prime_pair_lift'] for p in pp])),
            'max_missing_mass':float(max(p['missing_real_mass_fraction'] for p in pp)),
            'max_model_negative_mass_fraction':float(max(p['model_negative_mass_fraction_21'] for p in pp))}
    # Same probabilities, different coherent phases: entropy is not determined
    # by measurement probabilities if phase conventions are left unspecified.
    uniform=np.ones((2,2))/4
    positive=np.sqrt(uniform);phase=positive.copy();phase[1,1]*=-1
    absent=np.array([[0,.5],[.5,0]])
    out['encoding_examples']={'same_uniform_P_positive_amplitudes_S':entropy_amplitudes(positive),
        'same_uniform_P_joint_phase_S':entropy_amplitudes(phase),
        'uniform_P_classical_MI':ent(uniform)[1],
        'P11_zero_but_coherent_entropy':ent(absent)[0]}
    # Exact group/operator identities on a small finite cyclic group.
    length=17;v=np.arange(length,dtype=float);idx=np.arange(length)
    reflect=lambda f,n:f[(n-idx)%length]
    relation=reflect(np.roll(reflect(v,0),1),0)
    maxn=4096;L=16384
    f=np.zeros(L);f[primes(maxn)]=1
    a=np.zeros(L);a[2:maxn+1]=1/np.log(np.arange(2,maxn+1))
    q=f-a;Q=np.fft.fft(q)
    convolution=np.fft.ifft(Q*Q).real
    ns=[100,1000,4000]
    errors=[]
    for n in ns:
        direct=float(np.dot(q,q[(n-np.arange(L))%L]))
        errors.append(abs(direct-convolution[n]))
    fshift=np.roll(f,1)
    power_error=float(np.max(abs(abs(np.fft.fft(f))**2-abs(np.fft.fft(fshift))**2)))
    pair=np.fft.ifft(np.fft.fft(f)**2).real
    shifted_pair=np.fft.ifft(np.fft.fft(fshift)**2).real
    out['group_phase_checks']={'dihedral_RTR_equals_Tinverse':bool(np.array_equal(relation,np.roll(v,-1))),
        'max_reflection_vs_Qsquare_error':float(max(errors)),
        'same_power_after_translation_max_numerical_error':power_error,
        'different_pair_counts_same_power_example':{'N':100,'original':float(pair[100]),'shifted':float(shifted_pair[100])}}
    out['limitations']=['No independent rerun of the 1e8 integer/null tables.',
        'The saved null entropies cannot be independently recomputed without the unsaved shuffled tables.',
        'The small phase examples are information-loss demonstrations, not arithmetic prime surrogates.',
        'No direction-B predictive hypothesis is claimed validated by these exact identities.']
    OUT.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'out':str(OUT),'groups':out['groups'],
        'max_entropy_recompute_error':max(abs(p['entropy_recompute_error']) for p in out['per_N']),
        'max_model_recompute_error':max(abs(p['model_entropy_recompute_error']) for p in out['per_N']),
        'encoding_examples':out['encoding_examples'],'group_phase_checks':out['group_phase_checks']},indent=2))


if __name__=='__main__':
    with threadpool_limits(limits=2):main()
