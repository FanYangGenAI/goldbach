"""B1--B3 audit: exact block-sampling means and symmetric diagnostics.

Reproduces all 398 z=100/z=30 draws with real FFTs, saves their 80-vectors,
and keeps the original preregistered statistic distinct from new diagnostics.
Also checks the rough-set parity identity without treating it as new evidence.
"""
from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from scipy import fft, integrate

ROOT=Path(__file__).resolve().parents[2]
REF='fca0ca4'
M=1_100_000;L=2310*953;BLOCK=4096
C2=.6601618158468695739278121100145
OUT=Path(__file__).with_name('audit_results.json')


def load(path):
    return json.loads(subprocess.check_output(['git','show',f'{REF}:{path}'],cwd=ROOT))


def sieve(n):
    s=np.ones(n+1,bool);s[:2]=False
    for p in range(2,math.isqrt(n)+1):
        if s[p]:s[p*p::p]=False
    return s


def odd_factors(n):
    out=[]
    while n%2==0:n//=2
    p=3
    while p*p<=n:
        if n%p==0:
            out.append(p)
            while n%p==0:n//=p
        p+=2
    if n>1:out.append(n)
    return out


def ratio(n,z):
    a=b=1.
    for p in odd_factors(int(n)):
        b*=(p-1)/(p-2)
        if p<=z:a*=(p-1)/(p-2)
    return a/b


def b2(n):
    ss=2*C2
    for p in odd_factors(n):ss*=(p-1)/(p-2)
    points=sorted({2/n,1e-5,.001,.05,.5})
    val=0.
    for lo,hi in zip(points,points[1:]):
        v,_=integrate.quad(lambda u:1/(math.log(n*u)*math.log(n-n*u)),lo,hi,epsabs=1e-13,epsrel=1e-11,limit=300)
        val+=2*n*v
    return ss*val


def toy_mean_check():
    length=32;sv=np.array([2,3,6,7,10]);h=len(sv);m=2;rho=m/h
    mask=np.ones(length//2+1);mask[[0,1,-1]]=0
    xs=[]
    for chosen in itertools.combinations(sv,m):
        x=np.zeros(length);x[list(chosen)]=1
        xs.append(fft.irfft(fft.rfft(x)**2*mask,n=length))
    mu=np.zeros(length);mu[sv]=rho
    cov=np.full((h,h),-rho*(1-rho)/(h-1));np.fill_diagonal(cov,rho*(1-rho))
    c=np.zeros(length)
    for i,a in enumerate(sv):
        for j,b in enumerate(sv):c[(a+b)%length]+=cov[i,j]
    expected=fft.irfft((fft.rfft(mu)**2+fft.rfft(c))*mask,n=length)
    return float(np.max(abs(np.mean(xs,axis=0)-expected)))


def main():
    source=load('research/B3_sieve_null/out/b3_summary.json')
    b2source=load('research/B2_rigidity/out/b2_summary.json')
    ns=np.array(source['Ns'],int);grp=ns%3==0
    def center(x):
        x=np.asarray(x,float).copy()
        for g in (True,False):x[...,grp==g]-=x[...,grp==g].mean(axis=-1,keepdims=True)
        return x
    def stat(v,baseline):return float(np.sum(center((v-baseline)/ns)**2))
    mask_full=np.zeros(L,bool)
    for k in np.arange(0,L,L//210):mask_full[(k+np.arange(-16,17))%L]=True
    mask_full|=mask_full[(-np.arange(L))%L];mask_full[[0,L//2]]=True
    keep=mask_full[:L//2+1];unlocked=~keep
    s=sieve(M);primes=np.flatnonzero(s)
    d=np.zeros(L);d[2:M+1]=1/np.log(np.arange(2,M+1))
    D=fft.rfft(d,workers=2)
    def V(f):
        padded=np.zeros(L);padded[:M+1]=f
        Q=fft.rfft(padded,workers=2)-D
        return fft.irfft(Q*Q*unlocked,n=L,workers=2)[ns]
    vr=V(s);bb=np.array([b2(int(n)) for n in ns])
    tau_full=bb*np.array([1-ratio(n,7) for n in ns])
    real_old=stat(vr,tau_full)
    out={'source_commit':subprocess.check_output(['git','rev-parse',REF],cwd=ROOT).decode().strip(),
         'Ns':ns.tolist(),'source_T_real':source['T_real'],'recomputed_T_real':real_old,
         'toy_exact_mean_max_error':toy_mean_check(),'models':{}}
    used=set()
    for seed,scales,size in [(11,[10**6,10**7,10**8],30),(21,[10**6,10**7,10**8,10**9],10),
        (31,[10**6,10**7,10**8,10**9],10),(41,[10**6,10**7,10**8,10**9],100),
        (51,[10**6,10**7,10**8],60),(61,[10**6,10**7,10**8,10**9],20),
        (71,[10**6,10**7,10**8,10**9],20),(20261125,[10**6,10**7,10**8],30)]:
        rr=np.random.default_rng(seed)
        for scale in scales:used.update(int(x) for x in 2*rr.integers(scale//2,int(1.1*scale)//2,size=size))
    used.update(load('research/B1_phase/out/b1_summary.json')['Ns']);used.update(b2source['Ns'])
    rng=np.random.default_rng(2026112408);g3=[];g0=[]
    while len(g3)<40 or len(g0)<40:
        n=int(2*rng.integers(10**6//2,11*10**5//2))
        if n in used or n in g3 or n in g0:continue
        if n%3==0 and len(g3)<40:g3.append(n)
        elif n%3!=0 and len(g0)<40:g0.append(n)
    out['sampling_checks']={'n_unique':len(set(ns.tolist())),
         'seed_sequence_match':ns.tolist()==g3+g0,'overlap_with_declared_prior':len(set(ns.tolist())&used)}

    def build(z):
        small=primes[primes<=z]
        rough=np.ones(M+1,bool);rough[:z+1]=False
        for p in small:rough[p::p]=False
        rough[:z+1]=False
        sv=np.flatnonzero(rough);blocks=sv//BLOCK
        need=np.bincount(primes[primes>z]//BLOCK,minlength=M//BLOCK+1)
        have=np.bincount(blocks,minlength=M//BLOCK+1)
        starts=np.r_[0,np.cumsum(have)[:-1]]
        mu=np.zeros(M+1);mu[small]=1;mu[sv]=need[blocks]/have[blocks]
        cov_sequence=np.zeros(L)
        for b,h0 in enumerate(have):
            h=int(h0);m=int(need[b])
            if h<=1 or m==0 or m==h:continue
            p=m/h;v=p*(1-p)
            inds=sv[starts[b]:starts[b]+h]
            cov_sequence[2*inds]+=v*h/(h-1)
            start=b*BLOCK;size=min(BLOCK,M+1-start)
            indicator=np.zeros(size);indicator[inds-start]=1
            nfft=fft.next_fast_len(2*size-1)
            conv=fft.irfft(fft.rfft(indicator,n=nfft)**2,n=nfft)[:2*size-1]
            cov_sequence[2*start:2*start+len(conv)]-=v/(h-1)*conv
        padded=np.zeros(L);padded[:M+1]=mu
        mq=fft.rfft(padded,workers=2)-D
        mean_only=fft.irfft(mq*mq*unlocked,n=L,workers=2)[ns]
        trace=fft.irfft(fft.rfft(cov_sequence,workers=2)*unlocked,n=L,workers=2)[ns]
        expectation=mean_only+trace
        tau=bb*np.array([ratio(n,z)-ratio(n,7) for n in ns])
        return {'z':z,'small':small,'rough':rough,'sv':sv,'blocks':blocks,'need':need,'have':have,'starts':starts,
                'mean':expectation,'trace':trace,'tau':tau,'cov_total_sum':float(cov_sequence.sum())}

    models={z:build(z) for z in (30,100,math.ceil(math.sqrt(M)))}
    # Omega from prime powers, independently of the source SPF program.
    omega=np.zeros(M+1,np.uint8)
    for p in primes:
        power=int(p)
        while power<=M:omega[power::power]+=1;power*=int(p)
    for z,obj in models.items():
        counts=np.bincount(omega[obj['sv']]).tolist()
        out['models'][str(z)]={'n_survivors':len(obj['sv']),'n_selected':int(obj['need'].sum()),
            'Omega_counts_survivors':counts,'mean_sampling_fraction':float(obj['need'].sum()/len(obj['sv'])),
            'expected_V':obj['mean'].tolist(),'sampling_covariance_trace':obj['trace'].tolist(),
            'tau_source_style':obj['tau'].tolist(),
            'T_deterministic_null_mean_minus_tau':stat(obj['mean'],obj['tau']),
            'T_real_common_tau_z':stat(vr,obj['tau']),
            'T_real_exact_conditional_mean':stat(vr,obj['mean']),
            'covariance_sum_check':obj['cov_total_sum']}
    endpoint=models[math.ceil(math.sqrt(M))]
    fixed=np.zeros(M+1,bool);fixed[endpoint['small']]=True;fixed[endpoint['sv']]=True
    out['endpoint']={'z':endpoint['z'],'candidate_labeling_equals_primes':bool(np.array_equal(fixed,s)),
        'all_block_have_equals_need':bool(np.array_equal(endpoint['have'],endpoint['need'])),
        'source_style_stat_ratio_even_with_identical_sequence':stat(vr,endpoint['tau'])/real_old}

    # Cache raw replicate vectors only; recompute all statistics from them.
    # This avoids rerunning the already reproduced primary ensemble when adding
    # the preregistered z=30 sensitivity audit. Removing the output reruns both.
    cache={}
    if OUT.exists():
        prior=json.loads(OUT.read_text(encoding='utf-8'))
        if prior.get('source_commit')==out['source_commit'] and prior.get('Ns')==out['Ns']:
            cache=prior
    def compare(real,ts):
        return {'T_real':float(real),'median':float(np.median(ts)),'ratio':float(real/np.median(ts)),
            'q025':float(np.quantile(ts,.025)),'q975':float(np.quantile(ts,.975)),
            'rank':int(np.sum(ts<real)),'lower_rank_p':float((1+np.sum(ts<real))/200),
            'upper_rank_p':float((1+np.sum(ts>=real))/200)}
    for z,seed in ((100,2026112409),(30,2026112410)):
        obj=models[z];key=f'z{z}_replication'
        cached=cache.get(key,{}).get('samples_V')
        if cached is not None and np.shape(cached)==(199,len(ns)):
            samples=np.array(cached)
            print('reusing raw replicas',z,flush=True)
        else:
            rng=np.random.default_rng(seed);vv=[]
            for j in range(199):
                order=np.argsort(obj['blocks']+rng.random(len(obj['sv'])),kind='stable')
                ranks=np.arange(len(order))-obj['starts'][obj['blocks'][order]]
                chosen=obj['sv'][order[ranks<obj['need'][obj['blocks'][order]]]]
                f=np.zeros(M+1,bool);f[obj['small']]=True;f[chosen]=True
                vv.append(V(f))
                if j%25==0:print('replicas',z,j,flush=True)
            samples=np.array(vv)
        old_stats=np.sum(center((samples-obj['tau'])/ns)**2,axis=1)
        exact_stats=np.sum(center((samples-obj['mean'])/ns)**2,axis=1)
        out[key]={'max_T_difference_from_source':float(np.max(abs(old_stats-np.array(source[f'z{z}']['Ts'])))),
            'original_asymmetric':compare(real_old,old_stats),
            'same_tau_z_both_sides_exploratory':compare(stat(vr,obj['tau']),old_stats),
            'same_exact_mean_both_sides_exploratory':compare(stat(vr,obj['mean']),exact_stats),
            'null_source_T_mean':float(old_stats.mean()),'null_exact_center_T_mean':float(exact_stats.mean()),
            'null_mean_drift_T':stat(obj['mean'],obj['tau']),
            'samples_V':samples.tolist(),'source_Ts':old_stats.tolist(),'exact_center_Ts':exact_stats.tolist()}

    # Phase-null expected quadratic norm on the SAME N set, via its covariance FFT.
    fp=np.zeros(L);fp[:M+1]=s
    Q=fft.rfft(fp,workers=2)-D
    kernel=fft.irfft(abs(Q)**4*unlocked,n=L,workers=2)/L
    def phase_et(nn):
        nn=np.array(nn);cov=kernel[(nn[:,None]-nn[None,:])%L]/np.outer(nn,nn)
        h=np.eye(len(nn))
        for div in (True,False):
            ii=np.flatnonzero((nn%3==0)==div);h[np.ix_(ii,ii)]-=1/len(ii)
        return float(np.trace(h@cov@h))
    out['phase_null']={'ET_same_B3_Ns':phase_et(ns),'ET_B2_Ns':phase_et(b2source['Ns']),
        'ET_B2_source':b2source['main']['ET_pred_eq5']}

    # Parity statistic on the cube-root rough set is exactly a prime-pair rewrite.
    zcube=math.ceil(M**(1/3));cube=build(zcube);a=cube['rough'];b=a.astype(np.int8)*(1-2*(omega%2).astype(np.int8))
    parity=[]
    for n in ns:
        left=slice(2,n-1);right=slice(n-2,1,-1)
        A=int(np.sum(a[left]&a[right]));B=int(np.sum(b[left]*a[right]));C=int(np.sum(b[left]*b[right]))
        large=(s&a);rlarge=int(np.sum(large[left]&large[right]))
        parity.append({'N':int(n),'A':A,'B':B,'C':C,'r_prime_large':rlarge,
             'identity_error':float((A-2*B+C)/4-rlarge),'parity_covariance':float(C/A-(B/A)**2)})
    out['cube_root_parity']={'z':zcube,'max_Omega_on_survivors':int(omega[a].max()),'rows':parity}
    out['limitations']=['Original preregistered verdict retained; shared-baseline tests are new exploratory diagnostics.',
        'No causal percentage of rigidity is identified by comparing unrelated null families.',
        'The parity identity is algebraic and is not an independent estimate or proof of Goldbach.']
    OUT.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'out':str(OUT),'replication':{k:v for k,v in out['z100_replication'].items() if k not in ['samples_V','source_Ts','exact_center_Ts']},
                     'endpoint':out['endpoint'],'phase_null':out['phase_null']},indent=2),flush=True)


if __name__=='__main__':main()
