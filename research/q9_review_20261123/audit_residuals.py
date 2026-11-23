"""Independent exploratory audit of Q6; fixed git inputs, no expansion of N.

Run from repository root. Requires NumPy, SciPy, threadpoolctl.
Computes 64-point analytic backgrounds for the existing 300 samples, then
tests the one-system cross-term predictor on the existing 60 samples near 1e6.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
from scipy import integrate, signal, special
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
REF = "c3d27d933899637866e36416385b5969678d9d8c"
OUT = Path(__file__).with_name("audit_results.json")


def git_json(path):
    raw = subprocess.check_output(["git", "show", f"{REF}:{path}"], cwd=ROOT)
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def primes(n):
    s = np.ones(n + 1, bool)
    s[:2] = False
    for p in range(2, math.isqrt(n) + 1):
        if s[p]:
            s[p*p::p] = False
    return np.flatnonzero(s)


def factors(n):
    f = {}
    p = 2
    while p*p <= n:
        while n % p == 0:
            f[p] = f.get(p, 0) + 1
            n //= p
        p += 1
    if n > 1:
        f[n] = f.get(n, 0) + 1
    return f


class Background:
    """Same declared v2, independently integrated/analytically differentiated.

    The prime logarithm is evaluated with its convergent z-power series;
    the omitted z-dependent tail is below floating precision on |z|<=.5.
    The submitted E1 approximation to the prime tail is retained explicitly.
    """

    def __init__(self, order=400, side_order=200):
        self.pr = primes(1_000_000).astype(float)
        self.odd = self.pr[1:]
        x, w = np.polynomial.legendre.leggauss(order)
        v = (x+1)/2
        self.y = .45*v*v
        self.w = w*.45*v
        sigma = 1-self.y
        ps = self.pr[self.pr <= 200_000][None, :]**(-sigma[:, None])
        self.linear = (np.log1p(-ps)+ps).sum(1)
        power = ps*ps
        self.pk = []
        for k in range(2, 41):
            self.pk.append(power.sum(1)/k)
            power *= ps
        self.pk = np.array(self.pk)
        self.tail = special.exp1((2*sigma-1)*math.log(200_000))
        self.logzeta = np.log(np.abs(special.zeta(sigma)))
        self.ell = self.logzeta+self.linear-self.tail/2
        self.beta = np.exp(special.betaln(1-self.y[:,None],1-self.y[None,:]))
        self.qx,self.qw = np.polynomial.legendre.leggauss(side_order)
        self.cache = {}

    def prepare(self, z):
        z = np.asarray(z, complex)
        key = z.tobytes()
        if key not in self.cache:
            logphi = (z[:,None]*(self.logzeta+self.linear)
                      + z[:,None]**np.arange(2,41)[None,:] @ self.pk
                      + (z*z-z)[:,None]*self.tail/2)
            c = np.sin(np.pi*z[:,None])/np.pi*self.w*np.exp(logphi)
            lr=[]; ts=[]
            for zz in z:
                lr.append(np.log(1-(zz-1)**2/(self.odd-1)**2).sum())
                ts.append(np.sum(np.log(self.odd)*(-2*zz*(zz-1))/((self.odd-1)**2-(zz-1)**2)))
            self.cache[key] = c,np.array(lr),np.array(ts)
        return self.cache[key]

    def rt(self,n,z):
        _,lr0,t0=self.prepare(z)
        lr=lr0.copy(); t=t0.copy()
        for p,e in factors(n).items():
            if 2<p<=1_000_000:
                lr-=np.log(1-(z-1)**2/(p-1)**2)
                t-=math.log(p)*(-2*z*(z-1))/((p-1)**2-(z-1)**2)
            j=(p-2+z)/(p-z)
            dj=2*(p-1)/(p-z)**2
            for _ in range(e):
                dj,j=(2*z*j+z*z*dj)/p,1-1/p+z*z*j/p
            lr+=np.log(j/((p-1)/(p-z))**2)
            t+=math.log(p)*(z*dj/j-2*z/(p-z))
        return np.exp(lr),t

    def evaluate(self,n,z,full_patched_mellin=False):
        z=np.asarray(z,complex)
        c,_,_=self.prepare(z)
        a,b=self.y[:,None],self.y[None,:]
        # Exact truncated beta integral instead of leading endpoint subtraction.
        ib=n**(1-a-b)*self.beta*(1-special.betainc(1-a,1-b,2/n)-special.betainc(1-b,1-a,2/n))
        ci=c@ib
        i0=(ci*c).sum(1)
        i1=(ci*(-self.y*c)).sum(1)
        lo,hi=math.log(2/n),math.log(.5)
        v=lo+(self.qx+1)/2*(hi-lo)
        u=np.exp(v);du=self.qw/2*(hi-lo)*u
        x=n*u; xx=n-x
        px=np.exp(-self.y[:,None]*np.log(x))
        pxx=np.exp(-self.y[:,None]*np.log(xx))
        dx=c@px; dxx=c@pxx
        gx=x**(-.45)/np.log(x);gxx=xx**(-.45)/np.log(xx)
        j1=n*((dx*gxx+dxx*gx)@du)
        j2=2*n*np.sum(du*gx*gxx)
        itilde=i0+2*z*j1+z*z*j2
        if full_patched_mellin:
            ddx=(-self.y*c)@px;ddxx=(-self.y*c)@pxx
            dgx=gx*(-.45-1/np.log(x));dgxx=gxx*(-.45-1/np.log(xx))
            i1=n*(((ddx+z[:,None]*dgx)*(dxx+z[:,None]*gxx)
                   +(ddxx+z[:,None]*dgxx)*(dx+z[:,None]*gx))@du)
        r,t=self.rt(n,z)
        return r*(itilde-t*i1)

    def a2(self,t):
        t=np.asarray(t)
        return np.exp(-np.log(t)[...,None]*self.y)@(self.w*self.ell)

    def coefficients23(self,n):
        # Differentiate at z=0; independent from circle extraction.
        def a1(t): return 1/math.log(t)
        def a1cut(t): return (1-t**(-.45))/math.log(t)
        def da1cut(t): return .45*t**(-.45)/math.log(t)-(1-t**(-.45))/math.log(t)**2
        def integrands(u):
            x=n*u;y=n-x
            return np.array([2*a1(x)*a1(y),
                             a1(x)*float(self.a2(y))+a1(y)*float(self.a2(x)),
                             da1cut(x)*a1cut(y)+da1cut(y)*a1cut(x)])
        points=sorted({2/n,1e-5,.001,.05,.5})
        vals=np.zeros(3)
        for lo,hi in zip(points,points[1:]):
            val,_=integrate.quad_vec(integrands,lo,hi,epsabs=1e-12,epsrel=2e-10)
            vals+=n*val
        r0=self.rt(n,np.array([0j]))[0][0].real
        base=2/(self.odd*(self.odd-2))
        rp=base.sum();tp=np.sum(np.log(self.odd)*base)
        for p,e in factors(n).items():
            if 2<p<=1_000_000:
                rp-=2/(p*(p-2));tp-=math.log(p)*2/(p*(p-2))
            rp-=2/p;tp-=math.log(p)*2/p
        return r0*vals[0],r0*(2*vals[1]+rp*vals[0]-tp*vals[2])


def pca(d):
    dc=d-d.mean(0)
    w,v=np.linalg.eigh(dc.T@dc.conj())
    amp=dc@v[:,-1].conj()
    return float(w[-1]/w.sum()),amp


def complex_corr(a,b):
    a=a-a.mean();b=b-b.mean()
    return float(abs(np.vdot(a,b))/(np.linalg.norm(a)*np.linalg.norm(b)))


def stats(d,rows):
    pc,amp=pca(d)
    ff=np.fft.fft(d,axis=1)/d.shape[1]
    fc=ff-ff.mean(0)
    share=np.sum(abs(fc)**2,axis=0);share/=share.sum()
    within=d.copy()
    strata={}
    for g in (True,False):
        ii=[i for i,r in enumerate(rows) if (r['mod3']==0)==g]
        within[ii]-=d[ii].mean(0)
        if len(ii)>2:
            strata[str(g)]={'n':len(ii),'pc_fraction':pca(d[ii])[0]}
    coef_real=(ff[:,:6]/(.5**np.arange(6))).real
    corr_matrix=np.corrcoef(coef_real,rowvar=False)
    return {'pc_fraction':pc,'sector_variance_shares_m2_to_m9':share[:8].tolist(),
            'corr_pc_with_coeff_m2_to_m7':[complex_corr(amp,ff[:,j]) for j in range(6)],
            'coefficient_correlation_matrix_m2_to_m7':corr_matrix.tolist(),
            'unit_variance_coeff_pc_fraction':float(np.linalg.eigvalsh(corr_matrix)[-1]/6),
            'between_mod3_trace_fraction':float(1-np.linalg.norm(within)**2/np.linalg.norm(d-d.mean(0))**2),
            'within_mod3_pc_fraction':pca(within)[0],'strata':strata}


def one_system_trial(model,rows,coefs23):
    # This predictor uses only prime/semiprime indicators and smooth densities.
    m=max(r['N'] for r in rows)
    om=np.zeros(m+1,np.uint8)
    for p in primes(m):
        power=int(p)
        while power<=m:
            om[power::power]+=1
            power*=int(p)
    f1=(om==1).astype(float);f2=(om==2).astype(float)
    d1=np.zeros(m+1);d1[2:]=1/np.log(np.arange(2,m+1))
    predictions=[]
    for ng in (8193,16385):
        loggrid=np.linspace(math.log(2),math.log(m),ng)
        a2=model.a2(np.exp(loggrid))
        d2=np.zeros(m+1);d2[2:]=np.interp(np.log(np.arange(2,m+1)),loggrid,a2)
        c2=2*(signal.fftconvolve(d1,f1)-signal.fftconvolve(d1,d1))
        c3=2*(signal.fftconvolve(d1,f2)+signal.fftconvolve(d2,f1)-2*signal.fftconvolve(d1,d2))
        predictions.append(np.array([[c2[r['N']]/r['N'],c3[r['N']]/r['N']] for r in rows]))
    pred=predictions[-1]
    direct=[]
    for row in rows[:3]:
        n=row['N']
        left=slice(2,n-1);right=slice(n-2,1,-1)
        c3=2*(np.dot(d1[left],(f2-d2)[right])+np.dot(d2[left],(f1-d1)[right]))
        fftvalue=pred[len(direct),1]*n
        direct.append({'N':n,'h2_recount':int(np.dot(f1[left],f1[right])),
                       'h3_recount':int(2*np.dot(f1[left],f2[right])),
                       'h2_saved':row['h'][2],'h3_saved':row['h'][3],
                       'cross3_direct_minus_fft':float(c3-fftvalue)})
    obs=np.array([[(r['h'][2]-bb[0])/r['N'],(r['h'][3]-bb[1])/r['N']]
                  for r,bb in zip(rows,coefs23)])
    groups={}
    for name,ii in [('all',list(range(len(rows)))),('3div',[i for i,r in enumerate(rows) if r['mod3']==0]),
                    ('3nondiv',[i for i,r in enumerate(rows) if r['mod3']!=0])]:
        gg={}
        for j in (0,1):
            o=obs[ii,j];p=pred[ii,j];oc=o-o.mean();pc=p-p.mean()
            gg[f'm{j+2}']={'correlation':float(np.corrcoef(o,p)[0,1]),
                          'predicted_sd_over_observed_sd':float(np.std(p)/np.std(o)),
                          'centered_no_fit_R2':float(1-np.sum((oc-pc)**2)/np.sum(oc**2)),
                          'observed_mean':float(o.mean()),'prediction_mean':float(p.mean())}
        groups[name]=gg
    return {'max_N':m,'n':len(rows),'groups':groups,'direct_convolution_checks':direct,
            'max_interpolation_refinement_change':float(np.max(abs(predictions[0]-predictions[1]))),
            'per_N':[{'N':r['N'],'obs2_over_N':float(o[0]),'obs3_over_N':float(o[1]),
                      'cross2_over_N':float(p[0]),'cross3_over_N':float(p[1])} for r,o,p in zip(rows,obs,pred)],
            'limitations':'Exploratory low-scale diagnostic, not confirmation or a test at 1e9.'}


def main():
    q6,sha=git_json('research/Q6_residual/out/q6_summary.json')
    q4,_=git_json('research/Q1_fugacity/out/q4b_summary.json')
    q5,_=git_json('research/Q5_background/out/q5d_summary.json')
    samples={}
    for scale,group in q4.items():
        if isinstance(group,dict) and 'rows' in group:
            for r in group['rows']: samples[('Q4b',r['N'])]=r['h']
    for r in q5['rows']:samples[('Q5d',r['N'])]=r['h']
    rows=q6['rows']
    for r in rows:r['h']=samples[r['src'],r['N']]
    model=Background()
    z64=.5*np.exp(2j*np.pi*np.arange(64)/64)
    out={'source_commit':REF,'q6_json_sha256':sha,'n_rows':len(rows),'groups':{},'representative_checks':[]}
    circle=[]
    for i,r in enumerate(rows):
        z=z64
        b=model.evaluate(r['N'],z)
        h=np.polynomial.polynomial.polyval(z,np.array(r['h'],float))
        dd=(h-b)/z**2/r['N']
        circle.append(dd)
        if i%50==0:print('circle',i,flush=True)
    circle=np.array(circle)
    for src,scale in sorted({(r['src'],r['X']) for r in rows}):
        ii=[i for i,r in enumerate(rows) if r['src']==src and r['X']==scale]
        rr=[rows[i] for i in ii];dd=circle[ii]
        old=np.array([[complex(*v) for v in r['D_over_N']] for r in rr])
        fc=np.fft.fft(dd,axis=1)/64
        result={'n':len(ii),'radii':{},'submitted_16point':stats(old,rr),
                'max_recomputed_D_difference_16':float(np.max(abs(dd[:,::4]-old)))}
        for rho in (.1,.25,.35,.5):
            # Downward analytic continuation from resolved 64-point coefficients.
            small=np.fft.ifft(fc*(rho/.5)**np.arange(64),axis=1)*64
            result['radii'][str(rho)]=stats(small,rr)
        eps=np.array([[complex(*v) for v in r['eps']] for r in rr])
        result['imaginary_point']={'mean_eps_im':float(eps[:,4].imag.mean()),
                    'sem_im':float(eps[:,4].imag.std(ddof=1)/math.sqrt(len(rr))),
                    'sem_re':float(eps[:,4].real.std(ddof=1)/math.sqrt(len(rr)))}
        phase=np.angle(1+eps[:,4])
        result['imaginary_point']['mean_actual_phase_radians']=float(phase.mean())
        result['imaginary_point']['sem_actual_phase_radians']=float(phase.std(ddof=1)/math.sqrt(len(rr)))
        oldcf=np.fft.fft(old,axis=1)/16
        result['alias_sensitivity']={'max_abs_m2_over_N_change':float(np.max(abs(oldcf[:,0]-fc[:,0]))),
                    'max_abs_m3_over_N_change':float(np.max(abs((oldcf[:,1]-fc[:,1])/.5)))}
        out['groups'][f'{src} {scale}']=result
    for src,scale in [('Q4b',10**6),('Q5d',10**9)]:
        i=next(i for i,r in enumerate(rows) if r['src']==src and r['X']==scale)
        r=rows[i];b2,b3=model.coefficients23(r['N']);fc=np.fft.fft(circle[i])/64
        alt=model.evaluate(r['N'],z64,True)
        out['representative_checks'].append({'N':r['N'],'b2':b2,'b3':b3,
                 'coefficient2_over_N_circle_minus_exact':float((fc[0]-(r['h'][2]-b2)/r['N']).real),
                 'coefficient3_over_N_circle_minus_exact':float((fc[1]/.5-(r['h'][3]-b3)/r['N']).real),
                 'max_background_change_full_patched_mellin_over_N':float(np.max(abs(alt-model.evaluate(r['N'],z64)))/r['N'])})
    rr=[r for r in rows if r['src']=='Q4b' and r['X']==10**6]
    coefficients=[model.coefficients23(r['N']) for r in rr]
    out['one_system_cross_trial']=one_system_trial(model,rr,coefficients)
    out['notes']=['All conclusions exploratory; no source output is modified.',
                  '64-point resolution and prime-tail sensitivity are numerical checks, not interval bounds.',
                  'Smaller radii use downward continuation of resolved coefficients, not a new fitted background.']
    OUT.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'out':str(OUT),'one_system_trial':out['one_system_cross_trial']['groups']},indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=2):
        main()
