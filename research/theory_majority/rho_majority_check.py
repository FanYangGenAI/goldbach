"""Exploratory check (not preregistered): prime share rho_N in the two-sided N^(1/3)-rough set A_N
(ChatGPT theory-reset memo, Target A: rho_N > 1/2 implies r_>(N) > 0)."""
import numpy as np, numba as nb
@nb.njit(cache=False)
def lpf_sieve(n):
    lpf=np.zeros(n+1,dtype=np.int32)
    for i in range(2,n+1):
        if lpf[i]==0:
            lpf[i]=i; j=np.int64(i)*i
            while j<=n:
                if lpf[j]==0: lpf[j]=i
                j+=i
    return lpf
@nb.njit(parallel=True)
def rho(Ns,lpf):
    out=np.zeros((len(Ns),3))
    for k in nb.prange(len(Ns)):
        N=Ns[k]; z=int(round(N**(1/3)))
        while z*z*z>N: z-=1
        while (z+1)**3<=N: z+=1
        T=0; P=0; rr=0
        for n in range(2,N-1):
            a=lpf[n]; b=lpf[N-n]
            if a>z and b>z:
                T+=1
                if a==n:
                    P+=1
                    if b==N-n: rr+=1
        out[k,0]=T; out[k,1]=P; out[k,2]=rr
    return out
lpf=lpf_sieve(110_000_000)
rng=np.random.default_rng(1)
for N0 in [10**3,10**4,10**5,10**6,10**7,10**8]:
    Ns=(2*rng.integers(N0//2,int(1.1*N0)//2,size=200)).astype(np.int64)
    if N0<=10**5: Ns=np.arange(N0,int(1.1*N0),2,dtype=np.int64)
    o=rho(Ns,lpf); r=o[:,1]/o[:,0]
    print(N0, "rho mean %.4f min %.4f max %.4f"%(r.mean(),r.min(),r.max()), "min r_>/T %.4f"%(o[:,2]/o[:,0]).min())
# all even N up to 20000: min rho
Ns=np.arange(6,20001,2,dtype=np.int64); o=rho(Ns,lpf); r=o[:,1]/np.maximum(o[:,0],1)
bad=Ns[r<=0.5]; print("N<=2e4 with rho<=1/2:", len(bad), bad[:20], "max such:", bad.max() if len(bad) else None)
