"""Exploratory: compare P_N and S2_N with the linear-sieve normalisation (same V_1 for both)."""
import numpy as np, numba as nb, math, json, sys
@nb.njit
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
def counts(Ns,lpf):
    out=np.zeros((len(Ns),4))
    for k in nb.prange(len(Ns)):
        N=Ns[k]; z=int(round(N**(1/3)))
        while z*z*z>N: z-=1
        while (z+1)**3<=N: z+=1
        P=0; S2=0; Pmass=0; S2mass=0
        for n in range(z+1,N-1):
            a=lpf[n]
            if a<=z: continue
            isP = a==n
            if isP: Pmass+=1
            else: S2mass+=1
            if lpf[N-n]>z:
                if isP: P+=1
                else: S2+=1
        out[k,0]=P; out[k,1]=S2; out[k,2]=Pmass; out[k,3]=S2mass
    return out
NMAX=int(sys.argv[1]) if len(sys.argv)>1 else 10**7
lpf=lpf_sieve(int(1.1*NMAX)+10)
primes=np.nonzero(lpf==np.arange(len(lpf),dtype=np.int32))[0]
rng=np.random.default_rng(3); res={}
eg=math.exp(0.5772156649015329)
for N0 in [10**5,10**6,10**7]:
    if N0>NMAX: break
    Ns=(2*rng.integers(N0//2,int(1.1*N0)//2,size=40)).astype(np.int64)
    o=counts(Ns,lpf); rp=[]; rs=[]
    for i,N in enumerate(Ns):
        z=int(round(N**(1/3)))
        while z**3>N: z-=1
        while (z+1)**3<=N: z+=1
        ls=primes[(primes>=3)&(primes<=z)]; ls=ls[N%ls!=0]
        V1=np.prod(1-1/(ls-1.0))
        rp.append(o[i,0]/(V1*o[i,2])); rs.append(o[i,1]/(V1*o[i,3]))
    res[N0]=dict(P_over_M=float(np.mean(rp)),S2_over_Mprime=float(np.mean(rs)),mass_ratio=float(np.mean(o[:,3]/o[:,2])),rho=float(np.mean(o[:,0]/(o[:,0]+o[:,1]))))
    print(N0,res[N0])
print("sieve constants u=3: f(3)=%.4f F(3)=%.4f; heuristic true e^g*omega(3)=%.4f"%(2*eg*math.log(2)/3,2*eg/3,eg*(1+math.log(2))/3))
