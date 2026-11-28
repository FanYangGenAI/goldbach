# Phase-blind minor arc bound (2026-11-28)
Read gb/hub current_conclusions.md (blob eccb086cf87fcd2d707e7b4af4a0c67dc2fb21d2) and prior run's Hausdorff–Young calculation. That calculation is correct.
## Exact extremum
On the circle let M be invariant under alpha -> -alpha, and A>=0 be even with A in L2. For any integer N, among measurable S supported on M with |S|=A on M and S(-alpha)=conj(S(alpha)), the minimum of integral S(alpha)^2 exp(-2 pi i N alpha) is -integral_M A^2. Indeed the integral is real by symmetry and bounded below by triangle inequality. On representatives alpha in (0,1/2), choose S=i A exp(pi i N alpha), and extend by conjugation to negative representatives. Then the integrand equals -A^2 almost everywhere. The maximum is attained by replacing i by 1. Values at endpoints are immaterial. Thus even the entire magnitude profile, not just its moments, cannot yield a stronger universal one-sided bound in this relaxed class. This does not establish a barrier for trigonometric polynomials with nonnegative prime-supported coefficients; those constraints are lost. It also does not establish novelty.
## Finite grid experiment
Use odd grid size m, symmetric mask, target N even, S[k]=i sign(k) A[k] exp(pi i N k/m) for signed nonzero k, zero at k=0. Then inverse FFT at N of S^2 is exactly -sum A^2/m. Compare with original prime phases. Reproducible code follows (requires numpy):
```python
import numpy as np
m=8193; X=4096; N=3000
p=np.ones(X+1,dtype=bool); p[:2]=False
for j in range(2,int(X**.5)+1):
 if p[j]: p[j*j::j]=False
w=np.zeros(m); j=np.flatnonzero(p); w[j]=np.log(j)
s=np.fft.fft(w); k=np.arange(m); k=np.where(k>m//2,k-m,k)
mask=(abs(k)>32)&(abs(abs(k)-m/2)>32)
a=abs(s)*mask
v=1j*np.sign(k)*a*np.exp(-1j*np.pi*N*k/m)
print(np.fft.ifft((s*mask)**2)[N].real,np.fft.ifft(v*v)[N].real,-np.sum(a*a)/m)
```
Mask is illustrative, not a full circle-method minor arc definition. FFT sign convention is negative. Code supplied but not executed in this write step. Literature: Pintz arXiv:1804.09084 abstract reports E(X)<X^.72; Pintz–Althöfer arXiv:2610.15559 abstract reports 7/12 under fixed zero-free half-plane input; neither full proof audited. Next: exploit nonnegative coefficient and prime support constraints, not another phase-blind norm bound. No Goldbach proof.