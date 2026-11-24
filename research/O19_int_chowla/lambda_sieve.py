"""O19 helper: Liouville lambda(n) for 0 <= n <= TOP as int8 .npy (lambda(0) = 0).  Usage: python lambda_sieve.py TOP OUTFILE"""
import sys
import numba as nb
import numpy as np


@nb.njit(cache=True)
def omega_parity(n):
    spf = np.zeros(n + 1, dtype=np.int32)
    for i in range(2, n + 1):
        if spf[i] == 0:
            spf[i] = i
            j = np.int64(i) * i
            while j <= n:
                if spf[j] == 0:
                    spf[j] = i
                j += i
    lam = np.zeros(n + 1, dtype=np.int8)
    lam[1] = 1
    for m in range(2, n + 1):
        lam[m] = -lam[m // spf[m]]
    return lam


np.save(sys.argv[2], omega_parity(int(sys.argv[1])))
