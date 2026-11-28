"""Independent small-range reference and exact validation of cumulative logs."""
import argparse
import decimal
import re
import subprocess
from math import isqrt
from pathlib import Path


def parse_line(line):
    fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
    return {k: int(v) if k not in ('slope', 'elapsed') else v for k,v in fields.items()}


def check_log(path):
    previous = 0
    rows = [parse_line(line) for line in Path(path).read_text().splitlines() if line.startswith('N=')]
    assert rows, 'No result rows'
    for r in rows:
        n = r['N']; m = n//2; L = m-2
        assert n % 2 == 0 and n > previous
        assert r['count'] == m-1 and r['missing_within_cap'] == 0, r
        assert r['num'] == 2*(r['md']-r['dd'])
        assert r['den'] == L*(L+1)*(2*L+1)//6-2*r['md']+r['dd']
        previous = n
    r = rows[-1]
    decimal.getcontext().prec = 40
    print(f"{path}: N={r['N']}, count={r['count']}, slope={1+decimal.Decimal(r['num'])/decimal.Decimal(r['den'])}")
    return r


def reference(n):
    # Independent scalar search, no mod-6 restrictions or bit-parallel logic.
    prime = bytearray(b'\x01')*(n+1)
    prime[:2] = b'\x00\x00'
    for p in range(2,isqrt(n)+1):
        if prime[p]:
            prime[p*p::p] = bytes((n-p*p)//p+1)
    md = dd = maxd = 0
    for m in range(2,n//2+1):
        for d in range(m-1):
            if prime[m-d] and prime[m+d]:
                break
        else:
            raise AssertionError(f'No pair: {2*m}')
        md += (m-2)*d
        dd += d*d
        maxd = max(maxd,d)
    return dict(md=md,dd=dd,maxgap=2*maxd,count=n//2-1,missing_within_cap=0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--exe', help='Compiled binary to cross-check (optional)')
    ap.add_argument('logs', nargs='*')
    args = ap.parse_args()
    for path in args.logs:
        check_log(path)
    if args.exe:
        for n in [4, 10000, 199998, 200000, 200002, 200012, 200128, 1000000]:
            raw = subprocess.check_output([args.exe,str(n)],text=True)
            row = parse_line(raw.splitlines()[-1])
            ref = reference(n)
            assert all(row[k]==v for k,v in ref.items()), (n,row,ref)
            print(f'Independent reference PASS: {n}')
