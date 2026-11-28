from collections import deque
from math import isqrt
import argparse
import json


def is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    return all(n % d for d in range(3, isqrt(n) + 1, 2))


def forward_coverage(limit):
    if limit < 4:
        raise ValueError('limit must be >= 4')
    primes = {2, 3}
    pending = deque([2, 3])
    processed = []
    covered = {}
    while pending:
        r = pending.popleft()
        processed.append(r)
        for s in processed:
            n = r + s
            if n > limit or n % 2 or n in covered:
                continue
            covered[n] = (r, s)
            p = n - 1
            if p not in primes and is_prime(p):
                primes.add(p)
                pending.append(p)
    return primes, covered


def summarize(limit):
    primes, covered = forward_coverage(limit)
    missing = [n for n in range(4, limit + 1, 2) if n not in covered]
    assert all(r in primes and s in primes and r+s == n
               for n, (r, s) in covered.items())
    assert all(is_prime(p) for p in primes)
    return dict(limit=limit, generated_primes=len(primes),
                covered_evens=len(covered), missing_evens=missing,
                witness_10=covered.get(10))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=10_000)
    print(json.dumps(summarize(parser.parse_args().limit), indent=2))
