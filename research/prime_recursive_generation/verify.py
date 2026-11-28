import argparse
import json
from math import isqrt


def verify(limit):
    if limit < 3:
        raise ValueError('limit must be >= 3')
    prime = bytearray(b'\x01') * (limit + 1)
    prime[:2] = b'\x00\x00'
    for d in range(2, isqrt(limit) + 1):
        if prime[d]:
            prime[d*d::d] = b'\x00' * ((limit-d*d)//d+1)
    generated = bytearray(limit + 1)
    generated[2] = generated[3] = 1
    ordered = [2, 3]
    missing = []
    max_r = max_attempts = 0
    for p in range(5, limit + 1, 2):
        if not prime[p]:
            continue
        for attempts, r in enumerate(ordered, 1):
            s = p + 1 - r
            if r > s:
                missing.append(p)
                break
            if generated[s]:
                generated[p] = 1
                ordered.append(p)
                max_r = max(max_r, r)
                max_attempts = max(max_attempts, attempts)
                break
        else:
            missing.append(p)
    return dict(limit=limit, prime_count=sum(prime), generated_count=len(ordered), missing=missing, max_r=max_r, max_attempts=max_attempts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=10_000_000)
    print(json.dumps(verify(parser.parse_args().limit), indent=2))
