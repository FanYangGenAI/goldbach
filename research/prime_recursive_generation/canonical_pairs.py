from math import isqrt
import argparse
import json


def canonical_pairs(start, stop, *, allow_equal=True):
    """Return one closest prime pair per even n in the inclusive interval.

    None denotes an even n with no pair under the chosen constraint.
    """
    if start < 4 or stop < start:
        raise ValueError('require 4 <= start <= stop')
    prime = bytearray(b'\x01') * (stop + 1)
    prime[:2] = b'\x00\x00'
    for d in range(2, isqrt(stop) + 1):
        if prime[d]:
            prime[d*d::d] = b'\x00' * ((stop-d*d)//d+1)
    pairs = {}
    for n in range(start + start % 2, stop + 1, 2):
        upper = n // 2 - (not allow_equal)
        pairs[n] = next(((p, n-p) for p in range(upper, 1, -1)
                         if prime[p] and prime[n-p]), None)
    return pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', type=int, default=4)
    parser.add_argument('--stop', type=int, default=10_000)
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()
    pairs = canonical_pairs(args.start, args.stop, allow_equal=not args.strict)
    print(json.dumps(pairs, indent=2))
