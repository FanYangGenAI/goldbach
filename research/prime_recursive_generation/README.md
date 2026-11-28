# Recursive prime generation — research charter (2026-11-27)

## Motivation
For even N, if N/2 is prime we have a diagonal Goldbach pair. Otherwise seek distinct primes p < N/2 and q=N-p. The mirror map fixes q given p, but does not imply primality of q. For odd prime p, M=p+1 is even and smaller than N. Under strong induction on Goldbach, M=r+s for primes r,s, so p=r+s-1. Repeating on the smaller prime leads downward; reversing the process motivates generation from small seeds.

## Definition
Let P be the primes, S0={2,3}, and S(k+1)=S(k) union {r+s-1 in P: r,s in S(k)}. Let S be the union over all k. Repetitions r=s are allowed. For a new odd prime, both parents must be odd (2 with an odd prime gives an even composite), and both are strictly smaller than the child. A derivation is a finite binary tree with seed leaves.

**G1:** Is S=P? This is open here, not established by numerical verification. In fact G1 is equivalent to Goldbach restricted to even numbers M=p+1 for odd primes p>=5: forward follows from the first appearance of p; reverse follows by strong induction on p. Strong Goldbach implies G1, but the reverse implication has not been established.

## Proposed Goldbach search
Generate all reachable primes p<N/2; test primality of N-p. Return true on a witness, false only after exhausting candidates. The user's additional stopping band N/4<p<N/2 is a separate stronger existence condition, not a consequence of G1. Handle N/2 prime separately. A failed restricted search is not a counterexample to Goldbach.

## Numerical verification
A fresh local run on 2026-11-27 of the sieve + ascending dynamic closure algorithm for limit 10,000,000 returned:
- reference prime count 664579
- reachable count 664579
- missing []
- maximum smaller parent in first-success ascending search 601
- maximum attempts 110 (including the seed 2)
The sieve independently determines primality; this is a coverage test, not a prime generator without a primality oracle. Ascending dynamic closure is exact for the finite cutoff because parents are strictly smaller than each non-seed child. No asymptotic inference from the maximum parent statistic is justified.

## Next questions
1. Prove G1, or locate a structural obstruction; investigate small-parent sets and residue constraints.
2. Study the proposed middle-band witness condition independently.
3. Separate closure of primes from the actual two-prime mirror correlation required by Goldbach.

This branch is an independent exploratory route; no claim of a proof of Goldbach.
