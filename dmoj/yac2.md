# Yet Another Contest 2

## Summary

I did very bad at this contest, I was tired and only had 1 hour and 30 minutes.  But I thought
I found the solution for question 1, but it didn't work.  Question 3 I was stuck on how to improve it. 


# P1: Betting

## Solution: Math

Just need to derive 2 inequality with x and y, then solve for y, and you get an inequality
that if it is true means you satisfy both inequalities.  I solved the inequality correctly in contest
but there was precision error.  Cause I had division.  Here you can remove any division and have it
just in multiplication

```py
T = int(input())
for _ in range(T):
    A, B, C , D = map(int,input().split())
    if A*C < (B-A)*(D-C):
        print('YES')
    else:
        print('NO')
```

# P2: Secret Sequence

## Solution: Graph + Dynamic Programming



```py

```

# P3: Maximum Damage

## Solution 1: Dumb Prime Sieve

This is slow and only passes testcase 1, trying to get it to pass testcase 2

I think the question is how can I use sieve to prime factorization

```py
from collections import Counter, defaultdict
N, K = map(int,input().split())
H = list(map(int,input().split()))
top = max(H)+1
sieve = [Counter() for _ in range(top)]
primes = defaultdict(list)
for integer in range(2,top):
    if not len(sieve[integer]):
        for possibly_divisible_integer in range(integer,top,integer):
            current_integer = possibly_divisible_integer
            while not current_integer%integer:
                sieve[possibly_divisible_integer][integer] += 1
                current_integer //= integer
for h in H:
    for k, v in sieve[h].items():
        primes[k].append(v)
cnt = 0
for p in primes.values():
    sum_ = sum(p)
    cnt += min(sum_//2, sum_ - max(p))
print(cnt)
```