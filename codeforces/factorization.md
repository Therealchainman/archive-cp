# Factorization 

Factorization of integers in number theory.  Never forget that there are about 7 distinct prime integers at most for any 32 bit integer. and at most 32 prime integers in the prime factorization, if they are just 2^32,  which means there are log(n) prime integers in the prime factorization of an integer.  These facts are very important to remember.

## D. Soldier and Number Game

```py
def sieve(n):
    spf = [i for i in range(n+1)]
    for i in range(2, n + 1):
        if spf[i] != i: continue
        for j in range(i * i, n + 1, i):
            spf[j] = i
    return spf

def factorize(n):
    cnt = 0
    while n > 1:
        cnt += 1
        n //= spf[n]
    return cnt

N = 5_000_000

def main():
    a, b = map(int, input().split())
    res = dp[a] - dp[b]
    print(res)

if __name__ == '__main__':
    T = int(input())
    spf = sieve(N)
    dp = [0] * (N + 1)
    for i in range(2, N + 1):
        dp[i] = dp[i - 1] + factorize(i)
    for _ in range(T):
        main()
```

## C. Reducing Fractions

```py
def sieve(n):
    spf = [i for i in range(n+1)]
    for i in range(2, n + 1):
        if spf[i] != i: continue
        for j in range(i * i, n + 1, i):
            spf[j] = i
    return spf

def factorize(n, freq, i, map_):
    while n > 1:
        freq[spf[n]] += 1
        map_[i][spf[n]] += 1
        n //= spf[n]

def add(freq, res, map_):
    for i, counter in enumerate(map_):
        for k, v in counter.items():
            take = min(freq[k], v)
            freq[k] -= take
            res[i] *= pow(k, take)

N = 10**7

def main():
    n, m = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    afreq, bfreq = Counter(), Counter()
    amap, bmap = [Counter() for _ in range(n)], [Counter() for _ in range(m)]
    for i, a in enumerate(A):
        factorize(a, afreq, i, amap)
    for i, b in enumerate(B):
        factorize(b, bfreq, i, bmap)
    found = 0
    for k in afreq.keys():
        delta = min(afreq[k], bfreq[k])
        found |= delta > 0
        afreq[k] -= delta
        bfreq[k] -= delta
    resa, resb = [1] * n, [1] * m
    add(afreq, resa, amap)
    add(bfreq, resb, bmap)
    print(len(resa), len(resb))
    print(*resa, sep = " ")
    print(*resb, sep = " ")
            

if __name__ == '__main__':
    spf = sieve(N)
    main()
```

## 2015 German Collegiate Programming Contest (GCPC 15) F Divisions

```py
import math

N = 10**6

def factorize(n):
    cnt = cur = 1
    while n % 2 == 0:
        n //= 2
        cur += 1
    cnt *= cur
    for i in range(3, N, 2):
        cur = 1
        while n % i == 0:
            n //= i
            cur += 1
        cnt *= cur
    return cnt

def check_composite(n, a, d, s):
    x = pow(a, d, n)
    if x == 1 or x == n - 1: return False
    for r in range(1, s):
        x = x * x % n
        if x == n - 1: return False
    return True

def miller_rabin(n):
    if n < 4: return n == 2 or n == 3
    r = 0
    d = n - 1
    while d % 2 == 0:
        r += 1
        d >>= 1
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in bases:
        if n == a: return True
        if check_composite(n, a, d, r): return False
    return True

def main():
    n = int(input()) 
    res = factorize(n)
    if n == 1: print(res)
    elif miller_rabin(n): print(2 * res)
    elif math.isqrt(n) * math.isqrt(n) == n: print(3 * res)
    else: print(4 * res)

if __name__ == '__main__':
    main()
```