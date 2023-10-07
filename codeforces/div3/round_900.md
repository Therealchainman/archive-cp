# Codeforces Round 900 Div 3

## A. How Much Does Daytona Cost?

### Solution 1: 

```py
def main():
    n, k = map(int, input().split())
    arr = list(map(int, input().split()))
    res = "YES" if k in arr else "NO"
    print(res)
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## B. Aleksa and Stack

### Solution 1: 

```py
def main():
    n = int(input())
    res = list(range(1, n + 1))
    for i in range(1, n):
        res[i] = res[i - 1] + 1
        while (res[i - 1] + res[i]) % 3 == 0: res[i] += 1
        if i > 1 and (3 * res[i]) % (res[i - 2] + res[i - 1]) == 0: res[i] += 1
    print(*res)
 
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C. Vasilije in Cacak

### Solution 1: 

```py
def main():
    n, k, x = map(int, input().split())
    minv = k * (k + 1) // 2
    maxv = n * (n + 1) // 2 - (n - k) * (n - k + 1) // 2
    res = "YES" if minv <= x <= maxv else "NO"
    print(res)
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## D. Reverse Madness

### Solution 1:  two pointers + trick

```py
def main():
    n, k = map(int, input().split())
    s = list(input())
    l = list(map(int, input().split()))
    r = list(map(int, input().split()))
    q = int(input())
    queries = sorted(map(int, input().split()))
    segments = {}
    for i in range(k):
        left, right = l[i], r[i]
        while left < right:
            segments[(left, right)] = 0
            left += 1
            right -= 1
    i = 0
    for x in queries:
        while x < l[i] or x > r[i]: i += 1
        left, right = min(x, l[i] + r[i] - x), max(x, l[i] + r[i] - x)
        if left == right: continue
        segments[(left, right)] += 1
    for left, right in zip(l, r):
        cnt = 0
        while left < right:
            cnt += segments[(left, right)]
            if cnt & 1: s[left - 1], s[right - 1] = s[right - 1], s[left - 1]
            left += 1
            right -= 1
    print(*s, sep = "")
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## E. Iva & Pav

### Solution 1: binary search + prefix sum + bits

```py
def main():
    n = int(input())
    arr = list(map(int, input().split()))
    psum = [[0] * 31 for _ in range(n + 1)]
    for i in range(n):
        for j in range(31):
            psum[i + 1][j] = psum[i][j] + ((arr[i] >> j) & 1)
    q = int(input())
    def value(left, right):
        res = 0
        for i in range(31):
            cnt = psum[right + 1][i] - psum[left][i]
            if cnt == right - left + 1:
                res |= 1 << i
        return res
    def bsearch(start, target):
        left, right = start, n
        while left < right:
            mid = (left + right) >> 1
            if value(start, mid) < target:
                right = mid
            else:
                left = mid + 1
        return left
    result = [-1] * q
    for i in range(q):
        left, k = map(int, input().split())
        left -= 1
        right = bsearch(left, k)
        if left == right: continue
        result[i] = right
    print(*result)
 
 
if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## F. Vasilije Loves Number Theory

### Solution 1:  sieve of Erastothenes for smallest prime factor + log(x) prime factorizations of queries

Calculate the product of the frequency of the power of the prime factors for each integer, and basically it needs to be divisible by that. But since the number can be really large you need to calculate it modulo the product.  But you can use O(q^2) so you can just store queries since the last reset to n.

```py
import math

def sieve(n):
    spf = [i for i in range(n)]
    for i in range(2, n):
        if spf[i] != i: continue
        for j in range(i * i, n, i):
            spf[j] = i
    return spf

LIM = 10**6

def main():
    n, q = map(int, input().split())
    pfreq = Counter()
    def factorize(x):
        while x > 1:
            pfreq[spf[x]] += 1
            x //= spf[x]
    factorize(n)
    original_pfreq = pfreq.copy()
    integers = []
    for _ in range(q):
        query = list(map(int, input().split()))
        if query[0] == 1:
            x = query[1]
            integers.append(x)
            factorize(x)
            p = math.prod(v + 1 for v in pfreq.values())
            d = n
            for v in integers:
                d = (d * v) % p
            res = "YES" if d == 0 else "NO"
            print(res)
        else: # reset
            integers.clear()
            pfreq = original_pfreq.copy()
    print()

if __name__ == '__main__':
    T = int(input())
    spf = sieve(LIM + 1)
    for _ in range(T):
        main()
```

## G. wxhtzdy ORO Tree

### Solution 1: 

```py

```

