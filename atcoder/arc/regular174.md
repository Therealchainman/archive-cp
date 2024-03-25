# Atcoder Regular Contest 174

## A - A Multiply 

### Solution 1:  kadane, maximum subarray sum, dp

```py
def kadane(arr):
    n = len(arr)
    l = r = s = best = psum = 0
    for i in range(n):
        psum += arr[i]
        if psum < 0:
            psum = 0
            s = i + 1
        elif psum > best:
            best = psum
            r = i + 1
            l = s
    return l, r

def main():
    N, C = map(int, input().split())
    arr = list(map(int, input().split()))
    ans = sum(arr)
    l, r = kadane(arr) if C > 0 else kadane([-x for x in arr])
    lsum = sum(arr[:l])
    rsum = sum(arr[r:])
    msum = sum(arr[l : r])
    ans = max(ans, lsum + rsum + C * msum)
    print(ans)

if __name__ == '__main__':
    main()
```

## B - Bought Review

### Solution 1: math, algebra, average

```py
def main():
    A = list(map(int, input().split()))
    P = list(map(int, input().split()))
    target = 3
    lhs = target * sum(A)
    rhs = sum(i * a for i, a in enumerate(A, 1))
    delta = max(0, lhs - rhs)
    # bribe as many 5 star reviews if it is better
    # since 2 4 star reviews is equivalent to 1 5 star review
    if P[-1] < 2 * P[-2]:
        cost = delta // 2 * P[-1] + (min(P[-1], P[-2]) if delta & 1 else 0)
    else:
        cost = delta * P[-2]
    print(cost)

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## C - Catastrophic Roulette 

### Solution 1:  dp, geometric distribution, probability, infinite geometric series, binary tree

```py
from itertools import product
MOD = 998244353

def mod_inverse(v):
    return pow(v, MOD - 2, MOD)

def infinite_geometric_series(a, r):
    return (a * mod_inverse(1 - r)) % MOD

def main():
    N = int(input())
    ans = [0] * 2
    prob = [1, 0]
    for i in range(1, N):
        p = (i * mod_inverse(N)) % MOD
        geo = infinite_geometric_series(1, p ** 2) # 1 + p^2 + p^4, a + ar + ar^2 + ...
        prob_trans = [
            ((1 - p) * geo) % MOD,
            ((1 - p) * p * geo) % MOD
        ]
        nprob = [
            (prob[0] * prob_trans[1] + prob[1] * prob_trans[0]) % MOD,
            (prob[0] * prob_trans[0] + prob[1] * prob_trans[1]) % MOD
        ]
        fee_trans = [
            (p * geo) % MOD,
            (p ** 2 * geo) % MOD
        ]
        ans[0] = (ans[0] + prob[0] * fee_trans[1] + prob[1] * fee_trans[0]) % MOD
        ans[1] = (ans[1] + prob[0] * fee_trans[0] + prob[1] * fee_trans[1]) % MOD
        prob = nprob
    print(*ans)

if __name__ == '__main__':
    main()
```

## D - Digit vs Square Root 

### Solution 1: 

```py

```

## E - Existence Counting 

### Solution 1: 

```py

```

## 

### Solution 1: 

```py

```

## 

### Solution 1: 

```py

```

