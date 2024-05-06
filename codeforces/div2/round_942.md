# Codeforces Round 942 Div 2

##

### Solution 1: 

```py

```

## E. Fenwick Tree

### Solution 1:  fenwick tree, binomial coefficients, combinatorics, coefficient matrix, moving upwards in fenwick tree

```py
MOD = 998244353

def mod_inverse(x):
    return pow(x, MOD - 2, MOD)

def lowbit(x):
    return x & -x

def main():
    n, k = map(int, input().split())
    arr = [0] + list(map(int, input().split()))
    inv = [1] * (n + 1)
    for d in range(1, n + 1):
        inv[d] = mod_inverse(d)
    for i in range(1, n + 1):
        u = i + lowbit(i)
        d = c = 1
        while u <= n:
            c = (c * (d + k - 1) * inv[d]) % MOD
            arr[u] = (arr[u] - c * arr[i]) % MOD
            u += lowbit(u)
            d += 1
    print(*arr[1:])

if __name__ == '__main__':
    T = int(input())
    for _ in range(T):
        main()
```

## F. Long Way to be Non-decreasing

### Solution 1: 

```py

```