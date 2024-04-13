# Starters 127

## 

### Solution 1: 

```py

```

## Expected Components

### Solution 1:  combinatorics, tree, connected components, multiplicative modular inverse, 

```py
MOD = int(1e9) + 7
def mod_inverse(x):
    return pow(x, MOD - 2, MOD)
def divide(x, y):
    return (x * y) % MOD
def main():
    N, K = map(int, input().split())
    black = [0] * N
    arr = map(int, input().split())
    for v in arr:
        v -= 1
        black[v] = 1
    cww = cwb = cbb = 0
    for _ in range(N - 1):
        u, v = map(int, input().split())
        u -= 1; v -= 1
        if black[u] and black[v]: cbb += 1
        elif black[u] or black[v]: cwb += 1
        else: cww += 1
    ans = [0] * (K + 1)
    inv_bb = mod_inverse(K * (K - 1))
    inv_wb = mod_inverse(K)
    for i in range(1, K + 1):
        pbb = divide((K - i) * (K - i - 1), inv_bb)
        pwb = divide(K - i, inv_wb)
        expected_rem_edges = (cww + cbb * pbb + cwb * pwb) % MOD
        ans[i] = (N - i - expected_rem_edges) % MOD
    print(*ans[1:])

if __name__ == "__main__":
    T = int(input())
    for _ in range(T):
        main()
```

## 

### Solution 1: 

```py

```