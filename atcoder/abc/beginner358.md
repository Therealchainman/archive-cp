# Atcoder Beginner Contest 358

## E - Alphabet Tiles  

### Solution 1:  bounded knapsack, combinatorics, factorials

```py
MOD = 998244353
N = 26

def mod_inverse(x):
    return pow(x, MOD - 2, MOD)

def factorials(n):
    fact, inv_fact = [1] * (n + 1), [0] * (n + 1)
    for i in range(2, n + 1):
        fact[i] = (fact[i - 1] * i) % MOD
    inv_fact[-1] = mod_inverse(fact[-1])
    for i in reversed(range(n)):
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD
    return fact, inv_fact

def main():
    K = int(input())
    counts = list(map(int, input().split()))
    fact, inv_fact = factorials(K)
    def choose(n, r):
        return (fact[n] * inv_fact[r] * inv_fact[n - r]) % MOD if n >= r else 0
    dp = [0] * (K + 1)
    dp[0] = 1
    for i in range(N):
        ndp = dp[:]
        for j in range(1, counts[i] + 1):
            for cap in range(j, K + 1):
                ndp[cap] = (ndp[cap] + dp[cap - j] * choose(cap, j)) % MOD
        dp = ndp
    ans = sum(dp[1:]) % MOD
    print(ans)

if __name__ == '__main__':
    main()
```

## 

### Solution 1: 

```cpp

```