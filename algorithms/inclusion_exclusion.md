# inclusion exclusion technique

|A U B| = |A| + |B| - |A ∩ B|
size of union when size of intersection is known. 

## dynamic programming implementation of technique

In this case you can use the inclusion and exclusion technique to remove duplicates coming in the form of if a factor is divisible by another factor such as x is divisibly by y, then x will contain duplicates from y.  

```py
factors = []
for i in range(1, n):
    # i is factor if n is divisible by i
    if n % i == 0: factors.append(i)
m = len(factors)
dp = [[0] * m for _ in range(m)]
for i in range(m):
    dp[i][i] = 1 # i is divisible by i
    for j in range(i):
        if factors[i] % factors[j] == 0: dp[i][j] = 1 # factor_i is divisible by factor_j
# count the ways
counts = [0] * m
for i in range(m):
    # finds position that must be fixed, that is takahashi doesn't work that day so that '.'
    fixed = [0] * factors[i]
    for j in range(n):
        if s[j] == '.': fixed[j % factors[i]] = 1
    unset = factors[i] - sum(fixed)
    counts[i] = pow(2, unset, mod)
# dynamic programming to remove the duplicates
for i in range(m):
    for j in range(i):
        if dp[i][j]:
            counts[i] -= counts[j]
```