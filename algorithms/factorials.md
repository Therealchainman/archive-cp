# FACTORIALS

computing factorials and their inverses using memory to reduce time. 

Store all factorials up to some n, use mod_inverse to compute the inverse factorials

```py
n = 10**6
mod = 998244353
fact = [1]*(n + 1)
for i in range(1, n + 1):
    fact[i] = (fact[i - 1] * i) % mod
inv_fact = [1]*(n + 1)
inv_fact[-1] = mod_inverse(fact[-1], mod)
for i in range(n - 1, -1, -1):
    inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % mod
```