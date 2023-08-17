# FACTORIALS

computing factorials and their inverses using memory to reduce time. 

Store all factorials up to some n, use mod_inverse to compute the inverse factorials

Can be used to solve multinomial coefficients, combinations, permutations and so on. 

## precompute factorial and inverse factorials

```py
mod = int(1e9) + 7

def mod_inverse(v):
    return pow(v, mod - 2, mod)

def factorials(n):
    fact, inv_fact = [1] * (n + 1), [0] * (n + 1)
    for i in range(2, n + 1):
        fact[i] = (fact[i - 1] * i) % mod
    inv_fact[-1] = mod_inverse(fact[-1])
    for i in reversed(range(n)):
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % mod
    return fact, inv_fact
```