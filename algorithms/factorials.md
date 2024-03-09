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

## binomial coefficient or combinations

combinations pick r from n elements

factorials are precomputed for calculating combinations frequently

```py
def nCr(n, r):
    return (fact[n] * inv_fact[r] * inv_fact[n - r]) % mod if n >= r else 0
```

## log base 2 factorials

log base 2 factorials can be used to calculate large factorials without overflow.  This keeps them as small decimal values.  

```py
def log_factorials(n):
    log_facts = [0.0] * (n + 1)
    for i in range(2, n + 1):
        log_facts[i] = math.log2(i) + log_facts[i - 1]
    return log_facts

log_facts = log_factorials(n)
def choose(n, k):
    return log_facts[n] - log_facts[k] - log_facts[n - k]
```

Note you only need to call log_factorial once, don't call each time if it is ran against many tests all at once.

```cpp
void log_factorial() {
    log_fac[0] = 0.0;
    for (int i = 1; i < MAXN; i++) {
        log_fac[i] = log_fac[i - 1] + log2(i);
    }
}

long double choose(int n, int k) {
    return log_fac[n] - log_fac[k] - log_fac[n - k];
}
```