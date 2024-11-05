# FACTORIALS

computing factorials and their inverses using memory to reduce time. 

Store all factorials up to some n, use mod_inverse to compute the inverse factorials

Can be used to solve multinomial coefficients, combinations, permutations and so on. 

multinomial coefficients is you take fact[n], and multiply by all inv_fact[b1] * inv_fact[b2] ...

## precompute factorial and inverse factorials

```py
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
```

```cpp
int64 inv(int i, int64 m) {
  return i <= 1 ? i : m - (int64)(m/i) * inv(m % i, m) % m;
}

vector<int64> fact, inv_fact;

void factorials(int n, int m) {
    fact.assign(n + 1, 1);
    inv_fact.assign(n + 1, 0);
    for (int i = 2; i <= n; i++) {
        fact[i] = (fact[i - 1] * i) % m;
    }
    inv_fact.end()[-1] = inv(fact.end()[-1]);
    for (int i = n - 1; i >= 0; i--) {
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % m;
    }
}
```

## binomial coefficient or combinations

combinations pick r from n elements

factorials are precomputed for calculating combinations frequently

```py
def choose(n, r):
    return (fact[n] * inv_fact[r] * inv_fact[n - r]) % mod if n >= r else 0
```

```cpp
int64 choose(int n, int r, int m) {
    if (n < r) return 0;
    return (fact[n] * inv_fact[r] % m) * inv_fact[n - r] % m;
}
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

## Factorial trick with division by an integer

If you have N! / i, there is a way compute these with using suffix and prefix product precomputation, and to compute these on the fly

```cpp
for (int i = 1; i <= N; i++) {
    pprod[i] = (pprod[i - 1] * i) % M;
}
for (int i = N; i > 0; i--) {
    sprod[i] = (sprod[i + 1] * i) % M;
}

```

Then to compute N! / i => pprod[i - 1] * sprod[i + 1]