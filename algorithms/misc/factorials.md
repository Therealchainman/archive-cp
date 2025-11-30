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
  return i <= 1 ? i : m - (m / i) * inv(m % i, m) % m;
}

vector<int64> fact, inv_fact;

void factorials(int n, int64 m) {
    fact.assign(n + 1, 1);
    inv_fact.assign(n + 1, 0);
    for (int i = 2; i <= n; i++) {
        fact[i] = (fact[i - 1] * i) % m;
    }
    inv_fact.end()[-1] = inv(fact.end()[-1], m);
    for (int i = n - 1; i >= 0; i--) {
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % m;
    }
}

int64 choose(int n, int r, int64 m) {
    if (n < r) return 0;
    return (fact[n] * inv_fact[r] % m) * inv_fact[n - r] % m;
}
```

## binomial coefficient or combinations

combinations pick r from n elements

factorials are precomputed for calculating combinations frequently

```py
def choose(n, r):
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

## Multinomial theorem 

Count the number of permutations of a multiset.

An implementation I can use for multinomial calculations

Can count situations where you have balls of [2,2,2,3,5], so you have some identical, to find the number of ways you can use the multinomial theorem to place in 10 boxes or something. 

N! / (a! * b! * c! * d! * e!), where a + b + c + d + e = N

You can count how many sequences have these exact counts 
\($c_0, c_1, \dots, c_{n-1}$) by choosing positions for each label:

First choose $c_0$ positions out of the m for label 0:
$$
\binom{m}{c_0}.
$$

Then choose $c_1$ positions out of the remaining $m - c_0$ for label 1:
$$
\binom{m - c_0}{c_1}.
$$

Continue until all labels are placed.

Multiplying gives
$$
\binom{m}{c_0}
\binom{m - c_0}{c_1}
\cdots
\binom{m - c_0 - \cdots - c_{n-2}}{c_{n-1}}
\;=\;
\frac{m!}{c_0!\,c_1!\,\cdots\,c_{n-1}!}.
$$

ignore the division by 2, that must have been specific to problem it was used for.

```cpp
int multinomial(const vector<int> &items) {
    int cnt = accumulate(items.begin(), items.end(), 0LL);
    int ans = fact[cnt];
    for (int x : items) {
        ans = (ans * inv_fact[x / 2]) % MOD;
    }
    return ans;
}
```

Another method when you can't necessarily use modular arithmetic to prevent integer overflow, instead use a threshold to stop early if it gets too large.

```cpp
const int MAXK = 1e6;
//this is same as nCk formula 
// we know nCk is n! / (k! * (n - k)!)
// can be written as (n * (n - 1) * (n - 2) * ....... * (n - k + 1)) / (1 * 2 * 3 * ..... * k) 
int64 binomial(int64 N, int64 K) {
    if (K > N) return 0;
    if (K > N - K) K = N - K;
    int64 ans = 1;
    for (int64 i = 1; i <= K; ++i) {
        ans = ans * (N - i + 1) / i;
        if (ans >= MAXK) return MAXK;
    }
    return ans;
}
int multinomial(const vector<int>& arr) {
    int64 ans = 1, tot = 0;
    for (int cnt : arr) {
        tot += cnt;
        ans *= binomial(tot, cnt);
        if (ans >= MAXK) return MAXK;
    }
    return ans;
}
```