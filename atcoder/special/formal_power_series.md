# Formal Power Series

## Snack

### Solution 1: binomial expansion, extracting coefficients, combinatorics, generating functions

Generating function is built so that every ordered sequence contributes exactly one monomial whose exponent is its total cost and coefficients count how many sequences land on the same total.

1. Encode one day:
Pick on item with cost in S={1,3,4,5}. Write the one day polynomial
A(x) = x+x^3+x^4+x^6
2. Combine D days
D independent days means pick one term from A(x) on each day and multiply them.  So the polynomial for D days is A(x)^D. When you multiply monomials exponents add:
$x^{s_1} * x^{s_2} * \cdots * x^{s_D} = x^{s_1 + s_2 + ... + s_D}$ 
3. Group by total cost
All sequences whose totals equal N produce the same monomial $x^N$. In a polynomial or formal power series, the coefficient of $x^N$ is the sum of the contributions of all such sequences. Since each sequence contributes 1, that coefficient is exaclty the number of sequences with total N.

$$
[X^N](x + x^3 + x^4 + x^6)^D
= [X^N](x(1+x^2+x^3+x^5))^D
= [X^N]x^D(1+x^2+x^3+x^5)^D
$$

$$
= [X^{N-D}](1+x^2+x^3+x^5)^D
= [X^{N-D}]((1+x^2)(1+x^3))^D
$$

```cpp
const int64 MOD = 998244353;
const int MAXN = 1e6 + 5;
int N, D;

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

void solve() {
    cin >> D >> N;
    int M = N - D;
    int64 ans = 0;
    if (M <= 0) {
        cout << ans << endl;
        return;
    }
    for (int i = 0; i < M / 2; ++i) {
        int cand = M - 2 * i;
        if (cand % 3 != 0) continue;
        int j = cand / 3;
        int64 add = choose(D, i, MOD) * choose(D, j, MOD) % MOD;
        ans = (ans + add) % MOD;
    }
    cout << ans << endl;
}
```

## Tuple of Integers

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```

##

### Solution 1: 

```cpp

```