# 📘 Binomial Theorem

The **Binomial Theorem** gives a formula to expand expressions of the form:

$$(a + b)^n$$

where \( n \) is a non-negative integer.

### 🔹 General Formula:

$$(a + b)^n = \sum_{k=0}^{n} \binom{n}{k} a^{n-k} b^k$$

- \( \binom{n}{k} \) is a **binomial coefficient**, also written as "n choose k":
  $$
  \binom{n}{k} = \frac{n!}{k!(n-k)!}
  $$
- Each term in the expansion is of the form:
  $$
  \text{coefficient} \cdot a^{n-k} \cdot b^k
  $$

### 🔹 Example:

$$(x + y)^3 = \binom{3}{0}x^3y^0 + \binom{3}{1}x^2y^1 + \binom{3}{2}x^1y^2 + \binom{3}{3}x^0y^3$$

$$= 1x^3 + 3x^2y + 3xy^2 + 1y^3$$

### 🔹 What About \( (a - b)^n \)?

To expand a binomial with a minus sign, like \( (a - b)^n \), use the same formula, but alternate the signs:

$$(a - b)^n = \sum_{k=0}^{n} \binom{n}{k} a^{n-k} (-b)^k$$

Because of \( (-b)^k \), the signs alternate:
- Terms where \( k \) is even are positive.
- Terms where \( k \) is odd are negative.

#### Example:

$$(x - y)^3 = x^3 - 3x^2y + 3xy^2 - y^3$$

### 🔹 Properties of Binomial Coefficients:

- Symmetry: \( \binom{n}{k} = \binom{n}{n-k} \)
- Pascal’s Rule: 
  $$
  \binom{n}{k} = \binom{n-1}{k} + \binom{n-1}{k-1}
  $$

### 🔹 Applications:

- Algebraic expansions  
- Probability theory (binomial distributions)  
- Combinatorics (counting subsets)  
- Calculus (Taylor expansions)  



### 🔹 Implementation:

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
  
int64 binomialCoefficient(int n, int r, int64 m) {
    if (n < r) return 0;
    return (fact[n] * inv_fact[r] % m) * inv_fact[n - r] % m;
}

int64 exponentiation(int64 b, int64 p, int64 m) {
    int64 res = 1;
    while (p > 0) {
        if (p & 1) res = (res * b) % m;
        b = (b * b) % m;
        p >>= 1;
    }
    return res;
}
```

