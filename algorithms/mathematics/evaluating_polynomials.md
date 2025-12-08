# Evaluating Polynomials

## Paterson-Stockmeyer Algorithm

The Paterson-Stockmeyer algorithm is a method for efficiently evaluating polynomials at a given point. It reduces the number of multiplications required by breaking down the polynomial into smaller parts. The algorithm works by expressing the polynomial in terms of powers of a base, allowing for the reuse of previously computed powers.

User this pattern when all of these are true:

1. You have a polynomial / power series $H(z) = \sum h_k z^k$ of moderately large degree D (like thousands).
1. You want to evaluate $H$ at some object $F$ (matrix, series, operator).
1. Multiplying F by F (or by another F-derived series) is expensive (e.g., NTT convolution, matrix multiply).
1. You need many powers of F: all $F^k$ with nontrivial coefficients $h_k$.

O(√D) multiplications by reusing small/big powers.

```cpp
poly A = F;
for (int k = 2; k <= D; ++k) {
    A = A * F; // F^k
    // using A in a linear sum with some coefficients h[k]
}>)
```

- And you’re working in a ring where * is heavy (matrices, NTT polys),
- and you need many exponents k,

1. Define this exponential generating function (EGF): $E(x) = \sum_{n=0}^N f_n \frac{x^n}{n!}$. The coefficient $\frac{x^n}{n!}$ gives the number of structures on a labeled set of size n.
1. So I think you want to calculate the following $$N \cdot (N - 1)! \cdot \sum_{k \geq K}^{N - 1} \cdot [x^{N-1}] \cdot \frac{E(x)^k}{k!}$$
1. I can calculate this fast by using fast convolution algorithms to compute the powers of E(x) and summing the (N - 1)th coefficient.

1. remember how important big[0][0] = 1, which represents the identity sequence for convolution so the first sequence multiplied by it is equal to itself. 
1. Paterson-Stockeymer doesn't change how a single multiplication works; it's about how many times you need to multiply when evaluating a whole polynomial H(F).
1. $$S(x) = \sum_{k=K}^{N-1} \frac{E(x)^k}{k!}$$
1. $H(z) = \sum_{k=0}^{N - 1} \frac{z^k}{k!} = \sum_{k=0}^{N-1} h_k z^k$ and say S(x) = H(F(x))
1. $$H(z) = \sum_{q = 0}^{Q - 1} \sum_{r = 0}^{B - 1} h_{qB + r} z^{qB + r} = \sum_{q = 0}^{Q - 1} z^r \left( \sum_{r = 0}^{B - 1} h_{qB + r} z^{qB} \right)$$
1. $$H(z) = \sum_{q = 0}^{Q - 1} z^r Q_q(z)$$
1. $$S(x) = \sum_{q = 0}^{Q - 1} E(x)^r Q_q(E(x))$$
1. small and big, small powers $E^r$ encode all $0 \leq r \le B$ for inner polynomials $Q_q(E)$
1. And the exponent is $k = qB + r$ can be represented as small[r] * big[q] and $E^k = E^{qB + r} = E^r * E^{qB}$

```cpp
vector<int64> h(M + 1, 0);
for (int i = K; i <= M; ++i) {
    h[i] = inv_fact[i];
}

// Polynomial composition S(x) = H(F(x)) via Paterson–Stockmeyer.
// degree is M
int B = sqrt(M + 1) + 1; // block size
int Q = ceil(M + 1, B); // number of blocks

// small powers: F^0, F^1, ..., F^{B-1}
vector<vector<int64>> small(B, vector<int64>(M + 1, 0));
small[0][0] = 1; // F^0 = 1
if (B > 1) small[1] = f;
for (int i = 2; i < B; ++i) {
    small[i] = polynomialMultiplication(small[i - 1], f);
}

// big powers: G^q = (F^B)^q = F^{qB}
vector<vector<int64>> big(Q, vector<int64>(M + 1, 0));
big[0][0] = 1; // F^0
// compute F^B
vector<int64> FB = small[B - 1];
FB = polynomialMultiplication(FB, f);
if (Q > 1) big[1] = FB;
for (int q = 2; q < Q; ++q) {
    big[q] = polynomialMultiplication(big[q - 1], FB);
}

// result S(x)
vector<int64> S(M + 1, 0);
for (int q = 0; q < Q; ++q) {
    // temp(x) = sum_{r=0}^{B-1} h[qB + r] * F(x)^r
    vector<int64> temp(M + 1, 0);
    bool nonzero = false;
    for (int r = 0; r < B; ++r) {
        int idx = q * B + r;
        if (idx > M) break;
        if (idx < K) continue; // h[idx] = 0
        int64 coeff = h[idx];  // 1 / idx!
        if (coeff == 0) continue;
        nonzero = true;
        for (int i = 0; i <= M; ++i) {
            temp[i] = (temp[i] + coeff * small[r][i]) % MOD;
        }
    }
    if (!nonzero) continue;
    // add big[q](x) * temp(x)
    auto prod = polynomialMultiplication(big[q], temp);
    for (int i = 0; i <= M; ++i) {
        S[i] = (S[i] + prod[i]) % MOD;
    }
}
```