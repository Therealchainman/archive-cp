# BINOMIAL COEFFICIENTS

$\binom{n}{k}$
n choose k is the number of ways to select k objects from a total of n objects

are the number of ways to select a set of  
$k$  elements from  
$n$  different elements without taking into account the order of arrangement of these elements (i.e., the number of unordered sets)

This formula can be easily deduced from the problem of ordered arrangement (number of ways to select  
$k$  different elements from  
$n$  different elements). First, let's count the number of ordered selections of  
$k$  elements. There are  
$n$  ways to select the first element,  
$n-1$  ways to select the second element,  
$n-2$  ways to select the third element, and so on. As a result, we get the formula of the number of ordered arrangements:  


$n (n-1) (n-2) \cdots (n - k + 1) = \frac {n!} {(n-k)!}$ . We can easily move to unordered arrangements, noting that each unordered arrangement corresponds to exactly  
$k!$  ordered arrangements ( 
$k!$  is the number of possible permutations of  
$k$  elements). We get the final formula by dividing  


$\frac {n!} {(n-k)!}$  by  
$k!$ .

## Pascal's Triangle

recurrence relation

n >= 1 and 0 < k < n

$$\binom{n}{k} = \binom{n - 1}{k - 1} + \binom{n - 1}{k}$$

## faster way to calculate binomial coefficients

Fast way is to precompute the factorials and inverse factorials and use that to solve in O(1) for each time using the formula.

## Calculating binomial coefficients with dynamic programming

$c[i][j] = \binom{i}{j}$

transitions for dynamic programming

c[i+1][j+1] += c[i][j]; // chose bit i to be 1
c[i+1][j]   += c[i][j]; // chose bit i to be 0

time complexity is O(N^2), so it is not feasible for large N.
Use modular arithmetic to reduce the size of the numbers.

One of the benefits of this approach is you can use any modulus, not just prime moduli.

```cpp
const int MOD = 9998244353, MAXN = 61;
int C[MAXN][MAXN];

void binomial_coefficients() {
    for (int i = 0; i < MAXN; i++) {
        C[i][0] = C[i][i] = 1;
        for (int j = 1; j < i; j++) {
            C[i][j] = (C[i - 1][j - 1] + C[i - 1][j]) % MOD;
        }
    }
}
```

### Calculating binomial coefficients when N is large and K is small

because of cancelation you are left with just K terms in numerator and denominator.
$$\binom{N}{K} = \frac{N(N-1)(N-2)\cdots(N-K+1)}{K!} = \prod_{i=1}^{K} \frac{N - i + 1}{i}$$

```cpp
int64 C[MAXN];
void precompute(int n, int64 m) {
    memset(C, 0, sizeof(C));
    C[0] = 1;
    int64 cur = 1;
    for (int i = 1; i < MAXN; ++i) {
        int64 cand = static_cast<int128>(n - i + 1) * inv(i, m) % m;
        cur = cur * cand % m;
        C[i] = cur;
    }
}
```
A special solution for when you have stars and bars method being used.
If you need to calculate $\binom{N + K - 1}{K}$ you can do the following instead: which is linear in K

```cpp
int64 C[MAXN];
void precompute(int64 n, int64 m) {
    memset(C, 0, sizeof(C));
    C[0] = 1;
    int64 cur = 1;
    for (int i = 1; i < MAXN; ++i) {
        int64 cand = static_cast<int128>(n + i - 1) * inv(i, m) % m;
        cur = cur * cand % m;
        C[i] = cur;
    }
}
```
