# Principle of Inclusion Exclusion

The Principle of Inclusion and Exclusion (PIE) is a fundamental counting tool in combinatorics and probability theory.

|A U B| = |A| + |B| - |A ∩ B|
size of union when size of intersection is known. 

# Generalized inclusion–exclusion for “exactly M”

## Goal
Count objects where **exactly** \(M\) of a given family of properties hold.

## Setup
- Let \(E\) be a finite set of “properties” or “constraints.”
- For each \($F \subseteq E$), let  
  \(g(F)\) be the number of objects that satisfy **all** properties in \(F\)  
  with **no** restriction on the other properties in \($E \setminus F$).
- Let \(h(M)\) be the number of objects that satisfy **exactly** \(M\) properties from \(E\).

## The formula
For any integer ($M \ge 0$),
$$h(M)
= \sum_{F \subseteq E} g(F)\,\binom{|F|}{M}\,(-1)^{\,|F|-M}.$$

This reduces to the usual inclusion–exclusion when (M=0):
$$
h(0) = \sum_{F \subseteq E} g(F)\,(-1)^{|F|}.
$$

## Why it works (proof sketch)
Group subsets by size. Define
$$
B_k = \sum_{|F|=k} g(F).
$$
Each object that satisfies exactly \(i\) properties contributes \($\binom{i}{k}$) to \($B_k$), since it contains exactly \($\binom{i}{k}$) \(k\)-subsets of its satisfied property set. Hence
$$
B_k = \sum_{i \ge k} \binom{i}{k}\,h(i).
$$
This is a standard binomial inversion. Inverting gives
$$
h(M) = \sum_{k \ge M} \binom{k}{M}(-1)^{k-M} B_k
= \sum_{F \subseteq E} g(F)\,\binom{|F|}{M}(-1)^{|F|-M}.
$$

## Worked example
**Problem.** For a positive integer \(Y\) and a set of primes \($E=\{p_1,\dots,p_n\}$), count the integers \($1 \le x \le Y$) that are divisible by **exactly** \(M\) of the integers in \(E\).

**Fit to the template.**
- Properties are “divisible by \(p\)” for \($p \in E$).
- For \($F \subseteq E$), the numbers divisible by all primes in \(F\) are the multiples of \($\mathrm{lcm}(F)$).  
  So
  $$
  g(F) = \left\lfloor \frac{Y}{\mathrm{lcm}(F)} \right\rfloor,
  \qquad g(\varnothing) = Y \text{ by convention, and } \mathrm{lcm}(\varnothing)=1.
  $$
- Plug into the formula to get \(h(M)\).

**Concrete numbers.** Take \(Y=100\) and \(E=\{2,3,5\}\).

\[
\begin{aligned}
g(\varnothing)&=100,\\
g(\{2\})=50,\; g(\{3\})=33,\; g(\{5\})=20,\\
g(\{2,3\})=16,\; g(\{2,5\})=10,\; g(\{3,5\})=6,\\
g(\{2,3,5\})=3.
\end{aligned}
\]

Then
\[
\begin{aligned}
h(1)&=\sum_{F} g(F)\binom{|F|}{1}(-1)^{|F|-1}
= 48,\\
h(2)&=\sum_{F} g(F)\binom{|F|}{2}(-1)^{|F|-2}
= 23,\\
h(3)&=\sum_{F} g(F)\binom{|F|}{3}(-1)^{|F|-3}
= 3.
\end{aligned}
\]

You can also get \(h(0)=26\). The four values sum to \(100\) as a check.

## When to use this
- You can compute \(g(F)\) for every \($F \subseteq E$).
- You want the count split **by exact number** of satisfied properties, not just “at least one.”

## Common pitfalls
- In “divisibility” problems, use \($\left\lfloor Y/\mathrm{lcm}(F)\right\rfloor), not (Y/\mathrm{lcm}(F)$).
- Remember \($g(\varnothing)=$) total number of objects.
- The binomial \($\binom{|F|}{M}$) is zero if \(|F|<M\), so you can sum over all \(F\) without special casing.

## Template to copy
- Define \(E\) and what it means to “satisfy” a property.
- Prove or compute \(g(F)\) for all \($F \subseteq E$).
- Choose the target \(M\).  
- Evaluate
  $$
  h(M)=\sum_{F \subseteq E} g(F)\,\binom{|F|}{M}\,(-1)^{|F|-M}.
  $$

```cpp
int64 N, M, Y;
vector<int64> A;
int64 C[MAXN][MAXN];

void binomial_coefficients() {
    for (int i = 0; i < MAXN; i++) {
        C[i][0] = C[i][i] = 1;
        for (int j = 1; j < i; j++) {
            C[i][j] = C[i - 1][j - 1] + C[i - 1][j];
        }
    }
}

void solve() {
    cin >> N >> M >> Y;
    A.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    binomial_coefficients();
    int64 ans = 0;
    for (int mask = 0; mask < (1 << N); ++mask) {
        int sz = __builtin_popcount(mask);
        int64 combs = C[sz][M];
        // need to calculate the lcm
        __int128 l = 1;
        for (int i = 0; i < N; ++i) {
            if ((mask >> i) & 1) {
                l = lcm(l, A[i]);
                if (l > Y) {
                    l = Y + 1;
                    break;
                }
            }
        }
        if (sz % 2 == M % 2) {
            ans += (Y / l) * combs;
        } else {
            ans -= (Y / l) * combs;
        }
    }
    cout << ans << endl;
}
```

## An example

suppose you want to count the number of people with a cat or dog or fish as pet.  Suppose you have these elements, c, d, f, cd, cf, df, cdf.  Then if you split that into the sets of cat A_cat = {c, cd, cf, cdf}, and sets of dogs A_dog = {d, cd, df, cdf}, and sets containing fish A_fish = {f, cf, df, cdf}.  Then you want to use principle of inclusion and exclusion because the first time you can realize you are going to end up counting each group of elements, including intersections (yeah)
(c, 1), (d, 1), (f, 1), (cd, 2), (cf, 2), (df, 2), (cdf, 3),  so you double counted the intersections of size 2 and triple counted the intersections of size 3.  So you correct for this by subtracting intersections of size 2.  and so on. 

## Counting the number of coprime integers

This is an example problem you can use PIE. 

This counts the number of integers in the range [1, x] that are coprime with m.  Also you can easily calculate the inverse as well by switching the signs and everything.  That is you can calculate the number of integers in range [1, x] that are not coprime with integer m.  or just take x - num_coprimes(x, m) to get that. 

```cpp
// count number of coprimes of x to m
int num_coprimes(int x, int m) {
    int ans = x;
    int endMask = 1 << primes[m].size();
    for (int mask = 1; mask < endMask; mask++) {
        int numBits = 0, val = 1;
        for (int i = 0; i < primes[m].size(); i++) {
            if ((mask >> i) & 1) {
                numBits++;
                val *= primes[m][i];
            }
        }
        if (numBits % 2 == 0) { // even add
            ans += x / val;
        } else {
            ans -= x / val;
        }
    }
    return ans;
}
```

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