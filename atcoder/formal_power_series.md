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

## Sequence

### Solution 1: 

```cpp

```

## Sequence 2

### Solution 1: generating functions, sequences, infinite geometric series, binomial expansion, combinatorics

1. Find a way to represent a sequence of values summing to a specific integer such as M.
1. These should be infinite geometric series or something, so then rewrite in sum formulas.
1. Then do any factorization you need
1. Then find the binomial expansion to extract the coefficient for x^M
1. I think you need to think how to manipulate this so you can built a good model with generating functions and constraints.
1. construct a sequence which will work out with the math.
1. select a new sequence ($b_0, b_1, b_2, \dots b_N, b_{N+1}$), where you have the constraint that $b_0$=0 and $b_N+1$=M
1. Where you have the following constraints 
1. $b_1-b_0 \geq 0$ and $b_{N+1}-b_N \geq 0$
1. for the rest of i, $b_{i+1}-b_i$ has to be positive odd integer for the parity rule.
1. What is the sum of this sequence, $(b_1-b_0)+(b_2-b_1)+(b_3-b_2)+\cdots + (b_{N+1}-b_N)$ this is a telescoping sum, where you are left with $b_{N+1}-b_0=M$ 
1. So if we have generating functions we will want the coefficient for the $x^M$ term, cause the powers will sum to M and is what you want.
1. [$x^M$]$(\sum_{k=0}^{\infin}x^k)^2(\sum_{k=0}^{\infin}x^{2k+1})^{N-1}$
1. so now you just have to do the math to convert these infinite geometric series into something you can calculate.
1. using the formula you get $\frac{1}{1-x}$ and $\frac{x}{1-x^2}$
1. simplify to get the setup so you can do binomial expansion to find when x is to a specific power and extract the coefficient.
1. we get [$x^{M-N+1}$]=$(1-x)^{-2}(1-x^2)^{-(N-1)}$
1. binomial series for that negative exponent is $$(1-x)^{-n}=\sum_{k=0}^{\infin}\binom{n+k-1}{n-1}x^k$$, where $n \geq 0$
1. Now with ethe binomial series you have the following [$x^{M-N+1}$]$(\sum_{k=0}^{\infin}\binom{k+1}{1}x^k)$ $(\sum_{k=0}^{\infin}\binom{N+k-2}{N-2}x^{2k})$
1. This is what you can use to calculate in an algorithm.
1. $$\sum_{i \geq 0; j \geq 0; i + 2j = M - N + 1}^{\infin} \binom{i+1}{1} \cdot \binom{N+j-2}{N-2}$$
1. hard to remember but at end multiple by N! to account for permutations of the sequence.

```cpp
const int64 MOD = 998244353;
const int MAXN = 2e5 + 5;
int N, M;

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
    cin >> N >> M;
    if (N == 1) {
        cout << M + 1 << endl;
        return;
    }
    int target = M - N + 1;
    int64 ans = 0;
    for (int j = 0; j <= target; ++j) {
        int i = target - 2 * j;
        if (i < 0) break;
        ans = (ans + choose(i + 1, 1, MOD) * choose(N + j - 2, N - 2, MOD) % MOD) % MOD;
    }
    ans = ans * fact[N] % MOD;
    cout << ans << endl;
}
```

## Sequence 3

### Solution 1: exponential generating functions, combinatorics, multinomial coefficients, counting

Given a fixed frequency of all the values we want to count with the multinomial 

Given frequencies ($i_1, \dots, i_M$)
$\frac{N!}{i_1!i_2! \cdots 1_M!}$

So the total answer is 
$$\sum_{i_1, \dots, i_M \geq 0, i_m \leq m, \sum i_m = N} \frac{N!}{i_1! i_2! \cdots i_M!}$$
And you can factor out the factorial of N.

Now encode sum efficiently

$$F_m(x)=\sum_{k=0}^m \frac{x^k}{k!}$$
multiple all of these together
$F_1(x)F_2(x) \cdots F_M(x)$ 

formally you want to find the $x^N$ term, where the sum of frequencies equals N. 

the answer is $N! [x^N](F_1(x)F_2(x) \cdots F_M(x))$

Use exponential generating function because:
- The combinatorial formula involves factors of $\frac{1}{i_m!}$
- Exponential generating functions naturally encode these factorials. 

The reason you have to is because the positions 1, to N are all distinct(labelled)
So for each value m, choosing which positions get value m is like taking a labeled subset of positions. 

The EGF for "take a set of size i of labeled elements" is $\frac{x^i}{i!}$

```cpp
const int MAXN = 500, MOD = 998244353;
int N, M;

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

vector<int64> convolve(const vector<int64>& arr, const vector<int64>& brr) {
    int n = arr.size(), m = brr.size();
    vector<int64> res(N + 1, 0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (i + j > N) break;
            res[i + j] = (res[i + j] + arr[i] * brr[j]) % MOD;
        }
    }
    return res;
}

void solve() {
    cin >> N >> M;
    vector<int64> dp(N + 1, 0), arr(M + 1, 0);
    dp[0] = arr[0] = 1;
    for (int i = 1; i <= M; ++i) {
        arr[i] = inv_fact[i];
        dp = convolve(dp, arr);
    }
    int ans = fact[N] * dp[N] % MOD;
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    factorials(MAXN, MOD);
    solve();
    return 0;
}
```

## Colored Paper

### Solution 1:  exponential generating functions, combinatorics, labelled subsets of items, closed form

1. product simplifies to exponentials no convolution needed. 
1. everything was able to get into closed forms.

For EGF we want a formal power series, where $f_n$ is number of valid colorings for n sheets.

$$F(x)=\sum_{n=0}^{\infin} f_n \frac{x^n}{n!}$$

Red sheets are easy, any number is allowed so you just need EGF because you are taking subset of labeled items. 
$$R(x)=\sum_{n=0}^{\infin} \frac{x^n}{n!} = e^x$$

Blue sheets you want even only, so you want $$B(x)=\sum_{n \geq 0, \text{n is even}} \frac{x^n}{n!}$$

You can get this by doing the following trick to get just even terms for a formal power series. 

Define: $A(x)=\sum_{n=0}^{\infin} a_n x^n$

It turns out that $$\sum_{\text{even n}} a_n x^n = \frac{A(x) + A(-x)}{2}$$

Given that $A(x) = e^x$, you get the following for $B(x) = \frac{e^x + e^{-x}}{2}$

For yellow sheets:
Similarly for odd you have the following
$$\sum_{\text{n odd}} a_n x^n = \frac{A(x) - A(-x)}{2}$$
And this gives you for $Y(x) = \frac{e^x - e^{-x}}{2}$

To combine all of these you want the product of all of these
$\text{answer} = R(x)B(x)Y(x)$
And the answer for the Nth term is going to be $[\frac{x^N}{N!}]R(x)B(x)Y(x)$

But you have these nice closed forms for the EGFs for all terms, so let's try getting some solution from that.  If you do the math you will get $R(x)B(x)Y(x) = \frac{e^{3x} - e^{-x}}{4}$

Now extracting coefficient you can use the following rule. 

$$e^{ax} = \sum_{n=0}^{\infin} a^n \frac{x^n}{n!}$$

That means the Nth coefficient is $a^n$, so it is like $e^{3a} = 3^N$

The final answer is in a closed form: 
$$\frac{3^N - (-1)^N}{4}$$

Another way to view this is you have the following after taking product in the formal power series

$$
G(x)=\sum_{n=0}^{\infin} \frac{3^n - (-1)^n}{4} \frac{x^n}{n!}
$$

And so for the Nth term you want to extract just the coefficient not the N! part.


```cpp
const int MOD = 998244353;
int N;

int64 inv(int i, int64 m) {
  return i <= 1 ? i : m - (m / i) * inv(m % i, m) % m;
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

void solve() {
    cin >> N;
    int64 x = exponentiation(3, N, MOD);
    int64 y = N % 2 == 0 ? 1 : -1;
    int64 num = x - y;
    if (num < 0) num += MOD;
    int64 ans = (num * inv(4, MOD)) % MOD;
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Coin

### Solution 1: 

```cpp

```

## Jump

### Solution 1: 

1. you can only move 0 or 1 in x direction
1. 

```cpp

```

##

### Solution 1: 

```cpp

```