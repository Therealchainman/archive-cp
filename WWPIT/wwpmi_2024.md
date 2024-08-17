# WWPMI 2024

## 

### Solution 1: 

```py

```

## 

### Solution 1:  arithmetic progression, modular arithmetic, fast exponentiation

```py
def summation(n):
    return n * (n + 1) // 2
N = 10 ** 16
M = int(1e9) + 7
ans = summation(N) * pow(2, N - 1, M) % M
print(ans)
```

## 

### Solution 1:  Find all divisors and use dynamic programming, move each point to earliest position

```py
def divisors(n):
    factors = [n]
    i = 2
    while i * i <= n:
        if n % i == 0:
            factors.append(i)
            if i != n // i: factors.append(n // i)
        i += 1
    return factors
# N = 13
N = 10 ** 7
dp = [N] * (N + 1)
ans = 0
for i in reversed(range(2, N + 1)):
    for d in divisors(i):
        npos = i // d
        if i <= dp[d]:
            ans += dp[d] - i + 1
            dp[d] = npos
ans += sum(dp[1:])

```

## 

### Solution 1:  minimum spanning tree, greatest common divisor, count prime factors with using prime sieve to find all primes

```cpp
const int MAXN = 1e8 + 1;
int primes[MAXN], N, ans;

void sieve() {
    fill(primes, primes + MAXN, 1);
    primes[0] = primes[1] = 0;
    int p = 2;
    for (int p = 2; p * p <= N; p++) {
        if (primes[p]) {
            for (int i = p * p; i < N; i += p) {
                primes[i] = 0;
            }
        }
    }
}

void solve() {
    cin >> N;
    sieve();
    int cnt = 0;
    int start = N / 2;
    int num_nodes = N - 1;
    int num_edges = num_nodes - 1;
    if (start % 2 == 0) start++;
    for (int i = start; i <= N; i += 2) {
        if (primes[i]) {
            cnt++;
        }
    }
    int ans = 2 * (num_edges - cnt) + 3 * cnt;
    cout << ans << endl;
}

signed main() {
    freopen("input.txt", "r", stdin);
    freopen("output.txt", "w", stdout);
    solve();
    return 0;
}
```

## P7 Not a Topological Sort Problem

### Solution 1:  dynamic programming, cobinations, factorials, modular inverse

```cpp
const int MOD = 1e9 + 7;

int inv(int i) {
  return i <= 1 ? i : MOD - (int)(MOD/i) * inv(MOD % i) % MOD;
}

vector<int> fact, inv_fact;

void factorials(int n) {
    fact.assign(n + 1, 1);
    inv_fact.assign(n + 1, 0);
    for (int i = 2; i <= n; i++) {
        fact[i] = (fact[i - 1] * i) % MOD;
    }
    inv_fact.end()[-1] = inv(fact.end()[-1]);
    for (int i = n - 1; i >= 0; i--) {
        inv_fact[i] = (inv_fact[i + 1] * (i + 1)) % MOD;
    }
}

int choose(int n, int r) {
    if (n < r) return 0;
    return (fact[n] * inv_fact[r] % MOD) * inv_fact[n - r] % MOD;
}

int N, X;
vector<int> dp, ndp;

int exponentiation(int b, int p) {
    int res = 1;
    while (p > 0) {
        if (p & 1) res = (res * b) % MOD;
        b = (b * b) % MOD;
        p >>= 1;
    }
    return res;
}

void solve() {
    cin >> N >> X;
    factorials(2 * N);
    dp.assign(X + 1, 0);
    dp[0] = 1;
    vector<int> inv_choose(N + 1);
    
    for (int i = 1; i <= N; i++) {
        inv_choose[i] = inv(choose(X, i));
    }
    for (int i = 1; i <= N; i++) {
        ndp.assign(X + 1, 0);
        for (int j = 0; j <= X; j++) {
            int rem = X - (j - i);
            ndp[j] = dp[j];
            if (i > j) continue;
            ndp[j] = (ndp[j] + dp[j - i] * choose(rem, i) % MOD * inv_choose[i] % MOD) % MOD;
        }
        swap(dp, ndp);
    }
    int ans = 0;
    for (int x : dp) {
        ans = (ans + x) % MOD;
    }
    ans = (ans * inv(exponentiation(2, N))) % MOD;
    cout << ans << endl;
}
```

## P6 Plex Googol Plex

### Solution 1:  Chinese Remainder Theorem, Euler Totient Theorem, recursion, infinite power tower

```cpp
vector<int> phi;

// perform the Extended Euclidean Algorithm
// It returns gcd(a, b) and also updates x and y to satisfy ax + by = gcd(a, b)
int extended_gcd(int a, int b, int &x, int &y) {
    if (a == 0) {
        x = 0;
        y = 1;
        return b;
    }
    int x1, y1;
    int gcd = extended_gcd(b % a, a, x1, y1);
    x = y1 - (b / a) * x1;
    y = x1;
    return gcd;
}

// Function to find the modular inverse of a under modulo m using the extended Euclidean algorithm
int mod_inverse(int a, int m) {
    int x, y;
    int gcd = extended_gcd(a, m, x, y);
    if (gcd != 1) {
        throw invalid_argument("Modular inverse does not exist");
    }
    return (x % m + m) % m;
}

// Function to solve the system of congruences using the Chinese Remainder Theorem
// vector n is moduli, and vector a is remainders
int chinese_remainder_theorem(const vector<int>& n, const vector<int>& a) {
    int prod = 1;
    for (int ni : n) {
        prod *= ni;
    }

    int result = 0;
    for (size_t i = 0; i < n.size(); ++i) {
        int ni = n[i];
        int ai = a[i];
        int p = prod / ni;
        int inv = mod_inverse(p, ni);
        result += ai * inv * p;
    }

    return result % prod;
}

void phi_1_to_n(int n) {
    phi.resize(n + 1);
    for (int i = 0; i <= n; i++)
        phi[i] = i;

    for (int i = 2; i <= n; i++) {
        if (phi[i] == i) {
            for (int j = i; j <= n; j += i)
                phi[j] -= phi[j] / i;
        }
    }
}

int exponentiation(int b, int p, int m) {
    int res = 1;
    while (p > 0) {
        if (p & 1) res = (res * b) % m;
        b = (b * b) % m;
        p >>= 1;
    }
    return res;
}

int calc(int n) {
    if (n == 1) return 0;
    if (n % 5 == 0 || n % 2 == 0) { // split up and apply CRT
        int m = 1;
        while (n % 5 == 0) {
            m *= 5;
            n /= 5;
        }
        while (n % 2 == 0) {
            m *= 2;
            n /= 2;
        }
        int s1 = 0, s2 = exponentiation(10, calc(phi[n]), n);
        int crt = chinese_remainder_theorem({m, n}, {s1, s2});
        return crt;
    }
    // apply Euler Totient theorem
    int res = exponentiation(10, calc(phi[n]), n);
    return res;
}

void solve() {
    int n;
    cin >> n;
    phi_1_to_n(n);
    int ans = 0;
    for (int i = 1; i <= n; i++) {
        ans += calc(i);
    }
    cout << ans << endl;
}
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