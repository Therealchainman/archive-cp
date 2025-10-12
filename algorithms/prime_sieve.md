# PRIME SIEVE

prime sieve to generate prime numbers

primes holds the prime integers for the lim integer

sieve holds the prime factors for each integer from 0, 1, 2, ..., lim
So it gives you the prime factorization of all numbers in the range. 

the time complexity is O(n log log n) most likely

## prime sieve to precompute all integers that are prime

This prime sieve will just return if prime or not in an arrays.  So an array will represent the status of each integer in a continuous range whether it is prime or not prime (composite)

When primes[i] = 1, then i is prime.  When primes[i] = 0, then i is not prime.

```py
def prime_sieve(lim):
    primes = [1] * lim
    primes[0] = primes[1] = 0
    p = 2
    while p * p <= lim:
        if primes[p]:
            for i in range(p * p, lim, p):
                primes[i] = 0
        p += 1
    return primes
```

```cpp
const int MAXN = 4e5 + 5;
int N;
bool primes[MAXN];

void sieve(int n) {
    fill(primes, primes + n, true);
    primes[0] = primes[1] = false;
    int p = 2;
    for (int p = 2; p * p <= n; p++) {
        if (primes[p]) {
            for (int i = p * p; i < n; i += p) {
                primes[i] = false;;
            }
        }
    }
}
```

## fast prime sieve for prime factorizations of each integer

precomputes the prime factorization for each integer from 0 to upper_bound (inclusive).  This one is a bit worrisome, it may be too slow in actually.  A faster approach is below for how to get the prime factorization of integer queries. 

Note: this does not get the multiplicity of each prime factor, just the prime factors themselves.

```py
def prime_sieve(upper_bound):
    prime_factorizations = [[] for _ in range(upper_bound + 1)]
    for i in range(2, upper_bound + 1):
        if len(prime_factorizations[i]) > 0: continue # not a prime
        for j in range(i, upper_bound + 1, i):
            prime_factorizations[j].append(i)
    return prime_factorizations
```

```cpp
vector<vector<int>> primes;

void sieve(int n) {
    primes.assign(n + 1, vector<int>());
    for (int i = 2; i <= n; i++) {
        if (primes[i].empty()) {
            for (int j = i; j <= n; j += i) {
                primes[j].push_back(i);
            }
        }
    }
}
```

## prime sieve for smallest prime factor and fast prime factorization of integers

If you calculate the smallest prime factor for each integer, you can use that to speed up prime factorization of each integer from O(sqrt(n)) to O(log(n)).  While the prime sieve here is really fast and just nlog(log(n))

Just remember you want to call the sieve at the appropriate location, don't want to recompute it over and over it is a precomputation step that should only be done once. 

```py
def sieve(n):
    spf = [i for i in range(n + 1)]
    for i in range(2, n + 1):
        if spf[i] != i: continue
        for j in range(i * i, n + 1, i):
            if spf[j] != j: continue
            spf[j] = i
    return spf

# log(n) factorize
def factorize(x):
    factors = []
    while x > 1:
        factors.append(spf[x])
        x //= spf[x]
    return factors
```

Full example of smallest prime factor being used to count divisors in C++. 

```cpp
const int MAXN = 1e5 + 5;
int spf[MAXN];

// nloglog(n)
void sieve(int n) {
    for (int i = 0; i < n; i++) {
        spf[i] = i;
    }
    for (int64 i = 2; i < n; i++) {
        if (spf[i] != i) continue;
        for (int64 j = i * i; j < n; j += i) {
            if (spf[j] != j) continue;
            spf[j] = i;
        }
    }
}

vector<int> factorize(int x) {
    vector<int> factors;
    while (x > 1) {
        factors.emplace_back(spf[x]);
        x /= spf[x];
    }
    return factors;
}

// log(x) algorithm with spf
int count_divisors(int x) {
    int res = 1;
    int prev = -1;
    int cnt = 1;
    while (x > 1) {
        if (spf[x] != prev) {
            res *= cnt;
            cnt = 1;
        }
        cnt++;
        prev = spf[x];
        x /= spf[x];
    }
    res *= cnt;
    return res;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int n, x;
    cin >> n;
    sieve(MAXN);
    for (int i = 0; i < n; i++) {
        cin >> x;
        cout << count_divisors(x) << endl;
    }
    return 0;
}
```

You can also count the number of prime integers in the prime factorization of an integer, excluding 1, which is not prime anyways. 


```cpp
// log(x) algorithm to count prime factors, so 2^2, counts as 2 prime factors
int count_primes(int x) {
    int res = 0;
    while (x > 1) {
        x /= spf[x];
        res++;
    }
    return res;
}
```

## prime sieve to compute the sum of multiplicity for each integer

```py
def prime_sieve(lim):
    multi_sum = [0] * lim
    for i in range(2, lim):
        if multi_sum[i] > 0: continue 
        for j in range(i, lim, i):
            num = j
            while num % i == 0:
                multi_sum[j] += 1
                num //= i
    return multi_sum
```

## sieve of Eratosthenes to count the number of distinct primes that make up each integer

```cpp
void sieve(int n) {
    fill(isprime, isprime + n, true);
    isprime[0] = isprime[1] = false;
    for (int64 p = 2; p < n; p++) {
        if (isprime[p]) {
            for (int64 i = p; i < n; i += p) {
                isprime[i] = false;
                primesCount[i]++;
            }
        }
    }
}
```

## Sieve of Eratosthenes to get all divisors

```cpp
vector<vector<int>> fac(n+2);
for(int i = 1; i<=n; i++)
    for(int j = i; j<=n; j+=i)
        fac[j].emplace_back(i);
```

## Segmented Sieve

This segmented sieve is used to find all prime numbers in a range [l, r] where l and r can be very large, up to 10^12 or more.  the main requirement is that the range [l, r] should not be too large, ideally less than 10^7.  

time complexity is $\mathcal{O}\left((R - L + 1) \log\log R + \sqrt{R} \log\log \sqrt{R} \right)$


```cpp
int64 L, R;

int64 ceil(int64 x, int64 y) {
    return (x + y - 1) / y;
}

vector<bool> segmentedSieve(int64 l, int64 r) {
    int64 lim = sqrt(r);
    vector<bool> marked(lim + 1, false);
    vector<int64> primes;
    for (int64 i = 2; i <= lim; ++i) {
        if (marked[i]) continue;
        primes.emplace_back(i);
        for (int64 j = i * i; j <= lim; j += i) {
            marked[j] = true;
        }
    }
    vector<bool> isPrime(r - l + 1, true);
    for (int64 p : primes) {
        for (int64 i = max(p * p, ceil(L, i) * i); i <= r; i += p) {
            isPrime[l - i] = false;
        }
    }
    if (l == 1) isPrime[0] = false;
    return isPrime;
}
```

## Sieve over squares to compute square free part of every number

This code is a sieve over squares that computes the square-free part of every number.
dp[n] = n / g, where g is the largest perfect square dividing n.
This value is called the square-free part, square-free kernel or core of n.

Examples:
n = 72 = 2^3 * 3^2 → largest square divisor is 36 → dp[72] = 72 / 36 = 2
n = 12 = 2^2 * 3 → largest square divisor is 4 → dp[12] = 3
If n is square-free, dp[n] = n
If n is a perfect square, dp[n] = 1

Why it is sieve-like
Classic Eratosthenes crosses out multiples of i.
Here you “cross out” square factors by iterating over multiples of i^2 and dividing them away.

```cpp
const int MAXN = 1e6 + 5;
// sieve
for (int i = 0; i < MAXN; ++i) dp[i] = i;
for (int i = 2; 1LL * i * i < MAXN; ++i) {
    int sq = 1LL * i * i;
    for (int j = sq; j < MAXN; j += sq) {
        while (dp[j] % sq == 0) dp[j] /= sq;
    }
}
```