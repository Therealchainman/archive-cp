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

##

### Solution 1: 

```cpp

```