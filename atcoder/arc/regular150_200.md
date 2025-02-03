# Atcoder Regular Contest 186

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

# Atcoder Regular Contest 187

## Add and Swap

### Solution 1: 

```cpp

```

## Sum of CC

### Solution 1: 

```cpp

```

# Atcoder Regular Contest 188

## Symmetric Painting

### Solution 1:  number theory, modular arithmetic, gcd, parity

1.  Really need to know the rules of gcd and modular arithmetic, and how you can move about a circle. 

```cpp
int N, K;

void solve() {
    cin >> N >> K;
    int g = gcd(N, 2 * K);
    if (g == 1 || (g == 2 && K % 2 != (K + N / 2) % 2)) {
        cout << "Yes" << endl;
    } else {
        cout << "No" << endl;
    }
}

signed main() {
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}
```

# Atcoder Regular Contest 189

## A - Reversi 2 

### Solution 1:  parity, double factorial, factorials, combinatorics, interleaving, multinomial theorem

1. So you have at some point I have groups of sizes s1, s2, s3 and so want to calculate number of ways to interleave these.
1. Where si really represents number of operations, number of operations is always size of sequence of same values divided by 2, floor division
1. There are some obvious cases where it is false though. 
1. Then calculating the number of ways internally for any group of equal digit can be done with double factorial, draw out example to see why.

```cpp
const int MOD = 998244353;
int N;
vector<int> A;

int inv(int i, int m) {
  return i <= 1 ? i : m - (int)(m/i) * inv(m % i, m) % m;
}

vector<int> fact, inv_fact;
void factorials(int n, int m) {
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

int multinomial(const vector<int> &items) {
    int cnt = 0;
    for (int x : items) {
        cnt += x / 2;
    }
    int ans = fact[cnt];
    for (int x : items) {
        ans = (ans * inv_fact[x / 2]) % MOD;
    }
    return ans;
}

void solve() {
    cin >> N;
    A.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    A.emplace_back(2);
    if (A[0] == 0) {
        cout << 0 << endl;
        return;
    }
    factorials(N + 1, MOD);
    int last = 0;
    vector<int> groups;
    for (int i = 1; i <= N; i++) {
        if (A[i] != A[i - 1]) {
            int sz = i - last;
            if (sz % 2 == 0) {
                cout << 0 << endl;
                return;
            }
            groups.emplace_back(sz);
            last = i;
        }
    }
    int ans = multinomial(groups);
    for (int x : groups) {
        for (int i = x - 2; i > 0; i -= 2) {
            ans = (ans * i) % MOD;
        }
    }
    cout << ans << endl;
}

signed main() {
    solve();
    return 0;
}
```

## 

### Solution 1: 

```cpp

```

# Atcoder Regular Contest 191

## Replace Digits

### Solution 1:  greedy, constructive, implementation, strings, lexicographically sorting

1. The catch me is that the last element in T if not used must still be used so put it in the least worst place in S. Which is whereevery S[i] is equal to to T.back() or if there isn't just replace the last element in S with T.back().

```cpp
int N, M;
string S, T;

int decode(char c) {
    return c - '0';
}

char encode(int i) {
    return i + '0';
}

void solve() {
    cin >> N >> M >> S >> T;
    vector<vector<int>> adj(10, vector<int>());
    for (int i = 0; i < M; i++) {
        adj[decode(T[i])].emplace_back(i);
    }
    bool lastRemains = true;
    for (int i = 0; i < N; ++i) {
        int cur = decode(S[i]);
        for (int j = 9; j > cur; --j) {
            if (!adj[j].empty()) {
                S[i] = encode(j);
                if (adj[j].back() == M - 1) lastRemains = false;
                adj[j].pop_back();
                break;
            }
        }
    }
    for (int i = 0; i < N && lastRemains; ++i) {
        if (S[i] == T.back()) {
            S[i] = T.back();
            lastRemains = false;
        }
    }
    if (lastRemains) S.back() = T.back();
    cout << S << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## XOR = MOD

### Solution 1: 

```cpp

```

# Atcoder Regular Contest 192

## 

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```

