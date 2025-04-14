# Rutgers Contests

# Rutgers University Programming Contest Spring 2025

## Ramen Packs

### Solution 1: bitmask, binary, powers of two

```cpp
int N;
 
bool isSet(int mask, int i) {
    return (mask >> i) & 1;
}
 
void solve() {
    cin >> N;
    vector<int> A, B;
    for (int i = 0; i < 32; i++) {
        if (isSet(N, i)) {
            if (i % 2 == 0) {
                A.emplace_back(1 << i / 2);
            } else {
                B.emplace_back(1 << i / 2);
            }
        }
    }
    int K = A.size() + B.size();
    cout << K << " ";
    for (int x : A) {
        cout << "A" << x << " ";
    }
    for (int x : B) {
        cout << "B" << x << " ";
    }
    cout << endl;
}
 
signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}
```

## Thomas

### Solution 1: N-Dimensional Hypercube, bitmask, even parity

1. So a hypercube the adjacent vertices are the ones that differ by 1 bit, so we can use a bitmask to represent the vertices.
2. Also bipartite graph, and you just want one bipartite partition. 

```cpp
int K;
 
bool isSet(int mask, int i) {
    return (mask >> i) & 1;
}
 
void solve() {
    cin >> K;
    int cnt = 1 << (K - 1);
    cout << cnt << endl;
    for (int mask = 0; mask < (1 << K); mask++) {
        if (__builtin_popcount(mask) % 2 == 0) {
            string ans = "";
            for (int i = 0; i < K; i++) {
                if (isSet(mask, i)) ans += '1';
                else ans += '0';
            }
            cout << ans << endl;
        }
    }
}
 
signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Another Expected Value Problem

### Solution 1: expected values, average, cancel

1. Do the math for one layer, and you see that the expected values cancel for each operation, so just take average

```cpp

const int64 MOD = 1e9 + 7;
int N, K;
 
int inv(int64 i, int64 m) {
    return i <= 1 ? i : m - (m/i) * inv(m % i, m) % m;
}
 
void solve() {
    cin >> N >> K;
    int64 sum = 0;
    for (int i = 0, x; i < N; ++i) {
        cin >> x;
        sum = (sum + x) % MOD;
    }
    int64 ans = sum * inv(N, MOD) % MOD;
    cout << ans << endl;
}
 
signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}
```

## Unfair Game

### Solution 1: game, division

1. Really simulation and observed that alice can only win in the first and second move, if not it is lost for Alice and Bob wins.

```cpp
int a, b, n;
 
int ceil(int x, int y) {
    return (x + y - 1) / y;
}
int floor(int x, int y) {
    return x / y;
}
 
void solve() {
    cin >> a >> b >> n;
    if (n < a + b) {
        cout << "Alice" << endl;
        return;
    }
    if (ceil(n - a, 2) < b) {
        cout << "Alice" << endl;
        return;
    }
    if (floor(n - a, 2) >= a && ceil(n - a, 2) < 2 * b) {
        cout << "Alice" << endl;
        return;
    } 
    cout << "Bob" << endl;
}
 
signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}
```

## Subsequence Mex

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