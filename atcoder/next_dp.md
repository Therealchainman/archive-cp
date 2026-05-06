# Next DP

## A - Polyomino

### Solution 1: dp, recurrence optimization

The solution counts the number of valid polyomino configurations for size N using a recurrence. Instead of storing a full DP array, it keeps only the last three states: dp0, dp1, and dp2.

```cpp
int N;

void solve() {
    cin >> N;
    int64 dp0 = 1, dp1 = 1, dp2 = 2;
    for (int i = 3; i <= N; ++i) {
        int64 dp3 = 2LL * dp2 + dp0;
        dp0 = dp1;
        dp1 = dp2;
        dp2 = dp3;
    }
    if (N == 1) {
        cout << dp1 << endl;
        return;
    }
    cout << dp2 << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## B - DAG

### Solution 1: dag dp, topological sort

The solution counts the number of ways to reach node N - 1 from node 0 in a directed acyclic graph.

It uses Kahn’s algorithm to process nodes in topological order. The DP meaning is:

```cpp
const int MOD = 998244353;
int N, M;
vector<vector<int>> adj;
vector<int> ind;

void solve() {
    cin >> N >> M;
    adj.assign(N, vector<int>());
    ind.assign(N, 0);
    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].emplace_back(v);
        ind[v]++;
    }
    vector<int> dp(N, 0);
    queue<int> q;
    for (int i = 0; i < N; ++i) {
        if (ind[i] == 0) {
            q.emplace(i);
        }
    }
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        if (u == 0) {
            dp[u] = 1;
        }
        for (int v : adj[u]) {
            dp[v] = (dp[v] + dp[u]) % MOD;
            if (--ind[v] == 0) {
                q.emplace(v);
            }
        }
    }
    cout << dp[N - 1] << endl;
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

## C - String

### Solution 1: state tracking, multidimensional dp

The solution counts strings T of length N such that none of S1, S2, or S3 appears as a subsequence.

The DP state is:

dp[a][b][c]

where:

a = how many characters of S1 have been matched as a subsequence
b = how many characters of S2 have been matched
c = how many characters of S3 have been matched

For every position in T, the algorithm tries all 26 lowercase letters. If adding a character advances one of the forbidden strings to full length, that transition is rejected.

```cpp
const int MOD = 998244353;
int N;
string s1, s2, s3;

void solve() {
    cin >> N >> s1 >> s2 >> s3;
    int L1 = s1.size(), L2 = s2.size(), L3 = s3.size();
    vector<vector<vector<int>>> dp(L1, vector<vector<int>>(L2, vector<int>(L3, 0)));
    dp[0][0][0] = 1;
    for (int i = 0; i < N; ++i) {
        vector<vector<vector<int>>> ndp(L1, vector<vector<int>>(L2, vector<int>(L3, 0)));
        for (int a = 0; a < L1; ++a) {
            for (int b = 0; b < L2; ++b) {
                for (int c = 0; c < L3; ++c) {
                    int val = dp[a][b][c];
                    if (!val) continue;
                    for (char ch = 'a'; ch <= 'z'; ++ch) {
                        int na = a + (s1[a] == ch);
                        int nb = b + (s2[b] == ch);
                        int nc = c + (s3[c] == ch);
                        if (na < L1 && nb < L2 && nc < L3) {
                            ndp[na][nb][nc] = (ndp[na][nb][nc] + val) % MOD;
                        }
                    }
                }
            }
        }
        swap(dp, ndp);
    }
    int ans = 0;
    for (int i = 0; i < L1; ++i) {
        for (int j = 0; j < L2; ++j) {
            for (int k = 0; k < L3; ++k) {
                ans = (ans + dp[i][j][k]) % MOD;
            }
        }
    }
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

## D - Banknote

### Solution 1: digit dp

The solution processes the number from least significant digit to most significant digit.

It keeps two states:

dp0 = minimum cost so far with no carry
dp1 = minimum cost so far with carry

At each digit, you can either pay the digit directly or overpay to create a carry into the next digit.

```cpp
const int INF = numeric_limits<int>::max();
string S;

void solve() {
    cin >> S;
    int dp0 = 0, dp1 = INF;
    reverse(S.begin(), S.end());
    for (char ch : S) {
        int d = ch - '0';
        int ndp0 = INF, ndp1 = INF;
        // carry 0
        if (dp0 != INF) {
            ndp0 = min(ndp0, dp0 + d);
            ndp1 = min(ndp1, dp0 + 10 - d);
        }
        // carry 1
        if (dp1 != INF) {
            ndp0 = min(ndp0, dp1 + d + 1);
            ndp1 = min(ndp1, dp1 + 9 - d);
        }
        dp0 = ndp0;
        dp1 = ndp1;
    }
    cout << min(dp0, dp1 + 1) << endl;
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

## E - Summer Vacation

### Solution 1: greedy, binary jumping

For each starting position x, the solution precomputes the best interval to take next.

The greedy idea is:

To maximize the number of non-overlapping intervals, always choose the interval that ends earliest among intervals starting at or after the current position.

The array:

successor[x]

stores the minimum right endpoint among intervals with left endpoint at least x.

Then:

succ[0][x] = successor[x] + 1

means: after taking one interval starting from position x, the next available position is right after that interval.

Binary lifting builds:

succ[k][x]

which means: the position reached after taking 2^k intervals.

For each query [l, r], the algorithm greedily jumps as far as possible using binary lifting while staying inside the query range.

This gives fast query time:

O(log N)

after preprocessing.

```cpp
const int LOG = 20;
int N, M, Q;
vector<int> successor;

void solve() {
    cin >> N >> M >> Q;
    int INF = N + 1;
    successor.assign(N + 2, INF);
    vector<vector<int>> right(N, vector<int>());
    for (int i = 0; i < M; ++i) {
        int l, r;
        cin >> l >> r;
        l--, r--;
        right[l].emplace_back(r);
    }
    for (int i = N - 1; i >= 0; --i) {
        successor[i] = successor[i + 1];
        for (int r : right[i]) successor[i] = min(successor[i], r);
    }
    vector<vector<int>> succ(LOG, vector<int>(N + 2, INF));
    for (int j = 0; j <= N + 1; ++j) {
        if (j < N && successor[j] < INF) {
            succ[0][j] = successor[j] + 1;
        } else {
            succ[0][j] = INF;
        }
    }
    for (int i = 1; i < LOG; ++i) {
        for (int j = 0; j <= N + 1; ++j) {
            succ[i][j] = succ[i - 1][ succ[i - 1][j]];
        }
    }
    while (Q--) {
        int l, r;
        cin >> l >> r;
        l--, r--;
        int ans = 0;
        for (int k = LOG - 1; k >= 0; k--) {
            if (succ[k][l] <= r + 1) {
                ans += (1 << k);
                l = succ[k][l];
            }
        }
        cout << ans << endl;
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
