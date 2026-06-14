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

## F - Set

### Solution 1: 

```cpp

```

## G - Mouth

### Solution 1: segment tree, dynamic kadane's algorithm

Why contiguous? Because you can move left, right, or stay, and every move throws one candy at your current position. If you want to reach people at positions l and r, you must pass through every person between them.

For each position inside the chosen segment:

If A[i] > 0, you can give exactly A[i] candies, so the cost improves by A[i].

If A[i] == 0, visiting them forces you to give at least one candy, so the cost gets worse by 1.

The transformed array turns the problem into:

Find the maximum sum contiguous subarray.

That is exactly what Kadane’s algorithm solves.

```cpp
const int64 INF = 1e17;

struct Node {
    int64 pref, suf, sum, best;
    Node() : pref(-INF), suf(-INF), sum(0), best(-INF) {}
    Node(int64 val) : pref(val), suf(val), sum(val), best(val) {}
};

template<class Node>
struct SegmentTree {
    struct Configuration {
        const Node neutral;                           // identity for merge
        function<Node(const Node&, const Node&)> merge;           // combine two nodes
    } config;

    int size = 0;
    vector<Node> nodes;

    SegmentTree(int n, Configuration config) : config(config) { init(n); }

    void init(int num_nodes) {
        size = 1;
        while (size < num_nodes) size *= 2;
        nodes.assign(size * 2, config.neutral);
    }

    // this is for assign, for addition change to += val
    void update(int segment_idx, const Node& val) {
        segment_idx += size;
        nodes[segment_idx] = val; // += val if want addition, to track frequency
        for (segment_idx >>= 1; segment_idx >= 1; segment_idx >>= 1) pull(segment_idx);
    }

    Node query(int left, int right) {
        left += size, right += size;
        Node left_acc = config.neutral;
        Node right_acc = config.neutral;
        while (left <= right) {
           if (left & 1) {
                // res on left
                left_acc = config.merge(left_acc, nodes[left++]);
            }
            if (~right & 1) {
                // res on right
                right_acc = config.merge(nodes[right--], right_acc);
            }
            left >>= 1, right >>= 1;
        }
        return config.merge(left_acc, right_acc);
    }
    private:
        inline void pull(int segment_idx) { nodes[segment_idx] = config.merge(nodes[segment_idx << 1], nodes[segment_idx << 1 | 1]); }
};

int N, Q;
vector<int> A;

SegmentTree<Node>::Configuration cfg{
    Node(),
    [](const Node &x, const Node &y) {
        Node res;
        res.pref = max(x.pref, x.sum + y.pref);
        res.suf = max(y.suf, y.sum + x.suf);
        res.sum = x.sum + y.sum;
        res.best = max({x.best, y.best, x.suf + y.pref});
        return res;
    }
};

void solve() {
    cin >> N >> Q;
    int64 total = 0;
    A.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
        total += A[i];
    }
    SegmentTree<Node> seg(N, cfg);
    for (int i = 0; i < N; ++i) {
        int v = A[i] > 0 ? A[i] : -1;
        seg.update(i, Node(v));
    }
    while (Q--) {
        int i, v;
        cin >> i >> v;
        i--;
        // update A[i] = v
        total = total - A[i] + v;
        A[i] = v;
        if (v == 0) v = -1;
        seg.update(i, Node(v));
        int64 ans = total - max<int64>(0, seg.query(0, N - 1).best);
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

## H - Coin

### Solution 1: 

When a DP seems to require storing a set of possible values, check whether the set has structure.

1. Naive state space is too large.
2. Each cell’s reachable coin counts secretly form an interval.
3. The proof uses adjacent swaps of D/R path strings.
4. Min/max DP is enough.
5. Difference array efficiently counts interval coverage.
6. The problem asks about reachable cells, not number of paths.

```cpp
int N;
vector<vector<char>> grid;

void solve() {
    cin >> N;
    grid.assign(N, vector<char>(N));
    for (int i = 0; i < N; ++i) {
        string s;
        cin >> s;
        for (int j = 0; j < N; ++j) {
            grid[i][j] = s[j];
        }
    }
    vector<int> diff(2 * N, 0);
    vector<vector<int>> dpMin(N, vector<int>(N, N * N)), dpMax(N, vector<int>(N, 0));
    dpMin[0][0] = 0;
    for (int r = 0; r < N; ++r) {
        for (int c = 0; c < N; ++c) {
            if (r > 0) {
                dpMin[r][c] = min(dpMin[r][c], dpMin[r - 1][c]);
                dpMax[r][c] = max(dpMax[r][c], dpMax[r - 1][c]);
            }
            if (c > 0) {
                dpMin[r][c] = min(dpMin[r][c], dpMin[r][c - 1]);
                dpMax[r][c] = max(dpMax[r][c], dpMax[r][c - 1]);
            }
            if (grid[r][c] == '@') {
                dpMin[r][c]++;
                dpMax[r][c]++;
            }
            diff[dpMin[r][c]]++;
            diff[dpMax[r][c] + 1]--;
        }
    }
    int ans = 0;
    for (int i = 0; i < 2 * N - 1; ++i) {
        ans += diff[i];
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

## I - Update Positions

### Solution 1: insertion dp, combinatorics, stars and bars

Once larger values are already placed, inserting smaller values cannot change which of those larger values are prefix/suffix max updates.
The smaller values can only add a new prefix update if placed at the far left, and only add a new suffix update if placed at the far right.

DP definition
dp[i][l][r]

means:

Number of sequences using only values i, i + 1, ..., mx that have exactly l prefix max update positions and exactly r suffix max update positions.

So when processing value i, all values greater than i have already been arranged.

| Case        | New prefix? | New suffix? |   Stars |   Boxes | Formula              | Source state        |
| ----------- | ----------: | ----------: | ------: | ------: | -------------------- | ------------------- |
| Middle only |          No |          No |     `c` | `M - 1` | (\binom{c+M-2}{M-2}) | `dp[i+1][l][r]`     |
| Left side   |         Yes |          No | `c - 1` |     `M` | (\binom{c+M-2}{M-1}) | `dp[i+1][l-1][r]`   |
| Right side  |          No |         Yes | `c - 1` |     `M` | (\binom{c+M-2}{M-1}) | `dp[i+1][l][r-1]`   |
| Both sides  |         Yes |         Yes | `c - 2` | `M + 1` | (\binom{c+M-2}{M})   | `dp[i+1][l-1][r-1]` |


```cpp
const int MOD = 998244353, MAXN = 500;
int N, L, R, freq[MAXN];
vector<int> A;

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
    if (n < 0 || r < 0 || n < r) return 0;
    return (fact[n] * inv_fact[r] % m) * inv_fact[n - r] % m;
}

void solve() {
    cin >> N >> L >> R;
    A.resize(N);
    memset(freq, 0, sizeof(freq));
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
        freq[A[i]]++;
    }
    int mx = *max_element(A.begin(), A.end());
    vector<vector<vector<int64>>> dp(mx + 1, vector<vector<int64>>(L + 1, vector<int64>(R + 1, 0)));
    dp[mx][1][1] = 1;
    int M = freq[mx];
    for (int i = mx - 1; i > 0; --i) {
        int c = freq[i];
        for (int l = 1; l <= L; ++l) {
            for (int r = 1; r <= R; ++r) {
                if (!c) {
                    dp[i][l][r] = dp[i + 1][l][r];
                    continue;
                }
                // 1. add all to middle
                dp[i][l][r] = (dp[i][l][r] + dp[i + 1][l][r] * choose(c + M - 2, M - 2, MOD)) % MOD;
                // 2. add one to the leftmost
                dp[i][l][r] = (dp[i][l][r] + dp[i + 1][l - 1][r] * choose(c + M - 2, M - 1, MOD)) % MOD;
                // 3. add one to the rightmost
                dp[i][l][r] = (dp[i][l][r] + dp[i + 1][l][r - 1] * choose(c + M - 2, M - 1, MOD)) % MOD;
                // 4. add one to the leftmost and one to the rightmost
                if (c > 1) {
                    dp[i][l][r] = (dp[i][l][r] + dp[i + 1][l - 1][r - 1] * choose(c + M - 2, M, MOD)) % MOD;
                }
            }
        }
        M += c;
    }
    cout << dp[1][L][R] << endl;
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
