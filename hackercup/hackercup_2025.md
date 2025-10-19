# Meta Hacker Cup 2025

# Practice Round

## Warm up

### Solution 1: sorting, reverse iteration, hash map

```cpp
int N;
vector<int> A, B;

void solve() {
    cin >> N;
    A.assign(N, 0);
    B.assign(N, 0);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    for (int i = 0; i < N; i++) {
        cin >> B[i];
    }
    vector<pair<int, int>> pairs;
    for (int i = 0; i < N; i++) {
        pairs.emplace_back(A[i], i);
    }
    sort(pairs.rbegin(), pairs.rend());
    vector<int> last(N + 1, -1);
    vector<pair<int, int>> ans;
    for (const auto &[value, index] : pairs) {
        int target = B[index];
        if (value > target) {
            cout << -1 << endl;
            return;
        }
        if (value != target && last[target] == -1) {
            cout << -1 << endl;
            return;
        }
        if (value != target) {
            ans.emplace_back(index, last[target]);
        }
        last[value] = index;
    }
    reverse(ans.begin(), ans.end());
    cout << ans.size() << endl;
    for (const auto &[i, j] : ans) {
        cout << i + 1 << " " << j + 1 << endl;
    }
}
```

## Zone in

### Solution 1: multisource bfs on grid, flood fill with bfs

```cpp
int R, C, S;
vector<vector<char>> grid;

bool inBounds(int r, int c) {
    return r >= 0 && r < R && c >= 0 && c < C;
}

int bfs(int r, int c) {
    queue<pair<int, int>> q;
    q.emplace(r, c);
    grid[r][c] = '#';
    int ans = 0;
    while (!q.empty()) {
        auto [r, c] = q.front();
        q.pop();
        ans++;
        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                if (abs(dr) + abs(dc) != 1) continue;
                int nr = r + dr, nc = c + dc;
                if (!inBounds(nr, nc) || grid[nr][nc] == '#') continue;
                grid[nr][nc] = '#';
                q.emplace(nr, nc);
            }
        }
    }
    return ans;
}

void solve() {
    cin >> R >> C >> S;
    R += 2, C += 2;
    grid.assign(R, vector<char>(C, '#'));
    for (int i = 0; i < R - 2; ++i) {
        for (int j = 0; j < C - 2; ++j) {
            cin >> grid[i + 1][j + 1];
        }
    }
    queue<pair<int, int>> q;
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            if (grid[i][j] == '#') {
                q.emplace(i, j);
            }
        }
    }
    for (int i = 0; i < S; ++i) {
        int sz = q.size();
        while (sz--) {
            auto [r, c] = q.front();
            q.pop();
            for (int dr = -1; dr <= 1; ++dr) {
                for (int dc = -1; dc <= 1; ++dc) {
                    if (abs(dr) + abs(dc) != 1) continue;
                    int nr = r + dr, nc = c + dc;
                    if (!inBounds(nr, nc) || grid[nr][nc] == '#') continue;
                    grid[nr][nc] = '#';
                    q.emplace(nr, nc);
                }
            }
        }
    }
    int ans = 0;
    for (int i = 0; i < R; ++i) {
        for (int j = 0; j < C; ++j) {
            if (grid[i][j] == '#') continue;
            ans = max(ans, bfs(i, j));
        }
    }
    cout << ans << endl;
}
```

## Monkey Around

### Solution 1: constructive, modular arithmetic, weakly decreasing array

```cpp
int N;
vector<int> A;

void solve() {
    cin >> N;
    A.assign(N, 0);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    vector<int> seg, rot;
    int prv = N + 1, start = 1, seenOne = true;
    for (int x : A) {
        if (x == 1 && !seenOne) {
            seenOne = true;
        } else if (x == start || x != prv + 1) {
            seg.emplace_back(0);
            rot.emplace_back(x - 1);
            start = x;
            seenOne = x == 1;
        }
        seg.back()++;
        prv = x;
    }
    int M = seg.size();
    for (int i = M - 2; i >= 0; --i) {
        int v = (rot[i] - rot[i + 1]) % seg[i];
        if (v < 0) v += seg[i];
        v += seg[i];
        v %= seg[i];
        rot[i] = v + rot[i + 1];
    }
    int cnt = M + rot[0];
    seg.emplace_back(0); // dummy values
    rot.emplace_back(0);
    cout << cnt << endl;
    for (int i = 0; i < M; ++i) {
        cout << 1 << " " << seg[i] << endl; // place permutation
        int rotations = rot[i] - rot[i + 1];
        while (rotations--) {
            cout << 2 << endl;
        }
    }
}
```

## Plan Out

### Solution 1: Eulerian Circuit, undirected graph, connected components, Hierholzers algorithm

1. Connected component: a maximal set of vertices where every pair is connected by some path.
1. Eulerian component: a connected component whose every vertex has even degree. In an undirected graph, this condition is necessary and sufficient for the existence of an Eulerian circuit inside that component.
1. Use the handshaking lemma to know that with the odd degree vertex all be connected to some dummy node, means the degree of that dummy node will be even.
1. Even lengthed Eulerian circuits (even number of edges) will be balanced.
1. Odd lengthed Eulerian circuits will be imbalanced, where the starting node, will have imbalance of 2.

```cpp
int N, M;
vector<vector<pair<int, int>>> adj;
vector<int> day;
vector<bool> used;

// start node, eulerian circuits, all even degree nodes
void hierholzers(int source) {
    stack<pair<int, int>> stk;
    stk.emplace(source, -1);
    vector<int> edgeTour;
    while (!stk.empty()) {
        auto [u, eid] = stk.top();
        while (!adj[u].empty() && used[adj[u].back().second]) adj[u].pop_back();
        if (adj[u].empty()) {
            edgeTour.emplace_back(eid);
            stk.pop();
        } else {
            // take one neighbor and remove the edge from both sides
            auto [v, i] = adj[u].back();
            adj[u].pop_back();
            if (!used[i]) {
                used[i] = true;
                stk.emplace(v, i);
            }
        }
    }
    for (int i = 0; i < edgeTour.size(); ++i) {
        if (edgeTour[i] < 0) continue;
        day[edgeTour[i]] = i % 2 == 0 ? 1 : 2;
    }
}

void solve() {
    cin >> N >> M;
    adj.assign(N + 1, vector<pair<int, int>>());
    vector<pair<int, int>> edges;
    vector<int> deg(N + 1, 0);
    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        adj[u].emplace_back(v, i);
        adj[v].emplace_back(u, i);
        deg[u]++, deg[v]++;
        edges.emplace_back(u, v);
    }
    // for all odd degree vertices point them to the dummy node 0, so that we have all connected components with eulerian circuits
    for (int i = 1; i <= N; ++i) {
        if (deg[i] % 2 == 0) continue;
        adj.emplace_back(vector<pair<int, int>>());
        adj[0].emplace_back(i, M);
        adj[i].emplace_back(0, M);
        M++;
    }
    vector<bool> vis(N + 1, false);
    day.assign(M, 0);
    used.assign(M, false);
    for (int i = 0; i <= N; ++i) {
        if (vis[i]) continue;
        hierholzers(i);
    }
    string assigned;
    vector<int64> d1(N + 1, 0), d2(N + 1, 0);
    for (int i = 0; i < edges.size(); ++i) {
        auto [u, v] = edges[i];
        if (day[i] == 1) {
            d1[u]++;
            d1[v]++;
            assigned += '1';
        } else {
            d2[u]++;
            d2[v]++;
            assigned += '2';
        }
    }
    int64 ans = 0;
    for (int i = 1; i <= N; ++i) {
        ans += d1[i] * d1[i] + d2[i] * d2[i];
    }
    cout << ans << " " << assigned << endl;
}
```

## Pay Off

### Solution 1:

```cpp

```

# Round 1

## Snake Scales (Part 1)

### Solution 1: max absolute difference

```cpp
int N;
vector<int> A;

void solve() {
    cin >> N;
    A.assign(N, 0);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    int ans = 0;
    for (int i = 1; i < N; ++i) {
        ans = max(ans, abs(A[i] - A[i - 1]));
    }
    cout << ans << endl;
}
```

## Snake Scales (Part 2)

### Solution 1: greedy, binary search

```cpp
int N;
vector<int> A;
vector<bool> vis;

bool possible(int target) {
    vector<int> cands;
    int cnt = 0;
    vis.assign(N, false);
    for (int i = 0; i < N; ++i) {
        if (A[i] <= target) {
            cands.emplace_back(i); // jump to these from platform
            vis[i] = true;
            cnt++;
        }
    }
    for (int idx : cands) {
        for (int i = idx - 1; i >= 0; --i) {
            if (vis[i]) break;
            if (abs(A[i] - A[i + 1]) > target) break;
            vis[i] = true;
            cnt++;
        }
        for (int i = idx + 1; i < N; ++i) {
            if (vis[i]) break;
            if (abs(A[i] - A[i - 1]) > target) break;
            vis[i] = true;
            cnt++;
        }
    }
    return cnt == N;
}

void solve() {
    cin >> N;
    A.assign(N, 0);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    int lo = 0, hi = 2e9;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (possible(mid)) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    cout << lo << endl;
}
```

## Final Product (Part 1)

### Solution 1: constructive

```cpp
int N, A, B;

void solve() {
    cin >> N >> A >> B;
    int start = 1;
    for (int i = 1; i <= A; ++i) {
        if (B % i == 0) {
            start = i;
        }
    }
    cout << start << " ";
    for (int i = 1; i < 2 * N - 1; ++i) {
        cout << 1 << " ";
    }
    cout << B / start << endl;
}
```

## Final Product (Part 2)

### Solution 1: prime factorization, dfs with pruning, binomial coefficient, combinatorics, stars and bars method

Want to find the number of ordered N-tuples that multiple to d

```cpp
const int MOD = 1e9 + 7, MAXN = 50;
int64 N, A, B, ans, M, C[MAXN];
vector<int64> factors;
vector<int> expCounts;
map<int64, int> factorCount;

int64 inv(int i, int64 m) {
  return i <= 1 ? i : m - (m / i) * inv(m % i, m) % m;
}

void precompute(int n, int64 m) {
    memset(C, 0, sizeof(C));
    C[0] = 1;
    int64 cur = 1;
    for (int i = 1; i < MAXN; ++i) {
        int64 cand = static_cast<int128>(n + i - 1) * inv(i, m) % m;
        cur = cur * cand % m;
        C[i] = cur;
    }
}

void primeFactors(int64 n) {
    while (n % 2 == 0) {
        factorCount[2]++;
        n /= 2;
    }
    for (int64 i = 3; i * i <= n; i += 2) {
        while (n % i == 0) {
            factorCount[i]++;
            n /= i;
        }
    }
    if (n > 2) {
        factorCount[n]++;
    }
}

void dfs(int idx, int64 val, int64 ways) {
    if (idx == M) {
        ans = (ans + ways) % MOD;
        return;
    }
    int64 factor = factors[idx];
    int64 exp = expCounts[idx];
    int128 cur = val;
    dfs(idx + 1, val, ways * C[exp] % MOD);
    for (int i = 1; i <= exp; ++i) {
        cur *= factor;
        if (cur > A) break;
        int64 nways = ways * C[i] % MOD;
        nways = nways * C[exp - i] % MOD;
        dfs(idx + 1, cur, nways);
    }
}

void solve() {
    cin >> N >> A >> B;
    precompute(N, MOD);
    factorCount.clear();
    primeFactors(B);
    factors.clear();
    expCounts.clear();
    for (const auto &[factor, count] : factorCount) {
        if (!count) continue;
        factors.emplace_back(factor);
        expCounts.emplace_back(count);
    }
    M = factors.size();
    ans = 0;
    dfs(0, 1, 1);
    cout << ans << endl;
}
```

## Narrowing Down

### Solution 1: combinatorics, sum of lengths of all subarrays, combinatoric triplets, prefix xor,

at most m - 1 operations needs so start from the back.

sum of lengths of all subarrays

Looking for range of xor that equal to 0

```cpp
int N;
vector<int> A;

int64 choose2(int64 n) {
    return n * (n - 1) / 2;
}

int64 choose3(int64 n) {
    return n * (n - 1) * (n - 2) / 6;
}

void solve() {
    cin >> N;
    A.assign(N, 0);
    int pref = 0;
    unordered_map<int, int> freq;
    freq[0]++;
    for (int i = 0; i < N; i++) {
        cin >> A[i];
        pref ^= A[i];
        freq[pref]++;
    }
    int64 ans = choose3(N + 2), pairs = 0, triplets = 0;
    for (auto &[k, v] : freq) {
        pairs += choose2(v);
        triplets += choose3(v);
    }
    ans -= pairs;
    ans -= triplets;
    cout << ans << endl;
}
```

##

### Solution 1:

```cpp

```


# Round 2

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

# Round 3

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
