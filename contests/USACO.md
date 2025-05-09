# USACO

# USACO 2024 December Contest, Silver

## Cake Game

### Solution 1:  prefix sum, suffix sum

```cpp
int N;
vector<int> A;

void solve() {
    cin >> N;
    A.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    int psum = 0, ssum = 0;
    for (int i = 0; i < N / 2 - 1; i++) {
        psum += A[i];
    }
    int ans = psum;
    for (int i = N - 1, j = N / 2 - 2; i > N / 2; i--, j--) {
        ssum += A[i];
        psum -= A[j];
        ans = max(ans, psum + ssum);
    }
    int sum = accumulate(A.begin(), A.end(), 0LL);
    cout << sum - ans << " " << ans << endl;
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

## Deforestation

### Solution 1:  min heap, line sweep, sorting, binary search

```cpp
struct Event {
    int x, tp, r, t;
    Event() {}
    Event(int x, int tp, int r = 0, int t = 0) : x(x), tp(tp), r(r), t(t) {}
    bool operator<(const Event &other) const {
        if (x != other.x) {
            return x < other.x;
        }
        return tp < other.tp;
    }
};

int N, K;
vector<int> A;
vector<Event> events;

void solve() {
    cin >> N >> K;
    A.assign(N, 0);
    events.clear();
    for (int i = 0; i < N; i++) {
        cin >> A[i];
        events.emplace_back(A[i], 1);
    }
    sort(A.begin(), A.end());
    for (int i = 0; i < K; i++) {
        int l, r, t;
        cin >> l >> r >> t;
        int cnt = upper_bound(A.begin(), A.end(), r) - lower_bound(A.begin(), A.end(), l);
        events.emplace_back(l, 0, r, cnt - t);
    }
    sort(events.begin(), events.end());
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> minheap;
    int ans = 0;
    for (auto &[x, tp, r, t] : events) {
        while (!minheap.empty() && minheap.top().second < x) {
            minheap.pop();
        }
        if (tp == 0) {
            minheap.emplace(ans + t, r);
        } else {
            if (minheap.empty() || minheap.top().first > ans) {
                ans++;
            }
        }
    }
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

## 2D Conveyor Belt

### Solution 1: 

```cpp

```

# USACO 2025 January Contest, Silver

## Cow Checkups

### Solution 1:  prefix sums, aggregation, sorting, binary search, greedy

```cpp
int N;
vector<int> A, B;
vector<vector<int>> groups;
vector<vector<int64>> psums;

int64 chooseTwo(int64 n) {
    return n * (n + 1) / 2;
}

void solve() {
    cin >> N;
    A.resize(N);
    B.resize(N);
    groups.assign(N + 1, vector<int>());
    psums.assign(N + 1, vector<int64>());
    for (int i = 0; i < N; i++) {
        cin >> A[i];
        groups[A[i]].emplace_back(min(i + 1, N - i));
    }
    for (int i = 1; i <= N; i++) {
        sort(groups[i].begin(), groups[i].end());
        psums[i].emplace_back(0);
        for (int j = 0; j < groups[i].size(); j++) {
            psums[i].emplace_back(psums[i].back() + groups[i][j]);
        }
    }
    int64 ans = 0;
    for (int i = 0; i < N; i++) {
        cin >> B[i];
        int idx = min(i + 1, N - i);
        int j = upper_bound(groups[B[i]].begin(), groups[B[i]].end(), idx) - groups[B[i]].begin();
        ans += psums[B[i]][j];
        int64 thresholdSum = static_cast<int64>(groups[B[i]].size() - j) * idx;
        ans += thresholdSum;
        if (A[i] == B[i]) {
            ans += chooseTwo(i) + chooseTwo(N - i - 1);
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

## Farmer John's Favorite Operation

### Solution 1: 

help this is wrong somehow

```cpp
int64 N, M;
map<int64, vector<bool>> events;

int ceil(int64 a, int64 b) {
    return (a + b - 1) / b;
}

void solve() {
    cin >> N >> M;
    events.clear();
    int64 cur = 0, pos = 0, neg = 0, pt = 0;
    int64 H = M / 2;
    for (int i = 0; i < N; i++) {
        int64 x;
        cin >> x;
        x %= M;
        // event 0
        events[x].emplace_back(false);
        events[x + M].emplace_back(false);
        debug("x: " , x, "event 0: ", x, x + M, "\n");
        if (x >= ceil(M, 2)) x -= M;
        cur += abs(x);
        // event 1
        int64 ev1 = M + x - H, ev2 = 2LL * M + x - H;
        if (M & 1) {
            if (x == H) {
                ev1 = 0, ev2 = M;
            }
        }
        events[ev1].emplace_back(true);
        events[ev2].emplace_back(true);
        debug("x: ", x, "event 1: " , ev1, ev2, "\n");
        if (x > 0) ++pos;
        else ++neg;
    }
    int64 ans = cur;
    for (const auto& [t, eventTypes] : events) {
        assert(eventTypes.size() > 0);
        assert(pos + neg == N);
        int64 cnt0 = 0, cnt1 = 0;
        for (bool eventType : eventTypes) {
            if (eventType) {
                cnt1++;
            } else {
                cnt0++;
            }
        }
        int64 deltaT = t - pt;
        int64 delta = neg - pos;
        cur += deltaT * delta;
        if (M & 1 && deltaT > 0) cur -= cnt1;
        debug("t: ", t, "delta: ", delta, "cnt0: ", cnt0, "cnt1: ", cnt1, "cur: ", cur, "deltaT: ", deltaT, "\n");
        pt = t;
        pos += cnt1;
        neg -= cnt1;
        neg += cnt0;
        pos -= cnt0;
        ans = min(ans, cur);
    }
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

## Table Recovery

### Solution 1: 

```cpp

```

# USACO 2025 February Contest, Silver

## The Best Lineup

### Solution 1: range max query, sparse table, sorting, greedy

```cpp
const int LOG = 31, INF = 1e9;
int N;
vector<int> A, pos;
vector<vector<int>> st;

// [L, R]
int query(int L, int R) {
    if (L > R) return -INF;
	int k = log2(R - L + 1);
	return max(st[k][L], st[k][R - (1LL << k) + 1]);
}

// j < i, move i to j
void move(int i, int j) {
    int v = A[i];
    A.erase(A.begin() + i);
    A.insert(A.begin() + j, v);
}

void solve() {
    cin >> N;
    A.assign(N, 0);
    pos.assign(N, 0);
    st.assign(LOG, vector<int>(N, -INF));
    for (int i = 0; i < N; i++) {
        cin >> A[i];
        pos[i] = i;
        st[0][i] = A[i];
    }
    for (int i = 1; i < LOG; i++) {
        for (int j = 0; j + (1 << (i - 1)) < N; j++) {
            st[i][j] = max(st[i - 1][j], st[i - 1][j + (1 << (i - 1))]);
        }
    }
    sort(pos.begin(), pos.end(), [&](int i, int j) {
        if (A[i] == A[j]) return i > j;
        return A[i] > A[j];
    });
    int l = 0;
    for (int i = 0; i < N; i++) {
        int p = pos[i];
        int v = A[p];
        if (i + 1 < N && A[pos[i + 1]] == v) {
            l = max(l, pos[i + 1] + 1);
        }
        int maxL = query(l, p - 1);
        int maxR = query(p + 1, N - 1);
        if (maxL >= maxR) {
            move(p, l);
            break;
        }
        l = max(l, p + 1);
    }
    sort(pos.begin(), pos.end(), [&](int i, int j) {
        if (A[i] == A[j]) return i < j;
        return A[i] > A[j];
    });
    l = 0;
    vector<int> ans;
    for (int i = 0; i < N; i++) {
        int p = pos[i];
        int v = A[p];
        if (p < l) continue;
        ans.emplace_back(v);
        l = p + 1;
    }
    for (int i = 0; i < ans.size(); i++) {
        if (i + 1 == ans.size()) {
            cout << ans[i] << endl;
        } else {
            cout << ans[i] << " ";
        }
    }
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

## Vocabulary Quiz

### Solution 1: tree, leaf nodes, depth, set, reverse queries

```cpp
int N, M;
vector<int> depth, P;
vector<set<int>> depthSeen;
vector<bool> isLeaf;

void solve() {
    cin >> N;
    P.assign(N + 1, -1);
    depth.assign(N + 1, 0);
    int M = N + 1;
    isLeaf.assign(N + 1, true);
    depthSeen.assign(N + 1, set<int>());
    for (int i = 1; i <= N; i++) {
        cin >> P[i];
        depth[i] = depth[P[i]] + 1;
        if (isLeaf[P[i]]) {
            M--;
            isLeaf[P[i]] = false;
        }
    }
    vector<int> queries;
    vector<int> ans;
    while (M--) {
        int u;
        cin >> u;
        queries.emplace_back(u);
    }
    reverse(queries.begin(), queries.end());
    for (int u : queries) {
        u = P[u];
        bool isCompleted = true;
        while (u != -1) {
            if (depthSeen[depth[u]].count(u)) {
                ans.emplace_back(depth[u] + 1);
                isCompleted = false;
                break;
            }
            depthSeen[depth[u]].insert(u);
            u = P[u];
        }
        if (isCompleted) {
            ans.emplace_back(0);
        }
    }
    reverse(ans.begin(), ans.end());
    for (int u : ans) {
        cout << u << endl;
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

## Transforming pairs

### Solution 1: recursion

1. only partial credit, waiting for editorial to get the full credit solution, it is something number theory, extended euclidean algorithm? gcd?

```cpp
const int64 INF = 1e18;
int64 A, B, C, D;

int64 dfs(int64 a, int64 b) {
    if (a == C && b == D) return 0;
    if (a > C || b > D) return INF;
    int64 res = INF;
    int64 v1 = dfs(a + b, b);
    int64 v2 = dfs(a, a + b);
    if (v1 != INF) res = min(res, v1 + 1);
    if (v2 != INF) res = min(res, v2 + 1);
    return res;
}

void solve() {
    cin >> A >> B >> C >> D;
    if (C < A || D < B) {
        cout << -1 << endl;
        return;
    }
    int64 ans = dfs(A, B);
    cout << (ans != INF ? ans : -1) << endl;
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

# USACO 2025 US Open Contest, Gold

## Moo Decomposition

### Solution 1: binary exponentiation, binomial coefficients, factorials, combinatorics, reverse

```cpp
const int64 MOD = 1e9 + 7, MAXN = 1e6 + 5;
int64 K, N, L;
string S;

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
    cin >> K >> N >> L >> S;
    int64 ans = 1, cnt = 0;
    for (int i = N - 1; i >= 0; i--) {
        if (S[i] == 'M') {
            ans = (ans * choose(cnt, K, MOD)) % MOD;
            cnt -= K;
        } else {
            cnt++;
        }
    }
    ans = exponentiation(ans, L, MOD);
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

## Election Queries

### Solution 1: 

partial solve, O(NQ) time complexity

```cpp
int N, Q;
vector<int> A, counts;

void solve() {
    cin >> N >> Q;
    A.resize(N);
    counts.assign(N + 1, 0);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
        counts[A[i]]++;
    }
    priority_queue<pair<int, int>> mheap;
    for (int i = 1; i <= N; i++) {
        if (counts[i] == 0) continue;
        mheap.emplace(counts[i], i);
    }
    while (Q--) {
        int idx, val;
        cin >> idx >> val;
        idx--;
        counts[A[idx]]--;
        A[idx] = val;
        counts[val]++;
        mheap.emplace(counts[val], val);
        while (counts[mheap.top().second] != mheap.top().first) mheap.pop();
        auto [cmax, _] = mheap.top();
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> minheap;
        priority_queue<pair<int, int>> maxheap;
        for (int i = 1; i <= N; i++) {
            if (counts[i] == 0) continue;
            minheap.emplace(counts[i], i);
            maxheap.emplace(counts[i], i);
        }
        int ans = 0;
        int minIdx = N, maxIdx = 0;
        while (!minheap.empty()) {
            auto [cl, i] = minheap.top();
            minheap.pop();
            int cr = cmax - cl;
            while (!maxheap.empty() && maxheap.top().first >= cr) {
                auto [_, j] = maxheap.top();
                maxheap.pop();
                minIdx = min(minIdx, j);
                maxIdx = max(maxIdx, j);
            }
            ans = max(ans, maxIdx - i);
            ans = max(ans, i - minIdx);
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

## OohMoo Milk

### Solution 1: 

```cpp

```

# TBD

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