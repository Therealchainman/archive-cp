# Starters 135

## Graph Cost

### Solution 1:  suffix min, greedy

```cpp
const int INF = 1e9;
int N;
vector<int> A, smin;

void solve() {
    cin >> N;
    A.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    smin.assign(N, INF);
    for (int i = N - 1; i >= 0; i--) {
        smin[i] = A[i];
        if (i + 1 < N) smin[i] = min(smin[i], smin[i + 1]);
    }
    int ans = 0;
    int i = 0;
    for (int j = 0; j < N; j++) {
        if (A[j] <= A[i] || A[j] == smin[j]) {
            ans += (j - i) * max(A[i], A[j]);
            i = j;
        }
    }
    ans += (N - i - 1) * max(A[i], A.end()[-1]);
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

## Limit of MEX

### Solution 1:  monotonic stack, size of left and right, combinatorics

```cpp
int N;
vector<int> A, R, L, last;

int calc(int n) {
    return n * (n - 1) / 2;
}

void solve() {
    cin >> N;
    A.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    int ans = calc(N);
    R.resize(N);
    L.resize(N);
    stack<int> stk;
    for (int i = 0; i < N; i++) {
        while (!stk.empty() && A[i] >= A[stk.top()]) {
            stk.pop();
        }
        L[i] = i - (stk.empty() ? -1 : stk.top());
        stk.push(i);
    }
    while (!stk.empty()) {
        stk.pop();
    }
    for (int i = N - 1; i >= 0; i--) {
        while (!stk.empty() && A[i] > A[stk.top()]) {
            stk.pop();
        }
        R[i] = (stk.empty() ? N : stk.top()) - i;
        stk.push(i);
    }
    for (int i = 0; i < N; i++) {
        ans += A[i] * L[i] * R[i];
    }
    last.assign(N + 1, 0);
    for (int i = 0; i < N; i++) {
        if (!last[A[i]]) ans -= calc(N);
        ans += calc(i - last[A[i]]);
        last[A[i]] = i + 1;
    }
    for (int i = 0; i <= N; i++) {
        if (!last[i]) continue;
        ans += calc(N - last[i]);
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

## Milky-Dark Chocolates

### Solution 1:  dynammic programming, prefix sum, rearrangement of variables to simplify

```cpp
const int INF = 1e18;
int N, K;
vector<int> A, B, dp, ndp, pA, pB;

void solve() {
    cin >> N >> K;
    A.resize(N);
    B.resize(N);
    pA.resize(N);
    pB.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
        pA[i] = A[i];
        if (i > 0) pA[i] += pA[i - 1];
    }
    for (int i = 0; i < N; i++) {
        cin >> B[i];
        pB[i] = B[i];
        if (i > 0) pB[i] += pB[i - 1];
    }
    dp.assign(N + 1, INF);
    dp[0] = 0;
    for (int len = 0; len < K; len++) {
        ndp.assign(N + 1, INF);
        int minB = dp[0], minA = dp[0];
        for (int i = 1; i <= N; i++) {
            ndp[i] = min(minA + pA[i - 1], minB + pB[i - 1]);
            minB = min(minB, dp[i] - pB[i - 1]);
            minA = min(minA, dp[i] - pA[i - 1]);
        }
        swap(dp, ndp);
    }
    cout << dp.end()[-1] << endl;
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

## Envious Pile

### Solution 1:  dfs, spanning tree, path by dfs backtracking, undirected graph, connected components, greedy

```cpp
int N, W;
vector<vector<pair<int, int>>> adj;
vector<int> A, ans;
vector<bool> vis;

bool dfs(int u, int p = -1, int idx = -1) {
    if (idx != -1) {
        ans.push_back(idx);
    }
    if (u < A[0]) return true;
    for (auto [v, i]: adj[u]) {
        if (v == p || vis[v]) continue;
        vis[v] = true;
        if (dfs(v, u, i)) return true;
    }
    ans.pop_back();
    return false;
}

void solve() {
    cin >> N >> W;
    A.resize(N);
    int MAX = 0;
    for (int i = 0; i < N; i++) {
        cin >> A[i];
        MAX = max(MAX, A[i]);
    }
    adj.assign(MAX + 1, vector<pair<int, int>>());
    vis.assign(MAX + 1, false);
    // CONSTRUCT THE UNDIRECTED GRAPH
    for (int x = 1; x <= MAX; x++) {
        for (int i = 0; i < N; i++) {
            if (A[i] <= x) continue;
            adj[x].push_back({A[i] - x, i});
            adj[A[i] - x].push_back({x, i});
        }
    }
    ans.clear();
    if (!dfs(W)) {
        cout << -1 << endl;
        return;
    }
    cout << ans.size() + N << endl;
    for (int x: ans) {
        cout << x + 1 << " ";
    }
    for (int i = 1; i <= N; i++) {
        cout << i << " ";
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