# Codeforces 2026

# Codeforces Round 1076 (Div. 3)

## E. Product Queries

### Solution 1: dynamic programming, harmonic series

bfs

numbers are made up of x^p, they are taken to a power, where that power is at most 25 let's say, so p <= 25.

dynammic programming over 

It’s a dynamic programming over integers 1..N that computes, for every value x, the minimum number of “available” numbers whose product equals x

the time complexity is governed by a harmonic-series style sum.

1 + 2 + 3 + 4 + ... + N = NlogN

```cpp

const int INF = numeric_limits<int>::max();
int N;
vector<int> A;

void solve() {
    cin >> N;
    A.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
    }
    vector<int> ans(N + 1, INF);
    for (int x : A) {
        ans[x] = 1;
    }
    for (int i = 2; i <= N; ++i) {
        if (ans[i] == INF) continue;
        for (int j = 2; 1LL * i * j <= N; ++j) {
            if (ans[j] == INF) continue;
            ans[i * j] = min(ans[i * j], ans[i] + ans[j]);
        }
    }
    for (int i = 1; i <= N; ++i) {
        cout << (ans[i] < INF ? ans[i] : -1) << " ";
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

## F. Pizza Delivery

### Solution 1: map, dynamic programming

Always use transition to min or max y value for each x value, and pick the minimum. 

```cpp
const int INF = numeric_limits<int>::max();
int N, ax, ay, bx, by;
vector<int> X, Y;

void solve() {
    cin >> N >> ax >> ay >> bx >> by;
    X.assign(N, 0), Y.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        cin >> X[i];
    }
    for (int i = 0; i < N; ++i) {
        cin >> Y[i];
    }
    map<int, int> minMap, maxMap;
    for (int i = 0; i < N; ++i) {
        if (minMap.find(X[i]) == minMap.end()) {
            minMap[X[i]] = INF;
        }
        minMap[X[i]] = min(minMap[X[i]], Y[i]);
        maxMap[X[i]] = max(maxMap[X[i]], Y[i]);
    }
    minMap[bx] = maxMap[bx] = by;
    X.emplace_back(bx);
    sort(X.begin(), X.end());
    X.erase(unique(X.begin(), X.end()), X.end());
    int y1 = ay, y2 = ay;
    int64 a1 = 0, a2 = 0;
    for (const int x : X) {
        int mn = minMap[x], mx = maxMap[x]; // min, max
        int64 c1 = mx - mn + min(abs(y1 - mx) + a1, abs(y2 - mx) + a2);
        int64 c2 = mx - mn + min(abs(y1 - mn) + a1, abs(y2 - mn) + a2);
        y1 = mn, y2 = mx, a1 = c1, a2 = c2;
    }
    int64 ans = min(a1, a2) + abs(bx - ax);
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

## G. Paths in a Tree

### Solution 1: tree, dfs, preorder traversal, query pairs, process of elimination

```cpp
int N;
vector<vector<int>> adj;
vector<int> pre;

int query(int u, int v) {
    cout << "? " << u + 1 << " " << v + 1 << endl;
    cout.flush();
    int resp;
    cin >> resp;
    return resp; 
}

void answer(int u) {
    cout << "! " << u + 1 << endl;
    cout.flush();
}

void dfs(int u, int p = -1) {
    pre.emplace_back(u);
    for (int v : adj[u]) {
        if (v == p) continue;
        dfs(v, u);
    }
}

void solve() {
    cin >> N;
    adj.assign(N, vector<int>());
    for (int i = 0; i < N - 1; ++i) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].emplace_back(v);
        adj[v].emplace_back(u);
    }
    pre.clear();
    dfs(0);
    for (int i = 1; i < N; i += 2) {
        int u = pre[i - 1], v = pre[i];
        if (query(u, v)) {
            query(u, u) ? answer(u) : answer(v);
            return;
        }
    }
    answer(pre.back());
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

## H. Remove the Grail Tree

### Solution 1: undirected, directed graph, topological ordering, tree, dfs on tree, 

```cpp
int N;
vector<bool> vis;
vector<int> A, ind, cnt, sz;
vector<vector<int>> adj, dadj, adj1;

void dfs(int u, int p = -1) {
    vis[u] = true;
    sz[u] = 1;
    for (int v : adj[u]) {
        if (v == p) continue;
        dfs(v, u);
        if (sz[v] & 1) {
            dadj[u].emplace_back(v);
            ind[v]++;
        } else {
            dadj[v].emplace_back(u);
            ind[u]++;
        }
        sz[u] += sz[v];
    }
}

void solve() {
    cin >> N;
    A.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        cin >> A[i];
        A[i] %= 2;
    }
    adj.assign(N, vector<int>()); // 1-node undirected graph
    adj1.assign(N, vector<int>()); // edge from 1 -> 0 node
    cnt.assign(N, 0);
    for (int i = 0; i < N - 1; ++i) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        cnt[u] += A[v];
        cnt[v] += A[u];
        if (A[u] && A[v]) {
            adj[u].emplace_back(v);
            adj[v].emplace_back(u);
        }
        if (A[u] && !A[v]) {
            adj1[u].emplace_back(v);
        }
        if (!A[u] && A[v]) {
            adj1[v].emplace_back(u);
        }
    }
    dadj.assign(N, vector<int>());
    vis.assign(N, false);
    sz.assign(N, 0), ind.assign(N, 0);
    for (int i = 0; i < N; ++i) {
        if (!A[i] || vis[i]) continue;
        dfs(i);
        if (sz[i] % 2 == 0) {
            cout << "NO" << endl;
            return;
        }
    }
    vector<int> ans;
    vector<bool> zero(N, false);
    queue<int> q;
    for (int i = 0; i < N; ++i) {
        if (!A[i] && cnt[i] % 2) {
            ans.emplace_back(i);
        } else if (!A[i]) {
            zero[i] = true;
        } else if (A[i] && ind[i] == 0) {
            q.emplace(i);
        }
    }
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        ans.emplace_back(u);
        for (int v : adj1[u]) {
            if (!zero[v]) continue;
            zero[v] = false;
            ans.emplace_back(v);
        }
        for (int v : dadj[u]) {
            if (--ind[v] == 0) {
                q.emplace(v);
            }
        }
    }
    if (ans.size() < N) {
        cout << "NO" << endl;
        return;
    }
    cout << "YES" << endl;
    for (int x : ans) {
        cout << x + 1 << " ";
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