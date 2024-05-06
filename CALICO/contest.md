

```py

```

```py

```

```py

```

```py

```

```cpp
int N, S, E, C, L;
vector<vector<int>> adj;
vector<int> vis, cycle;
double cprob;

bool dfs(int u, int p) {
    if (vis[u]) {
        C = u;
        return true;
    }
    vis[u] = 1;
    bool res = false;
    for (int v : adj[u]) {
        if (v == p) continue;
        if (dfs(v, u)) {
            cycle[u] = 1;
            res = true;
        }
    }
    vis[u] = 0;
    return C == u ? false : res;
}

void dfs2(int u, int p) {
    if (vis[u]) return;
    int c = adj[u].size();
    double prob = 1.0;
    if (c > 0) {
        prob = 1.0 / c;
    }
    vis[u] = 1;
    for (int v : adj[u]) {
        if (cycle[u] && cycle[v]) {
            cprob *= prob;
        }
        if (v == p) continue;
        dfs2(v, u);
    }
    vis[u] = 0;
}

double converge(double pr, int depth) {
    double eps = 1e-12;
    double cur = pr * depth;
    double ans = 0.0;
    int loops = 0;
    while (cur > eps) {
        ans += cur;
        loops++;
        cur = pr * pow(cprob, loops) * (depth + loops * L);
    }
    return ans;
}

double dfs1(int u, int p, double pr = 1.0, int depth = 0, bool is_cycle = false) {
    if (vis[u]) return 0.0;
    int c = adj[u].size();
    double prob = 1.0;
    if (c > 0) {
        prob = 1.0 / c;
    }
    double ans = 0;
    vis[u] = 1;
    for (int v : adj[u]) {
        if (v == p) continue;
        ans += dfs1(v, u, pr * prob, depth + 1, is_cycle | cycle[u]);
    }
    if (c == 0) {
        if (is_cycle) {
            ans += converge(pr, depth);
        } else {
            ans += pr * depth;
        }
    }
    vis[u] = 0;
    return ans;
}

void solve() {
    cin >> N >> S >> E;
    S--; E--;
    adj.assign(N, vector<int>());
    cycle.assign(N, 0);
    C = -1;
    for (int i = 1; i < N; i++) {
        int p;
        cin >> p;
        p--;
        if (S == i && E == p) {
            C = p;
            cycle[i] = cycle[p] = 1;
        }
        adj[p].push_back(i);
    }
    adj[S].push_back(E);
    vis.assign(N, 0);
    if (C == -1) {
        dfs(0, -1);
    }
    L = accumulate(cycle.begin(), cycle.end(), 0);
    cprob = 1.0;
    if (L > 0) {
        vis.assign(N, 0);
        dfs2(0, -1);
    }
    vis.assign(N, 0);
    double ans = dfs1(0, -1);
    cout << fixed << setprecision(10) << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    // freopen("input.txt", "r", stdin);
    // freopen("output.txt", "w", stdout);
    int T;
    cin >> T;
    while (T--) {
        solve();
    }
    return 0;
}
```