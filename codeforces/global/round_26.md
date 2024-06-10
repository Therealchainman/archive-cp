# Codeforces Global Round 26

##

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp
int N, ans;
vector<vector<int>> adj;
vector<int> max_leaves, deg, sz, p_max_leaves;

void dfs1(int u, int p) {
    max_leaves[u] = 0;
    sz[u] = 1;
    bool is_leaf = true;
    for (int v : adj[u]) {
        if (v == p) continue;
        dfs1(v, u);
        max_leaves[u] += max_leaves[v];
        sz[u] += sz[v];
        is_leaf = false;
    }
    if (is_leaf) max_leaves[u]++;
    cout << u << " " << sz[u] << endl;
}

void dfs2(int u, int p) {
    int cur = p_max_leaves[u] + max_leaves[u];
    // cout << u << " " << max_leaves[u] << endl;
    if (p != -1 && deg[p] == 2 && N - sz[u] > 2) cur++;
    if (u == 0 and deg[u] == 1) cur++;
    for (int v : adj[u]) {
        if (v == p) continue;
        p_max_leaves[v] = p_max_leaves[u] + max_leaves[u] - max_leaves[v];
        dfs2(v, u);
        // cout << " u: " << u << " " << cur << endl;
        // cout << v << " " << deg[v] << " " << sz[v] << endl;
        if (deg[v] == 2 && sz[v] > 2) cur++;
    }
    ans = max(ans, cur);
}

void solve() {
    cin >> N;
    adj.assign(N, vector<int>());
    deg.assign(N, 0);
    for (int i = 0; i < N - 1; i++) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
        deg[u]++;
        deg[v]++;
    }
    max_leaves.resize(N);
    sz.resize(N);
    dfs1(0, -1); // calculate maximum number of leaves for each subtree
    ans = 0;
    p_max_leaves.assign(N, 0);
    dfs2(0, -1);
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

## E. Shuffle

### Solution 1:  maximum independent set, rerooting or dp on edges

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

