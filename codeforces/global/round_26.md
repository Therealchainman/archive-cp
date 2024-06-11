# Codeforces Global Round 26

##

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```

## E. Shuffle

### Solution 1:  maximum independent set, dp on tree, reroot dp

```cpp
int N, ans;
vector<vector<int>> adj, dp, dpp;
vector<int> deg;

void dfs1(int u, int p) {
    dp[u][1] = 1;
    for (int v : adj[u]) {
        if (v == p) continue;
        dfs1(v, u);
        dp[u][0] += max(dp[v][0], dp[v][1]);
        dp[u][1] += dp[v][0];
    }
}

void dfs2(int u, int p) {
    int cand;
    if (deg[u] > 1) {
        cand = max(dpp[u][0] + max(dp[u][0], dp[u][1]), dpp[u][1] + dp[u][0]);
    } else {
        cand = max(dpp[u][0] + max(dp[u][0] + 1, dp[u][1]), dpp[u][1] + dp[u][0] + 1);
    }
    ans = max(ans, cand);
    for (int v : adj[u]) {
        if (v == p) continue;
        dpp[v][0] = max(dpp[u][0], dpp[u][1]) + dp[u][0] - max(dp[v][0], dp[v][1]);
        dpp[v][1] = dpp[u][0] + dp[u][1] - dp[v][0];
        dfs2(v, u);
    }
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
    dp.assign(N, vector<int>(2, 0));
    dpp.assign(N, vector<int>(2, 0));
    ans = 0;
    dfs1(0, -1);
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

## F. Reconstruction

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

