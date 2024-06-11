# Reroot Tree DP

This is dp on a tree algorithm that is handy for many calculations on tree.  It is basically you perform a first dfs through tree to compute a value for the subtrees.  Then you have a second dfs that goes through the tree and keeps track of the value from the parent and is like rerooting the tree and solving for when 1, 2, ..., n are the root of the tree

## Example

A very basic example for calculating distance from each node to every other nodes

track descendent and ancestor distance

```cpp
int N;
vector<vector<int>> adj;
vector<int> ans, sz, anc, desc;
int dfs1(int u, int p) {
    sz[u] = 1;
    for (int v : adj[u]) {
        if (v == p) continue;
        sz[u] += dfs1(v, u);
        desc[u] += desc[v] + sz[v];
    }
    return sz[u];
}
void dfs2(int u, int p) {
    ans[u] = anc[u] + desc[u];
    for (int v : adj[u]) {
        if (v == p) continue;
        anc[v] = anc[u] + (desc[u] - desc[v] - sz[v]) + (N - sz[v]);
        dfs2(v, u);
    }
}
class Solution {
public:
    vector<int> sumOfDistancesInTree(int n, vector<vector<int>>& edges) {
        N = n;
        adj.assign(n, vector<int>());
        for (auto vec : edges) {
            int u = vec[0], v = vec[1];
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
        sz.assign(n, 0);
        desc.assign(n, 0);
        dfs1(0, -1);
        ans.resize(n);
        anc.assign(n, 0);
        dfs2(0, -1);
        return ans;
    }
};
```


## Example of rerooting and also computing the maximum independent set on a tree

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
        dpp[v][0] = max(dpp[u][0] + dp[u][0] - max(dp[v][0], dp[v][1]), dpp[u][1] + dp[u][0] - max(dp[v][0], dp[v][1]));
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
```