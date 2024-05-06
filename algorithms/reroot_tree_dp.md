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