# Lowest Common Ancestor

## Brute force solution

This was useful if you need to keep track of something as you are moving up to the lca of nodes u and v.

this is marking them as you move up the lca

This dfs can be used to construct the parent array and depth array

```cpp
void dfs(int u, int p = -1) {
    par[u] = p;
    for (int v : adj[u]) {
        if (v == p) continue;
        dep[v] = dep[u] + 1;
        dfs(v, u);
    }
}
```

Then it is used to climb up the tree to the LCA of nodes u and v. 

```cpp
u--, v--;
int ans = INF;
while (u != v) {
    if (dep[u] < dep[v]) swap(u, v);
    if (marked[A[u]] == i) {
        ans = 0;
        break;
    } else {
        marked[A[u]] = i;
    }
    u = par[u];
}
```