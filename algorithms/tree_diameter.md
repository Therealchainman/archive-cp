# Tree Diameter

This is the longest distance between two nodes in an undirected weighted or unweighted tree normally.

Another way to think of it the longest path graph int he tree is the diameter of the tree

## weighted undirected tree, twice DFS algorithm

This is the algorithm that is used to find the diameter of a tree.  
The first DFS is used to find the farthest node from any random starting node (although I pick node labeled as 0).  
The second DFS is finding the distance from the start node found from the first DFS to every other node.
And updating the diameter whenever you find a node that is farthest from the start node. 

Can use this for unweighted tree as well, easily, where the weight w = 1. 

```cpp
int N, diam, start_node, best;
vector<vector<pair<int, int>>> adj;
vector<int> dist;

// returns the farthest away node from u (random node)
void dfs1(int u, int p) {
    if (dist[u] > best) {
        best = dist[u];
        start_node = u;
    }
    for (auto [v, w]: adj[u]) {
        if (v == p) continue;
        dist[v] = dist[u] + w;
        dfs1(v, u);
    }
}

// Calculates the distance from the leaf node to every other node
void dfs2(int u, int p) {
    diam = max(diam, dist[u]);
    for (auto [v, w]: adj[u]) {
        if (v == p) continue;
        dist[v] = dist[u] + w;
        dfs2(v, u);
    }
}

void solve() {
    cin >> N;
    adj.assign(N, vector<pair<int, int>>());
    int total = 0;
    for (int i = 0; i < N - 1; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        u--; v--;
        adj[u].emplace_back(v, w);
        adj[v].emplace_back(u, w);
        total += 2 * w;
    }
    dist.assign(N, 0);
    best = 0;
    dfs1(0, -1);
    dist.assign(N, 0);
    dfs2(start_node, -1);
    cout << total - diam << endl;
}
```

## reroot dp to compute tree diameter

```cpp
int N, diam;
vector<vector<pair<int, int>>> adj;
vector<int> mx1, mx2, node1, node2, par;

// mx1[u] = maximum distance from u to any other node and mx2 (second max)
void dfs1(int u, int p) {
    for (auto [v, w]: adj[u]) {
        if (v == p) continue;
        dfs1(v, u);
        if (mx1[v] + w > mx1[u]) {
            mx2[u] = mx1[u];
            mx1[u] = mx1[v] + w;
            node2[u] = node1[u];
            node1[u] = v;
        } else if (mx1[v] + w > mx2[u]) {
            mx2[u] = mx1[v] + w;
            node2[u] = v;
        }
    }
}

// Calculates the diameter
void dfs2(int u, int p) {
    diam = max(diam, mx1[u] + par[u]);
    for (auto [v, w]: adj[u]) {
        if (v == p) continue;
        par[v] = par[u] + w;
        if (v != node1[u]) par[v] = max(par[v], mx1[u] + w);
        else par[v] = max(par[v], mx2[u] + w);
        dfs2(v, u);
    }
}

void solve() {
    cin >> N;
    adj.assign(N, vector<pair<int, int>>());
    int total = 0;
    for (int i = 0; i < N - 1; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        u--; v--;
        adj[u].emplace_back(v, w);
        adj[v].emplace_back(u, w);
        total += 2 * w;
    }
    mx1.assign(N, 0);
    mx2.assign(N, 0);
    node1.assign(N, -1);
    node2.assign(N, -1);
    par.assign(N, 0);
    dfs1(0, -1);
    dfs2(0, -1);
    cout << total - diam << endl;
}

signed main() {
    solve();
    return 0;
}
```