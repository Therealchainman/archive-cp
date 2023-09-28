# Bridge Tree

Using Tarjans bridge finding algorithm which is a recursive depth first search algorithm that tracks minimum ancestors and if there are back edges.  This algorithm can find the bridge edges between 2-edge connected components or blocks.

The block dfs will assign block id to each of the nodes to create the bridge tree that now has nodes that are blocks and edges that are bridge edges.

constructs the bridge_adj which represents the new tree.  The original undirected graph does not need be a tree of course. 

```cpp
vector<vector<pair<int,int>>> adj;
vector<vector<int>> bridge_adj;
vector<pair<int,int>> edges;
vector<bool> bridge_edge;
vector<int> tin, low, colors, blocks;
int n, m, timer, block_id;

const int inf = LLONG_MAX;

void dfs(int u, int p = -1) {
    colors[u] = 1;
    tin[u] = low[u] = timer++;
    for (auto &[v, i] : adj[u]) {
        if (v == p) continue;
        if (colors[v] != 0) { // back edge
            low[u] = min(low[u], tin[v]);
        } else {
            dfs(v, u);
            low[u] = min(low[u], low[v]);
            if (low[v] > tin[u]) bridge_edge[i] = true;
        }
    }
}

void block_dfs(int u, int p = -1) {
    colors[u] = 1;
    blocks[u] = block_id; // assigns block id to each node id
    for (auto &[v, i]: adj[u]) {
        if (v == p) continue;
        if (bridge_edge[i]) continue; // only dfs on block
        if (colors[v] != 0) continue;
        block_dfs(v, u);
    }
}

void bridge_tree() {
    timer = 0;
    colors.assign(n, 0);
    tin.assign(n, -1);
    low.assign(n, -1);
    bridge_edge.assign(m, false);
    dfs(0, -1); // Assume the graph is a one connected component
    block_id = 0;
    colors.assign(n, 0);
    blocks.assign(n, -1);
    for (int i = 0; i < n; i++) {
        if (colors[i] == 0) {
            block_dfs(i);
            block_id++;
        }
    }
    bridge_adj.assign(block_id, vector<int>{});
    // bridges will be edges in the bridge tree and blocks will be nodes
    for (int i = 0; i < m; i++) {
        if (bridge_edge[i]) {
            int u, v;
            tie(u, v) = edges[i];
            bridge_adj[blocks[u]].push_back(blocks[v]);
            bridge_adj[blocks[v]].push_back(blocks[u]);
        }
    }
}

int solve() {
    n = read(), m = read();
    adj.assign(n, vector<pair<int,int>>{});
    edges.resize(m);
    for (int i = 0; i < m; i++) {
        int u = read(), v = read();
        u--;
        v--;
        adj[u].push_back({v, i});
        adj[v].push_back({u, i});
        edges[i] = {u, v};
    }
    int res = 0;
    bridge_tree();
}
```