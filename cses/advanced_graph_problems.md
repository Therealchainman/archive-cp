# Advanced Graph Problems

## Nearest Shops

### Solution 1:

```cpp
```

## Prüfer Code

### Solution 1:

```cpp
```

## Tree Traversals

### Solution 1:

```cpp
```

## Course Schedule II

### Solution 1:

```cpp
```

## Acyclic Graph Edges

### Solution 1:

There is a simple trick for this one, hint is total ordering of nodes can be constructed by sorting each node (u, v) in increasing order.

```cpp
```

## Strongly Connected Edges

### Solution 1:

Robbins theorem can be related to this problem.

undirected edge is a bridge if removing it disconnects the graph.

2-edge connected means graph is connected and has no bridges.

```cpp
```

## Even Outdegree Edges

### Solution 1:

```cpp
```

## Graph Girth

### Solution 1:

```cpp
```

## Fixed Length Walk Queries

### Solution 1: parity shortest path, bfs, undirected graph

```cpp
const int INF = (1LL << 31) - 1;
int N, M, Q;
vector<vector<int>> adj;
vector<vector<int>> dist[2];

void bfs(int src) {
    queue<int> q;
    dist[0][src][src] = 0;
    q.emplace(src);
    int p = 0;
    while (!q.empty()) {
        int sz = q.size();
        for (int i = 0; i < sz; i++) {
            int u = q.front();
            q.pop();
            for (int v : adj[u]) {
                if (dist[p ^ 1][src][v] != INF) continue;
                dist[p ^ 1][src][v] = dist[p][src][u] + 1;
                q.emplace(v);
            }
        }
        p ^= 1;
    }
}

void solve() {
    cin >> N >> M >> Q;
    adj.assign(N, vector<int>());
    for (int i = 0; i < M; ++i) {
        int u, v;
        cin >> u >> v;
        --u, --v;
        adj[u].emplace_back(v);
        adj[v].emplace_back(u);
    }
    for (int i = 0; i < 2; i++) {
        dist[i].assign(N, vector<int>(N, INF));
    }
    for (int i = 0; i < N; i++) {
        bfs(i);
    }
    while (Q--) {
        int u, v, x;
        cin >> u >> v >> x;
        --u, --v;
        if (dist[x % 2][u][v] <= x) {
            cout << "YES" << endl;
        } else {
            cout << "NO" << endl;
        }
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Transfer Speeds Sum

### Solution 1:

```cpp
```

## MST Edge Check

### Solution 1: minimum spanning tree, union find, weighted undirected graph, map

The key to solving this problem is to realize the cut property of MSTs. 

The core idea relies on the cut property of MSTs:

For any cut (a partition of the vertices into two disjoint sets), the minimum weight edge crossing the cut must be part of some MST.

This algorithm leverages that fact efficiently by grouping edges with the same weight and checking whether they connect different components before they are merged in the union-find structure.

```cpp
int N, M;

struct Edge {
    int u, v, i;
    Edge() {}
    Edge(int u, int v, int i) : u(u), v(v), i(i) {}
};

struct UnionFind {
    vector<int> parents, size;
    UnionFind(int n) {
        parents.resize(n);
        iota(parents.begin(),parents.end(),0);
        size.assign(n,1);
    }

    int find(int i) {
        if (i==parents[i]) {
            return i;
        }
        return parents[i]=find(parents[i]);
    }

    bool same(int i, int j) {
        i = find(i), j = find(j);
        if (i!=j) {
            if (size[j]>size[i]) {
                swap(i,j);
            }
            size[i]+=size[j];
            parents[j]=i;
            return false;
        }
        return true;
    }
};

void solve() {
    cin >> N >> M;
    UnionFind dsu(N);
    map<int, vector<Edge>> edgesGroup;
    for (int i = 0; i < M; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        u--; v--;
        edgesGroup[w].emplace_back(u, v, i);
    }
    vector<bool> ans(M, false);
    for (auto [w, edges] : edgesGroup) {
        for (const Edge &edge : edges) {
            if (dsu.find(edge.u) != dsu.find(edge.v)) {
                ans[edge.i] = true;
            }
        }
        for (const Edge &edge : edges) {
            dsu.same(edge.u, edge.v);
        }
    }
    for (bool x : ans) {
        if (x) cout << "YES" << endl;
        else cout << "NO" << endl;
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## MST Edge Set Check

### Solution 1:

```cpp
```

## MST Edge Cost

### Solution 1: binary lifting, max path query on edges of tree, lca, kruskal's algorithm, union find, weighted undirectd graph

```cpp
const int INF = numeric_limits<int>::max();
int N, M;

struct UnionFind {
    vector<int> parents, size;
    UnionFind(int n) {
        parents.resize(n);
        iota(parents.begin(),parents.end(),0);
        size.assign(n,1);
    }

    int find(int i) {
        if (i==parents[i]) {
            return i;
        }
        return parents[i]=find(parents[i]);
    }

    bool same(int i, int j) {
        i = find(i), j = find(j);
        if (i!=j) {
            if (size[j]>size[i]) {
                swap(i,j);
            }
            size[i]+=size[j];
            parents[j]=i;
            return false;
        }
        return true;
    }
};

struct Tree {
    int N, LOG;
    vector<vector<pair<int,int>>> adj;
    vector<int> depth, parent, dist;
    vector<vector<int>> up, st;

    Tree(int n) : N(n) {
        LOG = 20;
        adj.assign(N, vector<pair<int, int>>());
        depth.assign(N, 0);
        parent.assign(N, -1);
        dist.assign(N, 0);
        up.assign(LOG, vector<int>(N, -1));
        st.assign(LOG, vector<int>(N, -INF));

    }
    void addEdge(int u, int v, int w = 1) {
        adj[u].emplace_back(v, w);
        adj[v].emplace_back(u, w);
    }
    void preprocess(int root = 0) {
        dfs(root);
        buildLiftingTable();
    }
    int query(int u, int v) const {
        if (depth[u] < depth[v]) swap(u, v);
        int ans = -INF;
        // Bring u up to the same depth as v
        int k = depth[u] - depth[v];
        for (int i = 0; i < LOG && u != -1; i++) {
            if ((k >> i) & 1) {
                ans = max(ans, st[i][u]);
                u = up[i][u];
            }
        }
        if (u == v) {
            return ans;
        }
        // Binary lift both
        for (int i = LOG - 1; i >= 0; i--) {
            if (up[i][u] != up[i][v]) {
                ans = max(ans, st[i][u]);
                ans = max(ans, st[i][v]);
                u = up[i][u];
                v = up[i][v];
            }
        }
        ans = max(ans, st[0][u]);
        ans = max(ans, st[0][v]);
        return ans;
    }
private:
    void dfs(int u, int p = -1) {
        parent[u] = p;
        up[0][u] = p;
        if (p == -1) st[0][u] = 0;
        for (auto &[v, w] : adj[u]) {
            if (v == p) {
                st[0][u] = w;
                continue;
            }
            depth[v] = depth[u] + 1;
            dist[v] = dist[u] + w;
            dfs(v, u);
        }
    }
    void buildLiftingTable() {
        for (int i = 1; i < LOG; i++) {
            for (int j = 0; j < N; j++) {
                if (up[i - 1][j] == -1) continue;
                up[i][j] = up[i - 1][up[i - 1][j]];
                st[i][j] = max(st[i - 1][j], st[i - 1][up[i - 1][j]]);
            }
        }
    }
};

struct Edge {
    int u, v, w, i;
    Edge() {}
    Edge(int u, int v, int w, int i) : u(u), v(v), w(w), i(i) {}
    bool operator<(const Edge &other) const {
        return w < other.w;
    }
};

void solve() {
    cin >> N >> M;
    vector<Edge> edges;
    for (int i = 0; i < M; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        u--; v--;
        edges.emplace_back(u, v, w, i);
    }
    sort(edges.begin(), edges.end());
    Tree tree(N);
    UnionFind dsu(N);
    int64 minCost = 0;
    for (const Edge &edge : edges) {
        if (!dsu.same(edge.u, edge.v)) {
            minCost += edge.w;
            tree.addEdge(edge.u, edge.v, edge.w);
        }
    }
    tree.preprocess();
    vector<int64> ans(M);
    for (const Edge &edge : edges) {
        int maxEdge = tree.query(edge.u, edge.v);
        ans[edge.i] = minCost + edge.w - maxEdge;
    }
    for (int64 x : ans) {
        cout << x << endl;
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Network Breakdown

### Solution 1:

```cpp
```

## Tree Coin Collecting I

### Solution 1: binary lifting, bfs, dfs, undirected graph, lca, minimum value on path query, distance on path query

```cpp
const int INF = (1LL << 31) - 1;
int N, Q;

struct Tree {
    int N, LOG;
    vector<vector<pair<int,int>>> adj;
    vector<int> depth, parent, dist, coins;
    vector<vector<int>> up, st;

    Tree(int n) : N(n) {
        LOG = 20;
        adj.assign(N, vector<pair<int, int>>());
        depth.assign(N, 0);
        parent.assign(N, -1);
        dist.assign(N, 0);
        coins.assign(N, INF);
        up.assign(LOG, vector<int>(N, -1));
        st.assign(LOG, vector<int>(N, INF));

    }
    void addEdge(int u, int v, int w = 1) {
        adj[u].emplace_back(v, w);
        adj[v].emplace_back(u, w);
    }
    void preprocess(int root = 0) {
        processCoins();
        dfs(root);
        buildLiftingTable();
    }
    int kthAncestor(int u, int k) const {
        for (int i = 0; i < LOG && u != -1; i++) {
            if ((k >> i) & 1) {
                u = up[i][u];
            }
        }
        return u;
    }
    int lca(int u, int v) const {
        if (depth[u] < depth[v]) swap(u, v);
        // Bring u up to the same depth as v
        u = kthAncestor(u, depth[u] - depth[v]);
        if (u == v) return u;
        // Binary lift both
        for (int i = LOG - 1; i >= 0; i--) {
            if (up[i][u] != up[i][v]) {
                u = up[i][u];
                v = up[i][v];
            }
        }
        // Now parents are equal
        return parent[u];
    }
    int query(int u, int v) const {
        if (depth[u] < depth[v]) swap(u, v);
        int ans = INF;
        // Bring u up to the same depth as v
        int k = depth[u] - depth[v];
        for (int i = 0; i < LOG && u != -1; i++) {
            if ((k >> i) & 1) {
                ans = min(ans, st[i][u]);
                u = up[i][u];
            }
        }
        if (u == v) {
            ans = min(ans, st[0][u]);
            return ans;
        }
        // Binary lift both
        for (int i = LOG - 1; i >= 0; i--) {
            if (up[i][u] != up[i][v]) {
                ans = min(ans, st[i][u]);
                ans = min(ans, st[i][v]);
                u = up[i][u];
                v = up[i][v];
            }
        }
        ans = min(ans, st[1][u]);
        ans = min(ans, st[1][v]);
        return ans;
    }
    int distance(int u, int v) const {
        int a = lca(u, v);
        return dist[u] + dist[v] - 2 * dist[a];
    }
private:
    void processCoins() {
        queue<pair<int, int>> q;
        for (int i = 0; i < N; i++) {
            if (!coins[i]) q.emplace(i, 0);
        }
        while (!q.empty()) {
            auto [u, d] = q.front();
            q.pop();
            for (auto [v, w] : adj[u]) {
                if (coins[v] != INF) continue;
                coins[v] = d + w;
                q.emplace(v, d + w);
            }
        }
    }
    void dfs(int u, int p = -1) {
        parent[u] = p;
        up[0][u] = p;
        st[0][u] = coins[u];
        for (auto &[v, w] : adj[u]) {
            if (v == p) continue;
            depth[v] = depth[u] + 1;
            dist[v] = dist[u] + w;
            dfs(v, u);
        }
    }
    void buildLiftingTable() {
        for (int i = 1; i < LOG; i++) {
            for (int j = 0; j < N; j++) {
                if (up[i - 1][j] == -1) continue;
                up[i][j] = up[i - 1][up[i - 1][j]];
                st[i][j] = min(st[i - 1][j], st[i - 1][up[i - 1][j]]);
            }
        }
    }
};

void solve() {
    cin >> N >> Q;
    Tree tree(N);
    for (int i = 0; i < N; i++) {
        int x;
        cin >> x;
        if (x == 1) {
            tree.coins[i] = 0;
        }
    }
    for (int i = 0; i < N - 1; i++) {
        int u, v;
        cin >> u >> v;
        u--; v--;
        tree.addEdge(u, v);
    }
    tree.preprocess();
    while (Q--) {
        int u, v;
        cin >> u >> v;
        u--; v--;
        int minCoinDistance = tree.query(u, v);
        int dist = tree.distance(u, v);
        int ans = dist + 2 * minCoinDistance;
        cout << ans << endl;
    }
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Tree Coin Collecting II

### Solution 1:

```cpp
```

## Tree Isomorphism I

### Solution 1:

```cpp
```

## Tree Isomorphism II

### Solution 1:

```cpp
```

## Flight Route Requests

### Solution 1:

```cpp
```

## Critical Cities

### Solution 1:

```cpp
```

## Visiting Cities

### Solution 1:

This is a crazy hard problem, involving multiple dijkstra and counting paths. 

```cpp
```

## Graph Coloring

### Solution 1:

```cpp
```

## Bus Companies

### Solution 1: Dijkstra, directed graph, shortest path, bipartite graph

The key idea is that each company or “hub” lets you travel between any pair of cities it connects to. If you draw this, you get a star-shaped graph: the company node in the center, connected to k cities. The trick is realizing that once you’ve paid the cost to access the hub, you can move freely (or cheaply) between any of the cities in that star.

To model this, use a directed graph:
- From each city → hub, add an edge with weight equal to the company’s ticket cost (you pay once to “enter” the hub).
- From the hub → each city, add an edge with weight 0 (you can now go to any connected city for free).

Why does this work with Dijkstra?

Whenever you see a structure where:
(a) A service covers a group of nodes,
(b) There’s a fixed cost to enter the service, and
(c) Travel within that group is free or uniformly cheap,
you can model it by adding a node for the service, connecting it to the group with directed edges as above. Then, standard shortest-path algorithms like Dijkstra will work perfectly on the augmented graph.

```cpp
const int64 INF = numeric_limits<int64>::max();
int N, M;
vector<int64> costs;
vector<int64> dist;
vector<vector<pair<int, int64>>> adj;

void solve() {
    cin >> N >> M;
    costs.resize(M);
    dist.assign(N + M, INF);
    dist[0] = 0;
    adj.assign(N + M, vector<pair<int, int64>>());
    for (int i = 0; i < M; i++) {
        cin >> costs[i];
    }
    for (int i = 0; i < M; i++) {
        int k;
        cin >> k;
        int company = N + i;
        for (int j = 0; j < k; j++) {
            int city;
            cin >> city;
            city--;
            adj[city].emplace_back(company, costs[i]);
            adj[company].emplace_back(city, 0);
        }
    }
    priority_queue<pair<int64, int>, vector<pair<int64, int>>, greater<pair<int64, int>>> minheap;
    minheap.emplace(0, 0); // start city 0
    while (!minheap.empty()) {
        auto [c, u] = minheap.top();
        minheap.pop();
        for (auto [v, w] : adj[u]) {
            if (c + w < dist[v]) {
                dist[v] = c + w;
                minheap.emplace(c + w, v);
            }
        }
    }
    for (int i = 0; i < N; i++) {
        cout << dist[i] << " ";
    }
    cout << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## Split into Two Paths

### Solution 1:

```cpp
```

## Network Renovation

### Solution 1:

```cpp
```

## Forbidden Cities

### Solution 1:

```cpp
```

## Creating Offices

### Solution 1:

```cpp
```

## New Flight Routes

### Solution 1:

```cpp
```
