# EULER TOUR TECHNIQUE

## DEFINITIONS

Euler Tour Technique to a random tree

relabel or index of the tree nodes


Necessary and sufficient conditions

An undirected graph has a closed Euler tour if and only if it is connected and each vertex has an even degree.

An undirected graph has an open Euler tour (Euler path) if it is connected, and each vertex, except for exactly two vertices, has an even degree. The two vertices of odd degree have to be 
the endpoints of the tour.

A directed graph has a closed Euler tour if and only if it is strongly connected and the in-degree of each vertex is equal to its out-degree.

Similarly, a directed graph has an open Euler tour (Euler path) if and only if for each vertex the difference between its in-degree and out-degree is 0, except for two vertices, 
where one has difference +1 (the start of the tour) and the other has difference -1 (the end of the tour) and, if you add an edge from the end to the start, the graph is strongly connected.

Definition 13.1.1.  A walk is closed if it begins and ends with the same vertex.
A trail is a walk in which no two vertices appear consecutively (in either order) more than once. (That is, no edge is used more than once.)

A tour is a closed trail.

An Euler trail is a trail in which every pair of adjacent vertices appear consecutively. (That is, every edge is used exactly once.)

An Euler tour is a closed Euler trail.

## EULER TOUR FOR SUBTREE QUERIES

Euler tour technique for subtree queries for a tree with root node 1
Note this is 1-indexed, that is the nodes are numbered from 1 to n

```py
# EULER TOUR TECHNIQUE
start, end = [0] * n, [0] * n
timer = 0
def dfs(node, parent):
    nonlocal timer
    start[node] = timer
    timer += 1
    for nei in adj_list[node]:
        if nei == parent: continue
        dfs(nei, node)
    end[node] = timer
dfs(0, -1)
bit = FenwickTree(timer + 1)
for i, val in enumerate(values):
    bit.update(start[i] + 1, val)
for _ in range(q):
    queries = list(map(int, input().split()))
    if queries[0] == 1:
        u, s = queries[1:]
        u -= 1
        delta = s - values[u]
        bit.update(start[u] + 1, delta)
        values[u] = s
    else:
        u = queries[1] - 1
        res = bit.query(end[u]) - bit.query(start[u])
        print(res)
```

Implemented in C++

```cpp
class EulerTour {
public:
    int num_nodes;
    vector<vector<int>> edges;
    vector<vector<int>> adj_list;
    int root_node;
    vector<int> enter_counter, exit_counter;
    int counter;

    EulerTour(int n, vector<vector<int>>& e) {
        num_nodes = n;
        edges = e;
        adj_list.resize(num_nodes + 1);
        root_node = 1;
        enter_counter.resize(num_nodes + 1);
        exit_counter.resize(num_nodes + 1);
        counter = 1;
        build_adj_list();
        euler_tour(root_node, -1);
    }

    void build_adj_list() {
        for (auto edge : edges) {
            int u = edge[0], v = edge[1];
            adj_list[u].push_back(v);
            adj_list[v].push_back(u);
        }
    }

    void euler_tour(int node, int parent_node) {
        enter_counter[node] = counter;
        counter++;
        for (auto child_node : adj_list[node]) {
            if (child_node != parent_node) {
                euler_tour(child_node, node);
            }
        }
        exit_counter[node] = counter - 1;
    }
};
```

## EULER TOUR FOR PATH QUERIES 

This one always increments the counter so that enter and exit counter will be differeent for each node. 

Allows to undo operation and get the sum along a path from root to a node in O(logn) time

Uses a fenwick tree to compute the sum along a path, from root you just do fenwick_tree.query(enter_counter[node]) get's sum from root to node.

This is 1-indexed, that is the nodes are numbered from 1 to n

Example of how need to update fenwick tree for each enter/exit counter for a node that is being updated, wich a delta value (change in value from current value in array)
fenwick_tree.update(enter_counter, delta) # update the fenwick tree
fenwick_tree.update(exit_counter, -delta)


```py
# EULER TOUR TECHNIQUE FOR PATH QUERIES
start, end = [0] * n, [0] * n
timer = 1
def dfs(node, parent):
    nonlocal timer
    start[node] = timer
    timer += 1
    for nei in adj_list[node]:
        if nei == parent: continue
        dfs(nei, node)
    timer += 1
    end[node] = timer
dfs(0, -1)
bit = FenwickTree(timer + 1)
for i, val in enumerate(values):
    bit.update(start[i], val)
    bit.update(end[i], -val)
for _ in range(q):
    queries = list(map(int, input().split()))
    if queries[0] == 1:
        u, s = queries[1:]
        u -= 1
        delta = s - values[u]
        bit.update(start[u], delta)
        bit.update(end[u], -delta)
        values[u] = s
    else:
        u = queries[1] - 1
        res = bit.query(start[u])
        print(res)
```

## Euler Tour and RMQ to find LCA of nodes u and v in tree

preprocess time O(nlogn) and lca query time O(1)

```cpp
const int MAXN = 2e5 + 5, LOG = 20;
int N, Q;
int par[MAXN];
vector<vector<int>> adj;
int last[MAXN];
int depth[2 * MAXN];
vector<int> tour;

void dfs(int u, int p, int dep) {
    depth[tour.size()] = dep;
    tour.push_back(u);
    for (int v : adj[u]) {
        if (v != p) {
            dfs(v, u, dep + 1);
            depth[tour.size()] = dep;
            tour.push_back(u);
        }
    }
    last[u] = tour.size() - 1;
}

struct RMQ {
    vector<vector<int>> st;
    void init() {
        int n = tour.size();
        st.assign(n, vector<int>(LOG));
        for (int i = 0; i < n; i++) {
            st[i][0] = i;
        }
        for (int j = 1; j < LOG; j++) {
            for (int i = 0; i + (1 << j) <= n; i++) {
                int x = st[i][j - 1];
                int y = st[i + (1 << (j - 1))][j - 1];
                st[i][j] = (depth[x] < depth[y] ? x : y);
            }
        }
    }

    int query(int a, int b) {
        int l = min(a, b), r = max(a, b);
        int j = 31 - __builtin_clz(r - l + 1);
        int x = st[l][j];
        int y = st[r - (1 << j) + 1][j];
        return (depth[x] < depth[y] ? x : y);
    }
};
signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    cin >> N >> Q;
    adj.assign(N + 1, vector<int>());
    for (int i = 2; i <= N; i++) {
        cin >> par[i];
        adj[par[i]].push_back(i);
        adj[i].push_back(par[i]);
    }
    memset(depth, 0, sizeof(depth));
    tour.clear();
    dfs(1, 0, 0);
    RMQ rmq;
    rmq.init();
    while (Q--) {
        int u, v;
        cin >> u >> v;
        int lca = tour[rmq.query(last[u], last[v])];
        cout << lca << endl;
    }
    return 0;
}
```