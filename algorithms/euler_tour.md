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

This technique is often used with a Fenwick Tree, but it can also be applied with a Segment Tree or a Sparse Table when there are no updates. It could even be implemented using a Lazy Segment Tree. The key idea lies in how it builds the tin and tout arrays using the Euler Tour technique. This involves a depth-first search (DFS) that constructs these arrays, enabling efficient range queries and allowing you to compute the sum of all nodes in a subtree rooted at a given node u.

```cpp
int N, Q, timer;
vector<vector<int>> adj;
vector<int> values, tin, tout;

void dfs(int u, int p = -1) {
    tin[u] = timer++;
    for (int v : adj[u]) {
        if (v == p) continue;
        dfs(v, u);
    }
    tout[u] = timer - 1;
}

int neutral = 0;
struct FenwickTree {
    vector<int> nodes;
    
    void init(int n) {
        nodes.assign(n + 1, neutral);
    }

    void update(int idx, int val) {
        while (idx < (int)nodes.size()) {
            nodes[idx] += val;
            idx += (idx & -idx);
        }
    }

    int query(int left, int right) {
        return right >= left ? query(right) - query(left - 1) : 0;
    }

    int query(int idx) {
        int result = neutral;
        while (idx > 0) {
            result += nodes[idx];
            idx -= (idx & -idx);
        }
        return result;
    }
};

void solve() {
    cin >> N >> Q;
    adj.assign(N, vector<int>());
    values.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> values[i];
    }
    for (int i = 0; i < N - 1; i++) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    tin.resize(N);
    tout.resize(N);
    dfs(0);
    FenwickTree ft;
    ft.init(N);
    for (int i = 0; i < N; i++) {
        ft.update(tin[i] + 1, values[i]); // 1-indexed
    }
    while (Q--) {
        int type;
        cin >> type;
        if (type == 1) {
            int u, x;
            cin >> u >> x;
            u--;
            int delta = x - values[u];
            values[u] = x;
            ft.update(tin[u] + 1, delta);
        } else if (type == 2) {
            int u;
            cin >> u;
            u--;
            int ans = ft.query(tin[u] + 1, tout[u] + 1);
            cout << ans << endl;
        }
    }
}
```

## Euler Tour for Subtree Queries for prefix or suffix sums, 

This works for if you want to find the maximum value or minimum value of all nodes in or not in the subtree. or finding range sums, but it only works when the values are static, otherwise need something more advanced with segment trees etc.

```cpp
vector<int> values, tin, tout, timerToNode;
vector<vector<int>> adj;

void dfs(int u, int p = -1) {
    tin[u] = ++timer;
    timerToNode[timer] = u;
    for (int v : adj[u]) {
        if (v == p) continue;
        dfs(v, u);
    }
    tout[u] = timer;
}

vector<int> pmax(N + 2, 0), smax(N + 2, 0);
for (int i = 1; i <= N; ++i) {
    pmax[i] = max(pmax[i - 1], values[timerToNode[i]]);
}
for (int i = N; i > 0; --i) {
    smax[i] = max(smax[i + 1], values[timerToNode[i]]);
}
int64 ans = 0;
for (int u = 0; u < N; ++u) {
    int maxAroundSubtree = max(pmax[tin[u] - 1], smax[tout[u] + 1]);
    if (maxAroundSubtree > values[u] && (!ans || values[u] > values[ans])) {
        ans = u;
    }
}
```

## EULER TOUR FOR PATH QUERIES 

This one always increments the counter so that enter and exit counter will be differeent for each node. 

Allows to undo operation and get the sum along a path from root to a node in O(logn) time

Uses a fenwick tree to compute the sum along a path, from root you just do fenwick_tree.query(enter_counter[node]) get's sum from root to node.

This is 0-indexed, that is the nodes are numbered from 0 to n - 1

But the timer and counter for fenwick tree is 1-indexed, it starts at 1.

Example of how need to update fenwick tree for each enter/exit counter for a node that is being updated, wich a delta value (change in value from current value in array)
fenwick_tree.update(enter_counter, delta) # update the fenwick tree
fenwick_tree.update(exit_counter, -delta)

This particular example you push the weight of the edge into the node, and are querying the edges by querying the nodes. 


```cpp
template <typename T>
struct FenwickTree {
    vector<T> nodes;
    T neutral;

    FenwickTree() : neutral(T(0)) {}

    void init(int n, T neutral_val = T(0)) {
        neutral = neutral_val;
        nodes.assign(n + 1, neutral);
    }

    void update(int idx, T val) {
        while (idx < (int)nodes.size()) {
            nodes[idx] += val;
            idx += (idx & -idx);
        }
    }

    T query(int idx) {
        T result = neutral;
        while (idx > 0) {
            result += nodes[idx];
            idx -= (idx & -idx);
        }
        return result;
    }

    T query(int left, int right) {
        return right >= left ? query(right) - query(left - 1) : T(0);
    }
};
class Solution {
private:
    int N, timer;
    vector<int> start, end_, values;
    vector<vector<pair<int, int>>> adj;
    void dfs(int u, int p = -1) {
        for (auto &[v, w] : adj[u]) {
            if (v == p) continue;
            values[v] = w;
            dfs(v, u);
        }
    }
    void dfs1(int u, int p = -1) {
        start[u] = ++timer;
        for (auto &[v, w] : adj[u]) {
            if (v == p) continue;
            dfs1(v, u);
        }
        end_[u] = ++timer;
    }
public:
    vector<int> treeQueries(int n, vector<vector<int>>& edges, vector<vector<int>>& queries) {
        N = n;
        adj.assign(N, vector<pair<int, int>>());
        values.assign(N, 0);
        start.resize(N);
        end_.resize(N);
        for (const auto &edge : edges) {
            int u = edge[0], v = edge[1], w = edge[2];
            u--, v--;
            adj[u].emplace_back(v, w);
            adj[v].emplace_back(u, w);
        }
        dfs(0);
        dfs1(0);
        FenwickTree<int> ft;
        ft.init(timer);
        for (int i = 0; i < N; i++) {
            ft.update(start[i], values[i]);
            ft.update(end_[i], -values[i]);
        }

        vector<int> ans;
        for (const auto &query : queries) {
            int t = query[0];
            if (t == 1) {
                int u = query[1], v = query[2], nw = query[3];
                u--, v--;
                if (start[v] > start[u]) swap(u, v);
                int delta = nw - values[u];
                ft.update(start[u], delta);
                ft.update(end_[u], -delta);
                values[u] = nw;
            } else {
                int u = query[1];
                u--;
                ans.emplace_back(ft.query(start[u]));
            }
        }
        return ans;
    }
};
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