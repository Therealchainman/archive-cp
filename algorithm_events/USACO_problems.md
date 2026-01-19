# USACO Problems

The most interesting problems from USACO Contests

## USACO 2018 January Contest, Gold, MooTube

### Solution 1: offline queries, sorting, reverse iteration, dsu

You have a tree.
Given the fact of how relevance is the minimum edge weight along a path, and defines the relevance of a pair of nodes.  It makes the most sense to perform the queries in order from largest to smallest k.
Doing this you can use a dsu to keep track of the size of connected components.  The answer for any query is basically the size of the connected component, because these are all nodes where the relevance would be >= k.

```cpp
struct UnionFind {
    vector<int> parent, size;
    UnionFind(int n) {
        parent.resize(n);
        iota(parent.begin(),parent.end(),0);
        size.assign(n,1);
    }

    int find(int i) {
        if (i == parent[i]) {
            return i;
        }
        return parent[i] = find(parent[i]);
    }

    void unite(int i, int j) {
        i = find(i), j = find(j);
        if (i != j) {
            if (size[j] > size[i]) {
                swap(i, j);
            }
            size[i] += size[j];
            parent[j] = i;
        }
    }

    bool same(int i, int j) {
        return find(i) == find(j);
    }
};

struct Edge {
    int u, v, w;
    Edge(int u, int v, int w) : u(u), v(v), w(w) {};
    friend ostream& operator<<(ostream& os, const Edge &e) {
        return os << "Edge(" << e.u << "," << e.v << "," << e.w << ")";
    }
    bool operator<(const Edge &other) const {
        return w < other.w;
    }
};

struct Query {
    int k, i, u;
    Query(int k, int i, int u) : k(k), i(i), u(u) {};
    friend ostream& operator<<(ostream& os, const Query &q) {
        return os << "Query(" << q.k << "," << q.i << "," << q.u << ")";
    }
    bool operator<(const Query &other) const {
        return k > other.k;
    }
};

int N, M;

void solve() {
    freopen("mootube.in", "r", stdin);
	freopen("mootube.out", "w", stdout);
    cin >> N >> M;
    vector<Edge> edges;
    for (int i = 0; i < N - 1; ++i) {
        int u, v, w;
        cin >> u >> v >> w;
        u--, v--;
        edges.emplace_back(u, v, w);
    }
    sort(edges.begin(), edges.end());
    vector<int> ans(M, 0);
    vector<Query> queries;
    for (int i = 0; i < M; ++i) {
        int k, u;
        cin >> k >> u;
        u--;
        queries.emplace_back(k, i, u);
    }
    sort(queries.begin(), queries.end());
    UnionFind dsu(N);
    for (const auto &[k, i, u] : queries) {
        while (!edges.empty() && edges.back().w >= k) {
            Edge edge = edges.back();
            edges.pop_back();
            dsu.unite(edge.u, edge.v);
        }
        int root = dsu.find(u);
        ans[i] = dsu.size[root] - 1;
    }
    for (int x : ans) {
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

## USACO 2020 January Contest, Silver, Wormhole Sort

### Solution 1: dsu, offline, reverse, sorting

You need to try adding edges from largest weight to smallest and uniting these, until you can get that the ith element is in the same connected component as P[i]th element.  
If they are in the same component, that means there is a way to swap via the edges to get them in the correct location.
So you just want to use the least amount of edges to get it possible to have all nodes that need to be in the same component.

```cpp
struct UnionFind {
    vector<int> parent, size;
    UnionFind(int n) {
        parent.resize(n);
        iota(parent.begin(),parent.end(),0);
        size.assign(n,1);
    }

    int find(int i) {
        if (i == parent[i]) {
            return i;
        }
        return parent[i] = find(parent[i]);
    }

    void unite(int i, int j) {
        i = find(i), j = find(j);
        if (i != j) {
            if (size[j] > size[i]) {
                swap(i, j);
            }
            size[i] += size[j];
            parent[j] = i;
        }
    }

    bool same(int i, int j) {
        return find(i) == find(j);
    }
};

struct Edge {
    int u, v, w;
    Edge(int u, int v, int w) : u(u), v(v), w(w) {};
    friend ostream& operator<<(ostream& os, const Edge &e) {
        return os << "Edge(" << e.u << "," << e.v << "," << e.w << ")";
    }
    bool operator<(const Edge &other) const {
        return w < other.w;
    }
};

int N, M;
vector<int> P;

void solve() {
    freopen("wormsort.in", "r", stdin);
    freopen("wormsort.out", "w", stdout);
    cin >> N >> M;
    P.resize(N);
    for (int i = 0; i < N; ++i) {
        cin >> P[i];
        P[i]--;
    }
    vector<Edge> edges;
    for (int i = 0; i < M; ++i) {
        int u, v, w;
        cin >> u >> v >> w;
        u--, v--;
        edges.emplace_back(u, v, w);
    }
    sort(edges.begin(), edges.end());
    UnionFind dsu(N);
    int ans = -1;
    for (int i = 0, j = M - 1; i < N; ++i) {
        while (!edges.empty() && dsu.find(i) != dsu.find(P[i])) {
            auto [u, v, w] = edges.back();
            edges.pop_back();
            dsu.unite(u, v);
            ans = w;
        }
        if (!dsu.same(i, P[i])) {
            cout << -1 << endl;
            return;
        }
    }
    cout << ans << endl;
}

signed main() {
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    solve();
    return 0;
}
```

## USACO 2025 January Contest, Gold, Reachable Pairs

### Solution 1: 

```cpp

```

## 

### Solution 1: 

```cpp

```