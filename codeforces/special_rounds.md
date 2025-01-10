# Special Codeforce Rounds

# Good Bye 2024: 2025 is NEAR

## D. Refined Product Optimality

### Solution 1:  binary search, sorting

1. don't overthink, this one is not that hard as it looks

```cpp
const int MOD = 998244353;
int N, Q;
vector<int> A, B, C, D;

int inv(int i, int m) {
  return i <= 1 ? i : m - (m/i) * inv(m % i, m) % m;
}

void solve() {
    cin >> N >> Q;
    A.assign(N, 0);
    B.assign(N, 0);
    C.assign(N, 0);
    D.assign(N, 0);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
        C[i] = A[i];
    }
    for (int i = 0; i < N; i++) {
        cin >> B[i];
        D[i] = B[i];
    }
    sort(C.begin(), C.end());
    sort(D.begin(), D.end());
    int ans = 1;
    for (int i = 0; i < N; i++) {
        ans = ans * min(C[i], D[i]) % MOD;
    }
    cout << ans << " ";
    while (Q--) {
        int x, y;
        cin >> x >> y;
        x--; y--;
        if (x == 0) {
            int i = upper_bound(C.begin(), C.end(), A[y]) - C.begin() - 1;
            ans = ans * inv(min(C[i], D[i]), MOD) % MOD;
            A[y]++; C[i]++;
            ans = ans * min(C[i], D[i]) % MOD;
        } else {
            int i = upper_bound(D.begin(), D.end(), B[y]) - D.begin() - 1;
            ans = ans * inv(min(C[i], D[i]), MOD) % MOD;
            B[y]++; D[i]++;
            ans = ans * min(C[i], D[i]) % MOD;
        }
        cout << ans << " ";
    }
    cout << endl;
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

## E. Resourceful Caterpillar Sequence

### Solution 1:  tree, dp on tree, reroot tree, leaf, degrees

1. There is the case when q is on a leaf node, that is precalculated.
1. Then it needs to calculate with dp the case when q is distance of 2 from a leaf node.  

```cpp
int N, ans;
vector<vector<int>> adj;
vector<int> deg, values;

void dfs(int u, int p = -1) {
    bool nonLeafChild = true;
    for (int v : adj[u]) {
        if (deg[v] == 1) nonLeafChild &= false;
        if (v == p) continue;
        dfs(v, u);
        values[u] += values[v];
    }
    if (deg[u] == 1) return;
    values[u] += nonLeafChild;
}

void dfs2(int u, int p = -1, int pval = 0) {
    int cntNonLeaf = 0;
    bool hasLeaf = false;
    for (int v : adj[u]) {
        if (deg[v] == 1) hasLeaf = true;
        else cntNonLeaf++;
        if (v == p) continue;
        int nval = pval + values[u] - values[v];
        dfs2(v, u, nval);
    }
    if (deg[u] == 1) return;
    if (hasLeaf && cntNonLeaf > 0) ans += (cntNonLeaf - 1) * (values[u] + pval);
}

void solve() {
    cin >> N;
    adj.assign(N, vector<int>());
    deg.assign(N, 0);
    values.assign(N, 0);
    for (int i = 0; i < N - 1; i++) {
        int u, v;
        cin >> u >> v;
        u--, v--;
        adj[u].push_back(v);
        adj[v].push_back(u);
        deg[u]++;
        deg[v]++;
    }
    int leaf = 0;
    for (int i = 0; i < N; i++) {
        if (deg[i] == 1) leaf++;
    }
    ans = leaf * (N - leaf);
    dfs(0);
    dfs2(0);
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

# Hello 2025

## D. Gifts Order

### Solution 1:  segment tree, math

1. rearrange the terms to derive (ar-r)-(al-l) and (al+l)-(ar+r) to figure out how segment tree helps here. 

```cpp
const int INF = 1e12;
int N, Q;
vector<int> A;

struct Node {
    int max1, min1, max2, min2, ans;
    void output() {
        cout << max1 << " " << min1 << " " << max2 << " " << min2 << " " << ans << endl;
    }
};

struct SegmentTree {
    int size;
    vector<Node> nodes;
    SegmentTree(int n) {
        size = 1;
        while (size < n) size *= 2;
        nodes.assign(size * 2, {-INF, INF, -INF, INF, 0});
    }

    Node func(Node left, Node right) {
        Node res;
        res.max1 = max(left.max1, right.max1);
        res.max2 = max(left.max2, right.max2);
        res.min1 = min(left.min1, right.min1);
        res.min2 = min(left.min2, right.min2);
        res.ans = max({
            left.ans,
            right.ans,
            right.max1 - left.min1,
            left.max2 - right.min2
        });
        return res;
    }

    void ascend(int segment_idx) {
        while (segment_idx > 0) {
            int left_segment_idx = 2 * segment_idx, right_segment_idx = 2 * segment_idx + 1;
            nodes[segment_idx] = func(nodes[left_segment_idx], nodes[right_segment_idx]);
            segment_idx >>= 1;
        }
    }

    void update(int segment_idx, int val) {
        int idx = segment_idx;
        segment_idx += size;
        nodes[segment_idx] = {val - idx, val - idx, val + idx, val + idx, 0};
        segment_idx >>= 1;
        ascend(segment_idx);
    }
};

void solve() {
    cin >> N >> Q;
    A.resize(N);
    for (int i = 0; i < N; i++) {
        cin >> A[i];
    }
    SegmentTree seg(N);
    for (int i = 0; i < N; i++) {
        seg.update(i, A[i]);
    }
    cout << seg.nodes[1].ans << endl;
    while (Q--) {
        int p, x;
        cin >> p >> x;
        p--;
        seg.update(p, x);
        cout << seg.nodes[1].ans << endl;
    }
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

## E1. Another Exercise on Graphs (Easy Version)

### Solution 1:  dynamic programming on graph, floyd warshall, weighted undirected graph, shortest paths, binary search

```cpp
struct Edge {
    int u, v, w;
    Edge() {}
    Edge(int u, int v, int w) : u(u), v(v), w(w) {}
    bool operator<(const Edge &other) const {
        return w < other.w;
    }
};

const int INF = 1e9;
int N, M, Q;
vector<vector<vector<int>>> dist;
vector<int> values;

void floyd_warshall(int n) {
    for (int k = 0; k < n; k++) {  // Intermediate vertex
        for (int i = 0; i < n; i++) {  // Source vertex
            for (int j = 0; j < n; j++) {  // Destination vertex
                dist[0][i][j] = min(dist[0][i][j], dist[0][i][k] + dist[0][k][j]);
            }
        }
    }
}

void solve() {
    cin >> N >> M >> Q;
    dist.assign(M + 1, vector<vector<int>>(N, vector<int>(N, INF)));
    for (int i = 0; i < N; i++) {
        dist[0][i][i] = 0;
    }
    vector<Edge> edges;
    for (int i = 0; i < M; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        u--; v--;
        edges.emplace_back(u, v, w);
        dist[0][u][v] = dist[0][v][u] = 1;
    }
    floyd_warshall(N);
    sort(edges.begin(), edges.end());
    values.assign(M + 1, 0);
    for (int k = 1; k <= M; k++) {
        auto [u, v, w] = edges[k - 1];
        values[k] = w;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                dist[k][i][j] = min({
                    dist[k - 1][i][j],
                    dist[k - 1][i][u] + dist[k - 1][v][j],
                    dist[k - 1][i][v] + dist[k - 1][u][j]
                });
            }
        }
    }
    while (Q--) {
        int a, b, k;
        cin >> a >> b >> k;
        a--; b--;
        int lo = 1, hi = M;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (dist[mid][a][b] < k) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        cout << values[lo] << " ";
    }
    cout << endl;
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

# 

## 

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

# 

## 

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