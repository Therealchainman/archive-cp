# Binary Jumping

Binary jumping is an algorithm that works for finding kth ancestor, LCA of trees and also for finding the kth successor in a functional/successor graph.

# Functional Graphs

## kth successor in functional graph

Example code for using it to find the kth successor in a functional graph. 

```cpp
vector<vector<int>> succ(LOG, vector<int>(n));
for (int j = 0; j < n; ++j) succ[0][j] = successor[j] - 1;
for (int i = 1; i < LOG; ++i) {
    for (int j = 0; j < n; ++j) {
        succ[i][j] = succ[i - 1][ succ[i - 1][j] ];
    }
}

cin >> q;
while (q--) {
    int x;
    long long k;
    cin >> x >> k;
    int v = x - 1;
    for (int i = 0; i < LOG; ++i) {
        if ((k >> i) & 1LL) v = succ[i][v];
    }
    cout << (v + 1) << '\n';
}
```

## Reachability in jumps

“is v on the forward orbit of u if I keep following next, and if yes how many steps does it take?”

```cpp
// where finding number of steps to reach j from i
int steps = 0;
for (int k = LOG - 1; k >= 0; k--) {
    if (succ[k][i] < j) {
        steps += (1 << k);
        i = succ[k][i];
    }
}
if (succ[0][i] < j) continue;
ans[q] = i < j ? steps + 1 : steps;
```

## sum from node to kth successor in functional graph

With the addition of the idea of sparse table for range sum queries, you can modify it to work with a successor graph and calculate range sums whenever you traverse k edges starting from any node. 

```py
LOG = 35
succ = [[0] * n for _ in range(LOG)]
succ[0] = successor[:]
st = [[0] * n for _ in range(LOG)]
st[0] = list(range(n))
for i in range(1, LOG):
    for j in range(n):
        st[i][j] = st[i - 1][j] + st[i - 1][succ[i - 1][j]]
        succ[i][j] = succ[i - 1][succ[i - 1][j]]
res = 0
for j in range(n):
    sum_ = 0
    for i in range(LOG):
        if ((k + 1) >> i) & 1:
            sum_ += st[i][j]
            j = succ[i][j]
    res = max(res, sum_)
return res
```

# rooted undirected tree

## binary jumping to find the lowest common ancestor (LCA) to use in path queries

This can be used to find sum for the path between two nodes or count/frequency between two nodes in a tree.

for u - v path
sum(u) + sum(v) - 2 * sum(lca(u, v))

Involves finding the depth first search to create the depth, parent, and freq or sum, whatever needs to be calculated
Then it creates the sparse table for looking up ancestors
Then it finds lca and first puts each node on same depth by using kth ancestor and then it moves them up through the tree equally. 

```py
adj_list = [[] for _ in range(n)]
for u, v, w in edges:
    adj_list[u].append((v, w))
    adj_list[v].append((u, w))
LOG = 14
depth = [0] * n
parent = [-1] * n
freq = [[0] * 27 for _ in range(n)]
# CONSTRUCT THE PARENT, DEPTH AND FREQUENCY ARRAY FROM ROOT
def bfs(root):
    queue = deque([root])
    vis = [0] * n
    vis[root] = 1
    dep = 0
    while queue:
        for _ in range(len(queue)):
            node = queue.popleft()
            depth[node] = dep
            for nei, wei in adj_list[node]:
                if vis[nei]: continue
                freq[nei] = freq[node][:]
                freq[nei][wei] += 1
                parent[nei] = node
                vis[nei] = 1
                queue.append(nei)
        dep += 1
bfs(0)
# CONSTRUCT THE SPARSE TABLE FOR THE BINARY JUMPING TO ANCESTORS IN TREE
ancestor = [[-1] * n for _ in range(LOG)]
ancestor[0] = parent[:]
for i in range(1, LOG):
    for j in range(n):
        if ancestor[i - 1][j] == -1: continue
        ancestor[i][j] = ancestor[i - 1][ancestor[i - 1][j]]
def kth_ancestor(node, k):
    for i in range(LOG):
        if (k >> i) & 1:
            node = ancestor[i][node]
    return node
def lca(u, v):
    # ASSUME NODE u IS DEEPER THAN NODE v   
    if depth[u] < depth[v]:
        u, v = v, u
    # PUT ON SAME DEPTH BY FINDING THE KTH ANCESTOR
    k = depth[u] - depth[v]
    u = kth_ancestor(u, k)
    if u == v: return u
    for i in reversed(range(LOG)):
        if ancestor[i][u] != ancestor[i][v]:
            u, v = ancestor[i][u], ancestor[i][v]
    return ancestor[0][u]
```

Implementation of the above code in C++

If unweighted edges, just use addEdge(u, v) without the weight parameter, it will set to 1 by default.

```cpp
struct Tree {
    int N, LOG;
    vector<vector<pair<int,int>>> adj;
    vector<int> depth, parent, dist;
    vector<vector<int>> up;

    Tree(int n) : N(n) {
        LOG = 20;
        adj.assign(N, vector<pair<int, int>>());
        depth.assign(N, 0);
        parent.assign(N, -1);
        dist.assign(N, 0);
        up.assign(LOG, vector<int>(N, -1));
    }
    void addEdge(int u, int v, int w = 1) {
        adj[u].emplace_back(v, w);
        adj[v].emplace_back(u, w);
    }
    void preprocess(int root = 0) {
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
    int distance(int u, int v) const {
        int a = lca(u, v);
        return dist[u] + dist[v] - 2 * dist[a];
    }
private:
    void dfs(int u, int p = -1) {
        parent[u] = p;
        up[0][u] = p;
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
            }
        }
    }
};
/*
example
Tree tree(N);
for (const vector<int> &edge : edges) {
    int u = edge[0], v = edge[1], w = edge[2];
    tree.addEdge(u, v, w);
}
tree.preprocess();
*/
```



## Binary jumping to calculate minimum node value on path queries in tree with LCA

So take care of the sparse table because it is looking at the nodes value it has a few edge cases.

But basically the sparse table over the nodes is along these lines, that is if you have a node u and looking at 4 away, the value over that sparse table is including node u and the next 3 nodes, while the ancestor would be pointing to the 4th node.  So it is behind the ancestor in a sense, so you must return st[0][u] and st[1][u] instead of what it does for lca. 

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
/*
Tree tree(N);
for (const vector<int> &edge : edges) {
    int u = edge[0], v = edge[1], w = edge[2];
    tree.addEdge(u, v, w);
}
tree.preprocess();
query(u, v); // returns minimum value on path from u to v
distance(u, v); // returns distance from u to v
*/

```

To modify a standard LCA-based tree structure (which typically handles vertex values) to instead compute the maximum or minimum edge weight along the path between two nodes, the following changes are necessary:

```cpp
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
```

##  Binary jumping, sparse table for next greater jumps and cost of jumps

Preprocessing with “Next Greater” jumps (Sparse Table / Binary Lifting)

This code uses a “binary lifting” (or sparse table) approach based on the “next greater element” information. For each index i, it stores where the next jump lands (stNextGreater) and the accumulated cost of that jump (stMinOperations). This lets you quickly “jump” in powers of two from one next-greater position to another, rather than moving step by step. The 
query(l,r) function uses these jumps to accumulate a specific cost from 
l up to 
r and makes a final adjustment by subtracting the subarray sum. Essentially, it provides an efficient way to calculate a certain “cost” (related to next greater elements) for subranges in 
O(logN) time per query.

```cpp
vector<int> stNextGreater[BITS];
vector<int64> stMinOperations[BITS];

int64 query(int l, int r) {
    int64 res = 0;
    int idx = l;
    for (int i = BITS - 1; i >= 0; i--) {
        int candIdx = stNextGreater[i][idx];
        if (candIdx <= r) {
            res += stMinOperations[i][idx];
            idx = candIdx;
        }
    }
    if (idx <= r) {
        int segmentLen = r - idx + 1;
        res += (int64)nums[idx] * segmentLen;
    }
    res -= rangeSum(l, r);
    return res;
}
for (int i = 0; i < BITS; i++) {
    stNextGreater[i].assign(N, N);
    stMinOperations[i].assign(N, 0);
}
for (int i = 0; i < N; i++) {
    psum[i] = nums[i];
    if (i > 0) psum[i] += psum[i - 1];
    stNextGreater[0][i] = nextGreater[i];
    stMinOperations[0][i] = (int64)nums[i] * (nextGreater[i] - i);
}
for (int i = 1; i < BITS; i++) {
    for (int j = 0; j < N; j++) {
        int mid = stNextGreater[i - 1][j];
        if (mid >= N) {
            stNextGreater[i][j] = mid;
            stMinOperations[i][j] = stMinOperations[i - 1][j];
        } else {
            stNextGreater[i][j] = stNextGreater[i - 1][mid];
            stMinOperations[i][j] = stMinOperations[i - 1][j] + stMinOperations[i - 1][mid];
        }
    }
}
```