# Binary Jumping

Binary jumping is an algorithm that works for finding kth ancestor, LCA of trees and also for finding the kth successor in a functional/successor graph.

# Functional Graphs

## kth successor in functional graph

Example code for using it to find the kth successor in a functional graph. 

```py
succ = [[0] * n for _ in range(LOG)]
succ[0] = [s - 1 for s in successor]
for i in range(1, LOG):
    for j in range(n):
        succ[i][j] = succ[i - 1][succ[i - 1][j]]
for x, k in queries:
    x -= 1
    for i in range(LOG):
        if (k >> i) & 1:
            x = succ[i][x]
    print(x + 1)
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

## Binary jumping to calculate minimum node value on path queries in tree with LCA

So take care of the sparse table because it is looking at the nodes value it has a few edge cases.

But basically the sparse table over the nodes is along these lines, that is if you have a node u and looking at 4 away, the value over that sparse table is including node u and the next 3 nodes, while the ancestor would be pointing to the 4th node.  So it is behind the ancestor in a sense, so you must return st[0][u] and st[1][u] instead of what it does for lca. 

```cpp
// binary lifting algorithm

vector<int> depth, parent;
const int LOG = 20, inf = INT_MAX;
vector<vector<int>> ancestor, st, adj;
vector int n;

// bfs from root of tree to calculate depth of nodes in the tree
void bfs(int root) {
    queue<int> q;
    depth.assign(block_id, inf);
    parent.assign(block_id, -1);
    q.push(root);
    depth[root] = 0;
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int v : adj[u]) {
            if (depth[u] + 1 < depth[v]) {
                depth[v] = depth[u] + 1;
                parent[v] = u;
                q.push(v);
            }
        }
    }
}

// preprocess the ancestor and sparse table for minimum array
void preprocess() {
    ancestor.assign(LOG, vector<int>(n, -1));
    st.assign(LOG, vector<int>(n, inf));
    for (int i = 0; i < n; i++) {
        ancestor[0][i] = parent[i];
        st[0][i] = dist[i];
    }
    for (int i = 1; i < LOG; i++) {
        for (int j = 0; j < n; j++) {
            if (ancestor[i - 1][j] != -1) {
                ancestor[i][j] = ancestor[i - 1][ancestor[i - 1][j]];
                st[i][j] = min(st[i - 1][j], st[i - 1][ancestor[i - 1][j]]);
            }
        }
    }
}

// LCA queries to calculate the minimum node value in path from u to v
int lca(int u, int v) {
    if (depth[u] < depth[v]) swap(u, v);
    int ans = inf;
    int k = depth[u] - depth[v];
    if (k > 0) {
        for (int i = 0; i < LOG; i++) {
            if ((k >> i) & 1) {
                ans = min(ans, st[i][u]);
                u = ancestor[i][u];
            }
        }
    }
    if (u == v) {
        ans = min(ans, st[0][u]);
        return ans;
    }
    for (int i = LOG - 1; i >= 0; i--) {
        if (ancestor[i][u] != -1 && ancestor[i][u] != ancestor[i][v]) {
            ans = min(ans, st[i][u]);
            ans = min(ans, st[i][v]);
            u = ancestor[i][u]; v = ancestor[i][v];
        }
    }
    ans = min(ans, st[1][u]);
    ans = min(ans, st[1][v]);
    return ans;
}
void solve() {
    bfs(0);
    preprocess();
    int Q = read();
    while (Q--) {
        int u = read(), v = read();
        // path min query on the bridge tree
        res += lca(u, v);
    }
}

```

# Sparse Tables

## Range Minimum Query 

RMQ + sparse tables + O(nlogn) precompute sparse tables + O(1) query since the ranges can overlap without affecting the in

This approach that only work on static arrays, i.e. it is not possible to change a value in the array without recomputing the complete data structure.

range minimum query can also be used with (value, index), where the index is location in an array and can be used in a divide and conquer type algorithm to split array into partition at the location of the minimum value in the current range. 


```py
n = len(nums)
lg = [0] * (n + 1)
for i in range(2, n + 1):
    lg[i] = lg[i // 2] + 1
LOG = lg[-1] + 1
st = [[math.inf] * n for _ in range(LOG)]
st[0] = nums[:]
for i in range(1, LOG):
    j = 0
    while (j + (1 << (i - 1))) < n:
        st[i][j] = min(st[i - 1][j], st[i - 1][j + (1 << (i - 1))])
        j += 1
def query(left, right):
    length = right - left + 1
    i = lg[length]
    return min(st[i][left], st[i][right - (1 << i) + 1])
```