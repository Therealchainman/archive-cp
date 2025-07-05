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

# Sparse Tables

sparse tables are best for range queries with an immutable array or static array.  They even have O(1) query for indempotent functions and the rest O(logn)

## Idempotent Functions

Idempotent functions are really useful for sparse tables because they allow O(1) operations.
Idempotent function means you can apply the function multiple times and it will not change the result.  For example, min, max, gcd, lcm, etc.  But not sum, product, etc.
f(x,x) = x is condition for idempotent function
sum(x,x) = 2x that is why sum function is not idempotent.
Because you don't care about applying function multiple times you can apply it over an overlapping range in the query, and you can cover any power of two lengthed segment
by using two power of two lengthed segments.

## Range Minimum Query 

RMQ + sparse tables + O(nlogn) precompute sparse tables + O(1) query since the ranges can overlap without affecting the in

This approach that only work on static arrays, i.e. it is not possible to change a value in the array without recomputing the complete data structure.

range minimum query can also be used with (value, index), where the index is location in an array and can be used in a divide and conquer type algorithm to split array into partition at the location of the minimum value in the current range. 


```py
class ST_Min:
    def __init__(self, nums):
        self.nums = nums
        self.n = len(nums)
        self.LOG = 14 # 10,000
        self.build()

    def op(self, x, y):
        return min(x, y)

    def build(self):
        self.lg = [0] * (self.n + 1)
        for i in range(2, self.n + 1):
            self.lg[i] = self.lg[i // 2] + 1
        self.st = [[0] * self.n for _ in range(self.LOG)]
        for i in range(self.n):
            self.st[0][i] = self.nums[i]
        # CONSTRUCT SPARSE TABLE
        for i in range(1, self.LOG):
            j = 0
            while (j + (1 << (i - 1))) < self.n:
                self.st[i][j] = self.op(self.st[i - 1][j], self.st[i - 1][j + (1 << (i - 1))])
                j += 1

    def query(self, l, r):
        length = r - l + 1
        i = self.lg[length]
        return self.op(self.st[i][l], self.st[i][r - (1 << i) + 1])
```

## C++ implementation

The query is inclusive so [L,R]

```cpp
const int LOG = 31;
vector<int> nums;
vector<vector<int>> st;

int query(int L, int R) {
	int k = log2(R - L + 1);
	return min(st[k][L], st[k][R - (1LL << k) + 1]);
}

void solve() {
	int N = read(), Q = read();
	nums.assign(N, 0);
	st.assign(LOG, vector<int>(N, INF));
	for (int i = 0; i < N; i++) {
		nums[i] = read();
		st[0][i] = nums[i];
	}
	for (int i = 1; i < LOG; i++) {
		for (int j = 0; j + (1LL << (i - 1)) < N; j++) {
			st[i][j] = min(st[i - 1][j], st[i - 1][j + (1LL << (i - 1))]);
		}
	}
	while (Q--) {
		int L = read(), R = read();
		printf("%lld\n", query(L, R - 1));
	}
}
```

## Range GCD Queries

This works only for static arrays, because each update requires recomputing the entire sparse table, which is a lot of work for single point update.

These are inclusive so query range [L, R]

Watch out for when gcd(0,0) = 0, what does that mean for the specific problem

Can also precompute logarithms at expense of memory

                                      
```py
st_gcd = [[0] * n for _ in range(LOG)]
st_gcd[0] = [x - y for x, y in zip(a, b)] # replace with whatever value
for i in range(1, LOG):
    j = 0
    while (j + (1 << (i - 1))) < n:
        st_gcd[i][j] = math.gcd(st_gcd[i - 1][j], st_gcd[i - 1][j + (1 << (i - 1))])
        j += 1
def query(left, right):
    length = right - left + 1
    k = int(math.log2(length))
    return math.gcd(st_gcd[k][left], st_gcd[k][right - (1 << k) + 1])
```

```cpp
const int LOG = 30;
struct SparseGCD {
    int N;
    vector<vector<int64>> st;
    SparseGCD(const vector<int> &arr) : N(arr.size()), st(LOG, vector<int64>(N, 0)) {
        for (int i = 0; i < N; i++) {
            st[0][i] = arr[i];
        }
        for (int i = 1; i < LOG; i++) {
            for (int j = 0; j + (1LL << i) <= N; j++) {
                st[i][j] = gcd(st[i - 1][j], st[i - 1][j + (1LL << (i - 1))]);
            }
        }
    }
    int64 query(int l, int r) const {
        int k = log2(r - l + 1);
        return gcd(st[k][l], st[k][r - (1LL << k) + 1]);
    }
};
```

## Range Bitwise Or Queries

It queries inclusive ranges [l, r], that are 0-indexed.

query is O(logn)
precompute is (nlogn)

think can change this to O(1), cause bitwise or is idempotent

```py
LOG = 14 # 200,000
st = [[0] * n for _ in range(LOG)]
for i in range(n): # this is not complete, initialize how needed
    st[0][i] = w
# CONSTRUCT SPARSE TABLE
for i in range(1, LOG):
    j = 0
    while (j + (1 << (i - 1))) < n:
        st[i][j] = st[i - 1][j] | st[i - 1][j + (1 << (i - 1))]
        j += 1
# QUERY SPARSE TABLE
def query(l, r):
    res = 0
    for i in reversed(range(LOG)):
        if (1 << i) <= r - l + 1:
            res |= st[i][l] 
            l += 1 << i
    return res
```

## Range Bitwise And Queries

It queries inclusive ranges [l, r], that are 0-indexed.

query is O(1) (idempotent)
precompute is (nlogn)

```py
class ST_And:
    def __init__(self, nums):
        self.nums = nums
        self.n = len(nums)
        self.LOG = 14 # 10,000
        self.build()

    def op(self, x, y):
        return x & y

    def build(self):
        self.lg = [0] * (self.n + 1)
        for i in range(2, self.n + 1):
            self.lg[i] = self.lg[i // 2] + 1
        self.st = [[0] * self.n for _ in range(self.LOG)]
        for i in range(self.n): 
            self.st[0][i] = self.nums[i]
        # CONSTRUCT SPARSE TABLE
        for i in range(1, self.LOG):
            j = 0
            while (j + (1 << (i - 1))) < self.n:
                self.st[i][j] = self.op(self.st[i - 1][j], self.st[i - 1][j + (1 << (i - 1))])
                j += 1

    def query(self, l, r):
        length = r - l + 1
        i = self.lg[length]
        return self.op(self.st[i][l], self.st[i][r - (1 << i) + 1])
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