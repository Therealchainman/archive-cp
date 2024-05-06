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
const int LOG = 21;
int st_gcd[21][LIM + 1];

int query_gcd(int L, int R) {
    int k = log2(R - L + 1);
    return gcd(st_gcd[k][L], st_gcd[k][R - (1LL << k) + 1]);
}
for (int i = 0; i < N; i++) {
    st_gcd[0][i] = A[i] - B[i] > 0 ? A[i] - B[i] : 1;
}
for (int i = 1; i < LOG; i++) {
    for (int j = 0; j + (1LL << (i - 1)) < N; j++) {
        st_gcd[i][j] = gcd(st_gcd[i - 1][j], st_gcd[i - 1][j + (1LL << (i - 1))]);
    }
}
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