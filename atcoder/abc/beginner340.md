# Atcoder Beginner Contest 340

## C - Divide and Divide

### Solution 1:  dynamic programming

```py
import math

def main():
    N = int(input())
    dp = Counter({N: 1})
    ans = 0
    while dp:
        ndp = Counter()
        for k, v in dp.items():
            ans += v * k
            floor = k // 2
            ceil = (k + 1) // 2
            if floor > 1: ndp[floor] += v
            if ceil > 1: ndp[ceil] += v
        dp = ndp
    print(ans)

if __name__ == '__main__':
    main()
```

## D - Super Takahashi Bros. 

### Solution 1:  dijkstra, directed graph, single source shortest path froms source to destination

```py
import heapq
def dijkstra(adj, src, dst):
    N = len(adj)
    min_heap = [(0, src)]
    vis = [0] * N
    while min_heap:
        cost, u = heapq.heappop(min_heap)
        if u == dst: return cost
        if vis[u]: continue
        vis[u] = 1
        for v, w in adj[u]:
            if vis[v]: continue
            heapq.heappush(min_heap, (cost + w, v))
    return -1
def main():
    N = int(input())
    adj = [[] for _ in range(N)]
    for i in range(N - 1):
        A, B, X = map(int, input().split())
        X -= 1
        adj[i].append((i + 1, A))
        adj[i].append((X, B))
    ans = dijkstra(adj, 0, N - 1)
    print(ans)

if __name__ == '__main__':
    main()
```

## E - Mancala 2

### Solution 1:  range update, point query, addition on segment, lazy propagation

```py
class LazySegmentTree:
    def __init__(self, n: int, neutral: int, noop: int):
        self.neutral = neutral
        self.size = 1
        self.noop = noop
        self.n = n 
        while self.size<n:
            self.size*=2
        self.tree = [neutral for _ in range(self.size*2)]

    def is_leaf_node(self, segment_right_bound, segment_left_bound) -> bool:
        return segment_right_bound - segment_left_bound == 1

    def operation(self, x: int, y: int) -> int:
        return x + y

    def propagate(self, segment_idx: int, segment_left_bound: int, segment_right_bound: int) -> None:
        # do not want to propagate if it is a leaf node
        if self.is_leaf_node(segment_right_bound, segment_left_bound): return
        left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
        self.tree[left_segment_idx] = self.operation(self.tree[left_segment_idx], self.tree[segment_idx])
        self.tree[right_segment_idx] = self.operation(self.tree[right_segment_idx], self.tree[segment_idx])
        self.tree[segment_idx] = self.noop
    
    def update(self, left: int, right: int, val: int) -> None:
        stack = [(0, self.size, 0)]
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                self.tree[segment_idx] = self.operation(self.tree[segment_idx], val)
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])

    def query(self, i: int) -> int:
        stack = [(0, self.size, 0)]
        while stack:
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if i < segment_left_bound or i >= segment_right_bound: continue
            # LEAF NODE
            if self.is_leaf_node(segment_right_bound, segment_left_bound): 
                return self.tree[segment_idx]
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.propagate(segment_idx, segment_left_bound, segment_right_bound)            
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
    
    def __repr__(self) -> str:
        return f"array: {self.tree}"

# [L, R)
def main():
    N, M = map(int, input().split())
    arr = list(map(int, input().split()))
    queries = list(map(int, input().split()))
    seg = LazySegmentTree(N, 0, 0)
    for i in range(N):
        seg.update(i, i + 1, arr[i])
    for i in range(M):
        idx = queries[i]
        balls = seg.query(idx)
        x = balls // N
        seg.update(idx, idx + 1, -balls)
        if x > 0:
            seg.update(0, N, x)
        if balls % N > 0:
            segment = [(idx + 1) % N, (idx + balls % N) % N]
            if segment[1] < segment[0]:
                seg.update(segment[0], N, 1)
                seg.update(0, segment[1] + 1, 1)
            else:
                seg.update(segment[0], segment[1] + 1, 1)
    ans = [seg.query(i) for i in range(N)]
    print(*ans)
```

## F - S = 1

### Solution 1:  linear diophantine equation, extended euclidean algorithm

```py
def extended_euclidean(a, b, x, y):
    if b == 0: return a, 1, 0
    g, x1, y1 = extended_euclidean(b, a % b, x, y)
    return g, y1, x1 - y1 * (a // b)

def main():
    A, B = map(int, input().split())
    C = 2
    # Ax + By = C
    # Bx - Ay = C
    g, x, y = extended_euclidean(B, -A, 0, 0)
    if C % g != 0: return print(-1)
    x *= 2 // g
    y *= 2 // g
    print(x, y)

if __name__ == '__main__':
    main()
```

## G - Leaf Color

### Solution 1:  virtual or aux tree, lca, binary jumping, dfs, dp on tree

```py
from collections import deque, defaultdict
LOG = 18
MOD = 998244353

def main():
    n = int(input())
    colors = list(map(int, input().split()))
    adj = [[] for _ in range(n)]
    virt = [[] for _ in range(n)]
    for _ in range(n - 1):
        u, v = map(int, input().split())
        u -= 1; v -= 1
        adj[u].append(v)
        adj[v].append(u)
    depth = [0] * n
    parent = [-1] * n
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
                for nei in adj[node]:
                    if vis[nei]: continue
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
    # CONSTRUCT THE DISCOVERY TIME ARRAY FOR EACH NODE
    disc = [0] * n
    timer = 0
    def dfs(u, p):
        nonlocal timer
        disc[u] = timer
        timer += 1
        for v in adj[u]:
            if v == p: continue
            dfs(v, u)
    dfs(0, -1)
    # CONSTRUCT THE AUXILIARY TREES FOR EACH SET S OF THE SAME COLOR
    tree_sets = defaultdict(list)
    for u in sorted(range(n), key = lambda i: disc[i]):
        tree_sets[colors[u]].append(u)
    # DP ON AUX TREES
    ans = 0
    def dp(u, c):
        nonlocal ans
        res, sum_ = 1, 0
        for v in virt[u]:
            child = dp(v, c)
            res = (res * (child + 1)) % MOD
            sum_ = (sum_ + child) % MOD
        if colors[u] == c:
            ans = (ans + res) % MOD
        else:
            ans = (ans + res - sum_ - 1) % MOD
            res = (res - 1) % MOD
        return res
    for c, S in tree_sets.items():
        P = set(S)
        m = len(S)
        for i in range(1, m):
            P.add(lca(S[i - 1], S[i]))
        P = sorted(P, key = lambda i: disc[i])
        parents = [None] * (len(P) - 1)
        for i in range(1, len(P)):
            par = lca(P[i - 1], P[i])
            parents[i - 1] = par
            virt[par].append(P[i])
        dp(P[0], c)
        for p in parents:
            virt[p].clear()
    print(ans)                 

if __name__ == '__main__':
    main()
```