# Binary Jumping

Binary jumping is an algorithm that works for finding kth ancestor, LCA of trees and also for finding the kth successor in a functional/successor graph.

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