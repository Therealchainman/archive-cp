# Leetcode Weekly Contest 361

## 2843. Count Symmetric Integers

### Solution 1: 

```py
class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        res = 0
        for i in range(low, high + 1):
            digits = str(i)
            n = len(digits)
            if n & 1: continue
            sum1, sum2 = sum(int(digits[j]) for j in range(n // 2)), sum(int(digits[j]) for j in range(n // 2, n))
            res += sum1 == sum2
        return res
```

## 2844. Minimum Operations to Make a Special Number

### Solution 1: 

```py
class Solution:
    def minimumOperations(self, num: str) -> int:
        stack = list(num)
        match = {"25", "50", "75", "00"}
        last = set()
        n = len(num)
        for i in reversed(range(n)):
            if num[i] in "27" and "5" in last:
                return n - i - 2
            if num[i] in "50" and "0" in last:
                return n - i - 2
            last.add(num[i])
        return n - ("0" in num)
```

## 2845. Count of Interesting Subarrays

### Solution 1:  dp + math

```py

```

## 2846. Minimum Edge Weight Equilibrium Queries in a Tree

### Solution 1:  binary jumping + lowest common ancestor (LCA) + frequency array for each weight + tree

The minimum operations is by not changing the one with highest frequency and thus would require most operations. 

```py
class Solution:
    def minOperationsQueries(self, n: int, edges: List[List[int]], queries: List[List[int]]) -> List[int]:
        adj_list = [[] for _ in range(n)]
        for u, v, w in edges:
            adj_list[u].append((v, w))
            adj_list[v].append((u, w))
        LOG = 14
        depth = [0] * n
        parent = [-1] * n
        freq = [[0] * 27 for _ in range(n)]
        # CONSTRUCT THE PARENT, DEPTH AND FREQUENCY ARRAY FROM ROOT
        def dfs(root):
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
        dfs(0)
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
        ans = [None] * len(queries)
        for i, (u, v) in enumerate(queries):
            lca_node = lca(u, v)
            freqs = [freq[u][w] + freq[v][w] - 2 * freq[lca_node][w] for w in range(27)]
            ans[i] = sum(freqs) - max(freqs)
        return ans
```

