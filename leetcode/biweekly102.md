# Leetcode Biweekly Contest 102

## 2639. Find the Width of Columns of a Grid

### Solution 1:  loop + string

```py
class Solution:
    def findColumnWidth(self, grid: List[List[int]]) -> List[int]:
        R, C = len(grid), len(grid[0])
        ans = [0]*C
        for r, c in product(range(R), range(C)):
            ans[c] = max(ans[c], len(str(grid[r][c])))
        return ans
```

## 2640. Find the Score of All Prefixes of an Array

### Solution 1:  prefix sum and max + accumulate

```py
class Solution:
    def findPrefixScore(self, nums: List[int]) -> List[int]:
        n = len(nums)
        pmax = 0
        conver = [0]*n
        for i, num in enumerate(nums):
            pmax = max(pmax, num)
            conver[i] = num + pmax
        return accumulate(conver)
```

```py
class Solution:
    def findPrefixScore(self, nums: List[int]) -> List[int]:
        return accumulate([num + pmax for num, pmax in zip(nums, accumulate(nums, max))])
```

## 2641. Cousins in Binary Tree II

### Solution 1:  2 dfs + first dfs to compute level sum + second dfs to compute cousins value by level_sum - children values

```py
class Solution:
    def replaceValueInTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        level_sum = Counter()
        def dfs1(node, depth):
            for child in filter(None, (node.left, node.right)):
                level_sum[depth] += child.val
                dfs1(child, depth + 1)
        dfs1(root, 1)
        def dfs2(node, root, depth):
            children_val = (root.left.val if root.left else 0) + (root.right.val if root.right else 0)
            cousins_val = level_sum[depth + 1] - children_val
            if root.left:
                node.left = TreeNode(cousins_val)
                dfs2(node.left, root.left, depth + 1)
            if root.right:
                node.right = TreeNode(cousins_val)
                dfs2(node.right, root.right, depth + 1)
            return node
        return dfs2(TreeNode(0), root, 0)
```

## 2642. Design Graph With Shortest Path Calculator

### Solution 1:  dijkstra

```py
class Graph:

    def __init__(self, n: int, edges: List[List[int]]):
        self.adj_list = [[] for _ in range(n)]
        for u, v, w in edges:
            self.adj_list[u].append((v, w))
        self.n = n
        
    def addEdge(self, edge: List[int]) -> None:
        u, v, w = edge
        self.adj_list[u].append((v, w))
        
    def shortestPath(self, node1: int, node2: int) -> int:
        minheap = [(0, node1)]
        min_dist = [math.inf]*self.n
        min_dist[node1] = 0
        while minheap:
            cost, node = heappop(minheap)
            if cost > min_dist[node]: continue
            if node == node2: return cost
            for nei, wei in self.adj_list[node]:
                ncost = cost + wei
                if ncost < min_dist[nei]:
                    heappush(minheap, (ncost, nei))
                    min_dist[nei] = ncost
        return -1
```