# Leetcode Weekly Contest 394

## 100290. Minimum Number of Operations to Satisfy Conditions

### Solution 1:  dynamic programming with pmin and smin, frequency count for each column

```py
class Solution:
    def minimumOperations(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        dp = [0] * 10
        for c in range(C):
            freq = [0] * 10
            for r in range(R):
                freq[grid[r][c]] += 1
            pmin = [math.inf] * 10
            smin = [math.inf] * 10
            for i in range(10):
                pmin[i] = dp[i]
                if i > 0: pmin[i] = min(pmin[i], pmin[i - 1])
            for i in reversed(range(10)):
                smin[i] = dp[i]
                if i < 9: smin[i] = min(smin[i], smin[i + 1])
            for i, f in enumerate(freq):
                dp[i] = R - f
                mn = math.inf
                if i > 0:
                    mn = min(mn, pmin[i - 1])
                if i < 9:
                    mn = min(mn, smin[i + 1])
                if mn != math.inf: dp[i] += mn
        return min(dp)
```

## 100276. Find Edges in Shortest Paths

### Solution 1:  double dijkstra algorithm

```py
def dijkstra(adj, src):
    N = len(adj)
    min_heap = [(0, src)]
    dist = [math.inf] * N
    while min_heap:
        cost, u = heapq.heappop(min_heap)
        if cost >= dist[u]: continue
        dist[u] = cost
        for v, w in adj[u]:
            if cost + w < dist[v]: heapq.heappush(min_heap, (cost + w, v))
    return dist

class Solution:
    def findAnswer(self, n: int, edges: List[List[int]]) -> List[bool]:
        m = len(edges)
        adj = [[] for _ in range(n)]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        src_dist = dijkstra(adj, 0)
        dst_dist = dijkstra(adj, n - 1)
        target = src_dist[n - 1]
        ans = [False] * m
        if target == math.inf: return ans
        for i, (u, v, w) in enumerate(edges):
            ans[i] = src_dist[u] + w + dst_dist[v] == target or src_dist[v] + w + dst_dist[u] == target
        return ans
```
