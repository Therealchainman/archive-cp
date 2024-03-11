# Leetcode BiWeekly Contest 125

## 3066. Minimum Operations to Exceed Threshold Value II

### Solution 1:  min heap

```py
class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        heapify(minheap := nums)
        ans = 0
        while minheap[0] < k:
            x, y = heappop(minheap), heappop(minheap)
            heappush(minheap, 2 * min(x, y) + max(x, y))
            ans += 1
        return ans
```

## 3067. Count Pairs of Connectable Servers in a Weighted Tree Network

### Solution 1:  combinatorics, dfs, tree

```py
class Solution:
    def countPairsOfConnectableServers(self, edges: List[List[int]], m: int) -> List[int]:
        n = len(edges) + 1
        ans = [0] * n
        adj = [[] for _ in range(n)]
        def dfs(u, p):
            sz[u] = 1 if dist[u] % m == 0 else 0
            for v, w in adj[u]:
                if v == p: continue
                dist[v] = dist[u] + w
                dfs(v, u)
                sz[u] += sz[v]
        for u, v, w in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        for r in range(n):
            dist, sz = [0] * n, [0] * n
            dfs(r, -1)
            cnt = 0
            for v, _ in adj[r]:
                ans[r] += cnt * sz[v]
                cnt += sz[v]
        return ans
```

## 3068. Find the Maximum Sum of Node Values

### Solution 1:  tree, cancelation property of applying xor even, bit manipulation, parity

```py
class Solution:
    def maximumValueSum(self, nums: List[int], k: int, edges: List[List[int]]) -> int:
        ans = sum(max(x, x ^ k) for x in nums)
        cnt = sum(1 for x in nums if x ^ k > x)
        if cnt & 1: # remove smallest
            delta = min(max(x, x ^ k) - min(x, x ^ k) for x in nums)
            ans -= delta
        return ans
```

