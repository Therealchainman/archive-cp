# Leetcode BiWeekly Contest 128

## 3112. Minimum Time to Visit Disappearing Nodes

### Solution 1:  dijkstra, shortest path, undirected graph

```py
def dijkstra(adj, src, dis):
    N = len(adj)
    min_heap = [(0, src)]
    dist = [math.inf] * N
    while min_heap:
        cost, u = heapq.heappop(min_heap)
        if cost >= dis[u]: continue
        if cost >= dist[u]: continue
        dist[u] = cost
        for v, w in adj[u]:
            if cost + w < dist[v]: heapq.heappush(min_heap, (cost + w, v))
    return dist
class Solution:
    def minimumTime(self, n: int, edges: List[List[int]], disappear: List[int]) -> List[int]:
        pedges = defaultdict(lambda: math.inf)
        adj = [[] for _ in range(n)]
        for u, v, w in edges:
            pedges[(u, v)] = min(pedges[(u, v)], w)
            pedges[(v, u)] = min(pedges[(v, u)], w)
        for (u, v), w in pedges.items():
            adj[u].append((v, w))
        dist_arr = dijkstra(adj, 0, disappear)
        ans = [-1] * n
        for u in range(n):
            if dist_arr[u] < disappear[u]: ans[u] = dist_arr[u]
        return ans
```

## 3113. Find the Number of Subarrays Where Boundary Elements Are Maximum

### Solution 1:  groups, sliding window, range maximum query sparse tables

```py

class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
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
                st[i][j] = max(st[i - 1][j], st[i - 1][j + (1 << (i - 1))])
                j += 1
        def query(left, right):
            length = right - left + 1
            i = lg[length]
            return max(st[i][left], st[i][right - (1 << i) + 1])
        pos = defaultdict(list)
        for i in range(n):
            pos[nums[i]].append(i)
        ans = 0
        for val, arr in pos.items():
            delta = 0
            start = arr[0]
            for end in arr:
                if query(start, end) != val:
                    start = end
                    delta = 0
                delta += 1
                ans += delta
        return ans
```