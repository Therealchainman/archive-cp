# Leetcode Biweekly Contest 101

## 2605. Form Smallest Number From Two Digit Arrays

### Solution 1:

```py
class Solution:
    def minNumber(self, nums1: List[int], nums2: List[int]) -> int:
        min1 = min2 = math.inf
        for dig in map(int, string.digits):
            if dig in nums1 and dig in nums2:
                return dig
            if dig in nums1:
                min1 = min(min1, dig)
            if dig in nums2:
                min2 = min(min2, dig)
        return min1*10 + min2 if min1 < min2 else min2*10 + min1
```

## 2606. Find the Substring With Maximum Cost

### Solution 1:  dp + kadanes

This is similar to kadanes' algo to get the maximum cost of a subarray. 

```py
class Solution:
    def maximumCostSubstring(self, s: str, chars: str, vals: List[int]) -> int:
        values = list(range(1, 27))
        unicode = lambda ch: ord(ch) - ord('a')
        for ch, v in zip(chars, vals):
            i = unicode(ch)
            values[i] = v
        res = cur = 0
        for ch in s:
            i = unicode(ch)
            cur = max(0, cur + values[i])
            res = max(res, cur)
        return res
```

## 2608. Shortest Cycle in a Graph

### Solution 1: dfs + depth array

```py
class Solution:
    def findShortestCycle(self, n: int, edges: List[List[int]]) -> int:
        adj_list = [[] for _ in range(n)]
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        res = math.inf
        depth = [0]*n
        def dfs(node, parent):
            nonlocal res
            for nei in adj_list[node]:
                if nei == parent: continue
                if depth[nei]: 
                    res = min(res, abs(depth[node] - depth[nei]) + 1)
                    continue
                depth[nei] = depth[node] + 1
                dfs(nei, node)
        for i in range(n):
            depth[i] = 1
            dfs(i, -1)
            depth = [0]*n
        return res if res < math.inf else -1
```

### Solution 2:  bfs + early termination + dist array 

can use the distance from root to know if it is parent or not, cause if distance is smaller, it is parent

total distance is distance from root to nei and node + 1 to get the size of the smallest cycle found, which is the first cycle. 
Although sometimes it counts the value as too large cause it is not always getting size of cycle if the root node does not belong to a cycle. 
But since minimizing when treating element in the cycle as root that will be the shortest cycle anyway it still works. 

```py
class Solution:
    def findShortestCycle(self, n: int, edges: List[List[int]]) -> int:
        adj_list = [[] for _ in range(n)]
        for u, v in edges:
            adj_list[u].append(v)
            adj_list[v].append(u)
        def bfs(root):
            # dist from root node
            dist = [-1]*n
            dist[root] = 0
            queue = deque([root])
            while queue:
                node = queue.popleft()
                for nei in adj_list[node]:
                    if dist[nei] == -1:
                        dist[nei] = dist[node] + 1
                        queue.append(nei)
                    elif dist[nei] >= dist[node]:
                        return dist[nei] + dist[node] + 1
            return math.inf
       
        res = min(map(bfs, range(n)))
        return res if res < math.inf else -1
```

## 2607. Make K-Subarray Sums Equal

### Solution 1:  gcd + cycle + sort

This one is weird, still don't have a fully understanding of it. Why does the gcd work I can't prove it. But you can also solve this
by creating a visited array and just finding the cycles that way. 

basic pattern is that you need certain elements equal such as n = 4, k = 2
a(0) + a(1) = a(1) + a(2) = ...

so basically need a(i) = a(i + k), so this can be solved with a loop and a secondardy loop that goes through increments of k to build the cycle 
and just store visited for the outer loop so don't redo. 

```py
class Solution:
    def makeSubKSumEqual(self, arr: List[int], k: int) -> int:
        n = len(arr)
        dist = gcd(n, k)
        res = 0
        for i in range(dist):
            cycle = sorted([arr[j] for j in range(i, n, dist)])
            median = cycle[len(cycle)//2]
            res += sum([abs(v - median) for v in cycle])
        return res
```