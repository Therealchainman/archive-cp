# Leetcode Weekly Contest 113

## 2855. Minimum Right Shifts to Sort the Array

### Solution 1: 

```py
class Solution:
    def minimumRightShifts(self, nums: List[int]) -> int:
        n = len(nums)
        i = 1
        while i < n and nums[i - 1] < nums[i]:
            i += 1
        for j in range(i, n):
            if j > i and nums[j] < nums[j - 1]: return -1
            if nums[j] > nums[i - 1]: return -1
        return n - i
```

## 2856. Minimum Array Length After Pair Removals

### Solution 1:  max heap + counter

```py
class Solution:
    def minLengthAfterRemovals(self, nums: List[int]) -> int:
        n = len(nums)
        freq = Counter(nums)
        max_heap = []
        for num, cnt in freq.items():
            heappush(max_heap, (-cnt, num))
        while len(max_heap) > 1:
            cnt_x, x = heappop(max_heap)
            cnt_y, y = heappop(max_heap)
            cnt_x, cnt_y = map(abs, (cnt_x, cnt_y))
            cnt_x -= 1
            cnt_y -= 1
            if cnt_x > 0:
                heappush(max_heap, (-cnt_x, x))
            if cnt_y > 0:
                heappush(max_heap, (-cnt_y, y))
        return sum(abs(cnt) for cnt, _ in max_heap)
```

## 2857. Count Pairs of Points With Distance k

### Solution 1:  bit manipulation + math

```py

```

## 2858. Minimum Edge Reversals So Every Node Is Reachable

### Solution 1:  reroot tree + tree dp

```py
class Solution:
    def minEdgeReversals(self, n: int, edges: List[List[int]]) -> List[int]:
        dp = [0] * n
        adj_list = [[] for _ in range(n)]
        for u, v in edges:
            adj_list[u].append((v, 0))
            adj_list[v].append((u, 1))
        def dfs(node, parent):
            s = 0
            for nei, wei in adj_list[node]:
                if nei == parent: continue
                s += wei
                s += dfs(nei, node)
            dp[node] = s
            return s
        dfs(0, -1)
        ans = [0] * n
        def dfs2(node, parent, psum):
            ans[node] = dp[node] + psum
            for nei, wei in adj_list[node]:
                if nei == parent: continue
                nsum = psum + (wei ^ 1) + dp[node] - dp[nei] - wei
                dfs2(nei, node, nsum)
        dfs2(0, -1, 0)
        return ans
```

