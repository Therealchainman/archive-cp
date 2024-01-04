# Leetcode Weekly Contest 114

## 2869. Minimum Operations to Collect Elements

### Solution 1:  visited

```py
class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        vis = [0] * k
        n = len(nums)
        for i in reversed(range(n)):
            if nums[i] <= k:
                vis[nums[i] - 1] = 1
            if sum(vis) == k: 
                return n - i
        return n
```

## 2870. Minimum Number of Operations to Make Array Empty

### Solution 1:  counter + remainder on division by 3

if remainder = 2, then when you subtract 2 you get remainder = 0 next so for instance it is ceiling of v / 3 
if remainder = 1, then you get 1 -> 2 -> 0

so it will take 1 or 2 operations before the integer will be divisible by 3.  And then you will just take 3 at a time.

```py
class Solution:
    def minOperations(self, nums: List[int]) -> int:
        counts = Counter(nums)
        if any(v == 1 for v in counts.values()): return -1
        return sum(math.ceil(v / 3) for v in counts.values())
```

## 2871. Split Array Into Maximum Number of Subarrays

### Solution 1:  bitwise and operator + greedy

```py
class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        target = reduce(operator.and_, nums)
        if target > 0: return 1
        cur = 0
        res = 0
        for i in range(n):
            cur = cur & nums[i] if cur > 0 else nums[i]
            if cur == 0: res += 1
        return res
```

## 2872. Maximum Number of K-Divisible Components

### Solution 1:  dp on tree + tree + dfs

```py
class Solution:
    def maxKDivisibleComponents(self, n: int, edges: List[List[int]], values: List[int], k: int) -> int:
        dp = [0] * n
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        def dfs(u, par):
            dp[u] = values[u]
            for v in adj[u]:
                if v == par: continue
                dp[u] += dfs(v, u)
            return dp[u]
        dfs(0, -1)
        return sum(1 for x in dp if x % k == 0)
```

