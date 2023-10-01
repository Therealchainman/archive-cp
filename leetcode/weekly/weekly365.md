# Leetcode Weekly Contest 365

## 2874. Maximum Value of an Ordered Triplet II

### Solution 1:  prefix max

```py
class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        pmax = delta = -math.inf
        res = 0
        for num in nums:
            res = max(res, num * delta)
            delta = max(delta, pmax - num)
            pmax = max(pmax, num)
        return res
```

### Solution 2:  prefix max + suffix max

```py

```

## 2875. Minimum Size Subarray in Infinite Array

### Solution 1:  modular arithmetic + sliding window

```py
class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        sum_ = sum(nums)
        middle_len = (target // sum_) * n
        target %= sum_
        cur = left = 0
        res = math.inf
        for right in range(2 * n):
            cur += nums[right % n]
            while cur > target:
                cur -= nums[left % n]
                left += 1
            if cur == target:
                res = min(res, middle_len + right - left + 1)
        return res if res < math.inf else -1
```

## 2876. Count Visited Nodes in a Directed Graph

### Solution 1:  functional graph + recover path + cycle detection

```py
class Solution:
    def countVisitedNodes(self, edges: List[int]) -> List[int]:
        n = len(edges)
        ans, vis = [0] * n, [0] * n
        def search(u):
            parent = {u: None}
            is_cycle = False
            while True:
                vis[u] = 1
                v = edges[u]
                if v in parent: 
                    is_cycle = True
                    break
                if vis[v]: break
                parent[v] = u
                u = v
            if is_cycle:
                crit_point = parent[edges[u]]
                cycle_path = []
                while u != crit_point:
                    cycle_path.append(u)
                    u = parent[u]
                len_ = len(cycle_path)
                for val in cycle_path:
                    ans[val] = len_
            while u is not None:
                ans[u] = ans[edges[u]] + 1
                u = parent[u]
        for i in range(n):
            if vis[i]: continue
            search(i)
        return ans
```

