# Leetcode Weekly Contest 392

## 3107. Minimum Operations to Make Median of Array Equal to K

### Solution 1:  binary search, median

```py
class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        n = len(nums)
        m = n // 2
        nums.sort()
        i = bisect_left(nums, k)
        ans = 0
        if i > m:
            for j in range(m, i):
                ans += k - nums[j]
        else:
            for j in range(i, m + 1):
                ans += nums[j] - k
        return ans
```

## 3108. Minimum Cost Walk in Weighted Graph

### Solution 1:  union find, bitwise and operation

```py
class UnionFind:
    def __init__(self, n: int):
        self.size = [1]*n
        self.parent = list(range(n))
    
    def find(self,i: int) -> int:
        while i != self.parent[i]:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i
    """
    returns true if the nodes were not union prior. 
    """
    def union(self,i: int,j: int) -> bool:
        i, j = self.find(i), self.find(j)
        if i!=j:
            if self.size[i] < self.size[j]:
                i,j=j,i
            self.parent[j] = i
            self.size[i] += self.size[j]
            return True
        return False
class Solution:
    def minimumCost(self, n: int, edges: List[List[int]], query: List[List[int]]) -> List[int]:
        BITS = 18
        dsu = UnionFind(n)
        for u, v, _ in edges:
            dsu.union(u, v)
        resp = defaultdict(lambda: (1 << BITS) - 1)
        for u, v, w in edges:
            root = dsu.find(u)
            resp[root] &= w
        m = len(query)
        ans = [-1] * m
        for i, (s, t) in enumerate(query):
            if dsu.find(s) != dsu.find(t): continue
            ans[i] = resp[dsu.find(t)] if s != t else 0
        return ans
```