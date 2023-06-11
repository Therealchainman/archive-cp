# Leetcode Weekly Contest 349

## 2733. Neither Minimum nor Maximum

### Solution 1:  min and max + scan

```py
class Solution:
    def findNonMinOrMax(self, nums: List[int]) -> int:
        min_, max_ = min(nums), max(nums)
        for num in nums:
            if num not in (min_, max_): return num
        return -1
```

## 2734. Lexicographically Smallest String After Substring Operation

### Solution 1:  greedy + decrease the earliest infix that is not containing a

```py
class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        start = n - 1
        for i in range(n):
            if s[i] != 'a':
                start = i
                break
        res = list(s[:start])
        for i in range(start, n):
            if i != start and s[i] == 'a': 
                res.extend(s[i:])
                break
            if s[i] == 'a':
                res.append('z')
                continue
            res.append(chr(ord(s[i]) - 1))
        return ''.join(res)
```

## 2735. Collecting Chocolates

### Solution 1:  simulation

```py
class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        res = sum(nums)
        mins = nums[:]
        for i in range(1, n):
            savings = 0
            v = []
            for j in range(n):
                if nums[(j + i) % n] < mins[j]:
                    savings += (mins[j] - nums[(j + i) % n])
                    v.append((j, nums[(j + i) % n]))
            savings -= x
            if savings <= 0: break
            if savings > 0:
                res -= savings
                for j, val in v:
                    mins[j] = val
        return res
```

## 2736. Maximum Sum Queries

### Solution 1:  two heaps + offline query + sort

```py
class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n, m = len(nums1), len(queries)
        ans = [-1] * m
        queries = sorted([(x, y, i) for i, (x, y) in enumerate(queries)], reverse = True)
        max_heap = []
        heap = []
        nums = sorted([(x1, x2) for x1, x2 in zip(nums1, nums2)], reverse = True)
        j = 0
        for x, y, i in queries:
            while j < n and nums[j][0] >= x:
                if nums[j][1] < y:
                    heappush(heap, (-nums[j][1], -sum(nums[j])))
                else:
                    heappush(max_heap, (-sum(nums[j]), nums[j][1]))
                j += 1
            while heap and abs(heap[0][0]) >= y:
                r, v = heappop(heap)
                heappush(max_heap, (v, abs(r)))
            while max_heap and max_heap[0][1] < y:
                v1, v2 = heappop(max_heap)
                heappush(heap, (-v2, v1))
            if max_heap: ans[i] = abs(max_heap[0][0]) 
        return ans
```


