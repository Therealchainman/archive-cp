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

### Solution 2:  maximum segment tree + offline queries + sort + coordinate compression

```py
class SegmentTree:
    def __init__(self, n: int, neutral: int, func):
        self.func = func
        self.neutral = neutral
        self.size = 1
        self.n = n
        while self.size<n:
            self.size*=2
        self.nodes = [neutral for _ in range(self.size*2)]

    def ascend(self, segment_idx: int) -> None:
        while segment_idx > 0:
            segment_idx -= 1
            segment_idx >>= 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            self.nodes[segment_idx] = self.func(self.nodes[left_segment_idx], self.nodes[right_segment_idx])
        
    def update(self, segment_idx: int, val: int) -> None:
        segment_idx += self.size - 1
        self.nodes[segment_idx] = self.func(self.nodes[segment_idx], val)
        self.ascend(segment_idx)
            
    def query(self, left: int, right: int) -> int:
        stack = [(0, self.size, 0)]
        result = self.neutral
        while stack:
            # BOUNDS FOR CURRENT INTERVAL and idx for tree
            segment_left_bound, segment_right_bound, segment_idx = stack.pop()
            # NO OVERLAP
            if segment_left_bound >= right or segment_right_bound <= left: continue
            # COMPLETE OVERLAP
            if segment_left_bound >= left and segment_right_bound <= right:
                result = self.func(result, self.nodes[segment_idx])
                continue
            # PARTIAL OVERLAP
            mid_point = (segment_left_bound + segment_right_bound) >> 1
            left_segment_idx, right_segment_idx = 2*segment_idx + 1, 2*segment_idx + 2
            stack.extend([(mid_point, segment_right_bound, right_segment_idx), (segment_left_bound, mid_point, left_segment_idx)])
        return result
    
    def __repr__(self) -> str:
        return f"nodes array: {self.nodes}, next array: {self.nodes}"

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        queries = sorted([(left, right, i) for i, (left, right) in enumerate(queries)], reverse = True)
        nums = sorted([(n1, n2) for n1, n2 in zip(nums1, nums2)], reverse = True)
        values = set()
        for _, v, _ in queries:
            values.add(v)
        for _, v in nums:
            values.add(v)
        compressed = {}
        for i, v in enumerate(sorted(values)):
            compressed[v] = i
        max_seg_tree = SegmentTree(len(compressed), -1, max)
        ans = [-1] * len(queries)
        i = 0
        for left, right, idx in queries:
            while i < n and nums[i][0] >= left:
                max_seg_tree.update(compressed[nums[i][1]], sum(nums[i]))
                i += 1
            ans[idx] = max_seg_tree.query(compressed[right], len(compressed))
        return ans
```


