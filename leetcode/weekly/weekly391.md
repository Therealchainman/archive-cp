# Leetcode Weekly Contest 391

## 3101. Count Alternating Subarrays

### Solution 1: sliding window

```py
class Solution:
    def countAlternatingSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        delta = ans = 0
        prev = None
        for x in nums:
            if x == prev:
                delta = 0
            delta += 1
            ans += delta
            prev = x
        return ans
```

## 3102. Minimize Manhattan Distances

### Solution 1:  maximum manhattan disatnce for pair of points

```py
class Solution:
    def max_manhattan_distance(self, points, remove = -1):
        smin = dmin = math.inf
        smax = dmax = -math.inf
        smax_i = smin_i = dmax_i = dmin_i = None
        for i, (x, y) in enumerate(points):
            if remove == i: continue
            s = x + y
            d = x - y
            if s > smax:
                smax = s
                smax_i = i
            if s < smin:
                smin = s
                smin_i = i
            if d > dmax:
                dmax = d
                dmax_i = i
            if d < dmin:
                dmin = d
                dmin_i = i
        return (smax_i, smin_i) if smax - smin >= dmax - dmin else (dmax_i, dmin_i)
    def minimumDistance(self, points: List[List[int]]) -> int:
        i, j = self.max_manhattan_distance(points)
        manhattan_distance = lambda i, j: abs(points[i][0] - points[j][0]) + abs(points[i][1] - points[j][1])
        return min(
            manhattan_distance(*self.max_manhattan_distance(points, i)),
            manhattan_distance(*self.max_manhattan_distance(points, j))
        )
```