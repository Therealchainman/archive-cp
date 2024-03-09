# Leetcode Weekly Contest 386

## 3047. Find the Largest Area of Square Inside Two Rectangles

### Solution 1: 

```py
class Solution:
    def largestSquareArea(self, bottomLeft: List[List[int]], topRight: List[List[int]]) -> int:
        n = len(bottomLeft)
        ans = 0
        intersection = lambda s1, e1, s2, e2: min(e1, e2) - max(s1, s2)
        for i in range(n):
            (x1, y1), (x2, y2) = bottomLeft[i], topRight[i]
            for j in range(i + 1, n):
                (x3, y3), (x4, y4) = bottomLeft[j], topRight[j]
                s = max(0, min(intersection(x1, x2, x3, x4), intersection(y1, y2, y3, y4)))
                ans = max(ans, s * s)
        return ans
```

## 3048. Earliest Second to Mark Indices I

### Solution 1:  sortedlist, greedy, simulation, O(nmlog(n))

```py
from sortedcontainers import SortedList
class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        changeIndices = [x - 1 for x in changeIndices]
        n, m = len(nums), len(changeIndices)
        index = [[] for _ in range(n)]
        marked = [0] * n
        for i in range(m):
            index[changeIndices[i]].append(i)
        sl = SortedList()
        for i in range(n):
            try:
                marked[i] = index[i].pop()
                sl.add((marked[i], i))
            except:
                return -1
        def search():
            for i in reversed(range(m)):
                prev = bag = 0
                for x, j in sl:
                    delta = x - prev
                    bag += delta
                    bag -= nums[j]
                    prev = x + 1
                    if bag < 0: return i + 2
                idx = changeIndices[i]
                sl.remove((marked[idx], idx))
                # need to process in order
                try:
                    marked[idx] = index[idx].pop()
                    sl.add((marked[idx], idx))
                except:
                    return i + 1
            return -1
        ans = search()
        return ans if 0 <= ans <= m else -1
```

### Solution 2:  binary search, O((n + m)log(m)), 

FFFFFT, return T

```py
class Solution:
    def earliestSecondToMarkIndices(self, nums: List[int], changeIndices: List[int]) -> int:
        changeIndices = [x - 1 for x in changeIndices]
        n, m = len(nums), len(changeIndices)
        left, right = 0, m + 1
        def possible(target):
            marked = [-1] * n
            for i in range(target):
                marked[changeIndices[i]] = i
            if any(m == -1 for m in marked): return False
            prev = bag = 0
            for i in sorted(range(n), key = lambda i: marked[i]):
                delta = marked[i] - prev
                bag += delta
                bag -= nums[i]
                prev = marked[i] + 1
                if bag < 0: return False
            return True
        while left < right:
            mid = (left + right) >> 1
            if possible(mid):
                right = mid
            else:
                left = mid + 1
        return left if left <= m else -1
```

## 3048. Earliest Second to Mark Indices II

### Solution 1: 

```py

```

