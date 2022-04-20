# Leetcode Weekly Contest 289

## Summary

## 2243. Calculate Digit Sum of a String

### Solution 1: generator to yield the sum of every k digits 

```py
class Solution:
    def digitSum(self, s: str, k: int) -> str:
        def get_digit_sum(digits):
            for i in range(0,len(digits),k):
                yield sum(map(int, digits[i:i+k]))
        while len(s) > k:
            s = "".join(map(str, get_digit_sum(s)))
        return s
```

## 2244. Minimum Rounds to Complete All Tasks

### Solution 1: Counter + hash table

```py
class Solution:
    def minimumRounds(self, tasks: List[int]) -> int:
        counter = Counter(tasks)
        if any(cnt==1 for cnt in counter.values()):
            return -1
        return sum(cnt//3 if cnt%3==0 else cnt//3+1 for cnt in counter.values())
```

## 2245. Maximum Trailing Zeros in a Cornered Path

### Solution 1: prefix sum with every pair 2 and 5 contributes to a trailing zero

```py
class Solution:
    def maxTrailingZeros(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        hor_prefix = [[[0,0] for _ in range(C+1)] for _ in range(R)]
        vert_prefix = [[[0,0] for _ in range(C)] for _ in range(R+1)]
        for r, c in product(range(R), range(C)):
            elem = grid[r][c]
            cnt2 = cnt5 = 0
            while elem > 0 and elem%2==0:
                cnt2 += 1
                elem//=2
            while elem > 0 and elem%5==0:
                cnt5 += 1
                elem//=5
            hor_prefix[r][c+1][0] = hor_prefix[r][c][0] + cnt2
            hor_prefix[r][c+1][1] = hor_prefix[r][c][1] + cnt5
            vert_prefix[r+1][c][0] = vert_prefix[r][c][0] + cnt2
            vert_prefix[r+1][c][1] = vert_prefix[r][c][1] + cnt5
        max_zeros = 0
        def pair(A, B):
            return min(A[0]+B[0], A[1]+B[1])
        for r, c in product(range(R), range(C)):
            right, left= list(map(lambda x: x[0]-x[1], zip(hor_prefix[r][C],hor_prefix[r][c]))), hor_prefix[r][c]
            bottom, top = list(map(lambda x: x[0]-x[1], zip(vert_prefix[R][c],vert_prefix[r+1][c]))), vert_prefix[r][c]
            # print(left, right, bottom, top)
            max_zeros = max(max_zeros, pair(left, top), pair(left, bottom), pair(right, top), pair(right, bottom))
        return max_zeros
```

## Solution 2: Numpy + np.cumsum + np.minimum + np.rot90

```py
import numpy as np

class Solution:
    def maxTrailingZeros(self, grid: List[List[int]]) -> int:
        A = np.array(grid)
        def prefix_sums(digit):
            sa = sum(A%digit**i==0 for i in range(1,10))
            return np.cumsum(sa, axis=0) + np.cumsum(sa, axis=1) - sa
        return max(np.minimum(prefix_sums(2), prefix_sums(5)).max() 
                  for _ in range(4) if bool([A := np.rot90(A)]))
```

## 2246. Longest Path With Different Adjacent Characters

### Solution 1:

```py

```