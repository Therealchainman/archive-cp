# Leetcode Weekly Contest 337


## 2595. Number of Even and Odd Bits

### Solution 1:  enumerate through binary representation + track index

```py
class Solution:
    def evenOddBit(self, n: int) -> List[int]:
        res = [0]*2
        for i, dig in enumerate(reversed(bin(n)[2:])):
            if dig == '1':
                res[i&1] += 1
        return res
```

## 2596. Check Knight Tour Configuration

### Solution 1:  hash table + check valid move by min needs be 1 and max needs be 2 amongst two values

```py
class Solution:
    def checkValidGrid(self, grid: List[List[int]]) -> bool:
        n = len(grid)
        if grid[0][0] != 0: return False
        cell_loc = {}
        for r, c in product(range(n), repeat = 2):
            cell_loc[grid[r][c]] = (r, c)
        for i in range(1, n*n):
            pr, pc = cell_loc[i - 1]
            r, c = cell_loc[i]
            dr, dc = abs(r - pr), abs(c - pc)
            if min(dr, dc) != 1 or max(dr, dc) != 2: return False
        return True
```

## 2597. The Number of Beautiful Subsets

### Solution 1:  sort + counter + backtrack + recursion

```py
class Solution:
    def beautifulSubsets(self, nums: List[int], k: int) -> int:
        n = len(nums)
        nums.sort()
        counts = [0]*1001
        cnt = 0
        def backtrack(i):
            nonlocal cnt
            if i == n: return int(cnt > 0)
            res = 0
            res += backtrack(i + 1) # skip
            prv = nums[i] - k
            if counts[prv] == 0:
                counts[nums[i]] += 1
                cnt += 1
                res += backtrack(i + 1)
                counts[nums[i]] -= 1
                cnt -= 1
            return res
        return backtrack(0)
```

## 2598. Smallest Missing Non-negative Integer After Operations

### Solution 1:  modular arithmetic + min with custom key

Find the minimum cnt that is the number of times you can wrap around the values. And then the min_idx is how far get on last wrap around so if wrap around once min_cnt is 1, and there might be min_idx of 2 or something so answer is 1*value + 2

```py
class Solution:
    def findSmallestInteger(self, nums: List[int], value: int) -> int:
        counts = [0]*value
        for v in map(lambda num: num%value, nums):
            counts[v] += 1
        min_idx, min_cnt = min(enumerate(counts), key = lambda pair: (pair[1], pair[0]))
        return value*min_cnt + min_idx
```