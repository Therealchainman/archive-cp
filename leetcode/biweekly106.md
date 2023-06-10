# Leetcode Biweekly Contest 106

## 2729. Check if The Number is Fascinating

### Solution 1: string + set

```py
class Solution:
    def isFascinating(self, n: int) -> bool:
        num = str(n) + str(2 * n) + str(3 * n)
        return len(set(num)) == 9 and len(num) == 9 and '0' not in set(num)
```

## 2730. Find the Longest Semi-Repetitive Substring

### Solution 1:  brute force + loops + substrings + O(n^3)

```py
class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        res = 0
        for i in range(n):
            for j in range(i + 1, n + 1):
                pairs = 0
                for k in range(i + 1, j):
                    if s[k - 1] == s[k]:
                        pairs += 1
                if pairs <= 1:
                    res = max(res, j - i)
        return res
```

## 2731. Movement of Robots

### Solution 1:

```py

```

## 2732. Find a Good Subset of the Matrix

### Solution 1:  greedy + bitmask

If a good subset exists, than there will also exist a good subset with 1 or 2 rows.  So just need to check for good subsets of length 1 if all columns are 0, and subsets of length 2 are good by using bitmask, that is there cannot be two 1s in both columns, must be at most 1 1 in a column

```py
class Solution:
    def goodSubsetofBinaryMatrix(self, grid: List[List[int]]) -> List[int]:
        R, C = len(grid), len(grid[0])
        for r, row in enumerate(grid):
            if sum(row) == 0: return [r]
        states = {}
        for r, row in enumerate(grid):
            mask = sum(1 << c for c, val in enumerate(row) if val)
            for pmask in range(1 << C):
                if pmask & mask: continue
                if pmask in states:
                    return [states[pmask], r]
            states[mask] = r
        return []
```