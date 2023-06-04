# Leetcode Weekly Contest 348

## 2716. Minimize String Length

### Solution 1:  string + logic

```py
class Solution:
    def minimizedStringLength(self, s: str) -> int:
        return len(set(s))
```

## 2717. Semi-Ordered Permutation

### Solution 1:  math

the number of adjacent swaps to get the integer 1 to the 0 index is basically the index it is currently at. For the integer n, it is how far away it is from the last integer.  But if 1 is to the right of n, then one of the swaps will be to move 1 to the left or n.  In that case subtract 1. 

```py
class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        first, last = nums.index(1), n - nums.index(n) - 1
        return first + last - (1 if nums.index(n) < first else 0)
```

## 2718. Sum of Matrix After Queries

### Solution 1:  matrix + track how many rows and columns have been queried

For this problem, if you go through the queries in reverse then you can compute it in O(n) time.  Because each time you use a row or column, you fill it in with an integer, and never again can you fill in that row that will contribute to the final result. So mark as visited. But also track how many unique rows and columns have been filled with value.  This way when you fill rows, you need to subtract how many columns have been filled up to that point to compute the actual value can get in current query.

```py
class Solution:
    def matrixSumQueries(self, n: int, queries: List[List[int]]) -> int:
        # type, index, value
        res = row_count = col_count = 0
        rows, cols = [0] * n, [0] * n
        for t, i, v in reversed(queries):
            if t == 0:
                if rows[i]: continue
                res += n * v - col_count * v
                row_count += 1
                rows[i] = 1
            else:
                if cols[i]: continue
                res += n * v - row_count * v
                col_count += 1
                cols[i] = 1
        return res
```

## 2719. Count of Integers

### Solution 1:  digit dp

```py

```


