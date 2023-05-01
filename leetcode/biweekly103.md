# Leetcode Biweekly Contest 103

## 2656. Maximum Sum With Exactly K Elements 

### Solution 1:  max + math

Use the summation of natural numbers from 1 + 2 + ... + k - 1

```py
class Solution:
    def maximizeSum(self, nums: List[int], k: int) -> int:
        return k*max(nums) + (k - 1)*k//2
```

## 2657. Find the Prefix Common Array of Two Arrays

### Solution 1: prefix count + frequency array

```py
class Solution:
    def findThePrefixCommonArray(self, A: List[int], B: List[int]) -> List[int]:
        n = len(A)
        counts = [0]*(n + 1)
        res = [0]*n
        prefix_count = 0
        for i in range(n):
            counts[A[i]] += 1
            prefix_count += counts[A[i]] == 2
            counts[B[i]] += 1
            prefix_count += counts[B[i]] == 2
            res[i] = prefix_count
        return res
```

## 2658. Maximum Number of Fish in a Grid

### Solution 1:  dfs + constant space memoization + modify grid in-place

```py
class Solution:
    def findMaxFish(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
        def dfs(r, c):
            stack = [(r, c, grid[r][c])]
            res = 0
            grid[r][c] = 0
            while stack:
                r, c, val = stack.pop()
                res += val
                for nr, nc in [(r + 1, c), (r - 1, c), (r, c - 1), (r, c + 1)]:
                    if not in_bounds(nr, nc) or grid[nr][nc] == 0: continue
                    stack.append((nr, nc, grid[nr][nc]))
                    grid[nr][nc] = 0
            return res
        result = 0
        for r, c in product(range(R), range(C)):
            if grid[r][c] == 0: continue
            result = max(result, bfs(r, c))
        return result
```

## 2659. Make Array Empty

### Solution 1:  modular arithmetic + bit(fenwick tree) + pointer + sort

Use a pointer to track current index and sort the order of index to visit, so then go and visit the index in order.  So you can find the distance between previous and current index in the array.  There are two different cases for when you wrap around and when you don't wrap around in the array. Then need to use a binary indexed tree for range sum queries and range updates that are fast.  It is just going to store the index that have already been removed, because those should be skipped and not counted as an operation.  So you store the count in the bit. 

```py
class FenwickTree:
    def __init__(self, N):
        self.sums = [0 for _ in range(N+1)]

    def update(self, i, delta):
        while i < len(self.sums):
            self.sums[i] += delta
            i += i & (-i)

    def query(self, i):
        res = 0
        while i > 0:
            res += self.sums[i]
            i -= i & (-i)
        return res

    def __repr__(self):
        return f"array: {self.sums}"

class Solution:
    def countOperationsToEmptyArray(self, nums: List[int]) -> int:
        n = len(nums)
        index = sorted(list(range(n)), key = lambda i: nums[i])
        fenwick = FenwickTree(n)
        i = res = 0
        for idx in index:
            delta = 0
            if idx >= i:
                delta = idx - i + 1
                left = fenwick.query(i)
                i += delta
                right = fenwick.query(i)
                i %= n
                seg_sum = right - left
                delta -= seg_sum
            else:
                delta = n - i + idx + 1
                left = fenwick.query(i)
                right = fenwick.query(n)
                left_seg_sum = right - left
                i += delta
                i %= n
                right_seg_sum = fenwick.query(i)
                delta = delta - left_seg_sum - right_seg_sum
            res += delta
            fenwick.update(idx + 1, 1)
        return res
```

### Solution 2:  sort + position + simulation

The idea is that you are simulation how many rounds it takes to remove all the elements.  In this manner each round will have the number of operations be the remaining elements.  So just need to iterate through sorted nums and check if position is less than previous to indicate that it has began a round starting from the beginning that will not need to add the remaining elements to the total operations required.

```py
class Solution:
    def countOperationsToEmptyArray(self, nums: List[int]) -> int:
        n = len(nums)
        pos = {num: i for i, num in enumerate(nums)}
        res = n
        nums.sort()
        for i in range(1, n):
            if pos[nums[i]] < pos[nums[i - 1]]:
                res += n - i
        return res
```