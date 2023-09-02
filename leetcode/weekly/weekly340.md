# Leetcode Weekly Contest 340

## 2614. Prime In Diagonal

### Solution 1:  matrix + prime + math + O(sqrt(n)) primality check

```py
class Solution:
    def diagonalPrime(self, nums: List[List[int]]) -> int:
        n = len(nums)
        memo = {}
        def is_prime(x: int) -> bool:
            if x in memo: return memo[x]
            if x < 2: return False
            for i in range(2, int(math.sqrt(x)) + 1):
                if x % i == 0: return False
            return True
        res = 0
        for i in range(n):
            if is_prime(nums[i][i]):
                res = max(res, nums[i][i])
            if is_prime(nums[i][~i]):
                res = max(res, nums[i][~i])
        return res
```

### Solution 2:  Sieve of Eratosthenes + precompute primality

```py

```

## 2615. Sum of Distances

### Solution 1:  prefix + suffix sums and counts + line sweep + hash table

```py
class Solution:
    def distance(self, nums: List[int]) -> List[int]:
        n = len(nums)
        last_index = Counter()
        suffix = Counter()
        suffix_cnt = Counter()
        prefix, pcnter = Counter(), Counter()
        for i, num in enumerate(nums):
            suffix[num] += i
            suffix_cnt[num] += 1
        ans = [0]*n
        for i, num in enumerate(nums):
            delta = i - last_index[num]
            suffix[num] -= delta*suffix_cnt[num]
            prefix[num] += delta*pcnter[num]
            ans[i] = prefix[num] + suffix[num]
            suffix_cnt[num] -= 1
            pcnter[num] += 1
            last_index[num] = i
        return ans
```

## 2616. Minimize the Maximum Difference of Pairs

### Solution 1:  greedy binary search

count every other greedily to check if it has enough pais where it is less than or equal to target.  But just know you have to move iterator two forward, so there is no overlap

```py
class Solution:
    def minimizeMax(self, nums: List[int], p: int) -> int:
        if p == 0: return 0
        n = len(nums)
        nums.sort()
        def possible(target):
            cnt = 0
            i = 1
            while i < n:
                if nums[i] - nums[i - 1] <= target:
                    cnt += 1
                    i += 1
                i += 1
            return cnt >= p
        left, right = 0, nums[-1] - nums[0]
        while left < right:
            mid = (left + right) >> 1
            if not possible(mid):
                left = mid + 1
            else:
                right = mid
        return left
```

## 2617. Minimum Number of Visited Cells in a Grid

### Solution 1:  bfs with boundary or frontier optimization

```py
class Solution:
    def minimumVisitedCells(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        # FRONTIERS
        max_row, max_col = [0]*C, [0]*R
        queue = deque([(0, 0)])
        dist = 1
        while queue:
            for _ in range(len(queue)):
                r, c = queue.popleft()
                if (r, c) == (R - 1, C - 1): return dist
                # RIGHTWARD MOVEMENT    
                for k in range(max(c, max_col[r]) + 1, min(grid[r][c] + c, C - 1) + 1):
                    queue.append((r, k))
                # DOWNWARD MOVEMENT
                for k in range(max(r, max_row[c]) + 1, min(grid[r][c] + r, R - 1) + 1):
                    queue.append((k, c))
                # UPDATE FRONTIERS
                max_col[r] = max(max_col[r], grid[r][c] + c)
                max_row[c] = max(max_row[c], grid[r][c] + r)
            dist += 1
        return -1
```

### Solution 2:  sortedlist to track non visited nodes + irange to quickly find next nodes

```py
from sortedcontainers import SortedList
class Solution:
    def minimumVisitedCells(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        rows, cols = [SortedList(range(R)) for _ in range(C)], [SortedList(range(C)) for _ in range(R)]
        queue = deque([(0, 0)])
        dist = 1
        while queue:
            for _ in range(len(queue)):
                r, c = queue.popleft()
                if (r, c) == (R - 1, C - 1): return dist
                # RIGHTWARD MOVEMENT
                for k in list(cols[r].irange(c + 1, grid[r][c] + c)):
                    queue.append((r, k))
                    cols[r].remove(k)
                    rows[k].remove(r)
                # DOWNWARD MOVEMENT
                for k in list(rows[c].irange(r + 1, grid[r][c] + r)):
                    queue.append((k, c))
                    rows[c].remove(k)
                    cols[k].remove(c)
            dist += 1
        return -1
```