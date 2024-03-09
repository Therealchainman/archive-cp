# Leetcode Weekly Contest 387

## 3070. Count Submatrices with Top-Left Element and Sum Less Than k

### Solution 1:  columnwise 2d prefix sum

```py
class Solution:
    def countSubmatrices(self, grid: List[List[int]], k: int) -> int:
        R, C = len(grid), len(grid[0])
        ps = [[0] * C for _ in range(R)]
        # build columnwise prefix sum
        for r, c in product(range(R), range(C)):
            ps[r][c] = grid[r][c]
            if r > 0: ps[r][c] += ps[r - 1][c]
        ans = 0
        for r in range(R):
            sum_ = 0
            for c in range(C):
                sum_ += ps[r][c]
                if sum_ > k: break
                ans += 1
        return ans
```

## 3071. Minimum Operations to Write the Letter Y on a Grid

### Solution 1:  counter, matrix

```py
class Solution:
    def minimumOperationsToWriteY(self, grid: List[List[int]]) -> int:
        N = len(grid)
        ycounts, counts = [0] * 3, [0] * 3
        for r, c in product(range(N), repeat = 2):
            if (r >= N // 2 and c == N // 2) or (r < N // 2 and c in (r, N - r - 1)): ycounts[grid[r][c]] += 1
            else: counts[grid[r][c]] += 1
        ysum, sum_ = sum(ycounts), sum(counts)
        ans = math.inf
        for i, j in product(range(3), repeat = 2):
            if i == j: continue # set y to be i, and complement of y to be j
            ans = min(ans, ysum - ycounts[i] + sum_ - counts[j])
        return ans
```

## 3072. Distribute Elements Into Two Arrays II

### Solution 1:  coordinate compression fenwick tree, range counts queries

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

    def query_range(self, i, j):
        return self.query(j) - self.query(i - 1)

    def __repr__(self):
        return f"array: {self.sums}"

class Solution:
    def resultArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        coords = {}
        for v in sorted(nums):
            if v not in coords: coords[v] = len(coords) + 1
        m = len(coords)
        fenwicks = list(map(lambda _: FenwickTree(m), range(2)))
        arrs = [[] for _ in range(2)]
        for i in range(2):
            arrs[i].append(nums[i])
            fenwicks[i].update(coords[nums[i]], 1)
        for val in nums[2:]:
            l, r = coords[val], len(coords)
            counts = [fenwicks[i].query_range(l + 1, r) for i in range(2)]
            if counts[0] != counts[1]:
                idx = int(counts[0] < counts[1])
            else:
                idx = int(len(arrs[0]) > len(arrs[1]))
            arrs[idx].append(val)
            fenwicks[idx].update(l, 1)
        return sum(arrs, [])
```

