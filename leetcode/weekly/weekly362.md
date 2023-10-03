# Leetcode Weekly Contest 362

## 2848. Points That Intersect With Cars

### Solution 1: 

```py
class Solution:
    def numberOfPoints(self, nums: List[List[int]]) -> int:
        vis = [0] * 101
        for s, e in nums:
            for i in range(s, e + 1):
                vis[i] = 1
        return sum(vis)
```

## 2849. Determine if a Cell Is Reachable at a Given Time

### Solution 1:  math

```py
class Solution:
    def isReachableAtTime(self, sx: int, sy: int, fx: int, fy: int, t: int) -> bool:
        dx, dy = abs(sx - fx), abs(sy - fy)
        x = min(dx, dy) + max(dx - min(dx, dy), dy - min(dx, dy))
        if x == 0:
            return t != 1
        return x <= t
```

## 2850. Minimum Moves to Spread Stones Over Grid

### Solution 1:  grid + backtracking

```py
class Solution:
    def minimumMoves(self, grid: List[List[int]]) -> int:
        n = len(grid)
        res = math.inf
        cur = 0
        manhattan_dist = lambda r1, c1, r2, c2: abs(r1 - r2) + abs(c1 - c2)
        def backtrack(i):
            nonlocal res, cur
            if i == len(cells):
                if all(grid[r][c] == 1 for r, c in product(range(n), repeat = 2)): res = min(res, cur)
                return
            row, col = cells[i]
            for r, c in product(range(n), repeat = 2):
                if (r, c) == (row, col): continue
                if grid[r][c] > 0:
                    dist = manhattan_dist(r, c, row, col)
                    cur += dist
                    grid[r][c] -= 1
                    grid[row][col] += 1
                    backtrack(i + 1)
                    cur -= dist
                    grid[r][c] += 1
                    grid[row][col] -= 1     
        cells = [(r, c) for r, c in product(range(n), repeat = 2) if grid[r][c] == 0]
        backtrack(0)
        return res
```

## 2851. String Transformation

### Solution 1:  matrix exponentiation + kmp + lcp

```py
"""
matrix multiplication with modulus
"""
def mat_mul(mat1, mat2, mod):
    result_matrix = []
    for i in range(len(mat1)):
        result_matrix.append([0]*len(mat2[0]))
        for j in range(len(mat2[0])):
            for k in range(len(mat1[0])):
                result_matrix[i][j] += (mat1[i][k]*mat2[k][j])%mod
    return result_matrix

"""
matrix exponentiation with modulus
matrix is represented as list of lists in python
"""
def mat_pow(matrix, power, mod):
    if power<=0:
        print('n must be non-negative integer')
        return None
    if power==1:
        return matrix
    if power==2:
        return mat_mul(matrix, matrix, mod)
    t1 = mat_pow(matrix, power//2, mod)
    if power%2 == 0:
        return mat_mul(t1, t1, mod)
    return mat_mul(t1, mat_mul(matrix, t1, mod), mod)

class Solution:
    def numberOfWays(self, s: str, t: str, k: int) -> int:
        n = len(s)
        mod = int(1e9) + 7
        def lcp(pat):
            dp = [0] * n
            j = 0
            for i in range(1, n):
                if pat[i] == pat[j]:
                    j += 1
                    dp[i] = j
                    continue
                while j > 0 and pat[i] != pat[j]:
                    j -= 1
                dp[i] = j
                if pat[i] == pat[j]:
                    j += 1
            return dp
        def kmp(text, pat):
            j = cnt = 0
            for i in range(2 * n - 1):
                while j > 0 and text[i % n] != pat[j]:
                    j = lcp_arr[j - 1]
                if text[i % n] == pat[j]:
                    j += 1
                if j == n:
                    cnt += 1
                    j = lcp_arr[j - 1]
            return cnt
        lcp_arr = lcp(t)
        m = kmp(s, t)
        res = 0
        T = [[max(0, n - m - 1), n - m], [m, max(0, m - 1)]]
        B = [[int(s != t)], [int(s == t)]]
        T_k = mat_pow(T, k, mod)
        M = mat_mul(T_k, B, mod)
        res = M[1][0]
        return res
```

