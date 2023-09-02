# Leetcode Weekly Contest 347

## 2710. Remove Trailing Zeros From a String

### Solution 1:  string

```py
class Solution:
    def removeTrailingZeros(self, num: str) -> str:
        x = int(num[::-1])
        return str(x)[::-1]
```

## 2711. Difference of Number of Distinct Values on Diagonals

### Solution 1:  matrix hash with r - c key + counter for top and bottom diagonals for each diagonal

```py
class Solution:
    def differenceOfDistinctValues(self, grid: List[List[int]]) -> List[List[int]]:
        R, C = len(grid), len(grid[0])
        top_diags, bot_diags = defaultdict(Counter), defaultdict(Counter)
        for r, c in product(range(R), range(C)):
            bot_diags[r - c][grid[r][c]] += 1
        ans = [[0]*C for _ in range(R)]
        for r, c in product(range(R), range(C)):
            v = grid[r][c]
            bot_diags[r - c][v] -= 1
            if bot_diags[r - c][v] == 0: del bot_diags[r - c][v]
            ans[r][c] = abs(len(bot_diags[r - c]) - len(top_diags[r - c]))
            top_diags[r - c][v] += 1
        return ans
```

## 2712. Minimum Cost to Make All Characters Equal

### Solution 1:  prefix and suffix array of difference points + find min(prefix[i] + suffix[i]) when swapping to 1s or 0s

```py
class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        # CONSTRUCT PREFIX AND SUFFIX ARRAY OF DIFFERENCE POINTS
        parr = []
        for i in range(1, n):
            if s[i] != s[i - 1]:
                parr.append(i - 1)
        parr.append(n - 1)
        sarr = []
        for i in range(n - 2, -1, -1):
            if s[i] != s[i + 1]:
                sarr.append(i + 1)
        sarr.append(0)
        sarr = sarr[::-1]
        def prefix(ch):
            dp = [0] * (len(parr) + 1)
            dp[1] = parr[0] + 1 if s[parr[0]] == ch else 0
            for i in range(1, len(parr)):
                idx = parr[i]
                if s[idx] == ch:
                    dp[i + 1] = dp[i] + idx + 1 + parr[i - 1] + 1
                else:
                    dp[i + 1] = dp[i]
            return dp
        def suffix(ch):
            dp = [0]*(len(sarr) + 1)
            dp[-2] = n - sarr[-1] if s[sarr[-1]] == ch else 0
            for i in range(len(sarr) - 2, -1, -1):
                idx = sarr[i]
                if s[idx] == ch:
                    dp[i] = dp[i + 1] + (n - idx) + (n - sarr[i + 1])
                else:
                    dp[i] = dp[i + 1]
            return dp
        pref_cost, suf_cost = prefix('1'), suffix('1') # invert 1s to 0s
        res = math.inf
        for i in range(len(parr)):
            res = min(res, pref_cost[i] + suf_cost[i])
        pref_cost, suf_cost = prefix('0'), suffix('0') # invert 0s to 1s
        for i in range(len(parr)):
            res = min(res, pref_cost[i] + suf_cost[i])
        return res
```

### Solution 2:  observation

If you draw it out can find this pattern, but still haven't proved it. 

```py
class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        return sum(mi-n(i, n - i) for i in range(1, n) if s[i] != s[i - 1])
```

## 2713. Maximum Strictly Increasing Cells in a Matrix

### Solution 1:  dynamic programming + start with largest value and work way backwards and take size for each row and column + sort coordinates by value

```py
class Solution:
    def maxIncreasingCells(self, mat: List[List[int]]) -> int:
        R, C = len(mat), len(mat[0])
        row_size, col_size = [0]*R, [0]*C
        prev_row_size, prev_col_size = [0]*R, [0]*C
        prev_row, prev_col = [-math.inf]*R, [-math.inf]*C
        prev_prev_row, prev_prev_col = [-math.inf]*R, [-math.inf]*C
        coords = sorted([(r, c) for r, c in product(range(R), range(C))], key = lambda x: mat[x[0]][x[1]], reverse = True)
        for r, c in coords:
            v = mat[r][c]
            rsize, csize = row_size[r], col_size[c]
            if v == prev_row[r]:
                if prev_prev_row[r] != -math.inf:
                    rsize = prev_row_size[r]
                else:
                    rsize = 0
            if v == prev_col[c]:
                if prev_prev_col[c] != -math.inf:
                    csize = prev_col_size[c]
                else:
                    csize = 0
            size = max(rsize, csize) + 1
            if v != prev_row[r]:
                prev_row_size[r] = row_size[r]
                prev_prev_row[r] = prev_row[r]
                prev_row[r] = v
            if v != prev_col[c]:
                prev_col_size[c] = col_size[c]
                prev_prev_col[c] = prev_col[c]
                prev_col[c] = v
            row_size[r] = max(row_size[r], size)
            col_size[c] = max(col_size[c], size)
        return max(max(row_size), max(col_size))
```
