# Leetcode Weekly Contest 397

## 3147. Taking Maximum Energy From the Mystic Dungeon

### Solution 1:  dynamic programming, kadane's algorithm, take largest amongst last k

```py
class Solution:
    def maximumEnergy(self, energy: List[int], k: int) -> int:
        n = len(energy)
        dp = [0] * n
        for i in range(n):
            dp[i] = max(energy[i], energy[i] + (dp[i - k] if i >= k else 0))
        return max(dp[-k:])
```

## 3148. Maximum Difference Score in a Grid

### Solution 1:  dynamic programming, matrix, row and column max

```py
class Solution:
    def maxScore(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        ans = -math.inf
        rmax = [-math.inf] * R
        cmax = [-math.inf] * C
        for r, c in product(range(R), range(C)):
            mx = max(rmax[r], cmax[c])
            cur = mx + grid[r][c]
            ans = max(ans, cur)
            cur = max(-grid[r][c], mx)
            rmax[r] = max(rmax[r], cur)
            cmax[c] = max(cmax[c], cur)
        return ans
```

## 3149. Find the Minimum Cost Array Permutation

### Solution 1:  recursive, bitmask, dp, tracking

```py
class Solution:
    def findPermutation(self, nums: List[int]) -> List[int]:
        n = len(nums)
        ptr = [[-1] * n for _ in range(1 << n)]
        dp = [[math.inf] * n for _ in range(1 << n)]
        def dfs(mask, p):
            if mask.bit_count() == n: return abs(nums[0] - p)
            if dp[mask][p] == math.inf:
                for i in range(n):
                    if (mask >> i) & 1: continue
                    val = abs(nums[i] - p) + dfs(mask | (1 << i), i)
                    if val < dp[mask][p]:
                        dp[mask][p] = val
                        ptr[mask][p] = i
            return dp[mask][p]
        dfs(1, 0)
        ans = [0]
        mask = 1
        for _ in range(n - 1):
            ans.append(ptr[mask][ans[-1]])
            mask |= (1 << ans[-1])
        return ans
```