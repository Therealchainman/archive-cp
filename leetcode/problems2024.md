## 1043. Partition Array for Maximum Sum

### Solution 1:  dynammic programming, O(n^2)

dp[i] = maximum sum of partitioning arr[:i + 1] into segments of length at most k when setting the values equal to the max value in each segment. 

For each position i it computes the maximum sum that can be achieved by partitioning the array up to and including the ith element.

Then it increases the size of the current partition that includes i, by moving the j pointer back until it reaches the max size of k.  And it tracks the maximum element in that partition, as that will be the value of all elements in the partition.  And then it computes the value by taking the maximum sum of the partition up to j, and adding the value of the partition to the sum.  And then it updates the dp[i + 1] with the maximum value of the partition.

```py
class Solution:
    def maxSumAfterPartitioning(self, arr: List[int], k: int) -> int:
        n = len(arr)
        dp = [-math.inf] * (n + 1)
        dp[0] = 0
        for i in range(n):
            segmax = -math.inf
            for j in range(i, max(-1, i - k), -1):
                segmax = max(segmax, arr[j])
                dp[i + 1] = max(dp[i + 1], dp[j] + (i - j + 1) * segmax)
        return dp[-1]
```

## 49. Group Anagrams

### Solution 1:  sort, groupby, counter

```py
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        ans = []
        prev = Counter({"i": -1})
        for s in sorted(strs, key = lambda x: sorted(list(x))):
            freq = Counter(s)
            if prev == freq:
                ans[-1].append(s)
            else:
                ans.append([s])
            prev = freq
        return ans
```

```py
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        ans = []
        strs.sort(key = sorted)
        for k, grp in groupby(strs, key = sorted):
            ans.append(list(grp))
        return ans
```

## 368. Largest Divisible Subset

### Solution 1:  sort, dynamic programming, parent array to track best path, backtrack in parent array

```py
class Solution:
    def largestDivisibleSubset(self, nums: List[int]) -> List[int]:
        nums.sort()
        n = len(nums)
        dp = [0] * n
        parent = [-1] * n
        for i in range(n):
            for j in range(i):
                if nums[i] % nums[j] == 0 and dp[j] >= dp[i]:
                    dp[i] = dp[j] + 1
                    parent[i] = j
        ans = []
        i = max(range(n), key = lambda i: dp[i])
        while i != -1:
            ans.append(nums[i])
            i = parent[i]
        return ans
```

## 1463. Cherry Pickup II

### Solution 1:  iterative dp, space optimized, maximize
(column robot 1 occupies, column robot 2 occupies)
And just compute maximum for every possible transition.  

```py
class Solution:
    def cherryPickup(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        dp = [[-math.inf] * C for _ in range(C)]
        dp[0][-1] = grid[0][0] + grid[0][-1]
        in_bounds = lambda c: 0 <= c < C
        for r in range(1, R):
            ndp = [[-math.inf] * C for _ in range(C)]
            for c1, c2 in product(range(C), repeat = 2):
                if dp[c1][c2] == -math.inf: continue
                for nc1, nc2 in product(range(c1 - 1, c1 + 2), range(c2 - 1, c2 + 2)):
                    if not in_bounds(nc1) or not in_bounds(nc2): continue
                    ndp[nc1][nc2] = max(ndp[nc1][nc2], dp[c1][c2] + grid[r][nc1] + (grid[r][nc2] if nc1 != nc2 else 0))
            dp = ndp
        return max(max(row) for row in dp)
```

## 169. Majority Element

### Solution 1:  Boyer-Moore Voting Algorithm

```py
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        n = len(nums)
        ans = cnt = 0
        for num in nums:
            if cnt == 0: ans = num
            if ans == num: cnt += 1
            else: cnt -= 1
        return ans
```

##

### Solution 1:

```py

```

##

### Solution 1:

```py

```

##

### Solution 1:

```py

```

##

### Solution 1:

```py

```

##

### Solution 1:

```py

```

##

### Solution 1:

```py

```