# Leetcode Weekly Contest 353

## 2769. Find the Maximum Achievable Number

### Solution 1:  math

```py
class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
        return num + 2 * t
```

## 2770. Maximum Number of Jumps to Reach the Last Index

### Solution 1:  dynamic programming + O(n^2)

dp[i] = the maximum number of jumps to get to the nums[j]

```py
class Solution:
    def maximumJumps(self, nums: List[int], target: int) -> int:
        n = len(nums)
        dp = [0] + [-math.inf] * (n - 1)
        for j in range(1, n):
            for i in range(j):
                if abs(nums[i] - nums[j]) <= target:
                    dp[j] = max(dp[j], dp[i] + 1)
        return dp[-1] if dp[-1] != -math.inf else -1
```

## 2771. Longest Non-decreasing Subarray From Two Arrays

### Solution 1:  dynamic programming + space optimized

dp1[i] is maximum longest non-decreasing substring ending with nums1[i]
dp2[i] is maximum longest non-decreasing substring ending with nums2[i]

```py
class Solution:
    def maxNonDecreasingLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        dp1 = dp2 = res = 1
        for i in range(1, n1):
            ndp1 = ndp2 = 1
            if nums1[i] >= nums1[i - 1]:
                ndp1 = max(ndp1, dp1 + 1)
            if nums1[i] >= nums2[i - 1]:
                ndp1 = max(ndp1, dp2 + 1)
            if nums2[i] >= nums2[i - 1]:
                ndp2 = max(ndp2, dp2 + 1)
            if nums2[i] >= nums1[i - 1]:
                ndp2 = max(ndp2, dp1 + 1)
            dp1, dp2 = ndp1, ndp2
            res = max(res, dp1, dp2)
        return res
```

## 2772. Apply Operations to Make All Array Elements Equal to Zero

### Solution 1:  difference array + construct array from all 0s + backwards

```py
class Solution:
    def checkArray(self, nums: List[int], k: int) -> bool:
        n = len(nums)
        diff = [0] * (n + 1)
        cur = 0
        for i in range(n):
            cur += diff[i] # update the changes
            delta = nums[i] - cur
            if delta < 0: return False
            if delta > 0 and i + k > n: return False
            if delta > 0:
                cur += delta
                diff[i + k] -= delta
        return True
```