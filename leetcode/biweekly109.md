# Leetcode Biweekly Contest 109

## 6930. Check if Array is Good

### Solution 1:  counts

```py
class Solution:
    def isGood(self, nums: List[int]) -> bool:
        n = max(nums)
        cnt = [0] * (n + 1)
        for num in nums:
            cnt[num] += 1
        return all(cnt[i] == 1 for i in range(1, n)) and cnt[n] == 2
```

## 6926. Sort Vowels in a String

### Solution 1:  sort + string

```py
class Solution:
    def sortVowels(self, s: str) -> str:
        indices, vows = [], []
        vowels = "AEIOUaeiou"
        for i, c in enumerate(s):
            if c in vowels:
                indices.append(i)
                vows.append(c)
        vows.sort()
        res = list(s)
        for i, v in zip(indices, vows):
            res[i] = v
        return ''.join(res)
```

## 6931. Visit Array Positions to Maximize Score

### Solution 1:  dynamic programming

just store the max score so far considering the parity of current element, it has two previous it can come from.

```py
class Solution:
    def maxScore(self, nums: List[int], x: int) -> int:
        if nums[0] & 1:
            odd = nums[0]
            even = -math.inf
        else:
            even = nums[0]
            odd = -math.inf
        for num in nums[1:]:
            if num & 1:
                odd = max(odd + num, even + num - x)
            else:
                even = max(even + num, odd + num - x)
        return max(even, odd)
```

## 6922. Ways to Express an Integer as Sum of Powers

### Solution 1:  dynamic programming + counter

```py
class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        mod = int(1e9) + 7
        dp = Counter({0: 1})
        for i in range(1, n + 1):
            v = pow(i, x)
            if v > n: break
            ndp = dp.copy()
            for k, cnt in dp.items():
                if k + v > n: continue
                ndp[k + v] += cnt
                ndp[k + v] %= mod
            dp = ndp
        return dp[n]
```