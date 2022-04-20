# Leetcode Biweekly Contest 75

## Summary

## 2220. Minimum Bit Flips to Convert Number

### Solution 1: xor + bit_count

```py
class Solution:
    def minBitFlips(self, start: int, goal: int) -> int:
        return (start ^ goal).bit_count()
```

## 2221. Find Triangular Sum of an Array

### Solution 1: optimized space solution, reuse nums array

```py
class Solution:
    def triangularSum(self, nums: List[int]) -> int:
        for end in range(len(nums)+1)[::-1]:
            for i in range(1, end):
                nums[i-1] = (nums[i-1]+nums[i])%10
        return nums[0]
```

### Solution 2: pairwise to get the two adjacent elements in pair

```py
class Solution:
    def triangularSum(self, nums: List[int]) -> int:
        while len(nums) > 1:
            nums = [(a+b)%10 for a,b in pairwise(nums)]
        return nums[0]
```

## 2222. Number of Ways to Select Buildings

### Solution 1: prefix sums for only 2 patterns 101, 010

```py
class Solution:
    def numberOfWays(self, s: str) -> int:
        n = len(s)
        prefix_zeros, prefix_ones = [0]*(n+1), [0]*(n+1)
        for i in range(n):
            prefix_zeros[i+1] = prefix_zeros[i] + (s[i]=='0')
            prefix_ones[i+1] = prefix_ones[i] + (s[i]=='1')
        ways = 0
        for i in range(n):
            if s[i] == '0':
                ways += prefix_ones[i]*(prefix_ones[-1]-prefix_ones[i])
            if s[i] == '1':
                ways += prefix_zeros[i]*(prefix_zeros[-1]-prefix_zeros[i])
        return ways
```

## 2223. Sum of Scores of Built Strings

### Solution 1: Brute force z-algorithm (z-array)

This will TLE cause it is O(n^2)

```py
class Solution:
    def sumScores(self, s: str) -> int:
        n = len(s)
        z = [0]*n
        for i in range(1, n):
            while z[i]+i < n and s[z[i]+i] == s[z[i]]:
                z[i] += 1
        return sum(z) + n
```

### Solution 2: optimized z-algorithm

```py

```
