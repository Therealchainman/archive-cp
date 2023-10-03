# Leetcode Weekly Contest 111

## 2824. Count Pairs Whose Sum is Less than Target

### Solution 1: 

```py
class Solution:
    def countPairs(self, nums: List[int], target: int) -> int:
        res = 0
        n = len(nums)
        for i in range(n):
            for j in range(i + 1, n):
                res += nums[i] + nums[j] < target
        return res
```

## 2825. Make String a Subsequence Using Cyclic Increments

### Solution 1: 

```py
class Solution:
    def canMakeSubsequence(self, str1: str, str2: str) -> bool:
        n1, n2 = len(str1), len(str2)
        unicode = lambda ch: ord(ch) - ord('a')
        i = 0
        for ch in str2:
            while i < n1 and str1[i] != ch and chr(((unicode(str1[i]) + 1) % 26) + ord('a')) != ch:
                i += 1
            if i == n1: return False
            i += 1
        return True
```

## 2826. Sorting Three Groups

### Solution 1: 

```py
class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        psum = [[0] * 4 for _ in range(n + 1)]
        for i in range(n):
            psum[i + 1] = psum[i].copy()
            psum[i + 1][nums[i]] += 1
        res = n
        for i in range(n + 1):
            nones = psum[i][2] + psum[i][3]
            for j in range(i, n + 1):
                ntwos = psum[j][1] + psum[j][3] - psum[i][1] - psum[i][3]
                nthrees = psum[-1][1] + psum[-1][2] - psum[j][1] - psum[j][2]
                res = min(res, nones + ntwos + nthrees)
        return res
```

## 2827. Number of Beautiful Integers in the Range

### Solution 1:  digit dp

```py

```

