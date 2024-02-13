# Leetcode Weekly Contest 384

## 3033. Modify the Matrix

### Solution 1:  max column

```py
class Solution:
    def modifiedMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        R, C = len(mat), len(mat[0])
        maxcol = [0] * C
        for r, c in product(range(R), range(C)):
            maxcol[c] = max(maxcol[c], mat[r][c])
        for r, c in product(range(R), range(C)):
            if mat[r][c] == -1: mat[r][c] = maxcol[c]
        return mat
```

## 3035. Maximum Palindromes After Operations

### Solution 1:  sort, greedy, count pairs and singles of characters

```py
class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        n = len(words)
        ans = 0
        freq = Counter()
        for w in words: freq.update(Counter(w))
        pairs = sum(v // 2 for v in freq.values())
        singles = sum(1 for v in freq.values() if v & 1)
        for w in sorted(words, key = len):
            sz = len(w)
            if sz & 1:
                if singles > 0: singles -= 1
                else: pairs -= 1; singles += 1
            if pairs >= sz // 2:
                pairs -= sz // 2
                ans += 1
        return ans
```

## 3036. Number of Subarrays That Match a Pattern II

### Solution 1:  z algorithm, pattern matching

```py
def z_algorithm(s) -> list[int]:
    n = len(s)
    z = [0]*n
    left = right = 0
    for i in range(1,n):
        # BEYOND CURRENT MATCHED SEGMENT, TRY TO MATCH WITH PREFIX
        if i > right:
            left = right = i
            while right < n and s[right-left] == s[right]:
                right += 1
            z[i] = right - left
            right -= 1
        else:
            k = i - left
            # IF PREVIOUS MATCHED SEGMENT IS NOT TOUCHING BOUNDARIES OF CURRENT MATCHED SEGMENT
            if z[k] < right - i + 1:
                z[i] = z[k]
            # IF PREVIOUS MATCHED SEGMENT TOUCHES OR PASSES THE RIGHT BOUNDARY OF CURRENT MATCHED SEGMENT
            else:
                left = i
                while right < n and s[right-left] == s[right]:
                    right += 1
                z[i] = right - left
                right -= 1
    return z
class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        n, m = len(nums), len(pattern)
        diff = [0] * (n - 1)
        for i in range(n - 1):
            if nums[i + 1] > nums[i]: diff[i] = 1
            elif nums[i + 1] < nums[i]: diff[i] = -1
        encoded = pattern + [2] + diff
        z_arr = z_algorithm(encoded)
        ans = sum(1 for x in z_arr if x == m)
        return ans
```

### Solution 2:  Rolling hash

```py
class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        n, m = len(nums), len(pattern)
        p, MOD = 31, int(1e9)+7
        coefficient = lambda x: x + 2
        pat_hash = 0
        for v in pattern:
            pat_hash = (pat_hash * p + coefficient(v)) % MOD
        diff = [0] * (n - 1)
        for i in range(n - 1):
            if nums[i + 1] > nums[i]: diff[i] = 1
            elif nums[i + 1] < nums[i]: diff[i] = -1
        POW = 1
        for _ in range(m - 1):
            POW = (POW * p) % MOD
        ans = cur_hash = 0
        for i, v in enumerate(diff):
            cur_hash = (cur_hash * p + coefficient(v)) % MOD
            if i >= m - 1:
                if cur_hash == pat_hash: ans += 1
                cur_hash = (cur_hash - coefficient(diff[i - m + 1]) * POW) % MOD
        return ans
```

### Solution 3:  KMP algorithm, prefix array

```py
def kmp(s):
    n = len(s)
    pi = [0] * n
    for i in range(1, n):
        j = pi[i - 1]
        while j > 0 and s[i] != s[j]: 
            j = pi[j - 1]
        if s[j] == s[i]: j += 1
        pi[i] = j
    return pi
class Solution:
    def countMatchingSubarrays(self, nums: List[int], pattern: List[int]) -> int:
        n, m = len(nums), len(pattern)
        diff = [0] * (n - 1)
        for i in range(n - 1):
            if nums[i + 1] > nums[i]: diff[i] = 1
            elif nums[i + 1] < nums[i]: diff[i] = -1
        encoded = pattern + [2] + diff
        parr = kmp(encoded)
        ans = sum(1 for x in parr if x == m)
        return ans
```

