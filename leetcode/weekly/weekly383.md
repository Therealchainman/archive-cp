# Leetcode Weekly Contest 383

## 3028. Ant on the Boundary

### Solution 1:  simulation

```py
class Solution:
    def returnToBoundaryCount(self, nums: List[int]) -> int:
        ans = pos = 0
        for num in nums:
            pos += num
            if pos == 0: ans += 1
        return ans
```

## 3030. Find the Grid of Region Average

### Solution 1: 

```py

```

## 3031. Minimum Time to Revert Word to Initial State II

### Solution 1:  KMP and prefix array algorithm, divisibility

the index i of the suffix that matches prefix is at i = n - j.  That needs to be divisible by k, 
cause give abacaba and k = 3
you are doing this, cabaxxx at first operation
axxxxxx at the second operation, just need a to match prefix, cause xxxxxxx can be assigned any character we desired.  So easy to make it match the original string. 

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
    def minimumTimeToInitialState(self, word: str, k: int) -> int:
        n = len(word)
        prefix_arr = kmp(word)
        j = prefix_arr[-1]
        while j > 0 and (n - j) % k != 0:
            j = prefix_arr[j - 1]
        return math.ceil((n - j) / k)
```

