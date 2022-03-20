# 5. Longest Palindromic Substring

## Solution: expanding about the center of each potential palindrome

This algorithm runs in O(n^2) it is an intermediate algorithm. 

These centers are useful because you can solve in constant space. 

But this can be also solved with dynamic programming

```py
class Solution:
    def longestPalindrome(self, s: str) -> str:
        n = len(s)
        def expand(i,j):
            while i>=0 and j<n and s[i]==s[j]:
                i-=1
                j+=1
            return j-i, i+1, j
        longest = 0
        indices = None
        for i in range(n):
            plen, j, k = expand(i,i)
            if plen > longest:
                longest = plen
                indices = (j,k)
            plen, j, k = expand(i,i+1)
            if plen > longest:
                longest = plen
                indices = (j,k)
        return s[indices[0]:indices[1]]
```

recursive DP doesn't work I get MLE in python.  