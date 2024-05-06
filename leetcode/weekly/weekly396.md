# Leetcode Weekly Contest 396

## 3138. Minimum Length of Anagram Concatenation

### Solution 1:  factorization, highly composite number, frequency count

```py
class Solution:
    def minAnagramLength(self, s: str) -> int:
        n = len(s)
        div = []
        for x in range(1, int(math.sqrt(n)) + 1):
            if n % x == 0: 
                div.append(x)
                if n // x != x: div.append(n // x)
        div.sort()
        unicode = lambda ch: ord(ch) - ord("a")
        def possible(target):
            freq = [0] * 26
            for i in range(target):
                freq[unicode(s[i])] += 1
            for i in range(target, n):
                if i % target == 0: nfreq = [0] * 26
                v = unicode(s[i])
                nfreq[v] += 1
                if nfreq[v] > freq[v]: return False
            return True
        for x in div:
            if possible(x): return x
        return n
```

## 3139. Minimum Cost to Equalize Array

### Solution 1: 

```py

```