# Leetcode Weekly Contest 121

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

## 2999. Count the Number of Powerful Integers

### Solution 1:  digit dp

```py
class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        # states (index, suffix, tight)
        # suppose n1 = 2, n2 = 2 then 2 - 2 = 0
        # suppose n1 = 2, n2 = 3, then 3 - 2 = 1 if i >= n2 - n1
        # then if n1 = 2, n2 = 4,, and 4 - 2 = 2, and i = 2, then you want -(4 - 2) = -2, or -(n2 - i) is the reversed index, cause it will increase as i increases, and move farther to the back of the suffix string.
        # n2 >= n1
        def solve(upper):
            n1, n2 = len(s), len(upper)
            if n2 < n1: return 0
            dp = Counter({(1, 1): 1})
            for i, d in enumerate(map(int, upper)):
                ndp = Counter()
                for (suffix, tight), cnt in dp.items():
                    for dig in range(limit + 1 if not tight else min(limit, d) + 1):
                        nsuffix = suffix and dig == int(s[-(n2 - i)]) if i >= n2 - n1 else suffix
                        ntight = tight and dig == d
                        ndp[(nsuffix, ntight)] += cnt
                dp = ndp
            return dp[(1, 0)] + dp[(1, 1)]
        return solve(str(finish)) - solve(str(start - 1))
```

