# Leetcode Weekly Contest 289

## Summary

## 2243. Calculate Digit Sum of a String

### Solution 1: generator to yield the sum of every k digits 

```py
class Solution:
    def digitSum(self, s: str, k: int) -> str:
        def get_digit_sum(digits):
            for i in range(0,len(digits),k):
                yield sum(map(int, digits[i:i+k]))
        while len(s) > k:
            s = "".join(map(str, get_digit_sum(s)))
        return s
```

## 2244. Minimum Rounds to Complete All Tasks

### Solution 1: Counter + hash table

```py
class Solution:
    def minimumRounds(self, tasks: List[int]) -> int:
        counter = Counter(tasks)
        if any(cnt==1 for cnt in counter.values()):
            return -1
        return sum(cnt//3 if cnt%3==0 else cnt//3+1 for cnt in counter.values())
```

##

### Solution 1:

```py

```

##

### Solution 1:

```py

```