# Leetcode Weekly Contest 356

## 2798. Number of Employees Who Met the Target

### Solution 1:  sum

```py
class Solution:
    def numberOfEmployeesWhoMetTarget(self, hours: List[int], target: int) -> int:
        return sum(1 for h in hours if h >= target)
```

## 2799. Count Complete Subarrays in an Array

### Solution 1:  set + brute force

```py
class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        n = len(nums)
        t = len(set(nums))
        res = 0
        for i in range(n):
            cur = set()
            for j in range(i, n):
                cur.add(nums[j])
                if len(cur) == t: res += 1
        return res
```

## 2800. Shortest String That Contains Three Strings

### Solution 1:  try all permutations

```py

```

## 2801. Count Stepping Numbers in Range

### Solution 1:  digit dp

three tight values, because sometimes you can have a prefix that is larger than current, or smaller or it could be equal.  
0 = smaller, 1 = equal, 2 = larger
Then just need to use correct argument to update the count for states, but also need to track how many of these integers that are smaller than digits which contributes to the number of stepping integers.

For instance if it is a tight bound, and it is smaller than you can move to the smaller
if it is a tight bound, and it is larger than you can move to the larger
if it is a tight bound, and it is equal than you stay with tight bound

if it is smaller than you can add any valid digit that satisfies the constraint.
if it is larger you can add it unless it is for the last digit, then you can't add it because it will be larger than the upper bound.

```py
class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        mod = int(1e9) + 7
        # (last_dig, tight, zero)
        def solve(upper):
            dp = Counter({(0, 1, 1): 1})
            for d in map(int, upper):
                ndp = Counter()
                for (last_dig, tight, zero), cnt in dp.items():
                    for dig in range(10 if not tight else d + 1):
                        if not zero and abs(last_dig - dig) != 1: continue
                        ntight, nzero = tight and dig == d, zero and dig == 0
                        ndp[(dig, ntight, nzero)] = (ndp[(dig, ntight, nzero)] + cnt) % mod
                dp = ndp
            return sum(dp[(dig, t, 0)] for dig, t in product(range(10), range(2))) % mod
        low_is_stepping_int = all(abs(x - y) == 1 for x, y in zip(map(int ,low), map(int, low[1:])))
        return (solve(high) - solve(low) + low_is_stepping_int) % mod
```