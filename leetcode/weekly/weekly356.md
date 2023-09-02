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
        def f(digits):
            # dp(digit, tight)
            if len(digits) == 1: return int(digits[0])
            dp = [[0] * 3 for _ in range(10)]
            for i in range(1, 10):
                if i == int(digits[0]): 
                    dp[i][1] += 1
                elif i < int(digits[0]):
                    dp[i][0] += 1
                else:
                    dp[i][2] += 1
            res = 9
            for i in range(1, len(digits)):
                ndp = [[0] * 3 for _ in range(10)]
                cur_dig = int(digits[i])
                for dig, tight in product(range(10), range(3)):
                    for j in range(10):
                        if tight == 1 and j != cur_dig and abs(j - dig) == 1: 
                            if j < cur_dig:
                                ndp[j][0] += dp[dig][tight]
                            elif i < len(digits) - 1:
                                ndp[j][2] += dp[dig][tight]
                        if tight == 1 and j == cur_dig and abs(j - dig) == 1:
                            ndp[j][1] += dp[dig][tight]
                        if tight == 0 and abs(j - dig) == 1:
                                ndp[j][0] += dp[dig][tight]
                        if tight == 2 and abs(j - dig) == 1 and i < len(digits) - 1:
                            ndp[j][2] += dp[dig][tight]
                dp = ndp
                res += sum(sum(row) for row in dp)
                res %= mod
            return res
        return (f(high) - f(low) + all(abs(y - x) == 1 for x, y in zip(map(int, low), map(int, low[1:]))) + mod) % mod
```