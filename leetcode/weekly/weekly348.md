# Leetcode Weekly Contest 348

## 2716. Minimize String Length

### Solution 1:  string + logic

```py
class Solution:
    def minimizedStringLength(self, s: str) -> int:
        return len(set(s))
```

## 2717. Semi-Ordered Permutation

### Solution 1:  math

the number of adjacent swaps to get the integer 1 to the 0 index is basically the index it is currently at. For the integer n, it is how far away it is from the last integer.  But if 1 is to the right of n, then one of the swaps will be to move 1 to the left or n.  In that case subtract 1. 

```py
class Solution:
    def semiOrderedPermutation(self, nums: List[int]) -> int:
        n = len(nums)
        first, last = nums.index(1), n - nums.index(n) - 1
        return first + last - (1 if nums.index(n) < first else 0)
```

## 2718. Sum of Matrix After Queries

### Solution 1:  matrix + track how many rows and columns have been queried

For this problem, if you go through the queries in reverse then you can compute it in O(n) time.  Because each time you use a row or column, you fill it in with an integer, and never again can you fill in that row that will contribute to the final result. So mark as visited. But also track how many unique rows and columns have been filled with value.  This way when you fill rows, you need to subtract how many columns have been filled up to that point to compute the actual value can get in current query.

```py
class Solution:
    def matrixSumQueries(self, n: int, queries: List[List[int]]) -> int:
        # type, index, value
        res = row_count = col_count = 0
        rows, cols = [0] * n, [0] * n
        for t, i, v in reversed(queries):
            if t == 0:
                if rows[i]: continue
                res += n * v - col_count * v
                row_count += 1
                rows[i] = 1
            else:
                if cols[i]: continue
                res += n * v - row_count * v
                col_count += 1
                cols[i] = 1
        return res
```

## 2719. Count of Integers

### Solution 1:  digit dp

dp(i, j, t) = number of integers with i digits, sum of digits is j, t represents tight bound

```py
class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        mod = 10**9 + 7
        def f(num):
            digits = str(num)
            n = len(digits)
            # dp(i, j, t) ith index in digits, j sum of digits, t represents tight bound
            dp = [[[0] * 2 for _ in range(max_sum + 1)] for _ in range(n + 1)]
            for i in range(min(int(digits[0]), max_sum) + 1):
                dp[1][i][1 if i == int(digits[0]) else 0] += 1
            for i, t, j in product(range(1, n), range(2), range(max_sum + 1)):
                for k in range(10):
                    if t and k > int(digits[i]): break
                    if j + k > max_sum: break
                    dp[i + 1][j + k][t and k == int(digits[i])] += dp[i][j][t]
            return sum(dp[n][j][t] for j, t in product(range(min_sum, max_sum + 1), range(2))) % mod
        num1, num2 = int(num1), int(num2)
        return (f(num2) - f(num1 - 1) + mod) % mod
```

```py
class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        mod = int(1e9) + 7
        def solve(upper):
            dp = Counter({(0, 1): 1})
            for d in map(int, upper):
                ndp = Counter()
                for (dig_sum, tight), cnt in dp.items():
                    for dig in range(10 if not tight else d + 1):
                        ndig_sum = dig_sum + dig
                        if ndig_sum > max_sum: break
                        ntight = tight and dig == d
                        ndp[(ndig_sum, ntight)] = (ndp[(ndig_sum, ntight)] + cnt) % mod
                dp = ndp
            return sum(dp[(ds, t)] for ds, t in product(range(min_sum, max_sum + 1), range(2))) % mod
        return (solve(num2) - solve(str(int(num1) - 1))) % mod
```


