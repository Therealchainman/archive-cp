from itertools import product
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
            return sum(dp[n][j][t] for j, t in product(range(min_sum, max_sum + 1), range(2)) % mod
        num1, num2 = int(num1), int(num2)
        return (f(num2) - f(num1 - 1) + mod) % mod