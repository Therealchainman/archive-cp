from typing import *

class Solution:
    def new21Game(self, n: int, k: int, m: int) -> float:
        window_sum = 0
        dp = [0]*(n + 1)
        dp[0] = 1
        for i in range(1, n + 1):
            if i - m - 1 >= 0:
                window_sum = max(0, window_sum - dp[i - m - 1])
            dp[i] = window_sum/m
            if i < k:
                window_sum += dp[i]
        return sum(dp[k:])