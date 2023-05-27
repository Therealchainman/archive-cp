class Solution:
    def stoneGameII(self, piles: List[int]) -> int:
        n = len(piles)
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                dp[i][j] = dp[i + 1][j] + piles[i]
        for i in range(n - 1, -1, -1):
            for j in range(n - 1, -1, -1):
                for k in range(1, 2 * j + 1):
                    dp[i][j] = max(dp[i][j], dp[i + k][max(j, k)])
        return dp[0][1]