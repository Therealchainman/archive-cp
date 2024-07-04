class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        k = 2
        n = len(nums)
        dp = [[0] * k for _ in range(n)] # dp[n][k]
        ans = 0
        for i in range(1, n):
            for j in range(i):
                val = (nums[i] + nums[j]) % k
                dp[i][val] = max(dp[i][val], dp[j][val] + 1, 2)
                ans = max(ans, dp[i][val])
        return ans
