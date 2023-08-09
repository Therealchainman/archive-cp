class Solution:
    def minimumTime(self, nums1: List[int], nums2: List[int], x: int) -> int:
        n = len(nums1)
        nums = sorted([(x1, x2) for x1, x2 in zip(nums1, nums2)], key = lambda pair: pair[1])
        print(nums)
        s1, s2 = sum(nums1), sum(nums2)
        dp = [[0] * (n + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        for i, j in product(range(n), repeat = 2):
            dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i][j] + nums[i][0] + j * nums[i][1])
        for i in range(n + 1):
            if s1 + s2 * i - dp[n][i] <= x:
                return i
        return - 1