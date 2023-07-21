class Solution:
    def findNumberOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        arr = []
        dp = [Counter() for _ in range(n + 1)]
        dp[0][-math.inf] = 1
        for num in nums:
            i = bisect.bisect_left(arr, num)
            for k, v in dp[i].items():
                if k < num:
                    dp[i + 1][num] += v
            if i == len(arr):
                arr.append(num)
            else:
                arr[i] = num
        print(dp[-1])
        return max(dp[-1].values())