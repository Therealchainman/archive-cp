class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        nums.sort()
        res = 0
        for i in range(n):
            if i < nums[i] or i + 1 > nums[i]: res += 1
        return res
