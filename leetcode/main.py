from typing import *

class Solution:
    def maximumOr(self, nums: List[int], k: int) -> int:
        n = len(nums)
        pref_xor, suf_xor = 0, [0]*(n+1)
        for i in range(n - 1, -1, -1):
            suf_xor[i] = suf_xor[i+1] | nums[i]
        res = 0
        for i in range(n):
            val = nums[i] << k
            res = max(res, pref_xor | val | suf_xor[i+1])
            pref_xor |= nums[i]
        return res
            