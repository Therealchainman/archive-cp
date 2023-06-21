class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        n = len(nums)
        mod = int(1e9) + 7
        dp = {(1 << i, i): 1 for i in range(n)}
        print(dp)
        for _ in range(n):
            ndp = Counter()
            for (mask, j), v in dp.items():
                for i in range(n):
                    if (k >> i) & 1: continue
                    if nums[i] % nums[j] != 0 and nums[j] % nums[i] != 0: continue
                    nstate = (mask | (1 << i), i)
                    ndp[nstate] = (ndp[nstate] + v) % mod
            dp = ndp
        return sum(dp.values()) % mod