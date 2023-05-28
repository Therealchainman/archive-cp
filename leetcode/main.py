class Solution:
    def minimumCost(self, s: str) -> int:
        n = len(s)
        # CONSTRUCT PREFIX AND SUFFIX ARRAY OF DIFFERENCE POINTS
        parr = []
        for i in range(1, n):
            if s[i] != s[i - 1]:
                parr.append(i - 1)
        parr.append(n - 1)
        sarr = []
        for i in range(n - 2, -1, -1):
            if s[i] != s[i + 1]:
                sarr.append(i + 1)
        sarr.append(0)
        sarr = sarr[::-1]
        def prefix(ch):
            dp = [0] * (len(parr) + 1)
            dp[1] = parr[0] + 1 if s[parr[0]] == ch else 0
            for i in range(1, len(parr)):
                idx = parr[i]
                if s[idx] == ch:
                    dp[i + 1] = dp[i] + idx + 1 + parr[i - 1] + 1
                else:
                    dp[i + 1] = dp[i]
            return dp
        def suffix(ch):
            dp = [0]*(len(sarr) + 1)
            dp[-2] = n - sarr[-1] if s[sarr[-1]] == ch else 0
            for i in range(len(sarr) - 2, -1, -1):
                idx = sarr[i]
                if s[idx] == ch:
                    dp[i] = dp[i + 1] + (n - idx) + (n - sarr[i + 1])
                else:
                    dp[i] = dp[i + 1]
            return dp
        pref_cost, suf_cost = prefix('1'), suffix('1') # invert 1s to 0s
        res = math.inf
        for i in range(len(parr)):
            res = min(res, pref_cost[i] + suf_cost[i])
        pref_cost, suf_cost = prefix('0'), suffix('0') # invert 0s to 1s
        for i in range(len(parr)):
            res = min(res, pref_cost[i] + suf_cost[i])
        return res
