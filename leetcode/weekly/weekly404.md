# Leetcode Weekly Contest 404

## Find the Maximum Length of Valid Subsequence I

### Solution 1:  dynammic programming, binary, base 2

```py
class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        nums = [x % 2 for x in nums]
        dp = [[0] * 2 for _ in range(2)] # (last, rem)
        for i in range(2):
            dp[nums[0]][i] = 1
        for num in nums[1:]:
            ndp = dp[:]
            for rem in range(2): # rem
                ndp[num][rem] = max(ndp[num][rem], dp[rem ^ num][rem] + 1)
            dp = ndp
        return max(max(row) for row in dp)
```

## Find the Maximum Length of Valid Subsequence II

### Solution 1:  dynamic programming

```py
class Solution:
    def maximumLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [[0] * (k + 1) for _ in range(n + 1)] # dp[n][k]
        ans = 0
        for i in range(1, n):
            for j in range(i):
                val = (nums[i] + nums[j]) % k
                dp[i][val] = max(dp[i][val], dp[j][val] + 1, 2)
                ans = max(ans, dp[i][val])
        return ans
```

## 3203. Find Minimum Diameter After Merging Two Trees

### Solution 1:  tree rerooting dp, dfs, tree diameter

```py
class Solution:
    def minimumDiameterAfterMerge(self, edges1: List[List[int]], edges2: List[List[int]]) -> int:
        n, m = len(edges1) + 1, len(edges2) + 1
        ans = 0
        def calc(edges):
            nonlocal ans
            n = len(edges) + 1
            adj = [[] for _ in range(n + 1)]
            st1 = [0] * (n + 1)
            st2 = [0] * (n + 1)
            n1 = [-1] * (n + 1)
            n2 = [-1] * (n + 1)
            par = [0] * (n + 1)
            diam = 0
            res = math.inf
            for u, v in edges:
                adj[u].append(v)
                adj[v].append(u)
            def dfs1(u, p):
                for v in adj[u]:
                    if v == p: continue
                    dfs1(v, u)
                    # update dp[u]
                    if st1[v] + 1 > st1[u]:
                        n2[u] = n1[u]
                        n1[u] = v
                        st2[u] = st1[u]
                        st1[u] = st1[v] + 1
                    elif st1[v] + 1 > st2[u]:
                        st2[u] = st1[v] + 1
                        n2[u] = v
            def dfs2(u, p):
                nonlocal res, diam
                # get answer based on new subtree and subtree of u
                res = min(res, max(par[u], st1[u]))
                diam = max(diam, par[u] + st1[u])
                for v in adj[u]:
                    if v == p: continue
                    # update par[v] based on par[u] and other children
                    par[v] = par[u] + 1
                    if n1[u] != v: par[v] = max(par[v], st1[u] + 1)
                    if n2[u] != v: par[v] = max(par[v], st2[u] + 1)
                    dfs2(v, u)
            dfs1(0, -1)
            dfs2(0, -1)
            ans = max(ans, diam)
            return res
        merged_tree = calc(edges1) + calc(edges2) + 1
        ans = max(ans, merged_tree)
        return ans
```