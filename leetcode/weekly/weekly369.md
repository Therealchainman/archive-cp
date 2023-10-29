# Leetcode Weekly Contest 369

## 2918. Minimum Equal Sum of Two Arrays After Replacing Zeros

### Solution 1:  greedy

```py
class Solution:
    def minSum(self, nums1: List[int], nums2: List[int]) -> int:
        n1, n2 = len(nums1), len(nums2)
        c1, c2 = nums1.count(0), nums2.count(0)
        s1, s2 = sum(nums1), sum(nums2)
        if c1 == 0 and c2 == 0 and s1 != s2: return -1
        elif c1 == 0 and s1 < s2 + c2: return -1
        elif c2 == 0 and s2 < s1 + c1: return -1
        return max(s1 + c1, s2 + c2)
```

## 2919. Minimum Increment Operations to Make Array Beautiful

### Solution 1:  dynamic programming + sliding window 

![images](images/Minimum_Increment_Operations_to_Make_Array_Beautiful.png)

```py
class Solution:
    def minIncrementOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        diff = [max(0, k - num) for num in nums]
        dp = diff[:3]
        for i in range(3, n):
            dp[i % 3] = min(dp) + diff[i]
        return min(dp)
```

## 2920. Maximum Points After Collecting Coins From All Nodes

### Solution 1:  depth first search, tree, dynamic programming on a tree

```py
class Solution:
    def maximumPoints(self, edges: List[List[int]], coins: List[int], k: int) -> int:
        n = len(coins)
        adj = [[] for _ in range(n)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        LOG = 15
        def dfs(u, p):
            dp = [-math.inf] * LOG
            dp_childs = [0] * LOG
            child_cnt = 0
            for v in adj[u]:
                if v == p: continue
                child_cnt += 1
                dp_child = dfs(v, u)
                for i in range(LOG):
                    dp_childs[i] += dp_child[i]
            if child_cnt == 0: # initialize everything
                for i in range(LOG):
                    coin = coins[u] // (1 << i)
                    dp[i] = max(coin - k, coin // 2)
            else:
                for i in range(LOG): # not half it
                    coin = coins[u] // (1 << i)
                    dp[i] = max(dp[i], coin - k + dp_childs[i])
                for i in range(LOG): # halved it here
                    coin = coins[u] // (1 << (i + 1))
                    if i + 1 < LOG:
                        dp[i] = max(dp[i], coin + dp_childs[i + 1])
                    else:
                        dp[i] = max(dp[i], coin + dp_childs[i])
            return dp
        res = dfs(0, -1)
        return max(res)
```

