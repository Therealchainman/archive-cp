# Leetcode Weekly Contest 388

## 3075. Maximize Happiness of Selected Children

### Solution 1:  sort, max, sum, greedy

```py
class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        happiness.sort(reverse = True)
        return sum(max(0, happiness[i] - i) for i in range(k))
```

## 3076. Shortest Uncommon Substring in an Array

### Solution 1:  brute force, counter

```py
class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        counts = Counter()
        n = len(arr)
        ans = [""] * n
        for word in arr:
            nw = len(word)
            for i in range(nw + 1):
                for j in range(i):
                    counts[word[j : i]] += 1
        for i in range(n):
            nw = len(arr[i])
            for j in range(nw + 1):
                for k in range(j):
                    counts[arr[i][k : j]] -= 1
            for len_ in range(1, nw + 1):
                for l in range(nw - len_ + 1):
                    r = l + len_
                    cur = arr[i][l : r]
                    if counts[cur] == 0 and (not ans[i] or cur < ans[i]):
                        ans[i] = cur
                if ans[i]: break
            for j in range(nw + 1):
                for k in range(j):
                    counts[arr[i][k : j]] += 1
        return ans
```

## 3077. Maximum Strength of K Disjoint Subarrays

### Solution 1: dp, O(2*n*k)

```py
class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        dp = [[-math.inf] * 2 for _ in range(k + 1)] # (subarray, started)
        # 0 not started, 1 started
        dp[0][0] = 0
        for num in nums:
            ndp = [[-math.inf] * 2 for _ in range(k + 1)]
            for i in range(k + 1):
                # continuation of a subarray
                score = num * (k - i) * (1 if i % 2 == 0 else -1)
                ndp[i][1] = max(ndp[i][1], dp[i][1] + score)
                # start new subarray
                if i > 0: ndp[i][1] = max(ndp[i][1], dp[i - 1][1] + score)
                ndp[i][1] = max(ndp[i][1], dp[i][0] + score)
                # skip element and section
                if i > 0: ndp[i][0] = max(ndp[i][0], dp[i - 1][1])
                ndp[i][0] = max(ndp[i][0], dp[i][0])
            dp = ndp
        return max(dp[-2][1], dp[-1][0])
```

