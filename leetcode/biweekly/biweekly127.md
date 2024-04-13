# Leetcode BiWeekly Contest 127

## 3096. Minimum Levels to Gain More Points

### Solution 1:  prefix sum, suffix sum

```py
class Solution:
    def minimumLevels(self, arr: List[int]) -> int:
        n = len(arr)
        ssum = 0
        for x in arr:
            if x == 1: ssum += 1
            else: ssum -= 1
        psum = 0
        for i in range(n - 1):
            psum += 1 if arr[i] == 1 else -1
            ssum -= 1 if arr[i] == 1 else -1
            if psum > ssum: return i + 1
        return -1
```

## 3097. Shortest Subarray With OR at Least K II

### Solution 1:  bit manipulation, sliding window, frequency array

```py
class Solution:
    def minimumSubarrayLength(self, nums: List[int], K: int) -> int:
        n = len(nums)
        ans = math.inf
        cur = j = 0
        freq = [0] * 30
        for i in range(n):
            for k in range(30):
                if (nums[i] >> k) & 1:
                    freq[k] += 1
                    if freq[k] == 1: cur |= 1 << k
            while cur >= K and j <= i:
                ans = min(ans, i - j + 1)
                for k in range(30):
                    if (nums[j] >> k) & 1:
                        freq[k] -= 1
                        if freq[k] == 0: cur ^= 1 << k
                j += 1
        return ans if ans < math.inf else -1
```

## 3098. Find the Sum of Subsequence Powers

### Solution 1:  sort, dp, countings

```py
class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        n = len(nums)
        mod = int(1e9) + 7
        nums.sort()
        @cache
        def dp(idx, sz, last, mind):
            if idx == n:
                if sz == k: return mind
                return 0
            ans = 0
            # take
            if sz < k:
                ans = (ans + dp(idx + 1, sz + 1, nums[idx], min(mind, nums[idx] - last))) % mod
            ans = (ans + dp(idx + 1, sz, last, mind)) % mod
            return ans
        return dp(0, 0, -math.inf, math.inf)
```

```py
class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        n = len(nums)
        mod = int(1e9) + 7
        nums.sort()
        dp = Counter({(0, -math.inf, math.inf): 1})
        for x in nums:
            ndp = Counter()
            for (sz, last, mind), cnt in dp.items():
                if sz < k:
                    ndp[(sz + 1, x, min(mind, x - last))] += cnt
                    ndp[(sz + 1, x, min(mind, x - last))] %= mod
                ndp[(sz, last, mind)] += cnt 
                ndp[(sz, last, mind)] %= mod
            dp = ndp
        ans = 0
        for (sz, _, mind), cnt in dp.items():
            if sz == k: ans = (ans + cnt * mind) % mod
        return ans
```

Could use coordinate compression and remove the Counter.
