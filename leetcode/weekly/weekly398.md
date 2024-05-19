# Leetcode Weekly Contest 398

## 3152. Special Array II

### Solution 1:  sort, offline query, line sweep

```py
class Solution:
    def isArraySpecial(self, nums: List[int], queries: List[List[int]]) -> List[bool]:
        n = len(nums)
        q = len(queries)
        ans = [True] * q
        events, on_stack = [], []
        for i, (l, r) in enumerate(queries):
            events.append((l, i, 0))
            events.append((r, i, 1))
        events.sort()
        idx = 0
        vis = [0] * q
        for i in range(n):
            if i > 0 and nums[i] % 2 == nums[i - 1] % 2:
                while on_stack:
                    pr = on_stack.pop()
                    if vis[pr]: ans[pr] = False
            while idx < len(events) and events[idx][0] == i:
                s, j, ev = events[idx]
                if ev == 0: 
                    on_stack.append(j)
                    vis[j] = 1
                else: vis[j] = 0 # no longer visited
                idx += 1
        return ans
```

## 3153. Sum of Digit Differences of All Pairs

### Solution 1:  frequency of digit at each position

```py
class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        m = len(str(nums[0]))
        freq = [[0] * 10 for _ in range(m)]
        for num in nums:
            for i, dig in enumerate(map(int, str(num))):
                freq[i][dig] += 1
        ans = 0
        for num in nums:
            for i, dig in enumerate(map(int, str(num))):
                for f in range(10):
                    if f == dig: continue
                    ans += freq[i][f]
                freq[i][dig] -= 1
        return ans
```

## 3154. Find Number of Ways to Reach the K-th Stair

### Solution 1:  dynamic programming, count

```py
class Solution:
    def waysToReachStair(self, k: int) -> int:
        dp = Counter({(1, 0, 0): 1})
        ans = 0
        while dp:
            ndp = Counter()
            for (a, b, c), v in dp.items():
                if a > k + 5: continue
                if a == k: ans += v
                if c == 0: ndp[(a - 1, b, 1)] += v
                ndp[(a + 2 ** b, b + 1, 0)] += v
            dp = ndp
        return ans
```