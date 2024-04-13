# Leetcode Weekly Contest 389

## 3084. Count Substrings Starting and Ending with Given Character

### Solution 1: counting, number of ways to pick 2 elements from n

```py
class Solution:
    def countSubstrings(self, s: str, c: str) -> int:
        n = s.count(c)
        return n * (n + 1) // 2
```

## 3085. Minimum Deletions to Make String K-Special

### Solution 1: frequency array, precompute prefix, greedy

```py
class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        n = len(word)
        unicode = lambda ch: ord(ch) - ord("a")
        freq = [0] * 26
        for ch in word:
            freq[unicode(ch)] += 1
        ans = n
        cur = 0
        freq.sort()
        for f in freq:
            if f == 0: continue 
            take = 0
            for j in range(26):
                if freq[j] < f: continue
                delta = max(0, freq[j] - f - k)
                take += delta
            ans = min(ans, cur + take)
            cur += f
        return ans
```

## 3086. Minimum Moves to Pick K Ones

### Solution 1:  prefix sum, rolling median deviation, greedy

```py
class Solution:
    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        ans = math.inf
        nums = [i for i, x in enumerate(nums) if x]
        n = len(nums)
        psum = list(accumulate(nums))
        def sum_(i, j):
            return psum[j] - (psum[i - 1] if i > 0 else 0)
        def deviation(i, j, mid):
            lsum = (mid - i + 1) * nums[mid] - sum_(i, mid)
            rsum = sum_(mid, j) - (j - mid + 1) * nums[mid]
            return lsum + rsum
        def RMD(nums, k): # rolling median deviation
            if k == 0: return 0
            ans = math.inf
            l = 0
            for r in range(k - 1, n):
                mid = (l + r) >> 1
                ans = min(ans, deviation(l, r, mid))
                if k % 2 == 0:
                    ans = min(ans, deviation(l, r, mid + 1))
                l += 1
            return ans
        L, R = max(0, min(k, maxChanges) - 3), min(k, maxChanges)
        for m in range(L, R + 1):
            ans = min(ans, RMD(nums, k - m) + 2 * m)
        return ans
```

