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
    def RMD(self, nums, k): # RMD
        n = len(nums)
        if k == 0: return 0
        if k > n: return math.inf
        psum = 0
        for i in range(k):
            psum += abs(nums[k // 2] - nums[i])
        ans = psum
        med_i = k // 2
        for i in range(k, n):
            delta = nums[med_i + 1] - nums[med_i]
            psum += delta
            psum += (k // 2) * delta
            v = k // 2 - (1 if k % 2 == 0 else 0)
            psum -= v * delta
            med_i += 1
            psum += nums[i] - nums[med_i] # element added into window
            psum -= nums[med_i] - nums[i - k] # element removed from window
            ans = min(ans, psum) 
        return ans

    def minimumMoves(self, nums: List[int], k: int, maxChanges: int) -> int:
        n = len(nums)
        ans = math.inf
        nums = [i for i, x in enumerate(nums) if x]
        L, R = max(0, min(k, maxChanges) - 3), min(k, maxChanges)
        for m in range(L, R + 1):
            ans = min(ans, self.RMD(nums, k - m) + 2 * m)
        return ans
```

