# Leetcode Biweekly Contest 108

## 2765. Longest Alternating Subarray

### Solution 1:  sliding window

```py
class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        res = -1
        left = 0
        n = len(nums)
        diff = [nums[i] - nums[i - 1] for i in range(1, n)]
        for right in range(n - 1):
            while left < right and diff[left] != 1:
                left += 1
            delta = right - left
            if ((delta & 1) and diff[right] == -1) or (delta % 2 == 0 and diff[right] == 1):
                res = max(res, delta + 2)
            else:
                left = right
        return res
```

## 2766. Relocate Marbles

### Solution 1:  counter

```py
class Solution:
    def relocateMarbles(self, nums: List[int], moveFrom: List[int], moveTo: List[int]) -> List[int]:
        locations = Counter(nums)
        for u, v in zip(moveFrom, moveTo):
            cnt = locations[u]
            locations[u] -= cnt
            locations[v] += cnt
        return sorted([k for k, v in locations.items() if v > 0])
```

## 2767. Partition String Into Minimum Beautiful Substrings

### Solution 1:  dfs + backtrack

```py
class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        n = len(s)
        fives = {1, 5, 25, 125, 625, 3125, 15625}
        substrings = []
        def backtrack(i):
            if i == n: return 0
            res = math.inf
            if s[i] == '0': return res
            for j in range(i, n):
                cand = int(s[i:j+1], 2)
                if cand in fives:
                    substrings.append(cand)
                    res = min(res, backtrack(j+1) + 1)
                    substrings.pop()
            return res
        res = backtrack(0)
        return res if res < math.inf else -1
```

### Solution 2:  dynamic programming + O(n^2)

dp[i] = minimum number of valid partitions ending at character s[i - 1]
base case is dp[0] for empty string and set to 0 partitions

```py
class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        n = len(s)
        fives = {1, 5, 25, 125, 625, 3125, 15625}
        dp = [0] + [math.inf] * n
        for i in range(1, n + 1):
            for j in range(i):
                if s[j] == '0': continue
                cand = int(s[j:i], 2)
                if cand not in fives: continue
                dp[i] = min(dp[i], dp[j] + 1)
        return dp[-1] if dp[-1] != math.inf else -1
```

## 2768. Number of Black Blocks

### Solution 1:  hash table + counters

For each black rock, add it to all the possibly 4 submatrices it can belong within.  

```py
class Solution:
    def countBlackBlocks(self, R: int, C: int, coordinates: List[List[int]]) -> List[int]:
        in_bounds = lambda r, c: 0 <= r < R - 1 and 0 <= c < C - 1
        neighborhood = lambda r, c: [(r - 1, c), (r - 1, c - 1), (r, c - 1), (r, c)]
        black_counter = Counter()
        for r, c in coordinates:
            for nr, nc in neighborhood(r, c):
                if not in_bounds(nr, nc): continue
                cell = nc + nr * C
                black_counter[cell] += 1
        counts = [0] * 5
        for cnt in black_counter.values():
            counts[cnt] += 1
        counts[0] = (R - 1) * (C - 1) - sum(counts)
        return counts
```