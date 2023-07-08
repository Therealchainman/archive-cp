# Leetcode Biweekly Contest 108

## 2765. Longest Alternating Subarray

### Solution 1:  sliding window

```py
class Solution:
    def alternatingSubarray(self, nums: List[int]) -> int:
        n = len(nums)
        res = -1
        for i in range(1, n):
            cur = nums[i] - nums[i - 1]
            if cur != 1: continue
            res = max(res, 2)
            for j in range(i + 1, n):
                if cur > 0 and nums[j] - nums[j - 1] == -1:
                    cur = nums[j] - nums[j - 1]
                    res = max(res, j - i + 2)
                elif cur < 0 and nums[j] - nums[j - 1] == 1:
                    cur = nums[j] - nums[j - 1]
                    res = max(res, j - i + 2)
                else: break
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

## 2768. Number of Black Blocks

### Solution 1:  set + brute force submatrices

```py
class Solution:
    def countBlackBlocks(self, m: int, n: int, coordinates: List[List[int]]) -> List[int]:
        R, C = m, n
        n = len(coordinates)
        coordinates = set([(r, c) for r, c in coordinates])
        neighborhood = lambda r, c: [(r - 1, c), (r - 1, c - 1), (r, c - 1), (r, c)]
        neighborhood2 = lambda r, c: [(r + 1, c), (r + 1, c + 1), (r, c + 1), (r, c)]
        in_bounds = lambda r, c: 0 <= r < R and 0 <= c < C
        in_bounds2 = lambda r, c: 0 <= r < R - 1 and 0 <= c < C - 1
        total = (R - 1) * (C - 1)
        vis = set()
        counts = [0] * 5
        black_rocks = lambda r, c: sum(1 for r, c in neighborhood2(r, c) if in_bounds(r, c) and (r, c) in coordinates)
        for r, c in coordinates:
            for nr, nc in neighborhood(r, c):
                if not in_bounds2(nr, nc) or (nr, nc) in vis: continue
                vis.add((nr, nc))
                cnt = black_rocks(nr, nc)
                counts[cnt] += 1
        counts[0] = total - sum(counts)
        return counts
```