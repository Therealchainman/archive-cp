# Leetcode Weekly Contest 367

## 2904. Shortest and Lexicographically Smallest Beautiful String

### Solution 1:  sliding window

```py
class Solution:
    def shortestBeautifulSubstring(self, s: str, k: int) -> str:
        res = ""
        mx = math.inf
        n = len(s)
        wcount = j = 0
        for i in range(n):
            wcount += s[i] == "1"
            while wcount >= k:
                if i - j + 1 < mx:
                    mx = i - j + 1
                    res = s[j : i + 1]
                elif i - j + 1 == mx:
                    res = min(res, s[j : i + 1])
                wcount -= s[j] == "1"
                j += 1
        return res
```

## 2905. Find Indices With Index and Value Difference II

### Solution 1:  offline query, sort, sliding window, min and max

```py
class Solution:
    def findIndices(self, nums: List[int], id: int, vd: int) -> List[int]:
        n = len(nums)
        queries = sorted([(v, i) for i, v in enumerate(nums)])
        window = deque()
        first = math.inf
        last = -math.inf
        for v, i in queries:
            window.append((v, i))
            while window and v - window[0][0] >= vd:
                pv, pi = window.popleft()
                first = min(first, pi)
                last = max(last, pi)
            if i - first >= id: return [first, i]
            if last - i >= id: return [i, last]
        return [-1, -1]
```

## 2906. Construct Product Matrix

### Solution 1:  prefix product, suffix product

```py
class Solution:
    def constructProductMatrix(self, grid: List[List[int]]) -> List[List[int]]:
        R, C = len(grid), len(grid[0])
        mod = 12345
        pprod = 1
        sprod = [1] * (R * C + 1)
        i = -2
        for r in reversed(range(R)):
            for c in reversed(range(C)):
                sprod[i] = (sprod[i + 1] * grid[r][c]) % mod
                i -= 1
        i = 1
        for r, c in product(range(R), range(C)):
            nprod = (pprod * grid[r][c]) % mod
            grid[r][c] = (pprod * sprod[i]) % mod
            pprod = nprod
            i += 1
        return grid
```