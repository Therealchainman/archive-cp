# Leetcode Weekly Contest 119

## 2956. Find Common Elements Between Two Arrays

### Solution 1:  counter

```py
class Solution:
    def findIntersectionValues(self, nums1: List[int], nums2: List[int]) -> List[int]:
        def count(nums, other):
            counts = [0] * 101
            for v in other:
                counts[v] = 1
            return sum(1 for num in nums if counts[num])
        ans = [count(nums1, nums2), count(nums2, nums1)]
        return ans
```

## 2957. Remove Adjacent Almost-Equal Characters

### Solution 1:  string

```py
class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        word = list(word)
        res = 0
        unicode = lambda ch: ord(ch) - ord('a')
        difference = lambda c1, c2: abs(unicode(c1) - unicode(c2))
        for i in range(1, n):
            if difference(word[i - 1], word[i]) <= 1:
                res += 1
                word[i] = "#"
        return res
```

## 2958. Length of Longest Subarray With at Most K Frequency

### Solution 1:  sliding window, counter

```py
class Solution:
    def maxSubarrayLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        left = res = 0
        freq = Counter()
        element = None
        for right in range(n):
            freq[nums[right]] += 1
            if freq[nums[right]] > k:
                element = nums[right]
            while element is not None:
                freq[nums[left]] -= 1
                if element == nums[left]: element = None
                left += 1
            res = max(res, right - left + 1)
        return res
```

## 2959. Number of Possible Sets of Closing Branches

### Solution 1:  bit mask, brute force, enumerate all sets, dijkstra, adjacency matrix

```py
class Solution:
    def numberOfSets(self, n: int, maxDistance: int, roads: List[List[int]]) -> int:
        # remove unecessary edges
        adj_mat = [[math.inf] * n for _ in range(n)]
        for u, v, w in roads:
            adj_mat[u][v] = min(adj_mat[u][v], w)
            adj_mat[v][u] = min(adj_mat[v][u], w)
        # enumerate every possible set of nodes
        def check(mask):
            # all pairs shortest distance
            for src in range(n):
                # finds shortest distance from source node to every other node using dijkstra
                if (mask >> src) & 1: continue # skip nodes that are in the removed set
                min_heap = [(0, src)]
                dist = [math.inf] * n
                dist[src] = 0
                while min_heap:
                    d, u = heappop(min_heap)
                    for v in range(n):
                        if (mask >> v) & 1: 
                            dist[v] = 0 # it is removed
                            continue
                        if v == u or adj_mat[u][v] == math.inf: continue
                        if dist[v] > d + adj_mat[u][v]:
                            dist[v] = d + adj_mat[u][v]
                            heappush(min_heap, (d + adj_mat[u][v], v))
                if any(d > maxDistance for d in dist): return False
            return True
        return sum(check(mask) for mask in range(1 << n))
```

