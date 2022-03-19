# Leetcode Biweekly Contest 74

## 2206. Divide Array Into Equal Pairs

### Solution: 

```py
class Solution:
    def divideArray(self, nums: List[int]) -> bool:
        counts = Counter(nums)
        for cnt in counts.values():
            if cnt%2==1: return False
        return True
```

```py
class Solution:
    def divideArray(self, nums: List[int]) -> bool:
        counts = Counter(nums)
        return not list(filter(lambda cnt: cnt%2==1, counts.values()))
```

## 2207. Maximize Number of Subsequences in a String

### Solution: Greedy 

```py
class Solution:
    def maximumSubsequenceCount(self, text: str, pattern: str) -> int:
        s = "".join([ch for ch in text if ch in pattern])
        def get_max(ss):
            sum_ = cnt = 0
            for ch in ss:
                if ch==pattern[1]:
                    sum_ += cnt
                if ch==pattern[0]:
                    cnt += 1
            return sum_
        return max(get_max(pattern[0] + s), get_max(s + pattern[1]))
```

## 2208. Minimum Operations to Halve Array Sum

### Solution: max heap datastructure

```py
class Solution:
    def halveArray(self, nums: List[int]) -> int:
        nums = [-x for x in nums]
        heapify(nums)
        half = -sum(nums)/2
        cnt = 0
        while half > 0:
            val = -heappop(nums)
            val/=2
            half -= val
            heappush(nums, -val)
            cnt += 1
        return cnt
```

## 2209. Minimum White Tiles After Covering With Carpets

### Solution: recursive DP

```py
class Solution:
    def minimumWhiteTiles(self, floor: str, numCarpets: int, carpetLen: int) -> int:
        n = len(floor)
        suffix = [0]*(n+1)
        for i in range(n-1,-1,-1):
            suffix[i] = suffix[i+1] + (floor[i]=='1')
        @cache
        def dfs(pos, count):
            if pos >= n: return 0
            if count == 0: return suffix[pos]
            return min(dfs(pos+carpetLen, count-1), dfs(pos+1, count) + (floor[pos] == '1'))
        return dfs(0, numCarpets)
```
