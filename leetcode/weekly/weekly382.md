# Leetcode Weekly Contest 382

## 3019. Number of Changing Keys

### Solution 1:  sum of adjacent diffs

```py
class Solution:
    def countKeyChanges(self, s: str) -> int:
        return sum(1 for i in range(1, len(s)) if s[i - 1].lower() != s[i].lower())
```

## 3020. Find the Maximum Number of Elements in Subset

### Solution 1:  counter, math

```py
class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        MAXN = 10**9
        freq = Counter(nums)
        ans = 0
        for k in sorted(freq):
            cnt = 0
            cur = k
            if cur == 1: 
                cnt += freq[cur]
                freq[cur] = 0
            while cur <= MAXN and freq[cur] > 0:
                cnt += 2
                if freq[cur] == 1: break
                freq[cur] = 0
                cur *= cur
            if cnt % 2 == 0: cnt -= 1
            ans = max(ans, cnt)
        return ans
```

## 3021. Alice and Bob Playing Flower Game

### Solution 1:  math, parity, trick

```py
class Solution:
    def flowerGame(self, n: int, m: int) -> int:
        return sum(m // 2 if i & 1 else (m + 1) // 2 for i in range(1, n + 1))
```

## 3022. Minimize OR of Remaining Elements Using Operations

### Solution 1: 

```py
import operator
class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        MAXB = 30
        n = len(nums)
        mask = ans = 0
        def min_operations(mask):
            cnt = 0
            val = (1 << MAXB) - 1
            for num in nums:
                if (num & mask) != 0:
                    val &= (num & mask)
                    if val == 0: 
                        val = (1 << MAXB) - 1
                        continue
                    cnt += 1
                else:
                    val = (1 << MAXB) - 1
            return cnt
        for i in reversed(range(MAXB)):
            x = min_operations(mask | (1 << i))
            if x <= k: mask |= (1 << i)
        return reduce(operator.or_, [(1 << i) for i in range(MAXB) if not ((mask >> i) & 1)], 0)
```

