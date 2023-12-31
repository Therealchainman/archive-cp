# Leetcode Weekly Contest 378

## 2980. Check if Bitwise OR Has Trailing Zeros

### Solution 1:  bitwise manipulation, parity

```py
class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        n = len(nums)
        for i in range(n):
            for j in range(i):
                if (nums[i] | nums[j]) % 2 == 0: return True
        return False
```

### Solution 2:  At least two even integers

```py
class Solution:
    def hasTrailingZeros(self, nums: List[int]) -> bool:
        return len([x for x in nums if x % 2 == 0]) >= 2
```

## 2982. Find Longest Special Substring That Occurs Thrice II

### Solution 1:  max heap, groupby, 26 groups

```py
class Solution:
    def maximumLength(self, s: str) -> int:
        ans = -1
        groups = defaultdict(list)
        for k, g in groupby(s):
            sz = len(list(g))
            heappush(groups[k], -sz)
        for k in groups:
            cnt = 0
            for _ in range(2):
                sz = -heappop(groups[k])
                sz -= 1
                heappush(groups[k], -sz)
            ans = max(ans, -groups[k][0])
        return ans if ans > 0 else -1
```

### Solution 2:  list of every group, 26 groups, sort, take the third largest

```py
class Solution:
    def maximumLength(self, s: str) -> int:
        ans = -1
        n = len(s)
        groups = defaultdict(list)
        groups[s[0]].append(1)
        cur = 1
        for i in range(1, n):
            if s[i] != s[i - 1]:
                cur = 0
            cur += 1
            groups[s[i]].append(cur)
        for lst in groups.values():
            if len(lst) < 3: continue
            lst.sort(reverse = True)
            ans = max(ans, lst[2])
        return ans
```

## 2983. Palindrome Rearrangement Queries

### Solution 1:  ranges, prefix sums, equivalent substrings

```py

```

