# Leetcode BiWeekly Contest 126

## 3081. Replace Question Marks in String to Minimize Its Value

### Solution 1:  min heap, greedy, lexicographically smallest

```py
class Solution:
    def minimizeStringValue(self, s: str) -> str:
        n = len(s)
        minheap = []
        for ch in string.ascii_lowercase:
            heappush(minheap, (s.count(ch), ch))
        add = []
        s = list(s)
        for _ in range(s.count("?")):
            x, ch = heappop(minheap)
            add.append(ch)
            heappush(minheap, (x + 1, ch))
        add.sort(reverse = True)
        for i in range(n):
            if s[i] != "?": continue 
            s[i] = add.pop()
        return "".join(s)
```

## 3082. Find the Sum of the Power of All Subsequences

### Solution 1:  combinatorics, dp, counting, subsequences

```py
class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        n = len(nums)
        freq = Counter({(1, 0): 1}) # (size, sum) -> freq
        ans = 0
        MOD = int(1e9) + 7
        for num in nums:
            nfreq = freq.copy()
            for (sz, su), f in freq.items():
                if su + num == k: ans = (ans + f * pow(2, n - sz, MOD)) % MOD
                if su + num >= k: continue 
                nfreq[(sz + 1, su + num)] += f
            freq = nfreq
        return ans
```

