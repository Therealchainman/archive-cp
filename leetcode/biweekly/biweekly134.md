# Leetcode BiWeekly Contest 134

## Number of Subarrays With AND Value of K

### Solution 1:  sparse table, range AND queries, binary search

```py
class ST_And:
    def __init__(self, nums):
        self.nums = nums
        self.n = len(nums)
        self.LOG = 21 # 10^9
        self.build()

    def op(self, x, y):
        return x & y

    def build(self):
        self.lg = [0] * (self.n + 1)
        for i in range(2, self.n + 1):
            self.lg[i] = self.lg[i // 2] + 1
        self.st = [[0] * self.n for _ in range(self.LOG)]
        for i in range(self.n): 
            self.st[0][i] = self.nums[i]
        # CONSTRUCT SPARSE TABLE
        for i in range(1, self.LOG):
            j = 0
            while (j + (1 << (i - 1))) < self.n:
                self.st[i][j] = self.op(self.st[i - 1][j], self.st[i - 1][j + (1 << (i - 1))])
                j += 1

    def query(self, l, r):
        length = r - l + 1
        i = self.lg[length]
        return self.op(self.st[i][l], self.st[i][r - (1 << i) + 1])
class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = 0
        st = ST_And(nums)
        def upper_bound(start):
            lo, hi = start, n - 1
            while lo < hi:
                mid = (lo + hi + 1) >> 1
                if st.query(start, mid) < k:
                    hi = mid - 1
                else:
                    lo = mid
            return lo
        def lower_bound(start):
            lo, hi = start, n - 1
            while lo < hi:
                mid = (lo + hi) >> 1
                if st.query(start, mid) <= k:
                    hi = mid
                else:
                    lo = mid + 1
            return lo
        for i in range(n):
            l, r = lower_bound(i), upper_bound(i)
            if r == l and st.query(i, l) != k: continue
            ans += r - l + 1
        return ans
```