# Leetcode Weekly Contest 400

## 3171. Find Subarray With Bitwise AND Closest to K

### Solution 1:  bitwise and range queries, static array, sparse table, binary search

```py
class ST_And:
    def __init__(self, nums):
        self.nums = nums
        self.n = len(nums)
        self.LOG = 18 # 10,000
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
    def minimumDifference(self, nums: List[int], k: int) -> int:
        n = len(nums)
        ans = math.inf
        st = ST_And(nums)
        def possible(src, dst):
            v = st.query(src, dst)
            return v >= k
        def bsearch(start):
            lo, hi = start, n - 1
            while lo < hi:
                mid = (lo + hi + 1) >> 1
                if possible(start, mid):
                    lo = mid
                else:
                    hi = mid - 1
            return lo
        for i in range(n):
            j = bsearch(i)
            v = st.query(i, j)
            ans = min(ans, abs(v - k))
            j += 1
            if j < n:
                v = st.query(i, j)
                ans = min(ans, abs(v - k))
        return ans
```