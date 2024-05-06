# Leetcode Weekly Contest 393

## 3116. Kth Smallest Amount With Single Denomination Combination

### Solution 1:  binary search, inclusion exclusion principle, lcm

```py
class Solution:
    def findKthSmallest(self, coins: List[int], k: int) -> int:
        n = len(coins)
        lcms = [[] for _ in range(n + 1)]
        for mask in range(1, 1 << n):
            cur = 1
            for i in range(n):
                if (mask >> i) & 1:
                    cur = math.lcm(cur, coins[i])
            lcms[mask.bit_count()].append(cur)
        def possible(target):
            ans = 0
            for i in range(1, n + 1):
                for v in lcms[i]:
                    if i & 1: ans += target // v
                    else: ans -= target // v
            return ans
        l, r = 0, 25 * 2 * 10 ** 9
        while l < r:
            m = (l + r) >> 1
            if possible(m) < k:
                l = m + 1
            else:
                r = m
        return l
```

## 3117. Minimum Sum of Values by Dividing Array

### Solution 1:  range bitwise or queries with sparse table, binary search, dynamic programming, min heap, line sweep

```py
class ST_And:
    def __init__(self, nums):
        self.nums = nums
        self.n = len(nums)
        self.LOG = 14 # 10,000
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
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n, m = len(nums), len(andValues)
        st_and = ST_And(nums)
        def ubsearch(start, target):
            l, r = start, n - 1
            while l < r:
                m = (l + r + 1) >> 1
                if st_and.query(start, m) >= target:
                    l = m
                else:
                    r = m - 1
            return l if st_and.query(start, l) == target else -1
        def lbsearch(start, target):
            l, r = start, n - 1
            while l < r:
                m = (l + r) >> 1
                if st_and.query(start, m) > target:
                    l = m + 1
                else:
                    r = m
            return l if st_and.query(start, l) == target else n
        dp = [[math.inf] * m for _ in range(n)]
        target = andValues[0]
        s, e = lbsearch(0, target), ubsearch(0, target)
        if s > e: return -1
        for i in range(s, e + 1):
            dp[i][0] = nums[i]
        for j in range(1, m):
            events = [(math.inf, math.inf)] * n
            activate = [(math.inf, math.inf)] * n
            minheap = []
            target = andValues[j]
            for i in range(1, n):
                if dp[i - 1][j - 1] == math.inf: continue
                s, e = lbsearch(i, target), ubsearch(i, target)
                if s > e: continue
                events[i] = (s, e)
            found = False
            for i in range(1, n):
                s, e = events[i]
                if s < math.inf: activate[s] = min(activate[s], (dp[i - 1][j - 1], e))
                if activate[i] is not None: heappush(minheap, activate[i])
                while minheap and minheap[0][1] < i: heappop(minheap)
                if minheap: 
                    dp[i][j] = minheap[0][0] + nums[i]
                    found = True
            if not found: return -1
        return dp[-1][-1] if dp[-1][-1] < math.inf else -1
```

### Solution 2:  range bitwise or queries with sparse table, binary search, range minimum query with sparse table, dynamic programming

```py
class ST_And:
    def __init__(self, nums):
        self.nums = nums
        self.n = len(nums)
        self.LOG = 14 # 10,000
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
class ST_Min:
    def __init__(self, nums):
        self.nums = nums
        self.n = len(nums)
        self.LOG = 14 # 10,000
        self.build()

    def op(self, x, y):
        return min(x, y)

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
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n, m = len(nums), len(andValues)
        st_and = ST_And(nums)
        def ubsearch(end, target):
            l, r = 0, end
            while l < r:
                m = (l + r + 1) >> 1
                if st_and.query(m, end) <= target:
                    l = m
                else:
                    r = m - 1
            return l if st_and.query(l, end) == target else -1
        def lbsearch(end, target):
            l, r = 0, end
            while l < r:
                m = (l + r) >> 1
                if st_and.query(m, end) < target:
                    l = m + 1
                else:
                    r = m
            return l if st_and.query(l, end) == target else n
        dp = [[math.inf] * n for _ in range(m)]
        # INITIALIZE
        target = andValues[0]
        for i in range(n):
            l, r = lbsearch(i, target), ubsearch(i, target)
            if l > r: continue
            dp[0][i] = nums[i]
        # MAIN DP ITERATIONS
        for j in range(1, m):
            st_min = ST_Min(dp[j - 1])
            target = andValues[j]
            for i in range(1, n):
                l, r = max(0, lbsearch(i, target) - 1), max(0, ubsearch(i, target) - 1)
                if l > r: continue
                dp[j][i] = st_min.query(l, r) + nums[i]
        return dp[-1][-1] if dp[-1][-1] < math.inf else -1
```

