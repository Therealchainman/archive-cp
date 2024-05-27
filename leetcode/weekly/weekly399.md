


```py
class FenwickTree:
    def __init__(self, N):
        self.sums = [0 for _ in range(N+1)]

    def update(self, i, delta):
        while i < len(self.sums):
            self.sums[i] += delta
            i += i & (-i)

    def query(self, i):
        res = 0
        while i > 0:
            res += self.sums[i]
            i -= i & (-i)
        return res

    def query_range(self, i, j):
        return self.query(j) - self.query(i - 1)

    def __repr__(self):
        return f"array: {self.sums}"
from sortedcontainers import Sortedlist
class Solution:
    def maximumSumSubsequence(self, nums: List[int], queries: List[List[int]]) -> int:
        n = len(nums)
        ans = cur = start = 0
        is_pos = False
        blocks = SortedList()
        podd, peven = FenwickTree(n), FenwickTree(n)
        for i in range(n):
            if i & 1: podd.update(i + 1, nums[i])
            else: peven.update(i + 1, nums[i])
            if nums[i] >= 0: cur += nums[i]
            if i == 0:
                is_pos = nums[i] >= 0
            if nums[i] < 0 and is_pos:
                blocks.add((start, i))
                is_pos = False
                start = i
            elif nums[i] >= 0 and not is_pos:
                blocks.add((start, i))
                is_pos = True
                start = i
        blocks.add((start, n))
        def update(i, x):
            if i & 1:
                podd.update(i + 1, -nums[i])
                podd.update(i + 1, x)
            else:
                peven.update(i + 1, -nums[i])
                peven.update(i + 1, x)
        print(blocks)
        for pos, x in queries:
            idx = blocks.bisect_left((pos, pos))
            l, r = blocks[idx]
            si = podd.query_range(l + 1, r) >= 0 or peven.query_ange(l + 1, r) >= 0
            if si:
                if x >= 0:
                    cur -= max(podd.query_range(l + 1, r), peven.query_range(l + 1, r))
                    update(pos, x)
                    cur += max(podd.query_range(l + 1, r), peven.query_range(l + 1, r))
                else:
                    cur -= max(podd.query_range(l + 1, r), peven.query_range(l + 1, r))
                    m = pos 
                    blocks.pop(idx)
                    blocks.add((l, m))
                    blocks.add((m + 1, r))
                    blocks.add((m, m + 1))
                    update(pos, x)
                    cur += max(podd.query_range(l + 1, m), peven.query_range(l + 1, m)) + max(podd.query_range(m + 1, r), peven.query_range(m + 1, r))
            else:
                if x < 0: update(pos, x)
                else:
                   m = pos
                   blocks.pop(idx)
                   blocks.add((l, m))
                   blocks.add((m + 1, r))
                   blocks.add((m, m + 1))
                   update(pos, x)
                   cur +=  

            nums[i] = x
            ans += cur
        return ans
```