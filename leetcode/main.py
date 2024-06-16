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
class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n, m = len(nums), len(queries)
        peaks = [0] * n
        for i in range(1, n - 1):
            if nums[i] > nums[i - 1] and nums[i] > nums[i + 1]:
                peaks[i] = 1
        ans = []
        ft = FenwickTree(n)
        for i in range(n):
            if peaks[i]: ft.update(i + 1, 1)
        for t, l, r in queries:
            if t == 1:
                res = ft.query_range(l + 1, r + 1)
                ans.append(res)
            else:
                nums[l] = r
                # update index peak
                if l > 0 and l + 1 < n:
                    if nums[l] > nums[l - 1] and nums[l] > nums[l + 1]:
                        if not peaks[l]:
                            peaks[l] = 1
                            ft.update(l + 1, 1)
                    else:
                        if peaks[l]:
                            peaks[l] = 0
                            ft.update(l + 1, -1)
                # update index + 1 peak
                if l + 2 < n:
                    if nums[l + 1] > nums[l] and nums[l + 1] > nums[l + 2]:
                        if not peaks[l + 1]:
                            peaks[l + 1] = 1
                            ft.update(l + 2, 1)
                    else:
                        if peaks[l + 1]:
                            peaks[l + 1] = 0
                            ft.update(l + 2, -1)
                # update index - 1 peak
                if l > 1:
                    if nums[l - 1] > nums[l - 2] and nums[l - 1] > nums[l]:
                        if not peaks[l - 1]:
                            peaks[l - 1] = 1
                            ft.update(l, 1)
                    else:
                        if peaks[l - 1]:
                            peaks[l - 1] = 0
                            ft.update(l, -1)
        return ans