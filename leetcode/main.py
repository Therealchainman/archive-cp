class Solution:
    def minimumArea(self, grid: List[List[int]]) -> int:
        R, C = len(grid), len(grid[0])
        minr, maxr, minc, maxc = R, -1, C, -1
        for r, c in product(range(R), range(C)):
            if grid[r][c]:
                minr = min(minr, r)
                maxr = max(maxr, r)
                minc = min(minc, c)
                maxc = max(maxc, c)
        return (maxr - minr + 1) * (maxc - minc + 1)