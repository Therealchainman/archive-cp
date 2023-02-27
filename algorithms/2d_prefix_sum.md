# 2D PREFIX SUM 


TODO: rewrite this with a class 2d prefix sum that includes a query function and the function that constructs the 2d array representation of a prefix sum. 

```py
class Solution:
    def recurse(self, min_row: int, min_col: int, max_row: int, max_col: int) -> 'Node':
        delta = max_row - min_row
        grid_sum = self.ps[max_row][max_col] - self.ps[max_row][min_col] - self.ps[min_row][max_col] + self.ps[min_row][min_col]
        if grid_sum == 0 or grid_sum == delta*delta:
            val = 0 if grid_sum == 0 else 1
            return Node(val, True)
        mid_row, mid_col = min_row + delta//2, min_col + delta//2
        topLeft, topRight, bottomLeft, bottomRight = (min_row, min_col, mid_row, mid_col), (min_row, mid_col, mid_row, max_col), (mid_row, min_col, max_row, mid_col), (mid_row, mid_col, max_row, max_col)
        return Node(0, False, self.recurse(*topLeft), self.recurse(*topRight), self.recurse(*bottomLeft), self.recurse(*bottomRight))

    def construct(self, grid: List[List[int]]) -> 'Node':
        n = len(grid)
        self.ps = [[0]*(n+1) for _ in range(n+1)]
        # BUILD 2D PREFIX SUM
        for r, c in product(range(1,n+1),range(1,n+1)):
            self.ps[r][c] = self.ps[r-1][c] + self.ps[r][c-1] + grid[r-1][c-1] - self.ps[r-1][c-1]
        return self.recurse(0, 0, n, n)
```