# 2D PREFIX SUM 

## C++ implementation of 2d prefix sum

This is inclusive range for submatrix so from upper left corner (r1, c1) to bottom right corner (r2, c2)

The inclusive nature makes it easy to work with so if I want to query the sum of a rectangle from (1, 1) to (3, 3) I just do:
query(ps, 1, 1, 3, 3)

```cpp
int query(const vector<vector<int>> &ps, int r1, int c1, int r2, int c2) {
    return ps[r2 + 1][c2 + 1] - ps[r1][c2 + 1] - ps[r2 + 1][c1] + ps[r1][c1];
}

vector<vector<int>> ps(R + 1, vector<int>(C + 1, 0));
for (int r = 1; r <= R; r++) {
    for (int c = 1; c <= C; c++) {
        ps[r][c] = ps[r - 1][c] + ps[r][c - 1] - ps[r - 1][c - 1] + mat[r - 1][c - 1];
    }
}
```

## Python implementation of 2d prefix sum

```py
R, C = len(matrix), len(matrix[0])
ps = [[0]*(C+1) for _ in range(R+1)]
# BUILD 2D PREFIX SUM
for r, c in product(range(1,R+1),range(1,C+1)):
    ps[r][c] = ps[r-1][c] + ps[r][c-1] + matrix[r-1][c-1] - ps[r-1][c-1]
# query
psum = ps[max_row][max_col] - ps[max_row][min_col] - ps[min_row][max_col] + ps[min_row][min_col]
```

## column-wise prefix sum

This is useful when you are going to add a rectangle with width = 1, so just at that column, but height could be whatever, cause now you are adding r1 to r2 at c,  and you can find the sum to add with this type of 2d prefix sum.

```py
R, C = len(matrix), len(matrix[0])
ps = [[0] * C for _ in range(R)]
# build columnwise prefix sum
for r, c in product(range(R), range(C)):
    ps[r][c] = matrix[r][c]
    if r > 0: ps[r][c] += ps[r - 1][c]
```

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

