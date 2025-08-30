# Transpose Matrix

```py
def transpose_matrix(matrix):
    return list(map(list, zip(*matrix)))
```

The transposition of a matrix is the process of flipping the matrix over its main diagonal. This means that the rows become columns and the columns become rows.

Transposition lets you treat column-wise problems as row-wise problems — which is often a big win, because many algorithms are more naturally expressed or efficiently implemented row-by-row. For example:

## Transpose square matrix in-place

```cpp
void transpose(vector<vector<int>>& mat) {
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            swap(mat[i][j], mat[j][i]);
        }
    }
}
```

## Transpose rectangular matrix

```cpp
vector<vector<char>> transpose(const vector<vector<char>>& mat) {
    vector<vector<char>> ans(C, vector<char>(R));
    for (int i = 0; i < R; i++) {
        for (int j = 0; j < C; j++) {
            ans[j][i] = mat[i][j];
        }
    }
    return ans;
}
```

## Rotate a matrix 90 degrees counterclockwise

```cpp
vector<vector<int>> rotate(const vector<vector<int>> &grid) {
    int R = grid.size(), C = grid[0].size();
    vector<vector<int>> ret(C, vector<int>(R));
    for (int r = 0; r < R; ++r) {
        for (int c = 0; c < C; ++c) {
            ret[C - c - 1][r] = grid[r][c];
        }
    }
    return ret;
}
```