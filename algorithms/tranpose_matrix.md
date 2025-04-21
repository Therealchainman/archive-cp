# Transpose Matrix

```py
def transpose_matrix(matrix):
    return list(map(list, zip(*matrix)))
```

Transposition lets you treat column-wise problems as row-wise problems — which is often a big win, because many algorithms are more naturally expressed or efficiently implemented row-by-row. For example:

```cpp
void transpose(vector<vector<int>>& mat) {
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            swap(mat[i][j], mat[j][i]);
        }
    }
}
```