# Grid Traversal

## Helpers for grids in cpp

Really a matrix in linear algebra

You can convert 2d point to 1d index and vice versa, check if a point is in bounds, and get the neighbors of a point in a grid.

```cpp
int mat_id(int i, int j) {
    return i * M + j;
}

pair<int, int> mat_ij(int id) {
    return {id / M, id % M};
}

bool in_bounds(int i, int j) {
    return i >= 0 && i < N && j >= 0 && j < M;
}

vector<pair<int, int>> neighborhood(int i, int j) {
    return {{i - 1, j}, {i + 1, j}, {i, j - 1}, {i, j + 1}};
}
```