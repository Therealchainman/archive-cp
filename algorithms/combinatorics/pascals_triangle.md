# Pascal's Triangle

## Problem Statement

Given a number `n`, generate the first `n` rows of Pascal's triangle.

Precompute the pascals triangle for all the N rows, it is O(N^2) time complexity and the values grow fast, while this works for the maxn = 55, it will not work if it is a little larger.  And then you must use modular arithmetic or something.

```cpp
const int MAXN = 55;
int64 pascal[MAXN][MAXN];

void precompute(int N) {
    for (int i = 0; i < N; i++) {
        pascal[i][0] = pascal[i][i] = 1;
        for (int j = 1; j < i; j++) {
            pascal[i][j] = (pascal[i - 1][j - 1] + pascal[i - 1][j]);
        }
    }
}
```