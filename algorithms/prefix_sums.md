# Prefix Sums

## Bitwise And

prefix sum for bitwise and is to take the prefix sum for the frequency of each bit, although the better solution is to use sparse tables, so look at binary jumping file.

```py
BITS = 18
psum = [[0] * BITS for _ in range(n)]
for i in range(n):
    for j in range(BITS):
        if (nums[i] >> j) & 1:
            psum[i][j] = 1
        if i > 0: psum[i][j] += psum[i - 1][j]
def sum_(i, j, b):
    return psum[j][b] - (psum[i - 1][b] if i > 0 else 0)
def sum_and(i, j):
    len_ = j - i + 1
    return sum(1 << b for b in range(BITS) if sum_(i, j, b) == len_)
```