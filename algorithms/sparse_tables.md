# SPARSE TABLES

## Range Minimum Queries (RMQ)

RMQ + sparse tables + O(nlogn) precompute sparse tables + O(1) query since the ranges can overlap without affecting the in

```py
lg = [0] * (n + 1)
lg[1] = 0
for i in range(2, n + 1):
    lg[i] = lg[i//2] + 1
max_power_two = 18
sparse_table = [[math.inf]*n for _ in range(max_power_two + 1)]
for i in range(max_power_two + 1):
    j = 0
    while j + (1 << i) <= n:
        if i == 0:
            sparse_table[i][j] = arr[j]
        else:
            sparse_table[i][j] = min(sparse_table[i - 1][j], sparse_table[i - 1][j + (1 << (i - 1))])
        j += 1
def query(left: int, right: int) -> int:
    length = right - left + 1
    power_two = lg[length]
    return min(sparse_table[power_two][left], sparse_table[power_two][right - (1 << power_two) + 1])
```