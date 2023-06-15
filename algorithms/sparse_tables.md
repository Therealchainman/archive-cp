# SPARSE TABLES

## Range Minimum Queries (RMQ)

RMQ + sparse tables + O(nlogn) precompute sparse tables + O(1) query since the ranges can overlap without affecting the in

This approach that only work on static arrays, i.e. it is not possible to change a value in the array without recomputing the complete data structure.

range minimum query can also be used with (value, index), where the index is location in an array and can be used in a divide and conquer type algorithm to split array into partition at the location of the minimum value in the current range. 

```py
import math

"""
n is size of array input
range query is [left, right]
"""
class RMQ:
    def __init__(self, n, arr):
        self.lg = [0] * (n + 1)
        self.lg[1] = 0
        for i in range(2, n + 1):
            self.lg[i] = self.lg[i//2] + 1
        max_power_two = 18
        self.sparse_table = [[math.inf]*n for _ in range(max_power_two + 1)]
        for i in range(max_power_two + 1):
            j = 0
            while j + (1 << i) <= n:
                if i == 0:
                    self.sparse_table[i][j] = arr[j]
                else:
                    self.sparse_table[i][j] = min(self.sparse_table[i - 1][j], self.sparse_table[i - 1][j + (1 << (i - 1))])
                j += 1
                
    def query(self, left: int, right: int) -> int:
        length = right - left + 1
        power_two = self.lg[length]
        return min(self.sparse_table[power_two][left], self.sparse_table[power_two][right - (1 << power_two) + 1])
```

Haven't used this one but it looks good

```py
class RangeQuery:
    def __init__(self, data, func=min):
        self.func = func
        self._data = _data = [list(data)]
        i, n = 1, len(_data[0])
        while 2 * i <= n:
            prev = _data[-1]
            _data.append([func(prev[j], prev[j + i]) for j in range(n - 2 * i + 1)])
            i <<= 1
 
    def query(self, start, stop):
        """func of data[start, stop)"""
        depth = (stop - start).bit_length() - 1
        return self.func(self._data[depth][start], self._data[depth][stop - (1 << depth)])
 
    def __getitem__(self, idx):
        return self._data[0][idx]
```