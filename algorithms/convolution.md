# Convolution

## Convolution of two sequences of numbers

This is a simple implementation of convolution of two sequences of numbers. It is not the fastest implementation, but it is easy to understand.
O(nm) time complexity

```py
from itertools import product
mod = 10**9 + 7

def convolution(arr1: List[int], arr2: List[int]) -> List[int]:
    """
    Convolution of two sequences of numbers modulo mod
    """
    n = len(arr1)
    m = len(arr2)
    res = [0] * (n + m - 1)
    for i, j in product(range(n), range(m)):
        res[i + j] += arr1[i] * arr2[j]
        res[i + j] %= mod
    return res
```