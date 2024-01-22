# Factors

## Precomputing all the factors for each integer

This can be solved in O(nlogn) time complexity.

The limitation of this algorithm is that n can't be too large, maybe 10^8 is about as reasonably large you'd want it.  And you can compute all the factors for each integer from 1 to 10^8.  They will also be sorted for each integer in order of ascending factors.

You want to compute this just once, not for each test case also.

```py
LIM = int(1e6)
factors = [[] for _ in range(LIM + 1)]
for i in range(1, LIM + 1):
    for j in range(i, LIM + 1, i):
        factors[j].append(i)
```

## Factorize in sqrt(n) time

The factorization algorithm that works in sqrt(n) time is the following, it can be modified what it returns, this one is just returning the smallest factor that is greater than 1. 

```py
import math
def factorable(num):
    for i in range(2, int(math.sqrt(num))+1):
        if num % i == 0: return i
    return 0
```