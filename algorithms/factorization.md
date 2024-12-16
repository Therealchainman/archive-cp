# Factorization

## Precomputing all the factors with factorization sieve

This can be solved in O(nlogn) time complexity.

The limitation of this algorithm is that n can't be too large, maybe 10^8 is about as reasonably large you'd want it.  And you can compute all the factors for each integer from 1 to 10^8.  They will also be sorted for each integer in order of ascending factors.

You want to compute this just once, not for each test case also.

```cpp
const int MAXN = 2e5 + 5;
bool precomputed;
vector<int> factors[MAXN];
void precomputeFactors(int n) {
    for (int i = 1; i < n; i++) {
        for (int j = i; j < n; j += i) {
            factors[j].emplace_back(i);
        }
    }
}
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