# Factorization

## Precomputing all the factors with factorization sieve

This algorithm efficiently precomputes the factors of all integers from 1 to n using a modified sieve approach in O(n log n) time complexity. It stores the factors of each number in a sorted order, making subsequent queries for factors extremely fast.

Algorithm Overview:
Time Complexity Analysis:
The outer loop runs from 1 to n, and the inner loop runs approximately n / i times for each i.
The total number of iterations across all loops is n (1 + 1/2 + 1/3 + ... + 1/n) ≈ O(n log n).
Limitations:
The algorithm requires O(n log n) time, so it is feasible for n up to approximately 10⁸.
Storing all factors requires O(n log n) space, which may become a bottleneck for very large values of n.
Usage Considerations:
This precomputation should be performed once per program execution, not per test case, to optimize performance.
The resulting factor lists can be used for efficient factor-related computations such as divisor sums, prime factorization, and number-theoretic applications.

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