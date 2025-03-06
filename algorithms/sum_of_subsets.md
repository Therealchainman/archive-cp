# Sum of Subsets

## sum of subsets dynamic programming variant

This approach efficiently computes the sum of subsets for numbers categorized by their most significant bit.
It is useful in problems involving bitwise subset sums, inclusion-exclusion principles, and combinatorial DP approaches.

```cpp
sos.assign(BITS, vector<int>(MAXN, 0));
for (int i = 0, x; i < N; i++) {
    cin >> x;
    int msb = log2(x);
    sos[msb][x ^ (1 << msb)]++;
}
// sum of subsets
for (int i = 0; i < BITS; i++) {
    for (int j = 0; j < i; j++) {
        for (int mask = 0; mask < (1 << i); mask++) {
            if (isSet(mask, j)) {
                sos[i][mask] += sos[i][mask ^ (1 << j)];
            }
        }
    }
}
```