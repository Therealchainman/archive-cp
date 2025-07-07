# Sum Over Subsets

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

## Sum over subsets dynamic programming

This is quite the standard approach, where it is iterating over n bits, 

outer loop is iterating over the bits, and the inner loop is iterating over all bitmasks. 

We want the “subset‐sums” (also called the subset‐zeta transform) of an array A

A brute-force double loop over x and its submasks $y \subseteq x$ costs $O(3^n)$.

Instead, we “push” partial sums one bit at a time: in pass $i$ we ensure that every mask $x$ has already accumulated contributions from all submasks that differ only in bit $i$.

time complexity of O(n * 2^n) is quite good, where n is the bit-width of the input.  Which is generally small (<= 20).

```cpp
vector<int> sos(1 << n);
sos = a;

for (int i = 0; i < n; i++) {
	for (int x = 0; x < (1 << n); x++) {
		if (x & (1 << i)) { sos[x] += sos[x ^ (1 << i)]; }
	}
}
```