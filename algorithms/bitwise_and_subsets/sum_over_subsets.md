# Sum Over Subsets

## subset zeta transform or sum over subsets dynamic programming

This is quite the standard approach, where it is iterating over n bits, 

outer loop is iterating over the bits, and the inner loop is iterating over all bitmasks. 

We want the “subset‐sums” (also called the subset‐zeta transform) of an array A

A brute-force double loop over x and its submasks $y \subseteq x$ costs $O(3^n)$.

Instead, we “push” partial sums one bit at a time: in pass $i$ we ensure that every mask $x$ has already accumulated contributions from all submasks that differ only in bit $i$.

time complexity of O(n * 2^n) is quite good, where n is the bit-width of the input.  Which is generally small (<= 20).

A related concept is the bit-wth, because that is needed. 

The bit-width is the mininmum number of binary digits to represent a number without loss of information. 
n = $\left\lfloor \log_2 N \right\rfloor + 1$

We represent sets with bitmasks.  We can imagine an n-dimensional hypercube, where each vertex represents an element in the bitmask or set.  

So if a few of these vertex are active that is a subset of the n-hypercube. 

The algorithm works by iterating over each vertex in hypercube, so we iterate over i. 
There is an inner loop that goes through all $2^n$ masks if i is set it pulls in the value from mask ^ (1 << i) (mask with i unset)

At the end this will calculate for each bitmask the sum over all subsets of that bitmask.

$$F(T) = \sum_{S \subseteq T} f(S)$$

Note the group operation is xor addition, but it can be addition, multiplication, etc.

```cpp
for (int i = 0; i < LOG; ++i) { // iterate over bits
    for (int mask = 0; mask <= endMask; ++mask) { // iterate over all masks
        if ((mask >> i) & 1) dp[mask] ^= dp[mask ^ (1 << i)]; // subset
    }
}
```

Can also be used for maximum over all subsets

```cpp
for (int i = 0; i < LOG; ++i) { // iterate over bits
    for (int mask = 0; mask <= endMask; ++mask) { // iterate over all masks
        if ((mask >> i) & 1) dp[mask] = max(dp[mask], dp[mask ^ (1 << i)]); // maximum subset
    }
}
```

## superset zeta transform or sum over supersets

$$F(T) = \sum_{S \supseteq T} f(S)$$

```cpp
for (int i = 0; i < LOG; ++i) { // iterate over bits
    for (int mask = 0; mask <= endMask; ++mask) { // iterate over all masks
        if ((mask >> i) & 1) dp[mask ^ (1 << i)] += dp[mask]; // superset
    }
}
```


## superset mobius inversion or inverse of sum over supersets

Given this 
$$G(T) = \sum_{S \supseteq T} f(S)$$
We want to calculate the following
$$f(T) = G(T) - \sum_{S \supset T} f(S)$$
By subtracting off the contributions of all strict supersets S in G(T) isolates the one term with S = T. 

This can also be calculated with bitmask dp approach. 

```cpp
for (int i = 0; i < LOG; ++i) { // iterate over
    for (int mask = 0; mask <= endMask; ++mask) { // iterate over all masks
        if ((mask >> i) & 1) dp[mask ^ (1 << i)] -= dp[mask]; // inverse superset
    }
}
```

## sum over subsets dynamic programming variant

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