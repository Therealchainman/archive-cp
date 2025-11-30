# Range XOR Queries

This is rather simple extension of range sum queries.  It is basically precomputing the prefix xor sum, and just using the xor to get the query, it can be used for tracking even/odd frequency of elements in a range.

Implementation is super simple

```py
def xor_sum(l, r):
    return pxor[r] ^ (pxor[l - 1] if l > 0 else 0)
```

```cpp
int xor_sum(int l, int r) {
    return pxor[r] ^ (l > 0 ? pxor[l - 1] : 0);
}
```