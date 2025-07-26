# Fast Walsh Hadamard Transform

This is an algorithm that runs in O(NlogN) using divide and conquer inspired algorithm, and butterfly steps similar to the Fast Fourier Transform.  This is used specifically for xor convolution. 
$$
(f *_\oplus g)[n]
\;=\;\sum_{k=0}^{N-1} f[k]\;g[n\oplus k]
\;=\;\sum_{i\oplus j = n} f[i]\;g[j].
$$

This is the algorithm that performs Hadamard transform

```cpp
// forward (and inverse) Walsh–Hadamard transform in place.
// A.size() == n must be a power of two.
void fwht(vector<int64>& A, int n) {
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            for (int j = 0; j < len; ++j) {
                int64 u = A[i + j];
                int64 v = A[i + j + len];
                A[i + j] = u + v;
                A[i + j + len] = u - v;
            }
        }
    }
}
```

Given arrays f and g, and you want to calculate the xor convolution you do the following:

1. FWHT both inputs
1. Point‑wise multiply
1. FWHT the product
1. Divide by N

```cpp
// XOR‐convolution of f and g into h.
// f, g must both be length n = 2^k.
// If they’re shorter or unequal, pad to the same power‐of‐two length.
vector<int64> xor_convolution(vector<int64> f, vector<int64> g) {
    int n = f.size();
    // 1) FWHT
    fwht(f, n);
    fwht(g, n);
    // 2) point‑wise multiply
    vector<int64> h(n);
    for (int i = 0; i < n; ++i)
        h[i] = f[i] * g[i];
    // 3) inverse = same transform
    fwht(h, n);
    // 4) normalize
    for (int i = 0; i < n; ++i)
        h[i] /= n;
    return h;
}

```

example of the padding and creating the arrays, this is just for freq, but imagine it for two arrays. 

```cpp
int bits = 0;
while (1 << bits <= maxVal) ++bits;
if (bits == 0) bits = 1;
int M = 1 << bits;

vector<int64> freq(M, 0);
for (int x : pref) {
    ++freq[x];
}
```