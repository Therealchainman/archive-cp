# XOR Basis


Gaussian elimination over GF(2) to find the basis of $\mathbb{Z}_2^{31}$.

Gaussian elimination to get the row echelon form which also represents the basis. 

This works because we are performing xor bitwise operations on each element of the vectors, and this is just addition modulo 2. 

```cpp
BITS = 31;
vector<int> basis(BITS, 0);
for (int i = 0; i < N; ++i) {
    int x;
    cin >> x;
    for (int b = BITS - 1; b >= 0; --b) {
        if (!((x >> b) & 1)) continue;
        if (!basis[b]) {
            basis[b] = x;
            break;
        }
        x ^= basis[b];
    }
}
```

There is another way to do basis though, which is important some times: 

```cpp
vector<int> basis;
for (int i = 0; i < N; ++i) {
    for (int b : basis) {
        A[i] = min(A[i], A[i] ^ b);
    }
    if (!A[i]) continue;
    for (int &b : basis) {
        b = min(b, b ^ A[i]);
    }
    basis.emplace_back(A[i]);
}
```