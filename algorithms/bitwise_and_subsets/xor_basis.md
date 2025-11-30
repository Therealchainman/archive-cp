# XOR Basis

Gaussian elimination over GF(2) to find the basis of $\mathbb{Z}_2^{31}$.

Gaussian elimination to get the reduced row echelon form which also represents the basis. 

This works because we are performing xor bitwise operations on each element of the vectors, and this is just addition modulo 2. 

Might have to sort to find max or smallest numbers. 

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