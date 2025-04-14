# Modular Multiplicative Inverse

Simple approach to find the modular multiplicative inverse of a number in python.

```py
res = (res*pow(cur, -1, mod))%mod
```

```py
def mod_inverse(num, mod):
    return pow(num, -1, mod)
```

Alternative that is needed sometimes, not all python3 versions support negative exponents in pow.

```py
def mod_inverse(num, mod):
    return pow(num, mod - 2, mod)
```

Binary Exponentiation is another algorithm to find it in log(m) time. 

Calculates 1/i, or the multiplicative inverse. 

## C++ implementation modular inverse

modular inverse of i under modulus m

```cpp
int inv(int i, int m) {
    return i <= 1 ? i : m - (m/i) * inv(m % i, m) % m;
}
```