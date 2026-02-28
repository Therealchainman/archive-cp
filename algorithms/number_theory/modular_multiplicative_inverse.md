# Modular Multiplicative Inverse

The modular multiplicative inverse of an integer is the number that undoes multiplication under modulo m.
In other words a multiplied by the modular multiplicative inverse of a under modulo m is congruent to 1 modulo m.

When does the multiplicative inverse exist? It exists if and only if a and m are coprime, i.e., gcd(a, m) = 1.

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
int64 inv(int i, int64 m) {
    return i <= 1 ? i : m - (m / i) * inv(m % i, m) % m;
}
```