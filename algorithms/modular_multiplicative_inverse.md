# Modular Multiplicative Inverse

Simple approach to find the modular multiplicative inverse of a number in python.

```py
res = (res*pow(cur, -1, mod))%mod
```

Binary Exponentiation is another algorithm to find it in log(m) time. 